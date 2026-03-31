"""Core Run class – the single object a researcher interacts with."""
from __future__ import annotations

import atexit
import logging
import os
import re
import secrets
import warnings
import signal
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .serialization import append_jsonl, normalize_config, read_json, write_json
from .outputs import save_output_tree
from .slurm import get_slurm_info
from .git_info import get_code_ref
from .environment import get_run_environment

log = logging.getLogger("euler_train")

_DATASET_META_NAMESPACES = ("euler_loading", "euler_train")


class Run:
    """A single experiment run backed by a directory on disk.

    Typically created via :func:`runlog.init` rather than directly.
    """

    def __init__(
        self,
        dir: str | Path | None = None,
        config: Any = None,
        meta: dict | None = None,
        output_formats: dict[str, str] | None = None,
        gpu_stats_every: int = 100,
        run_id: str | None = None,
        datasets: dict[str, Any] | None = None,
        run_name: str | None = None,
        evaluations: dict[str, dict[str, Any]] | None = None,
        mode: str | None = None,
    ) -> None:
        # ── detect run-directory shorthand ────────────────────────
        # When `dir` points to an existing run directory (contains
        # meta.json) and no explicit `run_id` is provided, treat it
        # as a resume: extract run_id from meta.json and derive the
        # project directory from the path.
        if dir is not None and run_id is None:
            candidate = Path(dir)
            meta_file = candidate / "meta.json"
            if meta_file.is_file():
                existing_meta = read_json(meta_file)
                run_id = existing_meta.get("run_id", candidate.name)
                dir = candidate.parent.parent

        self.project_dir: Path = Path(dir) if dir is not None else _infer_dir()
        resuming = run_id is not None
        self._resuming = resuming
        self.run_id: str = run_id if resuming else _generate_run_id()
        self.dir = self.project_dir / "runs" / self.run_id

        if resuming and not self.dir.exists():
            raise FileNotFoundError(
                f"Cannot resume run {self.run_id!r}: "
                f"directory {self.dir} does not exist"
            )
        self.dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir: Path | None = None
        self._output_formats: dict[str, str] = output_formats or {}
        self._start_time = time.time()
        self._finished = False
        self._gpu_handle: Any | None = None
        self._gpu_available: bool | None = None  # None = not yet probed
        self._gpu_stats_every: int = gpu_stats_every
        self.run_name: str | None = run_name
        self.mode: str | None = _normalize_mode(mode)

        # ── config ────────────────────────────────────────────────
        if resuming:
            config_path = self.dir / "config.json"
            self.config: dict = read_json(config_path) if config_path.exists() else {}
            if config is not None:
                self.config.update(normalize_config(config))
        else:
            self.config: dict = normalize_config(config)

        write_json(self.dir / "config.json", self.config)

        # ── code ref & run environment (fresh runs only) ─────────
        if not resuming:
            write_json(self.dir / "code_ref.json", get_code_ref())
            write_json(self.dir / "run_environment.json", get_run_environment())

        # ── meta ──────────────────────────────────────────────────
        if resuming:
            meta_path = self.dir / "meta.json"
            self._meta: dict[str, Any] = (
                read_json(meta_path) if meta_path.exists() else {}
            )
            self._meta.update(
                status="running",
                pid=os.getpid(),
                python=sys.version.split()[0],
                command=sys.argv,
            )
            self._clear_terminal_fields(self._meta, status="running")
        else:
            self._meta: dict[str, Any] = {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "status": "running",
                "start_time": self._start_time,
                "start_iso": _isotime(self._start_time),
                "end_time": None,
                "end_iso": None,
                "duration_sec": None,
                "pid": os.getpid(),
                "python": sys.version.split()[0],
                "command": sys.argv,
                "slurm": get_slurm_info(),
            }
        if meta:
            self._meta.update(meta)

        meta_run_name = self._meta.get("run_name")
        if isinstance(meta_run_name, str) or meta_run_name is None:
            self.run_name = meta_run_name

        checkpoint_dir = self._meta.get("checkpoint_dir")
        if isinstance(checkpoint_dir, str) and checkpoint_dir.strip():
            self.checkpoint_dir = Path(checkpoint_dir)

        # ── datasets (optional euler_loading integration) ────────
        if datasets is not None:
            self._meta["datasets"] = _build_datasets_meta(
                datasets=datasets,
                existing=self._meta.get("datasets"),
            )

        # ── evaluations (optional, typically on resume) ───────
        if evaluations is not None:
            self._meta["evaluations"] = _build_evaluations_meta(
                evaluations=evaluations,
                existing=self._meta.get("evaluations"),
            )
        self._mark_mode_running()
        self._flush_meta()
        self._setup_hooks()

        verb = "Resumed" if resuming else "Started"
        log.info("%s run %s", verb, self.run_id)
        log.info("  Run dir:     %s", self.dir)
        log.info("  Project dir: %s", self.project_dir)

    # ── evaluations ──────────────────────────────────────────────

    def add_evaluation(
        self,
        key: str,
        *,
        datasets: dict[str, Any] | None = None,
        name: str | None = None,
        status: str | None = None,
        checkpoint: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add or update a single evaluation entry in ``meta.json``.

        Processes *datasets* through the same modality-inference pipeline
        used for top-level ``datasets``.  Other fields are stored as-is.
        """
        entry: dict[str, Any] = {}
        if datasets is not None:
            entry["datasets"] = datasets
        if name is not None:
            entry["name"] = name
        if status is not None:
            entry["status"] = status
        if checkpoint is not None:
            entry["checkpoint"] = checkpoint
        if metadata is not None:
            entry["metadata"] = metadata

        self._meta["evaluations"] = _build_evaluations_meta(
            evaluations={key: entry},
            existing=self._meta.get("evaluations"),
        )
        self._flush_meta()

    def finish_evaluation(self, key: str, status: str = "completed") -> None:
        """Mark an existing evaluation as finished.

        Raises ``KeyError`` if *key* does not exist in evaluations.
        """
        evals = self._meta.get("evaluations")
        if not isinstance(evals, dict) or key not in evals:
            raise KeyError(
                f"Evaluation {key!r} not found — "
                f"available: {list(evals) if isinstance(evals, dict) else []}"
            )
        evals[key]["status"] = status
        self._flush_meta()

    # ── logging ───────────────────────────────────────────────────

    def log(
        self,
        metrics: dict[str, Any],
        *,
        step: int,
        epoch: int,
        mode: str = "train",
    ) -> None:
        """Append one record to ``train.jsonl`` or ``val.jsonl``.

        *metrics* is an arbitrary dict of scalar values (losses, learning
        rate, gradient norm, evaluation scores, …).  ``step``, ``epoch``,
        and ``wall_time`` are prepended automatically; ``elapsed_sec`` is
        added for training records.
        """
        record: dict[str, Any] = {
            "step": step,
            "epoch": epoch,
            "wall_time": time.time(),
        }
        if mode == "train":
            record["elapsed_sec"] = round(time.time() - self._start_time, 4)
        record.update(metrics)
        if step % self._gpu_stats_every == 0:
            record.update(self._get_gpu_stats())

        filename = "train.jsonl" if mode == "train" else "val.jsonl"
        append_jsonl(self.dir / filename, record)

    # ── GPU stats ─────────────────────────────────────────────────

    def _get_gpu_stats(self) -> dict[str, Any]:
        """Return GPU utilization & memory stats, or {} if unavailable."""
        if self._gpu_available is None:
            try:
                import pynvml

                pynvml.nvmlInit()
                self._gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._gpu_available = True
            except Exception:
                self._gpu_available = False
        if not self._gpu_available:
            return {}
        import pynvml

        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._gpu_handle)
            mem = pynvml.nvmlDeviceGetMemoryInfo(self._gpu_handle)
            return {
                "gpu_util_pct": util.gpu,
                "gpu_mem_util_pct": util.memory,
                "gpu_mem_used_gb": round(mem.used / 1e9, 3),
                "gpu_mem_total_gb": round(mem.total / 1e9, 3),
            }
        except Exception:
            return {}

    # ── visual / heavy outputs ────────────────────────────────────

    def save_outputs(
        self,
        *,
        epoch: int | None = None,
        step: int | None = None,
        **output_types: dict[str, Any],
    ) -> Path:
        """Save prediction / ground-truth / auxiliary arrays to disk.

        Each keyword argument is an *output type* (e.g. ``rgb``, ``depth``)
        whose value is a dict of **slots**::

            run.save_outputs(
                epoch=1, step=500,
                rgb  = dict(pred=pred_img, gt=gt_img),
                depth= dict(pred=depth_map, gt=gt_depth,
                            aux=dict(transmission=t_map)),
            )

        Accepted slot keys: ``pred``, ``gt``, ``input``, ``aux``.
        ``aux`` expects a sub-dict of named arrays.

        Returns the directory that was written to.
        """
        parts: list[str] = []
        if epoch is not None:
            parts.append(f"epoch_{epoch}")
        if step is not None:
            parts.append(f"step_{step}")
        dirname = "_".join(parts) or "unspecified"
        base = self.dir / "outputs" / dirname

        for output_type, slots in output_types.items():
            if slots is None:
                continue
            save_output_tree(
                base / output_type, slots, self._output_formats, output_type,
            )
        return base

    # ── checkpoints ───────────────────────────────────────────────

    def init_checkpoint_dir(self, base: str | Path | None = None) -> Path:
        """Set up an external checkpoint directory and record it in meta.

        Parameters
        ----------
        base:
            Explicit base path.  When *None*, the base is resolved as:

            1. ``$SCRATCH/euler_train/<project>/checkpoints``
            2. ``<project_dir>/checkpoints`` (same volume as logs).

        A subdirectory named after :attr:`run_name` (slugified) — or
        :attr:`run_id` when no name is set — is appended automatically.
        Fresh runs auto-disambiguate collisions by appending a suffix
        derived from :attr:`run_id`. Resumed runs reuse the recorded
        ``checkpoint_dir`` when available.

        Returns the created directory path.
        """
        if self.checkpoint_dir is not None:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            self._meta["checkpoint_dir"] = str(self.checkpoint_dir)
            self._flush_meta()
            log.info("Checkpoint dir: %s", self.checkpoint_dir)
            return self.checkpoint_dir

        if base is not None:
            ckpt_base = Path(base)
        else:
            ckpt_base = _infer_checkpoint_base(self.project_dir)

        slug = _checkpoint_dir_slug(self.run_name, self.run_id)
        self.checkpoint_dir = ckpt_base / slug
        if not self._resuming:
            self.checkpoint_dir = _disambiguate_path(
                self.checkpoint_dir,
                suffix=_slugify(self.run_id),
            )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._meta["checkpoint_dir"] = str(self.checkpoint_dir)
        self._flush_meta()
        log.info("Checkpoint dir: %s", self.checkpoint_dir)
        return self.checkpoint_dir

    def save_checkpoint(
        self,
        model: Any,
        *,
        epoch: int,
        step: int,
        optimizer: Any = None,
        **extra: Any,
    ) -> Path:
        """Save a checkpoint to ``epoch_{N}.pt``.

        When :meth:`init_checkpoint_dir` has been called, checkpoints are
        written there.  Otherwise they fall back to
        ``<run_dir>/checkpoints/``.

        If *model* (or *optimizer*) exposes ``.state_dict()``, it is called
        automatically.  Extra keyword arguments are included in the saved
        dict.
        """
        import torch

        ckpt_dir = self.checkpoint_dir or self.dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"epoch_{epoch}.pt"

        state: dict[str, Any] = {"epoch": epoch, "step": step}
        state["model"] = (
            model.state_dict() if hasattr(model, "state_dict") else model
        )
        if optimizer is not None:
            state["optimizer"] = (
                optimizer.state_dict()
                if hasattr(optimizer, "state_dict")
                else optimizer
            )
        state.update(extra)
        torch.save(state, path)
        self.log_saved_checkpoint(path, epoch=epoch, step=step)
        return path

    def log_saved_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int,
        step: int,
        is_best: bool = False,
    ) -> None:
        """Record a saved checkpoint in ``meta.json``.

        Call this after saving a checkpoint yourself, or let
        :meth:`save_checkpoint` call it automatically.  When *is_best*
        is ``True``, any previous checkpoint marked as best is cleared.
        """
        checkpoints: list[dict[str, Any]] = self._meta.get("checkpoints", [])

        if is_best:
            for entry in checkpoints:
                entry.pop("is_best", None)

        record: dict[str, Any] = {
            "path": str(path),
            "epoch": epoch,
            "step": step,
        }
        if is_best:
            record["is_best"] = True

        checkpoints.append(record)
        self._meta["checkpoints"] = checkpoints
        self._flush_meta()

    # ── architecture export ─────────────────────────────────────────

    def log_architecture(
        self,
        model: Any,
        dummy_input: Any,
    ) -> Path:
        """Export the model architecture to a lightweight ONNX file.

        The resulting ``architecture.onnx`` is optimized for Netron:
        redundant nodes are removed, operator fusions are applied, and
        weight tensors are stripped so only the graph topology remains.

        Requires the ``[architecture]`` optional dependencies
        (``onnx``, ``onnxruntime``, ``onnxsim``).

        Parameters
        ----------
        model:
            A PyTorch ``nn.Module``.
        dummy_input:
            Example input tensor(s) matching the model's forward signature.

        Returns
        -------
        Path
            Path to the saved ``architecture.onnx`` file.
        """
        from .architecture import export_architecture

        output_path = export_architecture(
            model, dummy_input, self.dir / "architecture.onnx"
        )
        self._meta["architecture"] = "architecture.onnx"
        self._flush_meta()
        return output_path

    # ── lifecycle ─────────────────────────────────────────────────

    def finish(self, status: str = "completed") -> None:
        """Mark the run as finished and write final metadata."""
        if self._finished:
            return
        self._finished = True
        self._teardown_hooks()
        end = time.time()
        duration = round(end - self._start_time, 3)
        self._meta.update(
            status=status,
            end_time=end,
            end_iso=_isotime(end),
            duration_sec=duration,
        )
        self._clear_terminal_fields(self._meta, status=status)
        self._finish_mode(status=status, end=end, duration=duration)
        self._flush_meta()

    def detach(self) -> None:
        """Disconnect from the run without changing its status.

        Tears down process hooks (atexit, signals, excepthook) so this
        process can exit without marking the run as completed or crashed.
        Any pending meta changes (e.g. evaluation entries) are flushed
        first.  The run's ``status``, ``end_time``, and ``duration_sec``
        are left untouched.

        Use this instead of :meth:`finish` when the run was resumed
        solely to attach evaluations while training is still active
        elsewhere.
        """
        if self._finished:
            return
        self._finished = True
        self._teardown_hooks()
        self._flush_meta()

    # ── context manager ───────────────────────────────────────────

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._finished:
            if exc_type is not None:
                self._record_terminal_error(
                    error=f"{exc_type.__name__}: {exc_val}",
                    tb_text=traceback.format_exc(),
                )
                self.finish(status="crashed")
            else:
                self.finish()
        return False  # never suppress exceptions

    # ── process hooks (atexit / signals / excepthook) ────────────

    def _setup_hooks(self) -> None:
        """Install process-level hooks so meta.json is updated on any exit."""
        atexit.register(self._on_exit)
        self._original_excepthook = sys.excepthook
        sys.excepthook = self._on_exception
        self._prev_sigterm = None
        self._prev_sigint = None
        try:
            self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_signal)
            self._prev_sigint = signal.signal(signal.SIGINT, self._on_signal)
        except ValueError:
            pass  # not on main thread

    def _teardown_hooks(self) -> None:
        atexit.unregister(self._on_exit)
        sys.excepthook = self._original_excepthook
        try:
            if self._prev_sigterm is not None:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
            if self._prev_sigint is not None:
                signal.signal(signal.SIGINT, self._prev_sigint)
        except ValueError:
            pass

    def _on_exit(self) -> None:
        if not self._finished:
            self.finish()

    def _on_exception(self, exc_type, exc_value, exc_tb) -> None:
        if not self._finished:
            self._record_terminal_error(
                error=f"{exc_type.__name__}: {exc_value}",
                tb_text="".join(
                    traceback.format_exception(exc_type, exc_value, exc_tb)
                ),
            )
            self.finish(status="crashed")
        self._original_excepthook(exc_type, exc_value, exc_tb)

    def _on_signal(self, signum, frame) -> None:
        if not self._finished:
            sig_name = signal.Signals(signum).name
            self._record_terminal_error(error=f"Signal: {sig_name}")
            self.finish(status="interrupted")
        sys.exit(128 + signum)

    # ── internals ─────────────────────────────────────────────────

    def _mark_mode_running(self) -> None:
        if self.mode is None:
            return

        self._modes_meta()[self.mode] = {
            "status": "running",
            "start_time": self._start_time,
            "start_iso": _isotime(self._start_time),
            "end_time": None,
            "end_iso": None,
            "duration_sec": None,
            "pid": os.getpid(),
            "command": sys.argv,
        }

    def _finish_mode(self, *, status: str, end: float, duration: float) -> None:
        if self.mode is None:
            return

        mode_meta = self._modes_meta().get(self.mode)
        if not isinstance(mode_meta, dict):
            self._mark_mode_running()
            mode_meta = self._modes_meta()[self.mode]

        mode_meta.update(
            status=status,
            end_time=end,
            end_iso=_isotime(end),
            duration_sec=duration,
        )
        self._clear_terminal_fields(mode_meta, status=status)

    def _record_terminal_error(
        self,
        *,
        error: str,
        tb_text: str | None = None,
    ) -> None:
        self._meta["error"] = error
        if tb_text is None:
            self._meta.pop("traceback", None)
        else:
            self._meta["traceback"] = tb_text

        if self.mode is None:
            return

        mode_meta = self._modes_meta().get(self.mode)
        if not isinstance(mode_meta, dict):
            self._mark_mode_running()
            mode_meta = self._modes_meta()[self.mode]

        mode_meta["error"] = error
        if tb_text is None:
            mode_meta.pop("traceback", None)
        else:
            mode_meta["traceback"] = tb_text

    @staticmethod
    def _clear_terminal_fields(target: dict[str, Any], *, status: str) -> None:
        if status in {"running", "completed"}:
            target.pop("error", None)
            target.pop("traceback", None)
            return
        if status == "interrupted":
            target.pop("traceback", None)

    def _modes_meta(self) -> dict[str, Any]:
        modes = self._meta.get("modes")
        if isinstance(modes, dict):
            return modes

        modes = {}
        self._meta["modes"] = modes
        return modes

    def _flush_meta(self) -> None:
        write_json(self.dir / "meta.json", self._meta)

    def __repr__(self) -> str:
        return f"Run(id={self.run_id!r}, dir={str(self.dir)!r}, status={self._meta['status']!r})"


def _infer_dir() -> Path:
    """Derive a default output directory from the git repo name or cwd.

    Returns ``<base>/<project_name>`` where *base* is resolved as:

    1. ``$ET_HOME`` environment variable (if set),
    2. ``~/euler_train`` (default).

    *project_name* is the git repository name, or the current working
    directory name when not inside a git repository.
    """
    et_home = os.environ.get("ET_HOME")
    base = Path(et_home) if et_home else Path.home() / "euler_train"

    from .git_info import _git

    toplevel = _git("rev-parse", "--show-toplevel")
    if toplevel is not None:
        project = Path(toplevel).name
    else:
        project = Path.cwd().name
    return base / project


def _infer_checkpoint_base(project_dir: Path) -> Path:
    """Derive a base directory for checkpoints.

    Resolution order:

    1. ``$SCRATCH/euler_train/<project>/checkpoints``
    2. ``<project_dir>/checkpoints`` (same volume as logs).
    """
    scratch = os.environ.get("SCRATCH")
    if scratch:
        project = project_dir.name
        return Path(scratch) / "euler_train" / project / "checkpoints"
    return project_dir / "checkpoints"


def _checkpoint_dir_slug(run_name: str | None, run_id: str) -> str:
    """Return the preferred leaf name for a checkpoint directory."""
    return _slugify(run_name) if run_name else run_id


def _disambiguate_path(path: Path, *, suffix: str) -> Path:
    """Return *path* or a collision-free sibling with *suffix* appended."""
    if not path.exists():
        return path

    candidate = path.with_name(f"{path.name}-{suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.name}-{suffix}-{counter}")
        counter += 1
    return candidate


def _slugify(text: str) -> str:
    """Turn *text* into a filesystem-safe slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return slug or "unnamed"


def _generate_run_id() -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    suffix = secrets.token_hex(2)
    return f"{ts}_{suffix}"


def _isotime(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _normalize_mode(mode: str | None) -> str | None:
    if mode is None:
        return None

    normalized = str(mode).strip()
    if not normalized:
        raise ValueError("mode must be a non-empty string when provided")
    return normalized


def _build_datasets_meta(
    datasets: Mapping[str, Any],
    existing: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    for split, dataset in datasets.items():
        split_name = str(split)
        merged[split_name] = _describe_dataset(dataset)

    return merged


def _build_evaluations_meta(
    evaluations: Mapping[str, Mapping[str, Any]],
    existing: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    for eval_key, eval_entry in evaluations.items():
        merged[str(eval_key)] = _build_single_evaluation(
            eval_entry, merged.get(str(eval_key)),
        )
    return merged


def _build_single_evaluation(
    entry: Mapping[str, Any],
    existing: Any,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    raw_datasets = entry.get("datasets")
    if raw_datasets is not None:
        result["datasets"] = _build_datasets_meta(
            datasets=raw_datasets,
            existing=result.get("datasets"),
        )

    for field in ("name", "status", "checkpoint", "metadata"):
        if field in entry:
            result[field] = entry[field]

    return result


_REQUIRED_MODALITY_FIELDS = ("used_as", "modality_type", "slot")


def _validate_modality_entries(entries: dict[str, dict[str, Any]]) -> None:
    for name, entry in entries.items():
        for field in _REQUIRED_MODALITY_FIELDS:
            if not entry.get(field):
                path = entry.get("path", "<unknown>")
                raise ValueError(
                    f"Modality {name!r} (path: {path}): missing required field "
                    f"{field!r}. Provide it via describe_for_runlog() or "
                    f"set 'properties.euler_train.{field}' explicitly."
                )


def _enrich_contract_entries(
    entries: dict[str, dict[str, Any]],
    ds_crawler_cache: dict[str, dict[str, Any]],
    *,
    is_hierarchical: bool,
) -> None:
    """Fill missing required fields in contract entries from ds_crawler."""
    for name, entry in entries.items():
        missing = [f for f in _REQUIRED_MODALITY_FIELDS if not entry.get(f)]
        if not missing:
            continue

        path = entry.get("path")
        if not path:
            continue

        descriptor = _get_ds_crawler_descriptor(path, ds_crawler_cache)
        properties = _properties_with_namespaces(descriptor.get("properties"))

        if "used_as" in missing:
            used_as, _ = _infer_used_as(
                name=name,
                properties=properties,
                is_hierarchical=is_hierarchical,
            )
            if used_as:
                entry["used_as"] = used_as

        if "modality_type" in missing:
            modality_type, _ = _infer_modality_type(
                name=name,
                path=path,
                descriptor=descriptor,
                properties=properties,
            )
            if modality_type:
                entry["modality_type"] = modality_type

        if "slot" in missing:
            used_as_val = entry.get("used_as")
            modality_type_val = entry.get("modality_type")
            if used_as_val and modality_type_val:
                slot = _infer_slot(
                    name=name,
                    used_as=used_as_val,
                    modality_type=modality_type_val,
                    properties=properties,
                )
                if slot:
                    entry["slot"] = slot


def _describe_dataset(dataset: Any) -> dict[str, Any]:
    described = _describe_dataset_via_contract(dataset)
    if described is not None:
        ds_crawler_cache: dict[str, dict[str, Any]] = {}
        _enrich_contract_entries(
            described.get("modalities", {}), ds_crawler_cache,
            is_hierarchical=False,
        )
        _enrich_contract_entries(
            described.get("hierarchical_modalities", {}), ds_crawler_cache,
            is_hierarchical=True,
        )
        _validate_modality_entries(described.get("modalities", {}))
        _validate_modality_entries(described.get("hierarchical_modalities", {}))
        return described

    modality_paths = _extract_modality_paths(
        dataset=dataset,
        method_name="modality_paths",
        attr_name="_modalities",
    )
    hierarchical_paths = _extract_modality_paths(
        dataset=dataset,
        method_name="hierarchical_modality_paths",
        attr_name="_hierarchical_modalities",
    )
    hierarchical_levels = _extract_hierarchical_levels(dataset)

    regular_names = list(modality_paths.keys())
    ds_crawler_meta_cache: dict[str, dict[str, Any]] = {}

    modalities: dict[str, dict[str, Any]] = {}
    for name, path in modality_paths.items():
        descriptor = _get_ds_crawler_descriptor(path, ds_crawler_meta_cache)
        modalities[name] = _build_modality_entry(
            name=name,
            path=path,
            descriptor=descriptor,
            is_hierarchical=False,
            regular_modality_names=regular_names,
            hierarchy_levels=None,
        )

    hierarchical_modalities: dict[str, dict[str, Any]] = {}
    for name, path in hierarchical_paths.items():
        descriptor = _get_ds_crawler_descriptor(path, ds_crawler_meta_cache)
        hierarchical_modalities[name] = _build_modality_entry(
            name=name,
            path=path,
            descriptor=descriptor,
            is_hierarchical=True,
            regular_modality_names=regular_names,
            hierarchy_levels=hierarchical_levels.get(name),
        )

    return {
        "modalities": modalities,
        "hierarchical_modalities": hierarchical_modalities,
    }


def _describe_dataset_via_contract(dataset: Any) -> dict[str, Any] | None:
    describe = getattr(dataset, "describe_for_runlog", None)
    if not callable(describe):
        return None

    try:
        raw = describe()
    except Exception:
        return None

    if not isinstance(raw, Mapping):
        return None

    modality_paths = _extract_modality_paths(
        dataset=dataset,
        method_name="modality_paths",
        attr_name="_modalities",
    )
    hierarchical_paths = _extract_modality_paths(
        dataset=dataset,
        method_name="hierarchical_modality_paths",
        attr_name="_hierarchical_modalities",
    )

    modalities = _normalize_contract_entries(
        raw_entries=raw.get("modalities"),
        fallback_paths=modality_paths,
    )
    hierarchical_modalities = _normalize_contract_entries(
        raw_entries=raw.get("hierarchical_modalities"),
        fallback_paths=hierarchical_paths,
    )
    return {
        "modalities": modalities,
        "hierarchical_modalities": hierarchical_modalities,
    }


def _normalize_contract_entries(
    *,
    raw_entries: Any,
    fallback_paths: dict[str, str],
) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_entries, Mapping):
        raw_entries = {}

    normalized: dict[str, dict[str, Any]] = {}
    for name, raw_entry in raw_entries.items():
        key = str(name)
        if not isinstance(raw_entry, Mapping):
            path = fallback_paths.get(key)
            normalized[key] = {"path": path} if path is not None else {}
            continue

        entry: dict[str, Any] = {
            str(k): v
            for k, v in raw_entry.items()
            if v is not None
        }
        if "path" in entry:
            entry["path"] = str(entry["path"])
        elif key in fallback_paths:
            entry["path"] = fallback_paths[key]
        normalized[key] = entry

    for name, path in fallback_paths.items():
        normalized.setdefault(name, {"path": path})

    return normalized


def _extract_modality_paths(
    dataset: Any,
    method_name: str,
    attr_name: str,
) -> dict[str, str]:
    method = getattr(dataset, method_name, None)
    if callable(method):
        raw = method()
        if isinstance(raw, Mapping):
            return {
                str(name): str(path)
                for name, path in raw.items()
                if path is not None
            }

    raw_attr = getattr(dataset, attr_name, None)
    if isinstance(raw_attr, Mapping):
        result: dict[str, str] = {}
        for name, modality in raw_attr.items():
            path = getattr(modality, "origin_path", None) or getattr(modality, "path", None)
            if path is not None:
                result[str(name)] = str(path)
        return result

    return {}


def _extract_hierarchical_levels(
    dataset: Any,
) -> dict[str, list[tuple[str, ...]]]:
    raw = getattr(dataset, "_hierarchical_lookups", None)
    if not isinstance(raw, Mapping):
        return {}

    result: dict[str, list[tuple[str, ...]]] = {}
    for name, levels in raw.items():
        if not isinstance(levels, Mapping):
            continue
        cleaned_levels: list[tuple[str, ...]] = []
        for level in levels.keys():
            if isinstance(level, tuple):
                cleaned_levels.append(tuple(str(part) for part in level))
        result[str(name)] = cleaned_levels
    return result


def _get_ds_crawler_descriptor(
    path: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if path not in cache:
        cache[path] = _read_ds_crawler_descriptor(path)
    return cache[path]


def _read_ds_crawler_descriptor(path: str) -> dict[str, Any]:
    try:
        from ds_crawler import get_dataset_contract, load_dataset_config
    except Exception:
        return {}

    try:
        cfg = load_dataset_config({"path": path})
    except Exception:
        try:
            contract = get_dataset_contract(path)
        except Exception:
            return {}
        return {
            "properties": contract.to_properties_dict(),
            "modality_key": contract.modality_key,
        }

    descriptor: dict[str, Any] = {}
    try:
        contract = get_dataset_contract(path)
    except Exception:
        contract = None
    if contract is not None:
        descriptor["properties"] = contract.to_properties_dict()
        descriptor["modality_key"] = contract.modality_key
    else:
        properties = cfg.properties if isinstance(cfg.properties, dict) else {}
        descriptor["properties"] = dict(properties)

    cfg_type = _as_non_empty_str(getattr(cfg, "type", None))
    if cfg_type is not None:
        descriptor["modality_key"] = cfg_type

    hierarchy_regex = _as_non_empty_str(getattr(cfg, "hierarchy_regex", None))
    if hierarchy_regex is not None:
        descriptor["hierarchy_regex"] = hierarchy_regex

    return descriptor


def _build_modality_entry(
    name: str,
    path: str,
    descriptor: Mapping[str, Any],
    *,
    is_hierarchical: bool,
    regular_modality_names: list[str],
    hierarchy_levels: list[tuple[str, ...]] | None,
) -> dict[str, Any]:
    properties = _properties_with_namespaces(descriptor.get("properties"))
    used_as, used_as_explicit = _infer_used_as(
        name=name,
        properties=properties,
        is_hierarchical=is_hierarchical,
    )
    modality_type, modality_type_explicit = _infer_modality_type(
        name=name,
        path=path,
        descriptor=descriptor,
        properties=properties,
    )

    if used_as is None:
        raise ValueError(
            f"Modality {name!r} (path: {path}): could not determine 'used_as'. "
            f"Set 'properties.euler_train.used_as' explicitly."
        )
    if modality_type is None:
        raise ValueError(
            f"Modality {name!r} (path: {path}): could not determine 'modality_type'. "
            f"Set 'properties.euler_train.modality_type' explicitly."
        )

    if not used_as_explicit:
        warnings.warn(
            f"Modality {name!r} (path: {path}): 'used_as' was inferred as "
            f"{used_as!r} from the modality name. "
            f"Set 'properties.euler_train.used_as' explicitly to suppress this warning.",
            stacklevel=2,
        )
    if not modality_type_explicit:
        warnings.warn(
            f"Modality {name!r} (path: {path}): 'modality_type' was inferred as "
            f"{modality_type!r} from the modality name/path. "
            f"Set 'properties.euler_train.modality_type' explicitly to suppress this warning.",
            stacklevel=2,
        )

    slot = _infer_slot(
        name=name,
        used_as=used_as,
        modality_type=modality_type,
        properties=properties,
    )
    if slot is None:
        raise ValueError(
            f"Modality {name!r} (path: {path}): could not determine 'slot'. "
            f"Set 'properties.euler_train.slot' explicitly."
        )

    entry: dict[str, Any] = {
        "path": path,
        "used_as": used_as,
        "slot": slot,
        "modality_type": modality_type,
    }

    if is_hierarchical:
        hierarchy_scope = _infer_hierarchy_scope(
            descriptor=descriptor,
            properties=properties,
            hierarchy_levels=hierarchy_levels,
        )
        if hierarchy_scope is not None:
            entry["hierarchy_scope"] = hierarchy_scope

        applies_to = _infer_applies_to(properties, regular_modality_names)
        entry["applies_to"] = applies_to

    return entry


def _properties_with_namespaces(value: Any) -> list[Mapping[str, Any]]:
    if not isinstance(value, Mapping):
        return []

    # Only consume explicit, namespaced metadata blocks.
    layers: list[Mapping[str, Any]] = []
    for namespaced_key in _DATASET_META_NAMESPACES:
        namespaced = value.get(namespaced_key)
        if isinstance(namespaced, Mapping):
            layers.append(namespaced)
    return layers


def _resolve_property(
    properties: list[Mapping[str, Any]],
    *keys: str,
) -> Any:
    for layer in properties:
        for key in keys:
            value = layer.get(key)
            if value is not None:
                return value
    return None


def _infer_used_as(
    *,
    name: str,
    properties: list[Mapping[str, Any]],
    is_hierarchical: bool,
) -> tuple[str | None, bool]:
    """Return ``(value, is_explicit)``."""
    explicit = _as_non_empty_str(
        _resolve_property(properties, "used_as")
    )
    if explicit is not None:
        return explicit, True

    lowered = name.lower()
    if any(
        token in lowered
        for token in ("condition", "cond", "camera", "intrinsics", "extrinsics", "pose")
    ):
        return "condition", False
    if any(token in lowered for token in ("target", "gt", "label", "clear", "clean")):
        return "target", False
    if any(token in lowered for token in ("input", "source", "src", "hazy", "noisy", "raw")):
        return "input", False
    if is_hierarchical:
        return "condition", False
    return None, False


def _infer_modality_type(
    *,
    name: str,
    path: str,
    descriptor: Mapping[str, Any],
    properties: list[Mapping[str, Any]],
) -> tuple[str | None, bool]:
    """Return ``(value, is_explicit)``."""
    explicit = _as_non_empty_str(descriptor.get("modality_key"))
    if explicit is not None:
        return explicit, True

    explicit = _as_non_empty_str(descriptor.get("modality_type"))
    if explicit is not None:
        return explicit, True

    explicit = _as_non_empty_str(
        _resolve_property(properties, "modality_type")
    )
    if explicit is not None:
        return explicit, True

    lowered = f"{name} {path}".lower()
    if any(token in lowered for token in ("rgb", "image", "img", "color", "colour")):
        return "rgb", False
    if any(token in lowered for token in ("depth", "disparity")):
        return "depth", False
    if any(token in lowered for token in ("segmentation", "segment", "mask", "semantic")):
        return "segmentation", False
    return None, False


def _infer_slot(
    *,
    name: str,
    used_as: str | None,
    modality_type: str | None,
    properties: list[Mapping[str, Any]],
) -> str | None:
    explicit = _as_non_empty_str(
        _resolve_property(properties, "slot")
    )
    if explicit is not None:
        return explicit
    if used_as is None:
        return None

    task = _as_non_empty_str(
        _resolve_property(properties, "task")
    )
    leaf = modality_type or name
    if task is not None:
        return f"{task}.{used_as}.{leaf}"
    return f"{used_as}.{leaf}"


def _infer_hierarchy_scope(
    *,
    descriptor: Mapping[str, Any],
    properties: list[Mapping[str, Any]],
    hierarchy_levels: list[tuple[str, ...]] | None,
) -> str | None:
    explicit = _as_non_empty_str(
        _resolve_property(properties, "hierarchy_scope")
    )
    if explicit is not None:
        return explicit

    regex_scope = _infer_hierarchy_scope_from_regex(descriptor.get("hierarchy_regex"))
    if regex_scope is not None:
        return regex_scope

    return _infer_hierarchy_scope_from_levels(hierarchy_levels)


def _infer_hierarchy_scope_from_regex(value: Any) -> str | None:
    regex = _as_non_empty_str(value)
    if regex is None:
        return None
    try:
        pattern = re.compile(regex)
    except re.error:
        return None

    if not pattern.groupindex:
        return None

    ordered_names = [
        name for name, _ in sorted(pattern.groupindex.items(), key=lambda item: item[1])
    ]
    if not ordered_names:
        return None
    return "_".join(ordered_names)


def _infer_hierarchy_scope_from_levels(
    levels: list[tuple[str, ...]] | None,
) -> str | None:
    if not levels:
        return None

    non_root_levels = [level for level in levels if level]
    if not non_root_levels:
        return "root"

    max_depth = max(len(level) for level in non_root_levels)
    deepest_levels = [level for level in non_root_levels if len(level) == max_depth]

    tokens: list[str] = []
    for idx in range(max_depth):
        candidates = {
            token
            for token in (_extract_hierarchy_token(level[idx]) for level in deepest_levels)
            if token is not None
        }
        if len(candidates) != 1:
            return f"level_{max_depth}"
        tokens.append(next(iter(candidates)))

    if not tokens:
        return f"level_{max_depth}"
    return "_".join(tokens)


def _extract_hierarchy_token(value: str) -> str | None:
    for separator in (":", "=", "__", "_", "-"):
        if separator not in value:
            continue
        prefix = value.split(separator, 1)[0].strip().lower()
        if prefix and any(ch.isalpha() for ch in prefix):
            return prefix
    return None


def _infer_applies_to(
    properties: list[Mapping[str, Any]],
    regular_modality_names: list[str],
) -> list[str]:
    explicit = _as_string_list(
        _resolve_property(
            properties,
            "applies_to",
        )
    )
    if explicit is not None:
        return explicit
    return list(regular_modality_names)


def _as_non_empty_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None

    if isinstance(value, (list, tuple, set)):
        parsed = [_as_non_empty_str(item) for item in value]
        return [item for item in parsed if item is not None]

    single = _as_non_empty_str(value)
    if single is None:
        return []
    return [single]
