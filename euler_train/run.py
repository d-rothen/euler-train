"""Core Run class – the single object a researcher interacts with."""
from __future__ import annotations

import atexit
import os
import re
import secrets
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

_DATASET_META_NAMESPACES = ("euler_loading", "euler_train")


class Run:
    """A single experiment run backed by a directory on disk.

    Typically created via :func:`runlog.init` rather than directly.
    """

    def __init__(
        self,
        dir: str | Path,
        config: Any = None,
        meta: dict | None = None,
        output_formats: dict[str, str] | None = None,
        gpu_stats_every: int = 100,
        run_id: str | None = None,
        datasets: dict[str, Any] | None = None,
        run_name: str | None = None,
    ) -> None:
        resuming = run_id is not None
        self.run_id: str = run_id if resuming else _generate_run_id()
        self.dir = Path(dir) / "runs" / self.run_id

        if resuming and not self.dir.exists():
            raise FileNotFoundError(
                f"Cannot resume run {self.run_id!r}: "
                f"directory {self.dir} does not exist"
            )
        self.dir.mkdir(parents=True, exist_ok=True)

        self._output_formats: dict[str, str] = output_formats or {}
        self._start_time = time.time()
        self._finished = False
        self._gpu_handle: Any | None = None
        self._gpu_available: bool | None = None  # None = not yet probed
        self._gpu_stats_every: int = gpu_stats_every
        self.run_name: str | None = run_name

        # ── config ────────────────────────────────────────────────
        if resuming:
            config_path = self.dir / "config.json"
            self.config: dict = read_json(config_path) if config_path.exists() else {}
            if config is not None:
                self.config.update(normalize_config(config))
        else:
            self.config: dict = normalize_config(config)

        write_json(self.dir / "config.json", self.config)

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

        # ── datasets (optional euler_loading integration) ────────
        if datasets is not None:
            self._meta["datasets"] = _build_datasets_meta(
                datasets=datasets,
                existing=self._meta.get("datasets"),
            )
        self._flush_meta()
        self._setup_hooks()

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

    def save_checkpoint(
        self,
        model: Any,
        *,
        epoch: int,
        optimizer: Any = None,
        **extra: Any,
    ) -> Path:
        """Save a checkpoint to ``checkpoints/epoch_{N}.pt``.

        If *model* (or *optimizer*) exposes ``.state_dict()``, it is called
        automatically.  Extra keyword arguments are included in the saved
        dict.
        """
        import torch

        ckpt_dir = self.dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"epoch_{epoch}.pt"

        state: dict[str, Any] = {"epoch": epoch}
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
        return path

    # ── lifecycle ─────────────────────────────────────────────────

    def finish(self, status: str = "completed") -> None:
        """Mark the run as finished and write final metadata."""
        if self._finished:
            return
        self._finished = True
        self._teardown_hooks()
        end = time.time()
        self._meta.update(
            status=status,
            end_time=end,
            end_iso=_isotime(end),
            duration_sec=round(end - self._start_time, 3),
        )
        self._flush_meta()

    # ── context manager ───────────────────────────────────────────

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if not self._finished:
            if exc_type is not None:
                self._meta["error"] = f"{exc_type.__name__}: {exc_val}"
                self._meta["traceback"] = traceback.format_exc()
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
            self._meta["error"] = f"{exc_type.__name__}: {exc_value}"
            self._meta["traceback"] = "".join(
                traceback.format_exception(exc_type, exc_value, exc_tb)
            )
            self.finish(status="crashed")
        self._original_excepthook(exc_type, exc_value, exc_tb)

    def _on_signal(self, signum, frame) -> None:
        if not self._finished:
            sig_name = signal.Signals(signum).name
            self._meta["error"] = f"Signal: {sig_name}"
            self.finish(status="interrupted")
        sys.exit(128 + signum)

    # ── internals ─────────────────────────────────────────────────

    def _flush_meta(self) -> None:
        write_json(self.dir / "meta.json", self._meta)

    def __repr__(self) -> str:
        return f"Run(id={self.run_id!r}, dir={str(self.dir)!r}, status={self._meta['status']!r})"


def _generate_run_id() -> str:
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    suffix = secrets.token_hex(2)
    return f"{ts}_{suffix}"


def _isotime(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _build_datasets_meta(
    datasets: Mapping[str, Any],
    existing: Any,
) -> dict[str, Any]:
    merged: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}

    for split, dataset in datasets.items():
        split_name = str(split)
        merged[split_name] = _describe_dataset(dataset)

    return merged


def _describe_dataset(dataset: Any) -> dict[str, Any]:
    described = _describe_dataset_via_contract(dataset)
    if described is not None:
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
            path = getattr(modality, "path", None)
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
        from ds_crawler import load_dataset_config
    except Exception:
        return {}

    try:
        cfg = load_dataset_config({"path": path})
    except Exception:
        return {}

    properties = cfg.properties if isinstance(cfg.properties, dict) else {}
    descriptor: dict[str, Any] = {"properties": dict(properties)}

    cfg_type = _as_non_empty_str(getattr(cfg, "type", None))
    if cfg_type is not None:
        descriptor["modality_type"] = cfg_type

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
    used_as = _infer_used_as(
        name=name,
        properties=properties,
        is_hierarchical=is_hierarchical,
    )
    modality_type = _infer_modality_type(
        name=name,
        path=path,
        descriptor=descriptor,
        properties=properties,
    )
    slot = _infer_slot(
        name=name,
        used_as=used_as,
        modality_type=modality_type,
        properties=properties,
    )

    entry: dict[str, Any] = {"path": path}
    if used_as is not None:
        entry["used_as"] = used_as
    if slot is not None:
        entry["slot"] = slot
    if modality_type is not None:
        entry["modality_type"] = modality_type

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
) -> str | None:
    explicit = _as_non_empty_str(
        _resolve_property(properties, "used_as")
    )
    if explicit is not None:
        return explicit

    lowered = name.lower()
    if any(
        token in lowered
        for token in ("condition", "cond", "camera", "intrinsics", "extrinsics", "pose")
    ):
        return "condition"
    if any(token in lowered for token in ("target", "gt", "label", "clear", "clean")):
        return "target"
    if any(token in lowered for token in ("input", "source", "src", "hazy", "noisy", "raw")):
        return "input"
    if is_hierarchical:
        return "condition"
    return None


def _infer_modality_type(
    *,
    name: str,
    path: str,
    descriptor: Mapping[str, Any],
    properties: list[Mapping[str, Any]],
) -> str | None:
    explicit = _as_non_empty_str(descriptor.get("modality_type"))
    if explicit is not None:
        return explicit

    explicit = _as_non_empty_str(
        _resolve_property(properties, "modality_type")
    )
    if explicit is not None:
        return explicit

    lowered = f"{name} {path}".lower()
    if any(token in lowered for token in ("rgb", "image", "img", "color", "colour")):
        return "rgb"
    if any(token in lowered for token in ("depth", "disparity")):
        return "depth"
    if any(token in lowered for token in ("segmentation", "segment", "mask", "semantic")):
        return "segmentation"
    return None


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
