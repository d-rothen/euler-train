"""Core Run class – the single object a researcher interacts with."""
from __future__ import annotations

import atexit
import os
import secrets
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .serialization import append_jsonl, normalize_config, write_json
from .outputs import save_output_tree


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
    ) -> None:
        self.run_id: str = _generate_run_id()
        self.dir = Path(dir) / "runs" / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)

        self._output_formats: dict[str, str] = output_formats or {}
        self._start_time = time.time()
        self._finished = False
        self._gpu_handle: Any | None = None
        self._gpu_available: bool | None = None  # None = not yet probed
        self._gpu_stats_every: int = gpu_stats_every

        # ── config ────────────────────────────────────────────────
        config_dict = normalize_config(config)
        write_json(self.dir / "config.json", config_dict)
        self.config: dict = config_dict

        # ── meta ──────────────────────────────────────────────────
        self._meta: dict[str, Any] = {
            "run_id": self.run_id,
            "status": "running",
            "start_time": self._start_time,
            "start_iso": _isotime(self._start_time),
            "end_time": None,
            "end_iso": None,
            "duration_sec": None,
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "command": sys.argv,
        }
        if meta:
            self._meta.update(meta)
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
