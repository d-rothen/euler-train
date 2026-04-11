"""Save prediction / ground-truth / auxiliary outputs to disk."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def save_output_tree(
    type_dir: Path,
    slots: dict[str, Any],
    format_overrides: dict[str, str],
    visualization_overrides: dict[str, Any],
    output_type: str,
) -> dict[str, Any]:
    """Persist all slots (pred, gt, input, aux/…) for one *output_type*.

    *slots* example::

        {
            "pred": array,
            "gt":   array,
            "aux":  {"transmission": array, "attention": array},
        }

    Returns a dict mapping slot keys to their manifest entries.
    """
    manifest: dict[str, Any] = {}
    for slot_name, data in slots.items():
        if data is None:
            continue
        if slot_name == "aux" and isinstance(data, dict):
            for aux_name, aux_data in data.items():
                if aux_data is None:
                    continue
                manifest[f"aux/{aux_name}"] = _save_slot(
                    type_dir / "aux" / aux_name,
                    aux_data,
                    output_type,
                    aux_name,
                    format_overrides,
                    visualization_overrides,
                )
        else:
            manifest[slot_name] = _save_slot(
                type_dir / slot_name,
                data,
                output_type,
                slot_name,
                format_overrides,
                visualization_overrides,
            )
    return manifest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save_slot(
    slot_dir: Path,
    data: Any,
    output_type: str,
    leaf_name: str,
    format_overrides: dict[str, str],
    visualization_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Save items for one slot and return a manifest entry."""
    slot_dir.mkdir(parents=True, exist_ok=True)
    visualization = _resolve_visualization(
        output_type, leaf_name, visualization_overrides,
    )

    # Dict with string keys → named outputs (e.g. {"scene_042": img}).
    if isinstance(data, dict):
        files: list[dict[str, Any]] = []
        for name, raw in data.items():
            item = _prepare(raw)
            fmt = _resolve_format(item, output_type, leaf_name, format_overrides)
            _save_item(slot_dir / f"{name}.{fmt}", item, fmt, visualization)
            files.append({"sample_id": name, "filename": f"{name}.{fmt}", "format": fmt})
        return {"id_mode": "named", "files": files}

    items = _unpack(data)
    files = []
    for idx, item in enumerate(items):
        fmt = _resolve_format(item, output_type, leaf_name, format_overrides)
        _save_item(slot_dir / f"{idx:04d}.{fmt}", item, fmt, visualization)
        files.append({"sample_id": idx, "filename": f"{idx:04d}.{fmt}", "format": fmt})
    return {"id_mode": "indexed", "files": files}


# ---- normalisation -------------------------------------------------------

def _unpack(data: Any) -> list:
    """Normalise *data* into a flat list of saveable items."""
    if isinstance(data, (list, tuple)):
        return [_prepare(d) for d in data]
    prepared = _prepare(data)
    # 4-D numpy → treat as batch
    if isinstance(prepared, np.ndarray) and prepared.ndim == 4:
        return [prepared[i] for i in range(prepared.shape[0])]
    return [prepared]


def _prepare(data: Any) -> Any:
    """Convert torch tensors → numpy; pass PIL images through unchanged."""
    # PIL Image — return as-is
    try:
        from PIL import Image as _PIL
        if isinstance(data, _PIL.Image):
            return data
    except ImportError:
        pass

    # torch Tensor → numpy, channels-first → channels-last
    if hasattr(data, "detach"):
        tensor = data.detach().cpu()
        try:
            arr: np.ndarray = tensor.numpy()
        except TypeError:
            # NumPy cannot represent some torch dtypes (for example bfloat16).
            arr = tensor.float().numpy()
        # (C, H, W) → (H, W, C)  when C looks like a channel dim
        if (
            arr.ndim == 3
            and arr.shape[0] in (1, 3, 4)
            and min(arr.shape[1:]) > 4
        ):
            arr = np.transpose(arr, (1, 2, 0))
        # (B, C, H, W) → (B, H, W, C)
        elif (
            arr.ndim == 4
            and arr.shape[1] in (1, 3, 4)
            and min(arr.shape[2:]) > 4
        ):
            arr = np.transpose(arr, (0, 2, 3, 1))
        return arr

    return np.asarray(data)


# ---- format inference ----------------------------------------------------

def _is_image_like(arr: np.ndarray) -> bool:
    """Heuristic: does this array look like it should be saved as a PNG?"""
    if arr.ndim == 2 and arr.dtype == np.uint8:
        return True  # grayscale uint8
    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        return True  # HxWx{1,3,4}
    return False


def _resolve_override(
    output_type: str,
    leaf_name: str,
    overrides: Mapping[str, Any],
) -> Any:
    specific = f"{output_type}.{leaf_name}"
    if specific in overrides:
        return overrides[specific]
    if output_type in overrides:
        return overrides[output_type]
    if leaf_name in overrides:
        return overrides[leaf_name]
    return None


def _resolve_format(
    item: Any,
    output_type: str,
    leaf_name: str,
    overrides: dict[str, str],
) -> str:
    """Pick save format: check overrides (most-specific first), then infer."""
    override = _resolve_override(output_type, leaf_name, overrides)
    if override is not None:
        return str(override)

    # PIL Image
    try:
        from PIL import Image as _PIL
        if isinstance(item, _PIL.Image):
            return "png"
    except ImportError:
        pass

    if isinstance(item, np.ndarray) and _is_image_like(item):
        return "png"
    return "npy"


def _resolve_visualization(
    output_type: str,
    leaf_name: str,
    overrides: Mapping[str, Any],
) -> dict[str, Any] | None:
    override = _resolve_override(output_type, leaf_name, overrides)
    if override is None:
        return None
    if isinstance(override, str):
        return {"mode": override}
    if isinstance(override, Mapping):
        return dict(override)
    raise TypeError(
        "Visualization overrides must be strings or mappings, "
        f"got {type(override)!r}",
    )


# ---- writers -------------------------------------------------------------

def _save_item(
    path: Path,
    item: Any,
    fmt: str,
    visualization: dict[str, Any] | None = None,
) -> None:
    if fmt == "png":
        _save_png(path, item, visualization)
    elif fmt == "npy":
        np.save(str(path), item if isinstance(item, np.ndarray) else np.asarray(item))
    elif fmt == "npz":
        np.savez_compressed(
            str(path),
            data=item if isinstance(item, np.ndarray) else np.asarray(item),
        )
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")


def _save_png(
    path: Path,
    item: Any,
    visualization: dict[str, Any] | None = None,
) -> None:
    from PIL import Image

    # PIL Image — save directly
    if isinstance(item, Image.Image):
        item.save(str(path))
        return

    arr: np.ndarray = item

    # float → visualization policy → uint8
    if np.issubdtype(arr.dtype, np.floating):
        arr = _render_float_png(arr, visualization)
    elif arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(str(path))
    elif arr.ndim == 3:
        c = arr.shape[2]
        mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(c)
        if mode is None:
            raise ValueError(f"Cannot save array with {c} channels as PNG")
        plane = arr[:, :, 0] if c == 1 else arr
        Image.fromarray(plane, mode=mode).save(str(path))
    else:
        raise ValueError(f"Cannot save {arr.ndim}D array as PNG")


def _render_float_png(
    arr: np.ndarray,
    visualization: dict[str, Any] | None,
) -> np.ndarray:
    policy = {"mode": "unit_range"}
    if visualization is not None:
        policy.update(visualization)

    mode = str(policy.get("mode", "unit_range"))
    finite = np.isfinite(arr)

    if mode == "unit_range":
        scaled = np.asarray(arr, dtype=np.float32)
    elif mode == "minmax":
        scaled = _normalize_minmax(arr, finite)
    elif mode == "percentile":
        pmin = float(policy.get("pmin", 1.0))
        pmax = float(policy.get("pmax", 99.0))
        if pmax <= pmin:
            raise ValueError(
                "Percentile visualization requires pmax > pmin, "
                f"got pmin={pmin} pmax={pmax}",
            )
        scaled = _normalize_percentile(arr, finite, pmin, pmax)
    elif mode == "fixed_range":
        vmin = float(policy.get("vmin", 0.0))
        vmax = float(policy.get("vmax", 1.0))
        if vmax <= vmin:
            raise ValueError(
                "Fixed-range visualization requires vmax > vmin, "
                f"got vmin={vmin} vmax={vmax}",
            )
        scaled = (np.asarray(arr, dtype=np.float32) - vmin) / (vmax - vmin)
    else:
        raise ValueError(f"Unsupported visualization mode: {mode!r}")

    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    if bool(policy.get("invert", False)):
        scaled = 1.0 - scaled
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _normalize_minmax(arr: np.ndarray, finite: np.ndarray) -> np.ndarray:
    values = arr[finite]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        return np.zeros_like(arr, dtype=np.float32)
    return (np.asarray(arr, dtype=np.float32) - vmin) / (vmax - vmin)


def _normalize_percentile(
    arr: np.ndarray,
    finite: np.ndarray,
    pmin: float,
    pmax: float,
) -> np.ndarray:
    values = arr[finite]
    if values.size == 0:
        return np.zeros_like(arr, dtype=np.float32)

    lo = float(np.percentile(values, pmin))
    hi = float(np.percentile(values, pmax))
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (np.asarray(arr, dtype=np.float32) - lo) / (hi - lo)
