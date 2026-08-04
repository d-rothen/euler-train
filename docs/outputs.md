# Saving model outputs

Metrics answer “how well?”; saved outputs answer “what did the model actually
produce?” euler-train stores both the values and a small manifest describing
how each snapshot is laid out.

PNG output requires the `[images]` extra:

```bash
pip install "euler-train[images]"
```

## Output types and slots

Call `save_outputs()` with one keyword argument per semantic output type. Each
output type contains one or more slots:

| Slot | Value |
|---|---|
| `pred` | Model prediction |
| `gt` | Ground truth or target |
| `input` | Model input |
| `aux` | Mapping of named auxiliary outputs; every value gets its own directory |

```python
output_dir = run.save_outputs(
    epoch=1,
    step=500,
    rgb={
        "pred": predicted_rgb,
        "gt": target_rgb,
        "input": hazy_rgb,
    },
    depth={
        "pred": predicted_depth,
        "gt": target_depth,
        "aux": {
            "confidence": confidence,
            "transmission": transmission,
        },
    },
)
```

This writes `outputs/epoch_1_step_500/` and returns that path. If only one of
`epoch` and `step` is supplied, only that component is used; if neither is
supplied, the snapshot directory is named `unspecified`.

Pass `None` for a slot or entire output type to skip it.

## Accepted values

A slot value may be:

- one NumPy array, PyTorch tensor or Pillow image;
- a list or tuple of those values;
- a four-dimensional NumPy array or tensor, split along its leading batch
  dimension; or
- a mapping from string sample IDs to values.

PyTorch tensors are detached, moved to CPU and converted to NumPy. Common
channels-first image tensors in `(C, H, W)` or `(B, C, H, W)` layout are
transposed to channels-last when the channel count is 1, 3 or 4.

Lists and batched arrays use sequential filenames (`0000.png`, `0001.png`, …).
A mapping uses its keys as filenames:

```python
run.save_outputs(
    epoch=1,
    step=500,
    rgb={
        "pred": {
            "scene_042": prediction_a,
            "scene_117": prediction_b,
        },
    },
)
```

This produces `rgb/pred/scene_042.png` and
`rgb/pred/scene_117.png`. Direct mapping keys become basenames, so they must be
unique, non-empty and filesystem-safe. Prefer `save_outputs_from_batch()` when
IDs originate outside your code; it sanitizes them automatically.

## Batch-aware naming

```python
run.save_outputs_from_batch(
    batch=batch,
    epoch=epoch,
    step=step,
    metadata={"dataset": "vkitti2", "split": "val"},
    dataset=val_dataset,
    n=8,
    rgb={"pred": prediction, "gt": batch["rgb"]},
    depth={
        "pred": depth_prediction,
        "aux": {"confidence": confidence},
    },
)
```

The helper:

1. reads the first list of IDs found under `full_id`, then `id`;
2. replaces characters outside `[A-Za-z0-9._-]` with `_`;
3. slices tensors to `n` items when requested and moves torch tensors to CPU;
4. wraps each slot in a named mapping; and
5. optionally adds `dataset.describe_id_schema()` below
   `metadata["id_schema"]`.

Pass `sample_ids=[...]` to override the batch IDs, or
`include_id_schema=False` to disable schema enrichment. An explicit
`id_schema={...}` takes precedence over the dataset method. Existing
`metadata["id_schema"]` always wins.

Metadata is optional, but when supplied it must contain a non-empty string
`dataset` field:

```python
metadata={
    "dataset": "vkitti2",
    "split": "val",
    "scene": "Scene01",
}
```

## Manifest

Every non-empty snapshot receives a `manifest.json`:

```json
{
  "version": 1,
  "epoch": 1,
  "step": 500,
  "metadata": {
    "dataset": "vkitti2",
    "split": "val"
  },
  "output_types": {
    "rgb": {
      "pred": {
        "id_mode": "named",
        "files": [
          {
            "sample_id": "scene_042",
            "filename": "scene_042.png",
            "format": "png"
          }
        ]
      },
      "gt": {
        "id_mode": "indexed",
        "files": [
          {
            "sample_id": 0,
            "filename": "0000.png",
            "format": "png"
          }
        ]
      }
    }
  },
  "format_overrides": {},
  "visualization_overrides": {}
}
```

When streaming is active and both epoch and step are present, the same logical
snapshot is emitted as an `output_snapshot` event.

## Format inference

| Value after conversion | Default format |
|---|---|
| Pillow image | PNG |
| `uint8` array shaped `(H, W)` | PNG |
| Any array shaped `(H, W, 1)`, `(H, W, 3)` or `(H, W, 4)` | PNG |
| Everything else, including a float `(H, W)` depth map | NPY |

Override the inferred format when initializing the run. Resolution is
most-specific first: `<output>.<slot>`, then `<output>`, then `<slot>`.

```python
run = euler_train.init(
    dir="experiments/depth",
    output_formats={
        "depth.pred": "npz",
        "depth": "npy",
        "confidence": "npz",
    },
)
```

Supported values are `png`, `npy` and `npz`. NPZ files store the array under
the `data` key.

## Float-to-PNG visualization

Floating-point image-like arrays must be mapped to 8-bit pixels. The default
policy clips values to `[0, 1]`; configure another policy at initialization:

```python
run = euler_train.init(
    dir="experiments/depth",
    output_visualization={
        "depth": {"mode": "percentile", "pmin": 1, "pmax": 99},
        "depth.pred": {
            "mode": "fixed_range",
            "vmin": 0.0,
            "vmax": 80.0,
        },
        "confidence": "minmax",
    },
)
```

| Mode | Parameters | Behavior |
|---|---|---|
| `unit_range` | — | Clip to `[0, 1]`, then scale to `[0, 255]`. |
| `minmax` | — | Normalize using the finite minimum and maximum of each item. |
| `percentile` | `pmin`, `pmax` | Normalize between finite-value percentiles. |
| `fixed_range` | `vmin`, `vmax` | Normalize against an explicit numeric range. |

All modes accept `invert: true`. NaN and infinities are converted to finite
pixel values before quantization.
