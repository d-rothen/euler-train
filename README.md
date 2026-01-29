# runlog

Lightweight, file-based experiment logger for PyTorch. No servers, no accounts — just structured files on disk.

## Install

```bash
pip install -e .

# with image saving support (Pillow)
pip install -e ".[images]"
```

## Quick start

```python
import runlog

run = runlog.init(
    dir="runs/experiment_01",
    config={"lr": 1e-4, "arch": "unet", "epochs": 50},
)

for epoch in range(50):
    for step, batch in enumerate(train_loader):
        loss = train_step(model, batch)
        run.log({"loss": loss.item(), "lr": scheduler.get_lr()}, step=step, epoch=epoch)

    metrics = evaluate(model, val_loader)
    run.log(metrics, step=step, epoch=epoch, mode="val")

    run.save_outputs(
        epoch=epoch, step=step,
        rgb=dict(pred=pred_img, gt=gt_img, input=input_img),
        depth=dict(pred=depth_map, aux=dict(transmission=t_map)),
    )
    run.save_checkpoint(model, epoch=epoch, optimizer=optimizer)

run.finish()
```

Use the context manager to auto-finish and capture crashes:

```python
with runlog.init(dir="runs/exp02", config=cfg) as run:
    ...  # if an exception is raised, meta.json records status="crashed" + traceback
```

## Directory structure

Each `runlog.init(dir=...)` call creates a timestamped subdirectory under `{dir}/runs/`:

```
{dir}/
└── runs/
    └── 2025-01-28_15-30-42_a3f2/   ← auto-generated run ID
        ├── meta.json
        ├── config.json
        ├── train.jsonl
        ├── val.jsonl
        ├── checkpoints/
        │   └── epoch_{N}.pt
        └── outputs/
            └── epoch_{N}_step_{M}/
                └── {output_type}/
                    ├── pred/
                    ├── gt/
                    ├── input/
                    └── aux/
                        ├── transmission/
                        └── attention_maps/
```

The run ID and directory are available as `run.run_id` and `run.dir`.

## API reference

### `runlog.init(dir, config=None, meta=None, output_formats=None) → Run`

Creates the run directory and writes `meta.json` + `config.json`.

| Parameter | Type | Description |
|---|---|---|
| `dir` | `str \| Path` | Project directory. Each call creates a unique run under `{dir}/runs/{timestamp_id}/`. |
| `config` | `dict \| str \| Path \| Namespace \| dataclass` | Hyperparameters. Paths to `.json` / `.yaml` files are loaded automatically. |
| `meta` | `dict \| None` | Extra fields merged into `meta.json` (e.g. `{"tags": ["baseline"]}`). |
| `output_formats` | `dict[str, str] \| None` | Override format inference (see [Format inference](#format-inference)). |

---

### `run.log(metrics, *, step, epoch, mode="train")`

Appends one JSON line to `train.jsonl` (default) or `val.jsonl`.

Fields `step`, `epoch`, and `wall_time` are added automatically. Training records also get `elapsed_sec`.

```python
run.log({"loss": 0.42, "lr": 3e-5, "grad_norm": 1.2}, step=100, epoch=1)
run.log({"rgb.psnr": 28.3, "depth.mae": 0.03}, step=100, epoch=1, mode="val")
```

---

### `run.save_outputs(*, epoch=None, step=None, **output_types)`

Saves arrays/images to `outputs/epoch_{N}_step_{M}/{output_type}/{slot}/`.

Each output type is a dict with these slot keys:

| Slot | Value |
|---|---|
| `pred` | Model prediction |
| `gt` | Ground truth |
| `input` | Model input |
| `aux` | Dict of named auxiliary outputs (each becomes a subdirectory) |

Values can be:
- A single numpy array, torch tensor, or PIL Image
- A list of the above (saved as `0000.ext`, `0001.ext`, ...)
- A 4D numpy/torch array (split along dim 0 as a batch)

```python
run.save_outputs(
    epoch=1, step=500,
    rgb=dict(pred=pred_rgb, gt=gt_rgb),
    depth=dict(
        pred=depth_map,
        gt=gt_depth,
        aux=dict(transmission=t_map, attention=attn_map),
    ),
)
```

Torch tensors in `(C,H,W)` or `(B,C,H,W)` layout are automatically transposed to channels-last before saving.

Pass `None` for any slot or output type to skip it.

---

### `run.save_checkpoint(model, *, epoch, optimizer=None, **extra) → Path`

Saves to `checkpoints/epoch_{N}.pt`. Calls `.state_dict()` on model/optimizer automatically if available. Extra keyword arguments are included in the saved dict.

```python
run.save_checkpoint(model, epoch=5, optimizer=opt, best_loss=0.12)
```

---

### `run.finish(status="completed")`

Writes final `end_time`, `duration_sec`, and `status` to `meta.json`. Called automatically when using the `with` block. Safe to call multiple times.

---

## Format inference

Arrays are saved as `.png` or `.npy` based on shape and dtype:

| Array | Format |
|---|---|
| `uint8` with shape `(H,W)` | `.png` (grayscale) |
| Any dtype with shape `(H,W,1)`, `(H,W,3)`, `(H,W,4)` | `.png` |
| Float `.png` | clipped to `[0,1]`, scaled to `[0,255]` |
| Everything else (e.g. `float32 (H,W)`) | `.npy` |
| PIL Image | `.png` |

### Overriding format

Pass `output_formats` at init. Keys are resolved most-specific-first:

```python
run = runlog.init(
    dir="runs/exp",
    config=cfg,
    output_formats={
        "depth.pred": "npz",   # only depth pred
        "depth": "npy",        # all depth slots (unless more specific key matches)
        "transmission": "npz", # any slot/aux named "transmission"
    },
)
```

Supported formats: `"png"`, `"npy"`, `"npz"`.

## `meta.json` schema

Auto-managed, not written to directly.

```json
{
  "run_id": "2025-01-28_15-30-42_a3f2",
  "status": "running | completed | crashed",
  "start_time": 1706400000.0,
  "start_iso": "2024-01-28T00:00:00",
  "end_time": 1706403600.0,
  "end_iso": "2024-01-28T01:00:00",
  "duration_sec": 3600.0,
  "pid": 12345,
  "python": "3.11.5",
  "command": ["train.py", "--lr", "1e-4"],
  "error": "RuntimeError: CUDA OOM",
  "traceback": "..."
}
```

`error` and `traceback` are only present when `status` is `"crashed"` (via context manager).

## Dev

```bash
pip install git+https://github.com/d-rothen/euler-train.git
uv pip install "euler-train[images,gpu] @ git+https://github.com/d-rothen/euler-train"
```
