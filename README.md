# euler_train

Lightweight, file-based experiment logger for PyTorch. No servers, no accounts — just structured files on disk.

## Install

```bash
pip install -e .

# with image saving support (Pillow)
pip install -e ".[images]"

# with GPU monitoring (nvidia-ml-py)
pip install -e ".[gpu]"
```

## Quick start

```python
import euler_train

run = euler_train.init(
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
with euler_train.init(dir="runs/exp02", config=cfg) as run:
    ...  # if an exception is raised, meta.json records status="crashed" + traceback
```

## Directory structure

Each `euler_train.init(dir=...)` call creates a timestamped subdirectory under `{dir}/runs/`:

```
{dir}/
└── runs/
    └── 2025-01-28_15-30-42_a3f2/   ← auto-generated run ID
        ├── meta.json
        ├── config.json
        ├── code_ref.json
        ├── run_environment.json
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

### `euler_train.init(dir, config=None, meta=None, output_formats=None, run_id=None, datasets=None, run_name=None, evaluations=None) → Run`

Creates the run directory and writes `meta.json`, `config.json`, `code_ref.json`, and `run_environment.json`. On resume (`run_id` provided), only `meta.json` and `config.json` are updated.

| Parameter | Type | Description |
|---|---|---|
| `dir` | `str \| Path` | Project directory. Each call creates a unique run under `{dir}/runs/{timestamp_id}/`. |
| `config` | `dict \| str \| Path \| Namespace \| dataclass` | Hyperparameters. Paths to `.json` / `.yaml` files are loaded automatically. |
| `meta` | `dict \| None` | Extra fields merged into `meta.json` (e.g. `{"tags": ["baseline"]}`). |
| `output_formats` | `dict[str, str] \| None` | Override format inference (see [Format inference](#format-inference)). |
| `run_id` | `str \| None` | Resume an existing run at `{dir}/runs/{run_id}` instead of creating a new one. |
| `datasets` | `dict[str, Any] \| None` | Optional split → dataset map. If a dataset exposes `describe_for_runlog()`, that contract is used directly; otherwise euler_train infers structured modality metadata (`path`, `used_as`, `slot`, `modality_type`, and hierarchical fields), resolving fixed namespaced properties from `properties.euler_loading` and `properties.euler_train` before heuristics. |
| `run_name` | `str \| None` | Optional human-readable run label stored in `meta.json`. |
| `evaluations` | `dict[str, dict] \| None` | Optional evaluation key → entry map. See [Evaluations](#evaluations). |

---

### `run.log(metrics, *, step, epoch, mode="train")`

Appends one JSON line to `train.jsonl` (default) or `val.jsonl`.

Fields `step`, `epoch`, and `wall_time` are added automatically. Training records also get `elapsed_sec`. When `nvidia-ml-py` is installed, GPU stats (`gpu_util_pct`, `gpu_mem_util_pct`, `gpu_mem_used_gb`, `gpu_mem_total_gb`) are appended every 100 steps.

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

### `run.add_evaluation(key, *, datasets=None, name=None, status=None, checkpoint=None, metadata=None)`

Adds or updates a single evaluation entry in `meta.json` under `evaluations[key]`. The `datasets` parameter accepts the same dataset objects as the top-level `datasets` parameter on `init()` and is processed through the same modality-inference pipeline. Flushes to disk immediately.

If the key already exists, existing fields are preserved and only the provided fields are updated (merge semantics).

```python
run.add_evaluation(
    "eval_rgb",
    datasets={"test": test_ds},
    name="RGB Eval",
    status="running",
    checkpoint={"epoch": 12, "step": 4800},
)
```

---

### `run.finish_evaluation(key, status="completed")`

Updates the `status` of an existing evaluation entry and flushes to disk. Raises `KeyError` if the key does not exist.

```python
run.finish_evaluation("eval_rgb")                    # status → "completed"
run.finish_evaluation("eval_depth", status="crashed") # custom status
```

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
run = euler_train.init(
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
  "run_name": "baseline_dehaze",
  "status": "running | completed | crashed | interrupted",
  "start_time": 1706400000.0,
  "start_iso": "2025-01-28T15:30:42",
  "end_time": 1706403600.0,
  "end_iso": "2025-01-28T16:30:42",
  "duration_sec": 3600.0,
  "pid": 12345,
  "python": "3.11.5",
  "command": ["train.py", "--lr", "1e-4"],
  "slurm": {
    "job_id": "123456",
    "job_name": "my_train_job",
    "node": "gpu-node-01",
    "partition": "gpu",
    "gpus": "1",
    "cpus": "8",
    "array_task_id": "0",
    "num_nodes": "1",
    "ntasks": "1",
    "ntasks_per_node": "1",
    "gpus_per_node": "1",
    "mem_per_node": "32000",
    "mem_per_cpu": "4000",
    "stdout_path": "/path/to/slurm-123456.out",
    "stderr_path": "/path/to/slurm-123456.err",
    "submit_dir": "/home/user/project"
  },
  "datasets": {
    "train": {
      "modalities": {
        "hazy_rgb": {
          "path": "/cluster/work/.../vkitti_rgb_hazy",
          "used_as": "input",
          "slot": "dehaze.input.rgb",
          "modality_type": "rgb"
        }
      },
      "hierarchical_modalities": {
        "camera_intrinsics": {
          "path": "/cluster/work/.../vkitti_intrinsics",
          "used_as": "condition",
          "slot": "dehaze.condition.camera_intrinsics",
          "hierarchy_scope": "scene_camera",
          "applies_to": ["hazy_rgb"]
        }
      }
    }
  },
  "evaluations": {
    "eval_rgb": {
      "name": "RGB Eval",
      "status": "completed",
      "checkpoint": { "epoch": 12, "step": 4800 },
      "metadata": { "runner": "eval_v2" },
      "datasets": {
        "test": {
          "modalities": {
            "rgb_input": { "path": "/mnt/ds/test/rgb", "used_as": "input" },
            "rgb_pred": { "path": "/mnt/ds/preds/rgb", "used_as": "output" }
          },
          "hierarchical_modalities": {}
        }
      }
    }
  },
  "error": "RuntimeError: CUDA OOM",
  "traceback": "Traceback (most recent call last):\n  ..."
}
```

- `end_time`, `end_iso`, `duration_sec` are `null` while `status` is `"running"`.
- `slurm` is `null` when not running under SLURM.
- `datasets` is only present when `datasets=...` is passed to `euler_train.init`.
- `evaluations` is only present when evaluations are provided via `evaluations=...` on `init()` or added via `run.add_evaluation()`.
- `error` and `traceback` are only present when `status` is `"crashed"` (context manager / excepthook) or `"interrupted"` (SIGTERM/SIGINT).

A formal JSON Schema for `meta.json` is available at [`meta-schema.json`](meta-schema.json).

## `code_ref.json` schema

Written once when a fresh run is created (not on resume). Captures git repository state at the time of the run.

```json
{
  "repo_url": "git@github.com:user/repo.git",
  "branch": "main",
  "commit_sha": "abc123def456...",
  "is_dirty": true,
  "dirty_diff": "diff --git a/train.py ...",
  "commit_message": "Add learning rate scheduler\n",
  "committed_at": "2025-01-28T15:20:00+01:00"
}
```

- `is_dirty` is `true` when there are uncommitted changes.
- `dirty_diff` contains the output of `git diff HEAD` when dirty, `null` otherwise.
- All fields are `null` if the project is not inside a git repository.

## `run_environment.json` schema

Written once when a fresh run is created (not on resume). Snapshots the runtime environment.

```json
{
  "name": "gpu-node-01",
  "python_version": "3.11.5",
  "cuda_version": "12.1",
  "gpu_type": "NVIDIA A100-SXM4-80GB",
  "gpu_count": 4,
  "packages_snapshot": {
    "torch": "2.1.0",
    "numpy": "1.26.2",
    "Pillow": "10.1.0"
  },
  "docker_image": null,
  "docker_digest": null,
  "metadata": null
}
```

- `name` is the hostname of the machine.
- `cuda_version` is detected from PyTorch, `nvcc`, or the `CUDA_VERSION` env var (first available).
- `gpu_type` and `gpu_count` are detected via `pynvml` or `nvidia-smi` (first available).
- `packages_snapshot` is the output of `pip freeze` (or `uv pip freeze`), parsed into a `{name: version}` dict.
- Fields are `null` when the corresponding tool/library is unavailable.

## Evaluations

Evaluations record model evaluation runs against test/validation splits, linking each evaluation to a checkpoint and its input/output datasets. They are written into the `evaluations` key of `meta.json` in the object form expected by downstream ingestion services (see `META_JSON_INGEST_README.md`).

### Typical usage: resume a trained run for evaluation

```python
import euler_train

# Resume the training run by its run_id
run = euler_train.init(
    dir="runs/experiment_01",
    run_id="2025-01-28_15-30-42_a3f2",
    evaluations={
        "eval_rgb": {
            "datasets": {"test": test_rgb_ds},
            "name": "RGB Eval",
            "status": "running",
            "checkpoint": {"epoch": 12, "step": 4800},
            "metadata": {"runner": "eval_v2"},
        },
    },
)

# ... run evaluation logic ...

run.finish_evaluation("eval_rgb")  # status → "completed"
run.finish()
```

### Evaluation entry fields

Each evaluation entry (the value under `evaluations[key]`) supports:

| Field | Type | Description |
|---|---|---|
| `datasets` | `dict[str, dataset]` | Split → dataset map (same objects as top-level `datasets`). Processed through the same modality-inference pipeline. |
| `name` | `str` | Human-readable evaluation label. |
| `status` | `str` | Evaluation status (`"running"`, `"completed"`, `"crashed"`, etc.). |
| `checkpoint` | `dict` | Checkpoint reference. Typically `{"epoch": int, "step": int}`, optionally with `"name"`. |
| `metadata` | `dict` | Arbitrary metadata (e.g. `{"runner": "eval_v2", "gpu": "A100"}`). |

All fields are optional. `datasets` is processed through `_build_datasets_meta` (contract → ds-crawler → heuristics); all other fields are stored as-is.

### Adding evaluations incrementally

Use `add_evaluation()` to register evaluations one at a time after init. This is useful when running multiple evaluations sequentially:

```python
run = euler_train.init(dir="runs/exp", run_id="2025-01-28_15-30-42_a3f2")

for split_name, ds in [("eval_rgb", test_rgb_ds), ("eval_depth", test_depth_ds)]:
    run.add_evaluation(
        split_name,
        datasets={"test": ds},
        status="running",
        checkpoint={"epoch": 12, "step": 4800},
    )
    evaluate(model, ds)
    run.finish_evaluation(split_name)

run.finish()
```

Each `add_evaluation()` call flushes `meta.json` immediately. Calling it with an existing key merges fields — existing fields not provided in the update are preserved.

### Merge semantics on resume

When resuming a run that already has evaluations in its `meta.json`, new evaluations are merged by key:

- Existing evaluation keys not present in the new `evaluations` dict are **preserved**.
- Existing keys present in the new dict are **updated** (field-level merge within each entry).
- New keys are **added**.

This means you can run evaluations across multiple sessions without losing previously recorded results.

## Dev

```bash
pip install git+https://github.com/d-rothen/euler-train.git
uv pip install "euler-train[images,gpu] @ git+https://github.com/d-rothen/euler-train"
```
