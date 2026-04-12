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

## Euler View stream mode

`euler_train` can dual-write runs to Euler View while still keeping the normal file-based run directory as the primary source of truth. Streaming is best-effort: local files are always written first, and stream events are sent alongside them when a stream consumer is configured.

Pass the stream config via `stream=...` on `euler_train.init()`:

```python
import os
import euler_train

run = euler_train.init(
    dir="runs/experiment_01",
    config=cfg,
    stream={
        "base_url": os.environ["EULER_VIEW_BASE_URL"],
        "model_id": 42,
        "api_token": os.environ["EULER_VIEW_API_TOKEN"],
        "stream_attach_token": os.environ.get("EULER_VIEW_STREAM_ATTACH_TOKEN"),
    },
)
```

What each field is for:

- `base_url`: Euler View server base URL, for example `https://view.example.com`.
- `model_id`: Euler View model that should receive this run.
- `api_token` or `access_token`: credential used to open a stream session. This is required whenever the package needs to negotiate its own short-lived ingest token, even if `stream_attach_token` is also provided.
- `stream_attach_token`: optional opaque launch-attachment token. When present, the producer sends it during the session handshake and Euler View attaches the stream to that launch directly. This is the preferred path for interactive or otherwise non-SLURM launches. Internally Euler View currently maps this token to `model_launches.id`, but the package treats it as an opaque token.
- `stream_token`: optional pre-issued ingest token. If you pass this, the package skips the session handshake entirely and posts events directly to the ingest endpoint.
- `datasource_id`: optional Euler View datasource hint for the run record.
- `euler_train_dir` and `run_dir`: optional path overrides. These are auto-derived in normal usage.
- `session_expires_in_sec`: optional requested session lifetime for the short-lived ingest token.

How the handshake works:

1. `euler_train` creates the local run directory and writes `meta.json`, `config.json`, and the other standard files as usual.
2. When the first stream flush happens, the package requests a session from Euler View at `/api/models/{model_id}/runs/{run_id}/stream/session`.
3. If `stream_attach_token` is configured, it is sent as `streamAttachToken` and used as the explicit launch attachment key.
4. If no `stream_attach_token` is configured, the package falls back to the SLURM metadata already captured in `meta.json["slurm"]["job_id"]`.
5. Euler View returns a short-lived ingest token plus the canonical `streamAttachToken`; the package caches that returned token and reuses it on later session refreshes.
6. Subsequent events are sent to `/api/model-run-stream/ingest`.

`stream_attach_token` is only the attachment/correlation key. It does not authenticate the session request by itself. In the current implementation you still need `api_token` or `access_token` for the session handshake, unless you already have a pre-issued `stream_token`.

### Dry-run connectivity check

If you need to verify from a cluster node whether Euler View is reachable and whether the stream handshake would be accepted before starting a real run, the package exposes the same dry-run check used by the stream protocol:

```bash
euler-train.stream.check \
  --api-url https://view.example.com \
  --model-id 42 \
  --api-key "$EULER_VIEW_API_TOKEN" \
  --stream-attach-token "$EULER_VIEW_STREAM_ATTACH_TOKEN"
```

This command calls:

- `POST /api/models/{model_id}/runs/{run_id}/stream/check`

It validates:

- network reachability to Euler View
- API token validity and scope
- model existence
- stream attachment resolution via `stream_attach_token`, when provided
- SLURM fallback matching via `--slurm-job-id`, when no attach token is provided

Useful flags:

- `--run-id`: override the dry-run run ID. If omitted, the CLI generates a temporary `stream-check-*` ID.
- `--stream-attach-token`: test the explicit launch attachment path.
- `--slurm-job-id`: test the fallback SLURM matching path.
- `--datasource-id`: include the datasource hint that a real stream session would send.
- `--euler-train-dir` and `--run-dir`: include path hints in the dry-run payload.
- `--json`: print the raw JSON response for automation.

The dry-run check does not start a run and does not mint an ingest token. It only verifies that the real session handshake would be accepted with the same request shape.

What has to be configured where:

- In the training script or job wrapper: pass `stream={...}` to `euler_train.init(...)`.
- In Euler View / the launcher: if the run should attach to a specific launch regardless of SLURM availability, provide a `stream_attach_token` to the runtime and forward it into the `stream` config.
- Under plain SLURM launches: no extra attachment config is required as long as Euler View can uniquely match the run by `slurm.job_id`.
- If neither `stream_attach_token` nor a unique SLURM match is available, run streaming still works, but launch-to-run linking in Euler View may remain unresolved.

For new integrations, prefer `stream_attach_token`. The older `launch_id` / `launchId` config keys are still accepted as compatibility aliases, but new code should treat the attachment value as an opaque stream token rather than a launch ID.

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
                ├── manifest.json
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

### `euler_train.init(dir, config=None, meta=None, output_formats=None, output_visualization=None, run_id=None, datasets=None, run_name=None, evaluations=None, mode=None, stream=None, metric_naming=None) → Run`

Creates the run directory and writes `meta.json`, `config.json`, `code_ref.json`, and `run_environment.json`. On resume (`run_id` provided), only `meta.json` and `config.json` are updated. `meta.json` also maintains an `updated_at` map with last-write timestamps for tracked artifacts.

| Parameter | Type | Description |
|---|---|---|
| `dir` | `str \| Path` | Project directory. Each call creates a unique run under `{dir}/runs/{timestamp_id}/`. |
| `config` | `dict \| str \| Path \| Namespace \| dataclass` | Hyperparameters. Paths to `.json` / `.yaml` files are loaded automatically. |
| `meta` | `dict \| None` | Extra fields merged into `meta.json` (e.g. `{"tags": ["baseline"]}`). |
| `output_formats` | `dict[str, str] \| None` | Override format inference (see [Format inference](#format-inference)). |
| `output_visualization` | `dict[str, Any] \| None` | Override PNG visualization policy per output type / slot (see [Output visualization](#output-visualization)). |
| `run_id` | `str \| None` | Resume an existing run at `{dir}/runs/{run_id}` instead of creating a new one. |
| `datasets` | `dict[str, Any] \| None` | Optional split → dataset map. If a dataset exposes `describe_for_runlog()`, that contract is used directly; otherwise euler_train infers structured modality metadata (`path`, `used_as`, `slot`, `modality_type`, and hierarchical fields), resolving fixed namespaced properties from `properties.euler_loading` and `properties.euler_train` before heuristics. |
| `run_name` | `str \| None` | Optional human-readable run label stored in `meta.json`. |
| `evaluations` | `dict[str, dict] \| None` | Optional evaluation key → entry map. See [Evaluations](#evaluations). |
| `mode` | `str \| None` | Optional process label such as `"train"`, `"val"`, or `"eval"`. When set, lifecycle and crash details are also written under `meta.json["modes"][mode]`. |
| `stream` | `dict \| EulerViewStreamConfig \| consumer \| sequence \| None` | Optional output stream config / consumer. A mapping with `base_url`, `model_id`, and `api_token` (or `access_token`) enables best-effort streaming to Euler View. Prefer `stream_attach_token` for explicit launch attachment; otherwise the package falls back to `meta.json["slurm"]["job_id"]` when available. |
| `metric_naming` | `dict[str, Any] \| None` | Optional structured metric naming declaration stored in `meta.json["metric_naming"]` and included in the stream init event. |

---

### `run.log(metrics, *, step, epoch, mode="train")`

Appends one JSON line to `train.jsonl` (default) or `val.jsonl`.

Fields `step`, `epoch`, and `wall_time` are added automatically. Training records also get `elapsed_sec`. When `nvidia-ml-py` is installed, GPU stats (`gpu_util_pct`, `gpu_mem_util_pct`, `gpu_mem_used_gb`, `gpu_mem_total_gb`) are appended every 100 steps.

```python
run.log({"loss": 0.42, "lr": 3e-5, "grad_norm": 1.2}, step=100, epoch=1)
run.log({"rgb.psnr": 28.3, "depth.mae": 0.03}, step=100, epoch=1, mode="val")
```

---

### `run.save_outputs(*, epoch=None, step=None, metadata=None, **output_types)`

Saves arrays/images to `outputs/epoch_{N}_step_{M}/{output_type}/{slot}/`.

Each output type is a dict with these slot keys:

| Slot | Value |
|---|---|
| `pred` | Model prediction |
| `gt` | Ground truth |
| `input` | Model input |
| `aux` | Dict of named auxiliary outputs (each becomes a subdirectory) |

Slot values can be:
- A single numpy array, torch tensor, or PIL Image
- A list of the above (saved as `0000.ext`, `0001.ext`, ...)
- A 4D numpy/torch array (split along dim 0 as a batch)
- A dict mapping custom string IDs to items (saved as `{id}.ext`)

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

#### Named sample IDs

When slot values are dicts with string keys, files are saved with the key as the basename instead of sequential indices. This is useful for matching outputs back to specific input samples:

```python
run.save_outputs(
    epoch=1, step=500,
    rgb=dict(pred={"scene_042": img_a, "scene_117": img_b}),
)
# produces: rgb/pred/scene_042.png, rgb/pred/scene_117.png
```

#### Source metadata

The optional `metadata` parameter attaches dataset provenance and other context to the output manifest. When provided, the `dataset` key is required and must be a non-empty string:

```python
run.save_outputs(
    epoch=1, step=500,
    metadata={"dataset": "vkitti2", "split": "val", "scene": "0001"},
    rgb=dict(pred={"frame_042": img_a}),
)
```

#### Output manifest

Each `save_outputs` call writes a `manifest.json` alongside the saved files, recording what was written and how. The manifest is also streamed via the `output_snapshot` event when streaming is active.

```json
{
  "version": 1,
  "epoch": 1,
  "step": 500,
  "metadata": {"dataset": "vkitti2", "split": "val"},
  "output_types": {
    "rgb": {
      "pred": {
        "id_mode": "named",
        "files": [
          {"sample_id": "scene_042", "filename": "scene_042.png", "format": "png"}
        ]
      },
      "gt": {
        "id_mode": "indexed",
        "files": [
          {"sample_id": 0, "filename": "0000.png", "format": "png"}
        ]
      }
    }
  },
  "format_overrides": {},
  "visualization_overrides": {}
}
```

Torch tensors in `(C,H,W)` or `(B,C,H,W)` layout are automatically transposed to channels-last before saving.

Pass `None` for any slot or output type to skip it.

---

### `run.init_checkpoint_dir(base=None) → Path`

Sets up an external checkpoint directory and records it in `meta.json["checkpoint_dir"]`.

When `base` is omitted, euler_train resolves it as:

1. `$SCRATCH/euler_train/<project>/checkpoints`
2. `<project_dir>/checkpoints`

The final leaf directory is the slugified `run_name`, or `run_id` when no name is set. Fresh runs auto-disambiguate collisions by appending a suffix derived from `run_id`. Resumed runs reuse the recorded `checkpoint_dir` automatically.

```python
ckpt_dir = run.init_checkpoint_dir()
run.save_checkpoint(model, epoch=5, step=2000, optimizer=opt)
```

---

### `run.save_checkpoint(model, *, epoch, optimizer=None, **extra) → Path`

Saves to `checkpoints/epoch_{N}.pt` inside the run directory by default. If `run.init_checkpoint_dir()` has been used, checkpoints are written there instead, and resumed runs keep using the recorded external directory automatically. Calls `.state_dict()` on model/optimizer automatically if available. Extra keyword arguments are included in the saved dict.

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
| Float `.png` | defaults to `unit_range`: clip to `[0,1]`, then scale to `[0,255]` |
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

## Output visualization

Float PNG outputs use a rendering policy before quantization to `uint8`. Policies are configured once at `init()` via `output_visualization`, and keys resolve most-specific-first just like `output_formats`:

```python
run = euler_train.init(
    dir="runs/exp",
    config=cfg,
    output_visualization={
        "depth": {"mode": "percentile", "pmin": 1, "pmax": 99},
        "depth.pred": {"mode": "fixed_range", "vmin": 0.0, "vmax": 80.0},
        "confidence": "minmax",
    },
)
```

Supported visualization modes:

| Mode | Parameters | Description |
|---|---|---|
| `unit_range` | none | Clip float data to `[0,1]` before saving. |
| `minmax` | none | Normalize using the finite-value min / max of the saved item. |
| `percentile` | `pmin`, `pmax` | Normalize using finite-value percentiles, useful for depth-like outputs. |
| `fixed_range` | `vmin`, `vmax` | Normalize against an explicit numeric range. |

All float PNG policies also accept `invert: true` to flip black ↔ white after normalization.

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
  "modes": {
    "train": {
      "status": "completed",
      "start_time": 1706400000.0,
      "start_iso": "2025-01-28T15:30:42",
      "end_time": 1706403000.0,
      "end_iso": "2025-01-28T16:20:42",
      "duration_sec": 3000.0,
      "pid": 12345,
      "command": ["train.py", "--lr", "1e-4"]
    },
    "eval": {
      "status": "crashed",
      "start_time": 1706403200.0,
      "start_iso": "2025-01-28T16:23:20",
      "end_time": 1706403300.0,
      "end_iso": "2025-01-28T16:25:00",
      "duration_sec": 100.0,
      "pid": 12399,
      "command": ["eval.py", "--ckpt", "epoch_12.pt"],
      "error": "RuntimeError: CUDA OOM",
      "traceback": "Traceback (most recent call last):\n  ..."
    }
  },
  "updated_at": {
    "meta.json": {
      "time": 1706403300.0,
      "iso": "2025-01-28T16:25:00"
    },
    "config.json": {
      "time": 1706400000.0,
      "iso": "2025-01-28T15:30:42"
    },
    "train.jsonl": {
      "time": 1706402999.0,
      "iso": "2025-01-28T16:19:59"
    },
    "outputs/epoch_12_step_4800": {
      "time": 1706403010.0,
      "iso": "2025-01-28T16:20:10"
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
- `modes` is only present when `mode=...` is passed to `euler_train.init`; each key stores the latest lifecycle snapshot for that mode.
- `updated_at` stores last-write timestamps for tracked artifacts. Keys are run-relative paths when the artifact lives inside the run directory (for example `train.jsonl` or `outputs/epoch_2_step_500`) and absolute paths for external artifacts such as custom checkpoint locations.
- `error` is only present when `status` is `"crashed"` (context manager / excepthook) or `"interrupted"` (SIGTERM/SIGINT). `traceback` is only present when `status` is `"crashed"`. When `mode=...` is set, the same fields are mirrored under `modes[mode]`.

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
