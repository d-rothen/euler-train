# Run file format

A run directory is the source of truth. Its core records are JSON or JSONL;
large tensors and images live in ordinary model/data formats. Consumers do not
need to import euler-train to inspect a completed run.

## Layout

```text
<project>/
└── runs/
    └── <run_id>/
        ├── meta.json
        ├── config.json
        ├── code_ref.json
        ├── run_environment.json
        ├── train.jsonl                 # after the first training log
        ├── val.jsonl                   # after the first validation log
        ├── validation/                 # optional: validation sets
        │   └── <key>/
        │       ├── set.json            # the set document, written once
        │       └── passes.jsonl        # one record per validation pass
        ├── architecture.onnx            # optional
        ├── checkpoints/                 # optional; may be external instead
        │   └── epoch_<N>.pt
        └── outputs/                     # optional
            └── epoch_<N>_step_<M>/
                ├── manifest.json
                └── <output_type>/
                    ├── pred/
                    ├── gt/
                    ├── input/
                    └── aux/
                        └── <aux_name>/
```

A run ID has the form `YYYY-MM-DD_HH-MM-SS_<hex>`, using the local clock and a
random suffix. Code should use the `run_id` value from `meta.json`, not infer
identity by parsing the directory name.

## `meta.json`

This is the mutable run index. euler-train rewrites it as lifecycle state or
artifact metadata changes.

| Field | Meaning |
|---|---|
| `run_id` | Stable generated identifier. |
| `run_name` | Optional human-readable label. |
| `status` | `running`, `completed`, `crashed` or `interrupted`. |
| `start_time`, `end_time` | Unix timestamps; end is `null` while running. |
| `start_iso`, `end_iso` | Local-time ISO-like representations of the same timestamps. |
| `duration_sec` | Process duration, or `null` while running. |
| `pid`, `python`, `command` | Process ID, Python version and `sys.argv`. |
| `slurm` | Selected SLURM allocation fields, or `null` outside a job. |
| `datasets` | Optional split-to-dataset contract. |
| `evaluations` | Optional named evaluation records. |
| `modes` | Optional lifecycle records keyed by process mode. |
| `checkpoints` | Checkpoint records in registration order. |
| `checkpoint_dir` | Optional external checkpoint directory. |
| `architecture` | Relative ONNX architecture path when exported. |
| `metric_naming` | Optional structured metric naming declaration; with `schema_version` `2.x` (training naming 2.0) every logged key is held to it. |
| `validation_sets` | Optional validation sets by key: paths of `set.json` and `passes.jsonl`, schema version, metric namespace, label. |
| `naming_report` | Naming problems of a training naming 2.0 run and of its validation sets, each counted once per key and code. |
| `pipeline` | Optional pipeline attachment identity. |
| `updated_at` | Last observed write time for tracked artifacts. |
| `error`, `traceback` | Terminal failure details when applicable. |

User fields passed with `meta={...}` are retained alongside managed fields.
Consumers should therefore ignore unknown keys.

The complete machine-readable contract is
[`meta-schema.json`](../meta-schema.json).

### Dataset entries

Datasets are keyed by caller-defined split names:

```json
{
  "datasets": {
    "train": {
      "modalities": {
        "hazy_rgb": {
          "path": "/data/vkitti2/hazy",
          "used_as": "input",
          "slot": "dehaze.input.rgb",
          "modality_type": "rgb"
        }
      },
      "hierarchical_modalities": {
        "camera_intrinsics": {
          "path": "/data/vkitti2/intrinsics",
          "used_as": "condition",
          "slot": "dehaze.condition.camera_intrinsics",
          "modality_type": "camera_intrinsics",
          "hierarchy_scope": "scene_camera",
          "applies_to": ["hazy_rgb"]
        }
      }
    }
  }
}
```

`used_as`, `slot` and `modality_type` are required for each modality captured
through the dataset integration. Additional contract fields are preserved.

### Evaluation entries

Evaluations are keyed so repeated resumes update the same logical record:

```json
{
  "evaluations": {
    "vkitti2-test": {
      "name": "Virtual KITTI 2 test",
      "status": "completed",
      "checkpoint": {"epoch": 12, "step": 4800},
      "metadata": {"runner": "eval-v2"},
      "datasets": {
        "test": {
          "modalities": {
            "rgb_input": {
              "path": "/data/vkitti2/test/rgb",
              "used_as": "input",
              "slot": "dehaze.input.rgb",
              "modality_type": "rgb"
            }
          },
          "hierarchical_modalities": {}
        }
      }
    }
  }
}
```

Evaluation status is caller-defined. Its datasets use the same shape as the
top-level dataset map.

### Mode lifecycle

Supplying `mode="eval"` mirrors lifecycle details below `modes.eval`:

```json
{
  "modes": {
    "eval": {
      "status": "crashed",
      "start_time": 1785853962.0,
      "start_iso": "2026-08-04T14:32:42",
      "end_time": 1785854062.0,
      "end_iso": "2026-08-04T14:34:22",
      "duration_sec": 100.0,
      "pid": 12399,
      "command": ["evaluate.py", "--checkpoint", "epoch_12.pt"],
      "error": "RuntimeError: CUDA out of memory",
      "traceback": "Traceback (most recent call last):\n..."
    }
  }
}
```

### Artifact timestamps

`updated_at` maps an artifact to its most recently observed filesystem write:

```json
{
  "updated_at": {
    "train.jsonl": {
      "time": 1785854040.5,
      "iso": "2026-08-04T14:34:00"
    },
    "outputs/epoch_12_step_4800": {
      "time": 1785854050.0,
      "iso": "2026-08-04T14:34:10"
    }
  }
}
```

Artifacts inside the run use run-relative keys. External checkpoint paths use
absolute or caller-supplied paths.

## `config.json`

The normalized training configuration is stored as one JSON object. A resumed
run loads this file; an explicitly supplied config is merged over its existing
keys.

NumPy/PyTorch scalar-like values, arrays, `Path` objects and dataclass values
are converted to JSON-compatible representations when written.

## `code_ref.json`

Captured once for a fresh run:

```json
{
  "repo_url": "git@github.com:owner/project.git",
  "branch": "main",
  "commit_sha": "abc123def456...",
  "is_dirty": true,
  "dirty_diff": "diff --git a/train.py ...",
  "commit_message": "Tune the learning-rate schedule",
  "committed_at": "2026-08-04T14:20:00+00:00"
}
```

`dirty_diff` contains the tracked `git diff HEAD`; untracked file contents are
not embedded. Git fields are `null` when the process is outside a repository or
Git cannot be queried.

## `run_environment.json`

Also captured once for a fresh run:

```json
{
  "name": "gpu-node-01",
  "python_version": "3.11.9",
  "cuda_version": "12.4",
  "gpu_type": "NVIDIA A100-SXM4-80GB",
  "gpu_count": 4,
  "packages_snapshot": {
    "numpy": "2.2.6",
    "torch": "2.7.1"
  },
  "docker_image": null,
  "docker_digest": null,
  "metadata": null
}
```

CUDA and GPU information is detected from the available PyTorch, NVML, NVIDIA
command-line and environment interfaces. Package versions come from `uv pip
freeze` or `pip freeze`. Unavailable values remain `null`.

## Metric JSONL

Every line in `train.jsonl` or `val.jsonl` is an independent JSON object:

```json
{"step": 100, "epoch": 1, "wall_time": 1785854040.5, "elapsed_sec": 93.2, "loss": 0.42, "lr": 0.00003}
```

This makes metric writes append-only and allows standard streaming tools to
read a file while training continues. Validation records omit `elapsed_sec`.

Under training naming 2.0 every key of these records decomposes under a
declared `<scope>.train` namespace and has a definition in `meta.json`
(`metric_naming`); evaluation names (`depth.eval.…`) never appear here.

## Validation sets

A validation set is an evaluation of the live model on a fixed population
([training.md](https://github.com/d-rothen/euler-metric-naming/blob/main/docs/training.md)
§3 of euler-metric-naming). `validation/<key>/set.json` holds its document,
written once: the metricSet envelope and the contract-v2 `protocol` of that
evaluation (references, configured parameters, population, definitions).
`validation/<key>/passes.jsonl` holds one record per pass:

```json
{"step": 7554, "epoch": 5, "wall_time": 1794651078.2,
 "metrics": {"sparsedepth.eval.metric.standard.image_mean.absrel": 0.4307},
 "parameters": {"spatial.method": "pointcloud_projection", "depth.alignment.applied": false},
 "observations": {"domain.grid": [768, 1536], "depth.cap.clipped": 0.0},
 "per_item": {"sparsedepth.eval.metric.standard.absrel": [0.41, 0.45]},
 "series": {"sparsedepth.eval.metric.standard.count": [18120, 17604]}}
```

A consumer turns a set and a pass into the evaluation it stands for
(`euler_metric_naming.contract.load_validation_pass`) and compares it with any
evaluation, or with another pass, using the contract's verdicts.

## Output manifests

Each output snapshot describes its files in a colocated `manifest.json`. See
[Saving model outputs](outputs.md#manifest) for its shape and format rules.

## Sensitive information

Run directories are experiment records, not sanitized publication bundles.
Configuration values, command arguments, repository URLs, tracked Git diffs,
package inventories and cluster paths can contain private information. Review a
run before sharing it or configuring an external stream destination. Never put
API tokens in `config` or `meta`; pass streaming credentials only in the
`stream` configuration or environment.
