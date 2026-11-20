# API and workflows

The package is built around one object, `Run`. Create it with
`euler_train.init()`, use it from the training loop, and close it with a
context manager or `finish()`.

## Initialize a run

```python
euler_train.init(
    dir=None,
    config=None,
    meta=None,
    output_formats=None,
    output_visualization=None,
    run_id=None,
    datasets=None,
    run_name=None,
    evaluations=None,
    mode=None,
    stream=None,
    metric_naming=None,
    pipeline=None,
    modality_roles=None,
    strict_naming=False,
) -> Run
```

| Parameter | Purpose |
|---|---|
| `dir` | Project/output directory. A fresh run is created below `<dir>/runs/<run_id>`. An existing run directory containing `meta.json` is resumed directly. |
| `config` | Hyperparameters as a mapping, JSON/YAML path, `argparse.Namespace`, dataclass or object with `__dict__`. YAML files require the `[yaml]` extra. |
| `meta` | Additional top-level fields for `meta.json`, such as `description` or `tags`. |
| `output_formats` | Per-output or per-slot `png`, `npy` or `npz` overrides. |
| `output_visualization` | Per-output or per-slot float-to-PNG rendering policies. |
| `run_id` | ID of an existing run below `<dir>/runs/` to resume. |
| `datasets` | Mapping of split name to a dataset object. Dataset metadata is captured in `meta.json`. |
| `run_name` | Human-readable label, independent of the generated run ID. |
| `evaluations` | Mapping of evaluation key to evaluation metadata, commonly supplied while resuming a run. |
| `mode` | Process label such as `train`, `val` or `eval`; adds a lifecycle record below `meta.json["modes"]`. |
| `stream` | Optional stream config, consumer or sequence of consumers. See [Streaming](streaming.md). |
| `metric_naming` | Structured naming declaration stored with the run and included in stream initialization. A training naming 2.0 declaration holds the run to it (see [Training naming 2.0](#training-naming-20)). |
| `pipeline` | Pipeline identity with a required non-empty `attach_id`. If omitted, `$EULER_SESSION_ID` is used when set. |
| `modality_roles` | Experiment role of each logged modality, e.g. `{"hazy": "input", "depth": "target"}`. Fills `used_as`/`slot` that the dataset does not declare. See [Dataset roles](#dataset-roles). |
| `strict_naming` | Raise `euler_train.NamingError` on the first naming problem instead of warning; for CI and smoke runs. Needs a training naming 2.0 declaration and euler-metric-naming 0.4 or newer. |

When `dir` is omitted, the project name comes from the Git root (or current
directory) and the base directory is resolved in this order:

1. `$ET_HOME/<project>`
2. `~/euler_train/<project>`

Useful attributes on the returned object are:

```python
run.run_id       # generated timestamp + random suffix
run.run_name     # optional human-readable label
run.project_dir  # directory containing runs/
run.dir          # this run's concrete directory
run.config       # normalized configuration mapping
```

## Log metrics

```python
run.log(metrics, *, step, epoch, mode="train")
```

`metrics` is a mapping of JSON-serializable values. `step`, `epoch` and a Unix
`wall_time` are added to each record. Training records also receive
`elapsed_sec`.

```python
run.log(
    {"loss": 0.42, "lr": 3e-5, "grad_norm": 1.2},
    step=100,
    epoch=1,
)
run.log(
    {"rgb.psnr": 28.3, "depth.mae": 0.03},
    step=100,
    epoch=1,
    mode="val",
)
```

Training records are appended to `train.jsonl`; all validation records are
appended to `val.jsonl`. With the `[gpu]` extra installed, GPU utilization and
memory values are sampled every 100 steps by default. If `metric_naming` is
configured, these automatic metrics use the `sys.train.*` namespace.

### Training naming 2.0

A declaration from `MetricNamespace(..., metrics=...)` (euler-metric-naming
0.4) carries `schema_version: "2.0"` and a metric definition for every base
metric ([training.md](https://github.com/d-rothen/euler-metric-naming/blob/main/docs/training.md)
there). Such a run is held to it, with no mixing of old and new names: every
key of `train.jsonl` and `val.jsonl` must decompose under a declared
`<scope>.train` namespace and have a definition, and every value must be a
finite number. With the `[naming]` extra installed, euler-train checks each
distinct key once:

| Code | The key |
|---|---|
| `no_namespace` | starts with no declared namespace (`loss`, `val/sky/mask_iou`) |
| `evaluation_name` | is an evaluation name (`depth.eval.…`): log it with `log_validation` |
| `no_decomposition` | does not decompose under its namespace's axes |
| `no_definition` | has a base metric without a definition |
| `not_a_number` | has a value that is not a finite number (booleans included) |

A problem is warned about once and counted in `meta.json["naming_report"]`
(source, key, code, count, first step); the value is still written. With
`strict_naming=True` the first problem raises `NamingError`. Schema-1
declarations and runs without one are not checked and behave as before.

## Validation sets

Validation scores that an evaluator also computes (AbsRel, δ < 1.25 on a
held-out set) are a **validation set**: an evaluation of the live model on a
fixed population, declared once and recorded per pass with the evaluator's
own metric names, so a consumer can judge a validation curve against the
final evaluation (`same protocol`, `compared on 64 shared files`, …).
euler-eval builds both documents (`euler_eval.validation.DepthValidationSet`):

```python
muses = DepthValidationSet("muses", sparse_depth_path=..., intrinsics_path=...,
                           camera_extrinsics_path=..., indices=range(64))
run.declare_validation_set(muses.key, muses.declaration())   # once per run

validation = muses.new_pass()
for index in muses.indices:
    sample = muses.dataset[index]
    validation.add(sample, model(sample), intrinsics=K)
run.log_validation(muses.key, validation.record(step=step, epoch=epoch))
```

| Method | Writes |
|---|---|
| `declare_validation_set(key, document)` | `validation/<key>/set.json` once, and `meta.json["validation_sets"][key]` (paths, schema version, metric namespace, label). The same document again (on resume) is a no-op; a different one raises `ValueError`. |
| `log_validation(key, record=None, *, step=None, epoch=None, **fields)` | one line of `validation/<key>/passes.jsonl`; `wall_time` is added when missing. The set must be declared (in this process or before a resume). |

Keys match `[a-z0-9][a-z0-9_-]{0,63}`. Nothing goes to `val.jsonl`, where
several sets would mix under shared names. Set documents and pass records are
checked with euler-metric-naming when it is installed; problems land in
`naming_report` under `validation/<key>`.

## Save model outputs

```python
run.save_outputs(*, epoch=None, step=None, metadata=None, **output_types)
run.save_outputs_from_batch(*, batch, epoch=None, step=None, ...)
```

These APIs store predictions, targets, inputs and auxiliary arrays below the
run directory. See [Saving model outputs](outputs.md) for the slot structure,
named sample IDs, format selection and visualization policies.

## Checkpoints

### Save with PyTorch

```python
path = run.save_checkpoint(
    model,
    epoch=5,
    step=2000,
    optimizer=optimizer,
    best_loss=0.12,
)
```

The model and optimizer are converted with `state_dict()` when available and
written with `torch.save`. Extra keyword arguments become fields in the saved
mapping. Without further configuration, the path is
`<run>/checkpoints/epoch_5.pt`.

### Use an external checkpoint directory

```python
checkpoint_dir = run.init_checkpoint_dir()
```

The default base is resolved as:

1. `$SCRATCH/euler_train/<project>/checkpoints`
2. `<project_dir>/checkpoints`

The final directory name is the slugified `run_name`, or `run_id` when the run
has no name. Fresh runs disambiguate collisions; resumed runs reuse the path
already recorded in `meta.json`. Pass `base=...` to choose an explicit base.

### Register a checkpoint written elsewhere

```python
trainer.save_checkpoint(path)
run.log_saved_checkpoint(path, epoch=5, step=2000, is_best=True)
```

This records the file without rewriting it. Marking a checkpoint `is_best=True`
clears that flag from earlier checkpoint entries.

### Record how a checkpoint was selected

```python
run.log_saved_checkpoint(
    path, epoch=5, step=7554, is_best=True,
    selection_metric="sparsedepth.eval.metric.standard.image_mean.absrel",
    selection_value=0.431,
    selection_direction="lower",       # from the metric's definition
    validation_set="muses",
)
run.save_checkpoint(model, epoch=5, step=7554, selection={
    "metric": "...", "value": 0.431, "direction": "lower",
    "validation_set": "muses", "is_best": True,
})
```

The fields land in the checkpoint's `meta.json` entry
(`selection_metric`, `selection_value`, `selection_direction`,
`validation_set`); a direction other than `lower` or `higher` raises.

## Model architecture

```python
path = run.log_architecture(model, dummy_input)
```

With the `[architecture]` extra installed, this exports
`<run>/architecture.onnx`, simplifies and optimizes the graph, and strips its
weights for lightweight inspection in [Netron](https://netron.app/). The
model's original train/eval state is restored after export.

The lower-level `euler_train.export_architecture(model, dummy_input,
output_path)` function writes the same representation to an arbitrary path.

## Resume a run

Pass the original project directory and ID:

```python
run = euler_train.init(
    dir="experiments/dehazing",
    run_id="2026-08-04_14-32-10_a3f2",
)
```

Or pass the concrete run directory itself:

```python
run = euler_train.init(
    dir="experiments/dehazing/runs/2026-08-04_14-32-10_a3f2",
)
```

Existing config is loaded and can be extended or overridden by a new `config`
mapping. Provenance files from the original process are preserved. The run is
marked `running` again until it is finished or detached.

## Datasets and evaluations

### Dataset metadata

Pass datasets by split to preserve the data contract beside the run:

```python
run = euler_train.init(
    dir="experiments/dehazing",
    config=config,
    datasets={"train": train_dataset, "val": val_dataset},
)
```

If a dataset exposes `describe_for_runlog()`, euler-train uses that structured
contract. Otherwise it reads modality paths and, when the `[datasets]` extra or
euler-loading provides ds-crawler, enriches them from indexed metadata before
using conservative naming heuristics. Regular and hierarchical modalities
record their path, role, semantic slot and modality type;
hierarchical entries can also record scope and applicability.

This integration is duck-typed. Importing euler-train does not require
euler-loading itself.

#### Dataset roles

Every logged modality needs `used_as` (`condition`, `input`, `output` or
`target`), `modality_type` and `slot`. Roles belong to the experiment, not to
the dataset: the same depth archive is a target for one model and a condition
for another, and datasets prepared by euler-known-datasets carry no roles.
Declare them once per run:

```python
run = euler_train.init(
    config=config,
    datasets={"train": train_dataset, "val": val_dataset},
    modality_roles={
        "hazy": "input",
        "clear": "target",
        "depth": "target",
        "intrinsics": "condition",
        "urbansyn.depth": {"used_as": "target", "task": "depth"},
    },
)
```

A key matches the full modality name, then the part after its last `.`, so
`"depth"` covers `urbansyn.depth` and `rds.depth` from a mixture. A value is a
`used_as` string or a mapping with `used_as`, `slot` and `task` (plus
`hierarchy_scope` and `applies_to` for conditions); the slot defaults to
`[task.]used_as.modality_type`.

Roles fill gaps only. A value the dataset's `describe_for_runlog()` already
reports (euler-loading `Modality(used_as=...)`, a head's
`addons.euler_train`/`addons.euler_loading` roles, or euler-loading's name
hints such as `hazy` → `input`) is kept; `modality_roles` wins over
euler-train's own name heuristics. When a role is still missing, `init` raises
one `MissingModalityFieldsError` (a `ValueError`) that lists every affected
modality of every split, before the run directory is created.

### Evaluation records

Evaluations are keyed records nested in the original run:

```python
run = euler_train.init(
    dir="experiments/dehazing",
    run_id=trained_run_id,
    mode="eval",
)

run.add_evaluation(
    "vkitti2-test",
    name="Virtual KITTI 2 test",
    status="running",
    checkpoint={"epoch": 12, "step": 4800},
    datasets={"test": test_dataset},
    metadata={"runner": "eval-v2"},
)

# ...evaluate...

run.finish_evaluation("vkitti2-test")
run.finish()
```

`add_evaluation()` merges fields into an existing entry and flushes immediately.
`finish_evaluation(key, status="completed")` updates its status and raises
`KeyError` for an unknown key. Existing evaluation keys survive later resumes.

## Modes and pipeline identity

`mode=` separates lifecycle state when train, validation and evaluation happen
in different processes. Top-level lifecycle fields still describe the most
recent process, while each named mode retains its own status, timestamps,
command and crash information.

Automated launchers can attach a stable pipeline identity:

```python
run = euler_train.init(
    pipeline={
        "attach_id": "pipeline-invocation-42",
        "stage": "training",
        "pipeline_id": "nightly-dehazing",
    },
)
```

Only `attach_id` is required; additional JSON-compatible fields are preserved.
When no explicit mapping is supplied, `$EULER_SESSION_ID` becomes the
`attach_id` automatically.

## Lifecycle

Prefer the context manager. It records terminal state even when the training
body raises:

```python
with euler_train.init(dir="experiments/dehazing", config=config) as run:
    train(run)
```

For manual lifecycle management:

```python
run.finish()                       # status="completed"
run.finish(status="interrupted")  # explicit terminal status
```

`finish()` is idempotent. It writes end timestamps and duration, flushes and
closes stream consumers, and removes the process hooks installed by the run.

Use `detach()` when a process must stop managing a resumed run without changing
its status—for example, after attaching evaluation metadata while another
process still owns the training lifecycle:

```python
run.detach()
```

Pending metadata is flushed and process hooks are removed, but `status`,
`end_time` and `duration_sec` are left unchanged.
