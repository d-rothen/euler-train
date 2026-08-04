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
| `metric_naming` | Structured naming declaration stored with the run and included in stream initialization. |
| `pipeline` | Pipeline identity with a required non-empty `attach_id`. If omitted, `$EULER_SESSION_ID` is used when set. |

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
