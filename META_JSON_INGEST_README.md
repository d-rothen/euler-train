# MODEL_RUN_META.md

This document defines how `meta.json` is read and ingested for model runs.

## Scope

`meta.json` currently affects these ingestion paths:

1. `model_runs` row update (`meta_json` persisted, `status` extracted)
2. `model_run_data_refs` + `model_run_data_ref_matches` sync
3. `model_evaluations` sync (from `meta_json.evaluations`)

It does **not** directly ingest into:

- `dataset_splits_eval`
- `per_file_metrics`
- `dataset_lineage`

Those are handled by separate flows/services.

## When ingestion runs

### A) During model run discovery/sync

Trigger paths:

- `GET /api/models?refresh=true`
- `POST /api/models/[id]/runs`
- Models page server load (`src/routes/models/+page.server.ts`), which calls full model sync

Flow:

1. Discover run directory.
2. Read `<runPath>/meta.json`.
3. Upsert `model_runs.meta_json`.
4. Extract `meta_json.status` if valid.
5. Call `syncRunDataRefs(modelRunId, metaJson)`.

### B) During manual reset of a single data ref

Trigger path:

- `POST /api/model-runs/[id]/data-refs/[refId]/reset` (with `resync: true`)

Flow:

1. Clear manual overrides for that ref (role/match as requested).
2. Re-run `syncRunDataRefs` using persisted `model_runs.meta_json` (if present).

## Mandatory vs optional by ingestion case

| Ingestion case | Mandatory in `meta.json` | Optional in `meta.json` | Outcome |
|---|---|---|---|
| Model run status sync | Nothing strictly required | `status` | `model_runs.meta_json` stored as-is; `status` only accepted if one of: `running`, `completed`, `crashed`, `interrupted` |
| Data-ref contract sync | `datasets` object (required for successful data-ref ingestion) | `evaluations` | Run-level refs + matches synced from `datasets` |
| Model evaluations sync | Only required if `evaluations` is present: valid eval keys and object shape | `name`, `status`, `checkpoint`, `metadata`, `datasets` | `model_evaluations` upserted by `(model_run_id, evaluation_key)` |
| Evaluation-scoped data refs | `evaluations[*].datasets` (or `evaluations.<key>.datasets`) when you want eval refs | all other eval fields | Eval-scoped refs + matches synced and linked by `model_evaluation_id` |

Important:

- If `meta_json.datasets` is missing or invalid, data-ref ingestion returns `success: false`.
- On validation failure, existing refs/matches/evaluations are preserved (no destructive cleanup).
- If `evaluations` is absent/null, all `model_evaluations` for that run are removed on successful sync.

## Top-level structure

```json
{
  "status": "running",
  "datasets": {},
  "evaluations": {}
}
```

Notes:

- `datasets` must exist and must be an object for data-ref ingestion to succeed.
- If you intentionally have no refs, use `"datasets": {}`.
- Other top-level keys are allowed and persisted in `model_runs.meta_json`, but ignored by data-ref ingestion.

## `datasets` contract (strict)

`datasets` is a map by split key:

```json
{
  "datasets": {
    "train": {
      "modalities": {
        "rgb": {
          "path": "/mnt/ds/train/rgb",
          "used_as": "input"
        }
      },
      "hierarchical_modalities": {
        "sequence_rgb": {
          "path": "/mnt/ds/train/hier/rgb",
          "used_as": "condition",
          "hierarchy_scope": "sequence"
        }
      }
    }
  }
}
```

### Split object rules

Mandatory:

- `modalities` (object)

Optional:

- `hierarchical_modalities` (object)

Rejected:

- Unknown keys at split level.

### Modality entry rules

Mandatory:

- `path`: non-empty string
- `used_as`: one of:
  - `input`
  - `target`
  - `condition`
  - `auxiliary`
  - `output`
  - `unknown`

Also accepted alias:

- `outputs` (normalized to `output`)

Optional:

- `slot`: non-empty string
- `modality_type`: non-empty string
- `hierarchy_scope`: non-empty string
- `applies_to`: any JSON value
- `metadata`: object

Rejected:

- Legacy string modality format (`"rgb": "/path"`)
- Unknown keys at modality level

### Path normalization

Ingested `path` is normalized before matching:

- trim whitespace
- `\\` to `/`
- collapse repeated slashes
- remove leading `./`
- remove trailing slash (except root)

## `evaluations` contract (optional)

`evaluations` may be either:

1. Object form
2. Array form

### Object form

```json
{
  "evaluations": {
    "eval_rgb": {
      "name": "RGB Eval",
      "status": "completed",
      "checkpoint": { "epoch": 2, "step": 400, "name": "epoch_2" },
      "metadata": { "runner": "eval_v2" },
      "datasets": {
        "test": {
          "modalities": {
            "rgb_input": {
              "path": "/mnt/ds/test/rgb",
              "used_as": "input"
            },
            "rgb_pred": {
              "path": "/mnt/ds/preds/rgb",
              "used_as": "outputs"
            }
          }
        }
      }
    }
  }
}
```

### Array form

```json
{
  "evaluations": [
    {
      "evaluation_key": "eval_rgb",
      "name": "RGB Eval",
      "datasets": {
        "test": {
          "modalities": {
            "rgb_input": { "path": "/mnt/ds/test/rgb", "used_as": "input" }
          }
        }
      }
    }
  ]
}
```

For array entries, evaluation key is resolved by first non-empty string in this order:

1. `evaluation_key`
2. `key`
3. `id`
4. `name`

Rules:

- Evaluation key must be unique within the run sync payload.
- `checkpoint.epoch` and `checkpoint.step` must be integers when present.
- `name`, `status`, `checkpoint.name` must be non-empty strings when present.
- `datasets` inside an evaluation follows the same strict contract as top-level `datasets`.

## Identity and upsert semantics

Data ref identity is scope-aware:

- Run-level ref identity: `(model_run_id, split_key, modality_kind, modality_name)`
- Eval-level ref identity: `(model_evaluation_id, split_key, modality_kind, modality_name)`

Behavior:

- Matching identity updates existing ref fields.
- Missing identity from new contract is pruned.
- Duplicate identities in one sync payload fail validation.

## Manual overrides vs sync

Manual controls:

- Role override: `PATCH /api/model-runs/[id]/data-refs/[refId]`
- Match override: `POST /api/model-runs/[id]/data-refs/[refId]/match`

Persistence behavior:

- `role_source = manual` is preserved across re-sync.
- `match_strategy = manual` is preserved across re-sync.

Reset behavior:

- `POST /api/model-runs/[id]/data-refs/[refId]/reset`
- Can clear role and/or match manually.
- Optional `resync: true` reapplies contract values from stored `model_runs.meta_json`.

## Recommended minimal templates

### Minimal valid meta for status + no refs

```json
{
  "status": "running",
  "datasets": {}
}
```

### Minimal valid meta with run refs

```json
{
  "status": "running",
  "datasets": {
    "train": {
      "modalities": {
        "rgb": {
          "path": "/mnt/ds/train/rgb",
          "used_as": "input"
        }
      }
    }
  }
}
```

### Valid meta with evaluation refs

```json
{
  "status": "completed",
  "datasets": {
    "train": {
      "modalities": {
        "rgb": { "path": "/mnt/ds/train/rgb", "used_as": "input" }
      }
    }
  },
  "evaluations": {
    "eval_rgb": {
      "status": "completed",
      "checkpoint": { "epoch": 12, "step": 4800 },
      "datasets": {
        "test": {
          "modalities": {
            "rgb_input": { "path": "/mnt/ds/test/rgb", "used_as": "input" },
            "rgb_pred": { "path": "/mnt/ds/preds/rgb", "used_as": "output" }
          }
        }
      }
    }
  }
}
```

## Current practical recommendation

Always emit at least:

1. `status`
2. `datasets` (use `{}` if empty)
3. `evaluations` only when available (with evaluation-level `datasets` for test input/output refs)

This keeps ingestion deterministic and avoids unnecessary validation failures during sync.
