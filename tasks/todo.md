- [x] Inspect current init/run crash logging code, tests, and schema/docs
- [x] Implement optional mode-aware crash metadata in the run lifecycle
- [x] Update tests/docs/schema and verify behavior

## Review

- Added optional `mode` support to `euler_train.init()` and mirrored lifecycle/error
  state into `meta.json["modes"][mode]` while keeping top-level crash fields as
  the latest-process summary.
- Cleared stale top-level error fields when resuming or completing a run so
  `meta.json` matches the documented status semantics.
- Verified with `pytest tests/test_runlog.py -q` (`88 passed`).
