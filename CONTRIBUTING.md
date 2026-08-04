# Contributing

Bug reports, documentation improvements and focused pull requests are welcome.
Please open an issue before starting a large API or file-format change so its
compatibility and migration path can be discussed first.

## Development setup

```bash
git clone https://github.com/d-rothen/euler-train.git
cd euler-train
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The development extra installs the dependencies used by the test suite. Add
`[architecture]`, `[gpu]` or `[naming]` when working on those integrations.

## Run the tests

```bash
pytest
```

The unit suite uses temporary directories and mocked HTTP/GPU integrations; it
does not require a dataset, accelerator, SLURM allocation or Euler View server.
Keep new tests similarly self-contained wherever possible.

Before submitting a packaging or documentation change, also build the wheel
and source distribution:

```bash
uv build
uvx check-jsonschema --check-metaschema meta-schema.json
uvx twine check dist/*
```

## Project conventions

- Python 3.9 is the minimum supported version. Keep public annotations valid on
  all versions covered by CI.
- The base package must remain importable without PyTorch, Pillow, NVML or the
  ONNX stack. Import optional dependencies only where their functionality is
  invoked and give the user an actionable install message.
- Local files are the source of truth. Optional stream failures must not break
  a training loop or prevent a local artifact from being written first.
- Treat the run directory and `meta.json` schema as public interfaces. Add a
  test and update `docs/run-format.md` and `meta-schema.json` when they change.
- Add regression tests for behavior changes. Match the existing type-annotated,
  small-helper style; no formatter-specific churn is needed.
- Keep the README concise. Put detailed behavior in the focused guide under
  `docs/` and link it from the documentation table.

## Releasing

PyPI publication uses trusted publishing from the
[publish workflow](https://github.com/d-rothen/euler-train/blob/main/.github/workflows/publish.yml).
To release:

1. Update `project.version` in `pyproject.toml`.
2. Run the tests and distribution checks.
3. Commit the version change.
4. Create and push a matching `v<version>` tag.

The publish workflow rejects a tag whose version does not match the project
metadata.
