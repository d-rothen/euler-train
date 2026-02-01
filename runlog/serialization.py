"""JSON / JSONL helpers and config normalisation."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# JSON default serialiser
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for numpy / torch scalars, paths, dataclasses."""
    # numpy / torch scalar → Python scalar
    if hasattr(obj, "item"):
        return obj.item()
    # numpy array → nested list
    if hasattr(obj, "tolist"):
        return obj.tolist()
    # pathlib
    if isinstance(obj, Path):
        return str(obj)
    # dataclass instance
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
        f.write("\n")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=_json_default) + "\n")


# ---------------------------------------------------------------------------
# Config normalisation
# ---------------------------------------------------------------------------

def normalize_config(config: Any) -> dict:
    """Accept dict | path | Namespace | dataclass → dict."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    if isinstance(config, (str, Path)):
        return _load_config_file(Path(config))
    # dataclass → dict
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return dataclasses.asdict(config)
    # argparse.Namespace or any object with __dict__
    if hasattr(config, "__dict__"):
        return vars(config)
    raise TypeError(f"Unsupported config type: {type(config).__name__}")


def _load_config_file(path: Path) -> dict:
    text = path.read_text()
    if path.suffix == ".json":
        return json.loads(text)
    if path.suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            raise ImportError(
                "PyYAML is required to load YAML config files: pip install pyyaml"
            ) from e
        return yaml.safe_load(text)
    raise ValueError(f"Unsupported config file extension: {path.suffix}")
