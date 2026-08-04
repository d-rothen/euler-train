"""Tests for the euler_train package — one section per public endpoint."""
from __future__ import annotations

import dataclasses
import json
import signal
import sys
from argparse import Namespace
from pathlib import Path
from urllib.error import URLError

import numpy as np
import pytest
import torch
from PIL import Image

import euler_train
import euler_train.stream as stream_mod
from euler_train.stream_check_cli import main as stream_check_cli_main

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _assert_updated_at_entry(entry: dict) -> None:
    assert isinstance(entry["time"], (int, float))
    assert isinstance(entry["iso"], str)
    assert entry["iso"]


class _DummyModality:
    def __init__(self, path: str) -> None:
        self.path = path


class _DummyDataset:
    def __init__(
        self,
        modalities: dict[str, str],
        hierarchical_modalities: dict[str, str] | None = None,
        hierarchical_lookups: dict[str, dict[tuple[str, ...], list[dict]]] | None = None,
    ) -> None:
        self._modalities = {
            name: _DummyModality(path) for name, path in modalities.items()
        }
        self._hierarchical_modalities = {
            name: _DummyModality(path)
            for name, path in (hierarchical_modalities or {}).items()
        }
        self._hierarchical_lookups = hierarchical_lookups or {}

    def modality_paths(self) -> dict[str, str]:
        return {name: mod.path for name, mod in self._modalities.items()}

    def hierarchical_modality_paths(self) -> dict[str, str]:
        return {
            name: mod.path
            for name, mod in self._hierarchical_modalities.items()
        }


class _DatasetWithRunlogDescription:
    def __init__(self, description: dict) -> None:
        self._description = description

    def describe_for_runlog(self) -> dict:
        return self._description


class _RecordingStreamConsumer:
    def __init__(self) -> None:
        self.context = None
        self.events: list[dict] = []
        self.flush_count = 0
        self.close_count = 0

    def bind(self, context) -> None:
        self.context = context

    def emit(self, event: dict) -> None:
        self.events.append(json.loads(json.dumps(event)))

    def flush(self) -> None:
        self.flush_count += 1

    def close(self) -> None:
        self.close_count += 1


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  init  /  config normalisation
# ═══════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_creates_run_subdirectory(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "proj"), config={"lr": 1e-3})
        run.finish()

        # run.dir should be {dir}/runs/{run_id}
        assert run.dir.parent == tmp_path / "proj" / "runs"
        assert run.dir.name == run.run_id
        assert (run.dir / "meta.json").exists()
        assert (run.dir / "config.json").exists()

    def test_run_id_in_meta(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        meta = _read_json(run.dir / "meta.json")
        assert meta["run_id"] == run.run_id
        run.finish()

    def test_config_from_dict(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={"lr": 0.01, "bs": 32})
        cfg = _read_json(run.dir / "config.json")
        assert cfg == {"lr": 0.01, "bs": 32}
        run.finish()

    def test_config_none_writes_empty(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"))
        cfg = _read_json(run.dir / "config.json")
        assert cfg == {}
        run.finish()

    def test_config_from_json_file(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"arch": "resnet", "depth": 50}))

        run = euler_train.init(dir=str(tmp_path / "r"), config=str(cfg_file))
        cfg = _read_json(run.dir / "config.json")
        assert cfg["arch"] == "resnet"
        assert cfg["depth"] == 50
        run.finish()

    def test_config_from_namespace(self, tmp_path):
        ns = Namespace(lr=1e-4, epochs=10)
        run = euler_train.init(dir=str(tmp_path / "r"), config=ns)
        cfg = _read_json(run.dir / "config.json")
        assert cfg == {"lr": 1e-4, "epochs": 10}
        run.finish()

    def test_config_from_dataclass(self, tmp_path):
        @dataclasses.dataclass
        class Cfg:
            lr: float = 1e-3
            arch: str = "unet"

        run = euler_train.init(dir=str(tmp_path / "r"), config=Cfg())
        cfg = _read_json(run.dir / "config.json")
        assert cfg == {"lr": 1e-3, "arch": "unet"}
        run.finish()

    def test_config_unsupported_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Unsupported config type"):
            euler_train.init(dir=str(tmp_path / "r"), config=42)

    def test_meta_initial_status_is_running(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "running"
        assert meta["start_time"] is not None
        assert meta["start_iso"] is not None
        assert meta["end_time"] is None
        run.finish()

    def test_meta_custom_fields_merged(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            meta={"description": "ablation A", "tags": ["v2", "fast"]},
        )
        meta = _read_json(run.dir / "meta.json")
        assert meta["description"] == "ablation A"
        assert meta["tags"] == ["v2", "fast"]
        assert meta["status"] == "running"  # built-in still present
        run.finish()

    def test_returns_run_object(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        assert isinstance(run, euler_train.Run)
        run.finish()

    def test_multiple_inits_create_separate_runs(self, tmp_path):
        run1 = euler_train.init(dir=str(tmp_path / "r"), config={})
        run2 = euler_train.init(dir=str(tmp_path / "r"), config={})
        assert run1.dir != run2.dir
        assert run1.run_id != run2.run_id
        run1.finish()
        run2.finish()

    def test_resume_from_run_directory(self, tmp_path):
        """Passing a full run directory (with meta.json) auto-resumes."""
        run = euler_train.init(dir=str(tmp_path / "proj"), config={"lr": 0.1})
        run_dir = run.dir
        run_id = run.run_id
        run.finish()

        # Resume by passing the run directory directly
        resumed = euler_train.init(dir=str(run_dir))
        assert resumed.run_id == run_id
        assert resumed.dir == run_dir
        assert resumed.project_dir == tmp_path / "proj"

        meta = _read_json(resumed.dir / "meta.json")
        assert meta["status"] == "running"
        resumed.finish()

    def test_resume_from_run_directory_preserves_config(self, tmp_path):
        """Resuming via run directory loads the existing config."""
        run = euler_train.init(dir=str(tmp_path / "proj"), config={"lr": 0.01})
        run_dir = run.dir
        run.finish()

        resumed = euler_train.init(dir=str(run_dir))
        assert resumed.config == {"lr": 0.01}
        resumed.finish()

    def test_resume_from_run_directory_with_config_override(self, tmp_path):
        """Resuming via run directory allows overriding config."""
        run = euler_train.init(dir=str(tmp_path / "proj"), config={"lr": 0.01})
        run_dir = run.dir
        run.finish()

        resumed = euler_train.init(dir=str(run_dir), config={"lr": 0.001})
        assert resumed.config["lr"] == 0.001
        resumed.finish()

    def test_explicit_run_id_takes_precedence_over_detection(self, tmp_path):
        """When run_id is given, dir is not treated as a run directory."""
        run1 = euler_train.init(dir=str(tmp_path / "proj"), config={})
        run1.finish()
        run2 = euler_train.init(dir=str(tmp_path / "proj"), config={})
        run2.finish()

        # Resume run2 explicitly even though we pass run1's dir
        # (this would be unusual but tests that explicit run_id wins)
        resumed = euler_train.init(
            dir=str(tmp_path / "proj"),
            run_id=run2.run_id,
        )
        assert resumed.run_id == run2.run_id
        resumed.finish()

    def test_dir_without_meta_json_creates_new_run(self, tmp_path):
        """A directory without meta.json is treated as a project dir."""
        project = tmp_path / "proj"
        project.mkdir()

        run = euler_train.init(dir=str(project), config={})
        assert run.project_dir == project
        assert run.dir.parent == project / "runs"
        run.finish()


class TestUpdatedAt:
    def test_init_tracks_bootstrap_artifacts(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})

        meta = _read_json(run.dir / "meta.json")
        updated_at = meta["updated_at"]

        for key in (
            "meta.json",
            "config.json",
            "code_ref.json",
            "run_environment.json",
        ):
            assert key in updated_at
            _assert_updated_at_entry(updated_at[key])
        run.finish()

    def test_log_tracks_train_and_val_jsonl(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})

        run.log({"loss": 1.0}, step=0, epoch=0)
        run.log({"metric": 0.9}, step=1, epoch=0, mode="val")

        meta = _read_json(run.dir / "meta.json")
        updated_at = meta["updated_at"]
        _assert_updated_at_entry(updated_at["train.jsonl"])
        _assert_updated_at_entry(updated_at["val.jsonl"])
        _assert_updated_at_entry(updated_at["meta.json"])
        run.finish()

    def test_save_outputs_tracks_output_directory(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)

        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        meta = _read_json(run.dir / "meta.json")
        _assert_updated_at_entry(meta["updated_at"]["outputs/epoch_0_step_0"])
        run.finish()

    def test_log_saved_checkpoint_tracks_checkpoint_path(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        checkpoint_path = tmp_path / "external" / "epoch_1.pt"

        run.log_saved_checkpoint(checkpoint_path, epoch=1, step=10)

        meta = _read_json(run.dir / "meta.json")
        _assert_updated_at_entry(meta["updated_at"][str(checkpoint_path)])
        run.finish()


class TestDatasetMetadata:
    def test_prefers_dataset_describe_for_runlog_contract(
        self, tmp_path, monkeypatch
    ):
        import euler_train.run as run_module

        def _should_not_call(_path):
            raise AssertionError("fallback ds-crawler descriptor should not be used")

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            _should_not_call,
        )

        contract = {
            "modalities": {
                "hazy_rgb": {
                    "path": "/datasets/hazy_rgb",
                    "used_as": "input",
                    "slot": "dehaze.input.rgb",
                    "modality_type": "rgb",
                }
            },
            "hierarchical_modalities": {
                "camera_intrinsics": {
                    "path": "/datasets/camera_intrinsics",
                    "used_as": "condition",
                    "slot": "dehaze.condition.camera_intrinsics",
                    "modality_type": "camera_intrinsics",
                    "hierarchy_scope": "scene_camera",
                    "applies_to": ["hazy_rgb"],
                }
            },
        }

        ds = _DatasetWithRunlogDescription(contract)
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            datasets={"train": ds},
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["datasets"]["train"] == contract

    def test_logs_rich_dataset_metadata_from_ds_crawler_properties(
        self, tmp_path, monkeypatch
    ):
        import euler_train.run as run_module

        descriptors = {
            "/datasets/hazy_rgb": {
                "modality_type": "rgb",
                "properties": {
                    "euler_train": {
                        "used_as": "input",
                        "slot": "dehaze.input.rgb",
                    }
                },
            },
            "/datasets/clear_rgb": {
                "modality_type": "rgb",
                "properties": {
                    "euler_train": {
                        "used_as": "target",
                        "slot": "dehaze.target.rgb",
                    }
                },
            },
            "/datasets/depth": {
                "modality_type": "depth",
                "properties": {
                    "euler_train": {
                        "used_as": "target",
                        "slot": "dehaze.target.depth",
                    }
                },
            },
            "/datasets/camera_intrinsics": {
                "modality_type": "camera_intrinsics",
                "properties": {
                    "euler_train": {
                        "used_as": "condition",
                        "slot": "dehaze.condition.camera_intrinsics",
                        "hierarchy_scope": "scene_camera",
                    }
                },
            },
        }

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: descriptors.get(path, {}),
        )

        ds = _DummyDataset(
            modalities={
                "hazy_rgb": "/datasets/hazy_rgb",
                "clear_rgb": "/datasets/clear_rgb",
                "depth": "/datasets/depth",
            },
            hierarchical_modalities={
                "camera_intrinsics": "/datasets/camera_intrinsics",
            },
        )

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            datasets={"train": ds},
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["datasets"]["train"] == {
            "modalities": {
                "hazy_rgb": {
                    "path": "/datasets/hazy_rgb",
                    "used_as": "input",
                    "slot": "dehaze.input.rgb",
                    "modality_type": "rgb",
                },
                "clear_rgb": {
                    "path": "/datasets/clear_rgb",
                    "used_as": "target",
                    "slot": "dehaze.target.rgb",
                    "modality_type": "rgb",
                },
                "depth": {
                    "path": "/datasets/depth",
                    "used_as": "target",
                    "slot": "dehaze.target.depth",
                    "modality_type": "depth",
                },
            },
            "hierarchical_modalities": {
                "camera_intrinsics": {
                    "path": "/datasets/camera_intrinsics",
                    "used_as": "condition",
                    "slot": "dehaze.condition.camera_intrinsics",
                    "modality_type": "camera_intrinsics",
                    "hierarchy_scope": "scene_camera",
                    "applies_to": ["hazy_rgb", "clear_rgb", "depth"],
                }
            },
        }

    def test_prefers_euler_loading_namespace_over_euler_train(
        self, tmp_path, monkeypatch
    ):
        import euler_train.run as run_module

        descriptors = {
            "/datasets/hazy_rgb": {
                "modality_type": "rgb",
                "properties": {
                    "euler_train": {
                        "used_as": "target",
                        "slot": "fallback.target.rgb",
                    },
                    "euler_loading": {
                        "used_as": "input",
                        "slot": "dehaze.input.rgb",
                        "modality_type": "rgb",
                    },
                },
            },
        }

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: descriptors.get(path, {}),
        )

        ds = _DummyDataset(modalities={"hazy_rgb": "/datasets/hazy_rgb"})
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            datasets={"train": ds},
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["datasets"]["train"] == {
            "modalities": {
                "hazy_rgb": {
                    "path": "/datasets/hazy_rgb",
                    "used_as": "input",
                    "slot": "dehaze.input.rgb",
                    "modality_type": "rgb",
                }
            },
            "hierarchical_modalities": {},
        }

    def test_logs_heuristic_dataset_metadata_without_ds_crawler(
        self, tmp_path, monkeypatch
    ):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: {},
        )

        ds = _DummyDataset(
            modalities={
                "hazy_rgb": "/datasets/hazy_rgb",
                "target_depth": "/datasets/target_depth",
            },
        )

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            datasets={"train": ds},
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["datasets"]["train"] == {
            "modalities": {
                "hazy_rgb": {
                    "path": "/datasets/hazy_rgb",
                    "used_as": "input",
                    "slot": "input.rgb",
                    "modality_type": "rgb",
                },
                "target_depth": {
                    "path": "/datasets/target_depth",
                    "used_as": "target",
                    "slot": "target.depth",
                    "modality_type": "depth",
                },
            },
            "hierarchical_modalities": {},
        }


# ═══════════════════════════════════════════════════════════════════════════
#  modality field enforcement (used_as, modality_type, slot)
# ═══════════════════════════════════════════════════════════════════════════

class TestModalityFieldEnforcement:
    def test_error_when_used_as_cannot_be_inferred(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(run_module, "_read_ds_crawler_descriptor", lambda path: {})

        ds = _DummyDataset(modalities={"mystery": "/datasets/mystery"})
        with pytest.raises(ValueError, match="could not determine 'used_as'"):
            euler_train.init(
                dir=str(tmp_path / "r"), config={},
                datasets={"train": ds},
            )

    def test_error_when_modality_type_cannot_be_inferred(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(run_module, "_read_ds_crawler_descriptor", lambda path: {})

        # "input_data" → used_as="input" works, but modality_type has no hint
        ds = _DummyDataset(modalities={"input_data": "/datasets/data"})
        with pytest.raises(ValueError, match="could not determine 'modality_type'"):
            euler_train.init(
                dir=str(tmp_path / "r"), config={},
                datasets={"train": ds},
            )

    def test_contract_missing_required_field_raises(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(run_module, "_read_ds_crawler_descriptor", lambda path: {})

        contract = {
            "modalities": {
                "mystery_data": {
                    "path": "/mnt/ds/preds/data",
                    "used_as": "output",
                    # missing modality_type and slot; name/path have no heuristic match
                }
            },
            "hierarchical_modalities": {},
        }
        ds = _DatasetWithRunlogDescription(contract)
        with pytest.raises(ValueError, match="missing required field 'modality_type'"):
            euler_train.init(
                dir=str(tmp_path / "r"), config={},
                datasets={"train": ds},
            )

    def test_contract_enriched_from_ds_crawler(self, tmp_path, monkeypatch):
        """Incomplete contract entries are filled from ds_crawler properties."""
        import euler_train.run as run_module

        descriptors = {
            "/datasets/depth": {
                "modality_type": "depth",
                "properties": {
                    "euler_train": {
                        "used_as": "target",
                        "slot": "dehaze.target.depth",
                    }
                },
            },
        }
        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: descriptors.get(path, {}),
        )

        # Contract provides only the path — the rest should come from ds_crawler
        contract = {
            "modalities": {
                "depth": {
                    "path": "/datasets/depth",
                }
            },
            "hierarchical_modalities": {},
        }
        ds = _DatasetWithRunlogDescription(contract)
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            datasets={"train": ds},
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["datasets"]["train"]["modalities"]["depth"] == {
            "path": "/datasets/depth",
            "used_as": "target",
            "slot": "dehaze.target.depth",
            "modality_type": "depth",
        }

    def test_output_json_fallback_when_no_ds_crawler_json(
        self, tmp_path, monkeypatch
    ):
        """When ds-crawler.json is absent, properties are read from output.json."""
        import euler_train.run as run_module

        output_json_data = {
            "/datasets/depth": [
                {
                    "name": "depth",
                    "type": "depth",
                    "euler_train": {
                        "used_as": "target",
                        "slot": "dehaze.target.depth",
                        "modality_type": "depth",
                    },
                    "dataset": {},
                }
            ],
        }

        def _fake_load_dataset_config(cfg):
            raise FileNotFoundError("no ds-crawler.json")

        def _fake_read_metadata_json(path, filename):
            return output_json_data.get(str(path), [])

        monkeypatch.setattr(
            run_module, "_read_ds_crawler_descriptor",
            run_module._read_ds_crawler_descriptor,  # use the real one
        )

        import ds_crawler
        monkeypatch.setattr(ds_crawler, "load_dataset_config", _fake_load_dataset_config)

        from ds_crawler import zip_utils
        monkeypatch.setattr(zip_utils, "read_metadata_json", _fake_read_metadata_json)

        contract = {
            "modalities": {
                "depth": {
                    "path": "/datasets/depth",
                }
            },
            "hierarchical_modalities": {},
        }
        ds = _DatasetWithRunlogDescription(contract)
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            datasets={"train": ds},
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["datasets"]["train"]["modalities"]["depth"] == {
            "path": "/datasets/depth",
            "used_as": "target",
            "slot": "dehaze.target.depth",
            "modality_type": "depth",
        }

    def test_warning_on_heuristic_used_as(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module, "_read_ds_crawler_descriptor",
            lambda path: {"modality_type": "rgb"},
        )

        ds = _DummyDataset(modalities={"hazy_rgb": "/datasets/hazy_rgb"})
        with pytest.warns(UserWarning, match="'used_as' was inferred"):
            euler_train.init(
                dir=str(tmp_path / "r"), config={},
                datasets={"train": ds},
            )

    def test_warning_on_heuristic_modality_type(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module, "_read_ds_crawler_descriptor",
            lambda path: {"properties": {"euler_train": {"used_as": "input"}}},
        )

        ds = _DummyDataset(modalities={"input_rgb": "/datasets/input_rgb"})
        with pytest.warns(UserWarning, match="'modality_type' was inferred"):
            euler_train.init(
                dir=str(tmp_path / "r"), config={},
                datasets={"train": ds},
            )

    def test_no_warning_when_explicit(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module, "_read_ds_crawler_descriptor",
            lambda path: {
                "modality_type": "rgb",
                "properties": {"euler_train": {"used_as": "input"}},
            },
        )

        ds = _DummyDataset(modalities={"hazy_rgb": "/datasets/hazy_rgb"})
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("error")
            run = euler_train.init(
                dir=str(tmp_path / "r"), config={},
                datasets={"train": ds},
            )
            run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  evaluations metadata
# ═══════════════════════════════════════════════════════════════════════════

class TestEvaluationMetadata:
    def test_evaluations_none_by_default(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        meta = _read_json(run.dir / "meta.json")
        assert "evaluations" not in meta
        run.finish()

    def test_evaluations_at_init_writes_to_meta(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: {},
        )

        test_ds = _DummyDataset(
            modalities={"rgb_input": "/mnt/ds/test/rgb"},
        )

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_rgb": {
                    "datasets": {"test": test_ds},
                    "name": "RGB Eval",
                    "status": "running",
                    "checkpoint": {"epoch": 12, "step": 4800},
                    "metadata": {"runner": "eval_v2"},
                },
            },
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert "evaluations" in meta
        ev = meta["evaluations"]["eval_rgb"]
        assert ev["name"] == "RGB Eval"
        assert ev["status"] == "running"
        assert ev["checkpoint"] == {"epoch": 12, "step": 4800}
        assert ev["metadata"] == {"runner": "eval_v2"}
        assert "test" in ev["datasets"]
        assert "rgb_input" in ev["datasets"]["test"]["modalities"]
        assert ev["datasets"]["test"]["modalities"]["rgb_input"]["path"] == "/mnt/ds/test/rgb"

    def test_evaluation_datasets_use_describe_for_runlog(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda _: (_ for _ in ()).throw(
                AssertionError("ds-crawler should not be called")
            ),
        )

        contract = {
            "modalities": {
                "rgb_pred": {
                    "path": "/mnt/ds/preds/rgb",
                    "used_as": "output",
                    "modality_type": "rgb",
                    "slot": "output.rgb",
                }
            },
            "hierarchical_modalities": {},
        }
        test_ds = _DatasetWithRunlogDescription(contract)

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_rgb": {"datasets": {"test": test_ds}},
            },
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        assert meta["evaluations"]["eval_rgb"]["datasets"]["test"] == contract

    def test_evaluation_datasets_use_heuristic_inference(
        self, tmp_path, monkeypatch
    ):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: {},
        )

        test_ds = _DummyDataset(
            modalities={
                "input_rgb": "/mnt/ds/test/rgb",
                "target_depth": "/mnt/ds/test/depth",
            },
        )

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_multi": {"datasets": {"test": test_ds}},
            },
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        mods = meta["evaluations"]["eval_multi"]["datasets"]["test"]["modalities"]
        assert mods["input_rgb"]["used_as"] == "input"
        assert mods["target_depth"]["used_as"] == "target"

    def test_evaluation_passthrough_fields(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_a": {
                    "name": "Eval A",
                    "status": "running",
                    "checkpoint": {"epoch": 5, "step": 2000, "name": "epoch_5"},
                    "metadata": {"gpu": "A100"},
                },
            },
        )
        meta = _read_json(run.dir / "meta.json")
        run.finish()

        ev = meta["evaluations"]["eval_a"]
        assert ev["name"] == "Eval A"
        assert ev["status"] == "running"
        assert ev["checkpoint"] == {"epoch": 5, "step": 2000, "name": "epoch_5"}
        assert ev["metadata"] == {"gpu": "A100"}
        # No datasets key when none provided
        assert "datasets" not in ev

    def test_evaluations_merged_on_resume(self, tmp_path):
        # Create initial run with an evaluation
        run1 = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_a": {"name": "First", "status": "completed"},
            },
        )
        run_id = run1.run_id
        run1.finish()

        # Resume and add a second evaluation
        run2 = euler_train.init(
            dir=str(tmp_path / "r"),
            run_id=run_id,
            evaluations={
                "eval_b": {"name": "Second", "status": "running"},
            },
        )
        meta = _read_json(run2.dir / "meta.json")
        run2.finish()

        # Both evaluations present
        assert "eval_a" in meta["evaluations"]
        assert "eval_b" in meta["evaluations"]
        assert meta["evaluations"]["eval_a"]["name"] == "First"
        assert meta["evaluations"]["eval_b"]["name"] == "Second"

    def test_add_evaluation_method(self, tmp_path, monkeypatch):
        import euler_train.run as run_module

        monkeypatch.setattr(
            run_module,
            "_read_ds_crawler_descriptor",
            lambda path: {},
        )

        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        test_ds = _DummyDataset(modalities={"input_rgb": "/mnt/ds/test/rgb"})

        run.add_evaluation(
            "eval_rgb",
            datasets={"test": test_ds},
            name="RGB Eval",
            status="running",
            checkpoint={"epoch": 10, "step": 5000},
        )

        meta = _read_json(run.dir / "meta.json")
        assert "evaluations" in meta
        ev = meta["evaluations"]["eval_rgb"]
        assert ev["name"] == "RGB Eval"
        assert ev["status"] == "running"
        assert ev["checkpoint"] == {"epoch": 10, "step": 5000}
        assert ev["datasets"]["test"]["modalities"]["input_rgb"]["path"] == "/mnt/ds/test/rgb"
        run.finish()

    def test_add_evaluation_updates_existing(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_a": {"name": "Original", "status": "running"},
            },
        )

        run.add_evaluation("eval_a", status="completed", metadata={"score": 0.95})
        meta = _read_json(run.dir / "meta.json")

        ev = meta["evaluations"]["eval_a"]
        assert ev["name"] == "Original"  # preserved from initial
        assert ev["status"] == "completed"  # updated
        assert ev["metadata"] == {"score": 0.95}  # added
        run.finish()

    def test_finish_evaluation(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_a": {"name": "Eval A", "status": "running"},
            },
        )

        run.finish_evaluation("eval_a")
        meta = _read_json(run.dir / "meta.json")

        assert meta["evaluations"]["eval_a"]["status"] == "completed"
        assert meta["evaluations"]["eval_a"]["name"] == "Eval A"
        run.finish()

    def test_finish_evaluation_custom_status(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            evaluations={
                "eval_a": {"status": "running"},
            },
        )

        run.finish_evaluation("eval_a", status="crashed")
        meta = _read_json(run.dir / "meta.json")

        assert meta["evaluations"]["eval_a"]["status"] == "crashed"
        run.finish()

    def test_finish_evaluation_missing_key_raises(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})

        with pytest.raises(KeyError, match="eval_missing"):
            run.finish_evaluation("eval_missing")
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  log  (train.jsonl  /  val.jsonl)
# ═══════════════════════════════════════════════════════════════════════════

class TestLog:
    def test_train_log_creates_file_and_appends(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log({"loss": 2.3, "lr": 1e-4}, step=0, epoch=0)
        run.log({"loss": 2.1, "lr": 1e-4}, step=1, epoch=0)

        records = _read_jsonl(run.dir / "train.jsonl")
        assert len(records) == 2
        assert records[0]["step"] == 0
        assert records[1]["loss"] == 2.1
        run.finish()

    def test_train_log_has_required_fields(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log({"loss": 1.0}, step=5, epoch=1)

        rec = _read_jsonl(run.dir / "train.jsonl")[0]
        assert rec["step"] == 5
        assert rec["epoch"] == 1
        assert "wall_time" in rec
        assert "elapsed_sec" in rec  # train-only field
        assert rec["loss"] == 1.0
        run.finish()

    def test_val_log_writes_to_val_file(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log({"rgb.psnr": 25.0, "rgb.ssim": 0.91}, step=100, epoch=1, mode="val")

        assert not (run.dir / "train.jsonl").exists()
        records = _read_jsonl(run.dir / "val.jsonl")
        assert len(records) == 1
        assert records[0]["rgb.psnr"] == 25.0
        run.finish()

    def test_val_log_no_elapsed_sec(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log({"metric": 1.0}, step=0, epoch=0, mode="val")

        rec = _read_jsonl(run.dir / "val.jsonl")[0]
        assert "elapsed_sec" not in rec
        run.finish()

    def test_log_dynamic_keys(self, tmp_path):
        """Caller can pass arbitrary metric names."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log({"custom_metric_xyz": 42, "another.nested.key": 0.5}, step=0, epoch=0)

        rec = _read_jsonl(run.dir / "train.jsonl")[0]
        assert rec["custom_metric_xyz"] == 42
        assert rec["another.nested.key"] == 0.5
        run.finish()

    def test_log_numpy_scalars_serialised(self, tmp_path):
        """numpy dtypes should be serialised to plain Python types."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log({"loss": np.float32(1.5), "n": np.int64(10)}, step=0, epoch=0)

        rec = _read_jsonl(run.dir / "train.jsonl")[0]
        assert isinstance(rec["loss"], float)
        assert isinstance(rec["n"], int)
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_outputs  — format inference
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveOutputsFormatInference:
    """Verify the right file extension is chosen based on array shape/dtype."""

    def _single_file(self, slot_dir: Path) -> Path:
        files = list(slot_dir.iterdir())
        assert len(files) == 1, f"Expected 1 file in {slot_dir}, got {len(files)}"
        return files[0]

    # ---- images → .png ---------------------------------------------------

    def test_uint8_hwc3_saved_as_png(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred")
        assert f.suffix == ".png"
        img = Image.open(f)
        assert img.size == (32, 32)
        assert img.mode == "RGB"
        run.finish()

    def test_uint8_hwc4_saved_as_rgba_png(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (16, 16, 4), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgba=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "rgba" / "pred")
        assert f.suffix == ".png"
        assert Image.open(f).mode == "RGBA"
        run.finish()

    def test_uint8_hwc1_saved_as_grayscale_png(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (16, 16, 1), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, gray=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "gray" / "pred")
        assert f.suffix == ".png"
        assert Image.open(f).mode == "L"
        run.finish()

    def test_uint8_hw_saved_as_grayscale_png(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, mask=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "mask" / "pred")
        assert f.suffix == ".png"
        run.finish()

    def test_float_hwc3_saved_as_png_scaled(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred")
        assert f.suffix == ".png"
        img = np.array(Image.open(f))
        # float [0,1] → uint8 [0,255]
        assert img.dtype == np.uint8
        assert img.max() > 0
        run.finish()

    def test_float_hwc3_clips_oob_values(self, tmp_path):
        """Values outside [0,1] should be clipped, not wrap around."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.full((8, 8, 3), 1.5, dtype=np.float32)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred")
        img = np.array(Image.open(f))
        assert (img == 255).all()
        run.finish()

    # ---- non-image arrays → .npy -----------------------------------------

    def test_float32_hw_saved_as_npy(self, tmp_path):
        """A float HxW array (e.g. depth map) should default to .npy."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        depth = np.random.rand(32, 32).astype(np.float32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=depth))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "depth" / "pred")
        assert f.suffix == ".npy"
        loaded = np.load(f)
        np.testing.assert_array_equal(loaded, depth)
        run.finish()

    def test_float64_hw_saved_as_npy(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.rand(16, 16)  # float64 by default
        run.save_outputs(epoch=0, step=0, field=dict(gt=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "field" / "gt")
        assert f.suffix == ".npy"
        run.finish()

    def test_high_channel_count_saved_as_npy(self, tmp_path):
        """HxWx64 is *not* image-like — should be .npy."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.rand(16, 16, 64).astype(np.float32)
        run.save_outputs(epoch=0, step=0, feat=dict(pred=arr))

        f = self._single_file(run.dir / "outputs" / "epoch_0_step_0" / "feat" / "pred")
        assert f.suffix == ".npy"
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_outputs  — format overrides
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveOutputsOverrides:
    def test_override_by_output_type(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"depth": "npz"},
        )
        arr = np.random.rand(16, 16).astype(np.float32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=arr))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "depth" / "pred").iterdir())[0]
        assert f.suffix == ".npz"
        run.finish()

    def test_override_by_leaf_name(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"pred": "npz"},
        )
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred").iterdir())[0]
        assert f.suffix == ".npz"
        run.finish()

    def test_override_specific_dotted_key(self, tmp_path):
        """'rgb.pred' override should only affect rgb/pred, not depth/pred."""
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"rgb.pred": "npy"},
        )
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        run.save_outputs(
            epoch=0, step=0,
            rgb=dict(pred=arr, gt=arr),
        )

        base = run.dir / "outputs" / "epoch_0_step_0" / "rgb"
        pred_f = list((base / "pred").iterdir())[0]
        gt_f = list((base / "gt").iterdir())[0]
        assert pred_f.suffix == ".npy"   # overridden
        assert gt_f.suffix == ".png"     # still inferred
        run.finish()


class TestSaveOutputsVisualization:
    def test_fixed_range_visualization_for_single_channel_png(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            output_visualization={
                "depth": {"mode": "fixed_range", "vmin": 0.0, "vmax": 10.0},
            },
        )
        depth = np.linspace(0.0, 10.0, 16, dtype=np.float32).reshape(4, 4, 1)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=depth))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "depth" / "pred").iterdir())[0]
        img = np.array(Image.open(f))
        assert f.suffix == ".png"
        assert img.min() == 0
        assert img.max() == 255
        run.finish()

    def test_visualization_override_specific_dotted_key(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            output_visualization={
                "depth.pred": {"mode": "fixed_range", "vmin": 5.0, "vmax": 15.0},
            },
        )
        depth = np.linspace(5.0, 15.0, 16, dtype=np.float32).reshape(4, 4, 1)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=depth, gt=depth))

        base = run.dir / "outputs" / "epoch_0_step_0" / "depth"
        pred_f = list((base / "pred").iterdir())[0]
        gt_f = list((base / "gt").iterdir())[0]
        pred_img = np.array(Image.open(pred_f))
        gt_img = np.array(Image.open(gt_f))
        assert pred_img.min() == 0
        assert pred_img.max() == 255
        assert gt_img.min() == 255
        assert gt_img.max() == 255
        run.finish()

    def test_torch_bfloat16_single_channel_png_promoted_to_float32(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        t = torch.linspace(0, 1, steps=16, dtype=torch.bfloat16).reshape(4, 4, 1)
        run.save_outputs(epoch=0, step=0, confidence=dict(pred=t))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "confidence" / "pred").iterdir())[0]
        img = np.array(Image.open(f))
        assert f.suffix == ".png"
        assert img.dtype == np.uint8
        assert img.min() == 0
        assert img.max() >= 254
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_outputs  — batches, lists, aux, PIL, torch
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveOutputsVariants:
    def test_list_of_arrays_saved_individually(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        imgs = [np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(4)]
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=imgs))

        pred_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        files = sorted(pred_dir.iterdir())
        assert len(files) == 4
        assert [f.name for f in files] == ["0000.png", "0001.png", "0002.png", "0003.png"]
        run.finish()

    def test_4d_array_split_as_batch(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        batch = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=batch))

        pred_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        assert len(list(pred_dir.iterdir())) == 3
        run.finish()

    def test_aux_creates_named_subdirectories(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        t_map = np.random.rand(16, 16).astype(np.float32)
        attn = np.random.rand(16, 16).astype(np.float32)
        run.save_outputs(
            epoch=1, step=50,
            depth=dict(aux=dict(transmission=t_map, attention=attn)),
        )

        aux_base = run.dir / "outputs" / "epoch_1_step_50" / "depth" / "aux"
        assert (aux_base / "transmission").is_dir()
        assert (aux_base / "attention").is_dir()
        assert list((aux_base / "transmission").iterdir())[0].suffix == ".npy"
        run.finish()

    def test_none_slots_are_skipped(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr, gt=None, input=None))

        rgb_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb"
        assert (rgb_dir / "pred").is_dir()
        assert not (rgb_dir / "gt").exists()
        assert not (rgb_dir / "input").exists()
        run.finish()

    def test_none_output_type_skipped(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr), depth=None)

        out_dir = run.dir / "outputs" / "epoch_0_step_0"
        assert (out_dir / "rgb").is_dir()
        assert not (out_dir / "depth").exists()
        run.finish()

    def test_pil_image_saved_as_png(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        img = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=img))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred").iterdir())[0]
        assert f.suffix == ".png"
        run.finish()

    def test_torch_tensor_chw_transposed_and_saved(self, tmp_path):
        """A (C,H,W) torch tensor should be transposed to (H,W,C) and saved."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        t = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=t))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred").iterdir())[0]
        assert f.suffix == ".png"
        img = np.array(Image.open(f))
        assert img.shape == (32, 32, 3)
        # Verify pixel values match the original tensor (CHW → HWC)
        expected = t.numpy().transpose(1, 2, 0)
        np.testing.assert_array_equal(img, expected)
        run.finish()

    def test_torch_tensor_bchw_becomes_batch(self, tmp_path):
        """A (B,C,H,W) torch tensor should produce B separate files."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        t = torch.randint(0, 255, (5, 3, 16, 16), dtype=torch.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=t))

        pred_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        assert len(list(pred_dir.iterdir())) == 5
        run.finish()

    def test_torch_float_tensor_saved_as_npy(self, tmp_path):
        """A 2-D float torch tensor (depth-like) should default to .npy."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        t = torch.rand(32, 32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=t))

        f = list((run.dir / "outputs" / "epoch_0_step_0" / "depth" / "pred").iterdir())[0]
        assert f.suffix == ".npy"
        loaded = np.load(f)
        np.testing.assert_allclose(loaded, t.numpy(), atol=1e-6)
        run.finish()

    def test_directory_naming_epoch_only(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(epoch=3, rgb=dict(pred=arr))
        assert path.name == "epoch_3"
        run.finish()

    def test_directory_naming_step_only(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(step=999, rgb=dict(pred=arr))
        assert path.name == "step_999"
        run.finish()

    def test_directory_naming_both(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(epoch=2, step=500, rgb=dict(pred=arr))
        assert path.name == "epoch_2_step_500"
        run.finish()

    def test_dict_slot_produces_named_files(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        b = np.ones((4, 4, 3), dtype=np.uint8) * 128
        run.save_outputs(epoch=0, step=0, rgb=dict(pred={"scene_a": a, "scene_b": b}))

        pred_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        names = sorted(f.name for f in pred_dir.iterdir())
        assert names == ["scene_a.png", "scene_b.png"]
        run.finish()

    def test_dict_slot_in_aux(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        m1 = np.ones((4, 4), dtype=np.float32)
        m2 = np.zeros((4, 4), dtype=np.float32)
        run.save_outputs(
            epoch=0, step=0,
            depth=dict(aux=dict(attn={"head_0": m1, "head_1": m2})),
        )

        attn_dir = run.dir / "outputs" / "epoch_0_step_0" / "depth" / "aux" / "attn"
        names = sorted(f.name for f in attn_dir.iterdir())
        assert names == ["head_0.npy", "head_1.npy"]
        run.finish()

    def test_mixed_dict_and_array_slots(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred={"frame_x": a}, gt=a))

        pred_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        gt_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "gt"
        assert sorted(f.name for f in pred_dir.iterdir()) == ["frame_x.png"]
        assert sorted(f.name for f in gt_dir.iterdir()) == ["0000.png"]
        run.finish()

    def test_returns_output_base_path(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))
        assert path == run.dir / "outputs" / "epoch_0_step_0"
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  Output manifest
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveOutputsManifest:
    def test_manifest_written_for_basic_save(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(epoch=1, step=500, rgb=dict(pred=arr))

        m = _read_json(run.dir / "outputs" / "epoch_1_step_500" / "manifest.json")
        assert m["version"] == 1
        assert m["epoch"] == 1
        assert m["step"] == 500
        assert "rgb" in m["output_types"]
        assert "pred" in m["output_types"]["rgb"]
        run.finish()

    def test_manifest_indexed_files(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        imgs = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=imgs))

        slot = _read_json(
            run.dir / "outputs" / "epoch_0_step_0" / "manifest.json",
        )["output_types"]["rgb"]["pred"]
        assert slot["id_mode"] == "indexed"
        assert len(slot["files"]) == 3
        assert slot["files"][0] == {"sample_id": 0, "filename": "0000.png", "format": "png"}
        assert slot["files"][2]["sample_id"] == 2
        run.finish()

    def test_manifest_named_files(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        a = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred={"scene_a": a, "scene_b": a}))

        slot = _read_json(
            run.dir / "outputs" / "epoch_0_step_0" / "manifest.json",
        )["output_types"]["rgb"]["pred"]
        assert slot["id_mode"] == "named"
        ids = [f["sample_id"] for f in slot["files"]]
        assert "scene_a" in ids
        assert "scene_b" in ids
        run.finish()

    def test_manifest_aux_slots(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.ones((4, 4), dtype=np.float32)
        run.save_outputs(epoch=0, step=0, depth=dict(aux=dict(transmission=arr)))

        m = _read_json(run.dir / "outputs" / "epoch_0_step_0" / "manifest.json")
        assert "aux/transmission" in m["output_types"]["depth"]
        run.finish()

    def test_manifest_none_slots_excluded(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr, gt=None))

        slots = _read_json(
            run.dir / "outputs" / "epoch_0_step_0" / "manifest.json",
        )["output_types"]["rgb"]
        assert "pred" in slots
        assert "gt" not in slots
        run.finish()

    def test_manifest_records_format_overrides(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"depth": "npz"},
        )
        arr = np.ones((4, 4), dtype=np.float32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=arr))

        m = _read_json(run.dir / "outputs" / "epoch_0_step_0" / "manifest.json")
        assert m["format_overrides"] == {"depth": "npz"}
        assert m["output_types"]["depth"]["pred"]["files"][0]["format"] == "npz"
        run.finish()

    def test_manifest_mixed_named_and_indexed(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred={"x": arr}, gt=arr))

        slots = _read_json(
            run.dir / "outputs" / "epoch_0_step_0" / "manifest.json",
        )["output_types"]["rgb"]
        assert slots["pred"]["id_mode"] == "named"
        assert slots["gt"]["id_mode"] == "indexed"
        run.finish()

    def test_manifest_metadata(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(
            epoch=0, step=0,
            metadata={"dataset": "vkitti2", "split": "val"},
            rgb=dict(pred=arr),
        )

        m = _read_json(run.dir / "outputs" / "epoch_0_step_0" / "manifest.json")
        assert m["metadata"] == {"dataset": "vkitti2", "split": "val"}
        run.finish()

    def test_manifest_metadata_absent_when_not_provided(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        m = _read_json(run.dir / "outputs" / "epoch_0_step_0" / "manifest.json")
        assert "metadata" not in m
        run.finish()

    def test_manifest_metadata_requires_dataset(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="dataset"):
            run.save_outputs(
                epoch=0, step=0,
                metadata={"split": "val"},
                rgb=dict(pred=arr),
            )
        run.finish()

    def test_manifest_metadata_rejects_empty_dataset(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="dataset"):
            run.save_outputs(
                epoch=0, step=0,
                metadata={"dataset": "  "},
                rgb=dict(pred=arr),
            )
        run.finish()

    def test_no_manifest_when_all_none(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.save_outputs(epoch=0, step=0, rgb=None, depth=None)

        manifest_path = run.dir / "outputs" / "epoch_0_step_0" / "manifest.json"
        assert not manifest_path.exists()
        run.finish()

    def test_manifest_multiple_output_types(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        dep = np.ones((4, 4), dtype=np.float32)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=img), depth=dict(pred=dep))

        m = _read_json(run.dir / "outputs" / "epoch_0_step_0" / "manifest.json")
        assert m["output_types"]["rgb"]["pred"]["files"][0]["format"] == "png"
        assert m["output_types"]["depth"]["pred"]["files"][0]["format"] == "npy"
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  init_checkpoint_dir
# ═══════════════════════════════════════════════════════════════════════════

class TestInitCheckpointDir:
    def test_uses_slugified_run_name(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            run_name="Baseline Dehaze",
        )

        path = run.init_checkpoint_dir(base=tmp_path / "external-checkpoints")
        meta = _read_json(run.dir / "meta.json")

        assert path == tmp_path / "external-checkpoints" / "baseline-dehaze"
        assert meta["checkpoint_dir"] == str(path)
        assert meta["run_name"] == "Baseline Dehaze"
        run.finish()

    def test_fresh_run_disambiguates_checkpoint_dir_collision(self, tmp_path):
        import euler_train.run as run_module

        existing = tmp_path / "external-checkpoints" / "baseline-dehaze"
        existing.mkdir(parents=True)
        (existing / "epoch_0.pt").write_text("old checkpoint")

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            run_name="Baseline Dehaze",
        )

        path = run.init_checkpoint_dir(base=tmp_path / "external-checkpoints")
        meta = _read_json(run.dir / "meta.json")

        assert path != existing
        assert path == tmp_path / "external-checkpoints" / (
            f"baseline-dehaze-{run_module._slugify(run.run_id)}"
        )
        assert meta["checkpoint_dir"] == str(path)
        assert meta["run_name"] == "Baseline Dehaze"
        run.finish()

    def test_resume_reuses_recorded_checkpoint_dir(self, tmp_path):
        model = torch.nn.Linear(4, 2)

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            run_name="Baseline Dehaze",
        )
        checkpoint_dir = run.init_checkpoint_dir(base=tmp_path / "external-checkpoints")
        run.save_checkpoint(model, epoch=1, step=100)
        run.finish()

        resumed = euler_train.init(
            dir=str(tmp_path / "r"),
            run_id=run.run_id,
        )

        assert resumed.run_name == "Baseline Dehaze"
        assert resumed.checkpoint_dir == checkpoint_dir
        assert resumed.init_checkpoint_dir(base=tmp_path / "ignored") == checkpoint_dir

        path = resumed.save_checkpoint(model, epoch=2, step=200)
        assert path.parent == checkpoint_dir
        resumed.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_checkpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveCheckpoint:
    def test_saves_model_state_dict(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=3, step=1200)

        assert path == run.dir / "checkpoints" / "epoch_3.pt"
        assert path.exists()
        ckpt = torch.load(path, weights_only=False)
        assert "model" in ckpt
        assert "epoch" in ckpt
        assert ckpt["epoch"] == 3
        assert ckpt["step"] == 1200
        # Verify weights match
        for k in model.state_dict():
            torch.testing.assert_close(ckpt["model"][k], model.state_dict()[k])
        run.finish()

    def test_saves_raw_dict_as_model(self, tmp_path):
        raw = {"weight": torch.randn(3, 3)}
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(raw, epoch=0, step=0)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["model"] is not None
        # raw dict has no state_dict(), so it's stored directly
        torch.testing.assert_close(ckpt["model"]["weight"], raw["weight"])
        run.finish()

    def test_saves_optimizer(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=1, step=500, optimizer=opt)

        ckpt = torch.load(path, weights_only=False)
        assert "optimizer" in ckpt
        assert "param_groups" in ckpt["optimizer"]
        run.finish()

    def test_saves_extra_kwargs(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=5, step=9999, best_loss=0.42)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["best_loss"] == 0.42
        assert ckpt["step"] == 9999
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  log_saved_checkpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestLogSavedCheckpoint:
    def test_save_checkpoint_logs_to_meta(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=3, step=1200)

        meta = _read_json(run.dir / "meta.json")
        assert "checkpoints" in meta
        assert len(meta["checkpoints"]) == 1
        assert meta["checkpoints"][0]["path"] == str(path)
        assert meta["checkpoints"][0]["epoch"] == 3
        assert meta["checkpoints"][0]["step"] == 1200
        run.finish()

    def test_log_saved_checkpoint_external(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log_saved_checkpoint("/custom/path/model.pt", epoch=5, step=2000)

        meta = _read_json(run.dir / "meta.json")
        assert len(meta["checkpoints"]) == 1
        entry = meta["checkpoints"][0]
        assert entry["path"] == "/custom/path/model.pt"
        assert entry["epoch"] == 5
        assert entry["step"] == 2000
        run.finish()

    def test_multiple_checkpoints_appended(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log_saved_checkpoint("/ckpt/e1.pt", epoch=1, step=400)
        run.log_saved_checkpoint("/ckpt/e2.pt", epoch=2, step=800)
        run.log_saved_checkpoint("/ckpt/e3.pt", epoch=3, step=1200)

        meta = _read_json(run.dir / "meta.json")
        assert len(meta["checkpoints"]) == 3
        assert [c["epoch"] for c in meta["checkpoints"]] == [1, 2, 3]
        assert [c["step"] for c in meta["checkpoints"]] == [400, 800, 1200]
        run.finish()

    def test_is_best_clears_previous(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.log_saved_checkpoint("/ckpt/e1.pt", epoch=1, step=400, is_best=True)
        run.log_saved_checkpoint("/ckpt/e2.pt", epoch=2, step=800)
        run.log_saved_checkpoint("/ckpt/e3.pt", epoch=3, step=1200, is_best=True)

        meta = _read_json(run.dir / "meta.json")
        assert len(meta["checkpoints"]) == 3
        # Only the last one should have is_best
        assert "is_best" not in meta["checkpoints"][0]
        assert "is_best" not in meta["checkpoints"][1]
        assert meta["checkpoints"][2]["is_best"] is True
        run.finish()

    def test_no_checkpoints_key_when_none_logged(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        meta = _read_json(run.dir / "meta.json")
        assert "checkpoints" not in meta
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  streaming
# ═══════════════════════════════════════════════════════════════════════════

class TestStreaming:
    @pytest.mark.parametrize(
        ("field", "replacement"),
        [
            ("access_token", "api_token"),
            ("accessToken", "api_token"),
            ("launch_id", "stream_attach_token"),
            ("launchId", "stream_attach_token"),
        ],
    )
    def test_removed_stream_config_aliases_are_rejected(
        self,
        field,
        replacement,
    ):
        with pytest.raises(
            ValueError,
            match=rf"{field!r}.*{replacement!r}",
        ):
            stream_mod.coerce_output_stream(
                {
                    "base_url": "https://sync.example",
                    "model_id": 42,
                    "api_token": "user-token",
                    field: "old-value",
                },
            )

    def test_run_emits_expected_stream_events(self, tmp_path):
        consumer = _RecordingStreamConsumer()

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={"lr": 1e-3},
            meta={"tags": ["baseline", "fast"]},
            stream=consumer,
        )
        run.add_evaluation(
            "eval_rgb",
            name="RGB Eval",
            status="running",
            metadata={"seed": 7},
        )
        run.finish_evaluation("eval_rgb")
        run.log({"loss": 0.5}, step=0, epoch=0, mode="train")
        run.log({"psnr": 28.1}, step=1, epoch=0, mode="val")
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        aux = np.ones((4, 4), dtype=np.float32)
        run.save_outputs(
            epoch=0,
            step=1,
            rgb=dict(pred=arr, gt=arr),
            depth=dict(aux=dict(transmission=aux)),
        )
        run.log_saved_checkpoint("/tmp/model.pt", epoch=0, step=1, is_best=True)
        run.finish()

        assert consumer.context.run_id == run.run_id
        assert consumer.context.run_dir == run.dir
        assert consumer.close_count == 1
        assert consumer.flush_count >= 2

        event_types = [event["type"] for event in consumer.events]
        assert event_types == [
            "init",
            "meta",
            "meta",
            "metric",
            "metric",
            "output_snapshot",
            "checkpoint",
            "finish",
        ]

        init_event = consumer.events[0]
        assert init_event["meta"]["run_id"] == run.run_id
        assert init_event["config"] == {"lr": 1e-3}
        assert init_event["tags"] == ["baseline", "fast"]
        assert isinstance(init_event["codeRef"], dict)
        assert isinstance(init_event["runEnvironment"], dict)

        add_eval_event = consumer.events[1]
        assert add_eval_event["patch"] == {
            "evaluations": {
                "eval_rgb": {
                    "name": "RGB Eval",
                    "status": "running",
                    "metadata": {"seed": 7},
                },
            },
        }

        finish_eval_event = consumer.events[2]
        assert finish_eval_event["patch"] == {
            "evaluations": {
                "eval_rgb": {
                    "status": "completed",
                },
            },
        }

        train_event = consumer.events[3]
        assert train_event["split"] == "train"
        assert train_event["records"][0]["loss"] == 0.5
        assert "elapsed_sec" in train_event["records"][0]

        val_event = consumer.events[4]
        assert val_event["split"] == "val"
        assert val_event["records"][0]["psnr"] == 28.1
        assert "elapsed_sec" not in val_event["records"][0]

        snapshot_event = consumer.events[5]
        snapshot = snapshot_event["snapshot"]
        assert snapshot["version"] == 1
        assert snapshot["epoch"] == 0
        assert snapshot["step"] == 1
        assert set(snapshot["output_types"]) == {"rgb", "depth"}
        assert set(snapshot["output_types"]["rgb"]) == {"pred", "gt"}
        assert "aux/transmission" in snapshot["output_types"]["depth"]

        checkpoint_event = consumer.events[6]
        assert checkpoint_event["checkpoint"] == {
            "path": "/tmp/model.pt",
            "epoch": 0,
            "step": 1,
            "is_best": True,
        }

        finish_event = consumer.events[7]
        assert finish_event["status"] == "completed"
        assert finish_event["patch"]["end_time"] is not None
        assert finish_event["patch"]["duration_sec"] >= 0

    def test_euler_view_http_consumer_uses_session_and_ingest_routes(
        self,
        tmp_path,
        monkeypatch,
    ):
        calls: list[dict] = []

        def fake_urlopen(request, timeout=0):
            headers = {k.lower(): v for k, v in request.header_items()}
            body = json.loads(request.data.decode("utf-8"))
            calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "headers": headers,
                    "body": body,
                    "timeout": timeout,
                },
            )
            if request.full_url == "https://sync.example/api/model-run-stream/session":
                return _FakeHttpResponse(
                    {
                        "token": "stream-token",
                        "expiresAt": "2099-01-01T00:00:00+00:00",
                        "ingestUrl": "https://sync.example/api/model-run-stream/ingest",
                    },
                )
            if request.full_url.endswith("/api/model-run-stream/ingest"):
                return _FakeHttpResponse(
                    {
                        "success": True,
                        "events": [],
                        "latestCursor": len(calls),
                    },
                )
            raise AssertionError(f"Unexpected URL {request.full_url}")

        monkeypatch.setattr(stream_mod, "urlopen", fake_urlopen)

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={"lr": 1e-3},
            stream={
                "base_url": "https://sync.example",
                "model_id": 42,
                "api_token": "user-token",
                "datasource_id": 7,
                "batch_size": 1,
            },
        )
        run.log({"loss": 0.25}, step=3, epoch=1)
        run.finish()

        assert calls[0]["url"] == "https://sync.example/api/model-run-stream/session"
        assert calls[0]["method"] == "POST"
        assert calls[0]["headers"]["authorization"] == "Bearer user-token"
        assert calls[0]["body"] == {
            "runId": run.run_id,
            "modelId": 42,
            "runDir": str(run.dir),
            "eulerTrainDir": str(run.dir.parent),
            "datasourceId": 7,
        }

        ingest_calls = calls[1:]
        assert [call["body"]["events"][0]["type"] for call in ingest_calls] == [
            "init",
            "metric",
            "finish",
        ]
        for ingest_call in ingest_calls:
            assert ingest_call["url"] == "https://sync.example/api/model-run-stream/ingest"
            assert ingest_call["headers"]["authorization"] == "Bearer stream-token"

        metric_call = ingest_calls[1]
        metric_event = metric_call["body"]["events"][0]
        assert metric_event["split"] == "train"
        assert metric_event["records"][0]["step"] == 3
        assert metric_event["records"][0]["epoch"] == 1
        assert metric_event["records"][0]["loss"] == 0.25

        finish_event = ingest_calls[2]["body"]["events"][0]
        assert finish_event["status"] == "completed"

    def test_euler_view_http_consumer_can_handshake_without_model_id_when_attach_token_is_present(
        self,
        tmp_path,
        monkeypatch,
    ):
        calls: list[dict] = []

        def fake_urlopen(request, timeout=0):
            headers = {k.lower(): v for k, v in request.header_items()}
            body = json.loads(request.data.decode("utf-8"))
            calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "headers": headers,
                    "body": body,
                    "timeout": timeout,
                },
            )
            if request.full_url.endswith("/api/model-run-stream/session"):
                return _FakeHttpResponse(
                    {
                        "token": "stream-token",
                        "expiresAt": "2099-01-01T00:00:00+00:00",
                        "ingestUrl": "https://sync.example/api/model-run-stream/ingest",
                        "run": {
                            "modelId": 42,
                            "runId": "placeholder",
                            "streamAttachToken": "attach-123",
                        },
                    },
                )
            if request.full_url.endswith("/api/model-run-stream/ingest"):
                return _FakeHttpResponse(
                    {
                        "success": True,
                        "events": [],
                        "latestCursor": len(calls),
                    },
                )
            raise AssertionError(f"Unexpected URL {request.full_url}")

        monkeypatch.setattr(stream_mod, "urlopen", fake_urlopen)

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={"lr": 1e-3},
            stream={
                "base_url": "https://sync.example",
                "api_token": "user-token",
                "stream_attach_token": "attach-123",
                "batch_size": 1,
            },
        )
        run.log({"loss": 0.25}, step=3, epoch=1)
        run.finish()

        assert calls[0]["url"] == "https://sync.example/api/model-run-stream/session"
        assert calls[0]["method"] == "POST"
        assert calls[0]["headers"]["authorization"] == "Bearer user-token"
        assert calls[0]["body"] == {
            "runId": run.run_id,
            "runDir": str(run.dir),
            "eulerTrainDir": str(run.dir.parent),
            "streamAttachToken": "attach-123",
        }

    def test_stream_network_failures_do_not_break_training(
        self,
        tmp_path,
        monkeypatch,
    ):
        def fake_urlopen(request, timeout=0):
            raise URLError("offline")

        monkeypatch.setattr(stream_mod, "urlopen", fake_urlopen)

        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            stream={
                "base_url": "https://sync.example",
                "model_id": 42,
                "api_token": "user-token",
                "batch_size": 1,
            },
        )
        run.log({"loss": 1.0}, step=0, epoch=0)
        run.finish()

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "completed"

    def test_check_stream_handshake_uses_dry_run_route(
        self,
        monkeypatch,
    ):
        calls: list[dict] = []

        def fake_urlopen(request, timeout=0):
            headers = {k.lower(): v for k, v in request.header_items()}
            body = json.loads(request.data.decode("utf-8"))
            calls.append(
                {
                    "url": request.full_url,
                    "method": request.get_method(),
                    "headers": headers,
                    "body": body,
                    "timeout": timeout,
                },
            )
            if request.full_url == "https://sync.example/api/model-run-stream/check":
                return _FakeHttpResponse(
                    {
                        "success": True,
                        "resolution": "stream_attach_token",
                        "ingestUrl": "https://sync.example/api/model-run-stream/ingest",
                        "run": {
                            "modelId": 42,
                            "runId": "dry-run-1",
                            "streamAttachToken": "attach-123",
                            "datasourceId": 7,
                            "eulerTrainDir": "/outputs",
                            "runDir": "/outputs/dry-run-1",
                        },
                    },
                )
            raise AssertionError(f"Unexpected URL {request.full_url}")

        monkeypatch.setattr(stream_mod, "urlopen", fake_urlopen)

        payload = stream_mod.check_stream_handshake(
            {
                "base_url": "https://sync.example",
                "api_token": "user-token",
                "stream_attach_token": "attach-123",
                "datasource_id": 7,
            },
            run_id="dry-run-1",
        )

        assert payload["success"] is True
        assert calls == [
            {
                "url": "https://sync.example/api/model-run-stream/check",
                "method": "POST",
                "headers": {
                    "accept": "application/json",
                    "authorization": "Bearer user-token",
                    "content-type": "application/json",
                },
                "body": {
                    "runId": "dry-run-1",
                    "streamAttachToken": "attach-123",
                    "datasourceId": 7,
                },
                "timeout": 10.0,
            },
        ]

    def test_stream_check_cli_prints_json_payload(
        self,
        monkeypatch,
        capsys,
    ):
        def fake_urlopen(request, timeout=0):
            if request.full_url != "https://sync.example/api/model-run-stream/check":
                raise AssertionError(f"Unexpected URL {request.full_url}")
            return _FakeHttpResponse(
                {
                    "success": True,
                    "resolution": "unresolved",
                    "ingestUrl": "https://sync.example/api/model-run-stream/ingest",
                    "run": {
                        "modelId": 42,
                        "runId": "dry-run-2",
                        "streamAttachToken": None,
                        "datasourceId": None,
                        "eulerTrainDir": None,
                        "runDir": None,
                    },
                },
            )

        monkeypatch.setattr(stream_mod, "urlopen", fake_urlopen)

        exit_code = stream_check_cli_main(
            [
                "--api-url",
                "https://sync.example",
                "--api-key",
                "user-token",
                "--run-id",
                "dry-run-2",
                "--stream-attach-token",
                "attach-123",
                "--json",
            ],
        )

        assert exit_code == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["success"] is True
        assert payload["resolution"] == "unresolved"
        assert payload["run"]["runId"] == "dry-run-2"


# ═══════════════════════════════════════════════════════════════════════════
#  lifecycle  (finish, context manager, crash)
# ═══════════════════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_finish_writes_completed(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.finish()

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "completed"
        assert meta["end_time"] is not None
        assert meta["end_iso"] is not None
        assert meta["duration_sec"] >= 0

    def test_double_finish_is_noop(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.finish()
        first_meta = _read_json(run.dir / "meta.json")

        run.finish()  # second call should be harmless
        second_meta = _read_json(run.dir / "meta.json")
        assert first_meta == second_meta

    def test_context_manager_clean_exit(self, tmp_path):
        with euler_train.init(dir=str(tmp_path / "r"), config={}) as run:
            run.log({"loss": 1.0}, step=0, epoch=0)

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "completed"

    def test_context_manager_crash_records_error(self, tmp_path):
        with pytest.raises(RuntimeError):
            with euler_train.init(dir=str(tmp_path / "r"), config={}) as run:
                raise RuntimeError("training exploded")

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "crashed"
        assert "RuntimeError: training exploded" in meta["error"]
        assert "traceback" in meta
        assert "training exploded" in meta["traceback"]

    def test_mode_scopes_lifecycle_and_crash_details(self, tmp_path):
        train_run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            mode="train",
        )
        train_run.finish()

        eval_run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            run_id=train_run.run_id,
            mode="eval",
        )
        eval_run._original_excepthook = lambda *args: None

        try:
            raise ValueError("eval exploded")
        except ValueError:
            eval_run._on_exception(*sys.exc_info())

        meta = _read_json(eval_run.dir / "meta.json")
        assert meta["status"] == "crashed"
        assert meta["modes"]["train"]["status"] == "completed"
        assert meta["modes"]["eval"]["status"] == "crashed"
        assert "ValueError: eval exploded" in meta["modes"]["eval"]["error"]
        assert "eval exploded" in meta["modes"]["eval"]["traceback"]

    def test_resume_clears_previous_error_fields(self, tmp_path):
        with pytest.raises(RuntimeError):
            with euler_train.init(dir=str(tmp_path / "r"), config={}) as run:
                raise RuntimeError("first failure")

        resumed = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            run_id=run.run_id,
        )
        resumed.finish()

        meta = _read_json(resumed.dir / "meta.json")
        assert meta["status"] == "completed"
        assert "error" not in meta
        assert "traceback" not in meta

    def test_context_manager_does_not_suppress_exception(self, tmp_path):
        with pytest.raises(ValueError, match="boom"):
            with euler_train.init(dir=str(tmp_path / "r"), config={}):
                raise ValueError("boom")

    def test_repr(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        r = repr(run)
        assert "status='running'" in r
        assert run.run_id in r
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  interrupt / exit hooks
# ═══════════════════════════════════════════════════════════════════════════

class TestInterruptHandling:
    def test_atexit_marks_completed(self, tmp_path):
        """Calling _on_exit (the atexit callback) marks status=completed."""
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run._on_exit()

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "completed"
        assert meta["end_time"] is not None

    def test_atexit_noop_if_already_finished(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        run.finish()
        first_meta = _read_json(run.dir / "meta.json")

        run._on_exit()
        second_meta = _read_json(run.dir / "meta.json")
        assert first_meta == second_meta

    def test_signal_marks_interrupted(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        with pytest.raises(SystemExit) as exc_info:
            run._on_signal(signal.SIGTERM, None)

        assert exc_info.value.code == 128 + signal.SIGTERM
        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "interrupted"
        assert "SIGTERM" in meta["error"]
        assert meta["end_time"] is not None

    def test_signal_sigint_records_name(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        with pytest.raises(SystemExit):
            run._on_signal(signal.SIGINT, None)

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "interrupted"
        assert "SIGINT" in meta["error"]
        assert "traceback" not in meta

    def test_excepthook_marks_crashed(self, tmp_path):
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        # Swap original hook with a no-op so it doesn't print during tests
        run._original_excepthook = lambda *args: None

        try:
            raise ValueError("test error")
        except ValueError:
            run._on_exception(*sys.exc_info())

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "crashed"
        assert "ValueError: test error" in meta["error"]
        assert "traceback" in meta
        assert "test error" in meta["traceback"]

    def test_hooks_cleaned_up_after_finish(self, tmp_path):
        original_excepthook = sys.excepthook
        run = euler_train.init(dir=str(tmp_path / "r"), config={})
        # After init, our hook is installed (different from original)
        assert sys.excepthook is not original_excepthook

        run.finish()
        # After finish, original is restored
        assert sys.excepthook is original_excepthook

    def test_signal_inside_context_manager_records_interrupted(self, tmp_path):
        """Signal during with-block should record 'interrupted', not 'crashed'."""
        with pytest.raises(SystemExit):
            with euler_train.init(dir=str(tmp_path / "r"), config={}) as run:
                run._on_signal(signal.SIGINT, None)

        meta = _read_json(run.dir / "meta.json")
        assert meta["status"] == "interrupted"
        assert "SIGINT" in meta["error"]


# ═══════════════════════════════════════════════════════════════════════════
#  Metric naming
# ═══════════════════════════════════════════════════════════════════════════

_SAMPLE_METRIC_NAMING = {
    "producer_key": "euler_train.weather_metric",
    "producer_version": "0.1.0",
    "namespaces": {
        "depth.train": {
            "axes": {
                "kind": {"position": 0, "optional": False, "values": ["loss", "diag", "stat"]},
                "stage": {"position": 1, "optional": True, "values": ["prior", "final"]},
            }
        },
        "sys.train": {
            "axes": {}
        },
    },
}


class TestMetricNaming:
    def test_metric_naming_stored_in_meta_json(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={"lr": 1e-3},
            metric_naming=_SAMPLE_METRIC_NAMING,
        )
        meta = _read_json(run.dir / "meta.json")
        assert meta["metric_naming"] == _SAMPLE_METRIC_NAMING
        run.finish()

    def test_metric_naming_none_by_default(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
        )
        meta = _read_json(run.dir / "meta.json")
        assert "metric_naming" not in meta
        run.finish()

    def test_metric_naming_in_stream_init_event(self, tmp_path):
        consumer = _RecordingStreamConsumer()
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            stream=consumer,
            metric_naming=_SAMPLE_METRIC_NAMING,
        )
        run.finish()

        init_event = consumer.events[0]
        assert init_event["type"] == "init"
        assert init_event["meta"]["metric_naming"] == _SAMPLE_METRIC_NAMING

    def test_metric_naming_not_in_stream_when_absent(self, tmp_path):
        consumer = _RecordingStreamConsumer()
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            stream=consumer,
        )
        run.finish()

        init_event = consumer.events[0]
        assert "metric_naming" not in init_event["meta"]

    def test_metric_naming_preserved_on_resume(self, tmp_path):
        run1 = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            metric_naming=_SAMPLE_METRIC_NAMING,
        )
        run1.finish()

        run2 = euler_train.init(
            dir=str(run1.dir),
        )
        meta = _read_json(run2.dir / "meta.json")
        assert meta["metric_naming"] == _SAMPLE_METRIC_NAMING
        run2.finish()

    def test_metric_naming_validation_not_dict(self, tmp_path):
        with pytest.raises(TypeError, match="metric_naming must be a dict"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                metric_naming="invalid",
            )

    def test_metric_naming_validation_missing_namespaces(self, tmp_path):
        with pytest.raises(ValueError, match="namespaces"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                metric_naming={"producer_key": "test"},
            )

    def test_metric_naming_validation_namespaces_not_dict(self, tmp_path):
        with pytest.raises(ValueError, match="namespaces"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                metric_naming={"namespaces": "invalid"},
            )

    def test_metric_naming_validation_namespace_entry_not_dict(self, tmp_path):
        with pytest.raises(ValueError, match="must be a dict"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                metric_naming={"namespaces": {"depth.train": "invalid"}},
            )

    def test_gpu_stats_namespaced_when_metric_naming_present(self, tmp_path):
        """GPU stats should use sys.train.* keys when metric_naming is set."""
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            metric_naming=_SAMPLE_METRIC_NAMING,
        )
        # Force GPU stats to be "available" with a mock
        run._gpu_available = True

        class _FakeUtil:
            gpu = 75
            memory = 50

        class _FakeMem:
            used = 4_000_000_000
            total = 8_000_000_000

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"pynvml": mock.MagicMock()}) as _:
            import sys as _sys
            pynvml_mock = _sys.modules["pynvml"]
            pynvml_mock.nvmlDeviceGetUtilizationRates.return_value = _FakeUtil()
            pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = _FakeMem()

            stats = run._get_gpu_stats()

        assert "sys.train.gpu_util_pct" in stats
        assert "sys.train.gpu_mem_used_gb" in stats
        assert "gpu_util_pct" not in stats
        run.finish()

    def test_gpu_stats_flat_when_no_metric_naming(self, tmp_path):
        """GPU stats should use flat keys when metric_naming is absent."""
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
        )
        run._gpu_available = True

        class _FakeUtil:
            gpu = 75
            memory = 50

        class _FakeMem:
            used = 4_000_000_000
            total = 8_000_000_000

        import unittest.mock as mock
        with mock.patch.dict("sys.modules", {"pynvml": mock.MagicMock()}) as _:
            import sys as _sys
            pynvml_mock = _sys.modules["pynvml"]
            pynvml_mock.nvmlDeviceGetUtilizationRates.return_value = _FakeUtil()
            pynvml_mock.nvmlDeviceGetMemoryInfo.return_value = _FakeMem()

            stats = run._get_gpu_stats()

        assert "gpu_util_pct" in stats
        assert "sys.train.gpu_util_pct" not in stats
        run.finish()


class TestPipeline:
    def test_pipeline_stored_in_meta_json(self, tmp_path):
        pipeline = {"attach_id": "pipe-abc-123"}
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={"lr": 1e-3},
            pipeline=pipeline,
        )
        meta = _read_json(run.dir / "meta.json")
        assert meta["pipeline"] == pipeline
        run.finish()

    def test_pipeline_none_by_default(self, tmp_path):
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
        )
        meta = _read_json(run.dir / "meta.json")
        assert "pipeline" not in meta
        run.finish()

    def test_pipeline_with_extra_fields(self, tmp_path):
        pipeline = {
            "attach_id": "pipe-xyz",
            "stage": "training",
            "pipeline_id": "nightly-run-42",
        }
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            pipeline=pipeline,
        )
        meta = _read_json(run.dir / "meta.json")
        assert meta["pipeline"]["attach_id"] == "pipe-xyz"
        assert meta["pipeline"]["stage"] == "training"
        assert meta["pipeline"]["pipeline_id"] == "nightly-run-42"
        run.finish()

    def test_pipeline_preserved_on_resume(self, tmp_path):
        pipeline = {"attach_id": "pipe-resume-test"}
        run1 = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            pipeline=pipeline,
        )
        run1.finish()

        run2 = euler_train.init(
            dir=str(run1.dir),
        )
        meta = _read_json(run2.dir / "meta.json")
        assert meta["pipeline"] == pipeline
        run2.finish()

    def test_pipeline_in_stream_init_event(self, tmp_path):
        consumer = _RecordingStreamConsumer()
        pipeline = {"attach_id": "stream-pipe-id"}
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            stream=consumer,
            pipeline=pipeline,
        )
        run.finish()

        init_event = consumer.events[0]
        assert init_event["type"] == "init"
        assert init_event["meta"]["pipeline"] == pipeline

    def test_pipeline_not_in_stream_when_absent(self, tmp_path):
        consumer = _RecordingStreamConsumer()
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            stream=consumer,
        )
        run.finish()

        init_event = consumer.events[0]
        assert "pipeline" not in init_event["meta"]

    def test_pipeline_validation_not_dict(self, tmp_path):
        with pytest.raises(TypeError, match="pipeline must be a dict"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                pipeline="invalid",
            )

    def test_pipeline_validation_missing_attach_id(self, tmp_path):
        with pytest.raises(ValueError, match="attach_id"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                pipeline={},
            )

    def test_pipeline_validation_empty_attach_id(self, tmp_path):
        with pytest.raises(ValueError, match="attach_id"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                pipeline={"attach_id": "  "},
            )

    def test_pipeline_validation_non_string_attach_id(self, tmp_path):
        with pytest.raises(ValueError, match="attach_id"):
            euler_train.init(
                dir=str(tmp_path / "r"),
                config={},
                pipeline={"attach_id": 42},
            )

    def test_pipeline_from_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EULER_SESSION_ID", "env-session-99")
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
        )
        meta = _read_json(run.dir / "meta.json")
        assert meta["pipeline"] == {"attach_id": "env-session-99"}
        run.finish()

    def test_pipeline_env_var_ignored_when_explicit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EULER_SESSION_ID", "env-session-99")
        explicit = {"attach_id": "explicit-id", "stage": "eval"}
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
            pipeline=explicit,
        )
        meta = _read_json(run.dir / "meta.json")
        assert meta["pipeline"] == explicit
        run.finish()

    def test_pipeline_env_var_empty_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("EULER_SESSION_ID", "  ")
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
        )
        meta = _read_json(run.dir / "meta.json")
        assert "pipeline" not in meta
        run.finish()

    def test_pipeline_env_var_unset_ignored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("EULER_SESSION_ID", raising=False)
        run = euler_train.init(
            dir=str(tmp_path / "r"),
            config={},
        )
        meta = _read_json(run.dir / "meta.json")
        assert "pipeline" not in meta
        run.finish()
