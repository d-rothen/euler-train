"""Trainer-declared modality roles (``euler_train.init(modality_roles=...)``).

Datasets prepared by euler-known-datasets keep experiment roles out of their
heads, so ``describe_for_runlog()`` reports no ``used_as`` for e.g. ``depth``.
The trainer declares the roles instead; datasets whose heads do carry roles
(the older Synscapes / VKITTI2 / Real Drive Sim archives) must log exactly
what they logged before.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import euler_train
import euler_train.run as run_module

_REAL_DESCRIPTOR_READER = run_module._read_ds_crawler_descriptor

# What mari's trainer knows about its sample keys.
MARI_ROLES = {
    "hazy": "input",
    "clear": "target",
    "depth": "target",
    "sky_mask": "target",
    "intrinsics": "condition",
}


class _Described:
    """Duck-typed stand-in for euler-loading / mari ``describe_for_runlog()``."""

    def __init__(self, description: dict) -> None:
        self._description = description

    def describe_for_runlog(self) -> dict:
        return json.loads(json.dumps(self._description))  # fresh copy per call


def _read_meta(run) -> dict:
    return json.loads((run.dir / "meta.json").read_text())


# What MultiModalDataset + mari's MixedDataset report for an UrbanSyn source:
# euler-loading resolved hazy/clear/intrinsics from their names, the head has
# no roles, so depth and sky_mask come without used_as/slot.
URBANSYN_NO_HEAD_ROLES = {
    "modalities": {
        "urbansyn.hazy": {
            "path": "/data/URBANSYN_euler_fog", "metadata_scope": "rgb",
            "used_as": "input", "slot": "input.rgb", "modality_type": "rgb",
        },
        "urbansyn.clear": {
            "path": "/data/URBANSYN_euler", "metadata_scope": "rgb",
            "used_as": "target", "slot": "target.rgb", "modality_type": "rgb",
        },
        "urbansyn.depth": {
            "path": "/data/URBANSYN_euler", "metadata_scope": "depth",
            "modality_type": "depth",
        },
        "urbansyn.sky_mask": {
            "path": "/data/URBANSYN_euler", "metadata_scope": "semantic_segmentation",
            "modality_type": "sky_mask",
        },
    },
    "hierarchical_modalities": {
        "urbansyn.intrinsics": {
            "path": "/data/URBANSYN_euler", "metadata_scope": "intrinsics",
            "used_as": "condition", "slot": "condition.intrinsics",
            "modality_type": "intrinsics", "hierarchy_scope": "level_2",
            "applies_to": ["hazy", "clear", "depth", "sky_mask"],
        },
    },
}

# What the same path reports for Real Drive Sim: every head carries
# addons.euler_train.used_as = "target" (even for the hazy input) and most
# carry a slot. Copied from a real meta.json (run 2026-11-13_02-33-46_2ec1).
RDS_HEAD_ROLES = {
    "modalities": {
        "rds.hazy": {
            "path": "/data/rds/cvpr/two", "metadata_scope": "rgb",
            "used_as": "target", "slot": "target.rgb", "modality_type": "rgb",
        },
        "rds.clear": {
            "path": "/data/rds/rgb.zip", "metadata_scope": "rgb",
            "used_as": "target", "slot": "target.rgb", "modality_type": "rgb",
        },
        "rds.depth": {
            "path": "/data/rds/depth.zip", "metadata_scope": "depth",
            "used_as": "target", "slot": "real_drive_sim.target.depth",
            "modality_type": "depth",
        },
        "rds.sky_mask": {
            "path": "/data/rds/semantic_segmentation_2d.zip",
            "metadata_scope": "sky_mask", "used_as": "target",
            "slot": "real_drive_sim.target.semantic_segmentation",
            "modality_type": "sky_mask",
        },
    },
    "hierarchical_modalities": {
        "rds.intrinsics": {
            "path": "/data/rds/calibration.zip", "metadata_scope": "intrinsics",
            "used_as": "target", "slot": "real_drive_sim.target.intrinsics",
            "modality_type": "intrinsics", "hierarchy_scope": "scene",
            "applies_to": ["hazy", "clear", "depth", "sky_mask"],
        },
    },
}


@pytest.fixture(autouse=True)
def _no_ds_crawler(monkeypatch):
    # The fixtures above are the whole truth; never read real metadata.
    monkeypatch.setattr(run_module, "_read_ds_crawler_descriptor", lambda path: {})


# ── new dataset, no roles in the head ────────────────────────────────────


def test_new_dataset_without_head_roles_lists_every_gap_once(tmp_path):
    hypersim = {
        "modalities": {
            "hypersim.depth": {
                "path": "/data/HYPERSIM_euler", "metadata_scope": "depth",
                "modality_type": "depth",
            },
        },
        "hierarchical_modalities": {},
    }
    with pytest.raises(run_module.MissingModalityFieldsError) as excinfo:
        euler_train.init(
            dir=str(tmp_path / "proj"), config={},
            datasets={
                "train": _Described(URBANSYN_NO_HEAD_ROLES),
                "val": _Described(hypersim),
            },
        )

    message = str(excinfo.value)
    # every gap of every split in one error, not the first one only
    assert [(m.split, m.name, m.fields) for m in excinfo.value.missing] == [
        ("train", "urbansyn.depth", ("used_as",)),
        ("train", "urbansyn.sky_mask", ("used_as",)),
        ("val", "hypersim.depth", ("used_as",)),
    ]
    assert "Modality 'urbansyn.depth' (path: /data/URBANSYN_euler, scope: depth)" in message
    assert "missing required field 'used_as'" in message
    # names the exact key to set, with the leaf names the trainer uses
    assert 'modality_roles={"depth": "<role>", "sky_mask": "<role>"}' in message
    # still a ValueError for callers that catch that
    assert isinstance(excinfo.value, ValueError)
    # nothing was written: the run directory is created only after validation
    assert not (tmp_path / "proj").exists()


def test_new_dataset_without_head_roles_logs_with_trainer_roles(tmp_path):
    run = euler_train.init(
        dir=str(tmp_path / "proj"), config={},
        datasets={"train": _Described(URBANSYN_NO_HEAD_ROLES)},
        modality_roles=MARI_ROLES,
    )
    train = _read_meta(run)["datasets"]["train"]
    run.finish()

    assert train["modalities"]["urbansyn.depth"] == {
        "path": "/data/URBANSYN_euler", "metadata_scope": "depth",
        "used_as": "target", "slot": "target.depth", "modality_type": "depth",
    }
    assert train["modalities"]["urbansyn.sky_mask"]["used_as"] == "target"
    assert train["modalities"]["urbansyn.sky_mask"]["slot"] == "target.sky_mask"
    # entries that were complete are untouched
    expected = URBANSYN_NO_HEAD_ROLES
    for name in ("urbansyn.hazy", "urbansyn.clear"):
        assert train["modalities"][name] == expected["modalities"][name]
    assert train["hierarchical_modalities"] == expected["hierarchical_modalities"]


def test_full_name_key_wins_over_leaf_key_and_mapping_sets_slot(tmp_path):
    run = euler_train.init(
        dir=str(tmp_path / "proj"), config={},
        datasets={"train": _Described(URBANSYN_NO_HEAD_ROLES)},
        modality_roles={
            "depth": "target",
            "sky_mask": "target",
            "urbansyn.depth": {"used_as": "target", "task": "depth"},
        },
    )
    modalities = _read_meta(run)["datasets"]["train"]["modalities"]
    run.finish()
    assert modalities["urbansyn.depth"]["slot"] == "depth.target.depth"
    assert modalities["urbansyn.sky_mask"]["slot"] == "target.sky_mask"


def test_head_role_from_ds_crawler_beats_trainer_role(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_module, "_read_ds_crawler_descriptor",
        lambda path: {"properties": {"euler_train": {"used_as": "target"}}},
    )
    ds = _Described({
        "modalities": {"depth": {"path": "/d", "modality_type": "depth"}},
        "hierarchical_modalities": {},
    })
    run = euler_train.init(
        dir=str(tmp_path / "proj"), config={}, datasets={"train": ds},
        modality_roles={"depth": "input"},
    )
    entry = _read_meta(run)["datasets"]["train"]["modalities"]["depth"]
    run.finish()
    assert entry["used_as"] == "target"


# ── older datasets, roles in the head: unchanged ─────────────────────────


@pytest.mark.parametrize("roles", [None, MARI_ROLES])
def test_old_dataset_with_head_roles_is_logged_unchanged(tmp_path, roles):
    """RDS/VKITTI2/Synscapes: identical meta with or without trainer roles.

    Their heads say ``used_as: target`` even for the hazy input; trainer roles
    fill gaps only, so ``rds.hazy`` stays ``target`` as before.
    """
    run = euler_train.init(
        dir=str(tmp_path / "proj"), config={},
        datasets={"train": _Described(RDS_HEAD_ROLES)},
        modality_roles=roles,
    )
    train = _read_meta(run)["datasets"]["train"]
    run.finish()
    assert train == RDS_HEAD_ROLES


def test_mixture_of_old_and_new_sources(tmp_path):
    mixture = {
        "modalities": {
            **RDS_HEAD_ROLES["modalities"],
            **URBANSYN_NO_HEAD_ROLES["modalities"],
        },
        "hierarchical_modalities": {
            **RDS_HEAD_ROLES["hierarchical_modalities"],
            **URBANSYN_NO_HEAD_ROLES["hierarchical_modalities"],
        },
    }
    run = euler_train.init(
        dir=str(tmp_path / "proj"), config={},
        datasets={"train": _Described(mixture)},
        modality_roles=MARI_ROLES,
    )
    train = _read_meta(run)["datasets"]["train"]
    run.finish()
    for name, entry in RDS_HEAD_ROLES["modalities"].items():
        assert train["modalities"][name] == entry
    assert train["modalities"]["urbansyn.depth"]["used_as"] == "target"


# ── other entry points and validation ────────────────────────────────────


def test_fallback_path_without_describe_for_runlog_uses_roles(tmp_path, monkeypatch):
    import warnings

    monkeypatch.setattr(
        run_module, "_read_ds_crawler_descriptor", lambda path: {"modality_type": "depth"},
    )

    class _Plain:
        def modality_paths(self):
            return {"mystery": "/datasets/mystery"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # explicit role: no "inferred" warning
        run = euler_train.init(
            dir=str(tmp_path / "proj"), config={}, datasets={"train": _Plain()},
            modality_roles={"mystery": "target"},
        )
    entry = _read_meta(run)["datasets"]["train"]["modalities"]["mystery"]
    run.finish()
    assert entry == {
        "path": "/datasets/mystery", "used_as": "target",
        "slot": "target.depth", "modality_type": "depth",
    }


def test_add_evaluation_uses_the_runs_roles(tmp_path):
    run = euler_train.init(
        dir=str(tmp_path / "proj"), config={}, modality_roles=MARI_ROLES,
    )
    run.add_evaluation("urbansyn", datasets={"test": _Described(URBANSYN_NO_HEAD_ROLES)})
    entry = _read_meta(run)["evaluations"]["urbansyn"]["datasets"]["test"]
    run.finish()
    assert entry["modalities"]["urbansyn.depth"]["used_as"] == "target"


@pytest.mark.parametrize(
    "roles, match",
    [
        ({"depth": "gt"}, r"used_as must be one of"),
        ({"depth": {"used_as": "target", "slot": "depth"}}, r"slot must match"),
        ({"depth": {"used_as": "target", "colour": "red"}}, r"Unknown .* key"),
        ({"intrinsics": {"used_as": "target", "hierarchy_scope": "scene"}},
         r"only allowed when used_as is 'condition'"),
    ],
)
def test_invalid_roles_are_rejected_before_the_run_exists(tmp_path, roles, match):
    with pytest.raises(ValueError, match=match):
        euler_train.init(dir=str(tmp_path / "proj"), config={}, modality_roles=roles)
    assert not (tmp_path / "proj").exists()


# ── real archives (skipped where /cluster/work is not mounted) ────────────

DATA = Path("/cluster/work/igp_psr/drothenpiele/data")


@pytest.mark.skipif(not (DATA / "URBANSYN_euler").is_dir(), reason="needs Euler data")
def test_real_urbansyn_and_rds_depth(tmp_path, monkeypatch):
    """Real heads: RDS depth.zip carries roles, URBANSYN_euler/depth does not."""
    from euler_loading import Modality, MultiModalDataset

    # undo the autouse stub: read the real heads
    monkeypatch.setattr(run_module, "_read_ds_crawler_descriptor", _REAL_DESCRIPTOR_READER)
    rds = MultiModalDataset(modalities={
        "depth": Modality(str(DATA / "rds/depth.zip"), modality_type="depth"),
    })
    urbansyn = MultiModalDataset(modalities={
        "depth": Modality(str(DATA / "URBANSYN_euler"), metadata_scope="depth",
                          modality_type="depth"),
    })
    rds_before = run_module._build_datasets_meta({"train": rds}, None)

    with pytest.raises(run_module.MissingModalityFieldsError, match="urbansyn|URBANSYN"):
        run_module._build_datasets_meta({"rds": rds, "urbansyn": urbansyn}, None)

    meta = run_module._build_datasets_meta(
        {"train": rds, "urbansyn": urbansyn}, None, modality_roles=MARI_ROLES,
    )
    assert meta["train"] == rds_before["train"]  # old archive unchanged
    assert meta["train"]["modalities"]["depth"]["slot"] == "real_drive_sim.target.depth"
    assert meta["urbansyn"]["modalities"]["depth"]["used_as"] == "target"
    assert meta["urbansyn"]["modalities"]["depth"]["slot"] == "target.depth"
