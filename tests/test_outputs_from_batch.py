"""Tests for euler_train.outputs_from_batch — batch-aware save_outputs layer."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import euler_train
from euler_train.outputs_from_batch import (
    build_save_kwargs,
    extract_sample_ids,
    make_named_slot,
    sanitize_sample_id,
    save_outputs_from_batch,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

class _StubDataset:
    """Duck-types ``describe_id_schema()`` without importing euler-loading."""

    def __init__(self, schema: dict | None) -> None:
        self._schema = schema

    def describe_id_schema(self) -> dict | None:
        return self._schema


class _FailingDataset:
    def describe_id_schema(self) -> dict:
        raise RuntimeError("boom")


# ═══════════════════════════════════════════════════════════════════════════
#  sanitize_sample_id
# ═══════════════════════════════════════════════════════════════════════════

class TestSanitizeSampleId:
    def test_safe_chars_passthrough(self):
        assert sanitize_sample_id("abc_123.png-v2") == "abc_123.png-v2"

    def test_unsafe_chars_collapsed_to_underscore(self):
        assert sanitize_sample_id("a/b\\c d") == "a_b_c_d"

    def test_leading_trailing_underscores_stripped(self):
        assert sanitize_sample_id("///foo///") == "foo"

    def test_empty_input_falls_back_to_default(self):
        assert sanitize_sample_id("") == "sample"
        assert sanitize_sample_id("///") == "sample"

    def test_non_string_converted_via_str(self):
        assert sanitize_sample_id(42) == "42"


# ═══════════════════════════════════════════════════════════════════════════
#  extract_sample_ids
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractSampleIds:
    def test_prefers_full_id_over_id(self):
        batch = {"full_id": ["/scene/a", "/scene/b"], "id": ["a", "b"]}
        assert extract_sample_ids(batch) == ["scene_a", "scene_b"]

    def test_falls_back_to_id_when_full_id_missing(self):
        batch = {"id": ["alpha", "beta"]}
        assert extract_sample_ids(batch) == ["alpha", "beta"]

    def test_returns_none_when_no_ids_present(self):
        assert extract_sample_ids({"foo": "bar"}) is None

    def test_returns_none_when_list_is_empty(self):
        assert extract_sample_ids({"full_id": []}) is None

    def test_returns_none_for_non_list_value(self):
        assert extract_sample_ids({"full_id": "single-string"}) is None

    def test_returns_none_for_mixed_content(self):
        assert extract_sample_ids({"full_id": ["a", 123]}) is None

    def test_sanitize_false_preserves_raw(self):
        batch = {"full_id": ["/scene/a", "/scene/b"]}
        assert extract_sample_ids(batch, sanitize=False) == ["/scene/a", "/scene/b"]

    def test_custom_keys_order(self):
        batch = {"full_id": ["x"], "id": ["y"]}
        assert extract_sample_ids(batch, keys=("id",)) == ["y"]

    def test_bytes_decoded_to_str(self):
        batch = {"full_id": [b"foo/bar", b"baz"]}
        assert extract_sample_ids(batch) == ["foo_bar", "baz"]


# ═══════════════════════════════════════════════════════════════════════════
#  make_named_slot
# ═══════════════════════════════════════════════════════════════════════════

class TestMakeNamedSlot:
    def test_none_tensor_returns_none(self):
        assert make_named_slot(None, ["a", "b"]) is None

    def test_none_ids_returns_tensor_passthrough(self):
        t = torch.zeros(2, 3, 3)
        out = make_named_slot(t, None)
        assert torch.is_tensor(out)
        assert out.shape == (2, 3, 3)

    def test_wraps_to_named_dict(self):
        t = torch.arange(6).view(2, 3)
        out = make_named_slot(t, ["s1", "s2"])
        assert set(out.keys()) == {"s1", "s2"}
        assert torch.equal(out["s1"], torch.tensor([0, 1, 2]))
        assert torch.equal(out["s2"], torch.tensor([3, 4, 5]))

    def test_n_prefix_limits_count(self):
        t = torch.arange(12).view(4, 3)
        out = make_named_slot(t, ["s0", "s1", "s2", "s3"], n=2)
        assert set(out.keys()) == {"s0", "s1"}

    def test_transform_applied_before_wrap(self):
        t = torch.ones(2, 2) * 3.0
        out = make_named_slot(t, ["a", "b"], transform=lambda x: x * 2)
        assert torch.equal(out["a"], torch.tensor([6.0, 6.0]))

    def test_length_mismatch_uses_min_safely(self):
        t = torch.arange(3).view(3, 1)
        out = make_named_slot(t, ["a", "b"])
        assert set(out.keys()) == {"a", "b"}

    def test_cuda_detach_cpu_path_on_plain_tensor(self):
        # detach()/cpu() exist on plain torch tensors even without CUDA;
        # this exercises the detach/cpu coercion path.
        t = torch.arange(4, dtype=torch.float32).view(2, 2).requires_grad_(True)
        out = make_named_slot(t, None)
        assert not out.requires_grad

    def test_numpy_array_accepted(self):
        arr = np.arange(6).reshape(2, 3)
        out = make_named_slot(arr, ["a", "b"])
        assert set(out.keys()) == {"a", "b"}
        np.testing.assert_array_equal(out["a"], np.array([0, 1, 2]))


# ═══════════════════════════════════════════════════════════════════════════
#  build_save_kwargs
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildSaveKwargs:
    def test_tensors_wrapped_into_named_dicts(self):
        slots = {"rgb": {"pred": torch.arange(6).view(2, 3)}}
        out = build_save_kwargs(slots, ["s0", "s1"])
        assert set(out["rgb"]["pred"].keys()) == {"s0", "s1"}

    def test_none_output_type_skipped(self):
        out = build_save_kwargs({"rgb": None, "depth": {"pred": torch.ones(1, 2)}}, ["a"])
        assert "rgb" not in out
        assert "depth" in out

    def test_none_slot_value_preserved_as_none(self):
        slots = {"rgb": {"pred": torch.ones(1, 2), "gt": None}}
        out = build_save_kwargs(slots, ["a"])
        assert out["rgb"]["gt"] is None
        assert isinstance(out["rgb"]["pred"], dict)

    def test_existing_mapping_passed_through(self):
        explicit = {"custom_id": torch.zeros(2, 2)}
        slots = {"rgb": {"pred": explicit}}
        out = build_save_kwargs(slots, ["a"])
        assert out["rgb"]["pred"] == explicit
        # must be a *copy*, not the same object
        assert out["rgb"]["pred"] is not explicit

    def test_aux_sub_dict_recurses(self):
        slots = {
            "depth": {
                "aux": {
                    "transmission": torch.arange(4).view(2, 2),
                    "attention": None,
                }
            }
        }
        out = build_save_kwargs(slots, ["a", "b"])
        aux = out["depth"]["aux"]
        assert isinstance(aux["transmission"], dict)
        assert set(aux["transmission"].keys()) == {"a", "b"}
        assert aux["attention"] is None

    def test_no_sample_ids_leaves_tensors_unwrapped(self):
        slots = {"rgb": {"pred": torch.arange(6).view(2, 3)}}
        out = build_save_kwargs(slots, None)
        assert torch.is_tensor(out["rgb"]["pred"])


# ═══════════════════════════════════════════════════════════════════════════
#  save_outputs_from_batch (free function + Run method)
# ═══════════════════════════════════════════════════════════════════════════

def _make_run(tmp_path: Path):
    return euler_train.init(dir=str(tmp_path / "r"), config={})


def _read_manifest(run_dir: Path, epoch: int, step: int) -> dict:
    return json.loads(
        (run_dir / "outputs" / f"epoch_{epoch}_step_{step}" / "manifest.json").read_text()
    )


class TestSaveOutputsFromBatch:
    def test_free_function_wraps_tensors_into_named_files(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/scene/img_001", "/scene/img_002"]}
        imgs = torch.randint(0, 255, (2, 3, 8, 8), dtype=torch.uint8)

        save_outputs_from_batch(
            run, batch=batch, epoch=1, step=10, rgb={"pred": imgs}
        )

        pred_dir = run.dir / "outputs" / "epoch_1_step_10" / "rgb" / "pred"
        names = sorted(f.name for f in pred_dir.iterdir())
        assert names == ["scene_img_001.png", "scene_img_002.png"]
        run.finish()

    def test_run_method_matches_free_function(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a", "/b"]}
        imgs = torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch, epoch=0, step=0, rgb={"pred": imgs}
        )

        pred_dir = run.dir / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        names = sorted(f.name for f in pred_dir.iterdir())
        assert names == ["a.png", "b.png"]
        run.finish()

    def test_id_schema_merged_from_dataset(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/s/a", "/s/b"]}
        imgs = torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8)
        ds = _StubDataset(
            {"hierarchy_separator": "-", "id_join_char": "+", "full_id_separator": "/"}
        )

        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=1,
            metadata={"dataset": "vkitti2"},
            dataset=ds,
            rgb={"pred": imgs},
        )

        manifest = _read_manifest(run.dir, 1, 1)
        assert manifest["metadata"]["id_schema"]["id_join_char"] == "+"
        assert manifest["metadata"]["id_schema"]["hierarchy_separator"] == "-"
        assert manifest["metadata"]["dataset"] == "vkitti2"
        run.finish()

    def test_explicit_id_schema_overrides_dataset(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)
        ds = _StubDataset({"id_join_char": "+"})

        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=2,
            metadata={"dataset": "vkitti2"},
            dataset=ds,
            id_schema={"id_join_char": "|", "hierarchy_separator": "-"},
            rgb={"pred": imgs},
        )

        manifest = _read_manifest(run.dir, 1, 2)
        assert manifest["metadata"]["id_schema"]["id_join_char"] == "|"
        run.finish()

    def test_existing_metadata_id_schema_wins_over_auto_injection(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)
        ds = _StubDataset({"id_join_char": "+"})

        preexisting = {"id_join_char": "preset"}
        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=3,
            metadata={"dataset": "vkitti2", "id_schema": preexisting},
            dataset=ds,
            rgb={"pred": imgs},
        )

        manifest = _read_manifest(run.dir, 1, 3)
        assert manifest["metadata"]["id_schema"] == preexisting
        run.finish()

    def test_include_id_schema_false_skips_injection(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)
        ds = _StubDataset({"id_join_char": "+"})

        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=4,
            metadata={"dataset": "vkitti2"},
            dataset=ds,
            include_id_schema=False,
            rgb={"pred": imgs},
        )

        manifest = _read_manifest(run.dir, 1, 4)
        assert "id_schema" not in manifest["metadata"]
        run.finish()

    def test_no_dataset_no_schema_no_injection(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=5,
            metadata={"dataset": "vkitti2"},
            rgb={"pred": imgs},
        )

        manifest = _read_manifest(run.dir, 1, 5)
        assert "id_schema" not in manifest["metadata"]
        run.finish()

    def test_failing_describe_id_schema_is_swallowed(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=6,
            metadata={"dataset": "vkitti2"},
            dataset=_FailingDataset(),
            rgb={"pred": imgs},
        )

        manifest = _read_manifest(run.dir, 1, 6)
        assert "id_schema" not in manifest["metadata"]
        run.finish()

    def test_no_metadata_skips_id_schema_injection(self, tmp_path):
        """When metadata is None, id_schema is not auto-injected.

        Injection would otherwise build a metadata dict with only
        ``id_schema`` (no ``dataset`` key), which ``save_outputs``
        rejects. The caller must supply ``metadata={"dataset": ...}``
        to receive schema enrichment.
        """
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch,
            epoch=1,
            step=7,
            dataset=_StubDataset({"id_join_char": "+"}),
            rgb={"pred": imgs},
        )

        pred_dir = run.dir / "outputs" / "epoch_1_step_7" / "rgb" / "pred"
        assert (pred_dir / "a.png").exists()
        manifest = _read_manifest(run.dir, 1, 7)
        assert "metadata" not in manifest
        run.finish()

    def test_no_ids_in_batch_uses_sequential_indices(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {}  # no full_id, no id
        imgs = torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch, epoch=0, step=1, rgb={"pred": imgs}
        )

        pred_dir = run.dir / "outputs" / "epoch_0_step_1" / "rgb" / "pred"
        names = sorted(f.name for f in pred_dir.iterdir())
        assert names == ["0000.png", "0001.png"]
        run.finish()

    def test_explicit_sample_ids_override_batch(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a", "/b"]}
        imgs = torch.randint(0, 255, (2, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch,
            epoch=0,
            step=2,
            sample_ids=["override_0", "override_1"],
            rgb={"pred": imgs},
        )

        pred_dir = run.dir / "outputs" / "epoch_0_step_2" / "rgb" / "pred"
        names = sorted(f.name for f in pred_dir.iterdir())
        assert names == ["override_0.png", "override_1.png"]
        run.finish()

    def test_n_prefix_limits_saved_files(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a", "/b", "/c", "/d"]}
        imgs = torch.randint(0, 255, (4, 3, 4, 4), dtype=torch.uint8)

        run.save_outputs_from_batch(
            batch=batch, epoch=0, step=3, n=2, rgb={"pred": imgs}
        )

        pred_dir = run.dir / "outputs" / "epoch_0_step_3" / "rgb" / "pred"
        names = sorted(f.name for f in pred_dir.iterdir())
        assert names == ["a.png", "b.png"]
        run.finish()

    def test_aux_slot_wraps_to_named_dicts(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a", "/b"]}
        t_map = torch.rand(2, 4, 4)
        depth_pred = torch.rand(2, 4, 4)

        run.save_outputs_from_batch(
            batch=batch,
            epoch=0,
            step=4,
            depth={"pred": depth_pred, "aux": {"transmission": t_map}},
        )

        trans_dir = (
            run.dir / "outputs" / "epoch_0_step_4" / "depth" / "aux" / "transmission"
        )
        names = sorted(f.name for f in trans_dir.iterdir())
        # float tensors default to .npy
        assert names == ["a.npy", "b.npy"]
        run.finish()

    def test_returns_output_directory(self, tmp_path):
        run = _make_run(tmp_path)
        batch = {"full_id": ["/a"]}
        imgs = torch.randint(0, 255, (1, 3, 4, 4), dtype=torch.uint8)

        result = run.save_outputs_from_batch(
            batch=batch, epoch=2, step=5, rgb={"pred": imgs}
        )
        assert result == run.dir / "outputs" / "epoch_2_step_5"
        run.finish()
