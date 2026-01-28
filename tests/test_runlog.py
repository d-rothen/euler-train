"""Tests for the runlog package — one section per public endpoint."""
from __future__ import annotations

import dataclasses
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

import runlog

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


# ═══════════════════════════════════════════════════════════════════════════
#  init  /  config normalisation
# ═══════════════════════════════════════════════════════════════════════════

class TestInit:
    def test_creates_directory_and_files(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "run1"), config={"lr": 1e-3})
        run.finish()

        assert (tmp_path / "run1" / "meta.json").exists()
        assert (tmp_path / "run1" / "config.json").exists()

    def test_config_from_dict(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={"lr": 0.01, "bs": 32})
        cfg = _read_json(tmp_path / "r" / "config.json")
        assert cfg == {"lr": 0.01, "bs": 32}
        run.finish()

    def test_config_none_writes_empty(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"))
        cfg = _read_json(tmp_path / "r" / "config.json")
        assert cfg == {}
        run.finish()

    def test_config_from_json_file(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"arch": "resnet", "depth": 50}))

        run = runlog.init(dir=str(tmp_path / "r"), config=str(cfg_file))
        cfg = _read_json(tmp_path / "r" / "config.json")
        assert cfg["arch"] == "resnet"
        assert cfg["depth"] == 50
        run.finish()

    def test_config_from_namespace(self, tmp_path):
        ns = Namespace(lr=1e-4, epochs=10)
        run = runlog.init(dir=str(tmp_path / "r"), config=ns)
        cfg = _read_json(tmp_path / "r" / "config.json")
        assert cfg == {"lr": 1e-4, "epochs": 10}
        run.finish()

    def test_config_from_dataclass(self, tmp_path):
        @dataclasses.dataclass
        class Cfg:
            lr: float = 1e-3
            arch: str = "unet"

        run = runlog.init(dir=str(tmp_path / "r"), config=Cfg())
        cfg = _read_json(tmp_path / "r" / "config.json")
        assert cfg == {"lr": 1e-3, "arch": "unet"}
        run.finish()

    def test_config_unsupported_type_raises(self, tmp_path):
        with pytest.raises(TypeError, match="Unsupported config type"):
            runlog.init(dir=str(tmp_path / "r"), config=42)

    def test_meta_initial_status_is_running(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        meta = _read_json(tmp_path / "r" / "meta.json")
        assert meta["status"] == "running"
        assert meta["start_time"] is not None
        assert meta["start_iso"] is not None
        assert meta["end_time"] is None
        run.finish()

    def test_meta_custom_fields_merged(self, tmp_path):
        run = runlog.init(
            dir=str(tmp_path / "r"), config={},
            meta={"description": "ablation A", "tags": ["v2", "fast"]},
        )
        meta = _read_json(tmp_path / "r" / "meta.json")
        assert meta["description"] == "ablation A"
        assert meta["tags"] == ["v2", "fast"]
        assert meta["status"] == "running"  # built-in still present
        run.finish()

    def test_returns_run_object(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        assert isinstance(run, runlog.Run)
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  log  (train.jsonl  /  val.jsonl)
# ═══════════════════════════════════════════════════════════════════════════

class TestLog:
    def test_train_log_creates_file_and_appends(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.log({"loss": 2.3, "lr": 1e-4}, step=0, epoch=0)
        run.log({"loss": 2.1, "lr": 1e-4}, step=1, epoch=0)

        records = _read_jsonl(tmp_path / "r" / "train.jsonl")
        assert len(records) == 2
        assert records[0]["step"] == 0
        assert records[1]["loss"] == 2.1
        run.finish()

    def test_train_log_has_required_fields(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.log({"loss": 1.0}, step=5, epoch=1)

        rec = _read_jsonl(tmp_path / "r" / "train.jsonl")[0]
        assert rec["step"] == 5
        assert rec["epoch"] == 1
        assert "wall_time" in rec
        assert "elapsed_sec" in rec  # train-only field
        assert rec["loss"] == 1.0
        run.finish()

    def test_val_log_writes_to_val_file(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.log({"rgb.psnr": 25.0, "rgb.ssim": 0.91}, step=100, epoch=1, mode="val")

        assert not (tmp_path / "r" / "train.jsonl").exists()
        records = _read_jsonl(tmp_path / "r" / "val.jsonl")
        assert len(records) == 1
        assert records[0]["rgb.psnr"] == 25.0
        run.finish()

    def test_val_log_no_elapsed_sec(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.log({"metric": 1.0}, step=0, epoch=0, mode="val")

        rec = _read_jsonl(tmp_path / "r" / "val.jsonl")[0]
        assert "elapsed_sec" not in rec
        run.finish()

    def test_log_dynamic_keys(self, tmp_path):
        """Caller can pass arbitrary metric names."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.log({"custom_metric_xyz": 42, "another.nested.key": 0.5}, step=0, epoch=0)

        rec = _read_jsonl(tmp_path / "r" / "train.jsonl")[0]
        assert rec["custom_metric_xyz"] == 42
        assert rec["another.nested.key"] == 0.5
        run.finish()

    def test_log_numpy_scalars_serialised(self, tmp_path):
        """numpy dtypes should be serialised to plain Python types."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.log({"loss": np.float32(1.5), "n": np.int64(10)}, step=0, epoch=0)

        rec = _read_jsonl(tmp_path / "r" / "train.jsonl")[0]
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
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred")
        assert f.suffix == ".png"
        img = Image.open(f)
        assert img.size == (32, 32)
        assert img.mode == "RGB"
        run.finish()

    def test_uint8_hwc4_saved_as_rgba_png(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (16, 16, 4), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgba=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgba" / "pred")
        assert f.suffix == ".png"
        assert Image.open(f).mode == "RGBA"
        run.finish()

    def test_uint8_hwc1_saved_as_grayscale_png(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (16, 16, 1), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, gray=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "gray" / "pred")
        assert f.suffix == ".png"
        assert Image.open(f).mode == "L"
        run.finish()

    def test_uint8_hw_saved_as_grayscale_png(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (16, 16), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, mask=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "mask" / "pred")
        assert f.suffix == ".png"
        run.finish()

    def test_float_hwc3_saved_as_png_scaled(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.rand(16, 16, 3).astype(np.float32)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred")
        assert f.suffix == ".png"
        img = np.array(Image.open(f))
        # float [0,1] → uint8 [0,255]
        assert img.dtype == np.uint8
        assert img.max() > 0
        run.finish()

    def test_float_hwc3_clips_oob_values(self, tmp_path):
        """Values outside [0,1] should be clipped, not wrap around."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.full((8, 8, 3), 1.5, dtype=np.float32)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred")
        img = np.array(Image.open(f))
        assert (img == 255).all()
        run.finish()

    # ---- non-image arrays → .npy -----------------------------------------

    def test_float32_hw_saved_as_npy(self, tmp_path):
        """A float HxW array (e.g. depth map) should default to .npy."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        depth = np.random.rand(32, 32).astype(np.float32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=depth))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "depth" / "pred")
        assert f.suffix == ".npy"
        loaded = np.load(f)
        np.testing.assert_array_equal(loaded, depth)
        run.finish()

    def test_float64_hw_saved_as_npy(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.rand(16, 16)  # float64 by default
        run.save_outputs(epoch=0, step=0, field=dict(gt=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "field" / "gt")
        assert f.suffix == ".npy"
        run.finish()

    def test_high_channel_count_saved_as_npy(self, tmp_path):
        """HxWx64 is *not* image-like — should be .npy."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.rand(16, 16, 64).astype(np.float32)
        run.save_outputs(epoch=0, step=0, feat=dict(pred=arr))

        f = self._single_file(tmp_path / "r" / "outputs" / "epoch_0_step_0" / "feat" / "pred")
        assert f.suffix == ".npy"
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_outputs  — format overrides
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveOutputsOverrides:
    def test_override_by_output_type(self, tmp_path):
        run = runlog.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"depth": "npz"},
        )
        arr = np.random.rand(16, 16).astype(np.float32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=arr))

        f = list((tmp_path / "r" / "outputs" / "epoch_0_step_0" / "depth" / "pred").iterdir())[0]
        assert f.suffix == ".npz"
        run.finish()

    def test_override_by_leaf_name(self, tmp_path):
        run = runlog.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"pred": "npz"},
        )
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))

        f = list((tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred").iterdir())[0]
        assert f.suffix == ".npz"
        run.finish()

    def test_override_specific_dotted_key(self, tmp_path):
        """'rgb.pred' override should only affect rgb/pred, not depth/pred."""
        run = runlog.init(
            dir=str(tmp_path / "r"), config={},
            output_formats={"rgb.pred": "npy"},
        )
        arr = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        run.save_outputs(
            epoch=0, step=0,
            rgb=dict(pred=arr, gt=arr),
        )

        base = tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb"
        pred_f = list((base / "pred").iterdir())[0]
        gt_f = list((base / "gt").iterdir())[0]
        assert pred_f.suffix == ".npy"   # overridden
        assert gt_f.suffix == ".png"     # still inferred
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_outputs  — batches, lists, aux, PIL, torch
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveOutputsVariants:
    def test_list_of_arrays_saved_individually(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        imgs = [np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8) for _ in range(4)]
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=imgs))

        pred_dir = tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        files = sorted(pred_dir.iterdir())
        assert len(files) == 4
        assert [f.name for f in files] == ["0000.png", "0001.png", "0002.png", "0003.png"]
        run.finish()

    def test_4d_array_split_as_batch(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        batch = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=batch))

        pred_dir = tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        assert len(list(pred_dir.iterdir())) == 3
        run.finish()

    def test_aux_creates_named_subdirectories(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        t_map = np.random.rand(16, 16).astype(np.float32)
        attn = np.random.rand(16, 16).astype(np.float32)
        run.save_outputs(
            epoch=1, step=50,
            depth=dict(aux=dict(transmission=t_map, attention=attn)),
        )

        aux_base = tmp_path / "r" / "outputs" / "epoch_1_step_50" / "depth" / "aux"
        assert (aux_base / "transmission").is_dir()
        assert (aux_base / "attention").is_dir()
        assert list((aux_base / "transmission").iterdir())[0].suffix == ".npy"
        run.finish()

    def test_none_slots_are_skipped(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr, gt=None, input=None))

        rgb_dir = tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb"
        assert (rgb_dir / "pred").is_dir()
        assert not (rgb_dir / "gt").exists()
        assert not (rgb_dir / "input").exists()
        run.finish()

    def test_none_output_type_skipped(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr), depth=None)

        out_dir = tmp_path / "r" / "outputs" / "epoch_0_step_0"
        assert (out_dir / "rgb").is_dir()
        assert not (out_dir / "depth").exists()
        run.finish()

    def test_pil_image_saved_as_png(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        img = Image.fromarray(np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8))
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=img))

        f = list((tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred").iterdir())[0]
        assert f.suffix == ".png"
        run.finish()

    def test_torch_tensor_chw_transposed_and_saved(self, tmp_path):
        """A (C,H,W) torch tensor should be transposed to (H,W,C) and saved."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        t = torch.randint(0, 255, (3, 32, 32), dtype=torch.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=t))

        f = list((tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred").iterdir())[0]
        assert f.suffix == ".png"
        img = np.array(Image.open(f))
        assert img.shape == (32, 32, 3)
        # Verify pixel values match the original tensor (CHW → HWC)
        expected = t.numpy().transpose(1, 2, 0)
        np.testing.assert_array_equal(img, expected)
        run.finish()

    def test_torch_tensor_bchw_becomes_batch(self, tmp_path):
        """A (B,C,H,W) torch tensor should produce B separate files."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        t = torch.randint(0, 255, (5, 3, 16, 16), dtype=torch.uint8)
        run.save_outputs(epoch=0, step=0, rgb=dict(pred=t))

        pred_dir = tmp_path / "r" / "outputs" / "epoch_0_step_0" / "rgb" / "pred"
        assert len(list(pred_dir.iterdir())) == 5
        run.finish()

    def test_torch_float_tensor_saved_as_npy(self, tmp_path):
        """A 2-D float torch tensor (depth-like) should default to .npy."""
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        t = torch.rand(32, 32)
        run.save_outputs(epoch=0, step=0, depth=dict(pred=t))

        f = list((tmp_path / "r" / "outputs" / "epoch_0_step_0" / "depth" / "pred").iterdir())[0]
        assert f.suffix == ".npy"
        loaded = np.load(f)
        np.testing.assert_allclose(loaded, t.numpy(), atol=1e-6)
        run.finish()

    def test_directory_naming_epoch_only(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(epoch=3, rgb=dict(pred=arr))
        assert path.name == "epoch_3"
        run.finish()

    def test_directory_naming_step_only(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(step=999, rgb=dict(pred=arr))
        assert path.name == "step_999"
        run.finish()

    def test_directory_naming_both(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(epoch=2, step=500, rgb=dict(pred=arr))
        assert path.name == "epoch_2_step_500"
        run.finish()

    def test_returns_output_base_path(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        path = run.save_outputs(epoch=0, step=0, rgb=dict(pred=arr))
        assert path == tmp_path / "r" / "outputs" / "epoch_0_step_0"
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  save_checkpoint
# ═══════════════════════════════════════════════════════════════════════════

class TestSaveCheckpoint:
    def test_saves_model_state_dict(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=3)

        assert path == tmp_path / "r" / "checkpoints" / "epoch_3.pt"
        assert path.exists()
        ckpt = torch.load(path, weights_only=False)
        assert "model" in ckpt
        assert "epoch" in ckpt
        assert ckpt["epoch"] == 3
        # Verify weights match
        for k in model.state_dict():
            torch.testing.assert_close(ckpt["model"][k], model.state_dict()[k])
        run.finish()

    def test_saves_raw_dict_as_model(self, tmp_path):
        raw = {"weight": torch.randn(3, 3)}
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(raw, epoch=0)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["model"] is not None
        # raw dict has no state_dict(), so it's stored directly
        torch.testing.assert_close(ckpt["model"]["weight"], raw["weight"])
        run.finish()

    def test_saves_optimizer(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        run = runlog.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=1, optimizer=opt)

        ckpt = torch.load(path, weights_only=False)
        assert "optimizer" in ckpt
        assert "param_groups" in ckpt["optimizer"]
        run.finish()

    def test_saves_extra_kwargs(self, tmp_path):
        model = torch.nn.Linear(4, 2)
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        path = run.save_checkpoint(model, epoch=5, best_loss=0.42, global_step=9999)

        ckpt = torch.load(path, weights_only=False)
        assert ckpt["best_loss"] == 0.42
        assert ckpt["global_step"] == 9999
        run.finish()


# ═══════════════════════════════════════════════════════════════════════════
#  lifecycle  (finish, context manager, crash)
# ═══════════════════════════════════════════════════════════════════════════

class TestLifecycle:
    def test_finish_writes_completed(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.finish()

        meta = _read_json(tmp_path / "r" / "meta.json")
        assert meta["status"] == "completed"
        assert meta["end_time"] is not None
        assert meta["end_iso"] is not None
        assert meta["duration_sec"] >= 0

    def test_double_finish_is_noop(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        run.finish()
        first_meta = _read_json(tmp_path / "r" / "meta.json")

        run.finish()  # second call should be harmless
        second_meta = _read_json(tmp_path / "r" / "meta.json")
        assert first_meta == second_meta

    def test_context_manager_clean_exit(self, tmp_path):
        with runlog.init(dir=str(tmp_path / "r"), config={}) as run:
            run.log({"loss": 1.0}, step=0, epoch=0)

        meta = _read_json(tmp_path / "r" / "meta.json")
        assert meta["status"] == "completed"

    def test_context_manager_crash_records_error(self, tmp_path):
        with pytest.raises(RuntimeError):
            with runlog.init(dir=str(tmp_path / "r"), config={}) as run:
                raise RuntimeError("training exploded")

        meta = _read_json(tmp_path / "r" / "meta.json")
        assert meta["status"] == "crashed"
        assert "RuntimeError: training exploded" in meta["error"]
        assert "traceback" in meta
        assert "training exploded" in meta["traceback"]

    def test_context_manager_does_not_suppress_exception(self, tmp_path):
        with pytest.raises(ValueError, match="boom"):
            with runlog.init(dir=str(tmp_path / "r"), config={}):
                raise ValueError("boom")

    def test_repr(self, tmp_path):
        run = runlog.init(dir=str(tmp_path / "r"), config={})
        r = repr(run)
        assert "status='running'" in r
        run.finish()
