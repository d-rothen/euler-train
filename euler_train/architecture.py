"""Export a PyTorch model to a lightweight ONNX graph for Netron visualization."""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

log = logging.getLogger("euler_train")

_MISSING_DEPS_MSG = (
    "Architecture export requires optional dependencies: onnx, onnxruntime, onnxsim. "
    "Install them with:  pip install euler-train[architecture]"
)


def export_architecture(
    model: Any,
    dummy_input: Any,
    output_path: str | Path = "architecture.onnx",
) -> Path:
    """Export a PyTorch model to a simplified, weightless ONNX graph.

    The resulting file is optimized for visual inspection in Netron:
    redundant nodes are removed, operator fusions are applied, and
    weight tensors are stripped so only the graph topology remains.

    Parameters
    ----------
    model:
        A PyTorch ``nn.Module``.  Temporarily set to eval mode for
        export; the original training/eval state is restored afterward.
    dummy_input:
        Example input tensor(s) matching the model's forward signature.
    output_path:
        Where to write the final ``.onnx`` file.

    Returns
    -------
    Path
        The written output path.
    """
    import torch

    try:
        import onnx
        import onnxruntime as ort
        from onnxsim import simplify
    except ImportError:
        raise ImportError(_MISSING_DEPS_MSG)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    try:
        return _do_export(model, dummy_input, output_path, onnx, ort, simplify, torch)
    finally:
        if was_training:
            model.train()


def _do_export(model, dummy_input, output_path, onnx, ort, simplify, torch):
    with tempfile.TemporaryDirectory() as tmpdir:
        raw_path = Path(tmpdir) / "raw.onnx"
        ort_path = Path(tmpdir) / "ort.onnx"

        # Step 1: Export to ONNX (with weights, needed for optimizer passes)
        _export_onnx(model, dummy_input, raw_path, torch=torch)

        # Step 2: Simplify — removes redundant glue nodes
        log.info("Simplifying ONNX graph with onnxsim...")
        raw_model = onnx.load(str(raw_path))
        simplified_model, check = simplify(raw_model)
        if not check:
            log.warning("onnxsim validation failed, continuing anyway.")
        onnx.save(simplified_model, str(raw_path))

        # Step 3: ORT optimization — fuses standard blocks (Conv+BN+ReLU, etc.)
        log.info("Applying ONNX Runtime graph optimizations...")
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.optimized_model_filepath = str(ort_path)
        ort.InferenceSession(str(raw_path), sess_options)

        # Step 4: Strip weights for a lightweight file
        log.info("Stripping weights...")
        fused_model = onnx.load(str(ort_path))
        while fused_model.graph.initializer:
            fused_model.graph.initializer.pop()

        onnx.save(fused_model, str(output_path))

        log.info("Architecture exported to %s", output_path)
        return output_path


def _export_onnx(model: Any, dummy_input: Any, path: Path, *, torch: Any) -> None:
    """Export using dynamo (PyTorch >= 2.1) or legacy torch.onnx.export."""
    # Try dynamo-based export first (produces a cleaner functional graph)
    if hasattr(torch.onnx, "dynamo_export"):
        try:
            log.info("Exporting ONNX via torch.onnx.dynamo_export (PyTorch 2.x)...")
            export_output = torch.onnx.dynamo_export(model, dummy_input)
            export_output.save(str(path))
            return
        except Exception as exc:
            log.warning(
                "dynamo_export failed (%s), falling back to legacy export.", exc
            )

    # Legacy export (PyTorch 1.x / 2.0 / dynamo fallback)
    log.info("Exporting ONNX via torch.onnx.export (legacy)...")
    torch.onnx.export(
        model,
        dummy_input,
        str(path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
    )
