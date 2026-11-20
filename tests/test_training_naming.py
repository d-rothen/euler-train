"""Training naming 2.0, validation sets and checkpoint selection (euler-metric-naming docs/training.md)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import euler_train
import euler_train.stream as stream_mod
from euler_train import NamingError

naming = pytest.importorskip("euler_metric_naming")
contract = pytest.importorskip("euler_metric_naming.contract")
if not hasattr(contract, "load_validation_pass"):
    pytest.skip("needs euler-metric-naming 0.4", allow_module_level=True)

from euler_metric_naming import MetricDescription, MetricNamespace
from euler_metric_naming.contract import (
    ProtocolBuilder,
    compare_files,
    load_validation_pass,
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _naming_v2() -> dict:
    return MetricNamespace(
        producer="euler_train.test",
        producer_version="1.0.0",
        modalities=("depth",),
        stages=("prior", "final"),
        metrics={
            "total": MetricDescription(is_higher_better=False, display_name="Total loss"),
            "depth_mae": {"kind": "score", "direction": "lower", "unit": "m"},
        },
    ).training_naming_config()


def _naming_v1() -> dict:
    return MetricNamespace(
        producer="euler_train.test", producer_version="1.0.0", modalities=("depth",),
    ).training_naming_config()


def _set_document(key: str = "cerema") -> dict:
    """A small validation set over two items, metric space, image mean AbsRel."""
    axes = {
        "space": {"position": 0, "optional": False, "values": ["native", "metric"]},
        "category": {"position": 1, "optional": True, "values": ["standard"]},
        "reduction": {"position": 2, "optional": True, "values": ["image_mean"]},
    }
    builder = ProtocolBuilder("depth.dense", ["euler.core", "euler.depth"])
    builder.reference("gt", identity="sha256:" + "0" * 64, modality="depth")
    builder.param("depth.cap.max_m", 80)
    builder.describe("spatial.method")
    builder.population(["scene~f0", "scene~f1"], unit="image")
    builder.axis("space", "space").axis("category", "family").axis("reduction", "reduction")
    builder.metric("absrel")
    builder.reducer("image_mean")
    return {
        "kind": "euler.validation_set",
        "schemaVersion": "1.0",
        "key": key,
        "label": "two frames",
        "metricSet": {"metricNamespace": "depth.eval", "axes": axes},
        "protocol": builder.build(),
    }


def _pass(step: int, value: float = 0.2) -> dict:
    return {
        "step": step,
        "epoch": step // 10,
        "metrics": {"depth.eval.metric.standard.image_mean.absrel": value},
        "parameters": {"spatial.method": "none"},
        "per_item": {"depth.eval.metric.standard.absrel": [value - 0.05, value + 0.05]},
    }


# -- naming checks --------------------------------------------------------------------------------


def test_a_conforming_run_has_an_empty_report(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=_naming_v2())
    run.log({"depth.train.loss.total": 0.5, "depth.train.diag.final.depth_mae": 0.3,
             "sys.train.lr": 1e-4}, step=0, epoch=0)
    run.log({"depth.train.diag.final.depth_mae": 0.25}, step=1, epoch=0, mode="val")
    run.finish()
    report = _read_json(run.dir / "meta.json")["naming_report"]
    assert report["schema_version"] == "2.0"
    assert report["checked"] is True
    assert report["validator"] == f"euler-metric-naming {naming.__version__}"
    assert report["problems"] == []
    assert report["truncated"] == 0


def test_problems_are_reported_once_and_counted(tmp_path, caplog):
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=_naming_v2())
    record = {
        "depth.train.loss.total": 0.5,
        "val/sky/mask_iou": 0.9,
        "depth.train.total": 1.0,
        "depth.train.diag.image_psnr": 30.0,
        "depth.eval.metric.standard.image_mean.absrel": 0.1,
        "depth.train.loss.final.total": True,
    }
    with caplog.at_level("WARNING", logger="euler_train"):
        for step in range(3):
            run.log(record, step=step, epoch=0, mode="val")
    run.finish()
    problems = {
        (p["key"], p["code"]): p for p in _read_json(run.dir / "meta.json")["naming_report"]["problems"]
    }
    assert set(problems) == {
        ("val/sky/mask_iou", "no_namespace"),
        ("depth.train.total", "no_decomposition"),
        ("depth.train.diag.image_psnr", "no_definition"),
        ("depth.eval.metric.standard.image_mean.absrel", "evaluation_name"),
        ("depth.train.loss.final.total", "not_a_number"),
    }
    for entry in problems.values():
        assert entry["source"] == "val.jsonl"
        assert entry["count"] == 3
        assert entry["first_step"] == 0
    assert sum("Naming:" in message for message in caplog.messages) == 5
    # nothing is dropped
    rows = _read_jsonl(run.dir / "val.jsonl")
    assert len(rows) == 3 and rows[0]["val/sky/mask_iou"] == 0.9


def test_strict_naming_raises_on_the_first_problem(tmp_path):
    run = euler_train.init(
        dir=str(tmp_path), config={}, metric_naming=_naming_v2(), strict_naming=True
    )
    run.log({"depth.train.loss.total": 0.5}, step=0, epoch=0)
    with pytest.raises(NamingError, match="no_namespace"):
        run.log({"loss": 0.5}, step=1, epoch=0)
    with pytest.raises(NamingError, match="training naming 2.0"):
        euler_train.init(dir=str(tmp_path / "v1"), config={}, metric_naming=_naming_v1(),
                         strict_naming=True)


def test_schema_1_and_undeclared_runs_are_not_checked(tmp_path):
    for name, declaration in (("v1", _naming_v1()), ("none", None)):
        run = euler_train.init(dir=str(tmp_path / name), config={}, metric_naming=declaration)
        run.log({"loss": 0.5, "val/sky/mask_iou": 0.9}, step=0, epoch=0)
        run.finish()
        meta = _read_json(run.dir / "meta.json")
        assert "naming_report" not in meta
        row = _read_jsonl(run.dir / "train.jsonl")[0]
        assert set(row) == {"step", "epoch", "wall_time", "elapsed_sec", "loss", "val/sky/mask_iou"}


def test_declaration_problems_are_reported(tmp_path):
    declaration = _naming_v2()
    declaration["namespaces"]["depth.train"]["metrics"]["absrel"] = {
        "id": "depth.absrel", "direction": "lower",
    }
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=declaration)
    run.finish()
    problems = _read_json(run.dir / "meta.json")["naming_report"]["problems"]
    assert problems == [{
        "source": "metric_naming", "key": "namespaces.depth.train.metrics.absrel",
        "code": "evaluation_id", "count": 1,
    }]


def test_a_resumed_run_keeps_counting(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=_naming_v2())
    run.log({"loss": 0.5}, step=0, epoch=0)
    run.finish()
    resumed = euler_train.init(dir=str(run.dir))
    resumed.log({"loss": 0.4}, step=1, epoch=0)
    resumed.finish()
    problems = _read_json(run.dir / "meta.json")["naming_report"]["problems"]
    assert problems == [{"source": "train.jsonl", "key": "loss", "code": "no_namespace",
                         "count": 2, "first_step": 0}]


# -- validation sets --------------------------------------------------------------------------------


def test_a_validation_set_is_declared_once_and_logged_per_pass(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=_naming_v2())
    document = _set_document()
    path = run.declare_validation_set("cerema", document)
    assert path == run.dir / "validation" / "cerema" / "set.json"
    assert _read_json(path) == document
    run.declare_validation_set("cerema", document)  # same document: no-op
    run.log_validation("cerema", _pass(10))
    run.log_validation("cerema", _pass(20, 0.18), epoch=3)
    run.finish()

    meta = _read_json(run.dir / "meta.json")
    assert meta["validation_sets"] == {"cerema": {
        "schema_version": "1.0",
        "set": "validation/cerema/set.json",
        "passes": "validation/cerema/passes.jsonl",
        "metric_namespace": "depth.eval",
        "label": "two frames",
    }}
    assert meta["naming_report"]["problems"] == []
    assert not (run.dir / "val.jsonl").exists()
    passes = _read_jsonl(run.dir / "validation" / "cerema" / "passes.jsonl")
    assert [record["step"] for record in passes] == [10, 20]
    assert passes[1]["epoch"] == 3
    assert all("wall_time" in record for record in passes)

    first = load_validation_pass(document, passes[0])
    second = load_validation_pass(document, passes[1])
    verdict = compare_files(first, second, "depth.eval.metric.standard.image_mean.absrel")
    assert verdict.label == "same protocol"


def test_a_set_cannot_change_its_document(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={})
    run.declare_validation_set("cerema", _set_document())
    changed = _set_document()
    changed["protocol"]["parameters"]["depth.cap.max_m"] = 40
    with pytest.raises(ValueError, match="different document"):
        run.declare_validation_set("cerema", changed)
    with pytest.raises(ValueError, match="key must match"):
        run.declare_validation_set("Bad Key", _set_document())
    with pytest.raises(ValueError, match="kind"):
        run.declare_validation_set("other", {"key": "other"})
    with pytest.raises(ValueError, match="declares key"):
        run.declare_validation_set("other", _set_document("cerema"))
    with pytest.raises(KeyError, match="not declared"):
        run.log_validation("muses", _pass(1))


def test_a_resumed_run_logs_to_its_declared_sets(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={})
    run.declare_validation_set("cerema", _set_document())
    run.log_validation("cerema", _pass(10))
    run.finish()
    resumed = euler_train.init(dir=str(run.dir))
    resumed.log_validation("cerema", step=20, metrics={"depth.eval.metric.standard.image_mean.absrel": 0.1})
    resumed.finish()
    passes = _read_jsonl(run.dir / "validation" / "cerema" / "passes.jsonl")
    assert [record["step"] for record in passes] == [10, 20]


def test_pass_problems_are_reported(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={})
    run.declare_validation_set("cerema", _set_document())
    record = _pass(10)
    record["metrics"]["depth.eval.metric.standard.image_mean.sqrel"] = 0.3
    record["parameters"]["lab.knob"] = 1
    run.log_validation("cerema", record)
    run.finish()
    problems = {
        (p["key"], p["code"]) for p in _read_json(run.dir / "meta.json")["naming_report"]["problems"]
    }
    assert problems == {
        ("metrics.depth.eval.metric.standard.image_mean.sqrel", "no_definition"),
        ("parameters.lab.knob", "undescribed"),
    }


# -- checkpoint selection -----------------------------------------------------------------------------


def test_checkpoint_selection_is_recorded(tmp_path):
    run = euler_train.init(dir=str(tmp_path), config={})
    run.log_saved_checkpoint(
        "/tmp/model.pt", epoch=5, step=7554, is_best=True,
        selection_metric="sparsedepth.eval.metric.standard.image_mean.absrel",
        selection_value=0.431, selection_direction="lower", validation_set="muses",
    )
    run.log_saved_checkpoint("/tmp/other.pt", epoch=6, step=8813)
    with pytest.raises(ValueError, match="selection_direction"):
        run.log_saved_checkpoint("/tmp/x.pt", epoch=7, step=1, selection_direction="min")
    run.finish()
    checkpoints = _read_json(run.dir / "meta.json")["checkpoints"]
    assert checkpoints[0] == {
        "path": "/tmp/model.pt", "epoch": 5, "step": 7554, "is_best": True,
        "selection_metric": "sparsedepth.eval.metric.standard.image_mean.absrel",
        "selection_value": 0.431, "selection_direction": "lower", "validation_set": "muses",
    }
    assert checkpoints[1] == {"path": "/tmp/other.pt", "epoch": 6, "step": 8813}


def test_save_checkpoint_takes_a_selection(tmp_path):
    torch = pytest.importorskip("torch")
    run = euler_train.init(dir=str(tmp_path), config={})
    model = torch.nn.Linear(2, 2)
    run.save_checkpoint(
        model, epoch=1, step=10,
        selection={"metric": "depth.eval.metric.standard.image_mean.absrel", "value": 0.2,
                   "direction": "lower", "validation_set": "cerema", "is_best": True},
    )
    with pytest.raises(ValueError, match="unknown selection fields"):
        run.save_checkpoint(model, epoch=2, step=20, selection={"metrik": "x"})
    run.finish()
    entry = _read_json(run.dir / "meta.json")["checkpoints"][0]
    assert entry["is_best"] is True
    assert entry["selection_direction"] == "lower"
    assert entry["validation_set"] == "cerema"


# -- streaming ---------------------------------------------------------------------------------


class _FakeHttpResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _streamed_types(tmp_path, monkeypatch, capabilities):
    calls: list[dict] = []

    def fake_urlopen(request, timeout=0):
        body = json.loads(request.data.decode("utf-8"))
        calls.append({"url": request.full_url, "body": body})
        if request.full_url.endswith("/api/model-run-stream/session"):
            session = {
                "token": "stream-token",
                "expiresAt": "2099-01-01T00:00:00+00:00",
                "ingestUrl": "https://sync.example/api/model-run-stream/ingest",
            }
            if capabilities is not None:
                session["capabilities"] = capabilities
            return _FakeHttpResponse(session)
        return _FakeHttpResponse({"success": True, "events": []})

    monkeypatch.setattr(stream_mod, "urlopen", fake_urlopen)
    run = euler_train.init(
        dir=str(tmp_path), config={},
        stream={"base_url": "https://sync.example", "model_id": 1, "api_token": "t",
                "batch_size": 1},
    )
    run.declare_validation_set("cerema", _set_document())
    run.log_validation("cerema", _pass(10))
    run.log({"loss": 0.1}, step=10, epoch=1)
    run.finish()
    return [
        event["type"]
        for call in calls
        if call["url"].endswith("/ingest")
        for event in call["body"]["events"]
    ]


def test_validation_events_are_streamed_only_when_supported(tmp_path, monkeypatch):
    without = _streamed_types(tmp_path / "a", monkeypatch, None)
    assert "validation_set" not in without and "validation" not in without
    assert without == ["init", "metric", "finish"]
    supported = _streamed_types(tmp_path / "b", monkeypatch, ["validation_sets/1"])
    assert supported == ["init", "validation_set", "validation", "metric", "finish"]
    # the files are written either way
    assert (tmp_path / "a").exists()


def test_meta_json_matches_the_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((Path(__file__).resolve().parents[1] / "meta-schema.json").read_text())
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=_naming_v2())
    run.log({"loss": 0.5}, step=0, epoch=0)
    run.declare_validation_set("cerema", _set_document())
    run.log_validation("cerema", _pass(10))
    run.log_saved_checkpoint(
        "/tmp/model.pt", epoch=1, step=10, is_best=True,
        selection_metric="depth.eval.metric.standard.image_mean.absrel",
        selection_value=0.2, selection_direction="lower", validation_set="cerema",
    )
    run.finish()
    meta = _read_json(run.dir / "meta.json")
    assert meta["naming_report"]["problems"]
    jsonschema.validate(meta, schema)


def test_strict_naming_fails_before_creating_the_run(tmp_path):
    with pytest.raises(NamingError, match="training naming 2.0"):
        euler_train.init(dir=str(tmp_path / "p"), config={}, strict_naming=True)
    assert not (tmp_path / "p").exists()


def test_without_a_validator_the_report_says_so(tmp_path, monkeypatch):
    import euler_train.naming as naming_mod

    monkeypatch.setattr(naming_mod, "_validator", lambda: (None, "euler-metric-naming is not installed"))
    run = euler_train.init(dir=str(tmp_path), config={}, metric_naming=_naming_v2())
    run.log({"loss": 0.5}, step=0, epoch=0)
    run.finish()
    report = _read_json(run.dir / "meta.json")["naming_report"]
    assert report["checked"] is False
    assert report["message"] == "euler-metric-naming is not installed"
    assert report["problems"] == []
    with pytest.raises(NamingError, match="not installed"):
        euler_train.init(dir=str(tmp_path / "s"), config={}, metric_naming=_naming_v2(),
                         strict_naming=True)
