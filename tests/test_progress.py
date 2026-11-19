"""euler-view progress of training, through euler-connector: epoch e of E, the steps within it."""

import pytest
from euler_connector import Connector, set_current
from euler_connector.transports.memory import MemoryTransport

import euler_train


@pytest.fixture
def records():
    """The records an euler-view launch would send, kept in memory."""
    memory = MemoryTransport()
    previous = set_current(Connector.from_environment({}, extra_transports=[memory]))
    yield memory
    set_current(previous)


def _sent(records):
    return [envelope.data for envelope in records.by_channel()["progress"]]


def _train(run, *, epochs, steps_per_epoch):
    for epoch in range(epochs):
        for step in range(epoch * steps_per_epoch, (epoch + 1) * steps_per_epoch):
            run.log({"loss": 0.5}, step=step, epoch=epoch)


def test_training_reports_its_epoch_and_the_steps_within_it(tmp_path, records):
    run = euler_train.init(dir=str(tmp_path), config={"training": {"num_epochs": 3}})
    run.set_progress(steps_per_epoch=4)
    _train(run, epochs=2, steps_per_epoch=4)
    run.finish()

    first, last = _sent(records)[0], _sent(records)[-1]
    assert (first["stage"], first["scope"], first["unit"]) == ("train", "epoch", "step")
    assert (first["epoch"]["index"], first["epoch"]["total"]) == (1, 3)  # from the config
    assert (first["current"], first["total"], first["state"]) == (1, 4, "running")
    assert (last["epoch"]["index"], last["current"], last["total"]) == (2, 4, 4)
    assert last["state"] == "done"


def test_without_set_progress_the_epochs_come_from_the_config(tmp_path, records):
    run = euler_train.init(dir=str(tmp_path), config={"trainer": {"max_epochs": 5}})
    run.log({"loss": 0.5}, step=0, epoch=0)
    run.log({"loss": 0.4}, step=0, epoch=0, mode="val")  # validation records do not count
    run.finish()

    record = _sent(records)[-1]
    assert (record["epoch"]["index"], record["epoch"]["total"]) == (1, 5)
    assert (record["current"], record["total"]) == (1, None)  # steps per epoch unknown


def test_a_failed_run_does_not_end_done(tmp_path, records):
    run = euler_train.init(dir=str(tmp_path), config={"epochs": 2})
    _train(run, epochs=1, steps_per_epoch=3)
    run.finish(status="failed")

    assert _sent(records)[-1]["state"] == "running"  # its last record: how far it got


def test_outside_a_launch_training_runs_as_before(tmp_path):
    previous = set_current(Connector.from_environment({}))  # no EULER_* variables
    try:
        run = euler_train.init(dir=str(tmp_path), config={"epochs": 1})
        _train(run, epochs=1, steps_per_epoch=2)
        run.finish()
    finally:
        set_current(previous)
