"""Hardening of the run and its background parts: bounded steps, bounded queues, forks, crashes."""

from __future__ import annotations

import importlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_forwarders import Blocking, Recording, use_forwarders
from fake_frameworks import FakeYoloModel
from support import batch, meta_repro

from cvpilot import api, runs
from cvpilot.core.errors import RunClosedError
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import MAX_STEP, MetricPoint
from cvpilot.hooks._guard import step_value
from cvpilot.logging import clear_request_context
from cvpilot.runs import forwarding

yh = importlib.import_module("cvpilot.hooks." + "ultra" + "lytics")


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


# -- 9: steps and poisoned batches -------------------------------------------------------------


def test_metric_steps_are_bounded_to_the_store_range() -> None:
    kw: dict[str, Any] = {"run_id": new_run_id(), "key": "a", "value": 1.0, "wall_time": 1.0}
    assert MetricPoint(step=MAX_STEP, **kw).step == MAX_STEP
    with pytest.raises(ValueError):
        MetricPoint(step=MAX_STEP + 1, **kw)


def test_log_refuses_a_step_beyond_the_store_range_and_queues_nothing() -> None:
    run = api.init("proj", task="detection")
    with pytest.raises(ValueError, match="step"):
        run.log({"a": 1.0}, step=MAX_STEP + 1)
    with pytest.raises(ValueError, match="epoch"):
        run.log({"a": 1.0}, step=1, epoch=MAX_STEP + 1)
    assert run._handle.buffered_points == 0
    run.log({"a": 1.0}, step=MAX_STEP)  # the largest step is fine
    run.finish()


def test_hook_steps_beyond_the_store_range_are_skipped_not_clamped() -> None:
    assert step_value(MAX_STEP) == MAX_STEP
    assert step_value(MAX_STEP + 1) is None and step_value(float(2**64)) is None


def test_a_batch_that_fails_with_a_data_error_is_dropped_not_requeued(tmp_path: Path) -> None:
    run = api.init("proj", task="detection")
    handle = run._handle
    poison = MetricPoint.model_construct(
        run_id=handle.run_id, key="a", step=MAX_STEP + 1, epoch=None, value=1.0, wall_time=1.0
    )
    handle._buffer.append(poison)
    with pytest.raises((ValueError, OverflowError, TypeError)):
        handle.flush()
    assert handle.buffered_points == 0  # the buffer cannot grow forever on a batch that never fits
    run.log({"b": 2.0}, step=1)
    run.finish()
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.stats.n_points == 1


def test_a_transient_store_error_still_keeps_the_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cvpilot.core.errors import StoreError

    run = api.init("proj", task="detection")
    handle = run._handle
    run.log({"a": 1.0}, step=1)
    with monkeypatch.context() as patch:

        def fail(points: Any) -> None:
            raise StoreError("disk")

        patch.setattr(handle._store, "append_points", fail)
        with pytest.raises(StoreError):
            handle.flush()
        assert handle.buffered_points == 1
    run.finish()  # the points were kept and are written now


# -- 10: queue by points, ranks, forks, crashes ---------------------------------------------------


def test_the_forwarder_queue_is_bounded_by_points_not_only_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert forwarding.MAX_QUEUED_POINTS == 200_000
    adapter = Blocking()
    meta, repro = meta_repro()
    fan = forwarding.Fanout(
        [adapter], meta, repro, queue_batches=1000, max_queued_points=10, close_timeout_s=0.2
    )
    fan.start()
    for step in range(30):
        fan.submit(batch(meta, step, 3))
    worker = fan._workers[0]
    assert worker.pending_points <= 10
    stats = fan.close("finished")
    assert stats["blocking"].dropped_batches >= 26 and stats["blocking"].dropped_points >= 78
    adapter.release.set()


def test_pending_points_fall_when_the_adapter_catches_up() -> None:
    adapter = Recording()
    meta, repro = meta_repro()
    fan = forwarding.Fanout([adapter], meta, repro, max_queued_points=10)
    fan.start()
    for step in range(50):
        fan.submit(batch(meta, step, 3))
        deadline = 0
        while fan._workers[0].pending_points > 6 and deadline < 200:
            threading.Event().wait(0.01)
            deadline += 1
    stats = fan.close("finished")
    assert stats["recording"].dropped_batches == 0 and len(adapter.logged) == 150


@pytest.mark.parametrize(("variable", "value"), [("RANK", "1"), ("LOCAL_RANK", "3"), ("RANK", "7")])
def test_non_zero_ranks_start_no_forwarders_and_no_threads(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, capsys: pytest.CaptureFixture[str]
) -> None:
    use_forwarders(monkeypatch, recording=Recording)
    monkeypatch.setenv(variable, value)
    before = {t.name for t in threading.enumerate()}
    run = api.init("proj", task="detection", forward_to=["recording"])
    run.log({"a": 1.0}, step=0)
    assert Recording.instances == []
    assert not [t for t in threading.enumerate() if t.name not in before and "forward" in t.name]
    run.finish()
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    skipped = [e for e in events if e["message"] == "forward_skipped_rank"]
    assert len(skipped) == 1 and skipped[0]["forwarders"] == ["recording"]


def test_rank_zero_and_unset_or_odd_rank_variables_keep_the_forwarders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for env_value in (None, "0", "junk", ""):
        use_forwarders(monkeypatch, recording=Recording)
        if env_value is None:
            monkeypatch.delenv("RANK", raising=False)
        else:
            monkeypatch.setenv("RANK", env_value)
        monkeypatch.setenv("LOCAL_RANK", "0")
        run = api.init("proj", task="detection", forward_to=["recording"])
        run.finish()
        assert len(Recording.instances) == 1, env_value


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_a_forked_child_neither_forwards_nor_touches_the_inherited_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_forwarders(monkeypatch, recording=Recording)
    run = api.init("proj", task="detection", forward_to=["recording"])
    run.log({"parent": 1.0}, step=0)
    run._handle.flush()
    read, write = os.pipe()
    pid = os.fork()
    if pid == 0:  # child: report what it observed through the pipe, never run test machinery
        try:
            observed = {"active": api._active is None}
            try:
                run.log({"child": 1.0}, step=1)
                observed["log"] = "accepted"
            except RunClosedError as exc:
                observed["log"] = f"refused: {exc}"
            try:
                api.log({"child": 2.0}, step=2)
                observed["api"] = "accepted"
            except Exception as exc:  # noqa: BLE001
                observed["api"] = type(exc).__name__
            observed["calls"] = len(Recording.instances[0].calls)
            os.write(write, json.dumps(observed).encode())
        finally:
            os._exit(0)
    os.close(write)
    seen = json.loads(os.read(read, 65536).decode())
    os.waitpid(pid, 0)
    assert (
        seen["active"] is True
        and seen["log"].startswith("refused")
        and "parent process" in seen["log"]
    )
    assert seen["api"] == "NoActiveRunError"
    calls_in_parent_before = len(Recording.instances[0].calls)
    assert seen["calls"] <= calls_in_parent_before
    run.log({"parent": 2.0}, step=1)  # the parent is unaffected
    run.finish()
    adapter = Recording.instances[0]
    assert {k for k, _, _ in adapter.logged} == {"parent"}
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.stats.n_points == 2


def test_a_crashed_yolo_train_marks_the_run_failed_and_reraises(tmp_path: Path) -> None:
    import importlib.metadata as md

    original_version = md.version
    md.version = lambda dist: "8.3.0"  # type: ignore[assignment]
    try:
        model = FakeYoloModel(fail_with=RuntimeError("CUDA out of memory"))
        run = api.autolog(model, project="proj", task="detection")
    finally:
        md.version = original_version
    with pytest.raises(RuntimeError, match="CUDA"):
        model.train(data="x.yaml")
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "failed" and view.stats.n_points > 0
    assert model.train_calls == [((), {"data": "x.yaml"})]  # arguments pass through untouched


def test_an_interrupted_yolo_train_is_marked_failed_too_and_a_clean_one_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "8.3.0")
    model = FakeYoloModel(fail_with=KeyboardInterrupt())
    run = api.autolog(model, project="proj", task="detection")
    with pytest.raises(KeyboardInterrupt):
        model.train()
    assert (
        runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store").meta.status == "failed"
    )
    ok_model = FakeYoloModel()
    ok = api.autolog(ok_model, project="proj", task="detection")
    assert ok_model.train() == "metrics"
    assert (
        runs.get_run_view("proj", ok.run_id, store_dir=tmp_path / "store").meta.status == "finished"
    )


def test_detach_restores_the_original_train() -> None:
    model = FakeYoloModel()
    hook = yh.YoloHook()
    from_class = type(model).train
    hook.attach(
        model,
        type(
            "S", (), {"status": "running", "log": lambda *a, **k: None, "finish": lambda *a: None}
        )(),
    )
    assert "train" in vars(model)
    hook.detach()
    assert "train" not in vars(model) and type(model).train is from_class
