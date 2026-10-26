from __future__ import annotations

import io
import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
import structlog

from cvpilot import runs
from cvpilot.core.errors import (
    DuplicateMetricError,
    InvalidResumeError,
    InvalidTransitionError,
    MetricOrderError,
    RunClosedError,
    RunNotFoundError,
    StoreError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, Repro, RunMeta
from cvpilot.logging import clear_request_context, configure_logging
from cvpilot.settings import Settings

SHA = "c" * 64


@pytest.fixture
def logs() -> Any:
    buf = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="dev", log_level="DEBUG"), stream=buf)
    yield lambda: [json.loads(line) for line in buf.getvalue().splitlines() if line.startswith("{")]
    clear_request_context()
    structlog.reset_defaults()


def make(project: str = "p", **over: Any) -> tuple[RunMeta, Repro]:
    rid = over.pop("run_id", None) or new_run_id()
    base: dict[str, Any] = {
        "run_id": rid,
        "project": project,
        "name": "n",
        "status": "running",
        "task": "detection",
        "framework": "python==3.12.1",
        "host": "h",
        "started_at": datetime.now(timezone.utc),
        "schema_version": SCHEMA_VERSION,
    }
    base.update(over)
    repro = Repro(
        run_id=rid,
        git_dirty=False,
        packages_lock_sha256=SHA,
        python_version="3.12",
        gpu_names=[],
        dataset_uri="",
        dataset_hash="merkle:" + SHA,
        config={},
        command_line="t.py",
    )
    return RunMeta(**base), repro


def start(tmp_path: Path, **kw: Any) -> runs.RunHandle:
    meta, repro = make(**{k: kw.pop(k) for k in ("status", "parent_run_id") if k in kw})
    return runs.start_run(meta, repro, store_dir=tmp_path, **kw)


def points_of(tmp_path: Path, run_id: str) -> list[Any]:
    from cvpilot.registry import stores

    store = stores.get("local")("p", tmp_path)
    try:
        return store.read_points(run_id)  # type: ignore[no-any-return]
    finally:
        store.close()


# --- start and events -------------------------------------------------------


def test_start_run_persists_the_run_and_logs_run_started(tmp_path: Path, logs: Any) -> None:
    h = start(tmp_path)
    assert h.status == "running"
    stored = runs.list_runs("p", store_dir=tmp_path)
    assert [r.run_id for r in stored] == [h.run_id]
    (event,) = [e for e in logs() if e["message"] == "run_started"]
    assert event["run_id"] == h.run_id and event["project"] == "p" and event["status"] == "running"
    assert event["service"] == "cvpilot-client" and event["level"] == "INFO"
    assert event["request_id"].startswith(f"{h.run_id}:")
    h.finish()


def test_dirty_diff_is_stored_as_an_artifact(tmp_path: Path) -> None:
    import hashlib

    diff = b"diff --git a b\n"
    meta, _ = make()
    repro = make(run_id=meta.run_id)[1].model_copy(
        update={
            "git_commit": "a" * 40,
            "git_dirty": True,
            "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        }
    )
    h = runs.start_run(meta, repro, store_dir=tmp_path, git_diff=diff)
    view = runs.get_run_view("p", h.run_id, store_dir=tmp_path)
    assert [a.kind for a in view.artifacts] == ["diff"]
    assert view.artifacts[0].sha256 == repro.git_diff_sha256
    h.finish()


def test_start_run_rejects_a_mismatched_project_pair(tmp_path: Path) -> None:
    meta, repro = make()
    other = make(run_id=new_run_id())[1]
    with pytest.raises(StoreError):
        runs.start_run(meta, other, store_dir=tmp_path)
    assert repro  # silence unused


# --- log, flush thresholds ----------------------------------------------------


def test_points_are_buffered_until_the_point_bound(tmp_path: Path, logs: Any) -> None:
    h = start(tmp_path, flush_points=4, flush_interval_s=3600)
    h.log({"a": 1.0, "b": 2.0}, step=0)
    assert points_of(tmp_path, h.run_id) == []  # buffered, no file yet
    h.log({"a": 1.0, "b": 2.0}, step=1)  # reaches 4 points: flush on the caller's thread
    assert len(points_of(tmp_path, h.run_id)) == 4
    flushed = [e for e in logs() if e["message"] == "metrics_flushed"]
    assert len(flushed) == 1
    assert (flushed[0]["n_points"], flushed[0]["file"], flushed[0]["level"]) == (
        4,
        "000001.parquet",
        "DEBUG",
    )
    h.finish()


def test_flush_happens_after_the_time_bound(tmp_path: Path) -> None:
    now = [100.0]
    h = start(tmp_path, flush_points=10_000, flush_interval_s=2.0, clock=lambda: now[0])
    h.log({"a": 1.0}, step=0)
    now[0] = 101.9
    h.log({"a": 1.0}, step=1)
    assert points_of(tmp_path, h.run_id) == []
    now[0] = 102.1
    h.log({"a": 1.0}, step=2)
    assert len(points_of(tmp_path, h.run_id)) == 3
    h.finish()


def test_buffer_never_exceeds_the_bound(tmp_path: Path) -> None:
    h = start(tmp_path, flush_points=50, flush_interval_s=3600)
    for step in range(500):
        h.log({"a": 1.0, "b": 2.0, "c": 3.0}, step=step)
        assert h.buffered_points <= 50
    h.finish()
    assert len(points_of(tmp_path, h.run_id)) == 1500


def test_finish_flushes_compacts_and_sets_status(tmp_path: Path, logs: Any) -> None:
    h = start(tmp_path, flush_points=10_000)
    for step in range(5):
        h.log({"loss": 1.0 / (step + 1)}, step=step, epoch=step // 2)
    h.finish()
    meta = runs.get_run_view("p", h.run_id, store_dir=tmp_path)
    assert meta.meta.status == "finished" and meta.meta.ended_at is not None
    assert meta.stats.n_points == 5 and meta.stats.n_files == 1
    assert [p.epoch for p in points_of(tmp_path, h.run_id)] == [0, 0, 1, 1, 2]
    (ev,) = [e for e in logs() if e["message"] == "run_finished"]
    assert (ev["run_id"], ev["status"], ev["n_points"]) == (h.run_id, "finished", 5)
    assert isinstance(ev["duration_ms"], int | float)


@pytest.mark.parametrize("status", ["failed", "preempted"])
def test_finish_with_other_terminal_statuses(tmp_path: Path, status: str) -> None:
    h = start(tmp_path)
    h.finish(status)  # type: ignore[arg-type]
    assert runs.list_runs("p", store_dir=tmp_path)[0].status == status


def test_finish_rejects_non_terminal_status(tmp_path: Path) -> None:
    h = start(tmp_path)
    with pytest.raises(InvalidTransitionError):
        h.finish("running")  # type: ignore[arg-type]
    h.finish()


def test_use_after_finish_is_an_error(tmp_path: Path) -> None:
    h = start(tmp_path)
    h.finish()
    with pytest.raises(RunClosedError):
        h.log({"a": 1.0}, step=0)
    with pytest.raises(RunClosedError):
        h.finish()
    with pytest.raises(RunClosedError):
        h.log_artifact(tmp_path / "x", kind="env")


def test_nonfinite_values_are_logged_not_rejected(tmp_path: Path) -> None:
    h = start(tmp_path)
    h.log({"loss": float("nan"), "g": float("inf")}, step=0)
    h.finish()
    assert len(points_of(tmp_path, h.run_id)) == 2


def test_numpy_style_numbers_are_accepted(tmp_path: Path) -> None:
    class F32:  # stands in for numpy.float32: not a float subclass, has __float__
        def __float__(self) -> float:
            return 0.5

    class I64:
        def __index__(self) -> int:
            return 3

    h = start(tmp_path)
    h.log({"a": F32()}, step=I64())  # type: ignore[dict-item,arg-type]
    h.finish()
    (p,) = points_of(tmp_path, h.run_id)
    assert (p.value, p.step) == (0.5, 3)


@pytest.mark.parametrize("value", ["1.0", True, None, [1.0]])
def test_non_numeric_values_are_rejected_naming_the_key(tmp_path: Path, value: Any) -> None:
    h = start(tmp_path)
    with pytest.raises(ValueError, match="'loss'"):
        h.log({"loss": value}, step=0)
    h.finish()


@pytest.mark.parametrize("step", [-1, 1.5, "3", None])
def test_bad_steps_are_rejected(tmp_path: Path, step: Any) -> None:
    h = start(tmp_path)
    with pytest.raises((ValueError, TypeError)):
        h.log({"a": 1.0}, step=step)
    h.finish()


def test_duplicate_step_rejects_the_whole_call(tmp_path: Path) -> None:
    h = start(tmp_path, flush_interval_s=3600)
    h.log({"a": 1.0}, step=5)
    with pytest.raises(DuplicateMetricError, match="'a'.*5"):
        h.log({"b": 1.0, "a": 2.0}, step=5)  # b must not sneak in
    h.log({"b": 1.0}, step=5)  # b was not recorded by the rejected call
    h.finish()
    assert sorted((p.key, p.step) for p in points_of(tmp_path, h.run_id)) == [("a", 5), ("b", 5)]


def test_backwards_step_is_rejected(tmp_path: Path) -> None:
    h = start(tmp_path)
    h.log({"a": 1.0}, step=5)
    with pytest.raises(MetricOrderError):
        h.log({"a": 1.0}, step=4)
    h.finish()


def test_empty_log_is_a_noop(tmp_path: Path) -> None:
    h = start(tmp_path)
    h.log({}, step=0)
    assert h.buffered_points == 0
    h.finish()


def test_failed_flush_keeps_the_points_for_the_next_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    h = start(tmp_path, flush_points=2, flush_interval_s=3600)
    real = h._store.append_points  # noqa: SLF001
    calls = {"n": 0}

    def flaky(batch: Any) -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise StoreError("disk full")
        return real(batch)  # type: ignore[no-any-return]

    monkeypatch.setattr(h._store, "append_points", flaky)  # noqa: SLF001
    with pytest.raises(StoreError, match="disk full"):
        h.log({"a": 1.0, "b": 1.0}, step=0)
    h.log({"a": 1.0}, step=1)  # retried together with the earlier points
    h.finish()
    assert len(points_of(tmp_path, h.run_id)) == 3


def test_concurrent_logging_from_threads(tmp_path: Path) -> None:
    h = start(tmp_path, flush_points=100)
    errors: list[BaseException] = []

    def work(key: str) -> None:
        try:
            for step in range(300):
                h.log({key: float(step)}, step=step)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(f"k{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    h.finish()
    assert errors == []
    got = points_of(tmp_path, h.run_id)
    assert len(got) == 2400 and len({(p.key, p.step) for p in got}) == 2400


# --- crash, recovery, resume ------------------------------------------------------


def test_unfinished_run_stays_running_and_can_be_marked_failed(tmp_path: Path, logs: Any) -> None:
    h = start(tmp_path, flush_points=2)
    h.log({"a": 1.0, "b": 2.0}, step=0)  # flushed; then the process "dies" without finish
    assert runs.list_runs("p", store_dir=tmp_path)[0].status == "running"

    runs.recover_run(
        "p", h.run_id, store_dir=tmp_path, force=True
    )  # no lease: the run never took one
    view = runs.get_run_view("p", h.run_id, store_dir=tmp_path)
    assert view.meta.status == "failed" and view.stats.n_points == 2
    assert any(e["message"] == "run_finished" and e["status"] == "failed" for e in logs())
    with pytest.raises(InvalidTransitionError):
        runs.recover_run("p", h.run_id, store_dir=tmp_path, force=True)


def test_the_run_functions_honour_the_store_directory_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``init`` and ``find_resume_point`` read CVPILOT_STORE_DIR; so must the functions a restart
    recipe calls next (``recover_run``, ``list_runs``, ``get_run_view``)."""
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path))
    h = start(tmp_path)
    assert [m.run_id for m in runs.list_runs("p")] == [h.run_id]
    runs.recover_run("p", h.run_id, force=True)
    assert runs.get_run_view("p", h.run_id).meta.status == "failed"


def test_recover_unknown_run(tmp_path: Path) -> None:
    with pytest.raises(RunNotFoundError):
        runs.recover_run("p", new_run_id(), store_dir=tmp_path)


def test_resume_from_builds_a_child_of_a_preempted_or_failed_run(tmp_path: Path) -> None:
    for status in ("preempted", "failed"):
        h = start(tmp_path)
        h.finish(status)  # type: ignore[arg-type]
        parent = runs.get_run_view("p", h.run_id, store_dir=tmp_path).meta
        rid = new_run_id()
        child = runs.resume_from(parent, run_id=rid, host="h2", name=None)
        assert (child.status, child.parent_run_id, child.run_id) == ("resumed", parent.run_id, rid)
        assert (child.project, child.task, child.framework, child.host) == (
            parent.project,
            parent.task,
            parent.framework,
            "h2",
        )
        assert child.started_at >= parent.started_at


@pytest.mark.parametrize("status", ["running", "resumed", "finished"])
def test_resume_from_refuses_other_parents(status: str) -> None:
    extra: dict[str, Any] = {"status": status}
    if status == "resumed":
        extra["parent_run_id"] = new_run_id()
    if status == "finished":
        extra["ended_at"] = datetime.now(timezone.utc) + timedelta(seconds=5)
    parent, _ = make(**extra)
    with pytest.raises(InvalidResumeError):
        runs.resume_from(parent, run_id=new_run_id(), host="h", name=None)


def test_resumed_run_is_started_with_run_resumed_event(tmp_path: Path, logs: Any) -> None:
    h = start(tmp_path)
    h.finish("preempted")
    parent = runs.get_run_view("p", h.run_id, store_dir=tmp_path).meta
    child_meta = runs.resume_from(parent, run_id=new_run_id(), host="h", name="again")
    _, repro = make(run_id=child_meta.run_id)
    child = runs.start_run(child_meta, repro, store_dir=tmp_path)
    events = [e["message"] for e in logs() if e.get("run_id") == child.run_id]
    assert events == ["run_started", "run_resumed"]
    assert runs.get_run_view("p", child.run_id, store_dir=tmp_path).meta.parent_run_id == h.run_id
    child.finish()


# --- queries -------------------------------------------------------------------------


def test_queries_on_a_project_without_a_store(tmp_path: Path) -> None:
    assert runs.list_runs("p", store_dir=tmp_path) == []
    assert not (tmp_path / "p").exists()
    with pytest.raises(RunNotFoundError):
        runs.get_run_view("p", new_run_id(), store_dir=tmp_path)


def test_list_runs_filters_by_status(tmp_path: Path) -> None:
    a, b = start(tmp_path), start(tmp_path)
    a.finish()
    assert [r.run_id for r in runs.list_runs("p", status="finished", store_dir=tmp_path)] == [
        a.run_id
    ]
    assert [r.run_id for r in runs.list_runs("p", status="running", store_dir=tmp_path)] == [
        b.run_id
    ]
    b.finish()


def test_recover_run_fails_closed_when_it_cannot_tell_whether_the_process_is_dead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # "no lease information" means unknown, not dead: marking a live run failed would end its
    # status while it still trains and writes
    monkeypatch.delenv("CVPILOT_CHECKPOINT_DIR", raising=False)
    h = start(tmp_path)
    with pytest.raises(InvalidTransitionError, match="force=True"):
        runs.recover_run("p", h.run_id, store_dir=tmp_path)  # no checkpoint directory known
    with pytest.raises(InvalidTransitionError, match="force=True"):
        runs.recover_run(  # a directory without a lease for this run: the run never took one
            "p", h.run_id, store_dir=tmp_path, checkpoint_dir=tmp_path / "ckpt"
        )
    assert runs.get_run_view("p", h.run_id, store_dir=tmp_path).meta.status == "running"
    runs.recover_run("p", h.run_id, store_dir=tmp_path, force=True)  # the operator vouches for it
    assert runs.get_run_view("p", h.run_id, store_dir=tmp_path).meta.status == "failed"


def test_recover_run_reads_the_checkpoint_directory_from_the_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np

    from cvpilot.ckpt.manager import CheckpointManager

    monkeypatch.setenv("CVPILOT_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    h = start(tmp_path)
    manager = CheckpointManager(h, every_n_steps=1, install_signal_handlers=False)
    manager.save_now(1, model={"w": np.ones(2)})
    with pytest.raises(InvalidTransitionError, match="live process"):
        runs.recover_run("p", h.run_id, store_dir=tmp_path, force=True)  # force never beats a lease
    manager.close()  # the lease is free: the process is provably gone
    runs.recover_run("p", h.run_id, store_dir=tmp_path)
    assert runs.get_run_view("p", h.run_id, store_dir=tmp_path).meta.status == "failed"
