"""The preemption path is the one place a failure must never reach the training loop, and the
run must be marked preempted as soon as the local checkpoint is safe: the grace window is short."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
URI = "s3://bucket-name/preempt"


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Exploding:
    """A model whose ``state_dict`` raises something a save does not list."""

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def state_dict(self) -> dict[str, Any]:
        raise self.error


def build(
    tmp: Path, *, remote: FakeRemote | None = None, clock: Clock | None = None, grace: float = 30.0
) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    clk = clock or Clock()
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=5, grace_seconds=grace,
        install_signal_handlers=False, clock=clk, remote=URI if remote else None,
        remote_store=remote, remote_sleep=lambda s: None, remote_clock=clk,
    )  # fmt: skip
    return run, mgr


def status(tmp: Path, run: runs.RunHandle) -> str:
    return runs.get_run_view("p", run.run_id, store_dir=tmp / "store").meta.status


@pytest.mark.parametrize(
    "error", [RuntimeError("boom"), KeyError("k"), AttributeError("a"), ZeroDivisionError()]
)
def test_an_unexpected_error_while_saving_never_reaches_the_loop(
    tmp_path: Path, events: Events, error: Exception
) -> None:
    run, mgr = build(tmp_path)
    mgr.flag.request("SIGTERM")
    assert mgr.maybe_save(5, model=Exploding(error)) is None  # no exception
    logged = {e["message"]: e for e in events()}
    assert logged["preemption_save_incomplete"]["reason"] == type(error).__name__
    assert logged["run_preempted"]["saved"] is False
    assert status(tmp_path, run) == "preempted"
    assert mgr.preempted()


def test_a_normal_save_still_raises_the_same_error_outside_a_preemption(tmp_path: Path) -> None:
    run, mgr = build(tmp_path)
    with pytest.raises(RuntimeError):
        mgr.save_now(1, model=Exploding(RuntimeError("boom")))
    run.finish()


def test_the_run_is_preempted_before_a_slow_upload_starts(tmp_path: Path) -> None:
    clock = Clock()
    seen: list[str] = []
    remote = FakeRemote()
    run, mgr = build(tmp_path, remote=remote, clock=clock)
    real_write = remote.write

    def slow_write(key: str, chunks: Any, *, append: bool = False) -> None:
        seen.append(status(tmp_path, run))  # what the store says while the upload is running
        clock.now += 4.0
        real_write(key, chunks, append=append)

    remote.write = slow_write  # type: ignore[method-assign]
    mgr.flag.request("SIGTERM")
    saved = mgr.maybe_save(3, model={"w": np.ones(2)})
    assert saved is not None and saved.step == 3
    assert seen and set(seen) == {"preempted"}  # marked right after the local save verified


def test_an_upload_that_would_outlast_the_grace_window_is_cut_short(
    tmp_path: Path, events: Events
) -> None:
    clock = Clock()
    remote = FakeRemote()
    run, mgr = build(tmp_path, remote=remote, clock=clock, grace=10.0)
    real_write = remote.write

    def slow_write(key: str, chunks: Any, *, append: bool = False) -> None:
        clock.now += 6.0  # two writes already exceed the 10 s window
        real_write(key, chunks, append=append)

    remote.write = slow_write  # type: ignore[method-assign]
    mgr.flag.request("SIGTERM")
    mgr.maybe_save(3, model={"w": np.ones(2)})
    assert clock.now - 1000.0 <= 10.0 + 6.0  # at most one write past the budget
    assert status(tmp_path, run) == "preempted"
    assert any(e["message"] == "preemption_upload_deferred" for e in events())


def test_nothing_is_uploaded_or_pruned_when_the_window_is_already_spent(
    tmp_path: Path, events: Events
) -> None:
    clock = Clock()
    remote = FakeRemote()
    run, mgr = build(tmp_path, remote=remote, clock=clock, grace=5.0)
    mgr.save_now(1, model={"w": np.ones(2)})
    uploaded = len(remote.writes)
    mgr.flag.request("SIGTERM")
    clock.now += 100.0  # the signal arrived long ago: grace expired
    assert mgr.maybe_save(2, model={"w": np.ones(2)}) is None
    assert len(remote.writes) == uploaded
    assert status(tmp_path, run) == "preempted"
    assert any(e["message"] == "preemption_save_incomplete" for e in events())


def test_retention_waits_until_after_the_run_is_marked(tmp_path: Path) -> None:
    clock = Clock()
    run, mgr = build(tmp_path, clock=clock)
    for step in (1, 2, 3, 4, 5):
        mgr.save_now(step, model={"w": np.ones(2)})
    marked: list[str] = []
    real_prune = mgr._own.prune

    def prune(*args: Any, **kw: Any) -> Any:
        marked.append(status(tmp_path, run))
        return real_prune(*args, **kw)

    mgr._own.prune = prune  # type: ignore[method-assign]
    mgr.flag.request("SIGTERM")
    mgr.maybe_save(6, model={"w": np.ones(2)})
    assert marked and marked[-1] == "preempted"


@pytest.mark.parametrize("key", [None, "9c" * 32])
def test_the_preemption_checkpoint_reaches_the_remote_after_the_run_is_marked(
    tmp_path: Path, events: Events, monkeypatch: pytest.MonkeyPatch, key: str | None
) -> None:
    # found by the SkyPilot walk-through: the run was marked preempted (its store closed) before
    # the upload, every lookup of the run store then failed (checkpoint_anchor_failed) and the
    # preemption checkpoint never left the node, which is exactly the one a spot job resumes from
    if key is not None:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", key)
    remote = FakeRemote()
    run, mgr = build(tmp_path, remote=remote)
    mgr.save_now(1, model={"w": np.ones(2)})
    mgr.flag.request("SIGTERM")
    manifest = mgr.maybe_save(2, model={"w": np.full(2, 2.0)})
    assert manifest is not None and status(tmp_path, run) == "preempted"
    base = f"{run.run_id}/last/step_000000002"
    assert f"{base}.manifest.json" in remote.objects and f"{base}.safetensors" in remote.objects
    names = [e["message"] for e in events()]
    assert "checkpoint_anchor_failed" not in names
    assert names.count("checkpoint_uploaded") == 2
    mgr.close()


def test_a_resumed_run_also_uploads_its_preemption_checkpoint(
    tmp_path: Path, events: Events
) -> None:
    remote = FakeRemote()
    parent, first = build(tmp_path, remote=remote)
    first.save_now(1, model={"w": np.ones(2)})
    first.close()
    parent.finish("preempted")
    meta = runs.get_run_view("p", parent.run_id, store_dir=tmp_path / "store").meta
    from ckpt_support import run_meta as fresh

    child_id = fresh()[0].run_id
    child_meta = runs.resume_from(meta, run_id=child_id, host="h", name=None)
    repro = fresh()[1].model_copy(update={"run_id": child_id})
    child = runs.start_run(child_meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False,
        remote=URI, remote_store=remote, remote_sleep=lambda s: None,
    )  # fmt: skip
    assert mgr.restore() is not None
    mgr.flag.request("SIGTERM")
    assert mgr.maybe_save(2, model={"w": np.full(2, 2.0)}) is not None
    assert f"{child_id}/last/step_000000002.manifest.json" in remote.objects
    names = [e["message"] for e in events()]
    assert "preemption_upload_deferred" not in names and "checkpoint_anchor_failed" not in names
    mgr.close()
