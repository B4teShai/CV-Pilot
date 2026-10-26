"""Preemption inside the manager: save inside the grace window, give up after it, never raise."""

from __future__ import annotations

import os
import signal
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import preempt
from cvpilot.ckpt.manager import CheckpointManager

Events = Callable[[], list[dict[str, Any]]]


class Clock:
    def __init__(self) -> None:
        self.now = 500.0

    def __call__(self) -> float:
        return self.now


def build(tmp_path: Path, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager, Clock]:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    clock = Clock()
    kw.setdefault("install_signal_handlers", False)
    kw.setdefault("every_n_steps", 1000)
    return run, CheckpointManager(run, dir=tmp_path / "ckpt", clock=clock, **kw), clock


def model(value: float = 1.0) -> dict[str, Any]:
    return {"w": np.full(4, value, dtype=np.float32)}


def names(events: Events) -> list[str]:
    return [e["message"] for e in events()]


def test_a_flag_inside_the_grace_window_saves_and_marks_the_run_preempted(
    tmp_path: Path, events: Events
) -> None:
    run, manager, clock = build(tmp_path, grace_seconds=30)
    assert manager.maybe_save(1, model=model()) is None and not manager.preempted()
    manager.flag.request("SIGTERM")
    clock.now += 10
    manifest = manager.maybe_save(2, model=model(2.0))
    assert manifest is not None and manifest.step == 2 and manager.preempted()
    assert run.status == "preempted"
    seen = names(events)
    order = [
        n
        for n in seen
        if n in {"preemption_signal_received", "checkpoint_saved", "run_preempted", "run_finished"}
    ]
    assert order == [
        "preemption_signal_received",
        "checkpoint_saved",
        "run_preempted",
        "run_finished",
    ]
    announced = next(e for e in events() if e["message"] == "preemption_signal_received")
    assert announced["signal"] == "SIGTERM" and announced["grace_seconds"] == 30.0
    assert manager.maybe_save(3, model=model()) is None  # handled once; the run is closed


def test_after_the_grace_window_nothing_is_saved_and_the_incomplete_save_is_logged(
    tmp_path: Path, events: Events
) -> None:
    run, manager, clock = build(tmp_path, grace_seconds=5)
    manager.save_now(1, model=model())
    manager.flag.request("SIGUSR1")
    clock.now += 6
    assert manager.maybe_save(2, model=model()) is None
    incomplete = [e for e in events() if e["message"] == "preemption_save_incomplete"]
    assert incomplete and incomplete[0]["reason"] == "grace_expired"
    assert run.status == "preempted"
    restored = manager.restore()
    assert (
        restored is not None and restored.manifest.step == 1
    )  # the previous one is the resume point


def test_a_failing_save_during_preemption_never_raises_into_the_trainer(
    tmp_path: Path, events: Events, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, manager, _ = build(tmp_path)
    manager.save_now(1, model=model())
    manager.flag.request("SIGTERM")

    def full(fd: int, data: Any) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", full)
    assert manager.maybe_save(2, model=model()) is None
    monkeypatch.undo()
    assert run.status == "preempted"
    assert any(e["message"] == "preemption_save_incomplete" for e in events())


def test_a_preemption_at_an_already_saved_step_just_marks_the_run(tmp_path: Path) -> None:
    run, manager, _ = build(tmp_path, every_n_steps=1)
    first = manager.maybe_save(1, model=model())
    assert first is not None
    manager.flag.request("SIGTERM")
    assert manager.maybe_save(1, model=model()) is None
    assert run.status == "preempted"


@pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
def test_handlers_are_installed_on_construction_and_restored_on_close(tmp_path: Path) -> None:
    before = signal.getsignal(signal.SIGUSR1)
    run, manager, _ = build(tmp_path, install_signal_handlers=True, signals=(signal.SIGUSR1,))
    assert manager.installed_signals == [signal.SIGUSR1]
    os.kill(os.getpid(), signal.SIGUSR1)
    assert manager.flag.is_set and not manager.preempted()  # requested, handled at the next save
    manager.close()
    assert signal.getsignal(signal.SIGUSR1) == before
    run.finish()


def test_the_manager_is_a_context_manager_that_restores_handlers(tmp_path: Path) -> None:
    before = signal.getsignal(signal.SIGUSR1)
    run, manager, _ = build(tmp_path, install_signal_handlers=True, signals=(signal.SIGUSR1,))
    with manager:
        assert signal.getsignal(signal.SIGUSR1) != before
    assert signal.getsignal(signal.SIGUSR1) == before
    run.finish()


class Notice:
    name = "fake-cloud"

    def check(self) -> bool:
        return True


def test_a_platform_notice_sets_the_flag_through_the_poller(tmp_path: Path) -> None:
    import time

    run, manager, _ = build(tmp_path, notice_source=Notice(), notice_interval=1.0)
    deadline = time.monotonic() + 3
    while not manager.flag.is_set and time.monotonic() < deadline:
        time.sleep(0.01)
    assert manager.flag.source == "fake-cloud"
    manager.close()
    run.finish()


def test_the_poller_is_off_by_default(tmp_path: Path) -> None:
    run, manager, _ = build(tmp_path)
    assert manager._poller is None
    run.finish()
    assert preempt.default_signals()


def test_a_signal_right_after_a_due_save_is_handled_by_the_next_maybe_save(tmp_path: Path) -> None:
    # found by the SkyPilot walk-through: the documented loop (maybe_save, then "if preempted():
    # break") broke out WITHOUT the preemption save and with the run still running when the signal
    # landed between the two calls, because preempted() reported the raw flag
    run, manager, _ = build(tmp_path, every_n_steps=1)
    assert manager.maybe_save(1, model=model()) is not None  # a regular save
    manager.flag.request("SIGTERM")  # the signal lands here, between the two calls
    assert not manager.preempted()  # not handled yet: the loop runs one more step
    assert run.status == "running"
    manifest = manager.maybe_save(2, model=model(2.0))
    assert manifest is not None and manifest.step == 2
    assert manager.preempted() and run.status == "preempted"
