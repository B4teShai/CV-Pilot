"""Installing handlers and a notice poller is all-or-nothing, restores what it replaced, chains to
the handler it displaced and says so once."""

from __future__ import annotations

import signal
import sys
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from types import FrameType
from typing import Any

import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import preempt
from cvpilot.ckpt.manager import CheckpointManager

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")
Events = Callable[[], list[dict[str, Any]]]
SIG = signal.SIGUSR1


@pytest.fixture
def usr1() -> Iterator[None]:
    before = signal.getsignal(SIG)
    yield
    signal.signal(SIG, before)


def start(tmp: Path) -> runs.RunHandle:
    meta, repro = run_meta()
    return runs.start_run(meta, repro, store_dir=tmp / "store")


def manager(run: runs.RunHandle, tmp: Path, **kw: Any) -> CheckpointManager:
    return CheckpointManager(run, dir=tmp / "ckpt", every_n_steps=1, **kw)


def test_a_failing_notice_source_leaves_no_handler_and_no_lease_behind(
    tmp_path: Path, usr1: None
) -> None:
    run = start(tmp_path)
    original = signal.getsignal(SIG)
    with pytest.raises(Exception):  # noqa: B017, PT011  # whatever the registry raises
        manager(run, tmp_path, notice_source="no-such-source")
    assert signal.getsignal(SIG) is original
    # the liveness lease was released: a new manager can take it and finds it free of the old one
    again = manager(run, tmp_path, install_signal_handlers=False)
    assert again._lease_fd is not None
    again.close()
    run.finish()


def test_an_invalid_poll_interval_leaves_the_handlers_alone(tmp_path: Path, usr1: None) -> None:
    class Never:
        name = "never"

        def check(self) -> bool:
            return False

    run = start(tmp_path)
    original = signal.getsignal(SIG)
    before = threading.active_count()
    with pytest.raises(ValueError, match="interval"):
        manager(run, tmp_path, notice_source=Never(), notice_interval=0.0)
    assert signal.getsignal(SIG) is original and threading.active_count() == before
    run.finish()


def test_a_failure_after_the_handlers_went_in_puts_them_back(
    tmp_path: Path, usr1: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = start(tmp_path)
    original = signal.getsignal(SIG)

    def boom(*_a: Any, **_k: Any) -> None:
        raise RuntimeError("poller could not start")

    monkeypatch.setattr(preempt.NoticePoller, "start", boom)

    class Never:
        name = "never"

        def check(self) -> bool:
            return False

    with pytest.raises(RuntimeError):
        manager(run, tmp_path, notice_source=Never())
    assert signal.getsignal(SIG) is original
    run.finish()


def test_the_displaced_handler_still_runs_after_the_flag_is_set(
    tmp_path: Path, usr1: None, events: Events
) -> None:
    calls: list[tuple[int, bool]] = []

    def app_handler(signum: int, frame: FrameType | None) -> None:
        calls.append((signum, "mgr" in locals()))

    signal.signal(SIG, app_handler)
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    assert SIG in mgr.installed_signals
    signal.raise_signal(SIG)
    assert mgr.flag.is_set and calls == [(SIG, False)]
    displaced = [e for e in events() if e["message"] == "signal_handler_displaced"]
    assert len(displaced) == 1 and displaced[0]["signal"] == "SIGUSR1"
    assert displaced[0]["level"] == "WARN" and displaced[0]["chained"] is True
    mgr.close()
    assert signal.getsignal(SIG) is app_handler  # put back exactly
    run.finish()


def test_ignored_and_default_dispositions_are_neither_chained_nor_reported(
    tmp_path: Path, usr1: None, events: Events
) -> None:
    signal.signal(SIG, signal.SIG_IGN)
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    signal.raise_signal(SIG)
    assert mgr.flag.is_set  # the process is still here: nothing chained to SIG_DFL
    assert [e for e in events() if e["message"] == "signal_handler_displaced"] == []
    mgr.close()
    assert signal.getsignal(SIG) == signal.SIG_IGN
    run.finish()


def test_a_second_signal_chains_again_but_is_reported_once(
    tmp_path: Path, usr1: None, events: Events
) -> None:
    seen: list[int] = []
    signal.signal(SIG, lambda s, f: seen.append(s))
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    signal.raise_signal(SIG)
    signal.raise_signal(SIG)
    assert seen == [SIG, SIG]
    assert len([e for e in events() if e["message"] == "signal_handler_displaced"]) == 1
    mgr.close()
    run.finish()


def test_the_flag_handler_itself_still_has_no_chaining_or_io() -> None:
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(preempt.PreemptionFlag.on_signal).strip())
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert calls <= {"self._clock", "signal.Signals", "getattr"}


def test_the_manager_logs_what_its_poller_does(tmp_path: Path, events: Events) -> None:
    import time

    class Flaky:
        name = "flaky"

        def __init__(self) -> None:
            self.answers: list[Any] = [OSError("down"), False, True]

        def check(self) -> bool:
            answer = self.answers.pop(0)
            if isinstance(answer, BaseException):
                raise answer
            return bool(answer)

    run = start(tmp_path)
    mgr = manager(
        run, tmp_path, notice_source=Flaky(), notice_interval=1.0, install_signal_handlers=False
    )
    assert mgr._poller is not None
    deadline = time.monotonic() + 8.0
    while not mgr.flag.is_set and time.monotonic() < deadline:
        time.sleep(0.05)
    mgr.close()
    names = [e["message"] for e in events()]
    assert mgr.flag.is_set and mgr.flag.source == "flaky"
    assert "notice_poller_degraded" in names and "notice_poller_recovered" in names
    stopped = [e for e in events() if e["message"] == "preemption_poller_stopped"]
    assert stopped and stopped[-1]["reason"] == "notice"
    run.finish()
