"""The preemption flag, the signal handler that only sets it, and the optional notice poller."""

from __future__ import annotations

import ast
import inspect
import os
import signal
import sys
import threading
import time
from typing import Any

import pytest

from cvpilot.ckpt import preempt

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")


def test_the_flag_starts_clear_and_records_the_first_signal() -> None:
    flag = preempt.PreemptionFlag(clock=lambda: 42.0)
    assert not flag.is_set and flag.source == ""
    flag.on_signal(signal.SIGTERM, None)
    assert flag.is_set and flag.source == "SIGTERM" and flag.since == 42.0
    flag.on_signal(signal.SIGUSR1, None)  # a second signal does not rewrite the first
    assert flag.source == "SIGTERM"


def test_the_handler_body_is_async_signal_safe() -> None:
    """Only attribute stores and one clock read: no logging, locks, I/O or allocation of locks."""
    tree = ast.parse(inspect.getsource(preempt.PreemptionFlag.on_signal).strip())
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    calls = [ast.unparse(node.func) for node in ast.walk(func) if isinstance(node, ast.Call)]
    assert set(calls) <= {"self._clock", "signal.Signals", "getattr"}, calls
    names = {n.id for n in ast.walk(func) if isinstance(n, ast.Name)}
    assert not names & {"print", "open", "logging", "log", "threading", "emit"}
    assert not any(isinstance(n, (ast.Raise, ast.With, ast.Await)) for n in ast.walk(func))


def test_the_handler_never_raises_even_when_the_clock_does() -> None:
    def broken() -> float:
        raise RuntimeError("clock")

    flag = preempt.PreemptionFlag(clock=broken)
    flag.on_signal(signal.SIGTERM, None)
    assert flag.is_set


def test_manual_request_sets_the_flag_with_a_source() -> None:
    flag = preempt.PreemptionFlag()
    flag.request("spot-notice")
    assert flag.is_set and flag.source == "spot-notice"


def test_installing_and_restoring_handlers() -> None:
    flag = preempt.PreemptionFlag()
    before = signal.getsignal(signal.SIGUSR1)
    guard = preempt.SignalGuard(flag, (signal.SIGUSR1,))
    assert guard.install() == [signal.SIGUSR1]
    os.kill(os.getpid(), signal.SIGUSR1)
    deadline = time.monotonic() + 2
    while not flag.is_set and time.monotonic() < deadline:
        time.sleep(0.01)
    assert flag.is_set and flag.source == "SIGUSR1"
    guard.restore()
    assert signal.getsignal(signal.SIGUSR1) == before
    guard.restore()  # idempotent


def test_installing_outside_the_main_thread_is_a_quiet_no_op() -> None:
    result: list[Any] = []

    def worker() -> None:
        result.append(preempt.SignalGuard(preempt.PreemptionFlag(), (signal.SIGUSR1,)).install())

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert result == [[]]


def test_signals_that_cannot_be_caught_are_skipped() -> None:
    guard = preempt.SignalGuard(preempt.PreemptionFlag(), (signal.SIGKILL,))
    assert guard.install() == []


def test_default_signals_are_sigterm_and_sigusr1() -> None:
    assert preempt.default_signals() == (signal.SIGTERM, signal.SIGUSR1)


class Scripted:
    name = "scripted"

    def __init__(self, answers: list[Any]) -> None:
        self.answers = list(answers)
        self.calls = 0

    def check(self) -> bool:
        self.calls += 1
        answer = self.answers.pop(0) if self.answers else False
        if isinstance(answer, BaseException):
            raise answer
        return bool(answer)


def wait_for(predicate: Any, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_poller_sets_the_flag_when_the_source_reports_a_notice() -> None:
    flag = preempt.PreemptionFlag()
    source = Scripted([False, OSError("down"), True])
    poller = preempt.NoticePoller(source, flag, interval=1.0, sleep=lambda s: None)
    poller.start()
    assert wait_for(lambda: flag.is_set)
    poller.stop()
    assert flag.source == "scripted" and source.calls == 3


def test_poller_never_gives_up_and_backs_off() -> None:
    flag = preempt.PreemptionFlag()
    source = Scripted([RuntimeError("x")] * 8 + [True])
    waits: list[float] = []
    poller = preempt.NoticePoller(
        source, flag, interval=1.0, max_errors=3, max_backoff=10.0, sleep=waits.append
    )
    poller.start()
    assert wait_for(lambda: flag.is_set)
    poller.stop()
    assert source.calls == 9 and poller.errors == 8
    assert waits[:5] == [1.0, 2.0, 4.0, 8.0, 10.0] and max(waits) == 10.0  # doubled, capped


def test_poller_reports_degradation_recovery_and_why_it_stopped() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    flag = preempt.PreemptionFlag()
    source = Scripted([OSError("a"), OSError("b"), OSError("c"), False, True])
    poller = preempt.NoticePoller(
        source, flag, interval=1.0, max_errors=3, sleep=lambda s: None,
        report=lambda event, **fields: events.append((event, fields)),
    )  # fmt: skip
    poller.start()
    assert wait_for(lambda: flag.is_set)
    poller.stop()
    assert wait_for(lambda: any(e == "preemption_poller_stopped" for e, _ in events))
    names = [e for e, _ in events]
    assert names.count("notice_poller_degraded") == 2  # at the first failure and the third
    assert names.count("notice_poller_recovered") == 1
    degraded = [f for e, f in events if e == "notice_poller_degraded"]
    assert degraded[0]["error_type"] == "OSError" and degraded[1]["consecutive"] == 3
    stopped = [f for e, f in events if e == "preemption_poller_stopped"]
    assert stopped == [{"source": "scripted", "reason": "notice", "errors": 3}]


def test_poller_reports_a_stop_request_and_a_crash() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    poller = preempt.NoticePoller(
        Scripted([]), preempt.PreemptionFlag(), interval=30.0,
        report=lambda event, **fields: events.append((event, fields)),
    )  # fmt: skip
    poller.start()
    poller.stop()
    assert events and events[-1][1]["reason"] == "stopped"

    crashing: list[tuple[str, dict[str, Any]]] = []
    bad = preempt.NoticePoller(
        Scripted([SystemExit()]),  # SystemExit ends a thread without a traceback
        preempt.PreemptionFlag(), interval=1.0, sleep=lambda s: None,
        report=lambda event, **fields: crashing.append((event, fields)),
    )  # fmt: skip
    bad.start()
    assert wait_for(lambda: any(e == "preemption_poller_stopped" for e, _ in crashing))
    assert crashing[-1][1]["reason"] == "crashed"


def test_a_failing_report_never_ends_the_poller() -> None:
    def broken(event: str, **fields: Any) -> None:
        raise RuntimeError("log sink down")

    flag = preempt.PreemptionFlag()
    poller = preempt.NoticePoller(
        Scripted([OSError("x"), True]), flag, interval=1.0, sleep=lambda s: None, report=broken
    )
    poller.start()
    assert wait_for(lambda: flag.is_set)
    poller.stop()


def test_poller_stops_promptly() -> None:
    flag = preempt.PreemptionFlag()
    poller = preempt.NoticePoller(Scripted([]), flag, interval=30.0)
    poller.start()
    started = time.monotonic()
    poller.stop()
    assert time.monotonic() - started < 2
    assert not poller.running


def test_poll_interval_is_bounded_below() -> None:
    with pytest.raises(ValueError):
        preempt.NoticePoller(Scripted([]), preempt.PreemptionFlag(), interval=0.1)
    with pytest.raises(ValueError):
        preempt.NoticePoller(Scripted([]), preempt.PreemptionFlag(), interval=10_000)
