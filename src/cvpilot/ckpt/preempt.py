"""Preemption: a flag that a signal handler (or a notice poller) sets and the training loop reads.

The handler does the least that works: it stores the signal number and a timestamp into plain
attributes and sets one boolean. It takes no lock, allocates no lock, logs nothing, does no I/O
and raises nothing (an exception in a handler would surface in whatever the main thread was
executing, which is the host's training loop). When it replaces a handler that was already
installed, that handler runs after the flag is set (its behaviour is the application's own), is
reported once as ``signal_handler_displaced`` and is put back by ``close``. Everything else (the
event, the save, the run status) happens later, on the training thread, at the next ``maybe_save``.

Installing handlers works only on the main thread; elsewhere ``install`` does nothing and the
flag can still be set by ``request`` or a poller. The poller is optional, off by default, polls
at most once per second per source, and gives up after ``max_errors`` consecutive failures.
"""

from __future__ import annotations

import contextlib
import signal
import threading
import time
from collections.abc import Callable, Sequence
from types import FrameType
from typing import Protocol

from cvpilot.core.errors import NoticeUnavailable  # noqa: F401  # re-exported for the poller's users

MIN_POLL_S = 1.0
MAX_POLL_S = 3600.0


def default_signals() -> tuple[int, ...]:
    """SIGTERM and, where it exists, SIGUSR1 (the usual scheduler warning signals)."""
    found = [getattr(signal, name, None) for name in ("SIGTERM", "SIGUSR1")]
    return tuple(int(s) for s in found if s is not None)


class PreemptionFlag:
    """Set once by the first signal or notice; read by ``maybe_save``."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._set = False
        self._source = ""
        self._since = 0.0

    @property
    def is_set(self) -> bool:
        return self._set

    @property
    def source(self) -> str:
        """``"SIGTERM"``, ``"SIGUSR1"``, a notice source name, or ``""`` while clear."""
        return self._source

    @property
    def since(self) -> float:
        """Clock reading when the flag was set (``0.0`` while clear)."""
        return self._since

    def on_signal(self, signum: int, frame: FrameType | None) -> None:
        if self._set:
            return
        try:
            self._source = signal.Signals(signum).name
            self._since = self._clock()
        except BaseException:  # noqa: BLE001  # a handler must never raise into the host
            self._since = 0.0
        self._set = True

    def request(self, source: str) -> None:
        """Set the flag from ordinary code (a poller thread, a test)."""
        if not self._set:
            self._source = source
            self._since = self._clock()
            self._set = True


class SignalGuard:
    """Install ``flag.on_signal`` for some signals and put the old handlers back on request."""

    def __init__(self, flag: PreemptionFlag, signals: Sequence[int]) -> None:
        self._flag = flag
        self._signals = tuple(signals)
        self._previous: dict[int, object] = {}
        self.displaced: list[tuple[int, bool]] = []
        """``(signal, chained)`` for each custom handler this guard replaced."""

    def _handler(self, previous: object) -> Callable[[int, FrameType | None], None]:
        flag = self._flag
        chain = previous if callable(previous) else None  # SIG_DFL and SIG_IGN are not callable

        def handle(signum: int, frame: FrameType | None) -> None:
            flag.on_signal(signum, frame)  # first, and nothing in it can raise
            if chain is not None:
                chain(signum, frame)  # then the handler that was there: its behaviour is yours

        return handle

    def install(self) -> list[int]:
        """Install the handlers; returns the signals that were installed (empty off the main
        thread and for signals that cannot be caught). A handler that was already there keeps
        running after the flag is set (``displaced`` lists them) and is put back by ``restore``."""
        installed: list[int] = []
        try:
            for signum in self._signals:
                try:
                    previous = signal.getsignal(signum)
                    signal.signal(signum, self._handler(previous))
                except (ValueError, OSError):
                    continue
                self._previous[signum] = previous
                installed.append(signum)
                if previous not in (signal.SIG_DFL, signal.SIG_IGN, None):
                    self.displaced.append((signum, callable(previous)))
        except BaseException:
            self.restore()
            raise
        return installed

    def restore(self) -> None:
        for signum, handler in list(self._previous.items()):
            try:
                signal.signal(signum, handler)  # type: ignore[arg-type]
            except (ValueError, OSError, TypeError):
                pass
            del self._previous[signum]


class NoticeSource(Protocol):
    """Reports whether the platform announced that this machine will be reclaimed."""

    name: str

    def check(self) -> bool: ...


Report = Callable[..., None]
"""``report(event, **fields)``: where the poller says what it is doing (the manager's log)."""


class NoticePoller:
    """A daemon thread that asks a ``NoticeSource`` every ``interval`` seconds.

    It never gives up quietly: a failing source is retried with exponential backoff (the interval
    doubles up to ``max_backoff``), the first failure and the ``max_errors``-th consecutive one are
    reported as ``notice_poller_degraded`` (with the error type and, when the service answered,
    its status), recovery as ``notice_poller_recovered``, and whenever the thread ends it says why
    (``preemption_poller_stopped`` with reason ``notice``, ``stopped`` or ``crashed``).
    """

    def __init__(
        self,
        source: NoticeSource,
        flag: PreemptionFlag,
        *,
        interval: float = 5.0,
        max_errors: int = 5,
        max_backoff: float = 300.0,
        sleep: Callable[[float], None] | None = None,
        report: Report | None = None,
    ) -> None:
        if not MIN_POLL_S <= interval <= MAX_POLL_S:
            raise ValueError(f"poll interval must be between {MIN_POLL_S} and {MAX_POLL_S} seconds")
        if max_errors < 1:
            raise ValueError("max_errors must be positive")
        if max_backoff < interval:
            raise ValueError("max_backoff must be at least the poll interval")
        self._source = source
        self._flag = flag
        self._interval = interval
        self._max_errors = max_errors
        self._max_backoff = max_backoff
        self._sleep = sleep
        self._report = report
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.errors = 0

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="cvpilot-spot-poll", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _wait(self, seconds: float) -> bool:
        if self._sleep is not None:
            self._sleep(seconds)
            return self._stop.is_set()
        return self._stop.wait(seconds)

    def _say(self, event: str, **fields: object) -> None:
        if self._report is not None:
            with contextlib.suppress(Exception):  # reporting must never end the poller
                self._report(event, source=self._source.name, **fields)

    def _run(self) -> None:
        reason = "crashed"
        try:
            reason = self._loop()
        finally:
            self._say("preemption_poller_stopped", reason=reason, errors=self.errors)

    def _loop(self) -> str:
        consecutive = 0
        while not self._stop.is_set():
            try:
                noticed = self._source.check()
            except Exception as exc:  # noqa: BLE001  # a metadata endpoint that fails is not an emergency
                self.errors += 1
                consecutive += 1
                if consecutive in (1, self._max_errors):
                    self._say(
                        "notice_poller_degraded", consecutive=consecutive,
                        error_type=type(exc).__name__, status=getattr(exc, "status", None),
                    )  # fmt: skip
                delay = min(self._interval * 2 ** (consecutive - 1), self._max_backoff)
            else:
                if consecutive:
                    self._say("notice_poller_recovered", failures=consecutive)
                consecutive = 0
                if noticed:
                    self._flag.request(self._source.name)
                    return "notice"
                delay = self._interval
            if self._wait(delay):
                return "stopped"
        return "stopped"
