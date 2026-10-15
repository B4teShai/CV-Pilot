"""Preemption: a flag that a signal handler (or a notice poller) sets and the training loop reads.

The handler does the least that works: it stores the signal number and a timestamp into plain
attributes and sets one boolean. It takes no lock, allocates no lock, logs nothing, does no I/O
and raises nothing (an exception in a handler would surface in whatever the main thread was
executing, which is the host's training loop). Everything else (the event, the save, the run
status) happens later, on the training thread, at the next ``maybe_save``.

Installing handlers works only on the main thread; elsewhere ``install`` does nothing and the
flag can still be set by ``request`` or a poller. The poller is optional, off by default, polls
at most once per second per source, and gives up after ``max_errors`` consecutive failures.
"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Sequence
from types import FrameType
from typing import Protocol

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

    def install(self) -> list[int]:
        """Install the handlers; returns the signals that were installed (empty off the main
        thread and for signals that cannot be caught)."""
        installed: list[int] = []
        for signum in self._signals:
            try:
                self._previous[signum] = signal.signal(signum, self._flag.on_signal)
                installed.append(signum)
            except (ValueError, OSError):
                continue
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


class NoticePoller:
    """A daemon thread that asks a ``NoticeSource`` every ``interval`` seconds."""

    def __init__(
        self,
        source: NoticeSource,
        flag: PreemptionFlag,
        *,
        interval: float = 5.0,
        max_errors: int = 5,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not MIN_POLL_S <= interval <= MAX_POLL_S:
            raise ValueError(f"poll interval must be between {MIN_POLL_S} and {MAX_POLL_S} seconds")
        if max_errors < 1:
            raise ValueError("max_errors must be positive")
        self._source = source
        self._flag = flag
        self._interval = interval
        self._max_errors = max_errors
        self._sleep = sleep
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

    def _wait(self) -> bool:
        if self._sleep is not None:
            self._sleep(self._interval)
            return self._stop.is_set()
        return self._stop.wait(self._interval)

    def _run(self) -> None:
        consecutive = 0
        while not self._stop.is_set():
            try:
                noticed = self._source.check()
            except Exception:  # noqa: BLE001  # a metadata endpoint that fails is not an emergency
                self.errors += 1
                consecutive += 1
                if consecutive >= self._max_errors:
                    return
            else:
                consecutive = 0
                if noticed:
                    self._flag.request(self._source.name)
                    return
            if self._wait():
                return
