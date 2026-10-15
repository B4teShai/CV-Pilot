"""Fan-out of flushed metric batches to tracker forwarders, isolated from training.

Guarantees (each has a test in ``tests/unit/test_forward_fanout.py``):

* ``Fanout.submit`` is called on the training thread after a batch was written to the store. It
  never blocks and never raises: it puts the batch on a bounded per-forwarder queue
  (``QUEUE_BATCHES`` batches) with ``put_nowait`` and drops it, counted, when the queue is full.
* Every adapter call (``open``, ``log_points``, ``close``) runs on that forwarder's own daemon
  worker thread. A slow or hung tracker can therefore only stall its own worker. A call that has
  run longer than ``CALL_TIMEOUT_S`` marks the forwarder stalled the next time a batch is
  offered; it is then disabled (batches counted as dropped) for the rest of the run.
* ``Fanout.close`` waits at most ``CLOSE_TIMEOUT_S`` in total for the workers to drain and call
  ``close``; a worker still busy after that is abandoned (daemon thread) and reported.
* An exception in an adapter is counted and the first one per forwarder is logged once as
  ``forward_failed`` (WARN) with the forwarder, run id, reason and the exception *type* only:
  exception text can carry URLs or credentials, so it is never logged. Later failures are only
  counted; ``close`` logs one ``forward_summary`` (WARN) with the counters when anything went
  wrong. The next batch is still offered to a forwarder whose call failed (retry), but not to
  one whose ``open`` failed or that stalled.

The time an adapter spends inside a tracker SDK is bounded by the SDK's own timeouts; this module
bounds what the *training* thread waits (nothing, except ``close`` at ``finish`` for at most
``CLOSE_TIMEOUT_S``).
"""

from __future__ import annotations

import contextvars
import os
import queue
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

from cvpilot.core.errors import CvpilotError
from cvpilot.core.schema import Forwarder, ForwarderFactory, MetricPoint, Repro, RunMeta, RunStatus
from cvpilot.logging import get_logger
from cvpilot.registry import Registry, check_requires
from cvpilot.registry import forwarders as default_registry

QUEUE_BATCHES = 256
"""Batches queued per forwarder; a batch holds at most ``FLUSH_POINTS`` (10 000) points."""
MAX_QUEUED_POINTS = 200_000
"""Points pending per forwarder (queued or being sent); a batch that would exceed it is dropped,
so the memory a stalled tracker can pin is bounded in points, not only in batches."""
CALL_TIMEOUT_S = 30.0
"""An adapter call running longer than this marks the forwarder stalled."""
CLOSE_TIMEOUT_S = 5.0
"""Longest ``Fanout.close`` waits for all workers together."""
_POLL_S = 0.05


@dataclass
class ForwarderStats:
    failed_calls: int = 0
    dropped_batches: int = 0
    dropped_points: int = 0
    disabled_reason: str | None = None
    abandoned: bool = False

    @property
    def clean(self) -> bool:
        return not (
            self.failed_calls or self.dropped_batches or self.disabled_reason or self.abandoned
        )


def resolve_forwarders(
    names: Sequence[str],
    *,
    allow_insecure: bool = False,
    allowed_hosts: Sequence[str] = (),
    registry: Registry[ForwarderFactory] | None = None,
) -> list[Forwarder]:
    """Instantiate the forwarders ``names`` through the registry (duplicates are ignored).

    Raises:
        UnknownPluginError: a name is not registered.
        MissingExtraError: the tracker package of a first-party forwarder is not installed.
        CvpilotError: an adapter rejected its configuration (for example an insecure endpoint).
    A plugin that fails with any other exception is skipped: ``forward_failed`` is logged once
    and training starts without it.
    """
    reg = registry if registry is not None else default_registry
    found: list[Forwarder] = []
    for name in dict.fromkeys(names):
        factory = reg.get(name)
        check_requires(name, factory)
        try:
            found.append(factory(allow_insecure=allow_insecure, allowed_hosts=tuple(allowed_hosts)))
        except CvpilotError:
            raise
        except Exception as exc:  # noqa: BLE001  # plugin boundary: isolate, count, log once
            get_logger("cvpilot.forward").warning(
                "forward_failed",
                forwarder=name,
                reason="construct_failed",
                error_type=type(exc).__name__,
            )
    return found


class _Worker:
    def __init__(self, adapter: Forwarder, fan: Fanout, name: str) -> None:
        self.adapter = adapter
        self.name = name
        self._fan = fan
        self.queue: queue.Queue[tuple[MetricPoint, ...]] = queue.Queue(maxsize=fan.queue_batches)
        self.stats = ForwarderStats()
        self.busy_since: float | None = None
        self.pending_points = 0
        self.stop = threading.Event()
        self.status: RunStatus = "finished"
        self.opened = False
        self._adapter_closed = False
        self._lock = threading.Lock()
        self._logged = False
        self.thread = threading.Thread(
            target=contextvars.copy_context().run,
            args=(self._loop,),
            name=f"cvpilot-forward-{name}",
            daemon=True,
        )

    # -- counters and the single log line --------------------------------------------------

    def _note(self, reason: str, *, exc: BaseException | None = None, n_points: int = 0) -> None:
        with self._lock:
            first = not self._logged
            self._logged = True
        if first:
            fields: dict[str, Any] = {
                "forwarder": self.name,
                "run_id": self._fan.meta.run_id,
                "n_points": n_points,
                "reason": reason,
            }
            if exc is not None:
                fields["error_type"] = type(exc).__name__
            get_logger("cvpilot.forward").warning("forward_failed", **fields)

    def _disable(self, reason: str) -> None:
        with self._lock:
            if self.stats.disabled_reason is None:
                self.stats.disabled_reason = reason

    def _drop(self, batch: Sequence[MetricPoint]) -> None:
        with self._lock:
            self.stats.dropped_batches += 1
            self.stats.dropped_points += len(batch)

    # -- training-thread side ---------------------------------------------------------------

    def offer(self, batch: tuple[MetricPoint, ...]) -> None:
        busy = self.busy_since
        if (
            self.stats.disabled_reason is None
            and busy is not None
            and self._fan.clock() - busy > self._fan.call_timeout_s
        ):
            self._disable("stalled")
            self._note("stalled", n_points=len(batch))
        if self.stats.disabled_reason is not None:
            self._drop(batch)
            return
        with self._lock:
            if self.pending_points + len(batch) > self._fan.max_queued_points:
                over = True
            else:
                over = False
                self.pending_points += len(batch)
        if over:
            self._drop(batch)
            self._note("queue_full", n_points=len(batch))
            return
        try:
            self.queue.put_nowait(batch)
        except queue.Full:
            with self._lock:
                self.pending_points -= len(batch)
            self._drop(batch)
            self._note("queue_full", n_points=len(batch))

    # -- worker thread ----------------------------------------------------------------------

    def _call(
        self, fn: Callable[[], None], *, n_points: int = 0, reason: str = "exception"
    ) -> bool:
        self.busy_since = self._fan.clock()
        try:
            fn()
            return True
        except Exception as exc:  # noqa: BLE001  # adapter boundary: count, log once, go on
            with self._lock:
                self.stats.failed_calls += 1
            self._note(reason, exc=exc, n_points=n_points)
            return False
        except BaseException as exc:  # an SDK calling sys.exit, an interrupt delivered here
            with self._lock:
                self.stats.failed_calls += 1
            self._note("base_exception", exc=exc, n_points=n_points)
            self._disable("base_exception")
            self._close_adapter("failed")
            return False
        finally:
            self.busy_since = None

    def _close_adapter(self, status: RunStatus) -> None:
        """Call ``adapter.close`` at most once, whatever it raises."""
        with self._lock:
            if self._adapter_closed or not self.opened:
                return
            self._adapter_closed = True
        try:
            self.adapter.close(status)
        except BaseException:  # noqa: BLE001, S110  # best effort: the forwarder is already disabled
            pass

    def _loop(self) -> None:
        meta, repro = self._fan.meta, self._fan.repro
        if self._call(lambda: self.adapter.open(meta, repro), reason="open_failed"):
            self.opened = True
        else:
            self._disable("open_failed")
        while True:
            try:
                batch = self.queue.get(timeout=_POLL_S)
            except queue.Empty:
                if self.stop.is_set():
                    break
                continue
            if self.stats.disabled_reason is not None or not self.opened:
                self._drop(batch)
            else:
                self._call(partial(self.adapter.log_points, batch), n_points=len(batch))
            with self._lock:
                self.pending_points -= len(batch)
        if self.opened and not self._adapter_closed:
            self._adapter_closed = True
            self._call(lambda: self.adapter.close(self.status))


class Fanout:
    """Delivers batches to several forwarders without ever involving the training thread."""

    def __init__(
        self,
        adapters: Sequence[Forwarder],
        meta: RunMeta,
        repro: Repro,
        *,
        queue_batches: int | None = None,
        max_queued_points: int | None = None,
        call_timeout_s: float | None = None,
        close_timeout_s: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.queue_batches = QUEUE_BATCHES if queue_batches is None else queue_batches
        if self.queue_batches < 1:
            raise ValueError("queue_batches must be positive")
        self.max_queued_points = (
            MAX_QUEUED_POINTS if max_queued_points is None else max_queued_points
        )
        self.meta = meta
        self.repro = repro
        self.call_timeout_s = CALL_TIMEOUT_S if call_timeout_s is None else call_timeout_s
        self.close_timeout_s = CLOSE_TIMEOUT_S if close_timeout_s is None else close_timeout_s
        self.clock = clock
        self._closed = False
        self._lock = threading.Lock()
        self._workers: list[_Worker] = []
        seen: dict[str, int] = {}
        for adapter in adapters:
            base = str(getattr(adapter, "name", type(adapter).__name__))
            seen[base] = seen.get(base, 0) + 1
            name = base if seen[base] == 1 else f"{base}-{seen[base]}"
            self._workers.append(_Worker(adapter, self, name))

    def start(self) -> None:
        _LIVE.add(self)
        for worker in self._workers:
            worker.thread.start()

    def _orphan(self) -> None:
        """In a forked child: the worker threads do not exist here; never deliver or join."""
        self._closed = True
        for worker in self._workers:
            worker.stats.disabled_reason = "forked"

    def submit(self, points: Sequence[MetricPoint]) -> None:
        """Offer one flushed batch to every forwarder; never blocks, never raises."""
        try:
            if self._closed:
                return
            batch = tuple(points)
            if not batch:
                return
            for worker in self._workers:
                worker.offer(batch)
        except Exception as exc:  # noqa: BLE001  # the training loop must never see this
            get_logger("cvpilot.forward").warning(
                "forward_failed", run_id=self.meta.run_id, reason="submit_failed",
                error_type=type(exc).__name__,
            )  # fmt: skip

    def close(self, status: RunStatus) -> dict[str, ForwarderStats]:
        """Drain and close the forwarders, waiting at most ``close_timeout_s`` in total."""
        with self._lock:
            first = not self._closed
            self._closed = True
        if first:
            for worker in self._workers:
                worker.status = status
                worker.stop.set()
            deadline = time.monotonic() + self.close_timeout_s
            for worker in self._workers:
                if worker.thread.is_alive():
                    worker.thread.join(max(0.0, deadline - time.monotonic()))
                if worker.thread.is_alive():
                    worker.stats.abandoned = True
            for worker in self._workers:
                s = worker.stats
                if not s.clean:
                    get_logger("cvpilot.forward").warning(
                        "forward_summary",
                        forwarder=worker.name,
                        run_id=self.meta.run_id,
                        failed_calls=s.failed_calls,
                        dropped_batches=s.dropped_batches,
                        dropped_points=s.dropped_points,
                        disabled_reason=s.disabled_reason,
                        abandoned=s.abandoned,
                    )
        return {w.name: w.stats for w in self._workers}


_LIVE: weakref.WeakSet[Fanout] = weakref.WeakSet()


def _orphan_fanouts() -> None:
    for fan in list(_LIVE):
        fan._orphan()
    _LIVE.clear()


if hasattr(os, "register_at_fork"):
    # A forked child (a dataloader worker, a DDP spawn) inherits the Fanout object but not its
    # threads; it must neither deliver the parent's points a second time nor wait for threads.
    os.register_at_fork(after_in_child=_orphan_fanouts)
