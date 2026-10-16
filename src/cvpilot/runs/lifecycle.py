"""Run lifecycle: start, log with a bounded buffer, finish, recover, resume.

``log`` never touches the disk except when it flushes: the buffer holds at most
``FLUSH_POINTS`` points and is written as one Parquet file on the caller's thread when it
reaches that bound or when ``FLUSH_INTERVAL_S`` has passed since the last flush (checked on
each ``log`` call; there is no timer thread). The worst case a training step waits on I/O is
therefore one batch write of at most ``FLUSH_POINTS`` points (tens of milliseconds). A run
that dies without ``finish`` keeps status ``running``; at most the unflushed buffer is lost.

Events (constant messages, ``request_id`` = ``<run_id>:<seq>``): ``run_started``,
``run_resumed``, ``metrics_flushed`` (DEBUG), ``run_finished`` (carries ``status``).
"""

from __future__ import annotations

import itertools
import os
import threading
import time
import weakref
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args

from cvpilot.core.errors import (
    DuplicateMetricError,
    InvalidResumeError,
    InvalidTransitionError,
    MetricOrderError,
    RunClosedError,
    RunNotFoundError,
)
from cvpilot.core.schema import (
    MAX_STEP,
    Artifact,
    ArtifactKind,
    CheckpointKind,
    CheckpointManifest,
    Forwarder,
    MetricPoint,
    Repro,
    RunMeta,
    RunStatus,
    RunStore,
    is_terminal,
)
from cvpilot.logging import bind_request_id, get_logger
from cvpilot.runs.forwarding import Fanout
from cvpilot.runs.queries import open_existing, open_store

if TYPE_CHECKING:
    from cvpilot.core.schema import EvalSnapshot
    from cvpilot.eval import AnnotationTable, PredictionTable

FLUSH_POINTS = 10_000
FLUSH_INTERVAL_S = 2.0
FinalStatus = Literal["finished", "failed", "preempted"]


_HANDLES: weakref.WeakSet[RunHandle] = weakref.WeakSet()
_ORPHANED_STORES: list[Any] = []


def _orphan_handles() -> None:
    for handle in list(_HANDLES):
        handle._orphan()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_orphan_handles)


def _log() -> Any:
    """A logger bound now, so it follows the configuration active at call time."""
    return get_logger("cvpilot.runs")


def _as_float(key: str, value: object) -> float:
    if isinstance(value, (bool, str, bytes)) or value is None:
        raise ValueError(f"value for metric {key!r} must be a number, got {type(value).__name__}")
    try:
        return float(value)  # type: ignore[arg-type]  # numpy scalars and Decimal implement __float__
    except (TypeError, ValueError):
        raise ValueError(
            f"value for metric {key!r} must be a number, got {type(value).__name__}"
        ) from None


def _as_int(name: str, value: object) -> int:
    if isinstance(value, (bool, str, float)) or value is None:
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}")
    try:
        return int(value.__index__())  # type: ignore[attr-defined]
    except (AttributeError, TypeError):
        raise ValueError(f"{name} must be an integer, got {type(value).__name__}") from None


class RunHandle:
    """One active run: validates and buffers points, flushes them, finishes the run."""

    def __init__(
        self,
        meta: RunMeta,
        store: RunStore,
        *,
        flush_points: int = FLUSH_POINTS,
        flush_interval_s: float = FLUSH_INTERVAL_S,
        clock: Callable[[], float] = time.monotonic,
        fanout: Fanout | None = None,
    ) -> None:
        if flush_points < 1:
            raise ValueError("flush_points must be positive")
        self.meta = meta
        self._store = store
        self._flush_points = flush_points
        self._flush_interval_s = flush_interval_s
        self._clock = clock
        self._buffer: list[MetricPoint] = []
        self._last_step: dict[str, int] = {}
        self._valid_keys: set[str] = set()
        self._last_flush = clock()
        self._started = clock()
        self._n_flushed = 0
        self._closed = False
        self._buffer_lock = threading.Lock()
        self._flush_lock = threading.Lock()
        self._seq = itertools.count(1)
        self._status: RunStatus = meta.status
        self._fanout = fanout
        self._orphaned = False
        _HANDLES.add(self)

    @property
    def run_id(self) -> str:
        return self.meta.run_id

    @property
    def status(self) -> RunStatus:
        return self._status

    @property
    def buffered_points(self) -> int:
        return len(self._buffer)

    @property
    def store_path(self) -> Path:
        return self._store.path

    def _orphan(self) -> None:
        """In a forked child: stop using the inherited store connection, buffer and forwarders.

        SQLite connections must not cross a fork, so the child never touches (or closes) the
        parent's connection; the object is kept alive so that garbage collection cannot close it.
        """
        self._orphaned = True
        self._closed = True
        self._buffer = []
        self._fanout = None
        _ORPHANED_STORES.append(self._store)

    def _closed_error(self) -> RunClosedError:
        if self._orphaned:
            return RunClosedError(
                f"run {self.run_id} belongs to the parent process; a forked child cannot use it"
            )
        return RunClosedError(f"run {self.run_id} is finished")

    def emit(self, level: str, event: str, **fields: Any) -> None:
        """Log one event with ``request_id`` = ``<run_id>:<seq>``."""
        bind_request_id(f"{self.run_id}:{next(self._seq)}")
        getattr(_log(), level)(event, run_id=self.run_id, **fields)

    # -- logging points ---------------------------------------------------

    def log(self, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
        """Validate and buffer ``points`` at ``step``; the whole call is rejected on any error.

        Raises:
            RunClosedError: the run has finished.
            ValueError: a value or the step is not a number, or a key is invalid.
            DuplicateMetricError: ``(key, step)`` was already logged.
            MetricOrderError: ``step`` is lower than the last step logged for the key.
            StoreError: a flush triggered by this call failed; the points stay buffered.
        """
        if self._closed:
            raise self._closed_error()
        if not points:
            return
        step_i = _as_int("step", step)
        epoch_i = None if epoch is None else _as_int("epoch", epoch)
        if step_i < 0 or (epoch_i is not None and epoch_i < 0):
            raise ValueError("step and epoch must be non-negative")
        if step_i > MAX_STEP:
            raise ValueError(f"step must be at most {MAX_STEP}")
        if epoch_i is not None and epoch_i > MAX_STEP:
            raise ValueError(f"epoch must be at most {MAX_STEP}")
        now = time.time()
        batch = []
        for key, value in points.items():
            number = _as_float(key, value)
            if key in self._valid_keys:
                # run_id, key, step, epoch and wall_time were validated above or on first use
                batch.append(
                    MetricPoint.model_construct(
                        run_id=self.run_id,
                        key=key,
                        step=step_i,
                        epoch=epoch_i,
                        value=number,
                        wall_time=now,
                    )
                )
                continue
            batch.append(  # first sight of this key: full model validation
                MetricPoint(
                    run_id=self.run_id,
                    key=key,
                    step=step_i,
                    epoch=epoch_i,
                    value=number,
                    wall_time=now,
                )
            )
            self._valid_keys.add(key)
        with self._buffer_lock:
            for p in batch:
                prior = self._last_step.get(p.key)
                if prior is not None and p.step == prior:
                    raise DuplicateMetricError(
                        f"duplicate metric point key={p.key!r} step={p.step}"
                    )
                if prior is not None and p.step < prior:
                    raise MetricOrderError(
                        f"step {p.step} for key {p.key!r} is not greater than the last step {prior}"
                    )
            for p in batch:
                self._last_step[p.key] = p.step
            self._buffer.extend(batch)
            due = (
                len(self._buffer) >= self._flush_points
                or self._clock() - self._last_flush >= self._flush_interval_s
            )
        if due:
            self.flush()

    def flush(self) -> None:
        """Write the buffered points as one batch. Points are kept if the write fails."""
        with self._flush_lock:
            with self._buffer_lock:
                batch, self._buffer = self._buffer, []
                self._last_flush = self._clock()
            if not batch:
                return
            try:
                name = self._store.append_points(batch)
            except (ValueError, TypeError, OverflowError) as exc:
                # A data error is deterministic: the same batch would fail forever and the buffer
                # would grow without bound, so the batch is dropped (counted, logged) and raised.
                self.emit(
                    "error", "metrics_dropped", n_points=len(batch), error_type=type(exc).__name__
                )
                raise
            except BaseException:
                with self._buffer_lock:
                    self._buffer[:0] = batch
                raise
            self._n_flushed += len(batch)
            self.emit("debug", "metrics_flushed", n_points=len(batch), file=name)
            if self._fanout is not None:
                self._fanout.submit(batch)  # never blocks, never raises (runs/forwarding.py)

    def log_artifact(
        self, path: Path, *, kind: ArtifactKind, meta: Mapping[str, Any] | None = None
    ) -> Artifact:
        """Store a file by content address and register it with the run."""
        if self._closed:
            raise self._closed_error()
        return self._store.put_artifact(self.run_id, Path(path), kind, dict(meta or {}))

    def log_eval(
        self,
        preds: PredictionTable,
        gts: AnnotationTable,
        *,
        step: int,
        class_names: Sequence[str] | None = None,
        slices: Mapping[str, str] | None = None,
        n_resamples: int = 1000,
        seed: int = 0,
        level: float = 0.95,
        provenance: Mapping[str, str | None] | None = None,
    ) -> EvalSnapshot:
        """Evaluate and store an immutable :class:`EvalSnapshot` keyed ``(run, set hash, step)``.

        The run's task selects detection or classification. Buffered metric points are not
        touched. See :func:`cvpilot.eval.record.record_eval` for the errors.

        Raises:
            RunClosedError: the run has finished.
        """
        if self._closed:
            raise self._closed_error()
        from cvpilot.eval.record import record_eval  # numpy is only needed once eval is used

        return record_eval(
            self._store,
            self.meta,
            preds,
            gts,
            step=step,
            class_names=class_names,
            slices=slices,
            n_resamples=n_resamples,
            seed=seed,
            level=level,
            emit=self.emit,
            provenance=provenance,
        )

    # -- checkpoints (what ``cvpilot.ckpt`` needs from a run) ---------------------

    def ancestors(self, limit: int = 64) -> list[str]:
        """Run ids this run resumed, nearest first (parent, grandparent, ...), at most ``limit``.

        The chain stops at a parent this store does not know and never revisits an id.
        """
        chain: list[str] = []
        seen = {self.run_id}
        current = self.meta.parent_run_id
        while current is not None and len(chain) < limit and current not in seen:
            chain.append(current)
            seen.add(current)
            try:
                current = self._store.get_run(current).parent_run_id
            except RunNotFoundError:
                break
        return chain

    def repro_facts(self) -> tuple[str, str | None]:
        """``(dataset_hash, git_commit)`` captured at ``init``, copied into checkpoint manifests."""
        repro = self._store.get_repro(self.run_id)
        return repro.dataset_hash, repro.git_commit

    def record_checkpoint(self, manifest: CheckpointManifest, manifest_sha256: str) -> None:
        """Record a checkpoint manifest's digest in the store (the directory's external anchor)."""
        if self._closed:
            raise self._closed_error()
        self._store.put_manifest(manifest, manifest_sha256)

    def checkpoint_digest(self, run_id: str, kind: CheckpointKind, step: int) -> str | None:
        """The digest recorded for a checkpoint manifest of this project, or None."""
        return self._store.get_manifest_digest(run_id, kind, step)

    def checkpoint_auth_evidence(
        self, run_id: str, kind: CheckpointKind
    ) -> tuple[int | None, list[str]]:
        """The first tagged checkpoint step the store recorded for a run and the key ids seen."""
        return self._store.auth_evidence(run_id, kind)

    def checkpoint_max_step(self, run_id: str, kind: CheckpointKind) -> int | None:
        """The highest checkpoint step the store recorded for a run, or None."""
        return self._store.max_manifest_step(run_id, kind)

    # -- finishing --------------------------------------------------------

    def finish(self, status: FinalStatus = "finished") -> None:
        """Flush, compact the metric files and set the final status.

        Raises:
            InvalidTransitionError: ``status`` is not ``finished``, ``failed`` or ``preempted``.
            RunClosedError: the run already finished.
        """
        if status not in get_args(FinalStatus):
            raise InvalidTransitionError(f"run status cannot change to {status}")
        if self._closed:
            raise RunClosedError(f"run {self.run_id} is already finished")
        self.flush()
        self._store.compact_metrics(self.run_id)
        self._store.set_status(self.run_id, status, datetime.now(timezone.utc))
        self._closed = True
        self._status = status
        self.emit(
            "info",
            "run_finished",
            project=self.meta.project,
            status=status,
            n_points=self._n_flushed,
            duration_ms=round((self._clock() - self._started) * 1000, 1),
        )
        if self._fanout is not None:
            self._fanout.close(status)  # waits at most CLOSE_TIMEOUT_S
        self._store.close()

    def abandon(self) -> None:
        """Flush what is buffered and release the store without changing the status."""
        if not self._closed:
            self.flush()
            if self._fanout is not None:
                self._fanout.close(self._status)
            self._store.close()
            self._closed = True


def start_run(
    meta: RunMeta,
    repro: Repro,
    *,
    store_dir: Path | None = None,
    git_diff: bytes | None = None,
    flush_points: int = FLUSH_POINTS,
    flush_interval_s: float = FLUSH_INTERVAL_S,
    clock: Callable[[], float] = time.monotonic,
    forwarders: Sequence[Forwarder] = (),
) -> RunHandle:
    """Persist a new run (and its diff artifact) and return its handle.

    ``forwarders`` (already resolved, see ``resolve_forwarders``) receive every flushed batch on
    their own worker threads; see ``cvpilot.runs.forwarding`` for the isolation guarantees.
    """
    store = open_store(meta.project, store_dir)
    try:
        store.put_run(meta, repro)
        if git_diff is not None:
            store.put_artifact_bytes(meta.run_id, git_diff, "diff", {})
    except BaseException:
        store.close()
        raise
    fanout: Fanout | None = None
    if forwarders:
        fanout = Fanout(forwarders, meta, repro)
        fanout.start()
    handle = RunHandle(
        meta,
        store,
        flush_points=flush_points,
        flush_interval_s=flush_interval_s,
        clock=clock,
        fanout=fanout,
    )
    handle.emit(
        "info",
        "run_started",
        project=meta.project,
        status=meta.status,
        task=meta.task,
        parent_run_id=meta.parent_run_id,
    )
    if meta.status == "resumed":
        handle.emit("info", "run_resumed", project=meta.project, parent_run_id=meta.parent_run_id)
    return handle


def log_points(
    run: RunHandle, points: Mapping[str, float], *, step: int, epoch: int | None = None
) -> None:
    run.log(points, step=step, epoch=epoch)


def finish_run(run: RunHandle, status: FinalStatus = "finished") -> None:
    run.finish(status)


def resume_from(parent: RunMeta, *, run_id: str, host: str, name: str | None) -> RunMeta:
    """The metadata of a new run that continues ``parent`` (status ``resumed``).

    Raises:
        InvalidResumeError: ``parent`` is not ``preempted`` or ``failed``.
    """
    if parent.status not in ("preempted", "failed"):
        raise InvalidResumeError(
            f"run {parent.run_id} has status {parent.status}; only preempted or failed runs resume"
        )
    return RunMeta(
        run_id=run_id,
        project=parent.project,
        name=name or parent.name,
        status="resumed",
        parent_run_id=parent.run_id,
        task=parent.task,
        framework=parent.framework,
        host=host,
        started_at=max(datetime.now(timezone.utc), parent.started_at),
        schema_version=parent.schema_version,
    )


def recover_run(project: str, run_id: str, *, store_dir: Path | None = None) -> None:
    """Mark a run left ``running`` by a dead process as ``failed`` (keeps its flushed points).

    Raises:
        RunNotFoundError, InvalidTransitionError: unknown run, or it already has a final status.
    """
    store = open_existing(project, run_id, store_dir)
    try:
        meta = store.get_run(run_id)
        if is_terminal(meta.status):
            raise InvalidTransitionError(f"run status cannot change from {meta.status} to failed")
        stats = store.metric_stats(run_id)
        store.compact_metrics(run_id)
        store.set_status(run_id, "failed", datetime.now(timezone.utc))
        bind_request_id(f"{run_id}:1")
        _log().info(
            "run_finished", run_id=run_id, project=project, status="failed", n_points=stats.n_points
        )
    finally:
        store.close()
