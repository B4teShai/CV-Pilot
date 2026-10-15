"""Public Python API: ``init``, ``log``, ``log_artifact``, ``finish`` and the ``Run`` handle.

One run is active per process. Use it directly or as a context manager (``failed`` on an
exception, ``finished`` otherwise). Settings come from ``CVPILOT_*`` variables read at
``init``; explicit arguments win.
"""

from __future__ import annotations

import atexit
import platform
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal

from cvpilot import runs
from cvpilot.core.errors import NoActiveRunError, RunAlreadyActiveError
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, Artifact, ArtifactKind, EvalSnapshot, RunMeta, Task
from cvpilot.logging import ensure_configured, get_logger
from cvpilot.repro import capture_repro
from cvpilot.settings import load_settings

if TYPE_CHECKING:
    from cvpilot.eval import AnnotationTable, PredictionTable

_state_lock = threading.Lock()
_active: Run | None = None
_atexit_registered = False


class Run:
    """Handle of one run, returned by :func:`init`."""

    def __init__(self, handle: runs.RunHandle) -> None:
        self._handle = handle

    @property
    def run_id(self) -> str:
        return self._handle.run_id

    @property
    def status(self) -> str:
        return self._handle.status

    def log(self, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
        """Buffer scalar metrics at ``step``; see :meth:`cvpilot.runs.RunHandle.log`."""
        self._handle.log(points, step=step, epoch=epoch)

    def log_artifact(
        self, path: str | Path, *, kind: ArtifactKind, meta: Mapping[str, Any] | None = None
    ) -> Artifact:
        """Store a file by content address and register it with the run."""
        return self._handle.log_artifact(Path(path), kind=kind, meta=meta)

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
        """Evaluate predictions against ground truth and store an immutable snapshot.

        Args:
            preds, gts: ``cvpilot.eval.PredictionTable`` and ``AnnotationTable``; the run's
                task decides between detection and top-1 classification.
            step: the training step the predictions come from.
            class_names: names by class id; when omitted, ``["0", "1", ...]`` up to the largest
                ground-truth class id (predictions with a higher id are rejected).
            slices: slice key to image metadata column, for example ``{"camera": "camera"}``.
            n_resamples, seed, level: bootstrap intervals over images (``0`` turns them off).
            provenance: optional ``{"checkpoint_sha256": ..., "git_commit": ...}`` stored in
                the snapshot.

        Raises:
            EvalInputError: the inputs fail validation.
            SnapshotExistsError: this ``(eval set, step)`` was already logged for the run.
            RunClosedError: the run has finished.
        """
        return self._handle.log_eval(
            preds,
            gts,
            step=step,
            class_names=class_names,
            slices=slices,
            n_resamples=n_resamples,
            seed=seed,
            level=level,
            provenance=provenance,
        )

    def finish(self, status: Literal["finished", "failed", "preempted"] = "finished") -> None:
        """Flush, compact and set the final status."""
        global _active
        self._handle.finish(status)
        with _state_lock:
            if _active is self:
                _active = None

    def abandon(self) -> None:
        """Flush and release the store without changing the status (crash-like exit)."""
        global _active
        self._handle.abandon()
        with _state_lock:
            if _active is self:
                _active = None

    def __enter__(self) -> Run:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self.status in ("running", "resumed"):
            self.finish("failed" if exc_type is not None else "finished")


def init(
    project: str | None = None,
    *,
    task: Task,
    name: str | None = None,
    config: Mapping[str, Any] | None = None,
    dataset_uri: str | Path | None = None,
    seed: int | None = None,
    framework: str | None = None,
    resume_from: str | None = None,
    store_dir: str | Path | None = None,
) -> Run:
    """Start a run and make it the active run of this process.

    Args:
        project: store directory name, ``[a-z0-9][a-z0-9_-]{0,63}``; defaults to
            ``CVPILOT_PROJECT``.
        task: ``"detection"`` or ``"classification"``.
        config: resolved training config; secrets are redacted by key before storage.
        dataset_uri: local dataset directory hashed into ``Repro.dataset_hash``.
        framework: ``"<name>==<version>"``; defaults to the running Python.
        resume_from: run id of a ``preempted`` or ``failed`` run to continue.
        store_dir: overrides ``CVPILOT_STORE_DIR`` (default ``~/.cvpilot``).

    Raises:
        SettingsError: an invalid ``CVPILOT_*`` variable.
        RunAlreadyActiveError: another run of this process is still active.
        InvalidResumeError, RunNotFoundError: ``resume_from`` cannot be resumed or is unknown.
        FileNotFoundError, NotADirectoryError: ``dataset_uri`` is unusable.
        StoreError: the store cannot be created or written.
    """
    global _active, _atexit_registered
    settings = load_settings()
    ensure_configured("cvpilot-client", settings)
    project = project or settings.project
    if project is None:
        raise ValueError("a project is required: pass project= or set CVPILOT_PROJECT")
    root = settings.store_dir if store_dir is None else Path(store_dir)
    with _state_lock:
        if _active is not None:
            raise RunAlreadyActiveError(
                f"run {_active.run_id} is still active; call finish() before init()"
            )
        run_id = new_run_id()
        captured = capture_repro(
            run_id,
            None if dataset_uri is None else str(dataset_uri),
            config,
            seed,
            store_command_line=settings.store_command_line,
        )
        framework = framework or f"python=={platform.python_version()}"
        started = datetime.now(timezone.utc)
        if resume_from is not None:
            parent = runs.get_run_view(project, resume_from, store_dir=root).meta
            if parent.task != task:
                raise ValueError(f"task {task!r} differs from the resumed run's {parent.task!r}")
            meta = runs.resume_from(parent, run_id=run_id, host=captured.host, name=name)
        else:
            meta = RunMeta(
                run_id=run_id,
                project=project,
                name=name or f"{framework.split('==')[0]}-{started.date().isoformat()}",
                status="running",
                task=task,
                framework=framework,
                host=captured.host,
                started_at=started,
                schema_version=SCHEMA_VERSION,
            )
        handle = runs.start_run(meta, captured.repro, store_dir=root, git_diff=captured.git_diff)
        _active = Run(handle)
        if not _atexit_registered:
            atexit.register(_flush_active_at_exit)
            _atexit_registered = True
        return _active


def _require_active() -> Run:
    if _active is None:
        raise NoActiveRunError("no active run; call cvpilot.init() first")
    return _active


def log(points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None:
    """Log scalars to the active run."""
    _require_active().log(points, step=step, epoch=epoch)


def log_eval(
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
    """Evaluate and store a snapshot on the active run; see :meth:`Run.log_eval`."""
    return _require_active().log_eval(
        preds,
        gts,
        step=step,
        class_names=class_names,
        slices=slices,
        n_resamples=n_resamples,
        seed=seed,
        level=level,
        provenance=provenance,
    )


def log_artifact(
    path: str | Path, *, kind: ArtifactKind, meta: Mapping[str, Any] | None = None
) -> Artifact:
    """Register a file with the active run."""
    return _require_active().log_artifact(path, kind=kind, meta=meta)


def finish(status: Literal["finished", "failed", "preempted"] = "finished") -> None:
    """Finish the active run."""
    _require_active().finish(status)


def _flush_active_at_exit() -> None:
    """Best effort at interpreter exit: write buffered points; the status stays ``running``."""
    run = _active
    if run is None:
        return
    try:
        run.abandon()
    except Exception as exc:  # noqa: BLE001  # exit hook: report, never raise during shutdown
        get_logger("cvpilot.api").warning(
            "exit_flush_failed", run_id=run.run_id, error_type=type(exc).__name__
        )
