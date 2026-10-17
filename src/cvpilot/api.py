"""Public Python API: ``init``, ``log``, ``log_artifact``, ``finish`` and the ``Run`` handle.

One run is active per process. Use it directly or as a context manager (``failed`` on an
exception, ``finished`` otherwise). Settings come from ``CVPILOT_*`` variables read at
``init``; explicit arguments win.
"""

from __future__ import annotations

import atexit
import os
import platform
import sys
import threading
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any, Literal, overload

from cvpilot import runs
from cvpilot.alerts.config import AlertConfig, load_alert_config, parse_alert_config
from cvpilot.core.errors import (
    CvpilotError,
    HookNotFoundError,
    NoActiveRunError,
    RunAlreadyActiveError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, Artifact, ArtifactKind, EvalSnapshot, RunMeta, Task
from cvpilot.logging import ensure_configured, get_logger
from cvpilot.registry import check_requires, hooks
from cvpilot.repro import capture_repro
from cvpilot.settings import is_rank_zero, load_settings, parse_forwarders

if TYPE_CHECKING:
    from cvpilot.ckpt.resume import ResumePoint
    from cvpilot.core.schema import RequireTrust
    from cvpilot.eval import AnnotationTable, PredictionTable

_state_lock = threading.Lock()
_active: Run | None = None
_atexit_registered = False
_uncaught = False
_hook_installed = False


class Run:
    """Handle of one run, returned by :func:`init`."""

    def __init__(self, handle: runs.RunHandle) -> None:
        self._handle = handle

    def __reduce__(self) -> Any:
        raise TypeError(
            "a cvpilot Run cannot be pickled: it holds an open store connection (use one run "
            "per process; strategy='ddp_spawn' pickles the trainer)"
        )

    @property
    def handle(self) -> runs.RunHandle:
        """The underlying run handle (advanced use: ``CheckpointManager`` reads lineage and
        records checkpoint digests through it)."""
        return self._handle

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
    forward_to: Sequence[str] | None = None,
    alerts: str | Path | Mapping[str, Any] | None = None,
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
        forward_to: forwarder names (``"mlflow"``, ``"wandb"``, ``"trackio"`` or a plugin) that
            receive every logged scalar on worker threads; defaults to ``CVPILOT_FORWARD_TO``.
            A tracker that is down or slow never raises into or blocks the training loop.
        alerts: alert rules for this run: the path of a JSON file or the same document as a
            mapping (see the user guide); defaults to the file named by ``CVPILOT_ALERTS``.
            Alerts are evaluated on ``log``, ``log_eval`` and ``finish`` and delivered by the
            configured channels on worker threads; a dead or slow channel never raises into or
            blocks the training loop.

    Raises:
        SettingsError: an invalid ``CVPILOT_*`` variable.
        RunAlreadyActiveError: another run of this process is still active.
        InvalidResumeError, RunNotFoundError: ``resume_from`` cannot be resumed or is unknown.
        FileNotFoundError, NotADirectoryError: ``dataset_uri`` is unusable.
        StoreError: the store cannot be created or written.
        UnknownPluginError, MissingExtraError: a forwarder or channel is not registered or its
            extra is missing (raised before any run is created).
        AlertConfigError, ForwardConfigError: the alert configuration is invalid, or a channel's
            endpoint or credential (from the environment) is missing or unsafe (before any run).
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
        names = settings.forward_to if forward_to is None else parse_forwarders(forward_to)
        if names and not is_rank_zero():
            get_logger("cvpilot.api").info("forward_skipped_rank", forwarders=list(names))
            names = ()  # only rank zero forwards: one tracker run, one copy of every point
        chosen = runs.resolve_forwarders(
            names, allow_insecure=settings.env == "dev", allowed_hosts=settings.forward_hosts
        )
        alert_config = _alert_config(alerts, settings.alerts)
        alert_channels = (
            runs.resolve_channels(
                alert_config.channels,
                allow_insecure=settings.env == "dev",
                allowed_hosts=settings.forward_hosts,
            )
            if alert_config is not None
            else []
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
        handle = runs.start_run(
            meta,
            captured.repro,
            store_dir=root,
            git_diff=captured.git_diff,
            forwarders=chosen,
            alerts=alert_config,
            channels=alert_channels,
        )
        _active = Run(handle)
        if alert_config is not None:
            _install_crash_marker()
        if not _atexit_registered:
            atexit.register(_flush_active_at_exit)
            _atexit_registered = True
        return _active


def _alert_config(
    given: str | Path | Mapping[str, Any] | None, from_settings: Path | None
) -> AlertConfig | None:
    """The alert configuration of this run (the argument, else ``CVPILOT_ALERTS``), or None.

    Only rank zero alerts: one alert per event, as for the tracker forwarders.
    """
    source = given if given is not None else from_settings
    if source is None:
        return None
    if not is_rank_zero():
        get_logger("cvpilot.api").info("alert_skipped_rank")
        return None
    return parse_alert_config(source) if isinstance(source, Mapping) else load_alert_config(source)


def _framework_version(requires: Sequence[str]) -> str | None:
    import importlib.metadata

    for requirement in requires:
        for dist in requirement.split("|"):
            try:
                return f"{dist}=={importlib.metadata.version(dist)}"
            except importlib.metadata.PackageNotFoundError:
                continue
    return None


def autolog(
    target: object,
    *,
    project: str | None = None,
    forward_to: Sequence[str] | None = None,
    **init_kwargs: Any,
) -> Run:
    """Start a run and attach the hook that matches ``target`` (a Lightning ``Trainer`` or a
    YOLO model) so its scalars are logged without changing the training loop.

    ``init_kwargs`` are passed to :func:`init` (``task`` is required). The hook's callbacks never
    raise into or block the framework. The YOLO framework's hook is an AGPL-3.0 extra: the notice is
    in the ``hook_attached`` event and in docs/architecture/05-runtime.md.

    Raises:
        HookNotFoundError: no installed hook matches ``type(target)``.
        MissingExtraError: the framework's distribution is not installed.
        Anything :func:`init` raises. If attaching fails the new run is marked ``failed``.
    """
    ensure_configured("cvpilot-client", load_settings())
    log = get_logger("cvpilot.api")
    factory = hooks.find(lambda candidate: candidate().matches(target))
    if factory is None:
        log.error("hook_missing", framework=type(target).__module__.split(".")[0])
        raise HookNotFoundError(
            f"no hook matches an object of type {type(target).__name__}; "
            f"known hooks: {', '.join(hooks.names()) or 'none'}"
        )
    hook = factory()
    try:
        check_requires(hook.name, hook)
    except CvpilotError:
        log.error("hook_missing", hook=hook.name, extra=hook.name)
        raise
    framework = _framework_version(hook.requires)
    if framework is not None:
        init_kwargs.setdefault("framework", framework)
    run = init(project, forward_to=forward_to, **init_kwargs)
    try:
        hook.attach(target, run)
    except BaseException:
        run.finish("failed")
        raise
    log.info(
        "hook_attached",
        hook=hook.name,
        framework=framework or hook.name,
        extra=hook.name,
        license_notice=hook.license_notice,
    )
    return run


@overload
def resume_path(
    project: str,
    *,
    checkpoint_dir: str | Path | None = ...,
    require_trust: RequireTrust | None = ...,
    accept_rollback_to: int | None = ...,
    with_trust: Literal[False] = ...,
) -> Path | None: ...


@overload
def resume_path(
    project: str,
    *,
    checkpoint_dir: str | Path | None = ...,
    require_trust: RequireTrust | None = ...,
    accept_rollback_to: int | None = ...,
    with_trust: Literal[True],
) -> ResumePoint | None: ...


def resume_path(
    project: str,
    *,
    checkpoint_dir: str | Path | None = None,
    require_trust: RequireTrust | None = None,
    accept_rollback_to: int | None = None,
    with_trust: bool = False,
) -> Path | ResumePoint | None:
    """The path of the newest verified ``last`` checkpoint to resume from, or ``None``.

    Searches the project's ``preempted``, ``failed`` and crashed (``running``) runs, newest
    first, in ``checkpoint_dir`` (default ``CVPILOT_CHECKPOINT_DIR``), and returns the first
    checkpoint that passes size, structure, SHA-256 and the trust policy; corrupt, truncated,
    forged or symlinked files are skipped. The policy is ``require_trust``, else
    ``CVPILOT_CKPT_REQUIRE_TRUST``, else ``authenticated`` when ``CVPILOT_CKPT_KEY`` is set (a
    key alone used to enforce nothing), else ``any``. Under the strict policies a checkpoint
    older than one known to have existed is a rollback and raises (``accept_rollback_to``
    accepts it on purpose).

    The return value is a ``Path`` (never the string ``"last"``); with ``with_trust=True`` it is
    the ``cvpilot.ckpt.ResumePoint``, which carries the trust level achieved, the run it came
    from and whether that run needs ``recover_run``.

    Raises:
        ValueError: no checkpoint directory is given or configured.
        CheckpointUntrustedError: checkpoints exist but none reaches the required trust level,
            or the newest one is a rollback.
    """
    from cvpilot.ckpt.resume import find_resume_point

    ensure_configured("cvpilot-client", load_settings())
    point = find_resume_point(
        project,
        checkpoint_dir=checkpoint_dir,
        require_trust=require_trust,
        accept_rollback_to=accept_rollback_to,
    )
    if point is None:
        return None
    return point if with_trust else point.path


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


def _forget_active_in_child() -> None:
    """A forked child inherits the active run but may not use it (see ``RunHandle._orphan``)."""
    global _active
    _active = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_active_in_child)


def _install_crash_marker() -> None:
    """Remember that an uncaught exception ended the main thread, so the exit handler can send
    the ``run_failed`` alert for a run that was never finished (no ``with`` block, no
    ``finish``). ``sys.exit`` does not pass through here and is not treated as a crash."""
    global _hook_installed
    if _hook_installed:
        return
    _hook_installed = True
    previous = sys.excepthook

    def hook(exc_type: type[BaseException], exc: BaseException, tb: TracebackType | None) -> None:
        global _uncaught
        _uncaught = True
        previous(exc_type, exc, tb)

    sys.excepthook = hook


def _flush_active_at_exit() -> None:
    """Best effort at interpreter exit: write buffered points; the status stays ``running``.
    After an uncaught exception the ``run_failed`` alert is sent too (bounded by the channel
    close timeout)."""
    global _active
    run = _active
    if run is None:
        return
    try:
        run.handle.abandon(crashed=_uncaught)
        with _state_lock:
            if _active is run:
                _active = None
    except Exception as exc:  # noqa: BLE001  # exit hook: report, never raise during shutdown
        get_logger("cvpilot.api").warning(
            "exit_flush_failed", run_id=run.run_id, error_type=type(exc).__name__
        )
