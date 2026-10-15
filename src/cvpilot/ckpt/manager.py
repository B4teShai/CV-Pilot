"""The checkpoint manager (ADR-0006): when to save, what a checkpoint holds, verified restore,
retention and preemption, on top of the verified directory in ``ckpt/directory.py``.

Typical use inside a training script::

    run = cvpilot.init("proj", task="detection")
    ckpt = CheckpointManager(run, dir="/checkpoint", keep_last=3, monitor="mAP50")
    state = ckpt.restore_latest_valid()          # None on a fresh start
    for step in ...:
        ...
        ckpt.maybe_save(step, model=model, optimizer=opt, metrics={"mAP50": value})
        if ckpt.preempted():
            break                                 # the run is already marked ``preempted``

``last`` checkpoints hold the full training state (model, optimizer, scheduler, scaler, RNG,
loader state); ``best`` checkpoints hold the model and an optional EMA only. Both are
safetensors files (tensors) with the non-tensor state as bounded JSON; see ``ckpt/state.py``.
Errors while saving (disk full, a file that does not verify) are raised to the caller after the
event ``checkpoint_save_failed``; the previous checkpoint is never touched. The preemption path
never raises: a failed save there is logged as ``preemption_save_incomplete`` and the run is
marked ``preempted`` with the previous checkpoint as its resume point.
"""

from __future__ import annotations

import functools
import hashlib
import math
import platform
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

from cvpilot.ckpt import auth, fsops, preempt
from cvpilot.ckpt.directory import (
    DEFAULT_MAX_BYTES,
    POLICIES,
    TRUST_REASONS,
    Entry,
    RunCheckpoints,
    Skipped,
)
from cvpilot.ckpt.remote import Mirror
from cvpilot.ckpt.rng import capture_rng
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointFormatError,
    CheckpointPathError,
    CheckpointUntrustedError,
    RunClosedError,
)
from cvpilot.core.schema import (
    CHECKPOINT_SUFFIX,
    CheckpointFormat,
    CheckpointKind,
    CheckpointManifest,
    RemoteStore,
    RequireTrust,
    TrustLevel,
)
from cvpilot.core.uri import parse_remote_uri
from cvpilot.registry import remotes, spot_notices
from cvpilot.settings import load_settings

DEFAULT_INTERVAL_MINUTES = 10.0
MAX_REMOTE_CANDIDATES = 10
MAX_GRACE_S = 3600.0
_WEIGHT_KEYS = ("model", "ema")
_STATE_OBJECTS = ("model", "optimizer", "scheduler", "scaler", "ema")
_RUN_METHODS = ("emit", "ancestors", "repro_facts", "record_checkpoint", "checkpoint_digest")


@dataclass
class Restored:
    """A verified checkpoint that was loaded: the state, its manifest and what was skipped."""

    state: dict[str, Any]
    manifest: CheckpointManifest
    run_id: str
    skipped: list[Skipped] = field(default_factory=list)
    trust: TrustLevel = "local_chain_only"
    """``authenticated`` (tag verified), ``anchored`` (run-store digest matched) or
    ``local_chain_only`` (only the directory's own chain vouches)."""


def _state_of(name: str, value: Any) -> Any:
    """``value.state_dict()`` for framework objects, the mapping itself for plain state."""
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    getter = getattr(value, "state_dict", None)
    if callable(getter):
        return getter()
    raise CheckpointError(f"{name} must be a mapping or have a state_dict() method")


def _head(data: bytes, n: int) -> bytes:
    return data[:n]


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class CheckpointManager:
    """Atomic, verified checkpoints for one run. See the module docstring."""

    def __init__(
        self,
        run: Any,
        *,
        dir: str | Path | None = None,  # noqa: A002  # the documented parameter name
        remote: str | None = None,
        remote_store: RemoteStore | None = None,
        remote_retries: int = 3,
        remote_deadline: float = 300.0,
        remote_timeout: float = 10.0,
        remote_sleep: Callable[[float], None] = time.sleep,
        remote_clock: Callable[[], float] = time.monotonic,
        keep_last: int = 3,
        keep_best: int = 2,
        monitor: str | None = None,
        mode: Literal["max", "min"] = "max",
        every_n_minutes: float | None = None,
        every_n_steps: int | None = None,
        signals: Sequence[int] | None = None,
        grace_seconds: float = 30.0,
        format: CheckpointFormat = "safetensors",  # noqa: A002
        max_checkpoint_bytes: int = DEFAULT_MAX_BYTES,
        verify_after_save: bool = True,
        install_signal_handlers: bool = True,
        notice_source: preempt.NoticeSource | str | None = None,
        notice_interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        lock_timeout: float = 30.0,
        require_trust: RequireTrust | None = None,
        allow_pt_format: bool | None = None,
    ) -> None:
        settings = load_settings()
        if dir is None:
            dir = settings.checkpoint_dir
        if dir is None:
            raise ValueError("dir is required: pass dir= or set CVPILOT_CHECKPOINT_DIR")
        if remote is None:
            remote = settings.checkpoint_remote
        if require_trust is None:
            require_trust = settings.ckpt_require_trust
        if require_trust not in POLICIES:
            raise ValueError(f"require_trust must be one of {', '.join(POLICIES)}")
        if allow_pt_format is None:
            allow_pt_format = settings.ckpt_allow_pt
        if format == "pt" and not allow_pt_format:
            raise ValueError(
                "the pt format needs allow_pt_format=True (or CVPILOT_CKPT_ALLOW_PT=1); "
                "prefer the default safetensors format"
            )
        if isinstance(keep_last, bool) or not isinstance(keep_last, int) or keep_last < 1:
            raise ValueError("keep_last must be an integer of at least 1")
        if isinstance(keep_best, bool) or not isinstance(keep_best, int) or keep_best < 1:
            raise ValueError("keep_best must be an integer of at least 1")
        if mode not in ("max", "min"):
            raise ValueError("mode must be 'max' or 'min'")
        if monitor is not None and (not isinstance(monitor, str) or not monitor):
            raise ValueError("monitor must be a non-empty metric name")
        if every_n_steps is not None and (
            isinstance(every_n_steps, bool)
            or not isinstance(every_n_steps, int)
            or every_n_steps < 1
        ):
            raise ValueError("every_n_steps must be an integer of at least 1")
        if every_n_minutes is not None and (
            not _is_number(every_n_minutes)
            or not math.isfinite(every_n_minutes)
            or every_n_minutes <= 0
        ):
            raise ValueError("every_n_minutes must be a positive number")
        if not _is_number(grace_seconds) or not 0 < grace_seconds <= MAX_GRACE_S:
            raise ValueError(f"grace_seconds must be in (0, {MAX_GRACE_S}]")
        if isinstance(max_checkpoint_bytes, bool) or max_checkpoint_bytes < 1:
            raise ValueError("max_checkpoint_bytes must be positive")
        if format not in ("safetensors", "pt"):
            raise ValueError("format must be 'safetensors' or 'pt'")
        self._serializer = get_serializer(format)
        self._handle = getattr(run, "handle", run)
        if not all(callable(getattr(self._handle, name, None)) for name in _RUN_METHODS):
            raise TypeError("run must be a cvpilot Run (the object returned by cvpilot.init)")
        self._run = run
        self._root = Path(dir)
        self.keep_last, self.keep_best = keep_last, keep_best
        self.monitor, self.mode = monitor, mode
        self.every_n_steps = every_n_steps
        self.every_n_minutes = (
            every_n_minutes
            if every_n_minutes is not None
            else (None if every_n_steps is not None else DEFAULT_INTERVAL_MINUTES)
        )
        self.grace_seconds = float(grace_seconds)
        self._format: CheckpointFormat = format
        self._max_bytes = max_checkpoint_bytes
        self._verify = verify_after_save
        self._lock_timeout = lock_timeout
        self._key = auth.load_key()  # environment only; never stored, logged or passed on
        self.require_trust: RequireTrust = require_trust
        self.allow_pt = allow_pt_format
        self._clock = clock
        self._dirs: dict[str, RunCheckpoints] = {}
        self._mirror: Mirror | None = None
        if remote is not None or remote_store is not None:
            if remote is None:
                raise ValueError("remote_store needs remote= (the location it serves)")
            try:
                location = parse_remote_uri(remote)
            except ValueError as exc:
                raise CheckpointPathError(str(exc)) from None
            self._mirror = Mirror(
                location,
                remote_store
                if remote_store is not None
                else remotes.get(location.scheme)(location, timeout=remote_timeout),
                retries=remote_retries,
                deadline=remote_deadline,
                sleep=remote_sleep,
                clock=remote_clock,
            )
        self._own = self._dir_for(self._handle.run_id)
        self._last_step: int | None = None
        self._last_time = clock()
        self._best_value: float | None = None
        self._best_loaded = False
        self.last_restore: Restored | None = None
        self.flag = preempt.PreemptionFlag(clock)
        self._announced = False
        self._handled = False
        self._guard = preempt.SignalGuard(
            self.flag, preempt.default_signals() if signals is None else signals
        )
        self.installed_signals: list[int] = self._guard.install() if install_signal_handlers else []
        self._poller: preempt.NoticePoller | None = None
        source = (
            spot_notices.get(notice_source)() if isinstance(notice_source, str) else notice_source
        )
        if source is not None:
            self._poller = preempt.NoticePoller(source, self.flag, interval=notice_interval)
            self._poller.start()

    # -- plumbing ---------------------------------------------------------------------------

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        owner = fields.pop("run_id", None)
        if owner is not None and owner != self._handle.run_id:
            fields["checkpoint_run_id"] = owner
        self._handle.emit(level, event, **fields)

    def _dir_for(self, run_id: str) -> RunCheckpoints:
        found = self._dirs.get(run_id)
        if found is None:
            found = RunCheckpoints(
                self._root,
                run_id,
                create=run_id == self._handle.run_id,
                max_bytes=self._max_bytes,
                emit=self._emit,
                anchor=self._handle.checkpoint_digest,
                lock_timeout=self._lock_timeout,
                key=self._key,
                require_trust=self.require_trust,
                allow_pt=self.allow_pt,
            )
            self._dirs[run_id] = found
        return found

    def _versions(self) -> dict[str, str]:
        from cvpilot import __version__

        versions = {"cvpilot": __version__, "python": platform.python_version()}
        framework = str(getattr(getattr(self._handle, "meta", None), "framework", ""))
        name, _, version = framework.partition("==")
        if name and version and name not in versions:
            versions[name[:64]] = version[:64]
        return versions

    @property
    def remote_url(self) -> str | None:
        """The validated remote location (no credentials can be in it), or None."""
        return None if self._mirror is None else self._mirror.location.url

    def close(self) -> None:
        """Stop the notice poller and put the previous signal handlers back."""
        if self._poller is not None:
            self._poller.stop()
        self._guard.restore()

    def __enter__(self) -> CheckpointManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    # -- saving -----------------------------------------------------------------------------

    def _build_state(self, kind: CheckpointKind, state: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in state.items():
            if key == "rng":
                if value is True:
                    out[key] = capture_rng()
                elif value not in (False, None):
                    out[key] = value
                continue
            converted = _state_of(key, value) if key in _STATE_OBJECTS else value
            if converted is not None:
                out[key] = converted
        if "model" not in out:
            raise CheckpointError(f"a {kind} checkpoint needs a model")
        if kind == "best":
            out = {k: v for k, v in out.items() if k in _WEIGHT_KEYS}
        return out

    def _save(
        self,
        kind: CheckpointKind,
        step: int,
        state: Mapping[str, Any],
        *,
        metric: tuple[str, float] | None = None,
    ) -> CheckpointManifest:
        started = time.perf_counter()
        try:
            data: bytes | Iterable[bytes] = self._serializer.dumps(self._build_state(kind, state))
        except (CheckpointError, OSError, ValueError, TypeError) as exc:
            self._emit(
                "error", "checkpoint_save_failed", step=step, kind=kind,
                error_type=type(exc).__name__, exc_info=exc,
            )  # fmt: skip
            raise
        return self._publish(kind, step, data, self._format, metric, started)

    def _publish(
        self,
        kind: CheckpointKind,
        step: int,
        data: bytes | Iterable[bytes],
        format: CheckpointFormat,  # noqa: A002
        metric: tuple[str, float] | None,
        started: float,
    ) -> CheckpointManifest:
        try:
            dataset_hash, git_commit = self._handle.repro_facts()
            manifest = self._own.save(
                kind=kind,
                step=step,
                data=data,
                format=format,
                versions=self._versions(),
                dataset_hash=dataset_hash,
                git_commit=git_commit,
                uri=self._intended_uri(kind, step, format),
                metric_key=None if metric is None else metric[0],
                metric_value=None if metric is None else metric[1],
                metric_mode=self.mode if metric is not None else None,
                verify=self._verify,
                before_publish=self._record,
            )
        except (CheckpointError, OSError, ValueError, TypeError) as exc:
            self._emit(
                "error",
                "checkpoint_save_failed",
                step=step,
                kind=kind,
                error_type=type(exc).__name__,
                exc_info=exc,
            )
            raise
        self._emit(
            "info",
            "checkpoint_saved",
            step=manifest.step,
            kind=kind,
            size_bytes=manifest.size_bytes,
            sha256_prefix=manifest.sha256[:12],
            is_best=kind == "best",
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
        if self._verify:
            self._emit(
                "info",
                "checkpoint_verified",
                step=manifest.step,
                kind=kind,
                sha256_prefix=manifest.sha256[:12],
            )
        self._last_time = self._clock()
        if kind == "last":
            self._last_step = step
        self._mirror_pending()
        self._retain(kind)
        return manifest

    def _intended_uri(self, kind: str, step: int, format: str) -> str | None:  # noqa: A002
        if self._mirror is None:
            return None
        name = f"step_{step:09d}.{CHECKPOINT_SUFFIX[format]}"
        return f"{self._mirror.location.url}/{self._handle.run_id}/{kind}/{name}"

    def _mirror_pending(self, limit: int = 3) -> None:
        """Upload the newest checkpoints that the remote does not hold yet (at most ``limit``);
        stop at the first failure, which is logged and never raised."""
        if self._mirror is None:
            return
        pending = [
            e.manifest
            for kind in ("last", "best")
            for e in self._own.inspect(kind, check_data=False)
            if e.manifest is not None and not self._own.is_uploaded(e.manifest)
        ]
        pending.sort(key=lambda m: (m.step, m.kind), reverse=True)
        for manifest in pending[:limit]:
            started = time.perf_counter()
            try:
                self._mirror.upload(self._own, manifest)
            except (OSError, CheckpointError) as exc:
                self._emit(
                    "warning", "checkpoint_upload_failed", step=manifest.step,
                    kind=manifest.kind, error_type=type(exc).__name__,
                    remote=self._mirror.label,
                )  # fmt: skip
                return
            self._emit(
                "info", "checkpoint_uploaded", step=manifest.step, kind=manifest.kind,
                size_bytes=manifest.size_bytes, sha256_prefix=manifest.sha256[:12],
                remote=self._mirror.label,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )  # fmt: skip

    def _record(self, manifest: CheckpointManifest, digest: str) -> None:
        """Store the manifest digest (the external anchor); a failure weakens trust, not saves."""
        try:
            self._handle.record_checkpoint(manifest, digest)
        except Exception as exc:  # noqa: BLE001  # the index must never break a verified save
            self._emit(
                "warning", "checkpoint_record_failed", step=manifest.step,
                kind=manifest.kind, error_type=type(exc).__name__,
            )  # fmt: skip

    def _retain(self, kind: CheckpointKind) -> None:
        keep = self.keep_last if kind == "last" else self.keep_best
        try:
            self._own.prune(
                kind,
                keep=keep,
                mode=self.mode,
                is_confirmed=self._own.is_uploaded if self._mirror is not None else None,
            )
        except CheckpointError as exc:
            self._emit(
                "warning", "checkpoint_prune_failed", kind=kind, error_type=type(exc).__name__
            )

    def save_now(
        self, step: int, *, kind: CheckpointKind = "last", **state: Any
    ) -> CheckpointManifest:
        """Save ``state`` (``model=``, ``optimizer=``, ``rng=True``, ...) at ``step`` now.

        Raises:
            CheckpointError: the state is unsupported or the write failed (disk full, a file
                that does not verify, another process holding the lock); the previous
                checkpoint is untouched.
            CheckpointStepError, CheckpointExistsError: the step does not advance or exists
                with other content.
        """
        if kind == "best":
            metric = None
            if self.monitor is not None and _is_number(state.get("metrics", {}).get(self.monitor)):
                metric = (self.monitor, float(state["metrics"][self.monitor]))
            if metric is None:
                raise CheckpointError("a best checkpoint needs the monitored metric in metrics=")
            state = {k: v for k, v in state.items() if k != "metrics"}
            return self._save("best", step, state, metric=metric)
        state = {k: v for k, v in state.items() if k != "metrics"}
        return self._save("last", step, state)

    def adopt(self, path: str | Path, step: int) -> CheckpointManifest:
        """Bring a checkpoint file a framework wrote (Lightning ``last.ckpt``, Ultralytics
        ``last.pt``) under verification: it is streamed into the ``last`` timeline as a ``pt``
        checkpoint with a manifest, a digest in the run store and a mirror upload, so that
        ``resume_path`` can hand back a file that was verified instead of one that merely
        exists. The source must be a regular file (never a symlink), within the size limit and
        a zip archive; its content is scanned again at resume time, before any framework loads it.

        Raises:
            CheckpointFormatError: the ``pt`` format was not opted into (``allow_pt_format``).
            CheckpointError: the source is missing, a symlink, too large or not an archive.
            CheckpointStepError, CheckpointExistsError: the step does not advance or exists
                with other content (identical bytes are a no-op).
        """
        if not self.allow_pt:
            raise CheckpointFormatError(
                "adopting a framework file stores it in the pt format; opt in with "
                "allow_pt_format=True (or CVPILOT_CKPT_ALLOW_PT=1), or save safetensors"
            )
        source = Path(path)
        started = time.perf_counter()
        digest = hashlib.sha256()
        size = 0
        for chunk in fsops.iter_regular_file(source, max_bytes=self._max_bytes):
            if size == 0:
                get_serializer("pt").check(functools.partial(_head, chunk), len(chunk))
            digest.update(chunk)
            size += len(chunk)
        if size == 0:
            raise CheckpointError("a checkpoint cannot be empty")
        existing = self._own.existing("last", step)
        if existing is not None and (existing.sha256, existing.size_bytes) == (
            digest.hexdigest(),
            size,
        ):
            return existing
        return self._publish(
            "last", step, fsops.iter_regular_file(source, max_bytes=self._max_bytes), "pt", None,
            started,
        )  # fmt: skip

    def _improves(self, value: float) -> bool:
        if not self._best_loaded:
            self._best_loaded = True
            newest = next(iter(self._own.iter_valid("best")), None)
            if newest is not None and newest.manifest.metric_value is not None:
                self._best_value = newest.manifest.metric_value
        if self._best_value is None:
            return True
        return value > self._best_value if self.mode == "max" else value < self._best_value

    def _due(self, step: int) -> bool:
        if self._last_step is not None and step <= self._last_step:
            return False
        if self.every_n_steps is not None and step - (self._last_step or 0) >= self.every_n_steps:
            return True
        minutes = self.every_n_minutes
        return minutes is not None and self._clock() - self._last_time >= minutes * 60.0

    def maybe_save(
        self,
        step: int,
        *,
        model: Any,
        optimizer: Any | None = None,
        scheduler: Any | None = None,
        scaler: Any | None = None,
        rng: bool = True,
        dataloader_state: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        ema: Any | None = None,
    ) -> CheckpointManifest | None:
        """Save when something is due: the interval elapsed (``last``), the monitored metric
        improved (``best``) or a preemption was signalled (an immediate ``last``, then the run
        is marked ``preempted``). Returns the ``last`` manifest if one was written, else the
        ``best`` one, else ``None``.
        """
        if self._handled:
            return None
        state: dict[str, Any] = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "ema": ema,
            "rng": rng,
            "dataloader_state": dataloader_state,
        }
        if self.flag.is_set:
            return self._on_preemption(step, state)
        best: CheckpointManifest | None = None
        value = None if metrics is None or self.monitor is None else metrics.get(self.monitor)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and self._improves(float(value))
        ):
            best = self._save("best", step, state, metric=(str(self.monitor), float(value)))
            self._best_value = float(value)
        last = self._save("last", step, state) if self._due(step) else None
        return last or best

    # -- preemption -------------------------------------------------------------------------

    def preempted(self) -> bool:
        """True once a signal or a platform notice asked this process to stop."""
        return self.flag.is_set

    def _on_preemption(self, step: int, state: Mapping[str, Any]) -> CheckpointManifest | None:
        if not self._announced:
            self._announced = True
            self._emit(
                "info",
                "preemption_signal_received",
                signal=self.flag.source,
                grace_seconds=self.grace_seconds,
            )
        manifest: CheckpointManifest | None = None
        reason = ""
        if self._clock() - self.flag.since > self.grace_seconds:
            reason = "grace_expired"
        elif self._last_step is not None and step <= self._last_step:
            reason = "step_already_saved"
        else:
            try:
                manifest = self._save("last", step, state)
            except (CheckpointError, OSError, ValueError, TypeError) as exc:
                reason = type(exc).__name__
        if manifest is None and reason != "step_already_saved":
            self._emit("warning", "preemption_save_incomplete", step=step, reason=reason)
        self._handled = True
        self._emit(
            "info", "run_preempted", signal=self.flag.source, step=step, saved=manifest is not None
        )
        try:
            self._run.finish("preempted")
        except RunClosedError:
            pass
        return manifest

    # -- restoring --------------------------------------------------------------------------

    def restore(
        self,
        kind: CheckpointKind = "last",
        *,
        map_location: str = "cpu",
        run_ids: Sequence[str] | None = None,
    ) -> Restored | None:
        """The newest checkpoint of ``kind`` that verifies and loads, searching this run first
        and then the runs it resumed (nearest first). Corrupt, truncated, forged, replaced or
        symlinked files are skipped and counted, never loaded; ``None`` when nothing verifies.

        Under the trust policy ``anchored`` or ``authenticated`` a candidate that lacks its own
        proof is never substituted at a lower level: if candidates exist but none qualifies,
        ``CheckpointUntrustedError`` is raised (event ``checkpoint_untrusted``).
        """
        order = (
            list(run_ids)
            if run_ids is not None
            else [self._handle.run_id, *self._handle.ancestors()]
        )
        skipped: list[Skipped] = []
        for run_id in order:
            found = self._restore_local(run_id, kind, map_location, skipped)
            if found is None and self._mirror is not None:
                found = self._restore_remote(run_id, kind, map_location, skipped)
            if found is not None:
                self.last_restore = found
                return found
        self.last_restore = None
        untrusted = [s for s in skipped if s.reason in TRUST_REASONS]
        if untrusted and self.require_trust != "any":
            self._emit(
                "error", "checkpoint_untrusted", required=self.require_trust,
                rejected=len(untrusted), reasons=sorted({s.reason for s in untrusted}),
            )  # fmt: skip
            raise CheckpointUntrustedError(
                f"{len(untrusted)} checkpoint(s) exist but none reaches the required trust level "
                f"{self.require_trust!r}; nothing was restored (see CVPILOT_CKPT_REQUIRE_TRUST)"
            )
        self._emit("info", "checkpoint_restore_empty", dir=str(self._root), skipped=len(skipped))
        return None

    def _restore_local(
        self, run_id: str, kind: CheckpointKind, map_location: str, skipped: list[Skipped]
    ) -> Restored | None:
        found = self._dir_for(run_id)
        for verified in found.iter_valid(kind, load=True):
            try:
                assert verified.data is not None  # noqa: S101  # load=True always sets it
                state = get_serializer(verified.manifest.format).loads(
                    verified.data, map_location=map_location
                )
            except (CheckpointCorruptError, CheckpointFormatError) as exc:
                reason = exc.reason if isinstance(exc, CheckpointCorruptError) else "unsafe_loader"
                skipped.append(
                    Skipped(verified.manifest.step, kind, verified.manifest.path, reason)
                )
                self._emit(
                    "warning", "checkpoint_skipped_corrupt", run_id=run_id,
                    step=verified.manifest.step, kind=kind, reason=reason,
                )  # fmt: skip
                continue
            skipped.extend(found.skipped)
            manifest = verified.manifest
            self._emit(
                "info", "checkpoint_restored", run_id=run_id, step=manifest.step, kind=kind,
                size_bytes=manifest.size_bytes, sha256_prefix=manifest.sha256[:12],
                skipped=len(skipped), trust_level=verified.trust,
            )  # fmt: skip
            return Restored(state, manifest, run_id, skipped, verified.trust)
        skipped.extend(found.skipped)
        return None

    def _restore_remote(
        self, run_id: str, kind: CheckpointKind, map_location: str, skipped: list[Skipped]
    ) -> Restored | None:
        """Nothing usable locally: fetch the newest remote checkpoints one by one (verified before
        they are kept) and restore through the same local checks."""
        assert self._mirror is not None  # noqa: S101  # the caller checked
        found = self._dir_for(run_id)
        try:
            steps = self._mirror.list_steps(run_id, kind)
        except OSError as exc:
            self._emit(
                "warning", "checkpoint_download_failed", kind=kind, error_type=type(exc).__name__
            )
            return None
        for step in steps[:MAX_REMOTE_CANDIDATES]:
            try:
                self._mirror.fetch(found, kind, step)
            except (CheckpointCorruptError, OSError) as exc:
                reason = (
                    exc.reason if isinstance(exc, CheckpointCorruptError) else "remote_unavailable"
                )
                skipped.append(Skipped(step, kind, f"remote:{kind}/{step}", reason))
                self._emit(
                    "warning", "checkpoint_skipped_corrupt", run_id=run_id, step=step, kind=kind,
                    reason=reason, remote=self._mirror.label,
                )  # fmt: skip
                continue
            restored = self._restore_local(run_id, kind, map_location, skipped)
            if restored is not None:
                return restored
        return None

    def restore_latest_valid(self, *, map_location: str = "cpu") -> dict[str, Any] | None:
        """The state of the newest valid ``last`` checkpoint, or ``None`` on a fresh start."""
        restored = self.restore("last", map_location=map_location)
        return None if restored is None else restored.state

    # -- inspection -------------------------------------------------------------------------

    def list(self) -> list[Entry]:
        """Every checkpoint of this run (both kinds, newest first) with its verification result."""
        return [
            *self._own.inspect("last"),
            *self._own.inspect("best"),
        ]

    def verify(self, manifest: CheckpointManifest) -> bool:
        """Re-verify one checkpoint against the directory (size, structure and SHA-256)."""
        found = self._dir_for(manifest.run_id)
        entries = [e for e in found.inspect(manifest.kind) if e.step == manifest.step]
        ok = bool(entries) and entries[0].status == "valid" and entries[0].manifest == manifest
        if ok:
            self._emit(
                "info", "checkpoint_verified", run_id=manifest.run_id, step=manifest.step,
                kind=manifest.kind, sha256_prefix=manifest.sha256[:12],
            )  # fmt: skip
        else:
            self._emit(
                "warning", "checkpoint_verify_failed", run_id=manifest.run_id,
                step=manifest.step, kind=manifest.kind,
                reason=entries[0].reason if entries else "missing",
            )  # fmt: skip
        return ok
