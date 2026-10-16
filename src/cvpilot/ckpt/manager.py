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
import os
import platform
import signal
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, overload

from cvpilot.ckpt import auth, fsops, preempt
from cvpilot.ckpt.directory import (
    DEFAULT_MAX_BYTES,
    DEFAULT_REMOTE_CANDIDATE_BYTES,
    DEFAULT_REMOTE_TOTAL_BYTES,
    POLICIES,
    TRUST_REASONS,
    AdmitContext,
    Entry,
    RunCheckpoints,
    Skipped,
    gate_failure,
)
from cvpilot.ckpt.remote import Mirror
from cvpilot.ckpt.rng import capture_rng, validate_rng
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
MAX_DISCOVERED_RUNS = 20  # newest remote runs looked at when discovering
MAX_GRACE_S = 3600.0
_WEIGHT_KEYS = ("model", "ema")
_STATE_OBJECTS = ("model", "optimizer", "scheduler", "scaler", "ema")
_RUN_METHODS = (
    "emit",
    "ancestors",
    "repro_facts",
    "record_checkpoint",
    "checkpoint_digest",
    "checkpoint_max_step",
)


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
    policy: str = "any"
    """The trust policy that was in force for this restore (after the monotone rule)."""
    weaker_allowed: bool = False
    """True when the manager was built with ``allow_weaker_trust=True``."""


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


class _HandleAnchor:
    """The run store as the directory's external anchor: manifest digests and the highest step."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def __call__(self, run_id: str, kind: CheckpointKind, step: int) -> str | None:
        digest: str | None = self._handle.checkpoint_digest(run_id, kind, step)
        return digest

    def max_step(self, run_id: str, kind: CheckpointKind) -> int | None:
        value: int | None = self._handle.checkpoint_max_step(run_id, kind)
        return value

    def auth_evidence(self, run_id: str, kind: CheckpointKind) -> tuple[int | None, list[str]]:
        evidence: tuple[int | None, list[str]] = self._handle.checkpoint_auth_evidence(run_id, kind)
        return evidence


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
        remote_max_candidate_bytes: int = DEFAULT_REMOTE_CANDIDATE_BYTES,
        remote_max_total_bytes: int = DEFAULT_REMOTE_TOTAL_BYTES,
        verify_after_save: bool = True,
        install_signal_handlers: bool = True,
        notice_source: preempt.NoticeSource | str | None = None,
        notice_interval: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
        lock_timeout: float = 30.0,
        require_trust: RequireTrust | None = None,
        allow_pt_format: bool | None = None,
        min_step: int | None = None,
        allow_weaker_trust: bool = False,
        restart_chain: bool = False,
        accept_key_change: bool = False,
    ) -> None:
        settings = load_settings()
        key = auth.load_key()  # environment only; never stored, logged or passed on
        if dir is None:
            dir = settings.checkpoint_dir
        if dir is None:
            raise ValueError("dir is required: pass dir= or set CVPILOT_CHECKPOINT_DIR")
        if remote is None:
            remote = settings.checkpoint_remote
        if not isinstance(allow_weaker_trust, bool):
            raise ValueError("allow_weaker_trust must be True or False")
        if not isinstance(restart_chain, bool) or not isinstance(accept_key_change, bool):
            raise ValueError("restart_chain and accept_key_change must be True or False")
        require_trust = auth.resolve_policy(  # type: ignore[assignment]
            require_trust, settings.ckpt_require_trust, key, allow_weaker=allow_weaker_trust
        )
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
        if remote_max_candidate_bytes < 1 or remote_max_total_bytes < remote_max_candidate_bytes:
            raise ValueError(
                "remote byte caps must be positive and the total at least one candidate"
            )
        self._remote_candidate_bytes = remote_max_candidate_bytes
        self._verify = verify_after_save
        self._lock_timeout = lock_timeout
        self._key = key
        for value in (min_step, settings.ckpt_min_step):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError("min_step must be a non-negative integer")
        # The operator's CVPILOT_CKPT_MIN_STEP and the constructor's argument both bind: the larger
        # wins. Only the explicit acknowledgement lets the constructor argument replace the setting.
        floors = [v for v in (min_step, settings.ckpt_min_step) if v is not None]
        if allow_weaker_trust and min_step is not None:
            floors = [min_step]
        self.min_step: int | None = max(floors) if floors else None
        self.allow_weaker_trust = allow_weaker_trust
        self.restart_chain = restart_chain  # explicit, never from the environment
        self.accept_key_change = accept_key_change  # explicit, never from the environment
        self._remote_max: dict[str, int | None] = {}
        self._anchor = _HandleAnchor(self._handle)
        weak = None if key is None else auth.weakness(key)
        if weak is not None:  # the reason only: the key itself is never in a log
            self._handle.emit("warning", "checkpoint_key_weak", reason=weak)
        self.require_trust: RequireTrust = require_trust
        self.allow_pt = allow_pt_format
        self._clock = clock
        self._dirs: dict[tuple[str, str], RunCheckpoints] = {}
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
                max_total_bytes=remote_max_total_bytes,
                sleep=remote_sleep,
                clock=remote_clock,
            )
        self._own = self._dir_for(self._handle.run_id)
        if allow_weaker_trust:
            self._emit(
                "warning", "checkpoint_weaker_trust_allowed", policy=self.require_trust,
                min_step=self.min_step,
            )  # fmt: skip
        self._lease_fd: int | None = self._own.hold_lease()  # dies with the process
        if self._lease_fd is None:  # no flock here (some FUSE and network file systems)
            self._emit("warning", "checkpoint_lease_unavailable", dir=str(self._root))
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
        self.installed_signals: list[int] = []
        self._poller: preempt.NoticePoller | None = None
        try:
            # everything that can fail comes before anything process-wide is changed
            source = (
                spot_notices.get(notice_source)()
                if isinstance(notice_source, str)
                else notice_source
            )
            poller = (
                None
                if source is None
                else preempt.NoticePoller(
                    source, self.flag, interval=notice_interval, report=self._report_poller
                )
            )
            if install_signal_handlers:
                self.installed_signals = self._guard.install()
                for signum, chained in self._guard.displaced:
                    self._emit(
                        "warning", "signal_handler_displaced",
                        signal=signal.Signals(signum).name, chained=chained,
                    )  # fmt: skip
            if poller is not None:
                self._poller = poller
                poller.start()
        except BaseException:
            self.close()  # handlers back, poller stopped, lease released: all or nothing
            raise

    # -- plumbing ---------------------------------------------------------------------------

    def _report_poller(self, event: str, **fields: Any) -> None:
        level = "info" if event in ("notice_poller_recovered",) else "warning"
        if event == "preemption_poller_stopped" and fields.get("reason") in ("notice", "stopped"):
            level = "info"
        self._emit(level, event, **fields)

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        owner = fields.pop("run_id", None)
        if owner is not None and owner != self._handle.run_id:
            fields["checkpoint_run_id"] = owner
        self._handle.emit(level, event, **fields)

    def _clamp(self, policy: str | None) -> RequireTrust:
        """The level a call may use: its own argument can only raise the manager's, unless the
        constructor said ``allow_weaker_trust=True``."""
        if policy is None:
            return self.require_trust
        if policy not in POLICIES:
            raise ValueError(f"require_trust must be one of {', '.join(POLICIES)}")
        if not self.allow_weaker_trust and auth.is_weaker(policy, self.require_trust):
            return self.require_trust
        return policy  # type: ignore[return-value]

    def _effective_policy(self, policy: str | None) -> RequireTrust:
        level = self._clamp(policy)
        if policy is not None and level != policy:
            self._emit(
                "warning", "checkpoint_trust_downgrade_ignored", requested=policy, effective=level
            )
        return level

    def _dir_for(self, run_id: str, policy: str | None = None) -> RunCheckpoints:
        level: Any = self._clamp(policy)
        found = self._dirs.get((run_id, level))
        if found is None:
            found = RunCheckpoints(
                self._root,
                run_id,
                create=run_id == self._handle.run_id,
                max_bytes=self._max_bytes,
                emit=self._emit,
                anchor=self._anchor,
                lock_timeout=self._lock_timeout,
                key=self._key,
                require_trust=level,
                allow_pt=self.allow_pt,
                accept_auth_downgrade=self.allow_weaker_trust,
                accept_key_change=self.accept_key_change,
            )
            self._dirs[(run_id, level)] = found
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
        """Stop the notice poller, put the previous signal handlers back and release the
        liveness lease."""
        if self._lease_fd is not None:
            try:
                os.close(self._lease_fd)
            except OSError:
                pass
            self._lease_fd = None
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
        finalize: bool = True,
    ) -> CheckpointManifest:
        started = time.perf_counter()
        try:
            data: bytes | Iterable[bytes] = self._serializer.dumps(self._build_state(kind, state))
        except Exception as exc:  # noqa: BLE001  # logged and re-raised; callers decide
            self._emit(
                "error", "checkpoint_save_failed", step=step, kind=kind,
                error_type=type(exc).__name__, exc_info=exc,
            )  # fmt: skip
            raise
        return self._publish(kind, step, data, self._format, metric, started, finalize=finalize)

    def _publish(
        self,
        kind: CheckpointKind,
        step: int,
        data: bytes | Iterable[bytes],
        format: CheckpointFormat,  # noqa: A002
        metric: tuple[str, float] | None,
        started: float,
        *,
        finalize: bool = True,
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
                floor_hint=self._floor_hint(kind),
                restart_chain=self.restart_chain,
                project=self._handle.meta.project,
                parent_run_id=self._handle.meta.parent_run_id,
            )
        except Exception as exc:  # noqa: BLE001  # logged and re-raised; callers decide
            self._emit(
                "error",
                "checkpoint_save_failed",
                step=step,
                kind=kind,
                error_type=type(exc).__name__,
                exc_info=exc,
            )
            raise
        if manifest.chain_reset:
            self._emit(
                "warning", "checkpoint_chain_reset", step=manifest.step, kind=kind,
                reason=manifest.chain_reset_reason,
            )  # fmt: skip
        self._emit(
            "info",
            "checkpoint_already_saved" if self._own.last_save_noop else "checkpoint_saved",
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
        if finalize:  # the preemption path defers both until the run is marked
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
            for e in self._own.inspect(kind, check_data=False, ctx=self._admit_ctx())
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
            known = self._remote_max.get(manifest.kind)
            if known is None or manifest.step > known:
                self._remote_max[manifest.kind] = manifest.step
            self._emit(
                "info", "checkpoint_uploaded", step=manifest.step, kind=manifest.kind,
                size_bytes=manifest.size_bytes, sha256_prefix=manifest.sha256[:12],
                remote=self._mirror.label,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )  # fmt: skip

    def _admit_ctx(self, accept_rollback_to: int | None = None) -> AdmitContext:
        """The one context every call into the gate uses (restore, list, verify, upload, best
        seeding): the operator's floor, the project binding, the lineage and the byte cap."""
        lineage = [self._handle.run_id, *self._handle.ancestors()]
        parents: dict[str, str | None] = {lineage[0]: self._handle.meta.parent_run_id}
        parents.update({a: b for a, b in zip(lineage[1:], lineage[2:], strict=False)})
        return AdmitContext(
            accept_rollback_to=accept_rollback_to,
            min_step=self.min_step,
            waive_min_step=self.allow_weaker_trust,
            max_candidate_bytes=self._remote_candidate_bytes,
            project=self._handle.meta.project,
            parents=parents,
        )

    def _floor_hint(self, kind: CheckpointKind) -> int | None:
        """A step known to have existed that no local file proves: ``min_step`` and the newest step
        the remote proves for this run (an admitted manifest with its data object). Under ``any`` a
        remote manifest that only chains to itself is advisory and never raises the floor: a bucket
        writer could otherwise plant a huge step and stop every save. A failed or empty lookup is
        not remembered."""
        hints = [self.min_step] if self.min_step is not None else []
        if self._mirror is not None:
            if kind not in self._remote_max:
                try:
                    found = self._mirror.admitted(self._own, kind, want=1)
                except (OSError, CheckpointUntrustedError):
                    self._emit(
                        "warning", "checkpoint_floor_unavailable", kind=kind, source="the remote"
                    )
                else:
                    step, level = found[0] if found else (None, None)
                    if (
                        step is not None
                        and self.require_trust == "any"
                        and level
                        not in (
                            "anchored",
                            "authenticated",
                        )
                    ):
                        self._emit("info", "checkpoint_remote_floor_ignored", kind=kind, step=step)
                        step = None
                    if step is not None:
                        self._remote_max[kind] = step
            remote = self._remote_max.get(kind)
            if remote is not None:
                hints.append(remote)
        return max(hints, default=None)

    def _record(self, manifest: CheckpointManifest, digest: str) -> None:
        """Store the manifest digest (the external anchor). Under ``any`` a failure weakens trust
        and is logged; under ``anchored`` or ``authenticated`` an unrecorded checkpoint would be
        refused by the very next restore (``not_anchored``), so the save fails before the pointer
        moves and the step is reported as unanchored."""
        try:
            self._handle.record_checkpoint(manifest, digest)
        except Exception as exc:  # noqa: BLE001  # the index must never break a verified save
            self._emit(
                "warning", "checkpoint_record_failed", step=manifest.step,
                kind=manifest.kind, error_type=type(exc).__name__,
            )  # fmt: skip
            if self.require_trust != "any":
                self._emit("error", "checkpoint_unanchored", step=manifest.step, kind=manifest.kind)
                raise CheckpointError(
                    f"checkpoint {manifest.step} was written but could not be recorded in the run "
                    f"store ({type(exc).__name__}); under {self.require_trust} it would be "
                    "refused, so the pointer was not moved"
                ) from None

    def _confirmed(self, manifest: CheckpointManifest) -> bool:
        """Retention may rely on an upload only if the marker says so AND the remote still holds
        exactly that checkpoint (without a key a marker is only a file anyone could write)."""
        return bool(
            self._mirror is not None
            and self._own.is_uploaded(manifest)
            and self._mirror.confirm(self._own, manifest)
        )

    def _retain(self, kind: CheckpointKind) -> None:
        keep = self.keep_last if kind == "last" else self.keep_best
        try:
            self._own.prune(
                kind,
                keep=keep,
                mode=self.mode,
                is_confirmed=self._confirmed if self._mirror is not None else None,
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
        """Bring a checkpoint file a framework wrote (Lightning ``last.ckpt``, a YOLO
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

    def _lineage_best(self) -> float | None:
        """The best monitored value among this run and the runs it resumed, read from checkpoints
        that pass the gate. Another metric or another mode is not comparable and is ignored."""
        found: list[float] = []
        for run_id in [self._handle.run_id, *self._handle.ancestors()]:
            try:
                newest = next(
                    iter(self._dir_for(run_id).iter_valid("best", ctx=self._admit_ctx())), None
                )
            except CheckpointError as exc:
                self._emit(
                    "warning", "best_seed_failed", checkpoint_run_id=run_id,
                    error_type=type(exc).__name__,
                )  # fmt: skip
                continue
            manifest = None if newest is None else newest.manifest
            if (
                manifest is not None
                and manifest.metric_value is not None
                and manifest.metric_key == self.monitor
                and manifest.metric_mode in (None, self.mode)
            ):
                found.append(manifest.metric_value)
        if not found:
            return None
        return max(found) if self.mode == "max" else min(found)

    def _improves(self, value: float) -> bool:
        if not self._best_loaded:
            self._best_loaded = True
            self._best_value = self._lineage_best()
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
                manifest = self._save("last", step, state, finalize=False)
            except Exception as exc:  # noqa: BLE001  # nothing may escape into the training loop
                reason = type(exc).__name__
        if manifest is None and reason != "step_already_saved":
            self._emit("warning", "preemption_save_incomplete", step=step, reason=reason)
        self._handled = True
        self._emit(
            "info", "run_preempted", signal=self.flag.source, step=step, saved=manifest is not None
        )
        try:  # marked as soon as the local checkpoint is safe: the window is short
            self._run.finish("preempted")
        except RunClosedError:
            pass
        except Exception as exc:  # noqa: BLE001  # nothing may escape into the training loop
            self._emit("warning", "run_finish_failed", error_type=type(exc).__name__)
        if manifest is not None:
            self._finish_preemption()
        return manifest

    def _finish_preemption(self) -> None:
        """Upload and prune with whatever is left of the grace window, never past it."""
        remaining = self.grace_seconds - (self._clock() - self.flag.since)
        if self._mirror is not None:
            if remaining <= 0:
                self._emit("warning", "preemption_upload_deferred", reason="grace_expired")
                return
            try:
                with self._mirror.budget(remaining):
                    self._mirror_pending()
            except Exception as exc:  # noqa: BLE001  # nothing may escape into the training loop
                self._emit("warning", "preemption_upload_deferred", reason=type(exc).__name__)
                return
            if self.grace_seconds - (self._clock() - self.flag.since) <= 0:
                self._emit("warning", "preemption_upload_deferred", reason="grace_expired")
                return
        try:
            self._retain("last")
        except Exception as exc:  # noqa: BLE001  # nothing may escape into the training loop
            self._emit("warning", "checkpoint_prune_failed", error_type=type(exc).__name__)

    # -- restoring --------------------------------------------------------------------------

    def restore(
        self,
        kind: CheckpointKind = "last",
        *,
        map_location: str = "cpu",
        run_ids: Sequence[str] | None = None,
        discover: bool = False,
        require_trust: RequireTrust | None = None,
        accept_rollback_to: int | None = None,
    ) -> Restored | None:
        """The newest checkpoint of ``kind`` that verifies and loads, searching this run first
        and then the runs it resumed (nearest first). Corrupt, truncated, forged, replaced or
        symlinked files are skipped and counted, never loaded; ``None`` when nothing verifies.

        ``require_trust`` can raise the manager's policy for this call but never lower it (a weaker
        request is ignored and logged as ``checkpoint_trust_downgrade_ignored``; only the
        constructor's ``allow_weaker_trust=True`` permits that, logged once at construction).
        ``min_step`` (also ``CVPILOT_CKPT_MIN_STEP``) is a floor no call can lower. Under
        ``anchored`` or ``authenticated`` a candidate that lacks its own proof is never substituted
        at a lower level: if candidates exist but none qualifies, ``CheckpointUntrustedError`` is
        raised (event ``checkpoint_untrusted``). The strict policies also look for rollback: if the
        newest checkpoint that qualifies is older than one the tag-checked pointer, the run store
        or ``min_step`` says existed, ``CheckpointUntrustedError`` with reason ``rolled_back`` is
        raised; ``accept_rollback_to=N`` accepts a restore at step N or later on purpose (logged;
        0 accepts starting from nothing). A newer checkpoint that is still present but damaged is
        corruption, not rollback.

        ``discover=True`` is for a restarted script on a node that has neither the run store nor
        the checkpoint directory of the earlier run (a new run has a new id and no lineage): when
        the lineage yields nothing, the newest ``MAX_DISCOVERED_RUNS`` run prefixes on the remote
        are listed, their candidates are ordered by step (newest run first on a tie) and each goes
        through the same gate as any other remote checkpoint. It needs a remote.
        """
        if discover and self._mirror is None:
            raise ValueError("discover=True needs a remote (remote= or CVPILOT_CHECKPOINT_REMOTE)")
        policy = self._effective_policy(require_trust)
        order = (
            list(run_ids)
            if run_ids is not None
            else [self._handle.run_id, *self._handle.ancestors()]
        )
        if self._mirror is not None:
            self._mirror.begin_restore()
        ctx = self._admit_ctx(accept_rollback_to)
        skipped: list[Skipped] = []
        for run_id in order:
            found = self._restore_local(run_id, kind, map_location, skipped, policy, ctx)
            if found is None and self._mirror is not None:
                found = self._restore_remote(run_id, kind, map_location, skipped, policy, ctx)
            if found is None:
                continue
            found.policy, found.weaker_allowed = policy, self.allow_weaker_trust
            self._announce(found, run_id, kind, len(skipped), policy)
            self.last_restore = found
            return found
        if discover:
            found = self._restore_discovered(order, kind, map_location, skipped, policy, ctx)
            if found is not None:
                found.policy, found.weaker_allowed = policy, self.allow_weaker_trust
                self._announce(found, found.run_id, kind, len(skipped), policy)
                self.last_restore = found
                return found
        self.last_restore = None
        if self.min_step is not None and not self.allow_weaker_trust:
            skipped.append(Skipped(None, kind, kind, "rolled_back", self.min_step))  # min_step
        failure = gate_failure(skipped, policy)
        if failure is not None:
            reason, message = failure
            rejected = [s for s in skipped if s.reason in TRUST_REASONS]
            self._emit(
                "error", "checkpoint_untrusted", required=policy, reason=reason,
                rejected=len(rejected), reasons=sorted({s.reason for s in rejected}),
            )  # fmt: skip
            raise CheckpointUntrustedError(message, reason=reason)
        self._emit("info", "checkpoint_restore_empty", dir=str(self._root), skipped=len(skipped))
        return None

    def _announce(
        self, found: Restored, run_id: str, kind: CheckpointKind, skipped: int, policy: str
    ) -> None:
        manifest = found.manifest
        if kind == "last":  # seed the save schedule: the next save is due one interval later
            self._last_step = max(self._last_step or 0, manifest.step)
            self._last_time = self._clock()
        self._emit(
            "info", "checkpoint_restored", run_id=run_id, step=manifest.step, kind=kind,
            size_bytes=manifest.size_bytes, sha256_prefix=manifest.sha256[:12],
            skipped=skipped, trust_level=found.trust,
        )  # fmt: skip
        if self._key is not None and policy == "any" and found.trust != "authenticated":
            self._emit(
                "warning", "checkpoint_untrusted_accepted", step=manifest.step, kind=kind,
                trust_level=found.trust, reason="a key is configured but the policy is any",
            )  # fmt: skip

    def _restore_local(
        self,
        run_id: str,
        kind: CheckpointKind,
        map_location: str,
        skipped: list[Skipped],
        policy: str | None = None,
        ctx: AdmitContext | None = None,
    ) -> Restored | None:
        found = self._dir_for(run_id, policy)
        for verified in found.iter_valid(kind, load=True, ctx=ctx):
            try:
                assert verified.data is not None  # noqa: S101  # load=True always sets it
                state = get_serializer(verified.manifest.format).loads(
                    verified.data, map_location=map_location
                )
                if "rng" in state:
                    try:
                        validate_rng(state["rng"])
                    except ValueError:
                        raise CheckpointCorruptError(
                            "the checkpoint's rng state is malformed", reason="unloadable"
                        ) from None
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
            return Restored(state, verified.manifest, run_id, skipped, verified.trust)
        skipped.extend(found.skipped)
        return None

    def _restore_remote(
        self,
        run_id: str,
        kind: CheckpointKind,
        map_location: str,
        skipped: list[Skipped],
        policy: str,
        ctx: AdmitContext,
    ) -> Restored | None:
        """Nothing usable locally: fetch the newest remote checkpoints one by one (verified before
        they are kept) and restore through the same local checks. ``present`` receives the steps
        the remote lists, which the rollback check counts as still existing."""
        assert self._mirror is not None  # noqa: S101  # the caller checked
        found = self._dir_for(run_id, policy)
        try:
            candidates = self._mirror.admitted(found, kind, ctx, want=MAX_REMOTE_CANDIDATES)
        except CheckpointUntrustedError:  # an OSError subclass: caught first on purpose
            skipped.append(Skipped(None, kind, f"remote:{kind}", "remote_untrusted"))
            return None
        except OSError as exc:
            self._emit(
                "warning", "checkpoint_download_failed", kind=kind, error_type=type(exc).__name__
            )
            return None
        for step, _level in candidates:
            if not self._fetch_one(found, run_id, kind, step, skipped, ctx):
                continue
            restored = self._restore_local(run_id, kind, map_location, skipped, policy, ctx)
            if restored is not None:
                return restored
        return None

    def _restore_discovered(
        self,
        known: Sequence[str],
        kind: CheckpointKind,
        map_location: str,
        skipped: list[Skipped],
        policy: str,
        ctx: AdmitContext,
    ) -> Restored | None:
        """Candidates from the newest remote runs outside ``known``, best step first."""
        assert self._mirror is not None  # noqa: S101  # the caller checked
        try:
            run_ids = [r for r in self._mirror.list_runs() if r not in set(known)]
            run_ids = run_ids[:MAX_DISCOVERED_RUNS]
        except OSError as exc:
            self._emit(
                "warning", "checkpoint_download_failed", kind=kind, error_type=type(exc).__name__
            )
            return None
        found_all: list[tuple[int, str]] = []
        for run_id in run_ids:
            try:
                for step, _level in self._mirror.admitted(
                    self._dir_for(run_id, policy), kind, ctx, want=3
                ):
                    found_all.append((step, run_id))
            except CheckpointUntrustedError:  # an OSError subclass: caught first on purpose
                skipped.append(Skipped(None, kind, f"remote:{run_id}", "remote_untrusted"))
            except OSError:
                continue
        for step, run_id in sorted(found_all, reverse=True)[:MAX_REMOTE_CANDIDATES]:
            found = self._dir_for(run_id, policy)
            if self._fetch_one(found, run_id, kind, step, skipped, ctx):
                restored = self._restore_local(run_id, kind, map_location, skipped, policy, ctx)
                if restored is not None:
                    return restored
        return None

    def _fetch_one(
        self,
        found: RunCheckpoints,
        run_id: str,
        kind: CheckpointKind,
        step: int,
        skipped: list[Skipped],
        ctx: AdmitContext,
    ) -> bool:
        """Fetch one remote candidate through the gate; False (counted, logged) when refused."""
        assert self._mirror is not None  # noqa: S101  # the caller checked
        try:
            self._mirror.fetch(found, kind, step, ctx)
        except (CheckpointCorruptError, OSError) as exc:
            reason = exc.reason if isinstance(exc, CheckpointCorruptError) else "remote_unavailable"
            skipped.append(Skipped(step, kind, f"remote:{kind}/{step}", reason))
            event = (
                "checkpoint_untrusted" if reason in TRUST_REASONS else "checkpoint_skipped_corrupt"
            )
            self._emit(
                "warning", event, run_id=run_id, step=step, kind=kind,
                reason=reason, remote=self._mirror.label,
            )  # fmt: skip
            return False
        return True

    @overload
    def restore_latest_valid(
        self,
        *,
        map_location: str = ...,
        run_ids: Sequence[str] | None = ...,
        discover: bool = ...,
        require_trust: RequireTrust | None = ...,
        accept_rollback_to: int | None = ...,
        with_trust: Literal[False] = ...,
    ) -> dict[str, Any] | None: ...

    @overload
    def restore_latest_valid(
        self,
        *,
        map_location: str = ...,
        run_ids: Sequence[str] | None = ...,
        discover: bool = ...,
        require_trust: RequireTrust | None = ...,
        accept_rollback_to: int | None = ...,
        with_trust: Literal[True],
    ) -> Restored | None: ...

    def restore_latest_valid(
        self,
        *,
        map_location: str = "cpu",
        run_ids: Sequence[str] | None = None,
        discover: bool = False,
        require_trust: RequireTrust | None = None,
        accept_rollback_to: int | None = None,
        with_trust: bool = False,
    ) -> dict[str, Any] | Restored | None:
        """The state of the newest valid ``last`` checkpoint, or ``None`` on a fresh start.

        ``require_trust`` can raise the policy for this call, never lower it. ``run_ids`` names
        the runs to search (default: this run and the runs it resumed); ``discover=True`` also
        looks through the newest runs on the remote (see ``restore``). By default only the state
        is returned (as before); with ``with_trust=True`` the whole ``Restored`` comes back, so
        the caller sees the trust level that was achieved (``Restored.trust``).
        """
        restored = self.restore(
            "last", map_location=map_location, run_ids=run_ids, discover=discover,
            require_trust=require_trust, accept_rollback_to=accept_rollback_to,
        )  # fmt: skip
        if restored is None:
            return None
        return restored if with_trust else restored.state

    # -- inspection -------------------------------------------------------------------------

    def list(self) -> list[Entry]:
        """Every checkpoint of this run (both kinds, newest first) with its verification result."""
        return [
            *self._own.inspect("last", ctx=self._admit_ctx()),
            *self._own.inspect("best", ctx=self._admit_ctx()),
        ]

    def verify(self, manifest: CheckpointManifest) -> bool:
        """Re-verify one checkpoint against the directory (size, structure and SHA-256)."""
        found = self._dir_for(manifest.run_id)
        entries = [
            e
            for e in found.inspect(manifest.kind, ctx=self._admit_ctx())
            if e.step == manifest.step
        ]
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
