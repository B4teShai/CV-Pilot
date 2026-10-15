"""One run's checkpoint directory: atomic save, trusted newest-first walk, retention.

Layout (``<root>/<run_id>/``)::

    LATEST, BEST                        pointers: manifest name, its digest and chain value
    last/step_<step:09d>.safetensors    the data file (or .pt), 0444
    last/step_<step:09d>.manifest.json  the CheckpointManifest, 0444, written before the rename
    last/step_<step:09d>.remote.json    marker: this checkpoint is uploaded and verified remotely
    best/...                            the same for the best checkpoints

Saving (under an exclusive ``flock``): remove debris of a killed save, write the data to an
exclusive temp file (0600, fsync, 0444), write the manifest the same way, rename manifest then
data into place, fsync the directory, re-read and re-hash the data (``verify``), let the caller
record the digest elsewhere (``before_publish``), and only then rewrite the pointer. A SIGKILL
at any point leaves either the previous state or a complete new checkpoint; the pointer is the
last thing that moves.

Walking back (``iter_valid``) never trusts a name. A manifest must parse, agree with its file
name and run, and have a chain value that recomputes. It must be anchored: it is the pointer's
target with the pointer's digest, or the run store holds its digest, or the chain of ``prev``
links connects it to the pointer target, and each trusted manifest vouches for its immediate
predecessor by digest. Then the data file must be a regular file (never a symlink) of exactly
the manifest's size whose structure parses and whose SHA-256 matches. Anything else is skipped,
counted and logged as ``checkpoint_skipped_corrupt`` with a reason, and the walk goes on to the
previous checkpoint. Nothing is signed: someone who can rewrite the directory *and* the run
store can forge a checkpoint; the store digest is the external anchor that closes the gap for
anyone who can only reach the directory (see ADR-0006).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import ValidationError

from cvpilot.ckpt import auth, fsops
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointPathError,
    CheckpointStepError,
    CheckpointUntrustedError,
)
from cvpilot.core.hashing import checkpoint_chain, pointer_digest
from cvpilot.core.schema import (
    CHECKPOINT_SUFFIX,
    GENESIS_DIGEST,
    MAX_MANIFEST_BYTES,
    MAX_STEP,
    CheckpointFormat,
    CheckpointKind,
    CheckpointManifest,
    CheckpointPointer,
    RequireTrust,
    TrustLevel,
)
from cvpilot.core.ulid import is_valid_ulid
from cvpilot.logging import get_logger

DEFAULT_MAX_BYTES = 8 * 1024 * 1024 * 1024
"""Largest checkpoint file written or read by default (8 GiB)."""
DEFAULT_REMOTE_CANDIDATE_BYTES = 4 * 1024 * 1024 * 1024
"""Most bytes one remote candidate may be before it is refused unread (4 GiB)."""
DEFAULT_REMOTE_TOTAL_BYTES = 8 * 1024 * 1024 * 1024
"""Most bytes one restore may download in all (8 GiB)."""
DEFAULT_MAX_ENTRIES = 10_000
"""Most files examined in one kind directory; bounds the work of a walk."""
KINDS: tuple[CheckpointKind, ...] = ("last", "best")
_POINTER_NAME: dict[str, str] = {"last": "LATEST", "best": "LATEST_BEST"}
_DATA = re.compile(r"step_([0-9]{9,19})\.(safetensors|pt)")
_MANIFEST = re.compile(r"step_([0-9]{9,19})\.manifest\.json")
_MARKER = re.compile(r"step_([0-9]{9,19})\.remote\.json")
_SHA = re.compile(r"[0-9a-f]{64}")
TRUST_REASONS = frozenset(
    {
        "auth_invalid",
        "auth_missing",
        "not_anchored",
        "anchor_error",
        "store_mismatch",
        "rolled_back",
        "checkpoint_missing",
        "binding_mismatch",
    }
)
"""Reasons that mean "not trusted at the required level" (event ``checkpoint_untrusted``)."""
POLICIES = ("authenticated", "anchored", "any")

Emit = Callable[..., None]
Anchor = Callable[[str, CheckpointKind, int], "str | None"]
"""``anchor(run_id, kind, step)``: the manifest digest recorded outside the directory, or None."""


def data_name(step: int, format: CheckpointFormat) -> str:
    return f"step_{step:09d}.{CHECKPOINT_SUFFIX[format]}"


def manifest_name(step: int) -> str:
    return f"step_{step:09d}.manifest.json"


def marker_name(step: int) -> str:
    return f"step_{step:09d}.remote.json"


@dataclass(frozen=True)
class Skipped:
    """A checkpoint the walk refused, and why (``reason`` is a short constant)."""

    step: int | None
    kind: str
    name: str
    reason: str
    floor: int | None = None  # for rolled_back / checkpoint_missing: the step known to have existed


@dataclass(frozen=True)
class Verified:
    """A checkpoint that passed every check; ``data`` is the verified bytes when loaded."""

    manifest: CheckpointManifest
    manifest_sha256: str
    data: bytes | None = None
    trust: TrustLevel = "local_chain_only"


@dataclass(frozen=True)
class Entry:
    """One line of ``inspect``: a checkpoint (or debris) and its verification result."""

    name: str
    step: int | None
    kind: str
    status: Literal["valid", "corrupt"]
    reason: str | None = None
    manifest: CheckpointManifest | None = None
    trust: TrustLevel | None = None


@dataclass(frozen=True)
class PruneResult:
    deleted: list[CheckpointManifest] = field(default_factory=list)
    kept: list[CheckpointManifest] = field(default_factory=list)
    held_back: bool = False
    """True when nothing was deleted because the newest checkpoint is not confirmed remotely."""


@dataclass
class _Cand:
    name: str
    step: int | None
    manifest: CheckpointManifest | None = None
    digest: str | None = None
    reason: str | None = None
    tag_state: str = "nokey"  # nokey | absent | valid | invalid
    trust: TrustLevel = "local_chain_only"


@dataclass(frozen=True)
class AdmitContext:
    """What a caller adds to the gate: steps known to exist elsewhere (a remote listing), a
    deliberate rollback acceptance, an external high-water mark and a per-candidate byte cap."""

    present: frozenset[int] = frozenset()
    accept_rollback_to: int | None = None
    min_step: int | None = None
    waive_min_step: bool = False  # only a constructor-level acknowledgement sets this
    max_candidate_bytes: int | None = None


class Admission:
    """The single decision: may this candidate be returned, fetched or counted as valid?

    ``decide`` combines, in this order, the trust verdict of the manifest (policy level,
    authentication, run-store anchor, binding), whether the format is allowed, the size caps and
    the rollback floor. Restore (local and remote), resume, the command line, retention and the
    remote fetch and install all call it; none has its own notion of "valid".

    The floor is the newest step known to have been saved (tag-checked pointer, run store,
    ``min_step``). A candidate older than the floor is admitted only when a trusted manifest at
    or above the floor is still present (so the newer one is merely damaged: corruption, counted
    and skipped, not rollback) or when the caller accepted the rollback on purpose. A store that
    cannot answer denies everything under the strict policies, with reason ``anchor_error``.
    """

    def __init__(
        self, rc: RunCheckpoints, kind: CheckpointKind, trusted: set[int], ctx: AdmitContext | None
    ) -> None:
        self._rc, self._kind = rc, kind
        self._ctx = ctx or AdmitContext()
        self._trusted = trusted | set(self._ctx.present)
        self._error: str | None = None
        known: int | None = None
        try:
            known = rc._floor(kind)
        except CheckpointUntrustedError as exc:
            self._error = exc.reason
        self._known = known
        marks = [m for m in (known, self._ctx.min_step) if m is not None]
        self.floor: int | None = max(marks) if marks else None
        self._accepted_logged = False

    def _rolled_back(self, step: int | None, floor: int | None = None) -> str | None:
        floor = self.floor if floor is None else floor
        mark = self._ctx.min_step
        if mark is not None and not self._ctx.waive_min_step and (step is None or step < mark):
            return "rolled_back"  # the operator's floor: a call cannot accept its way below it
        accept = self._ctx.accept_rollback_to
        if accept is not None and (accept == 0 if step is None else step >= accept):
            if not self._accepted_logged:
                self._accepted_logged = True
                self._rc._emit(
                    "warning", "checkpoint_rollback_accepted", step=step, kind=self._kind,
                    floor=floor, accepted_to=accept,
                )  # fmt: skip
            return None
        return "rolled_back"

    def _warn_progress_lost(self) -> None:
        """Under ``any`` rollback is not enforced, but an empty result must not look like a fresh
        start when the run store says checkpoints existed: say so, with the floor."""
        if self._rc.require_trust != "any":
            return
        try:
            stored = self._rc._anchor_max(self._kind)
        except CheckpointUntrustedError:
            return
        if stored:
            self._rc._emit(
                "warning", "checkpoint_progress_lost", kind=self._kind, floor=stored,
                reason="rolled_back", policy="any",
            )  # fmt: skip

    def decide(self, cand: _Cand | None) -> str | None:
        """None when ``cand`` may be used, else the reason it may not. ``None`` as the candidate
        asks whether finding nothing at all is acceptable (it is not when checkpoints are known to
        have existed)."""
        if self._error is not None:
            return self._error
        if cand is None:  # the external mark is checked by the caller across the whole lineage
            if self._known is None:
                self._warn_progress_lost()
                return None  # nothing is known to have existed: a genuine fresh start
            verdict = self._rolled_back(None, self._known)
            if verdict is None:
                return None  # starting over was accepted on purpose (logged)
            # the newest is still here but nothing verifies, or it is gone: never a fresh start
            damaged = any(t >= self._known for t in self._trusted)
            return "checkpoint_missing" if damaged else verdict
        if cand.reason is not None:
            return cand.reason
        manifest = cand.manifest
        if manifest is None or cand.digest is None or cand.step is None:
            return "manifest_invalid"
        if manifest.format == "pt" and not self._rc.allow_pt:
            return "pt_not_allowed"
        limit = self._ctx.max_candidate_bytes
        if manifest.size_bytes > self._rc.max_bytes or (
            limit is not None and manifest.size_bytes > limit
        ):
            return "too_large"
        if self.floor is not None and cand.step < self.floor:
            if any(t >= self.floor for t in self._trusted):
                return None  # the newer one is still here (damaged): corruption, not rollback
            return self._rolled_back(cand.step)
        return None


def gate_failure(skipped: Iterable[Skipped], policy: str) -> tuple[str, str] | None:
    """``(reason, message)`` when the gate, not damage, is why nothing could be returned: a
    rollback or nothing that verifies although progress is known, a failing run store, or (under a
    strict policy) candidates that lack the required proof. The caller logs it and raises
    ``CheckpointUntrustedError``."""
    items = list(skipped)
    reasons = [s.reason for s in items]
    floors = [s.floor for s in items if s.floor is not None]
    known = f"step {max(floors)}" if floors else "an earlier step"
    if "rolled_back" in reasons:
        return "rolled_back", (
            f"checkpoints were rolled back: {known} is known to have existed and nothing at or "
            "above it is left (restore it, or accept the rollback on purpose)"
        )
    if "checkpoint_missing" in reasons:
        return "checkpoint_missing", (
            f"{known} is known to have existed but no checkpoint at or above it verifies "
            "(restore the data, or accept starting over on purpose with accept_rollback_to)"
        )
    if "anchor_error" in reasons and policy != "any":
        return "anchor_error", (
            "the run store could not be consulted, so the required trust level cannot be proven"
        )
    untrusted = [r for r in reasons if r in TRUST_REASONS]
    if untrusted and policy != "any":
        return "untrusted", (
            f"{len(untrusted)} checkpoint(s) exist but none reaches the required trust level "
            f"{policy!r}; nothing was restored (see CVPILOT_CKPT_REQUIRE_TRUST)"
        )
    return None


def _check_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= MAX_STEP:
        raise CheckpointPathError(f"step must be an integer in [0, {MAX_STEP}]")
    return step


def _check_kind(kind: object) -> CheckpointKind:
    if kind not in KINDS:
        raise CheckpointPathError("kind must be 'last' or 'best'")
    return kind


def list_runs(root: Path, *, limit: int = 100_000) -> list[str]:
    """Run ids (valid ULIDs only) that have a directory below ``root``, oldest first."""
    fd = fsops.open_root(root, create=False)
    try:
        return [n for n in fsops.listdir(fd, limit=limit) if is_valid_ulid(n)]
    finally:
        import os

        os.close(fd)


class RunCheckpoints:
    """The checkpoints of one run below a checkpoint root. See the module docstring."""

    def __init__(
        self,
        root: Path | str,
        run_id: str,
        *,
        create: bool = False,
        max_bytes: int = DEFAULT_MAX_BYTES,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        emit: Emit | None = None,
        anchor: Anchor | None = None,
        lock_timeout: float = 30.0,
        key: bytes | None = None,
        require_trust: RequireTrust = "any",
        allow_pt: bool = False,
    ) -> None:
        if require_trust not in POLICIES:
            raise CheckpointError(f"require_trust must be one of {', '.join(POLICIES)}")
        if require_trust == "authenticated" and key is None:
            raise CheckpointError(
                f"require_trust=authenticated needs the key in {auth.KEY_ENV} (never stored)"
            )
        if not is_valid_ulid(run_id):
            raise CheckpointPathError(f"{run_id[:40]!r} is not a valid run id (26-character ULID)")
        if max_bytes < 1 or max_entries < 1:
            raise CheckpointPathError("max_bytes and max_entries must be positive")
        fsops.require_posix()
        self._root = Path(root)
        self.run_id = run_id
        self.max_bytes = max_bytes
        self.max_entries = max_entries
        self._emit_fn = emit
        self._anchor = anchor
        self._lock_timeout = lock_timeout
        self._key = key
        self.require_trust: RequireTrust = require_trust
        self.allow_pt = allow_pt
        self.skipped: list[Skipped] = []
        self._verified_files: dict[tuple[str, str], tuple[tuple[int, ...], str]] = {}
        try:
            self.path = Path(root).expanduser().resolve() / run_id
        except (OSError, RuntimeError) as exc:
            raise CheckpointPathError(f"cannot resolve the checkpoint directory: {exc}") from exc
        if create:
            with self._open_run(create=True):
                pass

    # -- plumbing ---------------------------------------------------------------------------

    def _emit(self, level: str, event: str, **fields: Any) -> None:
        if self._emit_fn is not None:
            self._emit_fn(level, event, run_id=self.run_id, **fields)
        else:
            getattr(get_logger("cvpilot.ckpt"), level)(event, run_id=self.run_id, **fields)

    @contextlib.contextmanager
    def _open_run(self, *, create: bool) -> Iterator[int | None]:
        """A descriptor for the run directory, or None when it does not exist (read only)."""
        import os

        if not create and not self.path.parent.is_dir():
            yield None  # a node that has never saved: nothing to find, nothing to fail
            return
        root_fd = fsops.open_root(self._root, create=create)
        run_fd: int | None = None
        try:
            try:
                run_fd = fsops.open_subdir(root_fd, self.run_id, create=create)
            except CheckpointCorruptError as exc:
                if exc.reason != "missing" or create:
                    raise
            yield run_fd
        finally:
            if run_fd is not None:
                os.close(run_fd)
            os.close(root_fd)

    def _open_kind(self, run_fd: int, kind: str, *, create: bool) -> int | None:
        try:
            return fsops.open_subdir(run_fd, kind, create=create)
        except CheckpointCorruptError as exc:
            if exc.reason == "missing":
                return None
            raise

    @contextlib.contextmanager
    def _locked(self) -> Iterator[int]:
        with self._open_run(create=True) as run_fd:
            assert run_fd is not None  # noqa: S101  # create=True never yields None
            with fsops.lock(run_fd, timeout=self._lock_timeout):
                yield run_fd

    def _skip(
        self, step: int | None, kind: str, name: str, reason: str, floor: int | None = None
    ) -> None:
        self.skipped.append(Skipped(step, kind, name, reason, floor))
        event = "checkpoint_untrusted" if reason in TRUST_REASONS else "checkpoint_skipped_corrupt"
        extra = {} if floor is None else {"floor": floor}
        self._emit("warning", event, step=step, kind=kind, reason=reason, **extra)

    # -- manifests and pointers ---------------------------------------------------------------

    def _list(self, kind_fd: int) -> list[str]:
        return fsops.listdir(kind_fd, limit=self.max_entries)

    def _load_manifest(self, kind_fd: int, kind: str, name: str, step: int) -> _Cand:
        if name != manifest_name(step):
            return _Cand(name=name, step=step, reason="name_mismatch")
        try:
            raw = fsops.read_regular(kind_fd, name, max_bytes=MAX_MANIFEST_BYTES)
        except CheckpointCorruptError:
            return _Cand(name=name, step=step, reason="manifest_unreadable")
        return self._cand_from_bytes(raw, kind, name, step)

    def _cand_from_bytes(self, raw: bytes, kind: str, name: str, step: int) -> _Cand:
        """Parse and self-check manifest bytes: schema, agreement with name and run, chain, tag."""
        cand = _Cand(name=name, step=step)
        try:
            manifest = CheckpointManifest.model_validate_json(raw)
        except (ValidationError, ValueError):
            cand.reason = "manifest_invalid"
            return cand
        if (
            manifest.run_id != self.run_id
            or manifest.kind != kind
            or manifest.step != step
            or not manifest.path.startswith(f"{kind}/step_{step:09d}.")
        ):
            cand.reason = "name_mismatch"
            return cand
        expected = checkpoint_chain(
            manifest.prev_chain_sha256,
            manifest.run_id,
            manifest.kind,
            manifest.step,
            manifest.sha256,
            manifest.size_bytes,
        )
        if expected != manifest.chain_sha256:
            cand.reason = "chain_mismatch"
            return cand
        cand.manifest = manifest
        cand.digest = hashlib.sha256(raw).hexdigest()
        if self._key is not None:
            if manifest.auth_tag is None:
                cand.tag_state = "absent"
            else:
                valid = auth.verify(
                    self._key,
                    auth.MANIFEST_LABEL,
                    self.run_id,
                    self._manifest_bytes(manifest),
                    manifest.auth_tag,
                )
                cand.tag_state = "valid" if valid else "invalid"
        return cand

    @staticmethod
    def _manifest_bytes(manifest: CheckpointManifest) -> bytes:
        """The canonical bytes a manifest tag covers: the record without its own tag."""
        return manifest.model_copy(update={"auth_tag": None}).model_dump_json().encode()

    @staticmethod
    def _pointer_bytes(pointer: CheckpointPointer) -> bytes:
        return pointer.model_copy(update={"auth_tag": None}).model_dump_json().encode()

    def _candidates(self, kind_fd: int, kind: str) -> list[_Cand]:
        cands: list[_Cand] = []
        for name in self._list(kind_fd):
            match = _MANIFEST.fullmatch(name)
            if match is not None:
                cands.append(self._load_manifest(kind_fd, kind, name, int(match.group(1))))
        cands.sort(key=lambda c: c.step if c.step is not None else -1, reverse=True)
        return cands

    def read_pointer(self, kind: CheckpointKind) -> CheckpointPointer | None:
        """The ``LATEST``/``BEST`` pointer if it exists, parses and agrees with itself."""
        with self._open_run(create=False) as run_fd:
            if run_fd is None:
                return None
            return self._pointer(run_fd, kind)

    def _pointer(self, run_fd: int, kind: str) -> CheckpointPointer | None:
        try:
            raw = fsops.read_regular(run_fd, _POINTER_NAME[kind], max_bytes=MAX_MANIFEST_BYTES)
            pointer = CheckpointPointer.model_validate_json(raw)
        except (CheckpointCorruptError, ValidationError, ValueError):
            return None
        expected = pointer_digest(
            pointer.run_id,
            pointer.kind,
            pointer.step,
            pointer.manifest,
            pointer.manifest_sha256,
            pointer.chain_sha256,
        )
        if (
            pointer.pointer_sha256 != expected
            or pointer.run_id != self.run_id
            or pointer.kind != kind
            or pointer.manifest != f"{kind}/{manifest_name(pointer.step)}"
        ):
            return None
        if self._key is not None and not auth.verify(
            self._key,
            auth.POINTER_LABEL,
            self.run_id,
            self._pointer_bytes(pointer),
            pointer.auth_tag,
        ):
            return None  # with a key, an untagged or wrongly tagged pointer is not believed
        return pointer

    def _write_pointer(self, run_fd: int, manifest: CheckpointManifest, digest: str) -> None:
        name = f"{manifest.kind}/{manifest_name(manifest.step)}"
        pointer = CheckpointPointer(
            run_id=self.run_id,
            kind=manifest.kind,
            step=manifest.step,
            manifest=name,
            manifest_sha256=digest,
            chain_sha256=manifest.chain_sha256,
            pointer_sha256=pointer_digest(
                self.run_id, manifest.kind, manifest.step, name, digest, manifest.chain_sha256
            ),
        )
        if self._key is not None:
            pointer = pointer.model_copy(
                update={
                    "auth_tag": auth.tag(
                        self._key, auth.POINTER_LABEL, self.run_id, self._pointer_bytes(pointer)
                    )
                }
            )
        target = _POINTER_NAME[manifest.kind]
        temp = fsops.temp_name(target)
        try:
            fsops.write_exclusive(run_fd, temp, (pointer.model_dump_json() + "\n").encode())
            fsops.replace(run_fd, temp, target)
        except BaseException:
            fsops.unlink_quiet(run_fd, temp)
            raise
        fsops.fsync_dir(run_fd)

    def _stored_digest(self, kind: str, step: int) -> tuple[str | None, bool]:
        """``(digest, errored)``: the digest recorded in the run store, if any, and whether the
        lookup failed (a failing store is not the same as a store with no row)."""
        if self._anchor is None:
            return None, False
        try:
            digest = self._anchor(self.run_id, _check_kind(kind), step)
        except Exception as exc:  # noqa: BLE001  # a broken anchor is reported, never ignored
            self._emit("warning", "checkpoint_anchor_failed", error_type=type(exc).__name__)
            return None, True
        return (digest if isinstance(digest, str) and _SHA.fullmatch(digest) else None), False

    # -- freshness: rollback and replay ---------------------------------------------------------

    def _anchor_max(self, kind: str) -> int | None:
        fn = getattr(self._anchor, "max_step", None)
        if not callable(fn):
            return None
        try:
            value = fn(self.run_id, kind)
        except Exception as exc:  # noqa: BLE001  # a store that cannot answer fails closed
            self._emit("warning", "checkpoint_anchor_failed", error_type=type(exc).__name__)
            raise CheckpointUntrustedError(
                "the run store could not say how far checkpoints got (anchor_error)",
                reason="anchor_error",
            ) from None
        return (
            value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None
        )

    def _floor(self, kind: CheckpointKind) -> int | None:
        """The newest step known to have been saved, or None when nothing is known."""
        _check_kind(kind)
        if self.require_trust == "any":
            return None
        known: list[int] = []
        if self._key is not None:  # an unauthenticated pointer proves nothing about freshness
            with self._open_run(create=False) as run_fd:
                pointer = None if run_fd is None else self._pointer(run_fd, kind)
            if pointer is not None:
                known.append(pointer.step)
        stored = self._anchor_max(kind)
        if stored is not None:
            known.append(stored)
        return max(known) if known else None

    def freshness_floor(self, kind: CheckpointKind) -> int:
        """The newest step known to have been saved: the larger of the tag-checked pointer's step
        and the run store's highest recorded step. 0 under the ``any`` policy (which does not look
        for rollback) and when nothing is known.

        Raises:
            CheckpointUntrustedError: the run store failed to answer (reason ``anchor_error``).
        """
        return self._floor(kind) or 0

    def trusted_steps(self, kind: CheckpointKind) -> set[int]:
        """Steps whose manifest passes every trust rule (their data is not examined)."""
        _check_kind(kind)
        with self._open_run(create=False) as run_fd:
            if run_fd is None:
                return set()
            kind_fd = self._open_kind(run_fd, kind, create=False)
            if kind_fd is None:
                return set()
            import os

            try:
                return {
                    c.step
                    for c in self._judge(run_fd, kind_fd, kind)
                    if c.reason is None and c.step is not None
                }
            finally:
                os.close(kind_fd)

    def admission(
        self,
        kind: CheckpointKind,
        ctx: AdmitContext | None = None,
        *,
        trusted: set[int] | None = None,
    ) -> Admission:
        """The one gate every path passes (see ``Admission.decide``)."""
        _check_kind(kind)
        return Admission(self, kind, self.trusted_steps(kind) if trusted is None else trusted, ctx)

    def _judge(self, run_fd: int, kind_fd: int, kind: str) -> list[_Cand]:
        """Decide which manifests are trusted at the required level, newest first."""
        cands = self._candidates(kind_fd, kind)
        if self.require_trust == "any":
            return self._judge_chain(run_fd, cands, kind)
        return self._judge_strict(cands, kind)

    def _judge_strict(self, cands: list[_Cand], kind: str) -> list[_Cand]:
        """Policies ``authenticated`` and ``anchored``: every candidate needs its own proof (a
        valid tag, or the digest in the run store). Nothing is vouched for by a neighbour, so a
        forged newer manifest cannot launder an older one."""
        for cand in cands:
            self._strict_verdict(cand, kind)
        return cands

    def _strict_verdict(self, cand: _Cand, kind: str) -> None:
        manifest = cand.manifest
        if manifest is None or cand.step is None or cand.digest is None:
            return
        if cand.tag_state == "invalid":
            cand.reason = "auth_invalid"
            return
        stored, errored = self._stored_digest(kind, cand.step)
        if stored is not None and stored != cand.digest:
            cand.reason = "store_mismatch"
            return
        authenticated = cand.tag_state == "valid"
        if self.require_trust == "authenticated" and not authenticated:
            cand.reason = "auth_missing"
        elif not authenticated and stored is None:
            cand.reason = "anchor_error" if errored else "not_anchored"
        else:
            cand.trust = "authenticated" if authenticated else "anchored"

    def judge_single(self, cand: _Cand, kind: str) -> _Cand:
        """The trust verdict for one candidate that is not part of the local walk (a remote
        manifest, before anything is downloaded). Under ``any`` only a wrong tag rejects."""
        if cand.manifest is None:
            return cand
        if self.require_trust != "any":
            self._strict_verdict(cand, kind)
            return cand
        if cand.tag_state == "invalid":
            cand.reason = "auth_invalid"
            return cand
        stored, _ = self._stored_digest(kind, cand.step or 0)
        if stored is not None and stored != cand.digest:
            cand.reason = "store_mismatch"
        elif cand.tag_state == "valid":
            cand.trust = "authenticated"
        elif stored is not None:
            cand.trust = "anchored"
        return cand

    def _judge_chain(self, run_fd: int, cands: list[_Cand], kind: str) -> list[_Cand]:
        """Policy ``any``: the chain rules of the module docstring. A wrong tag always rejects;
        the level reached is reported (``local_chain_only`` when neither a tag nor the store
        vouches)."""
        pointer = self._pointer(run_fd, kind)
        chained: set[int] = set()
        if pointer is not None:
            current = pointer.manifest_sha256
            for cand in reversed(cands):
                if cand.manifest is None or cand.step is None or cand.step <= pointer.step:
                    continue
                if cand.manifest.prev_manifest_sha256 == current and cand.digest is not None:
                    chained.add(cand.step)
                    current = cand.digest
        expected: str | None = None
        expected_step: int | None = None
        for cand in cands:
            manifest = cand.manifest
            if manifest is None or cand.step is None or cand.digest is None:
                expected = expected_step = None
                continue
            reason: str | None = None
            if cand.tag_state == "invalid":
                reason = "auth_invalid"
            stored: str | None = None
            if reason is None:
                stored, _ = self._stored_digest(kind, cand.step)
            # A checkpoint with its own proof (a valid tag, or the digest in the run store) does
            # not need its successor's vouching, so retention may leave gaps between the ones it
            # keeps (the best by metric plus the newest) without making the older ones "corrupt".
            own_proof = cand.tag_state == "valid" or (stored is not None and stored == cand.digest)
            if reason is None and not own_proof and expected is not None:
                gap = (
                    expected != GENESIS_DIGEST
                    and expected_step is not None
                    and cand.step < expected_step  # the vouched predecessor is gone: a gap
                )
                if expected == GENESIS_DIGEST or (cand.digest != expected and not gap):
                    reason = "link_mismatch"
            if (
                reason is None
                and pointer is not None
                and cand.step == pointer.step
                and (
                    cand.digest != pointer.manifest_sha256
                    or manifest.chain_sha256 != pointer.chain_sha256
                )
            ):
                reason = "pointer_mismatch"
            if reason is None and stored is not None and stored != cand.digest:
                reason = "store_mismatch"
            authenticated = cand.tag_state == "valid"
            if (
                reason is None
                and pointer is not None
                and cand.step > pointer.step
                and cand.step not in chained
                and stored is None
                and not authenticated
            ):
                reason = "unanchored"
            cand.reason = reason
            if reason is None:
                cand.trust = (
                    "authenticated"
                    if authenticated
                    else "anchored"
                    if stored is not None
                    else "local_chain_only"
                )
            expected = None if reason else manifest.prev_manifest_sha256
            expected_step = None if reason else manifest.prev_step
        return cands

    # -- verifying data ------------------------------------------------------------------------

    def _check_data(
        self, kind_fd: int, manifest: CheckpointManifest, *, load: bool
    ) -> bytes | None:
        name = manifest.path.split("/", 1)[1]
        if manifest.size_bytes > self.max_bytes:
            raise CheckpointCorruptError(f"{name} is over the size limit", reason="too_large")
        serializer = get_serializer(manifest.format)
        try:
            size = fsops.stat_regular(kind_fd, name, max_bytes=self.max_bytes)
        except CheckpointCorruptError as exc:
            raise CheckpointCorruptError(
                str(exc), reason="data_missing" if exc.reason == "missing" else exc.reason
            ) from None
        if size != manifest.size_bytes:
            raise CheckpointCorruptError(f"{name} has the wrong size", reason="size_mismatch")
        if load:
            try:
                data = fsops.read_regular(kind_fd, name, max_bytes=manifest.size_bytes)
            except CheckpointCorruptError as exc:
                raise CheckpointCorruptError(
                    str(exc), reason="size_mismatch" if exc.reason == "too_large" else exc.reason
                ) from None
            if len(data) != manifest.size_bytes:
                raise CheckpointCorruptError(f"{name} has the wrong size", reason="size_mismatch")
            serializer.check(lambda n: data[:n], len(data))
            if hashlib.sha256(data).hexdigest() != manifest.sha256:
                raise CheckpointCorruptError(f"{name} fails its digest", reason="digest_mismatch")
            return data
        serializer.check(
            lambda n: fsops.read_prefix(kind_fd, name, n, max_bytes=self.max_bytes), size
        )
        digest, total = fsops.hash_regular(kind_fd, name, max_bytes=self.max_bytes)
        if total != manifest.size_bytes:
            raise CheckpointCorruptError(f"{name} changed size while read", reason="size_mismatch")
        if digest != manifest.sha256:
            raise CheckpointCorruptError(f"{name} fails its digest", reason="digest_mismatch")
        return None

    def iter_valid(
        self, kind: CheckpointKind, *, load: bool = False, ctx: AdmitContext | None = None
    ) -> Iterator[Verified]:
        """Checkpoints of ``kind`` the gate admits and whose data verifies, newest step first;
        each refusal is counted in ``skipped`` and logged. With ``load`` the verified bytes are
        returned, read once. When nothing is yielded the gate is asked whether finding nothing is
        acceptable (it is not when checkpoints are known to have existed)."""
        _check_kind(kind)
        self.skipped = []
        import os

        yielded = False
        with self._open_run(create=False) as run_fd:
            kind_fd: int | None = None
            judged: list[_Cand] = []
            if run_fd is not None:
                try:
                    kind_fd = self._open_kind(run_fd, kind, create=False)
                except CheckpointCorruptError as exc:
                    self._skip(None, kind, kind, exc.reason)
                if kind_fd is not None:
                    judged = self._judge(run_fd, kind_fd, kind)
            try:
                gate = self.admission(
                    kind,
                    ctx,
                    trusted={c.step for c in judged if c.reason is None and c.step is not None},
                )
                for cand in judged:
                    reason = gate.decide(cand)
                    if reason is not None or cand.manifest is None or cand.digest is None:
                        self._skip(cand.step, kind, cand.name, reason or "manifest_invalid")
                        continue
                    assert kind_fd is not None  # noqa: S101  # judged is empty without it
                    try:
                        data = self._check_data(kind_fd, cand.manifest, load=load)
                    except CheckpointCorruptError as exc:
                        self._skip(cand.manifest.step, kind, cand.name, exc.reason)
                        continue
                    yielded = True
                    yield Verified(cand.manifest, cand.digest, data, cand.trust)
                if not yielded:
                    reason = gate.decide(None)
                    if reason is not None:
                        self._skip(None, kind, kind, reason, gate.floor)
            finally:
                if kind_fd is not None:
                    os.close(kind_fd)

    @staticmethod
    def _identity(kind_fd: int, name: str) -> tuple[int, ...] | None:
        """Inode, size and times of a file: what changes when it is replaced or rewritten."""
        import os

        try:
            st = os.stat(name, dir_fd=kind_fd, follow_symlinks=False)
        except OSError:
            return None
        return (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

    def _remember_verified(self, kind_fd: int, manifest: CheckpointManifest) -> None:
        name = manifest.path.split("/", 1)[1]
        identity = self._identity(kind_fd, name)
        if identity is not None:
            self._verified_files[(manifest.kind, name)] = (identity, manifest.sha256)

    def _data_ok(self, kind_fd: int, manifest: CheckpointManifest) -> bool:
        """Does the data of ``manifest`` verify? Used by retention only (a restore always
        hashes): a file this process verified (at save or earlier) whose inode, size and times
        are unchanged is not hashed again, so steady-state retention reads nothing."""
        name = manifest.path.split("/", 1)[1]
        key = (manifest.kind, name)
        identity = self._identity(kind_fd, name)
        if identity is not None and self._verified_files.get(key) == (identity, manifest.sha256):
            return True
        try:
            self._check_data(kind_fd, manifest, load=False)
        except CheckpointCorruptError:
            self._verified_files.pop(key, None)
            return False
        self._remember_verified(kind_fd, manifest)
        return True

    def inspect(
        self, kind: CheckpointKind, *, check_data: bool = True, ctx: AdmitContext | None = None
    ) -> list[Entry]:
        """Every checkpoint and every stray file of ``kind`` with its verification result,
        newest step first. ``check_data=False`` judges manifests only (no hashing)."""
        _check_kind(kind)
        import os

        entries: list[Entry] = []
        with self._open_run(create=False) as run_fd:
            if run_fd is None:
                return entries
            try:
                kind_fd = self._open_kind(run_fd, kind, create=False)
            except CheckpointCorruptError as exc:
                return [Entry(kind, None, kind, "corrupt", exc.reason)]
            if kind_fd is None:
                return entries
            try:
                judged = self._judge(run_fd, kind_fd, kind)
                names = self._list(kind_fd)
                have_manifest = {c.step for c in judged if c.step is not None}
                gate = self.admission(
                    kind,
                    ctx,
                    trusted={c.step for c in judged if c.reason is None and c.step is not None},
                )
                for cand in judged:
                    manifest = cand.manifest
                    reason = gate.decide(cand)
                    if reason is None and manifest is not None and check_data:
                        try:
                            self._check_data(kind_fd, manifest, load=False)
                        except CheckpointCorruptError as exc:
                            reason = exc.reason
                    entries.append(
                        Entry(
                            cand.name,
                            cand.step,
                            kind,
                            "corrupt" if reason else "valid",
                            reason,
                            manifest if reason is None else None,
                            cand.trust if reason is None else None,
                        )
                    )
                for name in names:
                    match = _DATA.fullmatch(name)
                    if match is not None and int(match.group(1)) not in have_manifest:
                        entries.append(
                            Entry(name, int(match.group(1)), kind, "corrupt", "no_manifest")
                        )
            finally:
                os.close(kind_fd)
        entries.sort(key=lambda e: e.step if e.step is not None else -1, reverse=True)
        return entries

    # -- saving ---------------------------------------------------------------------------------

    def _cleanup(self, kind_fd: int, run_fd: int, kind: str) -> None:
        """Remove debris of killed saves; safe because the caller holds the save lock."""
        names = self._list(kind_fd)
        steps_with_data = {int(m.group(1)) for n in names if (m := _DATA.fullmatch(n))}
        steps_with_manifest = {int(m.group(1)) for n in names if (m := _MANIFEST.fullmatch(n))}
        for name in names:
            removed: str | None = None
            if fsops.is_temp_name(name):
                removed = "temp_file"
            elif (m := _MANIFEST.fullmatch(name)) and int(m.group(1)) not in steps_with_data:
                removed = "manifest_without_data"
            elif (m := _DATA.fullmatch(name)) and int(m.group(1)) not in steps_with_manifest:
                removed = "no_manifest"
            elif (m := _MARKER.fullmatch(name)) and int(m.group(1)) not in steps_with_manifest:
                removed = "marker_without_manifest"
            if removed is not None:
                fsops.unlink_quiet(kind_fd, name)
                self._emit("warning", "checkpoint_debris_removed", kind=kind, reason=removed)
        for name in fsops.listdir(run_fd, limit=self.max_entries):
            if fsops.is_temp_name(name):
                fsops.unlink_quiet(run_fd, name)

    def save(
        self,
        *,
        kind: CheckpointKind,
        step: int,
        data: bytes | Iterable[bytes],
        format: CheckpointFormat,
        versions: dict[str, str],
        dataset_hash: str,
        metric_key: str | None = None,
        metric_value: float | None = None,
        metric_mode: Literal["max", "min"] | None = None,
        git_commit: str | None = None,
        uri: str | None = None,
        created_at: datetime | None = None,
        verify: bool = True,
        before_publish: Callable[[CheckpointManifest, str], None] | None = None,
    ) -> CheckpointManifest:
        """Write one checkpoint atomically and return its manifest.

        Raises:
            CheckpointPathError: invalid kind or step.
            CheckpointStepError: ``step`` does not exceed the newest step of ``kind``.
            CheckpointExistsError: this step exists with other content.
            CheckpointBusyError: another process is saving into this run directory.
            CheckpointError: the write failed (disk full, over ``max_bytes``) or verification
                after the write failed; the previous checkpoint is untouched.
        """
        _check_kind(kind)
        _check_step(step)
        serializer = get_serializer(format)
        with self._locked() as run_fd:
            kind_fd = self._open_kind(run_fd, kind, create=True)
            assert kind_fd is not None  # noqa: S101  # create=True never returns None
            import os

            try:
                return self._save_locked(
                    run_fd, kind_fd, kind, step, data, format, serializer.suffix, versions,
                    dataset_hash, metric_key, metric_value, metric_mode, git_commit, uri,
                    created_at, verify, before_publish,
                )  # fmt: skip
            finally:
                os.close(kind_fd)

    def _save_locked(
        self,
        run_fd: int,
        kind_fd: int,
        kind: CheckpointKind,
        step: int,
        data: bytes | Iterable[bytes],
        format: CheckpointFormat,
        suffix: str,
        versions: dict[str, str],
        dataset_hash: str,
        metric_key: str | None,
        metric_value: float | None,
        metric_mode: Literal["max", "min"] | None,
        git_commit: str | None,
        uri: str | None,
        created_at: datetime | None,
        verify: bool,
        before_publish: Callable[[CheckpointManifest, str], None] | None,
    ) -> CheckpointManifest:
        self._cleanup(kind_fd, run_fd, kind)
        # only manifests the gate admits count as "the newest" and as "the previous": a planted
        # manifest at a huge step must not block every later save, and a forged one must not
        # become the predecessor a genuine checkpoint chains to
        cands = [
            c
            for c in self._judge(run_fd, kind_fd, kind)
            if c.reason is None and c.manifest is not None
        ]
        newest = max((c.step for c in cands if c.step is not None), default=None)
        if newest is not None and step <= newest:
            same = next((c for c in cands if c.step == step and c.manifest is not None), None)
            if (
                same is not None
                and same.manifest is not None
                and isinstance(data, (bytes, bytearray))
                and hashlib.sha256(data).hexdigest() == same.manifest.sha256
                and len(data) == same.manifest.size_bytes
            ):
                return same.manifest
            if step == newest:
                raise CheckpointExistsError(
                    f"{kind} checkpoint at step {step} exists with other content; "
                    "checkpoints are immutable"
                )
            raise CheckpointStepError(
                f"step {step} does not exceed the newest {kind} checkpoint (step {newest})"
            )
        previous = next(
            (
                c
                for c in cands
                if c.manifest is not None
                and c.digest is not None
                and fsops.exists(kind_fd, c.manifest.path.split("/", 1)[1])
            ),
            None,
        )
        prev_digest = previous.digest if previous and previous.digest else GENESIS_DIGEST
        prev_chain = (
            previous.manifest.chain_sha256 if previous and previous.manifest else GENESIS_DIGEST
        )
        final_data = f"step_{step:09d}.{suffix}"
        final_manifest = manifest_name(step)
        temp_data, temp_manifest = fsops.temp_name(final_data), fsops.temp_name(final_manifest)
        published = False
        try:
            sha256, size = fsops.write_exclusive(kind_fd, temp_data, data, max_bytes=self.max_bytes)
            if size < 1:
                raise CheckpointError("a checkpoint cannot be empty")
            manifest = CheckpointManifest(
                run_id=self.run_id,
                step=step,
                kind=kind,
                path=f"{kind}/{final_data}",
                uri=uri,
                sha256=sha256,
                size_bytes=size,
                format=format,
                metric_key=metric_key,
                metric_value=metric_value,
                metric_mode=metric_mode,
                versions=versions,
                git_commit=git_commit,
                dataset_hash=dataset_hash,
                created_at=created_at or datetime.now(timezone.utc),
                prev_manifest_sha256=prev_digest,
                prev_chain_sha256=prev_chain,
                prev_step=previous.step if previous and previous.digest else None,
                chain_sha256=checkpoint_chain(prev_chain, self.run_id, kind, step, sha256, size),
            )
            if self._key is not None:
                manifest = manifest.model_copy(
                    update={
                        "auth_tag": auth.tag(
                            self._key,
                            auth.MANIFEST_LABEL,
                            self.run_id,
                            self._manifest_bytes(manifest),
                        )
                    }
                )
            raw = (manifest.model_dump_json() + "\n").encode()
            fsops.write_exclusive(kind_fd, temp_manifest, raw)
            fsops.fsync_dir(kind_fd)
            fsops.replace(kind_fd, temp_manifest, final_manifest)
            published = True
            fsops.replace(kind_fd, temp_data, final_data)
            fsops.fsync_dir(kind_fd)
            if verify:
                digest, total = fsops.hash_regular(kind_fd, final_data, max_bytes=self.max_bytes)
                if (digest, total) != (sha256, size):
                    raise CheckpointCorruptError(
                        f"{final_data} does not verify after it was written",
                        reason="digest_mismatch",
                    )
                self._remember_verified(kind_fd, manifest)
        except BaseException:
            fsops.unlink_quiet(kind_fd, temp_data)
            fsops.unlink_quiet(kind_fd, temp_manifest)
            if published:  # the pair is not complete or not verified: nothing may point at it
                fsops.unlink_quiet(kind_fd, final_manifest)
                fsops.unlink_quiet(kind_fd, final_data)
                fsops.fsync_dir(kind_fd)
            raise
        manifest_digest = hashlib.sha256(raw).hexdigest()
        if before_publish is not None:
            before_publish(manifest, manifest_digest)
        self._write_pointer(run_fd, manifest, manifest_digest)
        return manifest

    def existing(self, kind: CheckpointKind, step: int) -> CheckpointManifest | None:
        """The self-checked manifest at ``(kind, step)`` if there is one (data not re-hashed)."""
        _check_kind(kind)
        _check_step(step)
        with self._open_run(create=False) as run_fd:
            kind_fd = None if run_fd is None else self._open_kind(run_fd, kind, create=False)
            if kind_fd is None:
                return None
            import os

            try:
                cand = self._load_manifest(kind_fd, kind, manifest_name(step), step)
            except CheckpointError:
                return None
            finally:
                os.close(kind_fd)
        return cand.manifest

    # -- liveness -------------------------------------------------------------------------------

    def hold_lease(self) -> int | None:
        """Take the run's liveness lease (see ``fsops.hold_lease``); the caller keeps the
        descriptor open for the life of the process."""
        try:
            with self._open_run(create=True) as run_fd:
                return None if run_fd is None else fsops.hold_lease(run_fd)
        except CheckpointError:
            return None

    def lease_state(self) -> bool | None:
        """True: a live process holds the lease; False: the lease file exists and is free (the
        process is dead); None: no lease file."""
        with self._open_run(create=False) as run_fd:
            return None if run_fd is None else fsops.probe_lease(run_fd)

    def last_activity(self) -> float | None:
        """The newest modification time in the run's checkpoint folders (a heartbeat of sorts)."""
        import os

        newest: float | None = None
        with self._open_run(create=False) as run_fd:
            if run_fd is None:
                return None
            times = [fsops.newest_mtime(run_fd)]
            for kind in KINDS:
                try:
                    kind_fd = self._open_kind(run_fd, kind, create=False)
                except CheckpointCorruptError:
                    continue
                if kind_fd is not None:
                    times.append(fsops.newest_mtime(kind_fd))
                    os.close(kind_fd)
        for value in times:
            if value is not None and (newest is None or value > newest):
                newest = value
        return newest

    # -- bytes for the remote mirror ------------------------------------------------------------

    def read_range(self, manifest: CheckpointManifest, offset: int, length: int) -> bytes:
        """``length`` bytes of a checkpoint's data file at ``offset`` (regular files only)."""
        name = manifest.path.split("/", 1)[1]
        with self._open_run(create=False) as run_fd:
            kind_fd = (
                None if run_fd is None else self._open_kind(run_fd, manifest.kind, create=False)
            )
            if kind_fd is None:
                raise CheckpointCorruptError("the checkpoint directory is gone", reason="missing")
            import os

            try:
                return fsops.read_at(kind_fd, name, offset, length, max_bytes=self.max_bytes)
            finally:
                os.close(kind_fd)

    def read_manifest_bytes(self, manifest: CheckpointManifest) -> bytes:
        """The exact bytes of the manifest file (what the pointer and the store digest cover)."""
        import os

        with self._open_run(create=False) as run_fd:
            kind_fd = (
                None if run_fd is None else self._open_kind(run_fd, manifest.kind, create=False)
            )
            if kind_fd is None:
                raise CheckpointCorruptError("the checkpoint directory is gone", reason="missing")
            try:
                return fsops.read_regular(
                    kind_fd, manifest_name(manifest.step), max_bytes=MAX_MANIFEST_BYTES
                )
            finally:
                os.close(kind_fd)

    def install(
        self,
        kind: CheckpointKind,
        step: int,
        raw_manifest: bytes,
        chunks: Iterable[bytes],
        ctx: AdmitContext | None = None,
    ) -> CheckpointManifest | None:
        """Install a checkpoint fetched from the remote: the manifest must pass every self-check
        for this run, kind and step, and the data must match its size and SHA-256 while it is
        written. The pointer is not touched (the walk finds it); an existing step is left alone.

        Returns:
            The manifest, or None when this step already exists locally.

        Raises:
            CheckpointCorruptError: the manifest or the data does not verify; nothing is kept.
        """
        _check_kind(kind)
        _check_step(step)
        cand = self._cand_from_bytes(raw_manifest, kind, manifest_name(step), step)
        manifest = cand.manifest
        if manifest is None:
            raise CheckpointCorruptError(
                "the remote manifest does not verify", reason=cand.reason or "manifest_invalid"
            )
        refusal = self.admission(kind, ctx).decide(self.judge_single(cand, kind))
        if refusal is not None:
            raise CheckpointCorruptError(
                f"the remote checkpoint is not admitted ({refusal}); nothing was written",
                reason=refusal,
            )
        final_data = manifest.path.split("/", 1)[1]
        final_manifest = manifest_name(step)
        with self._locked() as run_fd:
            kind_fd = self._open_kind(run_fd, kind, create=True)
            assert kind_fd is not None  # noqa: S101  # create=True never returns None
            import os

            temp_data, temp_manifest = fsops.temp_name(final_data), fsops.temp_name(final_manifest)
            try:
                if fsops.exists(kind_fd, final_manifest) or fsops.exists(kind_fd, final_data):
                    return None
                sha256, size = fsops.write_exclusive(
                    kind_fd, temp_data, chunks, max_bytes=manifest.size_bytes
                )
                if (sha256, size) != (manifest.sha256, manifest.size_bytes):
                    raise CheckpointCorruptError(
                        "the downloaded checkpoint does not match its manifest",
                        reason="digest_mismatch",
                    )
                fsops.write_exclusive(kind_fd, temp_manifest, raw_manifest)
                fsops.fsync_dir(kind_fd)
                fsops.replace(kind_fd, temp_manifest, final_manifest)
                fsops.replace(kind_fd, temp_data, final_data)
                fsops.fsync_dir(kind_fd)
            except BaseException:
                fsops.unlink_quiet(kind_fd, temp_data)
                fsops.unlink_quiet(kind_fd, temp_manifest)
                raise
            finally:
                os.close(kind_fd)
        return manifest

    # -- remote markers -------------------------------------------------------------------------

    def _manifest_digest(self, kind_fd: int, manifest: CheckpointManifest) -> str:
        raw = fsops.read_regular(
            kind_fd, manifest_name(manifest.step), max_bytes=MAX_MANIFEST_BYTES
        )
        return hashlib.sha256(raw).hexdigest()

    def mark_uploaded(self, manifest: CheckpointManifest, uri: str) -> None:
        """Record that ``manifest``'s files are on the remote with matching size and digest."""
        import os

        with self._locked() as run_fd:
            kind_fd = self._open_kind(run_fd, manifest.kind, create=False)
            if kind_fd is None:
                raise CheckpointError("the checkpoint directory is gone")
            try:
                marker = {
                    "uri": uri,
                    "manifest_sha256": self._manifest_digest(kind_fd, manifest),
                    "sha256": manifest.sha256,
                    "size_bytes": manifest.size_bytes,
                }
                name = marker_name(manifest.step)
                temp = fsops.temp_name(name)
                try:
                    fsops.write_exclusive(kind_fd, temp, json.dumps(marker).encode())
                    fsops.replace(kind_fd, temp, name)
                except BaseException:
                    fsops.unlink_quiet(kind_fd, temp)
                    raise
                fsops.fsync_dir(kind_fd)
            finally:
                os.close(kind_fd)

    def is_uploaded(self, manifest: CheckpointManifest) -> bool:
        """True when a marker names this exact manifest (digest, file digest and size)."""
        import os

        with self._open_run(create=False) as run_fd:
            if run_fd is None:
                return False
            kind_fd = self._open_kind(run_fd, manifest.kind, create=False)
            if kind_fd is None:
                return False
            try:
                raw = fsops.read_regular(
                    kind_fd, marker_name(manifest.step), max_bytes=MAX_MANIFEST_BYTES
                )
                marker = json.loads(raw)
                return bool(
                    isinstance(marker, dict)
                    and marker.get("manifest_sha256") == self._manifest_digest(kind_fd, manifest)
                    and marker.get("sha256") == manifest.sha256
                    and marker.get("size_bytes") == manifest.size_bytes
                )
            except (CheckpointError, ValueError):
                return False
            finally:
                os.close(kind_fd)

    # -- retention ------------------------------------------------------------------------------

    def prune(
        self,
        kind: CheckpointKind,
        *,
        keep: int,
        mode: Literal["max", "min"] | None = None,
        is_confirmed: Callable[[CheckpointManifest], bool] | None = None,
        dry_run: bool = False,
    ) -> PruneResult:
        """Delete valid checkpoints of ``kind`` beyond the newest ``keep`` (``last``) or beyond
        the best ``keep`` by metric plus the newest (``best``).

        Only checkpoints that verify are counted or deleted, so a corrupt newest file never
        costs the last good one; the newest valid checkpoint is always kept; and when
        ``is_confirmed`` is given nothing is deleted until the newest valid checkpoint is
        confirmed (uploaded and verified remotely).

        ``mode`` says whether a higher or a lower metric is better for ``best``. Leave it ``None``
        to use the mode the checkpoints were saved with (``metric_mode`` in their manifests); a
        value that contradicts it is refused rather than silently ranking the wrong end.

        Raises:
            CheckpointError: ``keep`` is below 1, or ``mode`` contradicts the saved checkpoints.
        """
        _check_kind(kind)
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise CheckpointError(
                "keep must be at least 1: the newest valid checkpoint is never deleted"
            )
        import os

        with self._locked() as run_fd:
            kind_fd = self._open_kind(run_fd, kind, create=False)
            if kind_fd is None:
                return PruneResult()
            try:
                # manifests that pass the gate, newest first; their data is hashed only when it
                # matters and was not already verified by this process (see ``_data_ok``)
                cands = [
                    e.manifest
                    for e in self.inspect(kind, check_data=False)
                    if e.status == "valid" and e.manifest is not None
                ]
                newest = next((m for m in cands if self._data_ok(kind_fd, m)), None)
                if newest is None:
                    return PruneResult()
                kept: list[CheckpointManifest] = []
                if kind == "last":
                    for m in cands:
                        if len(kept) >= keep:
                            break
                        if self._data_ok(kind_fd, m):
                            kept.append(m)
                else:
                    saved_modes = {m.metric_mode for m in cands if m.metric_mode is not None}
                    if len(saved_modes) > 1 or (
                        mode is not None and saved_modes and saved_modes != {mode}
                    ):
                        raise CheckpointError(
                            "the --mode given contradicts the mode the checkpoints were saved "
                            f"with ({', '.join(sorted(saved_modes))}); nothing was deleted"
                        )
                    ranking = mode or (next(iter(saved_modes)) if saved_modes else "max")
                    sign = 1.0 if ranking == "max" else -1.0
                    ranked = sorted(
                        cands, key=lambda m: (sign * (m.metric_value or 0.0), m.step), reverse=True
                    )
                    for m in ranked:
                        if len(kept) >= keep:
                            break
                        if self._data_ok(kind_fd, m):
                            kept.append(m)
                    if newest not in kept:
                        kept.append(newest)
                # only a checkpoint that verifies is ever deleted: a damaged one is left alone
                doomed = sorted(
                    (m for m in cands if m not in kept and self._data_ok(kind_fd, m)),
                    key=lambda m: m.step,
                )
                if doomed and is_confirmed is not None and not is_confirmed(newest):
                    return PruneResult([], list(kept), held_back=True)
                if not dry_run:
                    for manifest in doomed:
                        for name in (
                            manifest_name(manifest.step),
                            manifest.path.split("/", 1)[1],
                            marker_name(manifest.step),
                        ):
                            fsops.unlink_quiet(kind_fd, name)
                    if doomed:
                        fsops.fsync_dir(kind_fd)
                return PruneResult(doomed, list(kept))
            finally:
                os.close(kind_fd)
