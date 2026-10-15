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

DEFAULT_MAX_BYTES = 16 * 1024 * 1024 * 1024
"""Largest checkpoint file written or read by default (16 GiB)."""
DEFAULT_MAX_ENTRIES = 10_000
"""Most files examined in one kind directory; bounds the work of a walk."""
KINDS: tuple[CheckpointKind, ...] = ("last", "best")
_POINTER_NAME: dict[str, str] = {"last": "LATEST", "best": "LATEST_BEST"}
_DATA = re.compile(r"step_([0-9]{9,19})\.(safetensors|pt)")
_MANIFEST = re.compile(r"step_([0-9]{9,19})\.manifest\.json")
_MARKER = re.compile(r"step_([0-9]{9,19})\.remote\.json")
_SHA = re.compile(r"[0-9a-f]{64}")
TRUST_REASONS = frozenset(
    {"auth_invalid", "auth_missing", "not_anchored", "anchor_error", "store_mismatch"}
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

    def _skip(self, step: int | None, kind: str, name: str, reason: str) -> None:
        self.skipped.append(Skipped(step, kind, name, reason))
        event = "checkpoint_untrusted" if reason in TRUST_REASONS else "checkpoint_skipped_corrupt"
        self._emit("warning", event, step=step, kind=kind, reason=reason)

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
            manifest = cand.manifest
            if manifest is None or cand.step is None or cand.digest is None:
                continue
            if cand.tag_state == "invalid":
                cand.reason = "auth_invalid"
                continue
            stored, errored = self._stored_digest(kind, cand.step)
            if stored is not None and stored != cand.digest:
                cand.reason = "store_mismatch"
                continue
            authenticated = cand.tag_state == "valid"
            if self.require_trust == "authenticated" and not authenticated:
                cand.reason = "auth_missing"
            elif not authenticated and stored is None:
                cand.reason = "anchor_error" if errored else "not_anchored"
            else:
                cand.trust = "authenticated" if authenticated else "anchored"
        return cands

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
        for cand in cands:
            manifest = cand.manifest
            if manifest is None or cand.step is None or cand.digest is None:
                expected = None
                continue
            reason: str | None = None
            if cand.tag_state == "invalid":
                reason = "auth_invalid"
            if (
                reason is None
                and expected is not None
                and (expected == GENESIS_DIGEST or cand.digest != expected)
            ):
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
            stored: str | None = None
            if reason is None:
                stored, _ = self._stored_digest(kind, cand.step)
            if stored is not None and stored != cand.digest:
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

    def iter_valid(self, kind: CheckpointKind, *, load: bool = False) -> Iterator[Verified]:
        """Checkpoints of ``kind`` that verify, newest step first; each refusal is counted in
        ``skipped`` and logged. With ``load`` the verified bytes are returned, read once."""
        _check_kind(kind)
        self.skipped = []
        with self._open_run(create=False) as run_fd:
            if run_fd is None:
                return
            try:
                kind_fd = self._open_kind(run_fd, kind, create=False)
            except CheckpointCorruptError as exc:
                self._skip(None, kind, kind, exc.reason)
                return
            if kind_fd is None:
                return
            import os

            try:
                for cand in self._judge(run_fd, kind_fd, kind):
                    manifest = cand.manifest
                    if cand.reason is not None or manifest is None or cand.digest is None:
                        self._skip(cand.step, kind, cand.name, cand.reason or "manifest_invalid")
                        continue
                    if manifest.format == "pt" and not self.allow_pt:
                        self._skip(manifest.step, kind, cand.name, "pt_not_allowed")
                        continue
                    try:
                        data = self._check_data(kind_fd, manifest, load=load)
                    except CheckpointCorruptError as exc:
                        self._skip(manifest.step, kind, cand.name, exc.reason)
                        continue
                    yield Verified(manifest, cand.digest, data, cand.trust)
            finally:
                os.close(kind_fd)

    def inspect(self, kind: CheckpointKind, *, check_data: bool = True) -> list[Entry]:
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
                for cand in judged:
                    manifest = cand.manifest
                    reason = cand.reason
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
        cands = self._candidates(kind_fd, kind)
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
        self, kind: CheckpointKind, step: int, raw_manifest: bytes, chunks: Iterable[bytes]
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
        if manifest.size_bytes > self.max_bytes:
            raise CheckpointCorruptError(
                "the remote checkpoint is over the size limit", reason="too_large"
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
        mode: Literal["max", "min"] = "max",
        is_confirmed: Callable[[CheckpointManifest], bool] | None = None,
        dry_run: bool = False,
    ) -> PruneResult:
        """Delete valid checkpoints of ``kind`` beyond the newest ``keep`` (``last``) or beyond
        the best ``keep`` by metric plus the newest (``best``).

        Only checkpoints that verify are counted or deleted, so a corrupt newest file never
        costs the last good one; the newest valid checkpoint is always kept; and when
        ``is_confirmed`` is given nothing is deleted until the newest valid checkpoint is
        confirmed (uploaded and verified remotely).

        Raises:
            CheckpointError: ``keep`` is below 1.
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
                valid = [
                    e.manifest
                    for e in self.inspect(kind, check_data=True)
                    if e.status == "valid" and e.manifest is not None
                ]
                if not valid:
                    return PruneResult()
                newest = valid[0]
                if kind == "last":
                    kept = valid[:keep]
                else:
                    sign = 1.0 if mode == "max" else -1.0
                    ranked = sorted(
                        valid, key=lambda m: (sign * (m.metric_value or 0.0), m.step), reverse=True
                    )
                    kept = ranked[:keep]
                    if newest not in kept:
                        kept.append(newest)
                doomed = sorted((m for m in valid if m not in kept), key=lambda m: m.step)
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
