"""Mirror checkpoints to object storage and fetch them back (ADR-0006 rule 5).

Upload happens after the local save verified. The data goes to ``<name>.part``; an interrupted
upload resumes from the remote part when its bytes equal the local prefix (otherwise it starts
over); the finished part is compared with the manifest by size and SHA-256 (read back) before it
is renamed into place, and the manifest is uploaded last, so a remote manifest implies verified
data. All storage calls share one deadline and are retried with exponential backoff only for
``OSError`` (a bug is never retried). Local bytes are read through the directory-descriptor layer
and hashed while they are sent, so a local file that changed mid-upload is caught.

Logs and errors name the scheme and bucket at most: never the path, never backend text.
"""

from __future__ import annotations

import functools
import hashlib
import re
import time
from collections.abc import Callable, Iterator
from typing import Any, TypeVar

from cvpilot.ckpt.directory import RunCheckpoints, manifest_name
from cvpilot.core.errors import CheckpointCorruptError, CheckpointError
from cvpilot.core.schema import MAX_MANIFEST_BYTES, CheckpointKind, CheckpointManifest, RemoteStore
from cvpilot.core.uri import RemoteLocation

T = TypeVar("T")
CHUNK = 8 * 1024 * 1024
MAX_BACKOFF_S = 8.0
_STEP_NAME = re.compile(r"step_([0-9]{9,19})\.manifest\.json")


def retrying(
    fn: Callable[[], T],
    *,
    retries: int,
    backoff: float,
    deadline: float,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    started: float | None = None,
) -> T:
    """Call ``fn``; on ``OSError`` retry up to ``retries`` attempts with backoff ``backoff * 2^n``
    (at most 8 s) while the ``deadline`` (seconds since ``started``) allows another wait."""
    begin = clock() if started is None else started
    attempt = 0
    while True:
        try:
            return fn()
        except OSError:
            attempt += 1
            delay = min(backoff * 2 ** (attempt - 1), MAX_BACKOFF_S)
            if attempt >= retries or clock() - begin + delay > deadline:
                raise
            sleep(delay)


class RemoteVerifyError(CheckpointError):
    """The remote copy does not match the manifest's size or digest."""

    event = "checkpoint_upload_failed"


class Mirror:
    """Upload and download for one remote location."""

    def __init__(
        self,
        location: RemoteLocation,
        store: RemoteStore,
        *,
        retries: int = 3,
        backoff: float = 0.5,
        deadline: float = 300.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if retries < 1 or backoff < 0 or deadline <= 0:
            raise ValueError("retries must be at least 1, backoff not negative, deadline positive")
        self.location = location
        self._store = store
        self._retries, self._backoff, self._deadline = retries, backoff, deadline
        self._sleep, self._clock = sleep, clock
        self._fetched = 0

    def begin_restore(self) -> None:
        """Start a restore: the byte budget of fetches begins again."""
        self._fetched = 0

    @property
    def label(self) -> str:
        """``scheme://bucket``: all that is ever logged about the location."""
        return f"{self.location.scheme}://{self.location.bucket}".rstrip("/")

    def _call(self, fn: Callable[[], T], started: float) -> T:
        return retrying(
            fn,
            retries=self._retries,
            backoff=self._backoff,
            deadline=self._deadline,
            sleep=self._sleep,
            clock=self._clock,
            started=started,
        )

    # -- digests -----------------------------------------------------------------------------

    def _remote_sha(self, key: str, length: int, started: float) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < length:
            want = min(CHUNK, length - offset)
            chunk = self._call(
                functools.partial(self._store.read_range, key, offset, want), started
            )
            if len(chunk) != want:
                raise RemoteVerifyError("the remote object is shorter than its size")
            digest.update(chunk)
            offset += want
        return digest.hexdigest()

    @staticmethod
    def _local_chunks(
        rc: RunCheckpoints, manifest: CheckpointManifest, start: int, hasher: Any
    ) -> Iterator[bytes]:
        offset = start
        while offset < manifest.size_bytes:
            chunk = rc.read_range(manifest, offset, min(CHUNK, manifest.size_bytes - offset))
            if not chunk:
                raise CheckpointCorruptError(
                    "the local checkpoint is shorter than its manifest", reason="size_mismatch"
                )
            hasher.update(chunk)
            offset += len(chunk)
            yield chunk

    def _local_prefix_sha(
        self, rc: RunCheckpoints, manifest: CheckpointManifest, length: int
    ) -> Any:
        hasher = hashlib.sha256()
        offset = 0
        while offset < length:
            chunk = rc.read_range(manifest, offset, min(CHUNK, length - offset))
            if not chunk:
                break
            hasher.update(chunk)
            offset += len(chunk)
        return hasher

    # -- upload ------------------------------------------------------------------------------

    def _data_present(self, key: str, manifest: CheckpointManifest, started: float) -> bool:
        size = self._call(lambda: self._store.size(key), started)
        return size == manifest.size_bytes and (
            self._remote_sha(key, size, started) == manifest.sha256
        )

    def _send_data(
        self, rc: RunCheckpoints, manifest: CheckpointManifest, part: str, started: float
    ) -> None:
        total = manifest.size_bytes
        existing = self._store.size(part)
        resume_at = 0
        hasher = hashlib.sha256()
        if existing is not None and 0 < existing <= total:
            local = self._local_prefix_sha(rc, manifest, existing)
            if self._remote_sha(part, existing, started) == local.hexdigest():
                resume_at, hasher = existing, local
        if existing is not None and resume_at == 0:
            self._store.delete(part)
        if resume_at < total:
            self._store.write(
                part, self._local_chunks(rc, manifest, resume_at, hasher), append=resume_at > 0
            )
        if hasher.hexdigest() != manifest.sha256:
            self._store.delete(part)
            raise CheckpointCorruptError(
                "the local checkpoint changed while it was uploaded", reason="digest_mismatch"
            )

    def upload(self, rc: RunCheckpoints, manifest: CheckpointManifest) -> bool:
        """Mirror one checkpoint; True when something was sent, False when the remote already
        held a verified copy. The manifest is written last.

        Raises:
            OSError: the storage kept failing (after retries, within the deadline).
            RemoteVerifyError: the remote copy does not match the manifest.
            CheckpointCorruptError: the local file no longer matches its manifest.
        """
        started = self._clock()
        base = f"{rc.run_id}/{manifest.kind}/"
        name = manifest.path.split("/", 1)[1]
        final, part = base + name, base + name + ".part"
        mfinal = base + manifest_name(manifest.step)
        raw = rc.read_manifest_bytes(manifest)
        sent = False
        if not self._data_present(final, manifest, started):
            try:
                self._call(lambda: self._send_data(rc, manifest, part, started), started)
                if self._call(lambda: self._store.size(part), started) != manifest.size_bytes:
                    raise RemoteVerifyError("the uploaded size does not match the manifest")
                if self._remote_sha(part, manifest.size_bytes, started) != manifest.sha256:
                    raise RemoteVerifyError("the uploaded digest does not match the manifest")
                self._call(lambda: self._store.move(part, final), started)
            except BaseException:
                try:
                    self._store.delete(part)
                except OSError:
                    pass
                raise
            sent = True
        present = self._call(lambda: self._store.size(mfinal), started)
        if (
            present != len(raw)
            or self._call(lambda: self._store.read_bounded(mfinal, MAX_MANIFEST_BYTES), started)
            != raw
        ):
            mpart = mfinal + ".part"
            self._call(lambda: self._store.write(mpart, [raw]), started)
            if (
                self._call(lambda: self._store.read_bounded(mpart, MAX_MANIFEST_BYTES), started)
                != raw
            ):
                self._store.delete(mpart)
                raise RemoteVerifyError("the uploaded manifest does not match")
            self._call(lambda: self._store.move(mpart, mfinal), started)
            sent = True
        rc.mark_uploaded(manifest, f"{self.location.url}/{rc.run_id}/{manifest.kind}/{name}")
        return sent

    # -- download ----------------------------------------------------------------------------

    def list_steps(self, run_id: str, kind: CheckpointKind) -> list[int]:
        """Steps that have a manifest on the remote, newest first."""
        started = self._clock()
        names = self._call(lambda: self._store.list(f"{run_id}/{kind}"), started)
        steps = [int(m.group(1)) for n in names if (m := _STEP_NAME.fullmatch(n))]
        return sorted(steps, reverse=True)

    def fetch(self, rc: RunCheckpoints, kind: CheckpointKind, step: int) -> bool:
        """Download one checkpoint into the local directory; the manifest and the data are
        verified before anything is kept.

        Returns:
            False when the step already exists locally.

        Raises:
            CheckpointCorruptError: the remote manifest or data does not verify.
            OSError: the storage kept failing.
        """
        started = self._clock()
        base = f"{rc.run_id}/{kind}/"
        raw = self._call(
            lambda: self._store.read_bounded(base + manifest_name(step), MAX_MANIFEST_BYTES),
            started,
        )
        probe = rc._cand_from_bytes(raw, kind, manifest_name(step), step)
        manifest = probe.manifest
        if manifest is None:
            raise CheckpointCorruptError(
                "the remote manifest does not verify", reason=probe.reason or "manifest_invalid"
            )
        key = base + manifest.path.split("/", 1)[1]
        size = self._call(lambda: self._store.size(key), started)
        if size != manifest.size_bytes:
            raise CheckpointCorruptError(
                "the remote file does not have the manifest's size", reason="size_mismatch"
            )

        def chunks() -> Iterator[bytes]:
            offset = 0
            while offset < manifest.size_bytes:
                want = min(CHUNK, manifest.size_bytes - offset)
                chunk = self._call(
                    functools.partial(self._store.read_range, key, offset, want), started
                )
                if not chunk:
                    return
                offset += len(chunk)
                yield chunk

        return rc.install(kind, step, raw, chunks()) is not None
