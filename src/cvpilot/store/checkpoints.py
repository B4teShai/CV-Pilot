"""The checkpoint index: one append-only row per saved checkpoint, holding the SHA-256 of the
exact manifest file bytes.

The checkpoint directory protects itself with digests and chains, but whoever can rewrite that
directory can recompute all of them. This table lives in the run store, a different place with
different write access (and on another disk when the checkpoints sit on a mounted bucket), so
a restore that consults it detects a manifest that was swapped, forged or rolled back in the
directory. Rows are written after the checkpoint verified and before the directory's pointer
moves, are never updated or deleted (triggers), and recording the same digest twice is a no-op.
It does not defend against someone who can write both the directory and ``runs.sqlite``.
"""

from __future__ import annotations

import re
import sqlite3
from typing import TYPE_CHECKING, Any

from cvpilot.core.errors import RunNotFoundError, StoreError
from cvpilot.core.schema import MAX_STEP, CheckpointManifest
from cvpilot.store.paths import require_run_id

if TYPE_CHECKING:
    from cvpilot.store.local import LocalRunStore

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _key(run_id: str, kind: str, step: int) -> tuple[str, str, int]:
    require_run_id(run_id)
    if kind not in ("last", "best"):
        raise StoreError("checkpoint kind must be 'last' or 'best'")
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= MAX_STEP:
        raise StoreError(f"step must be an integer in [0, {MAX_STEP}]")
    return run_id, kind, step


def put_manifest(store: LocalRunStore, manifest: CheckpointManifest, manifest_sha256: str) -> None:
    """Record a manifest digest; see ``LocalRunStore.put_manifest``."""
    from cvpilot.store import local

    if not _SHA256.fullmatch(manifest_sha256):
        raise StoreError("manifest digest must be 64 lower-case hex characters")
    run_id, kind, step = _key(manifest.run_id, manifest.kind, manifest.step)
    record = manifest.model_dump_json()
    with store._tx() as conn:
        if conn.execute("SELECT 1 FROM run_meta WHERE run_id=?", (run_id,)).fetchone() is None:
            raise RunNotFoundError(f"run {run_id} not found in {store.path}")
        row = conn.execute(
            "SELECT manifest_sha256 FROM checkpoint_manifest WHERE run_id=? AND kind=? AND step=?",
            (run_id, kind, step),
        ).fetchone()
        if row is not None:
            if row[0] != manifest_sha256:
                raise StoreError(
                    f"{kind} checkpoint at step {step} of run {run_id} is already recorded "
                    "with another digest; checkpoints are immutable"
                )
            return
        try:
            conn.execute(
                "INSERT INTO checkpoint_manifest VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    run_id, kind, step, manifest.sha256, manifest.size_bytes, manifest_sha256,
                    manifest.chain_sha256, record, local._ts(local._now()),
                ),
            )  # fmt: skip
        except sqlite3.IntegrityError as exc:
            raise StoreError(f"cannot record the checkpoint: {exc}") from exc


def get_manifest_digest(store: LocalRunStore, run_id: str, kind: str, step: int) -> str | None:
    """The recorded manifest digest of ``(run, kind, step)``, or None when none was recorded."""
    key = _key(run_id, kind, step)
    rows = store._query(
        "SELECT manifest_sha256 FROM checkpoint_manifest WHERE run_id=? AND kind=? AND step=?", key
    )
    if not rows:
        return None
    digest = rows[0][0]
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise StoreError("a recorded checkpoint digest is malformed")
    return digest


def max_manifest_step(store: LocalRunStore, run_id: str, kind: str) -> int | None:
    """The highest step recorded for ``(run, kind)``: the store's memory of how far saves got."""
    _key(run_id, kind, 0)
    rows = store._query(
        "SELECT max(step) FROM checkpoint_manifest WHERE run_id=? AND kind=?", (run_id, kind)
    )
    value = rows[0][0] if rows else None
    return None if value is None else int(value)


def auth_evidence(store: LocalRunStore, run_id: str, kind: str) -> tuple[int | None, list[str]]:
    """``(first tagged step, key ids)`` among the recorded manifests of ``(run, kind)``: the store's
    memory of whether, and with which key, this run was ever authenticated."""
    _key(run_id, kind, 0)
    rows = store._query(
        "SELECT step, record FROM checkpoint_manifest WHERE run_id=? AND kind=? ORDER BY step",
        (run_id, kind),
    )
    first: int | None = None
    key_ids: set[str] = set()
    for step, record in rows:
        try:
            manifest = CheckpointManifest.model_validate_json(record)
        except ValueError:
            continue
        if manifest.auth_tag is not None:
            first = step if first is None else min(first, step)
            if manifest.key_id is not None:
                key_ids.add(manifest.key_id)
    return first, sorted(key_ids)


def list_manifests(store: LocalRunStore, run_id: str) -> list[CheckpointManifest]:
    """The recorded manifests of a run in step order (kind, then step)."""
    require_run_id(run_id)
    rows: list[tuple[Any, ...]] = store._query(
        "SELECT record FROM checkpoint_manifest WHERE run_id=? ORDER BY kind, step", (run_id,)
    )
    out: list[CheckpointManifest] = []
    for (record,) in rows:
        try:
            out.append(CheckpointManifest.model_validate_json(record))
        except ValueError as exc:
            raise StoreError(f"a recorded checkpoint manifest is invalid: {exc}") from exc
    return sorted(out, key=lambda m: (m.step, m.kind))
