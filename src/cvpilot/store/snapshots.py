"""Eval snapshots in the local store: immutable, digested and hash-chained.

Each snapshot is a JSON file ``evals/<run_id>/<eval_set_hash[:12]>_<step:09d>.json`` (mode 0444,
created exclusively, never replaced) plus an index row in ``eval_snapshot``. The row holds the
SHA-256 of the exact bytes written and a chain digest
``sha256(previous_chain, run_id, eval_set_hash, step, snapshot_sha256)`` over the run's rows
in insertion order.

What this defends against: edits of a snapshot file (digest mismatch on every read), swapped
artifact addresses or changed key fields (the row is compared with the file), deletion or
reordering of an earlier row (chain mismatch), and two concurrent writers (existence check,
file creation and insert run in one ``BEGIN IMMEDIATE`` transaction).

What it does not defend against: someone who can write both the database and the files can
recompute every digest (nothing is signed), and deleting the newest row is invisible without
an external anchor of the last chain value. Database triggers are defence in depth.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Any

from cvpilot.core.errors import (
    SnapshotExistsError,
    SnapshotIntegrityError,
    SnapshotNotFoundError,
    StoreError,
)
from cvpilot.core.schema import MAX_STEP, EvalSnapshot
from cvpilot.store.paths import require_run_id, require_sha256

if TYPE_CHECKING:
    from cvpilot.store.local import LocalRunStore

MAX_SNAPSHOT_BYTES = 64 * 1024 * 1024
"""Largest snapshot JSON file read back; the writer produces far less (a few KB per class)."""
GENESIS = "0" * 64


def chain_digest(previous: str, run_id: str, eval_set_hash: str, step: int, digest: str) -> str:
    text = f"{previous}\n{run_id}\n{eval_set_hash}\n{step}\n{digest}"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _check_step(step: int) -> int:
    if isinstance(step, bool) or not isinstance(step, int) or not 0 <= step <= MAX_STEP:
        raise StoreError(f"step must be an integer in [0, {MAX_STEP}]")
    return step


def snapshot_path(store: LocalRunStore, run_id: str, name: str) -> Path:
    base = (store.path / "evals").resolve()
    target = (base / run_id / name).resolve()
    if not target.is_relative_to(base):
        raise StoreError("snapshot path resolves outside the store")
    return target


def has_snapshot(store: LocalRunStore, run_id: str, eval_set_hash: str, step: int) -> bool:
    require_run_id(run_id)
    require_sha256(eval_set_hash)
    _check_step(step)
    return bool(
        store._query(
            "SELECT 1 FROM eval_snapshot WHERE run_id=? AND eval_set_hash=? AND step=?",
            (run_id, eval_set_hash, step),
        )
    )


def _read_existing(path: Path) -> bytes | None:
    try:
        with path.open("rb") as handle:
            return handle.read(MAX_SNAPSHOT_BYTES + 1)
    except FileNotFoundError:
        return None


def _create_exclusive(target: Path, data: bytes) -> bool:
    """Create ``target`` (0444) with ``data`` unless it exists; True if this call created it.

    The bytes are written to a private temporary name and hard-linked into place, so the final
    name never holds a partial file and an existing file is never replaced. An existing file
    with identical bytes is adopted (a crash left it without a row); different bytes raise.
    """
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o444)
        try:
            os.link(tmp, target)
        except FileExistsError:
            existing = _read_existing(target)
            if existing == data:
                return False
            raise StoreError(
                f"snapshot file {target.name} already exists with other content"
            ) from None
        except OSError as exc:
            if exc.errno not in (errno.EPERM, errno.ENOTSUP, errno.EXDEV, errno.EMLINK):
                raise
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
            try:
                out = os.open(target, flags, 0o444)
            except FileExistsError:
                raise StoreError(f"snapshot file {target.name} already exists") from None
            with os.fdopen(out, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        return True
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink()


def put_snapshot(store: LocalRunStore, snapshot: EvalSnapshot) -> None:
    """Persist a snapshot once; see the module docstring and ``LocalRunStore.put_snapshot``."""
    from cvpilot.store import local

    run_id = require_run_id(snapshot.run_id)
    store._require_run(run_id)
    _check_step(snapshot.step)
    data = snapshot.model_dump_json().encode("utf-8")
    digest = hashlib.sha256(data).hexdigest()
    name = f"{snapshot.eval_set_hash[:12]}_{snapshot.step:09d}.json"
    target = snapshot_path(store, run_id, name)
    exists = SnapshotExistsError(
        f"snapshot ({snapshot.eval_set_hash[:12]}, step {snapshot.step}) of run {run_id} "
        "exists; snapshots are immutable"
    )
    created = False
    try:
        with store._tx() as conn:
            for column in ("confusion", "predictions"):
                sha = getattr(snapshot, f"{column}_sha256")
                rows = conn.execute(
                    "SELECT kind FROM artifact WHERE run_id=? AND sha256=?", (run_id, sha)
                ).fetchall()
                if not rows or rows[0][0] != column:
                    raise StoreError(f"run {run_id} has no {column} artifact {sha[:12]}")
            if conn.execute(
                "SELECT 1 FROM eval_snapshot WHERE run_id=? AND eval_set_hash=? AND step=?",
                (run_id, snapshot.eval_set_hash, snapshot.step),
            ).fetchone():
                raise exists
            if conn.execute(
                "SELECT 1 FROM eval_snapshot WHERE run_id=? AND file=?", (run_id, name)
            ).fetchone():
                raise StoreError(f"snapshot file name {name} of run {run_id} is already taken")
            last = conn.execute(
                "SELECT chain_sha256 FROM eval_snapshot WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
            chain = chain_digest(
                last[0] if last else GENESIS, run_id, snapshot.eval_set_hash, snapshot.step, digest
            )
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                created = _create_exclusive(target, data)
            except OSError as exc:
                raise StoreError(f"cannot write snapshot {name}: {exc}") from exc
            local._fsync_dir(target.parent)
            conn.execute(
                "INSERT INTO eval_snapshot VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id, snapshot.eval_set_hash, snapshot.step, snapshot.task,
                    snapshot.n_images, snapshot.confusion_sha256, snapshot.predictions_sha256,
                    name, local._ts(local._now()), digest, chain,
                ),
            )  # fmt: skip
    except sqlite3.IntegrityError as exc:
        if created:
            with contextlib.suppress(OSError):
                target.unlink()
        raise exists from exc
    except BaseException:
        if created:
            with contextlib.suppress(OSError):
                target.unlink()
        raise


def _read_snapshot(store: LocalRunStore, run_id: str, row: tuple[Any, ...]) -> EvalSnapshot:
    """Read and verify one snapshot; ``row`` is an ``eval_snapshot`` index row."""
    eval_set_hash, step, task, n_images, conf, pred, name, digest = row[:8]
    path = snapshot_path(store, run_id, name)
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        raise SnapshotIntegrityError(
            f"snapshot file {name} of run {run_id} is missing although the index lists it"
        ) from None
    except OSError as exc:
        raise StoreError(f"cannot read snapshot {name}: {exc}") from exc
    if size > MAX_SNAPSHOT_BYTES:
        raise StoreError(f"snapshot file {name} is too large ({size} bytes)")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise StoreError(f"cannot read snapshot {name}: {exc}") from exc
    if hashlib.sha256(data).hexdigest() != digest:
        raise SnapshotIntegrityError(f"snapshot file {name} of run {run_id}: digest mismatch")
    try:
        snapshot = EvalSnapshot.model_validate_json(data)
    except ValueError as exc:
        raise StoreError(f"cannot read snapshot {name} of run {run_id}: {exc}") from exc
    expected = {
        "run_id": run_id, "eval_set_hash": eval_set_hash, "step": step, "task": task,
        "n_images": n_images, "confusion_sha256": conf, "predictions_sha256": pred,
    }  # fmt: skip
    for field, value in expected.items():
        if getattr(snapshot, field) != value:
            raise SnapshotIntegrityError(
                f"snapshot file {name} of run {run_id}: {field} differs from the index row"
            )
    return snapshot


def verify_chain(store: LocalRunStore, run_id: str) -> None:
    """Recompute the hash chain of a run's snapshot rows.

    Raises:
        SnapshotIntegrityError: a row was changed, removed from the middle or reordered.
    """
    previous = GENESIS
    rows = store._query(
        "SELECT eval_set_hash, step, snapshot_sha256, chain_sha256 FROM eval_snapshot "
        "WHERE run_id=? ORDER BY rowid",
        (run_id,),
    )
    for eval_set_hash, step, digest, chain in rows:
        if chain_digest(previous, run_id, eval_set_hash, step, digest) != chain:
            raise SnapshotIntegrityError(
                f"snapshot hash chain of run {run_id} is broken at "
                f"({eval_set_hash[:12]}, step {step})"
            )
        previous = chain


_COLUMNS = (
    "eval_set_hash, step, task, n_images, confusion_sha256, predictions_sha256, file, "
    "snapshot_sha256"
)


def get_snapshot(
    store: LocalRunStore, run_id: str, eval_set_hash: str, step: int | None
) -> EvalSnapshot:
    store._require_run(run_id)
    require_sha256(eval_set_hash)
    sql = f"SELECT {_COLUMNS} FROM eval_snapshot WHERE run_id=? AND eval_set_hash=?"  # noqa: S608
    params: tuple[Any, ...] = (run_id, eval_set_hash)
    if step is not None:
        sql, params = sql + " AND step=?", (*params, _check_step(step))
    rows = store._query(sql + " ORDER BY step DESC LIMIT 1", params)
    if not rows:
        raise SnapshotNotFoundError(
            f"run {run_id} has no snapshot for eval set {eval_set_hash[:12]}"
            + ("" if step is None else f" at step {step}")
        )
    snapshot = _read_snapshot(store, run_id, rows[0])
    verify_chain(store, run_id)
    return snapshot


def list_snapshots(store: LocalRunStore, run_id: str) -> list[EvalSnapshot]:
    store._require_run(run_id)
    rows = store._query(
        f"SELECT {_COLUMNS} FROM eval_snapshot WHERE run_id=? "  # noqa: S608
        "ORDER BY eval_set_hash, step",
        (run_id,),
    )
    found = [_read_snapshot(store, run_id, row) for row in rows]
    verify_chain(store, run_id)
    return found
