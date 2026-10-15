"""Baselines in the local store: append-only pins of a snapshot under a tag.

Each ``baseline`` row holds the canonical JSON of the :class:`~cvpilot.core.schema.Baseline`
(``record``), its SHA-256 and a chain digest ``sha256(previous_chain, seq, tag, record_sha256)``
over all rows of the project in ``seq`` order. The record also carries the digest of the pinned
snapshot file. Pinning an existing tag appends a newer row (``replace``); nothing is updated or
deleted, so the history stays verifiable and a replacement is always visible.

What every read verifies: the record bytes against ``record_sha256``, the record against the
row's columns, the whole chain from the first row (an edited, removed, inserted or reordered row
breaks it), that the pinned snapshot still exists and that its stored digest equals the one
recorded when it was pinned. Callers that use the snapshot then read it through
``get_snapshot``, which verifies the file and the run's snapshot chain.

What it does not defend against: someone who can write the database and the snapshot files can
recompute every digest (nothing is signed); deleting the newest row or dropping the table is
invisible without an external anchor. Both make an affected tag unknown, and an unknown tag
fails every gate.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cvpilot.core.errors import (
    BaselineExistsError,
    BaselineIntegrityError,
    BaselineNotFoundError,
    BaselineRefusedError,
    RunNotFoundError,
    StoreError,
)
from cvpilot.core.schema import Baseline, is_valid_tag
from cvpilot.store.paths import require_run_id

if TYPE_CHECKING:
    from cvpilot.store.local import LocalRunStore

GENESIS = "0" * 64
MAX_BASELINE_ROWS = 10_000
"""Most baseline rows (pins and replacements) a project may hold; bounds the chain walk."""
MAX_RECORD_BYTES = 4096
"""Largest ``record`` read back; a real record is under 400 bytes."""


def chain_digest(previous: str, seq: int, tag: str, record_sha256: str) -> str:
    text = f"{previous}\n{seq}\n{tag}\n{record_sha256}"
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _ts(value: datetime) -> str:
    from cvpilot.store import local

    return local._ts(value)


def put_baseline(store: LocalRunStore, baseline: Baseline, *, replace: bool = False) -> None:
    """Append a pin; see ``LocalRunStore.put_baseline``."""
    run_id = require_run_id(baseline.run_id)
    fields = {
        "project": baseline.project,
        "tag": baseline.tag,
        "run_id": run_id,
        "eval_set_hash": baseline.eval_set_hash,
    }
    record = baseline.model_dump_json()
    digest = hashlib.sha256(record.encode("utf-8")).hexdigest()
    with store._tx() as conn:
        row = conn.execute("SELECT project FROM run_meta WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(f"run {run_id} not found in {store.path}")
        if row[0] != baseline.project:
            raise BaselineRefusedError(
                f"run {run_id} belongs to project {row[0]!r}, not {baseline.project!r}", **fields
            )
        snap = conn.execute(
            "SELECT snapshot_sha256 FROM eval_snapshot WHERE run_id=? AND eval_set_hash=? "
            "AND step=?",
            (run_id, baseline.eval_set_hash, baseline.step),
        ).fetchone()
        if snap is None:
            raise BaselineRefusedError(
                f"run {run_id} has no snapshot for eval set {baseline.eval_set_hash[:12]} "
                f"at step {baseline.step}",
                **fields,
            )
        if snap[0] != baseline.snapshot_sha256:
            raise BaselineRefusedError(
                f"the snapshot of run {run_id} changed since it was verified; not pinning it",
                **fields,
            )
        total = conn.execute("SELECT count(*) FROM baseline").fetchone()[0]
        if total >= MAX_BASELINE_ROWS:
            raise StoreError(f"too many baseline rows (limit {MAX_BASELINE_ROWS})")
        if (
            not replace
            and conn.execute("SELECT 1 FROM baseline WHERE tag=?", (baseline.tag,)).fetchone()
        ):
            raise BaselineExistsError(
                f"tag {baseline.tag!r} is already pinned; pass replace to pin it again", **fields
            )
        last = conn.execute(
            "SELECT seq, chain_sha256 FROM baseline ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        seq = (last[0] if last else 0) + 1
        chain = chain_digest(last[1] if last else GENESIS, seq, baseline.tag, digest)
        try:
            conn.execute(
                "INSERT INTO baseline (seq, tag, run_id, eval_set_hash, step, snapshot_sha256, "
                "pinned_at, record, record_sha256, chain_sha256) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    seq, baseline.tag, run_id, baseline.eval_set_hash, baseline.step,
                    baseline.snapshot_sha256, _ts(baseline.pinned_at), record, digest, chain,
                ),
            )  # fmt: skip
        except sqlite3.IntegrityError as exc:
            raise BaselineRefusedError(f"cannot pin tag {baseline.tag!r}: {exc}", **fields) from exc


_COLUMNS = (
    "seq, tag, run_id, eval_set_hash, step, snapshot_sha256, pinned_at, record, "
    "record_sha256, chain_sha256"
)


def _fail(message: str) -> BaselineIntegrityError:
    return BaselineIntegrityError(message)


def _row_baseline(store: LocalRunStore, row: tuple[Any, ...]) -> Baseline:
    seq, tag, run_id, eval_set_hash, step, snap_sha, pinned_at, record, record_sha, _ = row
    if not isinstance(record, str) or len(record) > MAX_RECORD_BYTES:
        raise _fail(f"baseline row {seq} has an invalid record")
    if hashlib.sha256(record.encode("utf-8")).hexdigest() != record_sha:
        raise _fail(f"baseline row {seq}: record digest mismatch")
    try:
        baseline = Baseline.model_validate_json(record)
    except ValueError:
        raise _fail(f"baseline row {seq}: the record is not a valid baseline") from None
    try:
        pinned = datetime.fromisoformat(pinned_at)
    except (TypeError, ValueError):
        raise _fail(f"baseline row {seq}: invalid pinned_at") from None
    columns = {
        "tag": tag, "run_id": run_id, "eval_set_hash": eval_set_hash, "step": step,
        "snapshot_sha256": snap_sha, "pinned_at": pinned,
    }  # fmt: skip
    for name, value in columns.items():
        if getattr(baseline, name) != value:
            raise _fail(f"baseline row {seq}: {name} differs from its record")
    current = store._query(
        "SELECT snapshot_sha256 FROM eval_snapshot WHERE run_id=? AND eval_set_hash=? AND step=?",
        (run_id, eval_set_hash, step),
    )
    if not current or current[0][0] != baseline.snapshot_sha256:
        raise _fail(
            f"baseline {tag!r} (row {seq}): the pinned snapshot is missing or changed since pinning"
        )
    return baseline


def _all_rows(store: LocalRunStore) -> list[tuple[Any, ...]]:
    count = store._query("SELECT count(*) FROM baseline")[0][0]
    if count > MAX_BASELINE_ROWS:
        raise StoreError(f"too many baseline rows ({count}; limit {MAX_BASELINE_ROWS})")
    rows = store._query(f"SELECT {_COLUMNS} FROM baseline ORDER BY seq")  # noqa: S608
    previous = GENESIS
    for row in rows:
        if chain_digest(previous, row[0], row[1], row[8]) != row[9]:
            raise _fail(f"baseline hash chain is broken at row {row[0]} (tag {str(row[1])[:32]!r})")
        previous = row[9]
    return rows


def get_baseline(store: LocalRunStore, tag: str) -> Baseline:
    """The newest pin of ``tag`` after verifying the whole baseline chain."""
    if not is_valid_tag(tag):
        raise BaselineNotFoundError(f"{tag[:40]!r} is not a valid baseline tag")
    rows = [row for row in _all_rows(store) if row[1] == tag]
    if not rows:
        raise BaselineNotFoundError(f"no baseline is pinned under tag {tag!r}")
    return _row_baseline(store, rows[-1])


def list_baselines(store: LocalRunStore, *, history: bool = False) -> list[Baseline]:
    """Verified pins: the newest per tag (ordered by tag), or every row in pin order."""
    rows = _all_rows(store)
    if not history:
        newest: dict[str, tuple[Any, ...]] = {}
        for row in rows:
            newest[row[1]] = row
        rows = [newest[tag] for tag in sorted(newest)]
    return [_row_baseline(store, row) for row in rows]
