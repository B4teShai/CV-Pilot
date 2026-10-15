"""Additive tables and guards: created idempotently when a store is opened, no version bump.

``docs/architecture/03-data-model.md``: adding a table is a minor schema change, so readers of
the same ``schema_version`` tolerate it. A migration (``store/migrations``) is reserved for
changes that rename, remove or re-key. ``ensure`` looks for the tables and triggers first and
only takes the write lock when something is missing, so opening a store to read it never
writes. It fails loudly instead of recreating an empty ``eval_snapshot`` table when snapshot
files exist on disk (the table was dropped), and when the table has an unknown layout.

The ``baseline`` table (pins of a snapshot under a tag, append-only, digested and chained, see
``store/baselines.py``) and the ``checkpoint_manifest`` table (the digest of every checkpoint
manifest, the external anchor of the checkpoint directory, see ``store/checkpoints.py``) are
created the same way. The triggers (no UPDATE or DELETE on
``eval_snapshot``, ``artifact``, ``baseline`` and ``checkpoint_manifest``) are defence in depth
against mistakes and casual edits. Anyone who can write the database file can drop them; the
real control is the SHA-256 digest of every snapshot file and the hash chain over the rows
(``store/snapshots.py``).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from cvpilot.core.errors import StoreError

COLUMNS = (
    "run_id", "eval_set_hash", "step", "task", "n_images", "confusion_sha256",
    "predictions_sha256", "file", "created_at", "snapshot_sha256", "chain_sha256",
)  # fmt: skip

EVAL_SNAPSHOT = (
    """
    CREATE TABLE IF NOT EXISTS eval_snapshot (
        run_id TEXT NOT NULL REFERENCES run_meta (run_id),
        eval_set_hash TEXT NOT NULL,
        step INTEGER NOT NULL CHECK (step >= 0),
        task TEXT NOT NULL,
        n_images INTEGER NOT NULL,
        confusion_sha256 TEXT NOT NULL,
        predictions_sha256 TEXT NOT NULL,
        file TEXT NOT NULL,
        created_at TEXT NOT NULL,
        snapshot_sha256 TEXT NOT NULL,
        chain_sha256 TEXT NOT NULL,
        PRIMARY KEY (run_id, eval_set_hash, step),
        UNIQUE (run_id, file),
        FOREIGN KEY (run_id, confusion_sha256) REFERENCES artifact (run_id, sha256),
        FOREIGN KEY (run_id, predictions_sha256) REFERENCES artifact (run_id, sha256)
    )
    """,
)

BASELINE_COLUMNS = (
    "seq", "tag", "run_id", "eval_set_hash", "step", "snapshot_sha256", "pinned_at", "record",
    "record_sha256", "chain_sha256",
)  # fmt: skip

BASELINE = (
    """
    CREATE TABLE IF NOT EXISTS baseline (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        tag TEXT NOT NULL,
        run_id TEXT NOT NULL REFERENCES run_meta (run_id),
        eval_set_hash TEXT NOT NULL,
        step INTEGER NOT NULL CHECK (step >= 0),
        snapshot_sha256 TEXT NOT NULL,
        pinned_at TEXT NOT NULL,
        record TEXT NOT NULL,
        record_sha256 TEXT NOT NULL,
        chain_sha256 TEXT NOT NULL,
        FOREIGN KEY (run_id, eval_set_hash, step)
            REFERENCES eval_snapshot (run_id, eval_set_hash, step)
    )
    """,
    "CREATE INDEX IF NOT EXISTS baseline_tag ON baseline (tag, seq)",
)

CHECKPOINT_COLUMNS = (
    "run_id", "kind", "step", "sha256", "size_bytes", "manifest_sha256", "chain_sha256", "record",
    "created_at",
)  # fmt: skip

CHECKPOINT = (
    """
    CREATE TABLE IF NOT EXISTS checkpoint_manifest (
        run_id TEXT NOT NULL REFERENCES run_meta (run_id),
        kind TEXT NOT NULL CHECK (kind IN ('last', 'best')),
        step INTEGER NOT NULL CHECK (step >= 0),
        sha256 TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        manifest_sha256 TEXT NOT NULL,
        chain_sha256 TEXT NOT NULL,
        record TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY (run_id, kind, step)
    )
    """,
)

_TRIGGER_BODY = "BEGIN SELECT RAISE(ABORT, '{table} rows are immutable'); END"
TRIGGERS = {
    f"{table}_no_{op}": (
        f"CREATE TRIGGER IF NOT EXISTS {table}_no_{op} BEFORE {op.upper()} ON {table} "
        + _TRIGGER_BODY.format(table=table if table != "eval_snapshot" else "eval snapshot")
    )
    for table in ("eval_snapshot", "artifact", "baseline", "checkpoint_manifest")
    for op in ("update", "delete")
}


def ensure(conn: sqlite3.Connection, evals_dir: Path) -> None:
    """Create the additive tables and triggers that are missing.

    Raises:
        StoreError: ``eval_snapshot`` is missing but snapshot files exist, or it has an
            unknown column layout.
    """
    present = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    triggers = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
    if "eval_snapshot" in present:
        columns = tuple(r[1] for r in conn.execute("PRAGMA table_info(eval_snapshot)"))
        if columns != COLUMNS:
            raise StoreError("eval_snapshot has an unsupported column layout")
    elif evals_dir.is_dir() and any(evals_dir.glob("*/*.json")):
        raise StoreError(
            "the eval_snapshot table is missing but snapshot files exist; "
            "refusing to recreate it empty (restore runs.sqlite)"
        )
    if "baseline" in present:
        columns = tuple(r[1] for r in conn.execute("PRAGMA table_info(baseline)"))
        if columns != BASELINE_COLUMNS:
            raise StoreError("baseline has an unsupported column layout")
    if "checkpoint_manifest" in present:
        columns = tuple(r[1] for r in conn.execute("PRAGMA table_info(checkpoint_manifest)"))
        if columns != CHECKPOINT_COLUMNS:
            raise StoreError("checkpoint_manifest has an unsupported column layout")
    if {"eval_snapshot", "baseline", "checkpoint_manifest"} <= present and set(
        TRIGGERS
    ) <= triggers:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in (*EVAL_SNAPSHOT, *BASELINE, *CHECKPOINT):
            conn.execute(statement)
        for statement in TRIGGERS.values():
            conn.execute(statement)
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
