"""Schema 2: ``run_meta.task`` also accepts ``obb`` (oriented bounding boxes, ADR-0013).

SQLite cannot alter a CHECK constraint, so the table is rebuilt the documented way: create the
new table, copy every row, drop the old one, rename, recreate its index. The runner turns
foreign-key enforcement off around migrations and runs ``PRAGMA foreign_key_check`` before it
commits, so the rows that reference ``run_meta`` keep pointing at the same run ids. A library of
schema 1 refuses a store of schema 2: it could not read a run of task ``obb``.
"""

from __future__ import annotations

import sqlite3

VERSION = 2
NAME = "obb_task"

_STATUSES = "'running','resumed','finished','failed','preempted'"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE run_meta_v2 (
            run_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ({_STATUSES})),
            parent_run_id TEXT REFERENCES run_meta (run_id),
            task TEXT NOT NULL CHECK (task IN ('detection','classification','obb')),
            framework TEXT NOT NULL,
            host TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            schema_version INTEGER NOT NULL,
            CHECK (parent_run_id IS NULL OR parent_run_id <> run_id)
        )
        """
    )
    conn.execute(
        """
        INSERT INTO run_meta_v2 (run_id, project, name, status, parent_run_id, task, framework,
                                 host, started_at, ended_at, schema_version)
        SELECT run_id, project, name, status, parent_run_id, task, framework, host, started_at,
               ended_at, schema_version
        FROM run_meta
        """
    )
    conn.execute("DROP TABLE run_meta")
    conn.execute("ALTER TABLE run_meta_v2 RENAME TO run_meta")
    conn.execute("CREATE INDEX idx_run_meta_started ON run_meta (started_at, run_id)")
