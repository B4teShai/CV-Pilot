"""Schema 1: runs, reproducibility rows, artifacts and the Parquet file index."""

from __future__ import annotations

import sqlite3

VERSION = 1
NAME = "initial"

_STATUSES = "'running','resumed','finished','failed','preempted'"


def upgrade(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE run_meta (
            run_id TEXT PRIMARY KEY,
            project TEXT NOT NULL,
            name TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ({_STATUSES})),
            parent_run_id TEXT REFERENCES run_meta (run_id),
            task TEXT NOT NULL CHECK (task IN ('detection','classification')),
            framework TEXT NOT NULL,
            host TEXT NOT NULL,
            started_at TEXT NOT NULL,
            ended_at TEXT,
            schema_version INTEGER NOT NULL,
            CHECK (parent_run_id IS NULL OR parent_run_id <> run_id)
        )
        """
    )
    conn.execute("CREATE INDEX idx_run_meta_started ON run_meta (started_at, run_id)")
    conn.execute(
        """
        CREATE TABLE repro (
            run_id TEXT PRIMARY KEY REFERENCES run_meta (run_id),
            git_commit TEXT,
            git_dirty INTEGER NOT NULL,
            git_diff_sha256 TEXT,
            packages_lock_sha256 TEXT NOT NULL,
            python_version TEXT NOT NULL,
            cuda TEXT,
            gpu_names TEXT NOT NULL,
            dataset_uri TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            config TEXT NOT NULL,
            seed INTEGER,
            command_line TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE artifact (
            run_id TEXT NOT NULL REFERENCES run_meta (run_id),
            sha256 TEXT NOT NULL,
            kind TEXT NOT NULL,
            uri TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            meta TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, sha256)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE metric_files (
            run_id TEXT NOT NULL REFERENCES run_meta (run_id),
            seq INTEGER NOT NULL,
            file TEXT NOT NULL,
            n_points INTEGER NOT NULL,
            min_step INTEGER NOT NULL,
            max_step INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, seq)
        )
        """
    )
