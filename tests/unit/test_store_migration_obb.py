"""Migration 2 (``obb_task``): a schema-1 store opens, keeps every row and accepts obb runs."""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from pathlib import Path

import pytest
from test_store_local import T0, make_run, pts

from cvpilot.core.errors import StoreError
from cvpilot.core.schema import SCHEMA_VERSION
from cvpilot.runs.lifecycle import resume_from
from cvpilot.store import migrations
from cvpilot.store.local import LocalRunStore, open_store


def v1_store(tmp_path: Path) -> tuple[str, str]:
    """A schema-1 store with a failed run, its resumed child, metrics and an artifact."""
    first = migrations.discover()[:1]
    s = LocalRunStore.open("p", tmp_path, migrations=first)
    try:
        assert s.schema_version == 1
        ended = T0 + timedelta(hours=1)
        parent, repro = make_run(schema_version=1, status="failed", ended_at=ended)
        s.put_run(parent, repro)
        child, child_repro = make_run(
            schema_version=1, status="resumed", parent_run_id=parent.run_id
        )
        s.put_run(child, child_repro)
        s.append_points(pts(child.run_id, "loss", range(3)))
        s.put_artifact_bytes(child.run_id, b"weights", "checkpoint", {})
        return parent.run_id, child.run_id
    finally:
        s.close()


def test_a_schema_1_store_migrates_with_a_backup_and_keeps_every_row(tmp_path: Path) -> None:
    parent_id, child_id = v1_store(tmp_path)
    with pytest.raises(StoreError, match="older than"):
        open_store("p", tmp_path, read_only=True)  # a reader never migrates
    s = open_store("p", tmp_path)
    try:
        assert s.schema_version == SCHEMA_VERSION == 2
        assert (tmp_path / "p" / "runs.sqlite.bak-1").exists()
        child = s.get_run(child_id)
        assert child.parent_run_id == parent_id and child.schema_version == 1
        assert s._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert s._conn.execute("PRAGMA foreign_key_check").fetchall() == []
        assert s._conn.execute(
            "SELECT count(*) FROM repro JOIN run_meta USING (run_id)"
        ).fetchone() == (2,)
        indexes = {r[1] for r in s._conn.execute("PRAGMA index_list(run_meta)")}
        assert "idx_run_meta_started" in indexes
        meta, repro = make_run(task="obb")
        s.put_run(meta, repro)
        assert s.get_run(meta.run_id).task == "obb"
        with pytest.raises(sqlite3.IntegrityError):
            with s._tx() as conn:  # the widened CHECK still rejects unknown tasks
                conn.execute("UPDATE run_meta SET task='segment' WHERE run_id=?", (meta.run_id,))
        parent = s.get_run(parent_id)
        resumed = resume_from(parent, run_id=make_run()[0].run_id, host="h", name=None)
        assert resumed.schema_version == SCHEMA_VERSION  # a run of schema 1 resumes in a new store
    finally:
        s.close()
    reader = open_store("p", tmp_path, read_only=True)
    reader.close()


def test_a_dangling_reference_rolls_the_migration_back(tmp_path: Path) -> None:
    v1_store(tmp_path)
    db = sqlite3.connect(tmp_path / "p" / "runs.sqlite")
    db.execute("PRAGMA foreign_keys=OFF")
    db.execute("UPDATE repro SET run_id='01ZZZZZZZZZZZZZZZZZZZZZZZZ' WHERE rowid=1")
    db.commit()
    db.close()
    with pytest.raises(StoreError, match="reference a missing run"):
        open_store("p", tmp_path)
    db = sqlite3.connect(tmp_path / "p" / "runs.sqlite")
    assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("1",)
    db.close()
