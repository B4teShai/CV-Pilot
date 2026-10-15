"""The checkpoint index in the run store: append-only digests that anchor the directory."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from support import make_run, open_project

from cvpilot.core.errors import RunNotFoundError, StoreError
from cvpilot.core.hashing import checkpoint_chain, new_run_id
from cvpilot.core.schema import GENESIS_DIGEST, CheckpointManifest
from cvpilot.store import additive
from cvpilot.store.local import LocalRunStore

SHA = "a" * 64


def manifest_for(
    run_id: str, step: int = 1, kind: str = "last", sha: str = SHA
) -> CheckpointManifest:
    return CheckpointManifest(
        run_id=run_id,
        step=step,
        kind=kind,  # type: ignore[arg-type]
        path=f"{kind}/step_{step:09d}.safetensors",
        sha256=sha,
        size_bytes=10,
        format="safetensors",
        metric_key="m" if kind == "best" else None,
        metric_value=0.5 if kind == "best" else None,
        versions={},
        dataset_hash="merkle:" + SHA,
        created_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
        prev_manifest_sha256=GENESIS_DIGEST,
        prev_chain_sha256=GENESIS_DIGEST,
        chain_sha256=checkpoint_chain(GENESIS_DIGEST, run_id, kind, step, sha, 10),
    )


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = open_project(tmp_path)
    yield s
    s.close()


def test_a_digest_is_stored_once_and_read_back(store: LocalRunStore) -> None:
    run = make_run(store)
    assert store.get_manifest_digest(run, "last", 1) is None
    store.put_manifest(manifest_for(run), "d" * 64)
    assert store.get_manifest_digest(run, "last", 1) == "d" * 64
    assert store.get_manifest_digest(run, "best", 1) is None
    assert store.get_manifest_digest(run, "last", 2) is None


def test_recording_the_same_digest_again_is_a_no_op_and_another_digest_is_refused(
    store: LocalRunStore,
) -> None:
    run = make_run(store)
    store.put_manifest(manifest_for(run), "d" * 64)
    store.put_manifest(manifest_for(run), "d" * 64)
    with pytest.raises(StoreError):
        store.put_manifest(manifest_for(run), "e" * 64)
    assert store.get_manifest_digest(run, "last", 1) == "d" * 64


def test_unknown_runs_and_malformed_digests_are_refused(store: LocalRunStore) -> None:
    with pytest.raises(RunNotFoundError):
        store.put_manifest(manifest_for(new_run_id()), "d" * 64)
    run = make_run(store)
    with pytest.raises(StoreError):
        store.put_manifest(manifest_for(run), "not-a-digest")
    with pytest.raises(Exception):  # noqa: B017  # an invalid run id never reaches SQL
        store.get_manifest_digest("x", "last", 1)
    with pytest.raises(StoreError):
        store.get_manifest_digest(run, "other", 1)  # type: ignore[arg-type]
    with pytest.raises(StoreError):
        store.get_manifest_digest(run, "last", -1)


def test_rows_cannot_be_updated_or_deleted(store: LocalRunStore) -> None:
    run = make_run(store)
    store.put_manifest(manifest_for(run), "d" * 64)
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("UPDATE checkpoint_manifest SET manifest_sha256='x'")
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("DELETE FROM checkpoint_manifest")


def test_list_manifests_returns_the_recorded_records_in_step_order(store: LocalRunStore) -> None:
    run = make_run(store)
    for step in (3, 1, 2):
        store.put_manifest(manifest_for(run, step), "d" * 64)
    assert [m.step for m in store.list_manifests(run)] == [1, 2, 3]


def test_an_unknown_table_layout_is_refused_at_open(tmp_path: Path) -> None:
    s = open_project(tmp_path)
    for trigger in [t for t in additive.TRIGGERS if t.startswith("checkpoint_manifest")]:
        s._conn.execute(f"DROP TRIGGER {trigger}")
    s._conn.execute("DROP TABLE checkpoint_manifest")
    s._conn.execute("CREATE TABLE checkpoint_manifest (x INTEGER)")
    db = s.path
    s.close()
    with pytest.raises(StoreError):
        LocalRunStore.open("proj", db.parent)


def test_an_older_store_gets_the_table_added_without_a_version_bump(tmp_path: Path) -> None:
    s = open_project(tmp_path)
    version = s.schema_version
    for trigger in [t for t in additive.TRIGGERS if t.startswith("checkpoint_manifest")]:
        s._conn.execute(f"DROP TRIGGER {trigger}")
    s._conn.execute("DROP TABLE checkpoint_manifest")
    root = s.path.parent
    s.close()
    again = LocalRunStore.open("proj", root)
    try:
        assert again.schema_version == version
        run = make_run(again)
        again.put_manifest(manifest_for(run), "d" * 64)
    finally:
        again.close()
