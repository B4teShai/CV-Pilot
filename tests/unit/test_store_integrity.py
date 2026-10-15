"""Integrity of snapshots and artifacts: digests, chain, exclusive writes, DB-level guards."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cvpilot.core.errors import (
    SnapshotExistsError,
    SnapshotIntegrityError,
    StoreError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, EvalSnapshot, Repro, RunMeta
from cvpilot.store.local import LocalRunStore, open_store

H1, H2 = "1" * 64, "2" * 64
SHA = "b" * 64


def make_run(store: LocalRunStore) -> RunMeta:
    rid = new_run_id()
    meta = RunMeta(
        run_id=rid, project="p", name="n", status="running", task="detection",
        framework="python==3.12.1", host="h",
        started_at=datetime(2026, 10, 14, tzinfo=timezone.utc), schema_version=SCHEMA_VERSION,
    )  # fmt: skip
    repro = Repro(
        run_id=rid, git_dirty=False, packages_lock_sha256=SHA, python_version="3.12",
        gpu_names=[], dataset_uri="/d", dataset_hash="merkle:" + SHA, config={},
        command_line="t.py",
    )  # fmt: skip
    store.put_run(meta, repro)
    return meta


def snap(run_id: str, conf: str, pred: str, *, h: str = H1, step: int = 0, m: float = 0.5) -> Any:
    return EvalSnapshot(
        run_id=run_id, eval_set_hash=h, step=step, task="detection", class_names=["a"],
        n_images=3, metrics={"mAP50_95": m}, per_class={"a": {"support": 3.0}}, slices={},
        confusion_sha256=conf, predictions_sha256=pred,
    )  # fmt: skip


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = open_store("p", tmp_path)
    yield s
    s.close()


@pytest.fixture
def run(store: LocalRunStore) -> RunMeta:
    return make_run(store)


def arts(store: LocalRunStore, rid: str) -> tuple[str, str]:
    c = store.put_artifact_bytes(rid, b"conf", "confusion", {})
    p = store.put_artifact_bytes(rid, b"pred", "predictions", {})
    return c.sha256, p.sha256


def path_of(store: LocalRunStore, rid: str, h: str = H1, step: int = 0) -> Path:
    return store.path / "evals" / rid / f"{h[:12]}_{step:09d}.json"


def test_editing_the_snapshot_file_is_detected_on_every_read(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = arts(store, run.run_id)
    original = snap(run.run_id, conf, pred)
    store.put_snapshot(original)
    path = path_of(store, run.run_id)
    assert path.stat().st_mode & 0o777 == 0o444
    path.chmod(0o644)
    forged = original.model_copy(update={"metrics": {"mAP50_95": 0.99}})
    path.write_text(forged.model_dump_json())
    for call in (
        lambda: store.get_snapshot(run.run_id, H1, 0),
        lambda: store.get_snapshot(run.run_id, H1),
        lambda: store.list_snapshots(run.run_id),
    ):
        with pytest.raises(SnapshotIntegrityError, match="digest"):
            call()
    assert issubclass(SnapshotIntegrityError, StoreError)


def test_swapped_artifact_shas_or_fields_are_detected_even_with_a_matching_digest(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = arts(store, run.run_id)
    store.put_snapshot(snap(run.run_id, conf, pred, step=4))
    # an attacker rewrites file AND digest column; the row fields still disagree
    forged = snap(run.run_id, pred, conf, step=4)
    data = forged.model_dump_json().encode()
    path = path_of(store, run.run_id, step=4)
    path.chmod(0o644)
    path.write_bytes(data)
    store._conn.execute("DROP TRIGGER eval_snapshot_no_update")
    store._conn.execute(
        "UPDATE eval_snapshot SET snapshot_sha256=?", (hashlib.sha256(data).hexdigest(),)
    )
    with pytest.raises(SnapshotIntegrityError, match="confusion"):
        store.get_snapshot(run.run_id, H1, 4)


def test_the_hash_chain_detects_a_deleted_or_reordered_row(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = arts(store, run.run_id)
    for step in (1, 2, 3):
        store.put_snapshot(snap(run.run_id, conf, pred, step=step))
    assert [s.step for s in store.list_snapshots(run.run_id)] == [1, 2, 3]
    store._conn.execute("DROP TRIGGER eval_snapshot_no_delete")
    store._conn.execute("DELETE FROM eval_snapshot WHERE step=2")
    with pytest.raises(SnapshotIntegrityError, match="chain"):
        store.list_snapshots(run.run_id)


def test_snapshot_files_are_created_exclusively_and_never_replaced(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = arts(store, run.run_id)
    path = path_of(store, run.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("squatter")  # an unreferenced file already sits on the name
    with pytest.raises(StoreError, match="already exists"):
        store.put_snapshot(snap(run.run_id, conf, pred))
    assert path.read_text() == "squatter"
    assert store.list_snapshots(run.run_id) == []


def test_an_orphan_file_with_identical_bytes_is_adopted(store: LocalRunStore, run: RunMeta) -> None:
    conf, pred = arts(store, run.run_id)
    s = snap(run.run_id, conf, pred)
    path = path_of(store, run.run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(s.model_dump_json().encode())  # a crash left the file but no row
    store.put_snapshot(s)
    assert store.get_snapshot(run.run_id, H1, 0) == s


def test_two_concurrent_writers_produce_exactly_one_snapshot(tmp_path: Path) -> None:
    first = open_store("p", tmp_path)
    run = make_run(first)
    conf, pred = arts(first, run.run_id)
    second = open_store("p", tmp_path)
    results: list[str] = []
    barrier = threading.Barrier(2)

    def write(store: LocalRunStore, metric: float) -> None:
        barrier.wait()
        try:
            store.put_snapshot(snap(run.run_id, conf, pred, m=metric))
            results.append("ok")
        except SnapshotExistsError:
            results.append("exists")

    threads = [
        threading.Thread(target=write, args=(first, 0.1)),
        threading.Thread(target=write, args=(second, 0.9)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["exists", "ok"]
    stored = first.get_snapshot(run.run_id, H1, 0)  # file and row agree: no torn state
    assert stored.metrics["mAP50_95"] in (0.1, 0.9)
    assert len(first.list_snapshots(run.run_id)) == 1
    first.close()
    second.close()


def test_insert_or_replace_cannot_replace_a_snapshot_row(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = arts(store, run.run_id)
    store.put_snapshot(snap(run.run_id, conf, pred))
    row = store._conn.execute("SELECT * FROM eval_snapshot").fetchone()
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store._conn.execute(
            "INSERT OR REPLACE INTO eval_snapshot VALUES (?,?,?,?,?,?,?,?,?,?,?)", row
        )
    assert store._conn.execute("SELECT COUNT(*) FROM eval_snapshot").fetchone() == (1,)


def test_pragmas_hold_on_every_connection(tmp_path: Path) -> None:
    for _ in range(2):
        s = open_store("p", tmp_path)
        assert s._conn.execute("PRAGMA foreign_keys").fetchone() == (1,)
        assert s._conn.execute("PRAGMA recursive_triggers").fetchone() == (1,)
        s.close()


def test_dropped_triggers_are_recreated_on_open(tmp_path: Path) -> None:
    s = open_store("p", tmp_path)
    s._conn.execute("DROP TRIGGER eval_snapshot_no_update")
    s._conn.execute("DROP TRIGGER artifact_no_delete")
    s.close()
    again = open_store("p", tmp_path)
    names = {
        r[0] for r in again._conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
    }
    assert {"eval_snapshot_no_update", "eval_snapshot_no_delete", "artifact_no_update",
            "artifact_no_delete"} <= names  # fmt: skip
    again.close()


def test_a_dropped_table_with_snapshot_files_on_disk_fails_loudly(tmp_path: Path) -> None:
    s = open_store("p", tmp_path)
    run = make_run(s)
    conf, pred = arts(s, run.run_id)
    s.put_snapshot(snap(run.run_id, conf, pred))
    s._conn.execute("DROP TRIGGER eval_snapshot_no_delete")
    s._conn.execute("DROP TABLE eval_snapshot")
    s.close()
    with pytest.raises(StoreError, match="snapshot files"):
        open_store("p", tmp_path)


def test_artifact_rows_cannot_be_updated_or_deleted(store: LocalRunStore, run: RunMeta) -> None:
    arts(store, run.run_id)
    for sql in ("UPDATE artifact SET kind='diff'", "DELETE FROM artifact"):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            store._conn.execute(sql)


# -- artifacts --------------------------------------------------------------------------------


def test_artifacts_are_read_only_and_a_corrupted_copy_is_healed_on_reingest(
    store: LocalRunStore, run: RunMeta
) -> None:
    art = store.put_artifact_bytes(run.run_id, b"payload", "config", {})
    path = store.artifact_path(art.sha256)
    assert path.stat().st_mode & 0o777 == 0o444
    path.chmod(0o644)
    path.write_bytes(b"tampered!")
    with pytest.raises(StoreError, match="integrity|digest"):
        store.open_artifact(art.sha256)
    again = store.put_artifact_bytes(run.run_id, b"payload", "config", {})
    assert again.sha256 == art.sha256 and path.read_bytes() == b"payload"
    with store.open_artifact(art.sha256) as handle:
        assert handle.read() == b"payload"


def test_open_artifact_validates_the_address_and_size(store: LocalRunStore, run: RunMeta) -> None:
    with pytest.raises(StoreError, match="64 lower-case hex"):
        store.open_artifact("../../etc/passwd")
    with pytest.raises(StoreError, match="missing"):
        store.open_artifact("a" * 64)


def test_reregistering_with_a_different_kind_or_meta_is_refused(
    store: LocalRunStore, run: RunMeta
) -> None:
    store.put_artifact_bytes(run.run_id, b"x", "config", {"a": 1})
    store.put_artifact_bytes(run.run_id, b"x", "config", {"a": 1})  # identical: fine
    with pytest.raises(StoreError, match="kind or meta"):
        store.put_artifact_bytes(run.run_id, b"x", "env", {"a": 1})
    with pytest.raises(StoreError, match="kind or meta"):
        store.put_artifact_bytes(run.run_id, b"x", "config", {"a": 2})


def test_evals_directory_is_private_and_steps_are_range_checked(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = arts(store, run.run_id)
    store.put_snapshot(snap(run.run_id, conf, pred))
    assert (store.path / "evals").stat().st_mode & 0o777 == 0o700
    assert (store.path / "evals" / run.run_id).stat().st_mode & 0o777 == 0o700
    for bad in (-1, 2**63, 2**70):
        with pytest.raises(StoreError, match="step"):
            store.has_snapshot(run.run_id, H1, bad)
        with pytest.raises(StoreError, match="step"):
            store.get_snapshot(run.run_id, H1, bad)
    assert os.path.isdir(store.path / "evals")
