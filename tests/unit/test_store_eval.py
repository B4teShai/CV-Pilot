"""Eval snapshots in the local store: immutable, keyed by (run_id, eval_set_hash, step)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from cvpilot.core.errors import (
    RunNotFoundError,
    SnapshotExistsError,
    SnapshotNotFoundError,
    StoreError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, EvalSnapshot, Repro, RunMeta
from cvpilot.store.local import LocalRunStore, open_store

H1, H2 = "1" * 64, "2" * 64
SHA = "b" * 64


def make_run() -> tuple[RunMeta, Repro]:
    rid = new_run_id()
    meta = RunMeta(
        run_id=rid,
        project="p",
        name="n",
        status="running",
        task="detection",
        framework="python==3.12.1",
        host="h",
        started_at=datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc),
        schema_version=SCHEMA_VERSION,
    )
    repro = Repro(
        run_id=rid,
        git_dirty=False,
        packages_lock_sha256=SHA,
        python_version="3.12.1",
        gpu_names=[],
        dataset_uri="/d",
        dataset_hash="merkle:" + SHA,
        config={},
        command_line="train.py",
    )
    return meta, repro


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = open_store("p", tmp_path)
    yield s
    s.close()


@pytest.fixture
def run(store: LocalRunStore) -> RunMeta:
    meta, repro = make_run()
    store.put_run(meta, repro)
    return meta


def artifacts(store: LocalRunStore, run_id: str, tag: bytes = b"") -> tuple[str, str]:
    conf = store.put_artifact_bytes(run_id, b"confusion" + tag, "confusion", {})
    pred = store.put_artifact_bytes(run_id, b"predictions" + tag, "predictions", {})
    return conf.sha256, pred.sha256


def snapshot(run_id: str, conf: str, pred: str, *, h: str = H1, step: int = 0) -> EvalSnapshot:
    return EvalSnapshot(
        run_id=run_id,
        eval_set_hash=h,
        step=step,
        task="detection",
        class_names=["a"],
        n_images=3,
        metrics={"mAP50_95": 0.5 + step / 100},
        per_class={"a": {"support": 3.0}},
        slices={},
        confusion_sha256=conf,
        predictions_sha256=pred,
    )


def test_snapshot_round_trips_and_is_written_under_evals(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = artifacts(store, run.run_id)
    snap = snapshot(run.run_id, conf, pred, step=7)
    store.put_snapshot(snap)
    assert store.get_snapshot(run.run_id, H1, 7) == snap
    path = store.path / "evals" / run.run_id / f"{H1[:12]}_{7:09d}.json"
    assert json.loads(path.read_text()) == json.loads(snap.model_dump_json())
    assert path.stat().st_mode & 0o777 == 0o444
    assert not list(path.parent.glob(".*tmp*"))


def test_latest_step_is_returned_without_a_step_and_listing_is_ordered(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = artifacts(store, run.run_id)
    for step in (5, 1, 3):
        store.put_snapshot(snapshot(run.run_id, conf, pred, step=step))
    store.put_snapshot(snapshot(run.run_id, conf, pred, h=H2, step=2))
    assert store.get_snapshot(run.run_id, H1).step == 5
    assert [(s.eval_set_hash[0], s.step) for s in store.list_snapshots(run.run_id)] == [
        ("1", 1),
        ("1", 3),
        ("1", 5),
        ("2", 2),
    ]


def test_a_snapshot_is_immutable(store: LocalRunStore, run: RunMeta) -> None:
    conf, pred = artifacts(store, run.run_id)
    first = snapshot(run.run_id, conf, pred)
    store.put_snapshot(first)
    path = store.path / "evals" / run.run_id / f"{H1[:12]}_{0:09d}.json"
    before = path.read_bytes()
    changed = first.model_copy(update={"metrics": {"mAP50_95": 0.99}})
    with pytest.raises(SnapshotExistsError, match="immutable"):
        store.put_snapshot(changed)
    assert path.read_bytes() == before and store.get_snapshot(run.run_id, H1, 0) == first
    assert store.has_snapshot(run.run_id, H1, 0) and not store.has_snapshot(run.run_id, H1, 1)


def test_the_database_itself_refuses_updates_and_deletes(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = artifacts(store, run.run_id)
    store.put_snapshot(snapshot(run.run_id, conf, pred))
    for sql in ("UPDATE eval_snapshot SET n_images=1", "DELETE FROM eval_snapshot"):
        with pytest.raises(sqlite3.DatabaseError, match="immutable"):
            store._conn.execute(sql)


def test_artifacts_must_exist_with_the_right_kind_in_the_same_run(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = artifacts(store, run.run_id)
    with pytest.raises(StoreError, match="predictions artifact"):
        store.put_snapshot(snapshot(run.run_id, conf, "f" * 64))
    with pytest.raises(StoreError, match="confusion artifact"):
        store.put_snapshot(snapshot(run.run_id, pred, pred))  # kind predictions, not confusion
    other, other_repro = make_run()
    store.put_run(other, other_repro)
    with pytest.raises(StoreError, match="confusion artifact"):
        store.put_snapshot(snapshot(other.run_id, conf, pred))  # artifacts belong to another run
    assert store.list_snapshots(run.run_id) == []


def test_unknown_run_and_missing_snapshot(store: LocalRunStore, run: RunMeta) -> None:
    with pytest.raises(RunNotFoundError):
        store.put_snapshot(snapshot(new_run_id(), "a" * 64, "b" * 64))
    with pytest.raises(SnapshotNotFoundError):
        store.get_snapshot(run.run_id, H1)
    with pytest.raises(SnapshotNotFoundError):
        store.get_snapshot(run.run_id, H1, 3)
    with pytest.raises(StoreError, match="64 lower-case hex"):
        store.get_snapshot(run.run_id, "../../etc/passwd")


def test_a_clash_of_the_twelve_character_file_prefix_is_refused(
    store: LocalRunStore, run: RunMeta
) -> None:
    conf, pred = artifacts(store, run.run_id)
    store.put_snapshot(snapshot(run.run_id, conf, pred, h="3" * 12 + "a" * 52))
    with pytest.raises(StoreError, match="file name"):
        store.put_snapshot(snapshot(run.run_id, conf, pred, h="3" * 12 + "b" * 52))


def test_an_oversized_or_missing_snapshot_file_is_a_store_error(
    store: LocalRunStore, run: RunMeta, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvpilot.store import snapshots

    conf, pred = artifacts(store, run.run_id)
    store.put_snapshot(snapshot(run.run_id, conf, pred))
    path = store.path / "evals" / run.run_id / f"{H1[:12]}_{0:09d}.json"
    monkeypatch.setattr(snapshots, "MAX_SNAPSHOT_BYTES", path.stat().st_size - 1)
    with pytest.raises(StoreError, match="too large"):
        store.get_snapshot(run.run_id, H1, 0)
    monkeypatch.undo()
    path.unlink()
    with pytest.raises(StoreError, match="missing"):
        store.get_snapshot(run.run_id, H1, 0)


def test_stores_from_before_eval_get_the_table_on_open_and_keep_their_version(
    tmp_path: Path,
) -> None:
    s = open_store("p", tmp_path)
    s._conn.execute("DROP TABLE eval_snapshot")
    s.close()
    again = open_store("p", tmp_path)
    try:
        assert again.schema_version == 1
        assert again._conn.execute("SELECT COUNT(*) FROM eval_snapshot").fetchone() == (0,)
    finally:
        again.close()
