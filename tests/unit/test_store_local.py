from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cvpilot.core.errors import (
    DuplicateMetricError,
    InvalidRunIdError,
    InvalidTransitionError,
    MetricOrderError,
    RunClosedError,
    RunNotFoundError,
    StoreError,
    StoreVersionError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import MetricPoint, Repro, RunMeta
from cvpilot.store.local import LocalRunStore, open_store

T0 = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
SHA = "b" * 64


def make_run(project: str = "p", **over: Any) -> tuple[RunMeta, Repro]:
    rid = over.pop("run_id", None) or new_run_id()
    base: dict[str, Any] = {
        "run_id": rid,
        "project": project,
        "name": "n",
        "status": "running",
        "task": "detection",
        "framework": "python==3.12.1",
        "host": "h",
        "started_at": T0,
        "schema_version": 1,
    }
    base.update(over)
    meta = RunMeta(**base)
    repro = Repro(
        run_id=rid,
        git_commit=None,
        git_dirty=False,
        packages_lock_sha256=SHA,
        python_version="3.12.1",
        cuda=None,
        gpu_names=[],
        dataset_uri="/d",
        dataset_hash="merkle:" + SHA,
        config={"lr": 0.1},
        seed=7,
        command_line="train.py",
    )
    return meta, repro


def pts(run_id: str, key: str, steps: range, value: float = 1.0) -> list[MetricPoint]:
    return [
        MetricPoint(run_id=run_id, key=key, step=s, value=value + s, wall_time=1.7e9 + s)
        for s in steps
    ]


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


# --- opening, WAL, migrations ---------------------------------------------


def test_open_creates_layout_in_wal_mode(tmp_path: Path) -> None:
    s = open_store("proj", tmp_path)
    try:
        assert s.path == (tmp_path / "proj").resolve()
        db = sqlite3.connect(s.path / "runs.sqlite")
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone() == ("1",)
        assert db.execute("SELECT version, name FROM migrations").fetchall() == [(1, "initial")]
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"run_meta", "repro", "artifact", "metric_files", "meta", "migrations"} <= tables
        db.close()
        assert s._conn.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert s._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    finally:
        s.close()


def test_reopen_is_idempotent_and_keeps_data(tmp_path: Path) -> None:
    s = open_store("p", tmp_path)
    meta, repro = make_run()
    s.put_run(meta, repro)
    s.close()
    s2 = open_store("p", tmp_path)
    assert s2.get_run(meta.run_id) == meta
    s2.close()


def test_open_without_create_requires_existing_store(tmp_path: Path) -> None:
    with pytest.raises(StoreError, match="no run store"):
        open_store("p", tmp_path, create=False)
    assert not (tmp_path / "p").exists()
    open_store("p", tmp_path).close()
    open_store("p", tmp_path, create=False).close()


def test_newer_schema_version_is_refused(tmp_path: Path) -> None:
    open_store("p", tmp_path).close()
    db = sqlite3.connect(tmp_path / "p" / "runs.sqlite")
    db.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    db.commit()
    db.close()
    with pytest.raises(StoreVersionError) as exc:
        open_store("p", tmp_path)
    assert (exc.value.store_version, exc.value.library_version) == (99, 1)
    assert exc.value.event == "store_version_unsupported"
    assert exc.value.log_fields() == {"store_version": 99, "library_version": 1}


def test_forward_migration_backs_up_then_applies(tmp_path: Path) -> None:
    from cvpilot.store import migrations

    open_store("p", tmp_path).close()

    second = ModuleType("0002_extra")
    second.VERSION = 2  # type: ignore[attr-defined]
    second.NAME = "extra"  # type: ignore[attr-defined]
    second.upgrade = lambda conn: conn.execute("CREATE TABLE extra (x INTEGER)")  # type: ignore[attr-defined]
    available = [*migrations.discover(), migrations.Migration(2, "extra", second.upgrade)]  # type: ignore[attr-defined]

    s = LocalRunStore.open("p", tmp_path, migrations=available)
    try:
        assert (tmp_path / "p" / "runs.sqlite.bak-1").exists()
        assert s.schema_version == 2
        names = [r[0] for r in s._conn.execute("SELECT name FROM migrations ORDER BY version")]
        assert names == ["initial", "extra"]
        s._conn.execute("SELECT * FROM extra")
    finally:
        s.close()


def test_failed_migration_rolls_back(tmp_path: Path) -> None:
    from cvpilot.store import migrations

    open_store("p", tmp_path).close()

    def boom(conn: sqlite3.Connection) -> None:
        conn.execute("CREATE TABLE half (x INTEGER)")
        raise RuntimeError("boom")

    available = [*migrations.discover(), migrations.Migration(2, "bad", boom)]
    with pytest.raises(StoreError, match="migration 2"):
        LocalRunStore.open("p", tmp_path, migrations=available)
    s = open_store("p", tmp_path)
    assert s.schema_version == 1
    assert not s._conn.execute("SELECT 1 FROM sqlite_master WHERE name='half'").fetchall()
    s.close()


def test_library_schema_constant_matches_latest_migration() -> None:
    from cvpilot.core.schema import SCHEMA_VERSION
    from cvpilot.store import migrations

    assert max(m.version for m in migrations.discover()) == SCHEMA_VERSION


# --- path constraints -----------------------------------------------------


@pytest.mark.parametrize("project", ["", "..", "../x", "a/b", "A", "x" * 65, "-a", "a\x00b"])
def test_invalid_project_names_are_refused(tmp_path: Path, project: str) -> None:
    with pytest.raises(StoreError, match="project"):
        open_store(project, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_project_symlink_escaping_the_root_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "p").symlink_to(outside)
    with pytest.raises(StoreError, match="outside"):
        open_store("p", root)
    assert not (outside / "runs.sqlite").exists()


def test_store_root_that_is_a_file_is_refused(tmp_path: Path) -> None:
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(StoreError, match="not a directory"):
        open_store("p", f)


def test_relative_root_is_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    s = open_store("p", Path("rel"))
    assert s.path == (tmp_path / "rel" / "p").resolve()
    s.close()


@pytest.mark.parametrize("bad", ["../../etc/passwd", "", "x" * 26, "01arz3ndektsv4rrffq69g5fav"])
def test_run_ids_are_validated_before_any_path_use(store: LocalRunStore, bad: str) -> None:
    for call in (store.read_points, store.get_run, store.metric_stats, store.compact_metrics):
        with pytest.raises(InvalidRunIdError):
            call(bad)
    with pytest.raises(InvalidRunIdError):
        store.append_points(
            [
                MetricPoint.model_construct(
                    run_id=bad, key="k", step=0, value=1.0, wall_time=1.0, epoch=None
                )
            ]
        )
    assert not (store.path / "metrics").exists() or not any((store.path / "metrics").iterdir())


# --- runs -----------------------------------------------------------------


def test_put_and_get_run_round_trip(store: LocalRunStore) -> None:
    meta, repro = make_run(name="résumé run")
    store.put_run(meta, repro)
    assert store.get_run(meta.run_id) == meta
    assert store.get_repro(meta.run_id) == repro


def test_duplicate_run_id_is_refused(store: LocalRunStore) -> None:
    meta, repro = make_run()
    store.put_run(meta, repro)
    with pytest.raises(StoreError, match="already exists"):
        store.put_run(meta, repro)


def test_resumed_run_requires_existing_parent(store: LocalRunStore, run: RunMeta) -> None:
    child, repro = make_run(status="resumed", parent_run_id=run.run_id)
    store.put_run(child, repro)
    assert store.get_run(child.run_id).parent_run_id == run.run_id
    orphan, orepro = make_run(status="resumed", parent_run_id=new_run_id())
    with pytest.raises(StoreError, match="parent"):
        store.put_run(orphan, orepro)


def test_put_run_rejects_foreign_schema_version(store: LocalRunStore) -> None:
    meta, repro = make_run(schema_version=2)
    with pytest.raises(StoreError, match="schema_version"):
        store.put_run(meta, repro)


def test_repro_and_meta_must_share_run_id(store: LocalRunStore) -> None:
    meta, _ = make_run()
    _, other = make_run()
    with pytest.raises(StoreError, match="run_id"):
        store.put_run(meta, other)


def test_unknown_run_raises(store: LocalRunStore) -> None:
    with pytest.raises(RunNotFoundError):
        store.get_run(new_run_id())
    with pytest.raises(RunNotFoundError):
        store.get_repro(new_run_id())


def test_list_runs_newest_first_and_filtered(store: LocalRunStore) -> None:
    ids = []
    for i in range(3):
        meta, repro = make_run(started_at=T0 + timedelta(minutes=i))
        store.put_run(meta, repro)
        ids.append(meta.run_id)
    store.set_status(ids[0], "finished", T0 + timedelta(hours=1))
    assert [r.run_id for r in store.list_runs()] == ids[::-1]
    assert [r.run_id for r in store.list_runs(status="finished")] == [ids[0]]
    assert [r.run_id for r in store.list_runs(status="running")] == [ids[2], ids[1]]
    assert store.list_runs(status="failed") == []


def test_status_transitions_are_enforced(store: LocalRunStore, run: RunMeta) -> None:
    end = T0 + timedelta(minutes=5)
    store.set_status(run.run_id, "failed", end)
    got = store.get_run(run.run_id)
    assert (got.status, got.ended_at) == ("failed", end)
    for new in ("finished", "running", "failed"):
        with pytest.raises(InvalidTransitionError):
            store.set_status(run.run_id, new, end)  # type: ignore[arg-type]
    assert store.get_run(run.run_id).status == "failed"


def test_set_status_rejects_non_terminal_target_and_bad_end(
    store: LocalRunStore, run: RunMeta
) -> None:
    with pytest.raises(InvalidTransitionError):
        store.set_status(run.run_id, "resumed", T0)
    with pytest.raises(StoreError, match="ended_at"):
        store.set_status(run.run_id, "finished", T0 - timedelta(seconds=1))
    assert store.get_run(run.run_id).status == "running"
    with pytest.raises(RunNotFoundError):
        store.set_status(new_run_id(), "finished", T0)


# --- metrics --------------------------------------------------------------


def test_points_round_trip_exactly_including_nonfinite(store: LocalRunStore, run: RunMeta) -> None:
    batch = [
        MetricPoint(run_id=run.run_id, key="a", step=0, epoch=1, value=math.nan, wall_time=1.5),
        MetricPoint(run_id=run.run_id, key="a", step=1, epoch=None, value=math.inf, wall_time=2.5),
        MetricPoint(run_id=run.run_id, key="b/c", step=1, value=-math.inf, wall_time=2.5),
        MetricPoint(run_id=run.run_id, key="ключ", step=2, value=0.1 + 0.2, wall_time=3.5),
    ]
    store.append_points(batch)
    got = store.read_points(run.run_id)
    assert len(got) == 4
    for want, have in zip(batch, got, strict=True):
        assert (want.key, want.step, want.epoch, want.wall_time) == (
            have.key,
            have.step,
            have.epoch,
            have.wall_time,
        )
        assert want.value == have.value or (math.isnan(want.value) and math.isnan(have.value))


def test_batches_become_numbered_parquet_files_with_schema_metadata(
    store: LocalRunStore, run: RunMeta
) -> None:
    import pyarrow.parquet as pq

    store.append_points(pts(run.run_id, "k", range(0, 5)))
    store.append_points(pts(run.run_id, "k", range(5, 8)))
    d = store.path / "metrics" / run.run_id
    assert sorted(p.name for p in d.iterdir()) == ["000001.parquet", "000002.parquet"]
    meta = pq.read_schema(d / "000001.parquet").metadata
    assert meta[b"cvpilot.schema_version"] == b"1"
    stats = store.metric_stats(run.run_id)
    assert (stats.n_points, stats.n_files, stats.min_step, stats.max_step) == (8, 2, 0, 7)
    assert [p.step for p in store.read_points(run.run_id)] == list(range(8))


def test_empty_batch_is_a_noop(store: LocalRunStore, run: RunMeta) -> None:
    store.append_points([])
    assert store.metric_stats(run.run_id).n_points == 0
    assert store.read_points(run.run_id) == []


def test_duplicate_inside_a_batch_rejects_the_whole_batch(
    store: LocalRunStore, run: RunMeta
) -> None:
    batch = [
        *pts(run.run_id, "k", range(3)),
        *pts(run.run_id, "j", range(2)),
        *pts(run.run_id, "k", range(2, 3)),
    ]
    with pytest.raises(DuplicateMetricError, match="key='k'.*step=2|k.*2"):
        store.append_points(batch)
    assert store.metric_stats(run.run_id).n_points == 0


def test_duplicate_across_batches_is_rejected(store: LocalRunStore, run: RunMeta) -> None:
    store.append_points(pts(run.run_id, "k", range(3)))
    with pytest.raises(DuplicateMetricError):
        store.append_points(pts(run.run_id, "k", range(2, 4)))
    assert store.metric_stats(run.run_id).n_points == 3


def test_step_going_backwards_per_key_is_rejected_but_other_keys_are_independent(
    store: LocalRunStore, run: RunMeta
) -> None:
    store.append_points(pts(run.run_id, "k", range(5, 8)))
    with pytest.raises(MetricOrderError):
        store.append_points(pts(run.run_id, "k", range(1, 2)))
    store.append_points(pts(run.run_id, "other", range(0, 2)))
    store.append_points(pts(run.run_id, "k", range(8, 9)))


def test_uniqueness_survives_reopening_the_store(tmp_path: Path) -> None:
    s = open_store("p", tmp_path)
    meta, repro = make_run()
    s.put_run(meta, repro)
    s.append_points(pts(meta.run_id, "k", range(4)))
    s.close()
    s2 = open_store("p", tmp_path)
    with pytest.raises(DuplicateMetricError):
        s2.append_points(pts(meta.run_id, "k", range(3, 5)))
    s2.append_points(pts(meta.run_id, "k", range(4, 6)))
    assert [p.step for p in s2.read_points(meta.run_id)] == list(range(6))
    s2.close()


def test_append_to_unknown_or_finished_run_is_refused(store: LocalRunStore, run: RunMeta) -> None:
    with pytest.raises(RunNotFoundError):
        store.append_points(pts(new_run_id(), "k", range(1)))
    store.set_status(run.run_id, "finished", T0)
    with pytest.raises(RunClosedError):
        store.append_points(pts(run.run_id, "k", range(1)))


def test_batch_spanning_runs_is_refused(store: LocalRunStore, run: RunMeta) -> None:
    other, repro = make_run()
    store.put_run(other, repro)
    with pytest.raises(ValueError, match="one run"):
        store.append_points([*pts(run.run_id, "k", range(1)), *pts(other.run_id, "k", range(1))])


def test_compaction_merges_files_and_preserves_order(store: LocalRunStore, run: RunMeta) -> None:
    for i in range(4):
        store.append_points(pts(run.run_id, "k", range(i * 10, i * 10 + 10)))
    before = store.read_points(run.run_id)
    store.compact_metrics(run.run_id)
    d = store.path / "metrics" / run.run_id
    assert [p.name for p in d.iterdir()] == ["all.parquet"]
    assert store.read_points(run.run_id) == before
    stats = store.metric_stats(run.run_id)
    assert (stats.n_points, stats.n_files) == (40, 1)
    store.compact_metrics(run.run_id)  # idempotent
    assert store.read_points(run.run_id) == before


def test_compaction_without_points_is_a_noop(store: LocalRunStore, run: RunMeta) -> None:
    store.compact_metrics(run.run_id)
    assert store.metric_stats(run.run_id).n_files == 0


def test_unindexed_orphan_file_is_ignored_and_never_read(
    store: LocalRunStore, run: RunMeta
) -> None:
    store.append_points(pts(run.run_id, "k", range(3)))
    d = store.path / "metrics" / run.run_id
    (d / "000002.parquet").write_bytes(b"PAR1 partial garbage")  # renamed but never indexed
    (d / ".000003.parquet.tmp").write_bytes(b"half")
    assert len(store.read_points(run.run_id)) == 3
    store.append_points(pts(run.run_id, "k", range(3, 5)))  # reuses seq 2, replacing the orphan
    assert [p.step for p in store.read_points(run.run_id)] == list(range(5))


def test_indexed_file_that_is_missing_or_corrupt_is_a_store_error(
    store: LocalRunStore, run: RunMeta
) -> None:
    store.append_points(pts(run.run_id, "k", range(3)))
    f = store.path / "metrics" / run.run_id / "000001.parquet"
    f.write_bytes(f.read_bytes()[:20])
    with pytest.raises(StoreError, match="000001.parquet"):
        store.read_points(run.run_id)
    f.unlink()
    with pytest.raises(StoreError, match="missing"):
        store.read_points(run.run_id)


def test_concurrent_appends_from_threads_are_serialised(tmp_path: Path) -> None:
    s = open_store("p", tmp_path)
    runs = []
    for _ in range(6):
        meta, repro = make_run()
        s.put_run(meta, repro)
        runs.append(meta.run_id)
    errors: list[BaseException] = []

    def work(rid: str) -> None:
        try:
            for i in range(15):
                s.append_points(pts(rid, "k", range(i * 4, i * 4 + 4)))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(r,)) for r in runs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    for rid in runs:
        assert [p.step for p in s.read_points(rid)] == list(range(60))
    s.close()


def test_two_processes_can_read_while_one_writes(tmp_path: Path) -> None:
    s = open_store("p", tmp_path)
    meta, repro = make_run()
    s.put_run(meta, repro)
    s.append_points(pts(meta.run_id, "k", range(10)))
    code = (
        "import sys; from pathlib import Path; from cvpilot.store.local import open_store;"
        "s = open_store('p', Path(sys.argv[1]), create=False);"
        "print(len(s.list_runs()), len(s.read_points(sys.argv[2])))"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code, str(tmp_path), meta.run_id],
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.split() == ["1", "10"]
    s.close()


# --- artifacts ------------------------------------------------------------


def test_artifact_is_stored_by_content_address(store: LocalRunStore, run: RunMeta) -> None:
    data = b"hello diff"
    sha = hashlib.sha256(data).hexdigest()
    art = store.put_artifact_bytes(run.run_id, data, kind="diff", meta={"n": 1})
    assert (
        art.sha256 == sha
        and art.size_bytes == len(data)
        and art.uri == f"artifacts/{sha[:2]}/{sha}"
    )
    assert store.artifact_path(sha).read_bytes() == data
    assert store.list_artifacts(run.run_id) == [art]


def test_artifact_from_file_and_dedup_across_runs(
    store: LocalRunStore, run: RunMeta, tmp_path: Path
) -> None:
    other, repro = make_run()
    store.put_run(other, repro)
    f = tmp_path / "cfg.json"
    f.write_text('{"a": 1}')
    a1 = store.put_artifact(run.run_id, f, kind="config", meta={})
    a2 = store.put_artifact(other.run_id, f, kind="config", meta={})
    assert a1.sha256 == a2.sha256
    files = [p for p in (store.path / "artifacts").rglob("*") if p.is_file()]
    assert len(files) == 1
    again = store.put_artifact(run.run_id, f, kind="config", meta={})  # same row, no error
    assert again.sha256 == a1.sha256
    assert len(store.list_artifacts(run.run_id)) == 1


def test_artifact_errors(store: LocalRunStore, run: RunMeta, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        store.put_artifact(run.run_id, tmp_path / "missing", kind="env", meta={})
    with pytest.raises(StoreError, match="regular file"):
        store.put_artifact(run.run_id, tmp_path, kind="env", meta={})
    with pytest.raises(RunNotFoundError):
        store.put_artifact_bytes(new_run_id(), b"x", kind="env", meta={})
    with pytest.raises(ValueError):
        store.put_artifact_bytes(run.run_id, b"x", kind="movie", meta={})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="JSON"):
        store.put_artifact_bytes(run.run_id, b"x", kind="env", meta={"o": object()})
    with pytest.raises(StoreError):
        store.artifact_path("../../etc/passwd")
    assert not list((store.path / "artifacts").rglob(".tmp*"))


def test_core_import_does_not_pull_pyarrow() -> None:
    code = (
        "import sys; import cvpilot.store.local, cvpilot.registry;"
        "assert 'pyarrow' not in sys.modules, 'pyarrow imported eagerly';"
        "assert 'to' + 'rch' not in sys.modules and 'pynvml' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


def test_sqlite_row_persisted_json_columns(store: LocalRunStore, run: RunMeta) -> None:
    row = store._conn.execute("SELECT config, gpu_names FROM repro").fetchone()
    assert json.loads(row[0]) == {"lr": 0.1} and json.loads(row[1]) == []
    assert os.stat(store.path / "runs.sqlite").st_mode & 0o077 == 0
