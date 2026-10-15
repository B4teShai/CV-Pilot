"""Baseline pins in the local store: append-only, digested, chained and bound to a snapshot."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from support import H1, H2, add_snapshot, make_run, open_project

from cvpilot.core.errors import (
    BaselineExistsError,
    BaselineIntegrityError,
    BaselineNotFoundError,
    BaselineRefusedError,
    RunNotFoundError,
    StoreError,
)
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import Baseline
from cvpilot.store import baselines as bl
from cvpilot.store.local import LocalRunStore

NOW = datetime(2026, 10, 15, 9, 0, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path: Path) -> Any:
    s = open_project(tmp_path)
    yield s
    s.close()


def pin(
    store: LocalRunStore, run_id: str, *, tag: str = "prod", h: str = H1, step: int = 0, **kw: Any
) -> Baseline:
    digest = store.snapshot_digest(run_id, h, step)
    baseline = Baseline(
        project="proj",
        tag=tag,
        run_id=run_id,
        eval_set_hash=h,
        step=step,
        snapshot_sha256=digest,
        pinned_at=NOW,
    )
    store.put_baseline(baseline, **kw)
    return baseline


@pytest.fixture
def run(store: LocalRunStore) -> str:
    rid = make_run(store)
    add_snapshot(store, rid)
    return rid


@pytest.mark.parametrize("tag", ["prod", "staging", "a", "v1_2-rc", "0day", "x" * 32])
def test_valid_tags(tag: str) -> None:
    Baseline(
        project="proj",
        tag=tag,
        run_id=new_run_id(),
        eval_set_hash=H1,
        step=0,
        snapshot_sha256=H2,
        pinned_at=NOW,
    )


@pytest.mark.parametrize(
    "tag", ["", "Prod", "-x", "_x", "x" * 33, "a b", "a/b", "../x", "a\n", "tag:prod", "é", "a.b"]
)
def test_invalid_tags_are_rejected(tag: str) -> None:
    with pytest.raises(ValidationError):
        Baseline(
            project="proj",
            tag=tag,
            run_id=new_run_id(),
            eval_set_hash=H1,
            step=0,
            snapshot_sha256=H2,
            pinned_at=NOW,
        )


def test_round_trip_and_latest_step_binding(store: LocalRunStore, run: str) -> None:
    pinned = pin(store, run)
    assert store.get_baseline("prod") == pinned
    assert store.list_baselines() == [pinned]


def test_unknown_tag_is_not_found(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    with pytest.raises(BaselineNotFoundError, match="staging"):
        store.get_baseline("staging")


def test_an_existing_tag_is_not_silently_replaced(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    other = make_run(store)
    add_snapshot(store, other)
    with pytest.raises(BaselineExistsError) as exc:
        pin(store, other)
    assert exc.value.log_fields()["tag"] == "prod"
    assert store.get_baseline("prod").run_id == run


def test_replace_appends_and_keeps_the_history(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    other = make_run(store)
    add_snapshot(store, other)
    pin(store, other, replace=True)
    assert store.get_baseline("prod").run_id == other
    history = store.list_baselines(history=True)
    assert [b.run_id for b in history] == [run, other]
    assert len(store.list_baselines()) == 1


def test_run_without_the_snapshot_is_refused(store: LocalRunStore, run: str) -> None:
    baseline = Baseline(
        project="proj",
        tag="prod",
        run_id=run,
        eval_set_hash=H2,
        step=0,
        snapshot_sha256="c" * 64,
        pinned_at=NOW,
    )
    with pytest.raises(BaselineRefusedError, match="no snapshot"):
        store_put(store, baseline)


def store_put(store: LocalRunStore, baseline: Baseline) -> None:
    store.put_baseline(baseline)


def test_unknown_run_is_refused(store: LocalRunStore) -> None:
    baseline = Baseline(
        project="proj",
        tag="prod",
        run_id=new_run_id(),
        eval_set_hash=H1,
        step=0,
        snapshot_sha256=H2,
        pinned_at=NOW,
    )
    with pytest.raises(RunNotFoundError):
        store.put_baseline(baseline)


def test_a_digest_that_does_not_match_the_stored_snapshot_is_refused(
    store: LocalRunStore, run: str
) -> None:
    baseline = Baseline(
        project="proj",
        tag="prod",
        run_id=run,
        eval_set_hash=H1,
        step=0,
        snapshot_sha256="c" * 64,
        pinned_at=NOW,
    )
    with pytest.raises(BaselineRefusedError, match="changed"):
        store.put_baseline(baseline)


def test_a_baseline_of_another_project_is_refused(store: LocalRunStore, run: str) -> None:
    baseline = Baseline(
        project="other",
        tag="prod",
        run_id=run,
        eval_set_hash=H1,
        step=0,
        snapshot_sha256=store.snapshot_digest(run, H1, 0),
        pinned_at=NOW,
    )
    with pytest.raises(BaselineRefusedError, match="project"):
        store.put_baseline(baseline)


def test_second_store_handle_cannot_overwrite_a_tag(tmp_path: Path, run: str, store: Any) -> None:
    pin(store, run)
    second = open_project(tmp_path)
    try:
        with pytest.raises(BaselineExistsError):
            pin(second, run)
    finally:
        second.close()


def test_database_refuses_update_and_delete(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store._conn.execute("UPDATE baseline SET tag='x'")
    with pytest.raises(sqlite3.DatabaseError, match="immutable"):
        store._conn.execute("DELETE FROM baseline")


def test_dropped_triggers_and_table_are_recreated_on_open(tmp_path: Path) -> None:
    s = open_project(tmp_path)
    s._conn.execute("DROP TRIGGER baseline_no_update")
    s.close()
    again = open_project(tmp_path)
    try:
        names = {
            r[0] for r in again._conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'")
        }
        assert {"baseline_no_update", "baseline_no_delete"} <= names
    finally:
        again.close()


def drop_triggers(store: LocalRunStore) -> None:
    for name in ("baseline_no_update", "baseline_no_delete"):
        store._conn.execute(f"DROP TRIGGER {name}")


def test_edited_record_text_fails_verification(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    drop_triggers(store)
    store._conn.execute("UPDATE baseline SET record=replace(record, 'prod', 'prop')")
    with pytest.raises(BaselineIntegrityError, match="digest"):
        store.get_baseline("prod")


def test_changed_columns_fail_verification(store: LocalRunStore, run: str) -> None:
    other = make_run(store)
    add_snapshot(store, other)
    pin(store, run)
    drop_triggers(store)
    store._conn.execute("UPDATE baseline SET run_id=?", (other,))
    with pytest.raises(BaselineIntegrityError, match="differs"):
        store.get_baseline("prod")


def test_removed_earlier_row_breaks_the_chain(store: LocalRunStore, run: str) -> None:
    pin(store, run, tag="a")
    pin(store, run, tag="b")
    drop_triggers(store)
    store._conn.execute("DELETE FROM baseline WHERE tag='a'")
    with pytest.raises(BaselineIntegrityError, match="chain"):
        store.get_baseline("b")
    with pytest.raises(BaselineIntegrityError, match="chain"):
        store.list_baselines()


def test_reordered_rows_break_the_chain(store: LocalRunStore, run: str) -> None:
    pin(store, run, tag="a")
    pin(store, run, tag="b")
    drop_triggers(store)
    store._conn.execute("UPDATE baseline SET seq=seq+100 WHERE tag='a'")
    with pytest.raises(BaselineIntegrityError, match="chain"):
        store.get_baseline("a")


def test_snapshot_swapped_after_pinning_is_detected(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    # an attacker who can write the database and the snapshot file recomputes the snapshot
    # digest and its chain; the baseline still holds the digest of what was pinned
    path = next((store.path / "evals" / run).glob("*.json"))
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"mAP50_95":0.6', '"mAP50_95":0.9'))
    import hashlib

    from cvpilot.store import snapshots as sn

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    store._conn.execute("DROP TRIGGER eval_snapshot_no_update")
    store._conn.execute(
        "UPDATE eval_snapshot SET snapshot_sha256=?, chain_sha256=? WHERE run_id=?",
        (digest, sn.chain_digest(sn.GENESIS, run, H1, 0, digest), run),
    )
    assert store.get_snapshot(run, H1, 0).metrics["mAP50_95"] == 0.9  # snapshot alone verifies
    with pytest.raises(BaselineIntegrityError, match="pinned snapshot"):
        store.get_baseline("prod")


def test_garbage_in_the_record_column_is_an_integrity_error(store: LocalRunStore, run: str) -> None:
    pin(store, run)
    drop_triggers(store)
    import hashlib

    garbage = "{not json"
    store._conn.execute(
        "UPDATE baseline SET record=?, record_sha256=?",
        (garbage, hashlib.sha256(garbage.encode()).hexdigest()),
    )
    with pytest.raises(BaselineIntegrityError):
        store.get_baseline("prod")


def test_row_count_is_bounded(
    store: LocalRunStore, run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(bl, "MAX_BASELINE_ROWS", 2)
    pin(store, run, tag="a")
    pin(store, run, tag="b")
    with pytest.raises(StoreError, match="too many"):
        pin(store, run, tag="c")
    assert len(store.list_baselines()) == 2
    monkeypatch.setattr(bl, "MAX_BASELINE_ROWS", 1)
    with pytest.raises(StoreError, match="too many"):
        store.list_baselines()


def test_snapshot_digest_requires_a_verified_snapshot(store: LocalRunStore, run: str) -> None:
    path = next((store.path / "evals" / run).glob("*.json"))
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"mAP50_95":0.6', '"mAP50_95":0.7'))
    from cvpilot.core.errors import SnapshotIntegrityError

    with pytest.raises(SnapshotIntegrityError):
        store.snapshot_digest(run, H1, 0)
