"""``cvpilot baseline set|list|show`` and the use cases behind them."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import CliResult
from support import H1, H2, add_snapshot, make_run, open_project

Invoke = Callable[..., CliResult]


@pytest.fixture
def seeded(store_root: Path) -> tuple[str, str]:
    store = open_project(store_root)
    try:
        run = make_run(store)
        add_snapshot(store, run, h=H1, step=1)
        add_snapshot(store, run, h=H1, step=5, metrics={"mAP50_95": 0.7})
        other = make_run(store)
        add_snapshot(store, other, h=H1)
        add_snapshot(store, other, h=H2)
    finally:
        store.close()
    return run, other


def test_set_pins_the_latest_step_and_list_shows_it(
    invoke: Invoke, seeded: tuple[str, str]
) -> None:
    run, _ = seeded
    res = invoke("baseline", "set", "prod", run, "--project", "proj", "--eval-set", H1[:12])
    assert res.code == 0, res.err
    assert res.event("baseline_set")[0]["step"] == 5
    listed = invoke("baseline", "list", "--project", "proj", "--json")
    rows = json.loads(listed.out)
    assert listed.code == 0 and [r["tag"] for r in rows] == ["prod"]
    assert rows[0]["run_id"] == run and rows[0]["step"] == 5 and rows[0]["eval_set_hash"] == H1


def test_set_with_a_step_pins_that_step(invoke: Invoke, seeded: tuple[str, str]) -> None:
    run, _ = seeded
    assert invoke("baseline", "set", "old", run, "--project", "proj", "--step", "1").code == 0
    shown = json.loads(invoke("baseline", "show", "old", "--project", "proj", "--json").out)
    assert shown["baseline"]["step"] == 1 and shown["metrics"]["mAP50_95"] == 0.6


def test_eval_set_is_inferred_when_unambiguous_and_refused_when_not(
    invoke: Invoke, seeded: tuple[str, str]
) -> None:
    run, other = seeded
    assert invoke("baseline", "set", "a", run, "--project", "proj").code == 0
    res = invoke("baseline", "set", "b", other, "--project", "proj")
    assert res.code == 1 and "--eval-set" in res.human
    refused = res.event("baseline_refused")
    assert len(refused) == 1 and refused[0]["tag"] == "b" and refused[0]["run_id"] == other


def test_unknown_eval_set_is_refused_with_the_event(
    invoke: Invoke, seeded: tuple[str, str]
) -> None:
    run, _ = seeded
    res = invoke("baseline", "set", "prod", run, "--project", "proj", "--eval-set", H2)
    assert res.code == 1 and res.event("baseline_refused")[0]["eval_set_hash"] == H2
    assert invoke("baseline", "list", "--project", "proj", "--json").out.strip() == "[]"


def test_existing_tag_needs_replace(invoke: Invoke, seeded: tuple[str, str]) -> None:
    run, other = seeded
    args = ("--project", "proj", "--eval-set", H1)
    assert invoke("baseline", "set", "prod", run, *args).code == 0
    res = invoke("baseline", "set", "prod", other, *args)
    assert res.code == 1 and res.event("baseline_refused")
    again = invoke("baseline", "set", "prod", other, *args, "--replace")
    assert again.code == 0 and again.event("baseline_set")[0]["replaced"] is True
    shown = json.loads(invoke("baseline", "show", "prod", "--project", "proj", "--json").out)
    assert shown["baseline"]["run_id"] == other and shown["history"] == 2


@pytest.mark.parametrize("tag", ["Prod", "-x", "a b", "x" * 33, "tag:prod", "../x"])
def test_bad_tags_are_usage_errors(invoke: Invoke, seeded: tuple[str, str], tag: str) -> None:
    run, _ = seeded
    assert invoke("baseline", "set", tag, run, "--project", "proj", "--eval-set", H1).code == 2


def test_bad_run_id_and_bad_eval_set_are_usage_errors(
    invoke: Invoke, seeded: tuple[str, str]
) -> None:
    run, _ = seeded
    assert invoke("baseline", "set", "prod", "nope", "--project", "proj").code == 2
    for bad in ("zz", "ABCDEF12", "abc", H1 + "0"):
        res = invoke("baseline", "set", "prod", run, "--project", "proj", "--eval-set", bad)
        assert res.code == 2, bad


def test_unknown_run_and_missing_project(invoke: Invoke, seeded: tuple[str, str]) -> None:
    res = invoke("baseline", "set", "prod", "0" * 26, "--project", "proj", "--eval-set", H1)
    assert res.code == 1
    assert invoke("baseline", "list").code == 2


def test_show_unknown_tag_fails_and_list_of_empty_store_is_empty(
    invoke: Invoke, seeded: tuple[str, str]
) -> None:
    assert invoke("baseline", "show", "prod", "--project", "proj").code == 1
    assert invoke("baseline", "list", "--project", "proj").code == 0
    assert invoke("baseline", "list", "--project", "missing").code == 0


def test_show_text_lists_the_pin_and_headline_metrics(
    invoke: Invoke, seeded: tuple[str, str]
) -> None:
    run, _ = seeded
    invoke("baseline", "set", "prod", run, "--project", "proj", "--eval-set", H1)
    res = invoke("baseline", "show", "prod", "--project", "proj")
    assert res.code == 0
    assert f"run_id: {run}" in res.out and "mAP50_95: 0.7000" in res.out
    assert "history: 1" in res.out


def test_tampered_snapshot_makes_show_and_set_fail_with_the_integrity_event(
    invoke: Invoke, seeded: tuple[str, str], store_root: Path
) -> None:
    run, _ = seeded
    invoke("baseline", "set", "prod", run, "--project", "proj", "--eval-set", H1)
    path = sorted((store_root / "proj" / "evals" / run).glob("*_000000005.json"))[0]
    path.chmod(0o644)
    path.write_text(path.read_text().replace("0.7", "0.9"))
    res = invoke("baseline", "show", "prod", "--project", "proj")
    assert res.code == 1 and res.event("snapshot_integrity_failed")
    res = invoke("baseline", "set", "again", run, "--project", "proj", "--eval-set", H1)
    assert res.code == 1 and res.event("snapshot_integrity_failed")


def test_tampered_baseline_row_is_reported_with_its_own_event(
    invoke: Invoke, seeded: tuple[str, str], store_root: Path
) -> None:
    run, _ = seeded
    invoke("baseline", "set", "prod", run, "--project", "proj", "--eval-set", H1)
    store = open_project(store_root)
    try:
        store._conn.execute("DROP TRIGGER baseline_no_update")
        store._conn.execute("UPDATE baseline SET tag='evil'")
    finally:
        store.close()
    res = invoke("baseline", "list", "--project", "proj")
    assert res.code == 1 and res.event("baseline_integrity_failed")


@pytest.mark.parametrize("status", ["running", "failed", "preempted"])
def test_set_refuses_a_run_that_is_not_finished(
    invoke: Invoke, store_root: Path, status: str
) -> None:
    store = open_project(store_root)
    try:
        run = make_run(store, status=status)
        add_snapshot(store, run, h=H1)
    finally:
        store.close()
    res = invoke("baseline", "set", "prod", run, "--project", "proj")
    assert res.code == 1 and "must be finished" in res.human and status in res.human
    assert len(res.event("baseline_refused")) == 1
    assert json.loads(invoke("baseline", "list", "--project", "proj", "--json").out) == []


def test_the_python_api_refuses_an_unfinished_run_too(store_root: Path) -> None:
    from cvpilot.compare.baselines import set_baseline
    from cvpilot.core.errors import BaselineRefusedError

    store = open_project(store_root)
    try:
        run = make_run(store, status="running")
        add_snapshot(store, run, h=H1)
    finally:
        store.close()
    with pytest.raises(BaselineRefusedError, match="must be finished"):
        set_baseline("proj", "prod", run, store_dir=store_root)
