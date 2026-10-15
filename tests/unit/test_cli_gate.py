"""``cvpilot compare`` and ``cvpilot gate`` through the CLI, with real evaluations and stores."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import CliResult
from support import add_snapshot, make_run, open_project, write_eval_files

Invoke = Callable[..., CliResult]


def evaluate_into(invoke: Invoke, store_root: Path, folder: Path, **kw: object) -> str:
    """Create a finished run and evaluate generated files into it with ``cvpilot eval``."""
    store = open_project(store_root)
    try:
        run = make_run(store)
    finally:
        store.close()
    preds, gts, classes = write_eval_files(folder, **kw)  # type: ignore[arg-type]
    res = invoke(
        "eval", run, "--project", "proj", "--preds", str(preds), "--gts", str(gts),
        "--classes", str(classes), "--slices", "camera=camera", "--allow-closed",
        "--resamples", "0",
    )  # fmt: skip
    assert res.code == 0, res.err
    return run


@pytest.fixture
def baseline_run(invoke: Invoke, store_root: Path, tmp_path: Path) -> str:
    run = evaluate_into(invoke, store_root, tmp_path / "base")
    assert invoke("baseline", "set", "prod", run, "--project", "proj").code == 0
    return run


def gate(invoke: Invoke, run: str, *extra: str) -> CliResult:
    return invoke(
        "gate", run, "--project", "proj", "--baseline", "tag:prod",
        "--min-images", "20", *extra,
    )  # fmt: skip


def test_planted_regression_exits_1_and_names_the_violation(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    res = gate(invoke, bad, "--max-drop", "mAP50_95=0.5", "--max-drop", "mAP50=0.5")
    assert res.code == 1, res.err
    assert res.out.startswith("gate: FAIL")
    assert "[headline_drop] mAP50_95" in res.out
    failed = res.event("gate_failed")
    assert len(failed) == 1 and failed[0]["run_id"] == bad and failed[0]["baseline_tag"] == "prod"
    assert failed[0]["n_violations"] == 2 and failed[0]["worst_delta"] < 0
    assert not res.event("gate_passed")


def test_equal_run_exits_0(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    res = gate(
        invoke, same, "--max-drop", "mAP50_95=0.5", "--per-class-max-drop", "AP50_95=2.0",
        "--slice-max-drop", "camera=*:1.5",
    )  # fmt: skip
    assert res.code == 0, res.err
    assert res.out.startswith("gate: PASS") and res.event("gate_passed")


def test_improvement_passes(invoke: Invoke, store_root: Path, tmp_path: Path) -> None:
    worse = evaluate_into(invoke, store_root, tmp_path / "w", broken=10)
    assert invoke("baseline", "set", "prod", worse, "--project", "proj").code == 0
    better = evaluate_into(invoke, store_root, tmp_path / "b")
    assert gate(invoke, better, "--max-drop", "mAP50_95=0").code == 0


def test_mismatched_eval_set_is_refused_with_exit_1_and_no_verdict(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    other = evaluate_into(invoke, store_root, tmp_path / "other", shift_gt=True)
    res = gate(invoke, other, "--max-drop", "mAP50_95=50")
    assert res.code == 1
    refused = res.event("compare_refused")
    assert len(refused) == 1 and refused[0]["reason"] == "eval_set_hash"
    assert refused[0]["candidate_eval_set_hash"] != refused[0]["baseline_eval_set_hash"]
    assert "gate: PASS" not in res.out and not res.event("gate_passed")
    cmp = invoke("compare", other, "--project", "proj", "--baseline", "tag:prod")
    assert cmp.code == 1 and cmp.event("compare_refused")


def test_candidate_without_any_snapshot_is_refused(
    invoke: Invoke, store_root: Path, baseline_run: str
) -> None:
    store = open_project(store_root)
    try:
        empty = make_run(store)
    finally:
        store.close()
    res = gate(invoke, empty, "--max-drop", "mAP50_95=50")
    assert res.code == 1 and res.event("compare_refused")


def test_unknown_tag_run_and_project_fail(invoke: Invoke, baseline_run: str) -> None:
    res = invoke(
        "gate",
        baseline_run,
        "--project",
        "proj",
        "--baseline",
        "tag:nope",
        "--max-drop",
        "mAP50_95=1",
    )
    assert res.code == 1 and "nope" in res.human
    res = invoke(
        "gate", "0" * 26, "--project", "proj", "--baseline", "tag:prod", "--max-drop", "mAP50_95=1"
    )
    assert res.code == 1
    res = invoke(
        "gate",
        baseline_run,
        "--project",
        "ghost",
        "--baseline",
        "tag:prod",
        "--max-drop",
        "mAP50_95=1",
    )
    assert res.code == 1


def test_missing_metric_and_too_few_images_fail_closed(invoke: Invoke, baseline_run: str) -> None:
    res = gate(invoke, baseline_run, "--max-drop", "mAP_small=1")
    assert res.code == 1 and "[headline_missing] mAP_small" in res.out
    res = invoke(
        "gate", baseline_run, "--project", "proj", "--baseline", "tag:prod",
        "--max-drop", "mAP50_95=1", "--min-images", "41",
    )  # fmt: skip
    assert res.code == 1 and "[too_few_images]" in res.out


def test_absent_slice_fails_closed(invoke: Invoke, baseline_run: str) -> None:
    res = gate(invoke, baseline_run, "--slice-max-drop", "site=*:1.5")
    assert res.code == 1 and "[slice_no_match]" in res.out
    res = gate(invoke, baseline_run, "--slice-max-drop", "camera=c9:1.5")
    assert res.code == 1 and "[slice_missing]" in res.out


def test_small_slices_fail_unless_skipping_is_requested(invoke: Invoke, baseline_run: str) -> None:
    args = ("--slice-max-drop", "camera=*:1.5", "--min-images", "30")
    res = invoke("gate", baseline_run, "--project", "proj", "--baseline", "tag:prod", *args)
    assert res.code == 1 and "[slice_too_small]" in res.out  # each camera has 20 images
    res = invoke(
        "gate",
        baseline_run,
        "--project",
        "proj",
        "--baseline",
        "tag:prod",
        *args,
        "--skip-small-slices",
    )
    assert res.code == 1 and "[slice_none_gated]" in res.out
    ok = invoke(
        "gate", baseline_run, "--project", "proj", "--baseline", "tag:prod",
        "--slice-max-drop", "camera=*:1.5", "--min-images", "20",
    )  # fmt: skip
    assert ok.code == 0, ok.out


@pytest.mark.parametrize(
    "extra",
    [
        [],  # no threshold at all
        ["--max-drop", "mAP50_95"],
        ["--max-drop", "mAP50_95=-1"],
        ["--max-drop", "mAP50_95=nan"],
        ["--max-drop", "mAP50_95=1", "--max-drop", "mAP50_95=2"],
        ["--per-class-max-drop", "=1"],
        ["--slice-max-drop", "camera=*"],
        ["--max-drop", "mAP50_95=1", "--min-images", "0"],
        ["--max-drop", "mAP50_95=1", "--baseline", "nonsense"],
    ],
)
def test_bad_threshold_syntax_is_a_usage_error(
    invoke: Invoke, baseline_run: str, extra: list[str]
) -> None:
    argv = ["gate", baseline_run, "--project", "proj", "--baseline", "tag:prod", *extra]
    res = invoke(*argv)
    assert res.code == 2, (extra, res.out, res.err)
    assert not res.event("gate_passed")


def test_gate_by_run_id_baseline(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    res = invoke(
        "gate", same, "--project", "proj", "--baseline", baseline_run, "--max-drop", "mAP50_95=0.5"
    )
    assert res.code == 0, res.err
    assert res.event("gate_passed")[0]["baseline_tag"] is None


def test_tampered_candidate_snapshot_fails_closed(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    cand = evaluate_into(invoke, store_root, tmp_path / "c", broken=10)
    path = next((store_root / "proj" / "evals" / cand).glob("*.json"))
    path.chmod(0o644)
    text = path.read_text()
    path.write_text(text.replace('"mAP50_95":0.', '"mAP50_95":0.9'))  # make it look better
    res = gate(invoke, cand, "--max-drop", "mAP50_95=0.5")
    assert res.code == 1 and res.event("snapshot_integrity_failed")
    assert not res.event("gate_passed")


def test_tampered_baseline_snapshot_and_row_fail_closed(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "s")
    path = next((store_root / "proj" / "evals" / baseline_run).glob("*.json"))
    path.chmod(0o644)
    path.write_text(path.read_text().replace('"mAP50_95":1.0', '"mAP50_95":0.1'))
    res = gate(invoke, same, "--max-drop", "mAP50_95=0.5")
    assert res.code == 1 and res.event("snapshot_integrity_failed")


def test_tampered_baseline_row_fails_closed(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "s")
    store = open_project(store_root)
    try:
        store._conn.execute("DROP TRIGGER baseline_no_update")
        store._conn.execute("UPDATE baseline SET chain_sha256=?", ("0" * 64,))
    finally:
        store.close()
    res = gate(invoke, same, "--max-drop", "mAP50_95=0.5")
    assert res.code == 1 and res.event("baseline_integrity_failed")


def test_a_planted_report_or_result_file_cannot_change_the_verdict(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    reports = store_root / "proj" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{bad}_vs_{baseline_run}.html").write_text("gate: PASS")
    (reports / f"{bad}_vs_{baseline_run}.json").write_text('{"passed": true, "violations": []}')
    (store_root / "proj" / "gate.json").write_text('{"passed": true}')
    assert gate(invoke, bad, "--max-drop", "mAP50_95=0.5").code == 1


def test_hostile_class_names_cannot_inject_ci_commands(invoke: Invoke, store_root: Path) -> None:
    names = ["::error::pwn", "::add-mask::x‮txt"]
    store = open_project(store_root)
    try:
        base = make_run(store)
        cand = make_run(store)
        per = lambda v: {n: {"AP50_95": v, "support": 9.0} for n in names}  # noqa: E731
        add_snapshot(store, base, class_names=names, per_class=per(0.9))
        add_snapshot(store, cand, class_names=names, per_class=per(0.1))
    finally:
        store.close()
    res = invoke(
        "gate", cand, "--project", "proj", "--baseline", base, "--per-class-max-drop",
        "AP50_95=1", "--min-images", "1",
    )  # fmt: skip
    assert res.code == 1
    for line in res.out.splitlines():
        assert line.startswith(("gate:", "  ")), line
    assert "‮" not in res.out and "\\u202e" in res.out


def test_compare_prints_headline_worst_classes_and_slices(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    res = invoke("compare", bad, "--project", "proj", "--baseline", "tag:prod")
    assert res.code == 0, res.err
    assert "mAP50_95" in res.out and "camera=c0" in res.out and "background" in res.out
    assert res.event("compare_completed")
    as_json = json.loads(
        invoke("compare", bad, "--project", "proj", "--baseline", "tag:prod", "--json").out
    )
    assert as_json["headline"][0]["name"] == "mAP50_95"


def test_an_unexpected_exception_in_the_gate_never_exits_0(
    invoke: Invoke, baseline_run: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvpilot.compare import service

    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("plugin exploded")

    monkeypatch.setattr(service, "gate", boom)
    res = gate(invoke, baseline_run, "--max-drop", "mAP50_95=50")
    assert res.code == 1 and res.event("command_failed") and not res.event("gate_passed")


def test_json_output_carries_the_verdict_and_the_units(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    res = gate(invoke, bad, "--max-drop", "mAP50_95=0.5", "--json")
    assert res.code == 1
    doc = json.loads(res.out)
    assert (
        doc["gate"]["passed"] is False and doc["gate"]["violations"][0]["kind"] == "headline_drop"
    )
    assert "0..100" in doc["gate"]["thresholds"]["unit"]
    assert doc["gate"]["thresholds"]["max_drop"] == {"mAP50_95": 0.5}


def test_reports_are_written_for_a_failing_gate_and_are_valid(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    html_path, json_path = tmp_path / "r.html", tmp_path / "r.json"
    res = gate(
        invoke, bad, "--max-drop", "mAP50_95=0.5", "--report", str(html_path),
        "--report-json", str(json_path),
    )  # fmt: skip
    assert res.code == 1
    page = html_path.read_text()
    assert "Gate FAIL" in page and "Content-Security-Policy" in page and "background" in page
    doc = json.loads(json_path.read_text())
    assert doc["gate"]["passed"] is False and doc["confusion"]["n_changed_cells"] > 0


def test_a_refused_comparison_writes_no_report(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    other = evaluate_into(invoke, store_root, tmp_path / "other", shift_gt=True)
    target = tmp_path / "r.html"
    res = gate(invoke, other, "--max-drop", "mAP50_95=50", "--report", str(target))
    assert res.code == 1 and not target.exists()


def test_a_report_that_cannot_be_written_fails_even_when_the_gate_passes(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    same = evaluate_into(invoke, store_root, tmp_path / "same")
    link = tmp_path / "link.html"
    link.symlink_to(tmp_path / "elsewhere.html")
    res = gate(invoke, same, "--max-drop", "mAP50_95=50", "--report", str(link))
    assert res.code == 1 and "symbolic link" in res.human
    assert not (tmp_path / "elsewhere.html").exists()


def test_compare_writes_both_reports(
    invoke: Invoke, store_root: Path, tmp_path: Path, baseline_run: str
) -> None:
    bad = evaluate_into(invoke, store_root, tmp_path / "bad", broken=10)
    res = invoke(
        "compare", bad, "--project", "proj", "--baseline", "tag:prod", "--report",
        str(tmp_path / "c.html"), "--report-json", str(tmp_path / "c.json"),
    )  # fmt: skip
    assert res.code == 0
    assert "<h2>Headline</h2>" in (tmp_path / "c.html").read_text()
    assert json.loads((tmp_path / "c.json").read_text())["schema"] == "cvpilot.compare/1"
