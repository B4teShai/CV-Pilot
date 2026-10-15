"""Where reports may be written: a report path must not become a way to overwrite other files."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest
from conftest import CliResult
from support import make_run, open_project, write_eval_files

from cvpilot.compare import html_report as hr
from cvpilot.core.errors import CvpilotError

Invoke = Callable[..., CliResult]


def policy(root: Path, *, store: Path | None = None, **kw: bool) -> hr.ReportPolicy:
    return hr.ReportPolicy(root=root, store_roots=(store,) if store else (), **kw)


def test_a_path_inside_the_root_is_written_and_marked_as_a_cvpilot_report(tmp_path: Path) -> None:
    target = tmp_path / "out" / "r.json"
    target.parent.mkdir()
    hr.write_report(target, '{"schema":"cvpilot.compare/1"}', policy(tmp_path))
    assert target.exists()


def test_a_path_outside_the_root_is_refused_unless_allowed(tmp_path: Path) -> None:
    root, other = tmp_path / "ws", tmp_path / "elsewhere"
    root.mkdir()
    other.mkdir()
    with pytest.raises(CvpilotError, match="outside"):
        hr.write_report(other / "r.html", "x", policy(root))
    hr.write_report(other / "r.html", "x", policy(root, allow_outside=True))
    assert (other / "r.html").exists()


def test_dot_dot_and_symlinked_parents_that_leave_the_root_are_refused(tmp_path: Path) -> None:
    root, other = tmp_path / "ws", tmp_path / "elsewhere"
    root.mkdir()
    other.mkdir()
    (root / "link").symlink_to(other, target_is_directory=True)
    with pytest.raises(CvpilotError, match="outside"):
        hr.write_report(root / "link" / "r.html", "x", policy(root))
    with pytest.raises(CvpilotError, match="outside"):
        hr.write_report(root / ".." / "elsewhere" / "r.html", "x", policy(root))
    assert not (other / "r.html").exists()
    (root / "inner").mkdir()
    (root / "alias").symlink_to(root / "inner", target_is_directory=True)
    hr.write_report(root / "alias" / "r.html", "x", policy(root))  # stays inside the root
    assert (root / "inner" / "r.html").exists()


def test_the_store_directory_is_never_a_report_target(tmp_path: Path) -> None:
    store = tmp_path / "store" / "proj"
    store.mkdir(parents=True)
    with pytest.raises(CvpilotError, match="store"):
        hr.write_report(store / "evals.html", "x", policy(tmp_path, store=tmp_path / "store"))
    with pytest.raises(CvpilotError, match="store"):
        hr.write_report(
            store / "evals.html",
            "x",
            policy(tmp_path, store=tmp_path / "store", allow_outside=True, force=True),
        )


@pytest.mark.parametrize(
    "variable",
    ["GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_PATH", "GITHUB_STEP_SUMMARY", "GITHUB_STATE"],
)
def test_files_named_by_the_runner_environment_are_never_report_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str
) -> None:
    runner_file = tmp_path / "runner.txt"
    runner_file.write_text("PATH=/safe\n")
    monkeypatch.setenv(variable, str(runner_file))
    for kw in ({}, {"force": True, "allow_outside": True}):
        with pytest.raises(CvpilotError, match=variable):
            hr.write_report(runner_file, "evil=1\n", policy(tmp_path, **kw))
    alias = tmp_path / "alias.txt"
    alias.symlink_to(runner_file)
    with pytest.raises(CvpilotError):
        hr.write_report(alias, "evil=1\n", policy(tmp_path, force=True))
    hard = tmp_path / "hard.txt"
    os.link(runner_file, hard)
    with pytest.raises(CvpilotError):
        hr.write_report(hard, "evil=1\n", policy(tmp_path))  # not ours: needs --force
    assert runner_file.read_text() == "PATH=/safe\n"


def test_a_file_the_tool_did_not_create_is_not_overwritten_without_force(tmp_path: Path) -> None:
    foreign = tmp_path / "notes.html"
    foreign.write_text("my own work")
    with pytest.raises(CvpilotError, match="force"):
        hr.write_report(foreign, "report", policy(tmp_path))
    assert foreign.read_text() == "my own work"
    hr.write_report(foreign, "report", policy(tmp_path, force=True))
    assert foreign.read_text() == "report"


def test_a_report_the_tool_created_can_be_replaced(tmp_path: Path) -> None:
    from support import build_snapshot

    from cvpilot.compare.engine import compare

    cmp = compare(build_snapshot(), build_snapshot())
    target = tmp_path / "r.html"
    hr.write_report(target, hr.render_html(cmp), policy(tmp_path))
    hr.write_report(target, hr.render_html(cmp), policy(tmp_path))  # recognised: no --force needed
    jsn = tmp_path / "r.json"
    from cvpilot.compare.report import report_json

    hr.write_report(jsn, report_json(cmp), policy(tmp_path))
    hr.write_report(jsn, report_json(cmp), policy(tmp_path))


def test_the_marker_is_in_every_report_format() -> None:
    from support import build_snapshot

    from cvpilot.compare.engine import compare
    from cvpilot.compare.report import report_json

    cmp = compare(build_snapshot(), build_snapshot())
    assert hr.REPORT_MARKER in hr.render_html(cmp) and hr.REPORT_MARKER in report_json(cmp)


def test_a_missing_parent_is_refused_with_a_policy(tmp_path: Path) -> None:
    with pytest.raises(CvpilotError, match="does not exist"):
        hr.write_report(tmp_path / "no" / "r.html", "x", policy(tmp_path))


# -- through the CLI ---------------------------------------------------------------------------


@pytest.fixture
def two_runs(invoke: Invoke, store_root: Path, tmp_path: Path) -> tuple[str, str]:
    ids = []
    for folder in ("a", "b"):
        store = open_project(store_root)
        try:
            run = make_run(store)
        finally:
            store.close()
        preds, gts, classes = write_eval_files(tmp_path / folder)
        res = invoke(
            "eval", run, "--project", "proj", "--preds", str(preds), "--gts", str(gts),
            "--classes", str(classes), "--slices", "camera=camera", "--allow-closed",
            "--resamples", "0",
        )  # fmt: skip
        assert res.code == 0, res.err
        ids.append(run)
    assert invoke("baseline", "set", "prod", ids[0], "--project", "proj").code == 0
    return ids[0], ids[1]


def run_gate(invoke: Invoke, run: str, *extra: str) -> CliResult:
    return invoke(
        "gate", run, "--project", "proj", "--baseline", "tag:prod", "--max-drop", "mAP50_95=0.5",
        "--min-images", "20", *extra,
    )  # fmt: skip


def test_the_cli_refuses_reports_in_the_store_or_outside_the_workspace(
    invoke: Invoke, store_root: Path, tmp_path: Path, two_runs: tuple[str, str]
) -> None:
    _, cand = two_runs
    res = run_gate(invoke, cand, "--report", str(store_root / "proj" / "r.html"))
    assert res.code == 1 and "store" in res.human
    outside = tmp_path.parent / "outside-report.html"
    res = run_gate(invoke, cand, "--report", str(outside))
    assert res.code == 1 and "outside" in res.human and not outside.exists()
    ok = run_gate(invoke, cand, "--report", str(outside), "--allow-outside")
    assert ok.code == 0 and outside.exists()
    outside.unlink()


def test_report_dir_sets_the_root_and_force_replaces_a_foreign_file(
    invoke: Invoke, tmp_path: Path, two_runs: tuple[str, str]
) -> None:
    _, cand = two_runs
    out = tmp_path / "artifacts"
    out.mkdir()
    ok = run_gate(invoke, cand, "--report-dir", str(out), "--report", str(out / "r.html"))
    assert ok.code == 0 and (out / "r.html").exists()
    res = run_gate(invoke, cand, "--report-dir", str(out), "--report", str(tmp_path / "r.html"))
    assert res.code == 1 and "outside" in res.human
    (out / "mine.json").write_text("{}")
    res = run_gate(invoke, cand, "--report-dir", str(out), "--report-json", str(out / "mine.json"))
    assert res.code == 1 and "force" in res.human and (out / "mine.json").read_text() == "{}"
    forced = run_gate(
        invoke, cand, "--report-dir", str(out), "--report-json", str(out / "mine.json"), "--force"
    )
    assert forced.code == 0 and '"schema"' in (out / "mine.json").read_text()


def test_the_runner_files_are_refused_through_the_cli(
    invoke: Invoke, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, two_runs: tuple[str, str]
) -> None:
    _, cand = two_runs
    summary = tmp_path / "summary.md"
    summary.write_text("# keep\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    res = run_gate(invoke, cand, "--report", str(summary), "--force", "--allow-outside")
    assert res.code == 1 and summary.read_text() == "# keep\n"
