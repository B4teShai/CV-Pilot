from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog

from cvpilot import api, runs
from cvpilot.cli import app as cli
from cvpilot.core.hashing import new_run_id
from cvpilot.logging import clear_request_context

REQUIRED = {"@timestamp", "level", "message", "service", "env", "request_id", "logger"}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in (
        "CVPILOT_ENV",
        "CVPILOT_LOG_LEVEL",
        "CVPILOT_PROJECT",
        "CVPILOT_STORE_COMMAND_LINE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()  # earlier tests may have left a closed stream configured
    yield
    clear_request_context()
    structlog.reset_defaults()


def main(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["cvpilot", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def logs(err: str) -> list[dict[str, Any]]:
    return [json.loads(x) for x in err.splitlines() if x.startswith("{")]


def make_run(
    monkeypatch: pytest.MonkeyPatch, project: str = "proj", status: str = "finished"
) -> str:
    monkeypatch.setattr(sys, "argv", ["train.py", "--lr", "0.1"])
    run = api.init(
        project, task="detection", name="first", config={"lr": 0.1, "api_key": "k"}, seed=5
    )
    api.log({"train/loss": 1.5, "val/mAP": 0.25}, step=0)
    api.log({"train/loss": 1.0}, step=1, epoch=0)
    api.finish(status)  # type: ignore[arg-type]
    return run.run_id


# --- runs list -----------------------------------------------------------------


def test_runs_list_prints_one_line_per_run_newest_first(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    a = make_run(monkeypatch)
    b = make_run(monkeypatch)
    capsys.readouterr()
    assert main(monkeypatch, "runs", "list", "--project", "proj") == 0
    out, err = capsys.readouterr()
    lines = out.strip().splitlines()
    assert lines[0].split()[:2] == ["RUN_ID", "STATUS"]
    assert [ln.split()[0] for ln in lines[1:]] == [b, a]
    assert "finished" in lines[1] and "first" in lines[1]
    assert logs(err) == []  # no events on success


def test_runs_ls_is_an_alias(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rid = make_run(monkeypatch)
    capsys.readouterr()
    assert main(monkeypatch, "runs", "ls", "--project", "proj") == 0
    assert rid in capsys.readouterr().out


def test_runs_list_json_status_filter_and_env_project(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    done = make_run(monkeypatch)
    running = make_run(monkeypatch, status="failed")
    capsys.readouterr()
    monkeypatch.setenv("CVPILOT_PROJECT", "proj")
    assert main(monkeypatch, "runs", "list", "--json", "--status", "failed") == 0
    data = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in data] == [running] and data[0]["status"] == "failed"
    assert done not in json.dumps(data)


def test_runs_list_on_an_empty_project_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(monkeypatch, "runs", "list", "--project", "nothing", "--json") == 0
    assert json.loads(capsys.readouterr().out) == []
    assert not (tmp_path / "store" / "nothing").exists()


def test_missing_project_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(monkeypatch, "runs", "list") == 2
    out, err = capsys.readouterr()
    assert out == "" and "CVPILOT_PROJECT" in err and "Traceback" not in err
    assert logs(err) == []


@pytest.mark.parametrize("project", ["../x", "Bad Name", "a/b"])
def test_invalid_project_option_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], project: str
) -> None:
    assert main(monkeypatch, "runs", "list", "--project", project) == 2
    assert capsys.readouterr().out == ""


def test_invalid_status_option_exits_2(monkeypatch: pytest.MonkeyPatch) -> None:
    assert main(monkeypatch, "runs", "list", "--project", "p", "--status", "banana") == 2


@pytest.mark.parametrize(
    ("var", "value"), [("CVPILOT_STORE_DIR", "/"), ("CVPILOT_PROJECT", "Bad Name")]
)
def test_invalid_settings_exit_2_before_any_logging(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], var: str, value: str
) -> None:
    monkeypatch.setenv(var, value)
    assert main(monkeypatch, "runs", "list", "--project", "p") == 2
    out, err = capsys.readouterr()
    assert out == "" and var in err and logs(err) == []
    assert len(err.strip().splitlines()) == 1


# --- runs show -----------------------------------------------------------------


def test_runs_show_matches_the_stored_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rid = make_run(monkeypatch)
    capsys.readouterr()
    assert main(monkeypatch, "runs", "show", rid, "--project", "proj") == 0
    out, err = capsys.readouterr()
    view = runs.get_run_view("proj", rid, store_dir=tmp_path / "store")
    fields = dict(line.split(":", 1) for line in out.splitlines() if ":" in line and line[0] != " ")
    fields = {k.strip(): v.strip() for k, v in fields.items()}
    assert fields["run_id"] == rid and fields["status"] == "finished"
    assert fields["project"] == "proj" and fields["task"] == "detection"
    assert fields["name"] == "first" and fields["framework"] == view.meta.framework
    assert fields["started_at"] == view.meta.started_at.isoformat()
    assert fields["ended_at"] == view.meta.ended_at.isoformat()  # type: ignore[union-attr]
    assert fields["dataset_hash"] == view.repro.dataset_hash
    assert fields["command_line"] == "train.py --lr 0.1"
    assert fields["seed"] == "5"
    assert json.loads(fields["config"]) == {"lr": 0.1, "api_key": "[REDACTED]"}
    assert fields["n_points"] == "3" and fields["steps"] == "0..1"
    assert logs(err) == []


def test_runs_show_json_is_the_stored_record(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    rid = make_run(monkeypatch)
    capsys.readouterr()
    assert main(monkeypatch, "runs", "show", rid, "--project", "proj", "--json") == 0
    data = json.loads(capsys.readouterr().out)
    view = runs.get_run_view("proj", rid, store_dir=tmp_path / "store")
    assert data["meta"] == json.loads(view.meta.model_dump_json())
    assert data["repro"] == json.loads(view.repro.model_dump_json())
    assert data["metrics"] == {"n_points": 3, "n_files": 1, "min_step": 0, "max_step": 1}
    assert data["artifacts"] == []


def test_unknown_run_exits_1_with_one_line_and_one_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(monkeypatch)
    capsys.readouterr()
    missing = new_run_id()
    assert main(monkeypatch, "runs", "show", missing, "--project", "proj") == 1
    out, err = capsys.readouterr()
    assert out == ""
    human = [ln for ln in err.splitlines() if not ln.startswith("{")]
    assert len(human) == 1 and missing in human[0] and "Traceback" not in human[0]
    (rec,) = logs(err)
    assert REQUIRED <= rec.keys()
    assert rec["message"] == "command_failed" and rec["command"] == "runs"
    assert rec["level"] == "ERROR" and rec["service"] == "cvpilot"


def test_unknown_run_in_a_project_without_a_store_is_also_exit_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(monkeypatch, "runs", "show", new_run_id(), "--project", "ghost") == 1


@pytest.mark.parametrize("bad", ["nope", "../../etc/passwd", "01ARZ3NDEKTSV4RRFFQ69G5FAI", ""])
def test_malformed_run_id_is_a_usage_error_exit_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], bad: str
) -> None:
    for cmd in (("runs", "show"), ("repro",)):
        assert main(monkeypatch, *cmd, bad, "--project", "proj") == 2
    assert capsys.readouterr().out == ""


def test_lowercase_run_id_is_accepted(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rid = make_run(monkeypatch)
    capsys.readouterr()
    assert main(monkeypatch, "runs", "show", rid.lower(), "--project", "proj") == 0


def test_store_from_a_newer_library_exits_1_with_the_store_event(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    import sqlite3

    make_run(monkeypatch)
    db = sqlite3.connect(tmp_path / "store" / "proj" / "runs.sqlite")
    db.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    db.commit()
    db.close()
    capsys.readouterr()
    assert main(monkeypatch, "runs", "list", "--project", "proj") == 1
    _, err = capsys.readouterr()
    (rec,) = logs(err)
    assert rec["message"] == "store_version_unsupported"
    assert (rec["store_version"], rec["library_version"]) == (99, 1)


# --- repro ---------------------------------------------------------------------------


def test_repro_prints_a_script_on_stdout_only(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rid = make_run(monkeypatch)
    capsys.readouterr()
    assert main(monkeypatch, "repro", rid, "--project", "proj") == 0
    out, err = capsys.readouterr()
    assert out.startswith("#!/usr/bin/env bash") and rid in out
    assert "train.py --lr 0.1" in out and logs(err) == []


def test_repro_unknown_run_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    make_run(monkeypatch)
    assert main(monkeypatch, "repro", new_run_id(), "--project", "proj") == 1
