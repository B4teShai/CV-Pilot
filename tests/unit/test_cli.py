import json
import sys
from typing import Any

import pytest
import typer

from cvpilot import __version__
from cvpilot.cli import app as cli
from cvpilot.logging import clear_request_context

REQUIRED = {"@timestamp", "level", "message", "service", "env", "request_id", "logger"}


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.delenv("CVPILOT_ENV", raising=False)
    monkeypatch.delenv("CVPILOT_LOG_LEVEL", raising=False)
    yield
    clear_request_context()


def run(monkeypatch: pytest.MonkeyPatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["cvpilot", *argv])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    return int(exc.value.code or 0)


def log_lines(err: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in err.splitlines() if line.startswith("{")]


@pytest.mark.parametrize("flag", ["--version", "--help"])
def test_version_and_help_are_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], flag: str
) -> None:
    assert run(monkeypatch, flag) == 0
    out, err = capsys.readouterr()
    assert out.strip() != "" and err == ""
    if flag == "--version":
        assert __version__ in out


def test_doctor_emits_one_compliant_log_line(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "doctor") == 0
    out, err = capsys.readouterr()
    assert out == "ok\n"
    (rec,) = log_lines(err)
    assert REQUIRED <= rec.keys()
    assert rec["message"] == "doctor_completed" and rec["service"] == "cvpilot"


@pytest.mark.parametrize("var", ["CVPILOT_ENV", "CVPILOT_LOG_LEVEL"])
def test_invalid_setting_exits_2_without_traceback(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], var: str
) -> None:
    monkeypatch.setenv(var, "banana")
    assert run(monkeypatch, "doctor") == 2
    out, err = capsys.readouterr()
    assert out == "" and var in err and "Traceback" not in err
    assert len(err.strip().splitlines()) == 1


def test_unknown_command_exits_2(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(monkeypatch, "nope") == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert not any(line.startswith("{") for line in err.splitlines())


def test_abort_exits_130_silently(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def abort() -> None:
        raise typer.Abort()

    monkeypatch.setattr(cli, "_run_doctor", abort)
    assert run(monkeypatch, "doctor") == 130
    out, err = capsys.readouterr()
    assert out == "" and err == ""


def test_exception_with_zero_exit_code_is_still_a_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    class Odd(Exception):
        exit_code = 0

        def show(self) -> None:
            raise AssertionError("must not be treated as a usage error")

    def boom() -> None:
        raise Odd("odd")

    monkeypatch.setattr(cli, "_run_doctor", boom)
    assert run(monkeypatch, "doctor") == 1
    _, err = capsys.readouterr()
    (rec,) = log_lines(err)
    assert rec["message"] == "command_failed"


def test_unexpected_error_is_logged_once_and_exits_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def boom() -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "_run_doctor", boom)
    assert run(monkeypatch, "doctor") == 1
    out, err = capsys.readouterr()
    assert out == ""
    (rec,) = log_lines(err)
    assert rec["message"] == "command_failed" and rec["level"] == "ERROR"
    assert rec["command"] == "doctor" and "RuntimeError: boom" in rec["exception"]
    assert err.count("Traceback") == 1  # only inside the JSON `exception` field


def test_doctor_reports_a_hook_whose_framework_patch_target_is_missing(
    invoke: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvpilot.hooks import rfdetr

    monkeypatch.setattr(rfdetr, "compatibility", lambda: "rfdetr.training has no build_trainer")
    result = invoke("doctor")
    assert result.code == 0 and result.out.strip() == "ok"  # a warning, not a failure
    (event,) = result.event("doctor_completed")
    assert event["hook_problems"] == ["rfdetr: rfdetr.training has no build_trainer"]
    assert [e["level"] for e in result.event("hook_incompatible")] == ["WARN"]
