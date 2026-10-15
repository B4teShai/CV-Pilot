"""Formatted exception text and error lines are scrubbed of secrets, home paths and ids."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog

from cvpilot import logging as cvlog
from cvpilot.cli import app as cli
from cvpilot.logging import clear_request_context, configure_logging, get_logger
from cvpilot.settings import Settings


@pytest.fixture(autouse=True)
def _clean() -> Any:
    yield
    clear_request_context()
    structlog.reset_defaults()


def emit_failure(message: str) -> dict[str, Any]:
    buf = io.StringIO()
    configure_logging("cvpilot", Settings(env="dev", log_level="DEBUG"), stream=buf)
    try:
        raise RuntimeError(message)
    except RuntimeError as exc:
        get_logger("t").error("command_failed", exc_info=exc)
    return json.loads(buf.getvalue().splitlines()[-1])  # type: ignore[no-any-return]


def test_scrub_removes_the_home_directory_and_credentials() -> None:
    home = str(Path.home())
    text = f"cannot open {home}/data/x.parquet; Authorization: Bearer abc.def-123 token=s3cr3t"
    out = cvlog.scrub(text)
    assert home not in out and "~/data/x.parquet" in out
    assert "abc.def-123" not in out and "s3cr3t" not in out and "[REDACTED]" in out


def test_formatted_tracebacks_in_log_events_are_scrubbed() -> None:
    home = str(Path.home())
    record = emit_failure(f"failed at {home}/secret/place with password=hunter2")
    blob = json.dumps(record)
    assert "hunter2" not in blob and f"{home}/secret" not in blob
    assert "~/secret/place" in record["exception"]


def test_error_lines_printed_by_the_cli_boundary_are_scrubbed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    from cvpilot.core.errors import CvpilotError

    home = str(Path.home())

    def boom(*a: object, **k: object) -> None:
        raise CvpilotError(f"store at {home}/.cvpilot/x is broken")

    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_run_doctor", boom)
    monkeypatch.setattr(sys, "argv", ["cvpilot", "doctor"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    err = capsys.readouterr().err
    assert exc.value.code == 1 and home not in err and "~/.cvpilot/x" in err


def test_validation_messages_do_not_echo_image_id_text() -> None:
    from cvpilot.core.errors import EvalInputError
    from cvpilot.eval import AnnotationTable, PredictionTable, evaluate

    person_file = "patient-jane-doe-0042.png"
    gts = AnnotationTable.from_columns(image_id=["other"], class_id=[0], boxes=[[0, 0, 5, 5]])
    preds = PredictionTable.from_columns(
        image_id=[person_file], class_id=[0], score=[0.5], boxes=[[0, 0, 5, 5]]
    )
    with pytest.raises(EvalInputError) as exc:
        evaluate(preds, gts, task="detection", class_names=["a"], n_resamples=0)
    assert person_file not in str(exc.value) and "sha256:" in str(exc.value)


def test_scrub_is_linear_on_hostile_text() -> None:
    import time

    start = time.perf_counter()
    for hostile in ("a" * 2_000_000, "token=" * 200_000, "Bearer " * 200_000, "-_" * 1_000_000):
        cvlog.scrub(hostile)
    assert time.perf_counter() - start < 5
