"""The harness that runs the tutorial must be strict about what it accepts: whole lines, in order,
on the stream they were promised on."""

from __future__ import annotations

from pathlib import Path

import pytest
from docs_harness import clean_env, run_page


def page(command: str, expected: str, *, stream: str = "") -> str:
    fence = "```"
    info = f"text expect {stream}".strip()
    return f"{fence}bash\n{command}\n{fence}\n\n{fence}{info}\n{expected}\n{fence}\n"


def run(tmp_path: Path, text: str) -> None:
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    run_page(text, tmp_path / "work", clean_env(home))


def test_a_whole_line_matches(tmp_path: Path) -> None:
    run(tmp_path, page("echo hello world", "hello world"))


def test_a_fragment_of_a_line_does_not(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="hello"):
        run(tmp_path, page("echo hello world", "hello"))


def test_a_longer_line_than_printed_does_not(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        run(tmp_path, page("echo hello", "hello world"))


def test_lines_must_come_in_order(tmp_path: Path) -> None:
    run(tmp_path, page("echo a; echo b; echo c", "a\nc"))
    with pytest.raises(AssertionError, match="order|b"):
        run(tmp_path, page("echo a; echo b", "b\na"))


def test_a_line_is_matched_once(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        run(tmp_path, page("echo a", "a\na"))


def test_wildcards_work_only_where_marked(tmp_path: Path) -> None:
    run(tmp_path, page("echo run 01ABC finished", "run {*} finished"))
    with pytest.raises(AssertionError):
        run(tmp_path, page("echo run 01ABC finished", "run * finished"))
    with pytest.raises(AssertionError):
        run(tmp_path, page("echo run 01ABC finished", "run .* finished"))


def test_stdout_and_stderr_are_checked_separately(tmp_path: Path) -> None:
    command = "echo to-out; echo to-err >&2"
    run(tmp_path, page(command, "to-out"))
    run(tmp_path, page(command, "to-err", stream="stderr"))
    with pytest.raises(AssertionError):
        run(tmp_path, page(command, "to-err"))  # promised on stdout, printed on stderr
    with pytest.raises(AssertionError):
        run(tmp_path, page(command, "to-out", stream="stderr"))


def test_a_json_log_line_is_matched_by_exact_field_values(tmp_path: Path) -> None:
    command = 'echo \'{"message": "run_started", "level": "INFO", "run_id": "01X"}\' >&2'
    run(tmp_path, page(command, "json: message=run_started level=INFO", stream="stderr"))
    with pytest.raises(AssertionError):
        run(tmp_path, page(command, "json: message=run_start", stream="stderr"))
    with pytest.raises(AssertionError):
        run(tmp_path, page(command, "json: message=run_started level=ERROR", stream="stderr"))


def test_an_expect_block_without_a_command_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="without a command"):
        run(tmp_path, "```text expect\nx\n```\n")


def test_a_failing_command_names_its_line(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="exit"):
        run(tmp_path, "```bash\nfalse\n```\n")
