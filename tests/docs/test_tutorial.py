"""Every command of docs/tutorial.md runs, and the output it promises is the output it gives."""

from __future__ import annotations

from pathlib import Path

import pytest
from docs_harness import blocks, clean_env, run_page

ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = ROOT / "docs" / "tutorial.md"


@pytest.mark.skipif(not hasattr(__import__("os"), "kill"), reason="needs POSIX signals")
def test_the_tutorial_runs_end_to_end(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    results = run_page(TUTORIAL.read_text(), tmp_path / "work", clean_env(home))
    assert len(results) >= 12
    joined = "\n".join(r.output for r in results)
    for milestone in (
        "preempted: checkpoint saved at step 14",
        "resumed from step 10",
        "gate: FAIL",
        "gate: PASS",
        "authenticated",
    ):
        assert milestone in joined, milestone


def test_the_tutorial_has_no_unexecuted_command_blocks() -> None:
    for block in blocks(TUTORIAL.read_text()):
        if block.lang in {"bash", "sh", "shell", "console"}:
            assert not block.notest, f"line {block.line}: tutorial commands must all run"
