"""The README's commands run, and every other command block in it says it is not run."""

from __future__ import annotations

from pathlib import Path

from docs_harness import RUNNABLE_LANGS, blocks, clean_env, run_page

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def test_the_readme_quickstart_runs_and_prints_what_it_promises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    results = run_page(README.read_text(), tmp_path / "work", clean_env(home))
    assert len(results) == 1 and "run_started" in results[0].output


def test_every_other_command_block_in_the_readme_is_marked_notest() -> None:
    for block in blocks(README.read_text()):
        if block.lang in RUNNABLE_LANGS and block.file is None:
            executed = block.lang in {"bash", "sh", "shell"} and not block.notest
            assert block.notest or executed, f"README.md:{block.line}: add 'notest'"


def test_the_readme_names_no_unbuilt_feature_as_available() -> None:
    text = README.read_text()
    assert "Planned usage" not in text and "Planned: resume-safe checkpoints" not in text
    assert "no checkpoint manager" not in text
