"""The README's commands run, and every other command block in it says it is not run."""

from __future__ import annotations

import re
from pathlib import Path

from docs_harness import RUNNABLE_LANGS, blocks, clean_env, run_page

ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "README.md"


def test_the_readme_quickstart_runs_and_prints_what_it_promises(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    results = run_page(README.read_text(), tmp_path / "work", clean_env(home))
    assert len(results) == 1
    assert "run_started" in results[0].stderr and "run_finished" in results[0].stderr


def test_every_other_command_block_in_the_readme_is_marked_notest() -> None:
    for block in blocks(README.read_text()):
        if block.lang in RUNNABLE_LANGS and block.file is None:
            executed = block.lang in {"bash", "sh", "shell"} and not block.notest
            assert block.notest or executed, f"README.md:{block.line}: add 'notest'"


def test_the_readme_names_no_unbuilt_feature_as_available() -> None:
    text = README.read_text()
    assert "Planned usage" not in text and "Planned: resume-safe checkpoints" not in text
    assert "no checkpoint manager" not in text


def test_the_recipes_that_left_the_readme_live_in_the_guide() -> None:
    """What the README no longer shows is on the page that now owns it."""
    docs = ROOT / "docs"
    assert "find_resume_point" in (docs / "guide" / "checkpoints.md").read_text()
    gating = (docs / "guide" / "gating-ci.md").read_text()
    for needle in ("--per-class-max-drop", "log_eval", "expect-baseline-chain", "iscrowd"):
        assert needle in gating, needle
    settings = (docs / "reference" / "settings.md").read_text()
    for name in ("CVPILOT_CKPT_KEY", "CVPILOT_FORWARD_TO", "CVPILOT_STORE_DIR", "130"):
        assert name in settings, name
    assert "Planned" in (docs / "status.md").read_text()


# The README is a front door: one plain sentence, one picture, one example, links.
MAX_LINES = 100
MAX_WORDS = 500
C4_IMAGES = ("context", "containers", "components", "data-model", "flow-a", "flow-b", "runtime")


def prose_words(text: str) -> int:
    """Words outside fenced code blocks; link targets do not count, link text does."""
    prose = re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.S | re.M)
    prose = re.sub(r"\]\([^)]*\)", "]", prose)
    return len(prose.split())


def test_the_readme_stays_short_enough_to_read_in_twenty_seconds() -> None:
    text = README.read_text()
    assert len(text.splitlines()) <= MAX_LINES
    assert prose_words(text) <= MAX_WORDS


def test_the_readme_has_exactly_one_picture_and_it_is_the_overview() -> None:
    images = re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", README.read_text())
    assert len(images) == 1 and images[0].endswith("/docs/diagrams/png/overview.png")
    assert not any(f"/png/{name}.png" in README.read_text() for name in C4_IMAGES)


def test_the_readme_leaves_internals_to_the_docs() -> None:
    text = README.read_text()
    for word in ("HMAC", "trust policy", "lineage", "quarantine", "eval_set_hash", "CVPILOT_"):
        assert word not in text, word


def test_every_git_url_install_in_the_readme_and_install_guide_is_pinned_to_a_full_commit() -> None:
    """A branch URL installs whatever the branch holds on the day: the default is a full commit."""
    pages = [README, ROOT / "docs" / "guide" / "install.md"]
    for page in pages:
        for number, line in enumerate(page.read_text().splitlines(), start=1):
            if "git+https://" in line:
                assert "<full-commit-sha>" in line, f"{page.name}:{number}: pin a commit: {line}"
    assert "git rev-parse HEAD" in (ROOT / "docs" / "guide" / "install.md").read_text()
