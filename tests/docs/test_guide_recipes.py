"""The restart recipe of the checkpoint guide is executed: a killed job and its restart."""

from __future__ import annotations

from pathlib import Path

from docs_harness import blocks, clean_env, run_page

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "docs" / "guide" / "checkpoints.md"

COMMANDS = """\
export CVPILOT_PROJECT=fridge-sku CVPILOT_STORE_DIR="$PWD/state/store" \\
  CVPILOT_CHECKPOINT_DIR="$PWD/state/ckpt"
python restart.py
python restart.py
"""

EXPECT_STDOUT = "starting at step 0\nstarting at step 3\n"
EXPECT_STDERR = (
    "json: message=run_started project=fridge-sku status=running\n"
    "json: message=run_started project=fridge-sku status=resumed\n"
    "json: message=checkpoint_restored step=2 kind=last trust_level=anchored\n"
)


def test_the_restart_recipe_in_the_checkpoint_guide_runs_twice_and_resumes(tmp_path: Path) -> None:
    """The guide's recipe, character for character, is a job killed at step 2 and its restart."""
    guide_blocks = blocks(GUIDE.read_text(encoding="utf-8"))
    recipe = next(b for b in guide_blocks if "find_resume_point" in b.body)
    page = (
        f"```python file=restart.py\n{recipe.body}```\n\n"
        f"```bash\n{COMMANDS}```\n\n"
        f"```text expect\n{EXPECT_STDOUT}```\n\n"
        f"```text expect stderr\n{EXPECT_STDERR}```\n"
    )
    home = tmp_path / "home"
    home.mkdir()
    results = run_page(page, tmp_path / "work", clean_env(home))
    assert len(results) == 1
