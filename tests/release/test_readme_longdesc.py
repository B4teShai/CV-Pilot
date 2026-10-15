"""The README is the package index page: absolute links, no frozen release status, and the release
workflow refuses to build while either is wrong."""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("check_readme", ROOT / "scripts" / "check_readme.py")
assert spec is not None and spec.loader is not None
check_readme = importlib.util.module_from_spec(spec)
sys.modules["check_readme"] = check_readme
spec.loader.exec_module(check_readme)


def test_the_readme_is_clean() -> None:
    assert check_readme.problems((ROOT / "README.md").read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "text",
    [
        "See [the guide](docs/guide/trust.md).",
        "![diagram](docs/diagrams/png/context.png)",
        "[a](http://example.com/x)",
        "[a](https://example.org/x)",
        "Version 0.1.0 (release candidate; nothing has been published to any package index).",
        "Nothing has been uploaded yet.",
        "## Install (from source)",
        "This is a pre-release.",
    ],
)
def test_the_checker_refuses_what_would_break_or_go_stale_on_the_index(text: str) -> None:
    assert check_readme.problems(text) != []


@pytest.mark.parametrize(
    "text",
    [
        "[docs](https://github.com/B4teShai/CV-Pilot/blob/main/docs/guide/trust.md)",
        "![d](https://raw.githubusercontent.com/B4teShai/CV-Pilot/main/docs/diagrams/png/context.png)",
        "[top](#install)",
        "```bash\n# a release candidate is mentioned in code only\n```",
    ],
)
def test_the_checker_accepts_absolute_project_links_anchors_and_code(text: str) -> None:
    assert check_readme.problems(text) == []


def test_the_script_exits_nonzero_on_a_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "README.md"
    bad.write_text("[x](docs/y.md)\n")
    assert check_readme.main(["check_readme", str(bad)]) == 1
    good = tmp_path / "GOOD.md"
    good.write_text("# Title\n\nPlain text.\n")
    assert check_readme.main(["check_readme", str(good)]) == 0


def test_the_release_build_refuses_while_the_readme_is_wrong() -> None:
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text()
    build = workflow.split("  publish:")[0]
    assert "scripts/check_readme.py" in build
    assert build.index("scripts/check_readme.py") < build.index("uv build")


def test_the_status_of_a_release_lives_in_the_runbook_not_the_readme() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert not re.search(r"release candidate", readme, re.I)
    runbook = (ROOT / "docs" / "engineering" / "release.md").read_text(encoding="utf-8")
    assert "check_readme" in runbook
