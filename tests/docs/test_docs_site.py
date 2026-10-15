"""The documentation site builds strictly, loads nothing from a third party and is not committed;
every command block in the user guide is executed or says it is not."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from docs_harness import RUNNABLE_LANGS, blocks

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
pytest.importorskip("mkdocs")

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@pytest.fixture(scope="module")
def site(tmp_path_factory: pytest.TempPathFactory) -> Path:
    target = tmp_path_factory.mktemp("site")
    env = {**os.environ, "NO_MKDOCS_2_WARNING": "1"}
    subprocess.run(  # noqa: S603
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(target)],
        cwd=ROOT, env=env, check=True, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    return target


def test_the_site_builds_in_strict_mode_with_every_page(site: Path) -> None:
    for page in ("index.html", "tutorial/index.html", "reference/cli/index.html",
                 "guide/trust/index.html", "adr/0006-checkpoint-semantics/index.html",
                 "engineering/release/index.html"):  # fmt: skip
        assert (site / page).is_file(), page


ASSET_ATTR = re.compile(
    r"""<(?:script|img|source|iframe|video|audio|embed|object)\b[^>]*?\b(?:src|data)=["'](?:https?:)?//""",
    re.I,
)
LINK_ASSET = re.compile(r"""<link\b[^>]*\bhref=["'](?:https?:)?//""", re.I)


def test_nothing_is_loaded_from_a_third_party(site: Path) -> None:
    for path in site.rglob("*.html"):
        text = path.read_text(encoding="utf-8")
        assert not ASSET_ATTR.search(text), f"{path.name}: a remote script, image or frame"
        assert not LINK_ASSET.search(text), f"{path.name}: a remote stylesheet, font or icon"
        for needle in (
            "fonts.googleapis",
            "fonts.gstatic",
            "google-analytics",
            "googletagmanager",
            "cdn.jsdelivr",
            "unpkg.com",
        ):
            assert needle not in text, f"{path.name}: {needle}"
    for path in list(site.rglob("*.css")) + list(site.rglob("*.js")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        assert "@import url(http" not in text and "fonts.googleapis" not in text, path.name


def test_the_configuration_has_no_analytics_and_no_remote_fonts() -> None:
    text = (ROOT / "mkdocs.yml").read_text()
    assert "analytics" not in text and "font: false" in text and "strict: true" in text


def test_documentation_tools_are_pinned_and_not_core_dependencies() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    docs = project["dependency-groups"]["docs"]
    assert docs and all("==" in item for item in docs)
    core = " ".join(project["project"]["dependencies"]).lower()
    assert "mkdocs" not in core and "material" not in core
    for extra in project["project"]["optional-dependencies"].values():
        assert not any("mkdocs" in item for item in extra)


def test_the_built_site_is_not_committed() -> None:
    assert "site/" in (ROOT / ".gitignore").read_text().splitlines()
    tracked = subprocess.run(  # noqa: S603
        ["git", "-C", str(ROOT), "ls-files", "site"],  # noqa: S607
        capture_output=True, text=True, check=False,
    ).stdout.split()  # fmt: skip
    assert tracked == []


SECRET = re.compile(
    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
    r"(?i:api[_-]?key|token|password|secret)\s*=\s*['\"][^'\"\s]{6,}['\"]"
)


def test_no_page_contains_a_secret() -> None:
    for page in DOCS.rglob("*.md"):
        assert not SECRET.search(page.read_text(encoding="utf-8")), page.name


GUIDE_PAGES = [
    *sorted((DOCS / "guide").glob("*.md")),
    DOCS / "index.md",
    DOCS / "reference" / "api.md",
]


@pytest.mark.parametrize("page", GUIDE_PAGES, ids=lambda p: p.name)
def test_every_command_block_is_marked_not_executable(page: Path) -> None:
    """Guide pages illustrate; the tutorial is the executed page. A shell or Python block here
    must say ``notest`` so that nobody mistakes it for a command that was run."""
    for block in blocks(page.read_text(encoding="utf-8")):
        if block.lang in RUNNABLE_LANGS:
            assert block.notest, (
                f"{page.name}:{block.line}: add 'notest' or move it to the tutorial"
            )


def test_the_tutorial_is_the_only_docs_page_with_executed_commands() -> None:
    """Every other page may show commands, but each says ``notest``: nothing claims a run that no
    test performed."""
    for page in DOCS.rglob("*.md"):
        if page.name == "tutorial.md":
            continue
        for block in blocks(page.read_text(encoding="utf-8")):
            if block.lang in RUNNABLE_LANGS:
                assert block.notest, f"{page.relative_to(DOCS)}:{block.line}: add 'notest'"


def test_the_checkpoint_guide_states_what_restoring_does_not_do_and_what_cannot_be_stored() -> None:
    guide = (DOCS / "guide" / "checkpoints.md").read_text(encoding="utf-8")
    api = (DOCS / "reference" / "api.md").read_text(encoding="utf-8")
    assert "apply_rng" in guide and "apply_rng" in api
    assert "Restoring does not touch your generators" in guide
    for dtype in ("bfloat16", "float8"):
        assert dtype in guide
    changelog = (DOCS.parent / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "RNG state capture and restore" not in changelog


def test_every_flow_heading_appears_once_and_flow_b_is_not_a_copy_of_flow_a() -> None:
    text = (DOCS / "architecture" / "04-flows.md").read_text(encoding="utf-8")
    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert len(headings) == len(set(headings)), headings
    assert sum(h.startswith("## Flow A") for h in headings) == 1
    assert sum(h.startswith("## Flow B") for h in headings) == 1
    flow_b = text[text.index("## Flow B") :]
    for flow_a_only in ("`gate_failed`", "compare_refused", "`metrics_rejected`", "baseline set"):
        assert flow_a_only not in flow_b, f"Flow B carries a row of Flow A: {flow_a_only}"
    for behaviour in ("checkpoint_missing", "discover=True", "run_preempted", "apply_rng"):
        assert behaviour in flow_b
    assert "Flow B is planned" not in text
    assert "or on `cvpilot ckpt verify`" not in text  # verify reads and reports, it uploads nothing


def test_the_docs_match_the_code_on_the_points_that_drifted() -> None:
    model = (DOCS / "architecture" / "03-data-model.md").read_text(encoding="utf-8")
    assert "Planned: `CheckpointManifest`" not in model and "prev_step" in model
    components = (DOCS / "architecture" / "02-components.md").read_text(encoding="utf-8")
    assert "put_manifest(self, manifest: CheckpointManifest, manifest_sha256: str)" in components
    adr = (DOCS / "adr" / "0006-checkpoint-semantics.md").read_text(encoding="utf-8")
    assert "the budget is not met" in adr and "Azure metadata" not in adr
    release = (DOCS / "engineering" / "release.md").read_text(encoding="utf-8")
    assert "shasum -a 256" in release and "dist/cvpilot-*.whl" not in release
    assert "`cvpilot` (no\nseparator) is a different one" in release
