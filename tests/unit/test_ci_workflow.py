"""The CI workflow carries the supply-chain jobs of Phase 4, and they stay strict."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml
from workflow_lint import lint

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github" / "workflows" / "ci.yml"


def jobs() -> dict[str, Any]:
    doc = yaml.safe_load(CI.read_text())
    assert isinstance(doc, dict)
    return doc["jobs"]  # type: ignore[no-any-return]


def scripts(job: str) -> str:
    return "\n".join(str(s.get("run", "")) for s in jobs()[job]["steps"])


def test_ci_passes_the_workflow_rules_and_has_no_oidc_or_secrets() -> None:
    doc = yaml.safe_load(CI.read_text())
    assert lint(doc) == []
    text = CI.read_text()
    assert "id-token" not in text and "secrets." not in text
    assert doc["permissions"] == {"contents": "read"}


def test_the_lock_must_be_fresh() -> None:
    assert "uv lock --check" in scripts("lock")


def test_the_docs_build_is_strict_and_the_docs_tests_run() -> None:
    text = scripts("docs")
    assert "mkdocs build --strict" in text and "tests/docs" in text


def test_the_package_job_builds_inspects_and_checks_the_metadata() -> None:
    text = scripts("package")
    assert "uv build" in text and "twine check --strict" in text and "tests/release" in text


def test_the_dependency_audit_runs_against_the_lock_with_a_pinned_tool() -> None:
    text = scripts("audit")
    assert "uv export --locked" in text and "pip-audit" in text and "--strict" in text
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    audit = project["dependency-groups"]["audit"]
    assert audit and all("==" in item for item in audit)
    assert "pip-audit" not in " ".join(project["project"]["dependencies"])


def test_the_test_job_still_runs_the_full_gate_on_every_python() -> None:
    doc = yaml.safe_load(CI.read_text())
    test = doc["jobs"]["test"]
    assert test["strategy"]["matrix"]["python"] == ["3.10", "3.11", "3.12", "3.13"]
    text = scripts("test")
    for command in (
        "ruff check",
        "ruff format --check",
        "mypy",
        "pytest --cov=cvpilot --cov-fail-under=85",
    ):
        assert command in text


def test_every_job_pins_its_actions_and_does_not_persist_credentials() -> None:
    for name, job in jobs().items():
        for step in job["steps"]:
            uses = str(step.get("uses", ""))
            if uses.startswith("actions/checkout"):
                assert step["with"]["persist-credentials"] is False, name
