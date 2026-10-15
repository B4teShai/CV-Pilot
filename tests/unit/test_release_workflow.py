"""The release workflow: tag-triggered, a secret-free build job, an OIDC publish job in a protected
environment, pinned actions, and a lint that catches every way of breaking that."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from workflow_lint import lint, lint_release

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / ".github" / "workflows" / "release.yml"
CI = ROOT / ".github" / "workflows" / "ci.yml"


def load(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict)
    return doc


def release() -> tuple[dict[str, Any], str]:
    return load(RELEASE), RELEASE.read_text()


def test_the_shipped_release_workflow_passes_its_rules() -> None:
    doc, text = release()
    assert lint_release(doc, text) == []


def test_id_token_write_is_allowed_only_where_the_release_lint_says() -> None:
    doc, _ = release()
    assert lint(doc) != []  # the generic lint refuses any write permission
    assert lint(doc, allow_oidc={"publish"}) == []
    ci = load(CI)
    assert lint(ci, allow_oidc={"publish"}) == []
    assert "id-token" not in CI.read_text()  # CI never gets an OIDC token


def trig(doc: dict[str, Any]) -> dict[str, Any]:
    """The ``on`` block (YAML 1.1 reads the bare key ``on`` as ``True``)."""
    block = doc["on"] if "on" in doc else doc[True]
    assert isinstance(block, dict)
    return block


def steps_by_job(doc: dict[str, Any], job: str) -> list[dict[str, Any]]:
    return list(doc["jobs"][job]["steps"])


MUTATIONS: dict[str, tuple[Callable[[dict[str, Any]], None], str]] = {
    "branch trigger": (lambda d: trig(d)["push"].update(branches=["main"]), "tags"),
    "pull request trigger": (lambda d: trig(d).update(pull_request={}), "trigger"),
    "manual trigger": (lambda d: trig(d).update(workflow_dispatch={}), "trigger"),
    "any tag": (lambda d: trig(d)["push"].update(tags=["*"]), "tags"),
    "workflow write permissions": (
        lambda d: d.update(permissions={"contents": "write"}),
        "permissions",
    ),
    "no top level permissions": (lambda d: d.pop("permissions"), "permissions"),
    "build can publish": (
        lambda d: d["jobs"]["build"]["permissions"].update({"id-token": "write"}),
        "build",
    ),
    "build has a protected environment": (
        lambda d: d["jobs"]["build"].update(environment="pypi"),
        "build",
    ),
    "build gets contents write": (
        lambda d: d["jobs"]["build"]["permissions"].update(contents="write"),
        "build",
    ),
    "publish without id-token": (
        lambda d: d["jobs"]["publish"].update(permissions={"contents": "read"}),
        "publish",
    ),
    "publish with extra permissions": (
        lambda d: d["jobs"]["publish"]["permissions"].update(contents="write"),
        "publish",
    ),
    "publish without environment": (
        lambda d: d["jobs"]["publish"].pop("environment"),
        "environment",
    ),
    "publish does not need build": (lambda d: d["jobs"]["publish"].pop("needs"), "needs"),
    "a third job": (
        lambda d: d["jobs"].update(
            extra={"runs-on": "ubuntu-latest", "permissions": {"contents": "read"}, "steps": []}
        ),
        "jobs",
    ),
    "publish checks out code": (
        lambda d: d["jobs"]["publish"]["steps"].insert(
            0, {"uses": "actions/checkout@" + "a" * 40 + " # v4"}
        ),
        "checkout",
    ),
    "unpinned action": (
        lambda d: d["jobs"]["build"]["steps"].append({"uses": "actions/checkout@v4"}),
        "pinned",
    ),
    "static token": (
        lambda d: d["jobs"]["publish"]["steps"][-1]["with"].update(password="x"),  # noqa: S106
        "token",
    ),
    "secret reference": (
        lambda d: d["jobs"]["build"]["steps"][0].update(env={"T": "${{ secrets.PYPI_TOKEN }}"}),
        "secret",
    ),
    "expression in a script": (
        lambda d: d["jobs"]["build"]["steps"].append({"run": 'echo "${{ github.ref_name }}"'}),
        "expression",
    ),
    "no tag check": (
        lambda d: d["jobs"]["build"]["steps"].__setitem__(
            slice(None),
            [
                s
                for s in d["jobs"]["build"]["steps"]
                if "print_version" not in str(s.get("run", ""))
            ],
        ),
        "version",
    ),
    "no digests": (
        lambda d: [
            s.update(run=str(s["run"]).replace("sha256sum", "true"))
            for j in d["jobs"].values()
            for s in j["steps"]
            if "run" in s
        ],
        "digest",
    ),
    "different artifact names": (
        lambda d: [
            s["with"].update(name="other")
            for s in d["jobs"]["publish"]["steps"]
            if str(s.get("uses", "")).startswith("actions/download-artifact")
        ],
        "artifact",
    ),
    "publish action in build": (
        lambda d: d["jobs"]["build"]["steps"].append(
            {"uses": "pypa/gh-action-pypi-publish@" + "a" * 40}
        ),
        "publish",
    ),
    "pull_request_target": (
        lambda d: trig(d).update(pull_request_target={}),
        "pull_request_target",
    ),
}


@pytest.mark.parametrize("name", list(MUTATIONS))
def test_the_lint_catches_each_way_of_breaking_the_release_pipeline(name: str) -> None:
    mutate, keyword = MUTATIONS[name]
    doc, text = release()
    broken = copy.deepcopy(doc)
    mutate(broken)
    problems = lint_release(broken, text)
    assert problems, f"{name}: nothing was reported"
    assert any(keyword in p.lower() for p in problems), (name, problems)


def test_every_uses_line_is_a_full_sha_with_a_version_comment() -> None:
    lines = [ln for ln in RELEASE.read_text().splitlines() if "uses:" in ln]
    assert lines
    for line in lines:
        assert re.search(r"uses: [\w./-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+$", line), line


def test_the_pins_are_the_verified_ones() -> None:
    """Resolved with ``gh api repos/<owner>/<repo>/git/ref/tags/<tag>`` (and, for an annotated tag,
    ``git/tags/<sha>``) on 2026-10-15; a changed pin must be re-verified the same way."""
    expected = {
        "actions/checkout": ("v4.2.2", "11bd71901bbe5b1630ceea73d27597364c9af683"),
        "astral-sh/setup-uv": ("v5.4.2", "d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86"),
        "actions/upload-artifact": ("v4.6.2", "ea165f8d65b6e75b540449e92b4886f43607fa02"),
        "actions/download-artifact": ("v4.3.0", "d3f86a106a0bac45b974a628896c90dbdf5c8093"),
        "pypa/gh-action-pypi-publish": ("v1.12.4", "76f52bc884231f62b9a034ebfe128415bbaabdfc"),
    }
    seen: dict[str, tuple[str, str]] = {}
    for path in (RELEASE, CI):
        for match in re.finditer(r"uses: ([\w./-]+)@([0-9a-f]{40}) # (v[\d.]+)", path.read_text()):
            seen[match.group(1)] = (match.group(3), match.group(2))
    for action, pin in seen.items():
        assert expected.get(action) == pin, f"{action} {pin} is not a verified pin"
    assert {"actions/upload-artifact", "pypa/gh-action-pypi-publish"} <= set(seen)


def test_the_publish_job_runs_no_repository_code_and_downloads_the_exact_artifact() -> None:
    doc, _ = release()
    publish = steps_by_job(doc, "publish")
    uses = [str(s.get("uses", "")) for s in publish]
    assert not any(u.startswith("actions/checkout") for u in uses)
    download = next(
        s for s in publish if str(s.get("uses", "")).startswith("actions/download-artifact")
    )
    upload = next(
        s
        for s in steps_by_job(doc, "build")
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
    )
    assert download["with"]["name"] == upload["with"]["name"] == "dist"
    assert upload["with"]["if-no-files-found"] == "error"
    scripts = "\n".join(str(s.get("run", "")) for s in publish)
    assert "sha256sum" in scripts and "EXPECTED_DIGESTS" in scripts and "TAG" in scripts


def test_the_version_check_reads_the_source_without_importing_it() -> None:
    script = (ROOT / "scripts" / "print_version.py").read_text()
    assert "import cvpilot" not in script and "re." in script
