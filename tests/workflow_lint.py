"""Static checks for workflow and composite-action YAML (a parsed document in, problems out).

The rules follow the Phase 3 security checklist, item 10c: untrusted values (inputs, branch
names, titles) never appear inside shell script text, third-party actions are pinned to a full
commit SHA, permissions are minimal and every use of an environment variable in a script is
quoted.
"""

from __future__ import annotations

import json
import re
from typing import Any

FULL_SHA = re.compile(r"[0-9a-f]{40}")
EXPRESSION = re.compile(r"\$\{\{.*?\}\}", re.S)
ENV_REF = re.compile(
    r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)[^}]*\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))"
)
SAFE_ENV_EXPRESSION = re.compile(r"\$\{\{\s*inputs\.[A-Za-z0-9_-]+\s*\}\}")


def steps_of(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every step of a workflow (all jobs) or of a composite action."""
    found: list[dict[str, Any]] = []
    for job in (doc.get("jobs") or {}).values():
        found.extend(job.get("steps") or [])
    found.extend((doc.get("runs") or {}).get("steps") or [])
    return found


def unquoted_env_refs(script: str) -> list[str]:
    """Variable references in ``script`` that are not inside double quotes (single quotes do not
    expand variables, so they are not an injection risk and are ignored)."""
    problems: list[str] = []
    for line in script.splitlines():
        code = line.split("#", 1)[0] if not re.search(r"""["'].*#.*["']""", line) else line
        for match in ENV_REF.finditer(code):
            before = code[: match.start()]
            in_double = before.count('"') % 2 == 1
            in_single = before.count("'") % 2 == 1 and not in_double
            if not in_double and not in_single:
                problems.append(match.group(0))
    return problems


def _writes(block: Any) -> list[str]:
    if block == "write-all":
        return ["write-all"]
    if isinstance(block, dict):
        return [name for name, value in block.items() if value == "write"]
    return []


def lint(
    doc: dict[str, Any],
    *,
    composite: bool = False,
    allow_oidc: frozenset[str] | set[str] = frozenset(),
) -> list[str]:
    """``allow_oidc`` names the jobs that may hold ``id-token: write`` (the trusted-publishing job
    of a release workflow); any other write permission, anywhere, is still refused."""
    problems: list[str] = []
    triggers = doc.get("on", doc.get(True)) or {}
    if "pull_request_target" in (triggers if isinstance(triggers, (dict, list)) else [triggers]):
        problems.append("pull_request_target runs untrusted code with secrets")
    if not composite:
        permissions = doc.get("permissions")
        jobs = doc.get("jobs") or {}
        job_permissions = [j.get("permissions") for j in jobs.values()]
        if permissions is None and not all(job_permissions):
            problems.append("permissions must be declared (workflow level or on every job)")
        if _writes(permissions):
            problems.append("write permissions are not minimal")
        for job_name, job in jobs.items():
            writes = _writes(job.get("permissions"))
            if job_name in allow_oidc:
                writes = [w for w in writes if w != "id-token"]
            if writes:
                problems.append("write permissions are not minimal")
    for index, step in enumerate(steps_of(doc)):
        label = step.get("name") or step.get("id") or f"step {index}"
        script = step.get("run")
        if script is not None:
            if EXPRESSION.search(script):
                problems.append(f"{label}: a ${{{{ }}}} expression inside a run script")
            if composite and "shell" not in step:
                problems.append(f"{label}: composite run steps need a shell")
            for ref in unquoted_env_refs(script):
                problems.append(f"{label}: unquoted variable {ref}")
        uses = step.get("uses")
        if (
            isinstance(uses, str)
            and uses.startswith("actions/checkout@")
            and (step.get("with") or {}).get("persist-credentials") is not False
        ):
            problems.append(f"{label}: checkout must set persist-credentials: false")
        if isinstance(uses, str) and not uses.startswith("./"):
            _, _, ref = uses.partition("@")
            if not FULL_SHA.fullmatch(ref):
                problems.append(f"{label}: {uses} is not pinned to a full commit SHA")
        if composite:
            for name, value in (step.get("env") or {}).items():
                if isinstance(value, str) and "${{" in value:
                    if not SAFE_ENV_EXPRESSION.fullmatch(value.strip()):
                        problems.append(f"{label}: env {name} holds something other than an input")
        if isinstance(uses, str) and uses.startswith("actions/github-script"):
            if EXPRESSION.search(str((step.get("with") or {}).get("script", ""))):
                problems.append(f"{label}: an expression inside a github-script script")
    return problems


def lint_trusted_gate(doc: dict[str, Any]) -> list[str]:
    """Extra rules for a workflow that gates a pull request (the documented CI example).

    The gate is a trust boundary only if the code that runs, the baseline it compares with and
    the settings it uses cannot be changed by the pull request being gated.
    """
    problems = lint(doc)
    triggers = doc.get("on", doc.get(True)) or {}
    names = set(triggers) if isinstance(triggers, (dict, list)) else {triggers}
    if "pull_request" not in names:
        problems.append("gate on pull_request, the only trigger that has no secrets for forks")
    if doc.get("permissions") != {"contents": "read"}:
        problems.append("permissions must be exactly contents: read")
    steps = steps_of(doc)
    checkouts = [s for s in steps if str(s.get("uses", "")).startswith("actions/checkout@")]
    if not checkouts:
        problems.append("check out the trusted base explicitly")
    for step in checkouts:
        with_ = step.get("with") or {}
        ref = str(with_.get("ref", ""))
        if "pull_request.base" not in ref or "head" in ref:
            problems.append("checkout must use the pull request's base commit, never its head")
        if with_.get("persist-credentials") is not False:
            problems.append("checkout must set persist-credentials: false")
    for step in steps:
        script = str(step.get("run", ""))
        if "pip install" in script and "--require-hashes" not in script:
            problems.append("install cvpilot from a pinned release with --require-hashes")
        if re.search(r"pip install[^\n]*\bcvpilot\b", script) and "-r " not in script:
            problems.append("do not install cvpilot by bare name")
    if not any("--require-hashes" in str(s.get("run", "")) for s in steps):
        problems.append("the example must install cvpilot with --require-hashes")
    gates = [s for s in steps if s.get("uses") == "./"]
    if len(gates) != 1:
        problems.append("use the gate action from the trusted checkout exactly once")
    for step in gates:
        with_ = step.get("with") or {}
        if not str(with_.get("baseline", "")).startswith("tag:"):
            problems.append("baseline must be a tag")
        if str(with_.get("recompute")) != "true":
            problems.append("the gate step must set recompute: true")
        for required in ("gt", "classes", "expect-baseline-chain", "require-commit"):
            if not with_.get(required):
                problems.append(f"the gate step must set {required}")
        if "allow-run-baseline" in with_ and str(with_["allow-run-baseline"]) == "true":
            problems.append("a baseline run id is not an anchor")
    return problems


_RELEASE_TAG = re.compile(r"v\[0-9\]\+\.\[0-9\]\+\.\[0-9\]\+")
_VERSION_COMMENT = re.compile(r"uses: [\w./-]+@[0-9a-f]{40} # v\d+\.\d+\.\d+$")


def _job_steps(doc: dict[str, Any], name: str) -> list[dict[str, Any]]:
    return list((doc.get("jobs") or {}).get(name, {}).get("steps") or [])


def _uses(steps: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [s for s in steps if str(s.get("uses", "")).startswith(prefix)]


def lint_release(doc: dict[str, Any], text: str) -> list[str]:
    """Rules for a workflow that publishes to PyPI with trusted publishing (Phase 4, item 1).

    One build job that can read the repository and nothing else, one publish job that can only
    mint an OIDC token, runs no repository code, sits in a protected environment and publishes
    the exact artifact the build job uploaded, on a version tag and nothing else.
    """
    problems = lint(doc, allow_oidc={"publish"})
    triggers = doc.get("on", doc.get(True))
    if not isinstance(triggers, dict) or set(triggers) != {"push"}:
        problems.append("trigger only on the push of version tags (no other trigger)")
    else:
        push = triggers.get("push") or {}
        tags = push.get("tags") if isinstance(push, dict) else None
        if not isinstance(push, dict) or set(push) != {"tags"} or not tags:
            problems.append("push must filter on tags only (no branches or paths)")
        elif not all(isinstance(t, str) and _RELEASE_TAG.fullmatch(t) for t in tags):
            problems.append("the tags filter must be exactly v[0-9]+.[0-9]+.[0-9]+")
    if doc.get("permissions") != {}:
        problems.append("top-level permissions must be {} (nothing by default)")
    jobs = doc.get("jobs") or {}
    if set(jobs) != {"build", "publish"}:
        problems.append("the workflow must have exactly the jobs build and publish")
        return problems
    build, publish = jobs["build"], jobs["publish"]
    if build.get("permissions") != {"contents": "read"}:
        problems.append("build: permissions must be exactly contents: read")
    if "environment" in build or "id-token" in str(build.get("permissions")):
        problems.append("build: no environment and no id-token (it holds no publish power)")
    if publish.get("permissions") != {"id-token": "write"}:
        problems.append("publish: permissions must be exactly id-token: write")
    needs = publish.get("needs")
    if needs not in ("build", ["build"]):
        problems.append("publish: needs must be build")
    environment = publish.get("environment")
    if not (isinstance(environment, str) and environment) and not (
        isinstance(environment, dict) and environment.get("name")
    ):
        problems.append("publish: a protected environment is required")
    build_steps, publish_steps = _job_steps(doc, "build"), _job_steps(doc, "publish")
    if _uses(publish_steps, "actions/checkout"):
        problems.append("publish: must not check out repository code")
    if _uses(build_steps, "pypa/gh-action-pypi-publish"):
        problems.append("publish action only in the publish job")
    publishers = _uses(publish_steps, "pypa/gh-action-pypi-publish")
    if len(publishers) != 1:
        problems.append("publish: exactly one trusted-publishing step")
    for step in publishers:
        with_ = step.get("with") or {}
        if any(k in with_ for k in ("password", "user", "username", "repository-url")):
            problems.append("publish: no static token or credential input (OIDC only)")
    blob = text + "\n" + json.dumps(doc, default=str)
    if re.search(r"secrets\.", blob) or re.search(r"(?i)pypi[_-]?(api[_-]?)?token", blob):
        problems.append("no secret references: trusted publishing needs none")
    upload = _uses(build_steps, "actions/upload-artifact")
    download = _uses(publish_steps, "actions/download-artifact")
    if len(upload) != 1 or len(download) != 1:
        problems.append("one upload-artifact in build and one download-artifact in publish")
    elif (upload[0].get("with") or {}).get("name") != (download[0].get("with") or {}).get("name"):
        problems.append("artifact names must match between build and publish")
    build_scripts = "\n".join(str(s.get("run", "")) for s in build_steps)
    publish_scripts = "\n".join(str(s.get("run", "")) for s in publish_steps)
    if "print_version" not in build_scripts or "TAG" not in build_scripts:
        problems.append("build: the tag must be checked against the package version")
    if "sha256sum" not in build_scripts or "sha256sum" not in publish_scripts:
        problems.append("artifact digests must be logged and re-checked (sha256sum)")
    if "workflow_dispatch" in text or "pull_request" in text.replace("pull_request_target", ""):
        problems.append("trigger only on version tags")
    for line in text.splitlines():
        if "uses:" in line and not _VERSION_COMMENT.search(line.strip().lstrip("- ")):
            problems.append(f"pin with a full SHA and a version comment: {line.strip()}")
    return problems
