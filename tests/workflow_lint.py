"""Static checks for workflow and composite-action YAML (a parsed document in, problems out).

The rules follow the Phase 3 security checklist, item 10c: untrusted values (inputs, branch
names, titles) never appear inside shell script text, third-party actions are pinned to a full
commit SHA, permissions are minimal and every use of an environment variable in a script is
quoted.
"""

from __future__ import annotations

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


def lint(doc: dict[str, Any], *, composite: bool = False) -> list[str]:
    problems: list[str] = []
    triggers = doc.get("on", doc.get(True)) or {}
    if "pull_request_target" in (triggers if isinstance(triggers, (dict, list)) else [triggers]):
        problems.append("pull_request_target runs untrusted code with secrets")
    if not composite:
        permissions = doc.get("permissions")
        job_permissions = [j.get("permissions") for j in (doc.get("jobs") or {}).values()]
        if permissions is None and not all(job_permissions):
            problems.append("permissions must be declared (workflow level or on every job)")
        for block in [permissions, *job_permissions]:
            if block == "write-all" or (
                isinstance(block, dict) and any(v == "write" for v in block.values())
            ):
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
