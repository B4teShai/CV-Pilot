"""The GitHub Action: static workflow rules and the real script run against hostile inputs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from conftest import CliResult
from support import make_run, open_project, write_eval_files
from workflow_lint import lint, lint_release, lint_trusted_gate, steps_of, unquoted_env_refs

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / "action.yml"


def load(path: Path) -> dict[str, Any]:
    doc = yaml.safe_load(path.read_text())
    assert isinstance(doc, dict)
    return doc


def gate_script() -> str:
    doc = load(ACTION)
    (step,) = [s for s in doc["runs"]["steps"] if s.get("id") == "gate"]
    return str(step["run"])


# -- the lint itself (it must catch what it claims to catch) ---------------------------------


def test_lint_rejects_an_expression_in_a_run_block() -> None:
    doc = {
        "runs": {
            "using": "composite",
            "steps": [{"shell": "bash", "run": 'echo "${{ inputs.x }}"'}],
        }
    }
    assert any("expression" in p for p in lint(doc, composite=True))
    doc = {
        "permissions": {"contents": "read"},
        "jobs": {"j": {"steps": [{"run": "echo ${{ github.head_ref }}"}]}},
    }
    assert any("expression" in p for p in lint(doc))


def test_lint_rejects_unpinned_third_party_actions_and_accepts_full_shas() -> None:
    bad = {"permissions": {}, "jobs": {"j": {"steps": [{"uses": "actions/checkout@v4"}]}}}
    assert any("pinned" in p for p in lint(bad))
    short = {"permissions": {}, "jobs": {"j": {"steps": [{"uses": "a/b@" + "a" * 39}]}}}
    assert any("pinned" in p for p in lint(short))
    good = {
        "permissions": {},
        "jobs": {"j": {"steps": [{"uses": "a/b@" + "a" * 40}, {"uses": "./"}]}},
    }
    assert lint(good) == []


def test_lint_rejects_missing_or_write_permissions_and_dangerous_triggers() -> None:
    assert any("permissions" in p for p in lint({"jobs": {"j": {"steps": []}}}))
    assert any("minimal" in p for p in lint({"permissions": "write-all", "jobs": {}}))
    assert any("minimal" in p for p in lint({"permissions": {"contents": "write"}, "jobs": {}}))
    assert any(
        "pull_request_target" in p
        for p in lint({"on": {"pull_request_target": {}}, "permissions": {}, "jobs": {}})
    )
    assert any(
        "pull_request_target" in p
        for p in lint(yaml.safe_load("on: pull_request_target\npermissions: {}\njobs: {}"))
    )


def test_lint_rejects_unquoted_variables_and_env_that_is_not_an_input() -> None:
    assert unquoted_env_refs("cvpilot gate $X") == ["$X"]
    assert unquoted_env_refs('cvpilot gate "$X" ${Y}') == ["${Y}"]
    assert unquoted_env_refs("echo '$X'") == []
    assert unquoted_env_refs('args+=("--a=$X")') == []
    doc = {
        "runs": {
            "steps": [{"shell": "bash", "run": "echo $X", "env": {"A": "${{ github.head_ref }}"}}]
        }
    }
    problems = lint(doc, composite=True)
    assert any("unquoted" in p for p in problems) and any(
        "other than an input" in p for p in problems
    )
    assert any("shell" in p for p in lint({"runs": {"steps": [{"run": "true"}]}}, composite=True))


# -- the shipped files -----------------------------------------------------------------------


def test_action_yml_passes_the_workflow_rules() -> None:
    doc = load(ACTION)
    assert doc["runs"]["using"] == "composite"
    assert lint(doc, composite=True) == []


def test_every_shipped_workflow_passes_the_workflow_rules() -> None:
    files = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
    assert files
    for path in files:
        if path.name == "release.yml":  # the one workflow that may hold an OIDC token
            assert lint_release(load(path), path.read_text()) == [], path.name
        else:
            assert lint(load(path)) == [], path.name


def test_no_run_block_anywhere_contains_an_expression() -> None:
    docs = [
        load(ACTION),
        *(load(p) for p in sorted((ROOT / ".github" / "workflows").glob("*.yml"))),
    ]
    for doc in docs:
        for step in steps_of(doc):
            assert "${{" not in str(step.get("run", "")), step


def guide_workflows() -> list[dict[str, Any]]:
    """The full workflow examples of the CI guide (the page that documents the Action)."""
    page = (ROOT / "docs" / "guide" / "gating-ci.md").read_text()
    return [yaml.safe_load(b) for b in re.findall(r"```yaml\n(.*?)```", page, re.S)]


def test_the_readme_stays_free_of_workflow_examples() -> None:
    assert "```yaml" not in (ROOT / "README.md").read_text()


def test_the_guide_example_workflow_follows_the_trust_model() -> None:
    docs = guide_workflows()
    assert docs, "the CI guide documents the Action with a workflow example"
    for doc in docs:
        assert lint_trusted_gate(doc) == []


def good_example() -> dict[str, Any]:
    return yaml.safe_load(
        """
on: pull_request
permissions:
  contents: read
jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          persist-credentials: false
      - run: python -m pip install --require-hashes -r ci/requirements.txt
      - uses: ./
        with:
          run-id: abc
          baseline: tag:prod
          recompute: "true"
          gt: ci/gt.parquet
          classes: ci/classes.txt
          expect-baseline-chain: ${{ vars.CHAIN }}
          require-commit: ${{ github.sha }}
"""
    )


def test_the_trusted_gate_lint_accepts_a_good_workflow_and_rejects_each_unsafe_pattern() -> None:
    assert lint_trusted_gate(good_example()) == []

    def broken(edit: Any) -> list[str]:
        doc = good_example()
        edit(doc)
        return lint_trusted_gate(doc)

    steps = lambda d: d["jobs"]["gate"]["steps"]  # noqa: E731
    cases: dict[str, Any] = {
        "pull_request_target": lambda d: d.update({"on": "pull_request_target"}),
        "head checkout": lambda d: steps(d)[0]["with"].update(
            {"ref": "${{ github.event.pull_request.head.sha }}"}
        ),
        "default checkout ref": lambda d: steps(d)[0]["with"].pop("ref"),
        "credentials kept": lambda d: steps(d)[0]["with"].pop("persist-credentials"),
        "unhashed install": lambda d: steps(d)[1].update({"run": "pip install cvpilot"}),
        "write permission": lambda d: d.update({"permissions": {"contents": "write"}}),
        "extra permission": lambda d: d.update(
            {"permissions": {"contents": "read", "id-token": "read"}}
        ),
        "no recompute": lambda d: steps(d)[2]["with"].pop("recompute"),
        "no anchor": lambda d: steps(d)[2]["with"].pop("expect-baseline-chain"),
        "no commit": lambda d: steps(d)[2]["with"].pop("require-commit"),
        "run baseline": lambda d: steps(d)[2]["with"].update(
            {"baseline": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
        ),
        "unpinned action": lambda d: steps(d).append({"uses": "actions/setup-python@v5"}),
        "expression in run": lambda d: steps(d).append({"run": "echo ${{ github.head_ref }}"}),
    }
    for name, edit in cases.items():
        assert broken(edit), name


def test_inputs_reach_the_script_only_through_env() -> None:
    doc = load(ACTION)
    (step,) = [s for s in doc["runs"]["steps"] if s.get("id") == "gate"]
    env_inputs = {m for v in step["env"].values() for m in re.findall(r"inputs\.([a-z-]+)", str(v))}
    assert env_inputs == set(doc["inputs"]), "every input is wired through env: and nothing else is"


def test_required_inputs_and_safe_defaults() -> None:
    inputs = load(ACTION)["inputs"]
    assert inputs["run-id"]["required"] is True and inputs["baseline"]["required"] is True
    assert inputs["min-images"]["default"] == "30"
    assert inputs["skip-small-slices"]["default"] == "false"
    for name, spec in inputs.items():
        assert spec.get("description"), name
        if name not in ("run-id", "baseline"):
            assert spec.get("required") in (None, False)


def test_the_action_has_no_third_party_steps() -> None:
    uses = [s["uses"] for s in load(ACTION)["runs"]["steps"] if "uses" in s]
    assert uses == []  # the caller sets up Python and installs cvpilot; nothing to pin


# -- running the script ----------------------------------------------------------------------


@pytest.fixture
def fake_cli(tmp_path: Path) -> Path:
    """A ``cvpilot`` that records its argv one entry per line and exits with $FAKE_EXIT."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "cvpilot"
    exe.write_text(
        "#!/bin/bash\n"
        'for a in "$@"; do printf "%s\\n" "$a" >> "$FAKE_ARGV"; done\n'
        '[ -n "${FAKE_STDOUT:-}" ] && printf "%s\\n" "$FAKE_STDOUT"\n'
        'exit "${FAKE_EXIT:-0}"\n'
    )
    exe.chmod(0o755)
    return bin_dir


BASE_ENV = {
    "GATE_RUN_ID": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "GATE_BASELINE": "tag:prod",
    "GATE_PROJECT": "",
    "GATE_STORE_DIR": "",
    "GATE_MAX_DROP": "",
    "GATE_PER_CLASS_MAX_DROP": "",
    "GATE_SLICE_MAX_DROP": "",
    "GATE_MIN_IMAGES": "30",
    "GATE_SKIP_SMALL_SLICES": "false",
    "GATE_MAX_SKIPPED": "0",
    "GATE_ALLOW_RUN_BASELINE": "false",
    "GATE_EXPECT_CHAIN": "",
    "GATE_REQUIRE_COMMIT": "",
    "GATE_RECOMPUTE": "false",
    "GATE_GT": "",
    "GATE_CLASSES": "",
    "GATE_EVAL_SET": "",
    "GATE_REPORT": "",
    "GATE_REPORT_JSON": "",
}


def run_script(
    tmp_path: Path, path_dirs: list[Path], overrides: dict[str, str], exit_code: int = 0
) -> tuple[subprocess.CompletedProcess[str], list[str], str]:
    argv_file, output = tmp_path / "argv.txt", tmp_path / "output.txt"
    env = {
        "PATH": os.pathsep.join([*map(str, path_dirs), "/usr/bin", "/bin"]),
        "HOME": str(tmp_path),
        "FAKE_ARGV": str(argv_file),
        "FAKE_EXIT": str(exit_code),
        "GITHUB_OUTPUT": str(output),
        **BASE_ENV,
        **overrides,
    }
    done = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", gate_script()],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    argv = argv_file.read_text().splitlines() if argv_file.exists() else []
    return done, argv, output.read_text() if output.exists() else ""


def test_script_builds_the_command_from_the_inputs(tmp_path: Path, fake_cli: Path) -> None:
    done, argv, output = run_script(
        tmp_path,
        [fake_cli],
        {
            "GATE_PROJECT": "demo",
            "GATE_MAX_DROP": "mAP50_95=0.5\nmAP50=1.0\n",
            "GATE_PER_CLASS_MAX_DROP": "AP50_95=2.0",
            "GATE_SLICE_MAX_DROP": "camera=*:1.5",
            "GATE_MIN_IMAGES": "50",
            "GATE_SKIP_SMALL_SLICES": "true",
            "GATE_EVAL_SET": "abcdef12",
            "GATE_REPORT": "out/cmp.html",
        },
    )
    assert done.returncode == 0 and output == "exit_code=0\n"
    assert argv == [
        "gate", "--baseline=tag:prod", "--project=demo", "--eval-set=abcdef12",
        "--max-drop=mAP50_95=0.5", "--max-drop=mAP50=1.0", "--per-class-max-drop=AP50_95=2.0",
        "--slice-max-drop=camera=*:1.5", "--min-images=50", "--max-skipped=0",
        "--skip-small-slices",
        "--report=out/cmp.html", "--", "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    ]  # fmt: skip


HOSTILE = [
    "$(touch pwned)",
    '"; touch pwned #',
    "`touch pwned`",
    'x"; touch pwned; echo "',
    "a b\tc",
    "--max-drop=1",
    "-x",
    "'; touch pwned; '",
    "${IFS}touch${IFS}pwned",
    "*",
    "\\",
]


@pytest.mark.parametrize("value", HOSTILE)
@pytest.mark.parametrize(
    "variable",
    [
        "GATE_RUN_ID",
        "GATE_BASELINE",
        "GATE_PROJECT",
        "GATE_MAX_DROP",
        "GATE_SLICE_MAX_DROP",
        "GATE_REPORT",
        "GATE_EVAL_SET",
        "GATE_MIN_IMAGES",
    ],
)
def test_hostile_input_values_never_execute_and_arrive_as_single_arguments(
    tmp_path: Path, fake_cli: Path, variable: str, value: str
) -> None:
    if variable == "GATE_BASELINE":
        value = "tag:" + value
    done, argv, _ = run_script(tmp_path, [fake_cli], {variable: value})
    assert not (tmp_path / "pwned").exists(), done.stderr
    assert done.returncode == 0
    needle = value.rstrip("\n")
    assert any(needle in a for a in argv if "\n" not in a) or "\n" in value
    assert argv[0] == "gate" and argv[-2] == "--"
    leading = [a for a in argv[:-2] if a.startswith("-")]  # after "--" nothing is an option
    for a in leading:  # every option-looking argument is one the script itself added
        assert re.match(
            r"--(baseline|project|eval-set|max-drop|per-class-max-drop|slice-max-drop|min-images|max-skipped|skip-small-slices|require-commit|expect-baseline-chain|gt|classes|recompute|report|report-json)(=|$)",
            a,
        ), a


def test_the_exit_code_of_cvpilot_is_the_exit_code_of_the_step(
    tmp_path: Path, fake_cli: Path
) -> None:
    for code in (0, 1, 2, 130):
        done, _, output = run_script(tmp_path, [fake_cli], {}, exit_code=code)
        assert done.returncode == code and output.splitlines()[-1] == f"exit_code={code}"


def test_a_missing_cvpilot_fails_the_step_instead_of_passing(tmp_path: Path) -> None:
    done, argv, output = run_script(tmp_path, [], {})
    assert done.returncode != 0 and argv == [] and "not installed" in done.stderr
    assert output == "exit_code=2\n"


def test_an_unknown_skip_small_slices_value_fails_closed(tmp_path: Path, fake_cli: Path) -> None:
    done, argv, _ = run_script(tmp_path, [fake_cli], {"GATE_SKIP_SMALL_SLICES": "maybe"})
    assert done.returncode != 0 and argv == []


def test_the_script_runs_without_github_output(tmp_path: Path, fake_cli: Path) -> None:
    argv_file = tmp_path / "argv.txt"
    env = {"PATH": f"{fake_cli}:/usr/bin:/bin", "FAKE_ARGV": str(argv_file), **BASE_ENV}
    done = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", gate_script()],
        env=env,
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )  # noqa: S603
    assert done.returncode == 0


# -- the script against the real CLI -----------------------------------------------------------


@pytest.fixture
def real_cli_dir() -> Path:
    exe = Path(sys.executable).parent / "cvpilot"
    assert exe.exists()
    return exe.parent


def test_the_script_gates_a_real_store(
    tmp_path: Path, store_root: Path, real_cli_dir: Path, invoke: Callable[..., CliResult]
) -> None:
    def evaluate(folder: str, **kw: Any) -> str:
        store = open_project(store_root)
        try:
            run = make_run(store)
        finally:
            store.close()
        preds, gts, classes = write_eval_files(tmp_path / folder, **kw)
        res = invoke(
            "eval",
            run,
            "--project",
            "proj",
            "--preds",
            str(preds),
            "--gts",
            str(gts),
            "--classes",
            str(classes),
            "--allow-closed",
            "--resamples",
            "0",
        )
        assert res.code == 0, res.err
        return run

    base, good, bad = evaluate("b"), evaluate("g"), evaluate("x", broken=12)
    assert invoke("baseline", "set", "prod", base, "--project", "proj").code == 0
    overrides = {
        "GATE_PROJECT": "proj",
        "GATE_STORE_DIR": str(store_root),
        "GATE_MAX_DROP": "mAP50_95=0.5",
        "GATE_MIN_IMAGES": "30",
    }
    ok, _, out = run_script(tmp_path, [real_cli_dir], {**overrides, "GATE_RUN_ID": good})
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert out == "exit_code=0\n" and "gate: PASS" in ok.stdout
    regressed, _, out = run_script(tmp_path, [real_cli_dir], {**overrides, "GATE_RUN_ID": bad})
    assert (
        regressed.returncode == 1
        and out.endswith("exit_code=1\n")
        and "gate: FAIL" in regressed.stdout
    )
    assert os.environ.get("CVPILOT_STORE_DIR") == str(
        store_root
    )  # the test's own setting, untouched


# -- trust options -----------------------------------------------------------------------------


def test_a_baseline_run_id_is_refused_unless_explicitly_allowed(
    tmp_path: Path, fake_cli: Path
) -> None:
    run_baseline = {"GATE_BASELINE": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}
    done, argv, out = run_script(tmp_path, [fake_cli], run_baseline)
    assert done.returncode == 2 and argv == [] and out == "exit_code=2\n"
    assert "tag:NAME" in done.stderr
    done, argv, _ = run_script(
        tmp_path, [fake_cli], {**run_baseline, "GATE_ALLOW_RUN_BASELINE": "true"}
    )
    assert done.returncode == 0 and argv[1] == "--baseline=01ARZ3NDEKTSV4RRFFQ69G5FAV"


def test_trust_inputs_are_passed_as_single_options(tmp_path: Path, fake_cli: Path) -> None:
    chain = "c" * 64
    done, argv, _ = run_script(
        tmp_path,
        [fake_cli],
        {
            "GATE_REQUIRE_COMMIT": "d" * 40,
            "GATE_EXPECT_CHAIN": chain,
            "GATE_RECOMPUTE": "true",
            "GATE_GT": "trusted/gt.parquet",
            "GATE_CLASSES": "trusted/c.txt",
            "GATE_MAX_SKIPPED": "2",
        },
    )
    assert done.returncode == 0
    for expected in (
        "--require-commit=" + "d" * 40,
        "--expect-baseline-chain=" + chain,
        "--recompute",
        "--gt=trusted/gt.parquet",
        "--classes=trusted/c.txt",
        "--max-skipped=2",
    ):
        assert expected in argv


@pytest.mark.parametrize(
    "overrides",
    [
        {"GATE_RECOMPUTE": "true"},
        {"GATE_RECOMPUTE": "true", "GATE_GT": "g.parquet"},
        {"GATE_GT": "g.parquet", "GATE_CLASSES": "c.txt"},
        {"GATE_RECOMPUTE": "yes"},
        {"GATE_ALLOW_RUN_BASELINE": "1"},
    ],
)
def test_inconsistent_trust_inputs_fail_closed(
    tmp_path: Path, fake_cli: Path, overrides: dict[str, str]
) -> None:
    done, argv, out = run_script(tmp_path, [fake_cli], overrides)
    assert done.returncode == 2 and argv == [] and out == "exit_code=2\n"


def test_the_step_summary_holds_the_verdict_in_a_fence_without_backticks(
    tmp_path: Path, fake_cli: Path
) -> None:
    summary = tmp_path / "summary.md"
    hostile = "gate: FAIL (1 violations)\n  skipped slice: `evil` ```\n# injected heading"
    env = {
        "PATH": f"{fake_cli}:/usr/bin:/bin",
        "FAKE_ARGV": str(tmp_path / "argv.txt"),
        "FAKE_STDOUT": hostile,
        "FAKE_EXIT": "1",
        "GITHUB_OUTPUT": str(tmp_path / "out.txt"),
        "GITHUB_STEP_SUMMARY": str(summary),
        **BASE_ENV,
    }
    done = subprocess.run(  # noqa: S603
        ["/bin/bash", "-c", gate_script()],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    text = summary.read_text()
    assert done.returncode == 1 and done.stdout.startswith("gate: FAIL")
    assert text.startswith("### cvpilot gate\n```text\n") and text.endswith("```\n")
    assert text.count("```") == 2 and "skipped slice: 'evil'" in text
