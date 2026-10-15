"""Golden: the script printed by `cvpilot repro` re-creates the run's code state and command."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cvpilot.runs import get_run_view, list_runs

TRAIN = """\
import json, os, pathlib
import cvpilot

value = float(pathlib.Path("value.txt").read_text())
run = cvpilot.init("fx", task="detection", config={"value": value}, seed=1)
for step in range(5):
    run.log({"train/loss": value / (step + 1)}, step=step)
run.finish()
"""


def sh(*cmd: str, cwd: Path, env: dict[str, str]) -> str:
    done = subprocess.run(  # noqa: S603
        list(cmd), cwd=cwd, env=env, check=True, capture_output=True, text=True
    )
    return done.stdout


@pytest.fixture
def fixture_repo(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {
        **os.environ,
        "CVPILOT_STORE_DIR": str(tmp_path / "store"),
        "CVPILOT_ENV": "prod",
        "GIT_CEILING_DIRECTORIES": str(tmp_path),
        "PYTHON": sys.executable,
    }
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    sh(*git, "init", "-q", cwd=repo, env=env)
    (repo / "train.py").write_text(TRAIN)
    (repo / "value.txt").write_text("1.0\n")
    sh(*git, "add", ".", cwd=repo, env=env)
    sh(*git, "commit", "-q", "-m", "fixture", cwd=repo, env=env)
    return repo, env


def test_repro_script_recreates_a_dirty_run(fixture_repo: tuple[Path, dict[str, str]]) -> None:
    repo, env = fixture_repo
    store = Path(env["CVPILOT_STORE_DIR"])
    (repo / "value.txt").write_text("4.0\n")  # uncommitted change that affects the result

    sh(sys.executable, "train.py", cwd=repo, env=env)
    (first,) = list_runs("fx", store_dir=store)
    v1 = get_run_view("fx", first.run_id, store_dir=store)
    assert v1.repro.git_dirty and v1.repro.git_diff_sha256 and v1.repro.command_line == "train.py"

    cli = "import sys; from cvpilot.cli.app import main; main()"
    script = sh(
        sys.executable, "-c", cli, "repro", first.run_id, "--project", "fx", cwd=repo, env=env
    )
    (repo / "reproduce.sh").write_text(script)

    # lose the uncommitted change, then let the script bring it back and re-run the command
    sh("git", "checkout", "--", "value.txt", cwd=repo, env=env)
    assert (repo / "value.txt").read_text() == "1.0\n"
    sh("bash", "reproduce.sh", cwd=repo, env=env)
    assert (repo / "value.txt").read_text() == "4.0\n"

    second, _ = sorted(list_runs("fx", store_dir=store), key=lambda r: r.started_at, reverse=True)
    v2 = get_run_view("fx", second.run_id, store_dir=store)
    assert v2.repro.git_commit == v1.repro.git_commit
    assert v2.repro.git_diff_sha256 == v1.repro.git_diff_sha256
    assert v2.repro.command_line == v1.repro.command_line
    assert v2.repro.config == v1.repro.config == {"value": 4.0}

    out1 = json.loads(
        sh(
            sys.executable,
            "-c",
            cli,
            "runs",
            "show",
            first.run_id,
            "--project",
            "fx",
            "--json",
            cwd=repo,
            env=env,
        )
    )
    assert out1["metrics"]["n_points"] == 5
