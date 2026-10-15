from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import MetricStats, Repro, RunMeta
from cvpilot.repro import script
from cvpilot.runs.queries import RunView

SHA = "d" * 64


def view(tmp_path: Path, **repro_over: Any) -> RunView:
    rid = new_run_id()
    meta = RunMeta(
        run_id=rid,
        project="p",
        name="n",
        status="finished",
        task="detection",
        framework="python==3.12.1",
        host="h",
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        ended_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        schema_version=1,
    )
    base: dict[str, Any] = {
        "run_id": rid,
        "git_commit": "a" * 40,
        "git_dirty": False,
        "packages_lock_sha256": SHA,
        "python_version": "3.12.1",
        "gpu_names": ["A10G"],
        "cuda": "12.4 / 550.54",
        "dataset_uri": "/data/set",
        "dataset_hash": "merkle:" + SHA,
        "config": {},
        "seed": 9,
        "command_line": "train.py --epochs 2",
    }
    base.update(repro_over)
    return RunView(
        meta=meta,
        repro=Repro(**base),
        stats=MetricStats(n_points=0, n_files=0),
        artifacts=[],
        store_path=tmp_path,
    )


def bash_syntax_ok(text: str) -> None:
    subprocess.run(["bash", "-n"], input=text.encode(), check=True)  # noqa: S603, S607


def test_script_has_checkout_environment_dataset_and_command(tmp_path: Path) -> None:
    text = script(view(tmp_path))
    bash_syntax_ok(text)
    assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
    assert f"git checkout --detach {'a' * 40}" in text
    assert "3.12.1" in text and "12.4 / 550.54" in text and "A10G" in text
    assert "/data/set" in text and "merkle:" + SHA in text and "seed: 9" in text
    assert text.rstrip().endswith('"${PYTHON:-python3}" train.py --epochs 2')


def test_script_applies_the_stored_diff_when_dirty(tmp_path: Path) -> None:
    from cvpilot.core.schema import Artifact

    v = view(tmp_path, git_dirty=True, git_diff_sha256=SHA)
    art = Artifact(
        run_id=v.meta.run_id,
        sha256=SHA,
        kind="diff",
        uri=f"artifacts/dd/{SHA}",
        size_bytes=1,
        meta={},
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    v = RunView(v.meta, v.repro, v.stats, [art], tmp_path)
    text = script(v)
    bash_syntax_ok(text)
    assert f"git apply -- {tmp_path}/artifacts/dd/{SHA}" in text


def test_script_without_git_commit_says_so(tmp_path: Path) -> None:
    text = script(view(tmp_path, git_commit=None))
    bash_syntax_ok(text)
    assert "git checkout --detach" not in text and "no git commit was recorded" in text


def test_script_for_non_python_commands_runs_them_as_recorded(tmp_path: Path) -> None:
    text = script(view(tmp_path, command_line="yolo train data=x.yaml"))
    assert text.rstrip().endswith("yolo train data=x.yaml")


def test_script_refuses_to_run_when_the_command_line_was_not_stored(tmp_path: Path) -> None:
    text = script(view(tmp_path, command_line="train.py [3 arguments omitted]"))
    bash_syntax_ok(text)
    assert "was not stored" in text and "exit 1" in text
    assert '"${PYTHON' not in text
    assert text.rstrip().endswith("exit 1")


def test_untrusted_strings_cannot_break_out_of_comments(tmp_path: Path) -> None:
    evil = "x\nrm -rf / #"
    text = script(view(tmp_path, dataset_uri=evil, python_version="3.12\nTOUCH", cuda=evil))
    bash_syntax_ok(text)
    for line in text.splitlines():
        assert not line.startswith("rm -rf") and not line.startswith("TOUCH")


def test_hostile_commit_value_is_quoted(tmp_path: Path) -> None:
    text = script(view(tmp_path, git_commit="a" * 40))
    assert "git checkout --detach " + "a" * 40 in text
    v = view(tmp_path)
    bad = v.repro.model_copy(update={"git_commit": "x; touch /tmp/pwned"})
    text = script(RunView(v.meta, bad, v.stats, [], tmp_path))
    assert "'x; touch /tmp/pwned'" in text
