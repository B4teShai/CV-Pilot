"""The SkyPilot recipe (docs/guide/skypilot.md): the task file is well formed and keeps the key out
of it, and the training script survives a preemption the way a managed spot job needs: node A gets
SIGTERM, saves, uploads and exits 143; node B, with nothing but the bucket, resumes from node A's
checkpoint and finishes. SkyPilot itself is not run (the recipe is labelled experimental)."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples" / "skypilot" / "cvpilot-spot.yaml"
SCRIPT = ROOT / "examples" / "skypilot" / "train.py"
NODE = Path(__file__).with_name("_skypilot_node.py")
GUIDE = ROOT / "docs" / "guide" / "skypilot.md"
KEY = "56" * 32  # an obviously fake test key

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals")


def test_the_task_file_asks_for_spot_recovery_and_keeps_the_key_out_of_it() -> None:
    task = yaml.safe_load(RECIPE.read_text())
    assert task["resources"]["use_spot"] is True
    assert task["resources"]["job_recovery"]["max_restarts_on_errors"] >= 1
    assert task["secrets"] == {"CVPILOT_CKPT_KEY": None}  # the value comes from --secret
    envs = task["envs"]
    assert "CVPILOT_CKPT_KEY" not in envs
    assert envs["CVPILOT_CKPT_REQUIRE_TRUST"] == "authenticated"
    # a per-user directory, never a shared one such as /tmp: the manager refuses a root that
    # others can write, and a predictable name there would let another account plant a run
    assert envs["CVPILOT_CHECKPOINT_DIR"].startswith("~/")
    assert "/tmp" not in envs["CVPILOT_CHECKPOINT_DIR"]
    # the environment comes from the committed lock, not from whatever pip resolves today
    setup = task["setup"]
    assert "uv sync --locked" in setup and "pip install -e" not in setup
    assert "uv==" in setup  # the installer itself is pinned
    assert "--no-sync" in task["run"] or "--locked" in task["run"]
    assert envs["CVPILOT_CHECKPOINT_REMOTE"].startswith("s3://")
    assert not any(w in k.lower() for k in envs for w in ("token", "secret", "password"))
    assert "examples/skypilot/train.py" in task["run"] and SCRIPT.exists()
    script = SCRIPT.read_text()
    for name in envs:  # every setting the task file passes is one the script or cvpilot reads
        assert name.startswith("CVPILOT_") or name in script, name
    assert "EXPERIMENTAL" in RECIPE.read_text()


def test_the_guide_shows_the_task_file_as_it_is_and_says_it_is_experimental() -> None:
    text = GUIDE.read_text()
    assert RECIPE.read_text().split("name:", 1)[1] in text
    assert "Experimental" in text and "not been run on a real spot instance" in text


def node(tmp: Path, name: str, bucket: Path, **extra: str) -> dict[str, str]:
    home = tmp / name
    home.mkdir()
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(home),
        "GIT_CEILING_DIRECTORIES": str(tmp),
        "CVPILOT_PROJECT": "fridge-sku",
        "CVPILOT_STORE_DIR": str(home / "store"),
        "CVPILOT_CHECKPOINT_DIR": str(home / "ckpt"),
        "CVPILOT_CHECKPOINT_REMOTE": "s3://my-bucket/cvpilot/fridge-sku",
        "CVPILOT_CKPT_REQUIRE_TRUST": "authenticated",
        "CVPILOT_CKPT_KEY": KEY,
        "TOTAL_STEPS": "400",
        "SAVE_EVERY": "50",
        "PYTHONUNBUFFERED": "1",
    }
    env.update(extra)
    return env


def test_a_preempted_node_saves_and_a_fresh_node_resumes_from_the_bucket(tmp_path: Path) -> None:
    pytest.importorskip("fsspec")
    bucket = tmp_path / "bucket"
    a = subprocess.Popen(  # noqa: S603
        [sys.executable, str(NODE), str(bucket)],
        env=node(tmp_path, "node-a", bucket, STEP_SECONDS="0.01"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )  # fmt: skip
    assert a.stderr is not None and a.stdout is not None
    lines: list[str] = []
    try:
        for line in a.stderr:  # wait for two uploaded checkpoints, then preempt
            lines.append(line)
            if sum('"checkpoint_uploaded"' in x for x in lines) == 2:
                break
    finally:
        os.kill(a.pid, signal.SIGTERM)
    lines.extend(a.stderr)  # the rest, until the process exits
    out = a.stdout.read()
    a.wait(timeout=120)
    err = "".join(lines)
    assert a.returncode == 143, out + err
    events = [json.loads(x) for x in lines if x.startswith("{")]
    names = [e["message"] for e in events]
    assert "preemption_signal_received" in names and "run_preempted" in names, names
    stopped = int(out.strip().splitlines()[-1].split()[-1])  # "preempted after step N"
    a_run = next(e["run_id"] for e in events if e["message"] == "run_preempted")
    b = subprocess.run(  # noqa: S603
        [sys.executable, str(NODE), str(bucket)], env=node(tmp_path, "node-b", bucket),
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert b.returncode == 0, b.stdout + b.stderr
    lines = b.stdout.strip().splitlines()
    assert lines[0] == f"starting at step {stopped + 1} from a checkpoint of run {a_run}"
    assert lines[-1] == "finished at step 399"
    restored = [
        json.loads(x) for x in b.stderr.splitlines()
        if x.startswith("{") and json.loads(x).get("message") == "checkpoint_restored"
    ]  # fmt: skip
    assert restored and restored[0]["trust_level"] == "authenticated"
    runs_in_bucket = sorted(
        p.name for p in (bucket / "my-bucket" / "cvpilot" / "fridge-sku").iterdir()
    )
    assert len(runs_in_bucket) == 2  # node A's run and node B's run


def test_a_node_with_another_key_restores_nothing_from_the_bucket(tmp_path: Path) -> None:
    pytest.importorskip("fsspec")
    bucket = tmp_path / "bucket"
    first = subprocess.run(  # noqa: S603
        [sys.executable, str(NODE), str(bucket)],
        env=node(tmp_path, "node-a", bucket, TOTAL_STEPS="120"),
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert first.returncode == 0, first.stderr
    other = subprocess.run(  # noqa: S603
        [sys.executable, str(NODE), str(bucket)],
        env=node(tmp_path, "node-b", bucket, CVPILOT_CKPT_KEY="78" * 32, TOTAL_STEPS="120"),
        capture_output=True, text=True, timeout=300, check=False,
    )  # fmt: skip
    assert other.returncode != 0  # fails closed: it neither restores nor silently starts over
    assert "CheckpointUntrustedError" in other.stderr
