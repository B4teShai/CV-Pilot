"""A restarted job on a node that has neither the run store nor the checkpoint directory of the
earlier run finds its checkpoints on the remote (across real processes), through the gate."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds"),
]
CHILD = Path(__file__).with_name("_ckpt_fresh_node.py")
pytest.importorskip("fsspec")


def child(mode: str, bucket: Path, node: Path, **env: str) -> subprocess.CompletedProcess[str]:
    full = {k: v for k, v in os.environ.items() if not k.startswith("CVPILOT_")}
    full.update(env)
    return subprocess.run(  # noqa: S603
        [sys.executable, str(CHILD), mode, str(bucket), str(node)],
        capture_output=True, text=True, env=full, timeout=120, check=False,
    )  # fmt: skip


def test_a_fresh_node_finds_the_previous_runs_checkpoints(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    wrote = child("write", bucket, tmp_path / "node-a")
    assert "WROTE 3" in wrote.stdout, wrote.stdout + wrote.stderr
    fresh = child("restore", bucket, tmp_path / "node-b")  # no store, no checkpoint directory
    assert "RESTORED 3 [3.0, 3.0]" in fresh.stdout, fresh.stdout + fresh.stderr


def test_the_same_with_a_key_set_on_both_nodes(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    key = {"CVPILOT_CKPT_KEY": "56" * 32}
    assert "WROTE 3" in child("write", bucket, tmp_path / "node-a", **key).stdout
    fresh = child("restore", bucket, tmp_path / "node-b", **key)
    assert "RESTORED 3 [3.0, 3.0]" in fresh.stdout, fresh.stdout + fresh.stderr


def test_a_key_on_the_fresh_node_refuses_checkpoints_written_without_one(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    assert "WROTE 3" in child("write", bucket, tmp_path / "node-a").stdout
    fresh = child("restore", bucket, tmp_path / "node-b", CVPILOT_CKPT_KEY="56" * 32)
    assert "RESTORED" not in fresh.stdout
    assert fresh.returncode != 0 and "CheckpointUntrustedError" in fresh.stderr
