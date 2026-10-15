"""kill -9 a real process inside a checkpoint save: restore must return a checkpoint that was
fully written and whose content is exactly what was saved, never anything else."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from cvpilot import runs
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.hashing import new_run_id

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX SIGKILL"),
]
WRITER = Path(__file__).with_name("_ckpt_writer.py")


def run_and_kill(
    tmp_path: Path, *, point: str = "none", min_acks: int = 3, delay: float = 0.0, mb: float = 4.0
) -> tuple[str, int, str | None]:
    """Start the writer, SIGKILL it, return ``(run_id, last acked step, stall point reached)``."""
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(WRITER), str(tmp_path), str(tmp_path / "store"), point, str(mb)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    run_id, acks, stalled = "", 0, None
    try:
        for line in proc.stdout:
            if line.startswith("RUN "):
                run_id = line.split()[1]
            elif line.startswith("ACK "):
                acks = int(line.split()[1])
                if point == "none" and acks >= min_acks:
                    time.sleep(delay)  # land the kill inside a later save
                    break
            elif line.startswith("STALL "):
                stalled = line.split()[1]
                break
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        proc.stdout.close()
    assert proc.returncode == -signal.SIGKILL
    return run_id, acks, stalled


def survey(tmp_path: Path, run_id: str) -> tuple[list[int], RunCheckpoints]:
    """Verified steps newest first; each loaded state must hold exactly what that step saved."""
    opened = runs.open_anchor("chaos", tmp_path / "store")
    anchor = opened[0] if opened else None
    found = RunCheckpoints(tmp_path / "ckpt", run_id, anchor=anchor)
    steps: list[int] = []
    for verified in found.iter_valid("last", load=True):
        assert verified.data is not None
        state = get_serializer("safetensors").loads(verified.data)
        weights = state["model"]["w"]
        assert weights.shape and np.all(weights == float(verified.manifest.step)), "wrong content"
        steps.append(verified.manifest.step)
    if opened:
        opened[1]()
    return steps, found


@pytest.mark.parametrize("delay", [0.0, 0.002, 0.007, 0.013, 0.021, 0.034])
def test_a_save_killed_at_a_random_moment_leaves_only_whole_checkpoints(
    tmp_path: Path, delay: float
) -> None:
    run_id, acked, _ = run_and_kill(tmp_path, delay=delay)
    steps, found = survey(tmp_path, run_id)
    assert steps, "at least the acknowledged checkpoints must restore"
    assert steps[0] >= acked  # everything acknowledged survives; at most one more completed
    assert steps == sorted(steps, reverse=True) and steps[-1] == 1
    assert len(set(steps)) == len(steps)
    for entry in found.inspect("last"):
        assert entry.status == "valid" or entry.reason in {
            "data_missing",
            "no_manifest",
            "size_mismatch",
            "manifest_unreadable",
            "unanchored",
        }, entry


@pytest.mark.parametrize(
    ("point", "newest", "skipped_reason"),
    [
        ("data_written", 2, None),  # data is a temp file, no manifest: invisible
        ("manifest_renamed", 2, "data_missing"),  # a manifest without its data: skipped
        ("before_record", 3, None),  # complete pair, pointer stale: found by the chained walk
        ("before_pointer", 3, None),  # complete pair, recorded in the store, pointer stale
    ],
)
def test_a_kill_at_each_step_of_the_save_restores_the_right_checkpoint(
    tmp_path: Path, point: str, newest: int, skipped_reason: str | None
) -> None:
    run_id, acked, stalled = run_and_kill(tmp_path, point=point)
    assert stalled == point and acked == 2
    steps, found = survey(tmp_path, run_id)
    assert steps[0] == newest
    assert steps == list(range(newest, 0, -1))
    reasons = {s.reason for s in found.skipped}
    assert reasons == ({skipped_reason} if skipped_reason else set())


def test_after_a_kill_the_next_run_resumes_cleans_the_debris_and_continues(
    tmp_path: Path,
) -> None:
    run_id, _, _ = run_and_kill(tmp_path, point="manifest_renamed")
    debris = [p.name for p in (tmp_path / "ckpt" / run_id / "last").iterdir()]
    assert any(n.endswith(".tmp") for n in debris)  # the half-written data file
    runs.recover_run("chaos", run_id, store_dir=tmp_path / "store")  # the dead run becomes failed
    from datetime import datetime, timezone

    from cvpilot.core.schema import Repro

    parent = runs.get_run_view("chaos", run_id, store_dir=tmp_path / "store").meta
    child_id = new_run_id()
    child_meta = runs.resume_from(parent, run_id=child_id, host="h", name=None)
    repro = Repro(
        run_id=child_id, git_dirty=False, packages_lock_sha256="0" * 64, python_version="3",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "0" * 64, config={},
        command_line="chaos",
    )  # fmt: skip
    assert datetime.now(timezone.utc) >= parent.started_at
    child = runs.start_run(child_meta, repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    restored = manager.restore()
    assert restored is not None
    assert restored.run_id == run_id and restored.manifest.step == 2
    assert [(s.step, s.reason) for s in restored.skipped] == [(3, "data_missing")]
    assert np.all(restored.state["model"]["w"] == 2.0)
    manager.save_now(3, model={"w": np.full(4, 3.0, dtype=np.float32)})
    own = sorted(p.name for p in (tmp_path / "ckpt" / child_id / "last").iterdir())
    assert own == ["step_000000003.manifest.json", "step_000000003.safetensors"]
    # the dead run's directory is untouched by the child: a later save in it would clean it
    parent_rc = RunCheckpoints(tmp_path / "ckpt", run_id, create=False)
    assert [e.status for e in parent_rc.inspect("last")][:2] == ["corrupt", "valid"]
    child.finish()
