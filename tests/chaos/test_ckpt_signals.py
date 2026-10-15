"""Real signals, real processes: the handler only sets a flag, the next maybe_save saves, the run
is marked preempted and a new run resumes from the checkpoint with its lineage linked."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.hashing import new_run_id

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX signals"),
]
CHILD = Path(__file__).with_name("_ckpt_preempt.py")


def preempt_child(tmp_path: Path, signum: int, signals_sent: int = 1) -> tuple[str, str, int]:
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(CHILD), str(tmp_path), str(tmp_path / "store")],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    run_id, done = "", ""
    for line in proc.stdout:
        if line.startswith("RUN "):
            run_id = line.split()[1]
        elif line.startswith("READY"):
            for _ in range(signals_sent):
                os.kill(proc.pid, signum)
        elif line.startswith("DONE "):
            done = line.split()[1]
            break
    code = proc.wait(timeout=60)
    proc.stdout.close()
    return run_id, done, code


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGUSR1])
def test_a_signal_saves_at_the_next_step_and_marks_the_run_preempted(
    tmp_path: Path, signum: int
) -> None:
    run_id, done, code = preempt_child(tmp_path, signum)
    assert code == 0, "the handler must not kill or crash the process"
    assert done != "none"
    view = runs.get_run_view("chaos", run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "preempted"
    meta, repro = run_meta("chaos")
    child_meta = runs.resume_from(view.meta, run_id=new_run_id(), host="h", name=None)
    child_repro = repro.model_copy(update={"run_id": child_meta.run_id})
    child = runs.start_run(child_meta, child_repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    restored = manager.restore()
    assert restored is not None and restored.run_id == run_id
    assert restored.manifest.step == int(done)
    assert np.all(restored.state["model"]["w"] == float(done))
    assert child.meta.parent_run_id == run_id and child.status == "resumed"
    child.finish()
    del meta


def test_repeated_signals_do_not_crash_or_save_twice(tmp_path: Path) -> None:
    run_id, done, code = preempt_child(tmp_path, signal.SIGTERM, signals_sent=3)
    assert code == 0 and done != "none"
    view = runs.get_run_view("chaos", run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "preempted"
    folder = tmp_path / "ckpt" / run_id / "last"
    assert sorted(p.name for p in folder.iterdir() if p.name.endswith(".safetensors")) == [
        f"step_{int(done):09d}.safetensors"
    ]
