"""Child process for the checkpoint chaos tests: saves checkpoints forever, acking each one.

``argv``: checkpoint root, store root, stall point (``none`` or one of ``data_written``,
``manifest_renamed``, ``before_record``, ``before_pointer``), megabytes per checkpoint. At the
stall point of the third save the child announces ``STALL <point>`` and sleeps until the parent
kills it, so the kill lands at a chosen place inside the save instead of by luck.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cvpilot import runs
from cvpilot.ckpt import fsops
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import Repro, RunMeta

STALL_STEP = 3


def stall(point: str) -> None:
    print(f"STALL {point}", flush=True)
    time.sleep(3600)


def main(root: str, store: str, point: str, megabytes: float) -> None:
    run_id = new_run_id()
    meta = RunMeta(
        run_id=run_id,
        project="chaos",
        name="chaos",
        status="running",
        task="detection",
        framework="python==3.12.1",
        host="h",
        started_at=datetime.now(timezone.utc),
        schema_version=1,
    )
    repro = Repro(
        run_id=run_id,
        git_dirty=False,
        packages_lock_sha256="0" * 64,
        python_version="3",
        gpu_names=[],
        dataset_uri="",
        dataset_hash="merkle:" + "0" * 64,
        config={},
        command_line="chaos",
    )
    run = runs.start_run(meta, repro, store_dir=Path(store))
    manager = CheckpointManager(
        run, dir=Path(root) / "ckpt", keep_last=100, every_n_steps=1, install_signal_handlers=False
    )
    current = {"step": 0, "replace_calls": 0}
    real_link = fsops.link_noreplace

    def link(dir_fd: int, source: str, target: str) -> None:
        if current["step"] == STALL_STEP:
            current["replace_calls"] += 1
            if (point, current["replace_calls"]) in {("data_written", 1), ("manifest_renamed", 2)}:
                stall(point)
        real_link(dir_fd, source, target)

    fsops.link_noreplace = link  # type: ignore[assignment]
    real_record = CheckpointManager._record
    real_pointer = RunCheckpoints._write_pointer

    def record(self: CheckpointManager, manifest: object, digest: str) -> None:
        if point == "before_record" and current["step"] == STALL_STEP:
            stall(point)
        real_record(self, manifest, digest)  # type: ignore[arg-type]

    def write_pointer(self: RunCheckpoints, run_fd: int, manifest: object, digest: str) -> None:
        if point == "before_pointer" and current["step"] == STALL_STEP:
            stall(point)
        real_pointer(self, run_fd, manifest, digest)  # type: ignore[arg-type]

    CheckpointManager._record = record  # type: ignore[method-assign]
    RunCheckpoints._write_pointer = write_pointer  # type: ignore[method-assign]
    print(f"RUN {run_id}", flush=True)
    width = int(megabytes * 1024 * 1024 / 4)
    step = 0
    while True:
        step += 1
        current["step"] = step
        current["replace_calls"] = 0
        weights = np.full(width, float(step), dtype=np.float32)
        manager.save_now(step, model={"w": weights})
        print(f"ACK {step}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))
