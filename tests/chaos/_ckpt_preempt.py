"""Child for the preemption tests: trains until it is told to stop, saving only on preemption.

``argv``: checkpoint root, store root. Prints ``READY`` once the loop runs, then ``DONE <step>``
(or ``DONE none``) after ``maybe_save`` reported the preemption and the loop left.
"""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import Repro, RunMeta


def main(root: str, store: str) -> None:
    run_id = new_run_id()
    meta = RunMeta(
        run_id=run_id, project="chaos", name="c", status="running", task="detection",
        framework="python==3.12.1", host="h", started_at=datetime.now(timezone.utc),
        schema_version=1,
    )  # fmt: skip
    repro = Repro(
        run_id=run_id, git_dirty=False, packages_lock_sha256="0" * 64, python_version="3",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "0" * 64, config={},
        command_line="c",
    )  # fmt: skip
    run = runs.start_run(meta, repro, store_dir=Path(store))
    manager = CheckpointManager(
        run, dir=Path(root) / "ckpt", every_n_steps=100_000, every_n_minutes=None,
        grace_seconds=30.0,
    )  # fmt: skip
    print(f"RUN {run_id}", flush=True)
    step = 0
    while True:
        step += 1
        time.sleep(0.01)
        if step == 5:
            print("READY", flush=True)
        saved = manager.maybe_save(step, model={"w": np.full(8, float(step), dtype=np.float32)})
        if manager.preempted():
            print(f"DONE {saved.step if saved else 'none'}", flush=True)
            return


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
