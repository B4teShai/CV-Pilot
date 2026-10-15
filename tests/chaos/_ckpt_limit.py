"""Child for the full-disk chaos test: a file size limit stands in for a full disk.

Saves a small checkpoint, then one that cannot fit (the kernel refuses the write with EFBIG),
then a small one again. Prints one line per outcome.
"""

import resource
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError
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
        run, dir=Path(root) / "ckpt", keep_last=10, install_signal_handlers=False
    )
    print(f"RUN {run_id}", flush=True)
    manager.save_now(1, model={"w": np.full(1000, 1.0, dtype=np.float32)})
    print("SAVED 1", flush=True)
    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    limit = 1_000_000
    resource.setrlimit(resource.RLIMIT_FSIZE, (limit, limit))
    try:
        manager.save_now(2, model={"w": np.full(2_000_000, 2.0, dtype=np.float32)})
        print("SAVED 2", flush=True)
    except CheckpointError as exc:
        print(f"FAILED 2 {type(exc).__name__}", flush=True)
    manager.save_now(3, model={"w": np.full(1000, 3.0, dtype=np.float32)})
    print("SAVED 3", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
