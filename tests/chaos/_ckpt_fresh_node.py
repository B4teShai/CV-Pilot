"""Child process for the fresh-node test: one process saves to a bucket, a later one on a
node with no run store and no checkpoint directory finds those checkpoints by itself.

``argv``: mode (``write`` or ``restore``), bucket directory, node directory. The bucket is a
local directory served under the ``s3`` scheme through fsspec, so it outlives each process.
"""

import sys
import types
from datetime import datetime, timezone
from pathlib import Path

import fsspec
import numpy as np
from fsspec.implementations.dirfs import DirFileSystem
from fsspec.implementations.local import LocalFileSystem

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import Repro, RunMeta

URI = "s3://bucket-name/fresh/"


def install_bucket(bucket: str) -> None:
    sys.modules.setdefault("s3fs", types.ModuleType("s3fs"))  # the backend check only imports it

    class Bucket(DirFileSystem):
        def __init__(self, **_options: object) -> None:
            super().__init__(path=bucket, fs=LocalFileSystem(auto_mkdir=True))

    fsspec.register_implementation("s3", Bucket, clobber=True)


def new_run(store: Path) -> runs.RunHandle:
    run_id = new_run_id()
    meta = RunMeta(
        run_id=run_id, project="fresh", name="fresh", status="running", task="detection",
        framework="python==3.12.1", host="h", started_at=datetime.now(timezone.utc),
        schema_version=1,
    )  # fmt: skip
    repro = Repro(
        run_id=run_id, git_dirty=False, packages_lock_sha256="0" * 64, python_version="3",
        gpu_names=[], dataset_uri="", dataset_hash="merkle:" + "0" * 64, config={},
        command_line="fresh",
    )  # fmt: skip
    return runs.start_run(meta, repro, store_dir=store)


def main(mode: str, bucket: str, node: str) -> None:
    install_bucket(bucket)
    root = Path(node)
    run = new_run(root / "store")
    manager = CheckpointManager(
        run, dir=root / "ckpt", remote=URI, every_n_steps=1, keep_last=10,
        install_signal_handlers=False,
    )  # fmt: skip
    if mode == "write":
        for step in (1, 2, 3):
            manager.save_now(step, model={"w": np.full(2, float(step), dtype=np.float32)})
        manager.close()
        run.finish("preempted")
        print("WROTE 3", flush=True)
        return
    restored = manager.restore(discover=True)
    if restored is None:
        print("RESTORED none", flush=True)
    else:
        print(
            f"RESTORED {restored.manifest.step} {restored.state['model']['w'].tolist()}", flush=True
        )
    run.finish()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
