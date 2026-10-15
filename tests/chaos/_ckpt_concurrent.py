"""Child for the concurrent-savers chaos test: one of several processes saving into one run."""

import sys
from pathlib import Path

import numpy as np

from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.errors import CheckpointBusyError, CheckpointStepError


def main(root: str, run_id: str, offset: int, stride: int, count: int) -> None:
    rc = RunCheckpoints(Path(root), run_id, create=True, lock_timeout=60.0)
    codec = get_serializer("safetensors")
    saved = refused = 0
    for step in range(offset, count + 1, stride):
        data = codec.dumps({"model": {"w": np.full(50_000, float(step), dtype=np.float32)}})
        try:
            rc.save(
                kind="last", step=step, data=data, format="safetensors", versions={},
                dataset_hash="merkle:" + "0" * 64,
            )  # fmt: skip
            saved += 1
        except CheckpointStepError:
            refused += 1  # another process already saved a newer step: the order is kept
        except CheckpointBusyError:
            refused += 1
    print(f"DONE {saved} {refused}", flush=True)


if __name__ == "__main__":
    main(*(sys.argv[1], sys.argv[2], *map(int, sys.argv[3:])))
