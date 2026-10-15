"""Child process for the SIGKILL chaos test: appends batches forever, acking each on stdout."""

import sys
from datetime import datetime, timezone
from pathlib import Path

from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import MetricPoint, Repro, RunMeta
from cvpilot.store.local import open_store

BATCH = 2000


def main(root: str) -> None:
    store = open_store("chaos", Path(root))
    rid = new_run_id()
    meta = RunMeta(
        run_id=rid,
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
        run_id=rid,
        git_dirty=False,
        packages_lock_sha256="0" * 64,
        python_version="3",
        gpu_names=[],
        dataset_uri="",
        dataset_hash="merkle:" + "0" * 64,
        config={},
        command_line="chaos",
    )
    store.put_run(meta, repro)
    print(f"RUN {rid}", flush=True)
    batch = 0
    while True:
        start = batch * BATCH
        points = [
            MetricPoint(run_id=rid, key="k", step=start + i, value=float(start + i), wall_time=1.0)
            for i in range(BATCH)
        ]
        store.append_points(points)
        batch += 1
        print(f"ACK {batch}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
