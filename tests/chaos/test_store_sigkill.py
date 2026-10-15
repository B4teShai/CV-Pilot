"""kill -9 a writer mid-batch: the store must open, earlier batches must read back intact."""

from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from cvpilot.core.schema import MetricPoint
from cvpilot.store.local import open_store

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX SIGKILL"),
]
WRITER = Path(__file__).with_name("_writer.py")
BATCH = 2000


def kill_writer(root: Path, min_acks: int, extra_delay: float) -> tuple[str, int]:
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, str(WRITER), str(root)],
        stdout=subprocess.PIPE,
        text=True,
    )
    assert proc.stdout is not None
    run_id, acks = "", 0
    try:
        for line in proc.stdout:
            if line.startswith("RUN "):
                run_id = line.split()[1]
            elif line.startswith("ACK "):
                acks = int(line.split()[1])
                if acks >= min_acks:
                    time.sleep(extra_delay)  # land the kill inside a later batch write
                    break
    finally:
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait()
        proc.stdout.close()
    assert proc.returncode == -signal.SIGKILL
    return run_id, acks


@pytest.mark.parametrize("extra_delay", [0.0, 0.003, 0.011])
def test_killed_writer_leaves_a_readable_store(tmp_path: Path, extra_delay: float) -> None:
    run_id, acked = kill_writer(tmp_path, min_acks=3, extra_delay=extra_delay)
    assert acked >= 3

    db = sqlite3.connect(tmp_path / "chaos" / "runs.sqlite")
    assert db.execute("PRAGMA integrity_check").fetchone() == ("ok",)
    db.close()

    store = open_store("chaos", tmp_path)
    points = store.read_points(run_id)
    # every acknowledged batch is present, and only whole batches ever become visible
    assert len(points) >= acked * BATCH
    assert len(points) % BATCH == 0
    assert [p.step for p in points] == list(range(len(points)))
    assert all(p.value == float(p.step) for p in points)
    assert store.get_run(run_id).status == "running"  # crashed runs stay running

    # the store is still usable: continue the same run, then compact it
    n = len(points)
    store.append_points(
        [
            MetricPoint(run_id=run_id, key="k", step=n + i, value=1.0, wall_time=1.0)
            for i in range(5)
        ]
    )
    store.compact_metrics(run_id)
    again = store.read_points(run_id)
    assert len(again) == n + 5 and [p.step for p in again[:n]] == list(range(n))
    store.close()
