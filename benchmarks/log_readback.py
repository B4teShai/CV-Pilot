"""Log 100 000 metric points through the public API and read them back identically.

Usage: uv run python benchmarks/log_readback.py [--steps 2000] [--keys 50]

Prints timings (total, per-call p50/p95/max of ``log`` with ``--keys`` scalars), the number of
Parquet files and bytes, and exits 1 on any mismatch. Uses a temporary store directory.
"""

from __future__ import annotations

import argparse
import math
import statistics
import tempfile
import time
from pathlib import Path

import cvpilot
from cvpilot.runs import get_run_view
from cvpilot.store.local import open_store


def expected(step: int, k: int) -> float:
    if (step + k) % 997 == 0:
        return math.nan
    if (step * 31 + k) % 1009 == 0:
        return math.inf
    return math.sin(step * 0.01 + k) * 10 ** (k % 7 - 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--keys", type=int, default=50)
    args = parser.parse_args()
    keys = [f"train/metric_{k:02d}" for k in range(args.keys)]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run = cvpilot.init("bench", task="detection", store_dir=root)
        latencies: list[float] = []
        t0 = time.perf_counter()
        for step in range(args.steps):
            points = {key: expected(step, k) for k, key in enumerate(keys)}
            t = time.perf_counter()
            run.log(points, step=step)
            latencies.append((time.perf_counter() - t) * 1000)
        t_log = time.perf_counter() - t0
        t1 = time.perf_counter()
        run.finish()
        t_finish = time.perf_counter() - t1

        t2 = time.perf_counter()
        store = open_store("bench", root, create=False)
        points_back = store.read_points(run.run_id)
        t_read = time.perf_counter() - t2
        stats = store.metric_stats(run.run_id)
        store.close()

        n = args.steps * args.keys
        got = {(p.key, p.step): p.value for p in points_back}
        bad = 0
        for step in range(args.steps):
            for k, key in enumerate(keys):
                want, have = expected(step, k), got.get((key, step))
                same = have is not None and (
                    (math.isnan(want) and math.isnan(have)) or want == have
                )
                bad += not same
        view = get_run_view("bench", run.run_id, store_dir=root)
        size = sum(f.stat().st_size for f in (view.store_path / "metrics").rglob("*.parquet"))
        qs = statistics.quantiles(latencies, n=100)
        print(f"points logged        : {n}")
        print(f"points read back     : {len(points_back)} (unique {len(got)})")
        print(f"mismatches           : {bad}")
        print(f"status               : {view.meta.status}")
        print(f"parquet files / bytes: {stats.n_files} / {size}")
        print(f"log total            : {t_log:.3f} s ({n / t_log:,.0f} points/s)")
        print(
            f"log() p50/p95/max    : {qs[49]:.3f} / {qs[94]:.3f} / {max(latencies):.3f} ms per call"
        )
        print(f"finish (flush+compact): {t_finish:.3f} s")
        print(f"read back            : {t_read:.3f} s")
        ok = bad == 0 and len(points_back) == n == len(got) and view.meta.status == "finished"
        print("RESULT               :", "identical" if ok else "MISMATCH")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
