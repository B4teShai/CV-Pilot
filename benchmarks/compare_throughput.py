"""Compare and gate cost on a 5 000-image evaluation (budgets: 01-overview.md).

Reference input: 5 000 images, 50 classes, 10 slices, 100 000 detections. Two runs are evaluated
into a temporary store (the second with 10 % of the detections dropped, so it regresses); the
timed stages are what ``cvpilot gate`` does: pin a baseline, resolve and verify both snapshots
and the confusion artifacts, compare, gate, render the HTML and JSON reports, and the whole CLI
in a child process (interpreter start included). The evaluation itself is not timed here (see
``eval_throughput.py``). Budgets: ``gate`` end to end including both reports <= 3 s, peak RSS of
the CLI <= 300 MB.

    uv run python benchmarks/compare_throughput.py [--images 5000] [--classes 50] [--repeat 5]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from eval_throughput import synth

from cvpilot.compare import service
from cvpilot.compare.baselines import set_baseline
from cvpilot.compare.engine import compare
from cvpilot.compare.gate import Thresholds, gate
from cvpilot.compare.html_report import render_html
from cvpilot.compare.report import report_json
from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, Repro, RunMeta
from cvpilot.eval import PredictionTable
from cvpilot.eval.record import record_eval
from cvpilot.store.local import open_store

SHA = "b" * 64


def new_run(store):  # type: ignore[no-untyped-def]
    run_id = new_run_id()
    started = datetime(2026, 10, 15, tzinfo=timezone.utc)
    meta = RunMeta(
        run_id=run_id, project="bench", name="b", status="finished", task="detection",
        framework="python==3.12.0", host="h", started_at=started, ended_at=started,
        schema_version=SCHEMA_VERSION,
    )  # fmt: skip
    repro = Repro(
        run_id=run_id, git_dirty=False, packages_lock_sha256=SHA, python_version="3.12.0",
        gpu_names=[], dataset_uri="/d", dataset_hash="merkle:" + SHA, config={}, command_line="x",
    )  # fmt: skip
    store.put_run(meta, repro)
    return meta


def timed(fn):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def rss_mb(who: int) -> float:
    peak = resource.getrusage(who).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=5000)
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--dets", type=int, default=100_000)
    ap.add_argument("--repeat", type=int, default=5)
    args = ap.parse_args()
    names = [f"class{k:03d}" for k in range(args.classes)]
    preds, gts = synth(args.images, args.classes, args.dets)
    keep = np.random.default_rng(1).random(len(preds)) > 0.10
    worse = PredictionTable.from_columns(
        image_id=preds.image_id[keep], class_id=preds.class_id[keep], score=preds.score[keep],
        boxes=preds.boxes[keep],
    )  # fmt: skip
    report: dict[str, object] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "images": args.images,
        "classes": args.classes,
        "slices": 10,
        "detections": len(preds),
    }
    with tempfile.TemporaryDirectory() as tmp:
        top = Path(tmp)
        root, out = top / "store", top / "out"  # reports never go inside the store
        out.mkdir()
        store = open_store("bench", root)
        ids = []
        for p in (preds, worse):
            meta = new_run(store)
            record_eval(
                store, meta, p, gts, step=1, class_names=names, slices={"camera": "camera"},
                n_resamples=200, allow_closed=True,
            )  # fmt: skip
            ids.append(meta.run_id)
        store.close()
        base_id, cand_id = ids
        _, report["pin_baseline_s"] = timed(
            lambda: set_baseline("bench", "prod", base_id, store_dir=root)
        )
        th = Thresholds(
            max_drop={"mAP50_95": 0.5}, per_class_max_drop={"AP50_95": 2.0},
            slice_max_drop={"camera=*": 1.5}, min_images=30,
        )  # fmt: skip
        times: dict[str, list[float]] = {
            k: [] for k in ("resolve", "compare_gate", "reports", "cli")
        }
        for _ in range(args.repeat):
            found, t = timed(lambda: service.resolve("bench", cand_id, "tag:prod", store_dir=root))  # noqa: B023
            times["resolve"].append(t)
            (result, cmp), t = timed(lambda: _cmp(found, th))  # noqa: B023
            times["compare_gate"].append(t)
            _, t = timed(
                lambda: (render_html(cmp, result, th), report_json(cmp, result, th))  # noqa: B023
            )
            times["reports"].append(t)
            for stale in ("r.html", "r.json"):  # every iteration writes its own reports
                (out / stale).unlink(missing_ok=True)
            done, t = timed(lambda: _cli(root, out, cand_id))
            if done.returncode != 1 or "gate: FAIL" not in done.stdout:  # the run must regress
                raise SystemExit(
                    f"unexpected gate result {done.returncode}: {done.stdout[:200]} "
                    f"{done.stderr[-600:]}"
                )
            times["cli"].append(t)
        report["gate_exit_code"] = done.returncode
        report["html_kb"] = round((out / "r.html").stat().st_size / 1024, 1)
        report["json_kb"] = round((out / "r.json").stat().st_size / 1024, 1)
    for key, values in times.items():
        report[f"{key}_median_s"] = round(statistics.median(values), 3)
        report[f"{key}_max_s"] = round(max(values), 3)
    report["cli_child_peak_rss_mb"] = round(rss_mb(resource.RUSAGE_CHILDREN), 1)
    report["in_process_peak_rss_mb"] = round(rss_mb(resource.RUSAGE_SELF), 1)
    print(json.dumps(report, indent=2))


def _cli(root: Path, out: Path, run_id: str) -> subprocess.CompletedProcess[str]:
    exe = str(Path(sys.executable).parent / "cvpilot")
    argv = [
        exe, "gate", run_id, "--project", "bench", "--baseline", "tag:prod",
        "--max-drop", "mAP50_95=0.5", "--per-class-max-drop", "AP50_95=2.0",
        "--slice-max-drop", "camera=*:1.5", "--report-dir", str(out),
        "--report", str(out / "r.html"), "--report-json", str(out / "r.json"),
    ]  # fmt: skip
    env = {**os.environ, "CVPILOT_STORE_DIR": str(root)}
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)  # noqa: S603


def _cmp(found, th):  # type: ignore[no-untyped-def]
    cmp = compare(found.candidate, found.baseline, confusion=found.confusion)
    return gate(cmp, th), cmp


if __name__ == "__main__":
    main()
