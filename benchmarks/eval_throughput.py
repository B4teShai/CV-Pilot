"""Eval engine throughput and memory on a synthetic detection set (budgets: 01-overview.md).

Reference input: 5 000 images, 50 classes, about 30 000 ground-truth boxes and 100 000
detections. Budgets: evaluation <= 10 s on 4 CPU cores, bootstrap with 1 000 resamples <= 5 s
extra, peak RSS <= 2 GB. Stages run in one process, so the RSS printed after a stage is the
peak so far (``ru_maxrss``); the end-to-end stage runs ``cvpilot eval`` in a child process and
reports the child's own peak.

    uv run python benchmarks/eval_throughput.py [--images 5000] [--classes 50] [--dets 100000]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable, evaluate


def synth(n_images: int, n_classes: int, n_dets: int, seed: int = 0):  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(seed)
    per_image = rng.poisson(6, n_images)
    g_img = np.repeat(np.arange(n_images), per_image)
    n_gt = g_img.size
    size = np.exp(rng.uniform(np.log(8), np.log(250), (n_gt, 2)))
    xy = rng.uniform(0, 640 - 250, (n_gt, 2))
    g_box = np.column_stack([xy, xy + size])
    g_cls = rng.integers(0, n_classes, n_gt)
    hit = rng.random(n_gt) < 0.8
    jitter = rng.normal(0, 0.04, (hit.sum(), 4)) * np.tile(size[hit], 2)
    p_img = [g_img[hit]]
    p_box = [g_box[hit] + jitter]
    wrong = rng.random(hit.sum()) < 0.05
    p_cls = [np.where(wrong, rng.integers(0, n_classes, hit.sum()), g_cls[hit])]
    extra = max(0, n_dets - hit.sum())
    e_size = np.exp(rng.uniform(np.log(8), np.log(250), (extra, 2)))
    e_xy = rng.uniform(0, 640 - 250, (extra, 2))
    p_img.append(rng.integers(0, n_images, extra))
    p_box.append(np.column_stack([e_xy, e_xy + e_size]))
    p_cls.append(rng.integers(0, n_classes, extra))
    img, box, cls = np.concatenate(p_img), np.vstack(p_box), np.concatenate(p_cls)
    score = rng.random(img.size)
    ids = np.array([f"img{i:06d}" for i in range(n_images)])
    images = ImageTable.from_columns(
        image_id=ids,
        width=np.full(n_images, 640),
        height=np.full(n_images, 640),
        meta={"camera": [f"cam{i % 10:02d}" for i in range(n_images)]},
    )
    gts = AnnotationTable.from_columns(
        image_id=ids[g_img], class_id=g_cls, boxes=g_box, images=images
    )
    preds = PredictionTable.from_columns(
        image_id=ids[img], class_id=cls, score=score, boxes=np.clip(box, 0, None)
    )
    return preds, gts


def rss_mb(who: int = resource.RUSAGE_SELF) -> float:
    peak = resource.getrusage(who).ru_maxrss
    return peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024


def timed(fn):  # type: ignore[no-untyped-def]
    start = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - start


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=5000)
    ap.add_argument("--classes", type=int, default=50)
    ap.add_argument("--dets", type=int, default=100_000)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4, help="bootstrap threads")
    args = ap.parse_args()
    names = [f"class{k:03d}" for k in range(args.classes)]
    preds, gts = synth(args.images, args.classes, args.dets)
    kw = {"task": "detection", "class_names": names}
    report: dict[str, object] = {
        "platform": f"{platform.system()} {platform.machine()}",
        "python": platform.python_version(),
        "cpus": os.cpu_count(),
        "images": args.images,
        "classes": args.classes,
        "ground_truth_boxes": len(gts),
        "detections": len(preds),
        "rss_after_generation_mb": round(rss_mb(), 1),
    }
    base, t_eval = timed(lambda: evaluate(preds, gts, n_resamples=0, **kw))
    report["evaluate_s"] = round(t_eval, 2)
    report["rss_after_evaluate_mb"] = round(rss_mb(), 1)
    report["mAP50_95"] = round(base.metrics["mAP50_95"], 4)
    boot, t_boot = timed(
        lambda: evaluate(preds, gts, n_resamples=args.resamples, workers=args.workers, **kw)
    )
    report["evaluate_with_bootstrap_s"] = round(t_boot, 2)
    report["bootstrap_extra_s"] = round(t_boot - t_eval, 2)
    report["bootstrap_resamples"] = args.resamples
    report["bootstrap_workers"] = args.workers
    report["rss_after_bootstrap_mb"] = round(rss_mb(), 1)
    _, t_slice = timed(
        lambda: evaluate(preds, gts, n_resamples=0, slices={"camera": "camera"}, **kw)
    )
    report["evaluate_with_10_slices_s"] = round(t_slice, 2)
    report["slice_extra_s"] = round(t_slice - t_eval, 2)
    with tempfile.TemporaryDirectory() as tmp:
        report["end_to_end_cli"] = end_to_end(Path(tmp), preds, gts, names, args)
    print(json.dumps(report, indent=2))


def end_to_end(tmp: Path, preds, gts, names, args) -> dict[str, float]:  # type: ignore[no-untyped-def]
    import pyarrow as pa
    import pyarrow.parquet as pq

    img = gts.images
    pos = {v: i for i, v in enumerate(img.image_id.tolist())}
    empty = sorted(set(img.image_id.tolist()) - set(gts.image_id.tolist()))
    ids = np.concatenate([gts.image_id, np.array(empty, dtype=str)])
    pad = [None] * len(empty)  # rows with a null cls declare images without objects
    cols = {"image_id": ids, "cls": pa.array([*gts.class_id.tolist(), *pad], type=pa.int64())}
    for i, name in enumerate(("x1", "y1", "x2", "y2")):
        cols[name] = pa.array([*gts.boxes[:, i].tolist(), *pad], type=pa.float64())
    rows = [pos[v] for v in ids.tolist()]
    cols["width"] = img.width[rows]
    cols["height"] = img.height[rows]
    pq.write_table(pa.table(cols), tmp / "g.parquet")
    pc = {"image_id": preds.image_id, "cls": preds.class_id, "score": preds.score}
    for i, name in enumerate(("x1", "y1", "x2", "y2")):
        pc[name] = preds.boxes[:, i]
    pq.write_table(pa.table(pc), tmp / "p.parquet")
    (tmp / "c.txt").write_text("\n".join(names))
    env = {**os.environ, "CVPILOT_STORE_DIR": str(tmp / "store"), "CVPILOT_PROJECT": "bench"}
    code = "import cvpilot; r = cvpilot.init(task='detection'); r.finish(); print(r.run_id)"
    run_id = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    ).stdout.strip()
    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    start = time.perf_counter()
    subprocess.run(  # noqa: S603
        [
            sys.executable, "-c", "from cvpilot.cli.app import main; main()",
            "eval", run_id, "--preds", str(tmp / "p.parquet"), "--gts", str(tmp / "g.parquet"),
            "--classes", str(tmp / "c.txt"), "--resamples", str(args.resamples), "--allow-closed",
        ],
        env=env, capture_output=True, text=True, check=True,
    )  # fmt: skip
    wall = time.perf_counter() - start
    peak = max(before, resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    peak_mb = peak / (1024 * 1024) if sys.platform == "darwin" else peak / 1024
    return {"wall_s": round(wall, 2), "child_peak_rss_mb": round(peak_mb, 1)}


if __name__ == "__main__":
    main()
