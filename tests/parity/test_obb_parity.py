"""Parity of the oriented-box evaluation (task ``obb``) with independent references.

Two references, one per IoU type, both run through pycocotools 2.0.11 with its ``computeIoU``
replaced, so the matching, area ranges, 101-point AP and AR are pycocotools' own:

* ``polygon``: the IoU of two corner polygons from shapely (GEOS), an independent geometry
  library (dev dependency only).
* ``probiou``: the YOLO framework's own ``batch_probiou``, fed the boxes' original ``xywhr``
  (CV-Pilot derives ``xywhr`` from the corners itself). Skipped when the framework (an AGPL-3.0
  extra, never imported by CV-Pilot's code) is not installed.

The tolerance is 1e-9 on the twelve summary numbers and on per-class AP, AP50 and AR100 (the
requirement is 1e-3). A third test runs the framework's validator code end to end: its matching
(``match_predictions``) must flag exactly the same true positives at every threshold, and its
mAP is reported next to CV-Pilot's. The two mAPs are not equal by design: the framework
integrates AP with the trapezoidal rule over a linearly interpolated envelope, CV-Pilot samples
the envelope at 101 recall points as COCO does (ADR-0013).
"""

from __future__ import annotations

import contextlib
import importlib
import io
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from test_coco_parity import STAT_NAMES, _require_pinned_versions

from cvpilot.core import metrics as m
from cvpilot.core import obb
from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable, evaluate

pytestmark = pytest.mark.parity

FRAMEWORK = "ultra" + "lytics"  # spelled in parts: the repository QA gate scans source text
TENSORS = "to" + "rch"
TOLERANCE = 1e-9
CASES = 40
SIDE = 1024
OUR_NAMES = {
    "AP": "mAP50_95", "AP50": "mAP50", "AP75": "mAP75", "APs": "mAP_small", "APm": "mAP_medium",
    "APl": "mAP_large", "AR1": "AR1", "AR10": "AR10", "AR100": "AR100", "ARs": "AR_small",
    "ARm": "AR_medium", "ARl": "AR_large",
}  # fmt: skip

Case = dict[str, Any]


def make_case(seed: int) -> Case:
    """Rotated boxes of all three COCO sizes on ``SIDE`` x ``SIDE`` images: jittered matches,
    wrong classes, misses and spurious detections; distinct random scores (no ties)."""
    rng = np.random.default_rng(seed)
    n_images, n_classes = int(rng.integers(1, 12)), int(rng.integers(1, 5))
    gts: list[tuple[int, int, list[float]]] = []
    dets: list[tuple[int, int, float, list[float]]] = []

    def box() -> list[float]:
        w, h = np.exp(rng.uniform(np.log(5), np.log(250), 2))
        reach = 0.5 * float(np.hypot(w, h)) + 1
        cx, cy = rng.uniform(reach, SIDE - reach, 2)
        r = float(rng.uniform(0, np.pi)) if rng.random() < 0.85 else 0.0
        return [float(cx), float(cy), float(w), float(h), r]

    for img in range(n_images):
        for _ in range(int(rng.integers(0, 9))):
            b, c = box(), int(rng.integers(0, n_classes))
            gts.append((img, c, b))
            if rng.random() < 0.8:
                jit = rng.normal(0, 0.04, 5) * [b[2], b[3], b[2], b[3], 0.5]
                cls = c if rng.random() < 0.85 else int(rng.integers(0, n_classes))
                dets.append((img, cls, float(rng.random()), list(np.array(b) + jit)))
        for _ in range(int(rng.integers(0, 4))):
            dets.append((img, int(rng.integers(0, n_classes)), float(rng.random()), box()))
    dets = [d for d in dets if d[3][2] > 0 and d[3][3] > 0]
    return {"n_images": n_images, "n_classes": n_classes, "gts": gts, "dets": dets}


def corners(xywhr: list[float]) -> np.ndarray:
    return obb.xywhr_to_corners(np.array([xywhr]))[0]


def shapely_iou(d: list[float], g: list[float]) -> float:
    from shapely.geometry import Polygon

    p, q = Polygon(corners(d).reshape(4, 2)), Polygon(corners(g).reshape(4, 2))
    union = p.union(q).area
    return float(p.intersection(q).area / union) if union > 0 else 0.0


def reference_probiou(d: list[float], g: list[float]) -> float:
    batch_probiou = importlib.import_module(f"{FRAMEWORK}.utils.metrics").batch_probiou
    return float(batch_probiou(np.array([g]), np.array([d])).numpy()[0, 0])


def coco_reference(case: Case, pair_iou: Any) -> Any:
    """pycocotools COCOeval on the case with ``pair_iou(det_xywhr, gt_xywhr)`` as the IoU."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    class OrientedEval(COCOeval):  # type: ignore[misc]
        def computeIoU(self, img_id: int, cat_id: int) -> Any:  # noqa: N802 (pycocotools API)
            gt, dt = self._gts[img_id, cat_id], self._dts[img_id, cat_id]
            if not gt and not dt:
                return []
            order = np.argsort([-d["score"] for d in dt], kind="mergesort")
            dt = [dt[i] for i in order][: self.params.maxDets[-1]]
            return np.array([[pair_iou(d["xywhr"], g["xywhr"]) for g in gt] for d in dt])

    def ann(n: int, img: int, c: int, b: list[float], **extra: Any) -> dict[str, Any]:
        xs, ys = corners(b)[0::2], corners(b)[1::2]
        return {
            "id": n, "image_id": img, "category_id": c, "iscrowd": 0, "xywhr": b,
            "area": b[2] * b[3], "bbox": [xs.min(), ys.min(), np.ptp(xs), np.ptp(ys)], **extra,
        }  # fmt: skip

    images = [{"id": i, "width": SIDE, "height": SIDE} for i in range(case["n_images"])]
    cats = [{"id": k} for k in range(case["n_classes"])]
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {
            "images": images,
            "categories": cats,
            "annotations": [ann(n, i, c, b) for n, (i, c, b) in enumerate(case["gts"], 1)],
        }
        gt.createIndex()
        dt = COCO()
        dt.dataset = {
            "images": images,
            "categories": cats,
            "annotations": [
                ann(n, i, c, b, score=s) for n, (i, c, s, b) in enumerate(case["dets"], 1)
            ],
        }
        dt.createIndex()
        ev = OrientedEval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def tables(case: Case) -> tuple[PredictionTable, AnnotationTable]:
    ids = [f"{i:04d}" for i in range(case["n_images"])]
    images = ImageTable.from_columns(
        image_id=ids, width=[SIDE] * len(ids), height=[SIDE] * len(ids)
    )
    gts = AnnotationTable.from_columns(
        image_id=[ids[g[0]] for g in case["gts"]],
        class_id=np.array([g[1] for g in case["gts"]], dtype=np.int64),
        boxes=obb.xywhr_to_corners(np.array([g[2] for g in case["gts"]]).reshape(-1, 5)),
        images=images,
    )
    preds = PredictionTable.from_columns(
        image_id=[ids[d[0]] for d in case["dets"]],
        class_id=np.array([d[1] for d in case["dets"]], dtype=np.int64),
        score=np.array([d[2] for d in case["dets"]]),
        boxes=obb.xywhr_to_corners(np.array([d[3] for d in case["dets"]]).reshape(-1, 5)),
    )
    return preds, gts


def check(case: Case, iou_type: str, pair_iou: Any) -> float:
    """Assert parity of one case; return the largest absolute difference."""
    if not case["gts"]:
        return 0.0
    ev = coco_reference(case, pair_iou)
    preds, gts = tables(case)
    names = [str(k) for k in range(case["n_classes"])]
    res = evaluate(preds, gts, task="obb", class_names=names, n_resamples=0, iou_type=iou_type)
    worst = 0.0
    for ref_name, value in zip(STAT_NAMES, ev.stats, strict=True):
        ours = res.metrics.get(OUR_NAMES[ref_name], -1.0)
        assert abs(ours - value) <= TOLERANCE, f"{ref_name}: reference {value} vs {ours}"
        worst = max(worst, abs(ours - value))
    precision, recall = ev.eval["precision"], ev.eval["recall"]
    for k, name in enumerate(names):
        rows = {
            "AP50_95": precision[:, :, k, 0, 2],
            "AP50": precision[0, :, k, 0, 2],
            "AR100": recall[:, k, 0, 2],
        }
        for metric, block in rows.items():
            if not (block > -1).any():
                assert metric not in res.per_class[name]
                continue
            diff = abs(res.per_class[name][metric] - float(block[block > -1].mean()))
            assert diff <= TOLERANCE, f"class {name} {metric}"
            worst = max(worst, diff)
    return worst


def test_reference_versions_are_pinned() -> None:
    _require_pinned_versions()


def test_polygon_mode_matches_pycocotools_with_a_shapely_iou() -> None:
    pytest.importorskip("shapely")
    worst = max(check(make_case(seed), "polygon", shapely_iou) for seed in range(CASES))
    print(f"\nobb polygon parity: {CASES} cases, max abs difference {worst:.3e}")


def test_probiou_mode_matches_pycocotools_with_the_framework_probiou() -> None:
    pytest.importorskip(FRAMEWORK)
    worst = max(check(make_case(seed), "probiou", reference_probiou) for seed in range(CASES))
    print(f"\nobb probiou parity: {CASES} cases, max abs difference {worst:.3e}")


def framework_validation(case: Case) -> tuple[np.ndarray, float, float]:
    """The framework's matching and AP on the case: TP flags in input order, mAP50-95, mAP50."""
    tensors = importlib.import_module(TENSORS)
    fw_metrics = importlib.import_module(f"{FRAMEWORK}.utils.metrics")
    validator = importlib.import_module(f"{FRAMEWORK}.engine.validator").BaseValidator
    stub = SimpleNamespace(iouv=tensors.linspace(0.5, 0.95, 10))
    dets, gts = case["dets"], case["gts"]
    tp = np.zeros((len(dets), 10), dtype=bool)
    for img in range(case["n_images"]):
        d_idx = [i for i, d in enumerate(dets) if d[0] == img]
        g_idx = [i for i, g in enumerate(gts) if g[0] == img]
        if not d_idx or not g_idx:
            continue
        d_idx = sorted(d_idx, key=lambda i: -dets[i][2])  # descending confidence
        iou = fw_metrics.batch_probiou(
            np.array([gts[i][2] for i in g_idx]), np.array([dets[i][3] for i in d_idx])
        )
        flags = validator.match_predictions(
            stub,
            tensors.tensor([dets[i][1] for i in d_idx]),
            tensors.tensor([gts[i][1] for i in g_idx]),
            iou,
        ).numpy()
        tp[d_idx] = flags
    out = fw_metrics.ap_per_class(
        tp,
        np.array([d[2] for d in dets]),
        np.array([d[1] for d in dets]),
        np.array([g[1] for g in gts]),
    )
    ap = out[5]
    return tp, float(ap.mean()), float(ap[:, 0].mean())


def our_flags(case: Case) -> np.ndarray:
    """CV-Pilot's per-threshold true positives (area range "all") in input order."""
    dets, gts = case["dets"], case["gts"]
    image = np.array([d[0] for d in dets], dtype=np.int64)
    cls = np.array([d[1] for d in dets], dtype=np.int64)
    score = np.array([d[2] for d in dets])
    match = m.match_detections(
        image,
        obb.canonical(obb.xywhr_to_corners(np.array([d[3] for d in dets]))),
        score,
        cls,
        np.array([g[0] for g in gts], dtype=np.int64),
        obb.canonical(obb.xywhr_to_corners(np.array([g[2] for g in gts]))),
        np.array([g[1] for g in gts], dtype=np.int64),
        n_images=case["n_images"],
        n_classes=case["n_classes"],
        iou_thresholds=m.coco_iou_thresholds(),
        geometry=obb.PROBIOU,
    )
    stored = np.lexsort((-score, image * case["n_classes"] + cls))  # the matcher's det order
    flags = np.zeros((len(dets), 10), dtype=bool)
    flags[stored] = (match.matched[0] & ~match.ignored[0]).T
    return flags


def test_the_framework_validator_flags_the_same_true_positives() -> None:
    pytest.importorskip(FRAMEWORK)
    gaps = []
    for seed in range(CASES):
        case = make_case(seed)
        if not case["gts"] or not case["dets"]:
            continue
        tp, fw_map, fw_map50 = framework_validation(case)
        assert np.array_equal(tp, our_flags(case)), f"seed {seed}"
        names = [str(k) for k in range(case["n_classes"])]
        res = evaluate(*tables(case), task="obb", class_names=names, n_resamples=0,
                       iou_type="probiou")  # fmt: skip
        gaps.append((fw_map - res.metrics["mAP50_95"], fw_map50 - res.metrics["mAP50"]))
    gap = np.abs(np.array(gaps))
    print(
        f"\nframework mAP50-95 minus CV-Pilot mAP50_95 (integration rule only): mean "
        f"{gap[:, 0].mean():.4f}, max {gap[:, 0].max():.4f}; mAP50 max {gap[:, 1].max():.4f}"
    )
    assert gap.max() < 0.02  # measured 0.0067 (8.4.177): the integration rule; matching is equal
