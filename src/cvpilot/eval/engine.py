"""``evaluate``: validated inputs in, an :class:`EvalResult` out. Deterministic and CPU-only.

Detection follows the COCO protocol (``cvpilot.core.metrics``); classification is top-1.
Metric values are fractions in ``[0, 1]``. Undefined metrics are left out of the result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from cvpilot.core import confusion as cf
from cvpilot.core import metrics as m
from cvpilot.core.errors import EvalInputError
from cvpilot.core.hashing import eval_set_hash as _hash
from cvpilot.core.metrics import I64, BoolArray
from cvpilot.core.schema import BootstrapInfo, CiStatus, EvalParams, EvalSnapshot, Task
from cvpilot.eval import bootstrap as bs
from cvpilot.eval import slices as sl
from cvpilot.eval import tables as t
from cvpilot.eval.validate import (
    GroundTruth,
    Predictions,
    prepare_ground_truth,
    prepare_predictions,
)

MAX_IOU_THRESHOLDS = 20
"""Most IoU thresholds a caller may pass; the matcher keeps one boolean plane per threshold."""

_DETECTION_HEADLINE = (
    ("mAP50", "AP50"),
    ("mAP75", "AP75"),
    ("mAP_small", "APs"),
    ("mAP_medium", "APm"),
    ("mAP_large", "APl"),
    ("AR1", "AR1"),
    ("AR10", "AR10"),
    ("AR100", "AR100"),
    ("AR_small", "ARs"),
    ("AR_medium", "ARm"),
    ("AR_large", "ARl"),
)


@dataclass(frozen=True)
class EvalResult:
    """Everything one evaluation produced, before it is tied to a run."""

    task: Task
    eval_set_hash: str
    class_names: list[str]
    n_images: int
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]]
    confusion: I64
    slices: dict[str, dict[str, float]] = field(default_factory=dict)
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)
    bootstrap: BootstrapInfo | None = None
    ci_status: CiStatus = "off"
    params: EvalParams | None = None
    slice_spec: dict[str, str] = field(default_factory=dict)

    def to_snapshot(
        self,
        *,
        run_id: str,
        step: int,
        confusion_sha256: str,
        predictions_sha256: str,
        provenance: Mapping[str, str | None] | None = None,
    ) -> EvalSnapshot:
        """The immutable record of this evaluation for ``run_id`` at ``step``."""
        return EvalSnapshot.model_validate(
            {
                "run_id": run_id,
                "eval_set_hash": self.eval_set_hash,
                "step": step,
                "task": self.task,
                "class_names": self.class_names,
                "n_images": self.n_images,
                "metrics": self.metrics,
                "per_class": self.per_class,
                "slices": self.slices,
                "ci": self.ci,
                "bootstrap": self.bootstrap,
                "ci_status": self.ci_status,
                "params": self.params,
                "provenance": dict(provenance) if provenance else None,
                "slice_spec": self.slice_spec,
                "confusion_sha256": confusion_sha256,
                "predictions_sha256": predictions_sha256,
            }
        )


def set_hash(gt: GroundTruth, class_names: Sequence[str]) -> str:
    """``eval_set_hash`` of a validated ground truth."""
    return _hash(
        class_names,
        gt.image_ids.tolist(),
        gt.width,
        gt.height,
        gt.ann_image,
        gt.ann_cls,
        gt.ann_boxes,
        gt.ann_crowd,
    )


def eval_set_hash(
    gts: t.AnnotationTable, class_names: Sequence[str], *, task: Task = "detection"
) -> str:
    """Content hash of the evaluation set: annotations, image list and class names only."""
    names = t.check_class_names(class_names)
    return set_hash(prepare_ground_truth(gts, len(names), task), names)


def _thresholds(values: Sequence[float] | None) -> m.F64:
    if values is None:
        return m.coco_iou_thresholds()
    arr = np.asarray(list(values), dtype=np.float64)
    if not 1 <= arr.size <= MAX_IOU_THRESHOLDS or not bool(np.isfinite(arr).all()):
        raise EvalInputError(f"iou_thresholds needs 1 to {MAX_IOU_THRESHOLDS} finite values")
    if bool((arr <= 0).any()) or bool((arr > 1).any()) or bool((np.diff(arr) <= 0).any()):
        raise EvalInputError("iou_thresholds must be strictly increasing values in (0, 1]")
    return arr


def _is_coco_grid(thresholds: m.F64) -> bool:
    return thresholds.size == 10 and bool(np.allclose(thresholds, m.coco_iou_thresholds()))


MAX_ESTIMATED_BYTES = 4 << 30
"""Largest estimated working set of one evaluation (4 GiB); see :func:`estimate_bytes`."""
WORKER_BYTES_PER_PREDICTION = 160
"""Bootstrap memory one worker thread adds per prediction (flag copies and cumulative sums)."""


def estimate_bytes(n_pred: int, n_gt: int, n_classes: int, n_thr: int, n_areas: int = 4) -> int:
    """Rough peak working set in bytes of ``evaluate``: the match flags per area and IoU
    threshold, the sorted prediction and box arrays, the precision tables and the confusion
    matrix. Measured peaks (benchmarks/eval_throughput.py) sit below it."""
    flags = 2 * n_areas * n_thr * n_pred
    arrays = 12 * 8 * n_pred + 10 * 8 * n_gt
    curves = n_thr * 101 * n_classes * n_areas * 3 * 8
    return flags + arrays + curves + (n_classes + 1) ** 2 * 8


def plan_workers(requested: int, n_pred: int, headroom: int = MAX_ESTIMATED_BYTES // 2) -> int:
    """Bootstrap threads that fit in ``headroom`` bytes (at least one)."""
    per_worker = max(1, WORKER_BYTES_PER_PREDICTION * n_pred)
    return max(1, min(int(requested), headroom // per_worker))


def evaluate(
    preds: t.PredictionTable,
    gts: t.AnnotationTable,
    *,
    task: Task,
    class_names: Sequence[str],
    slices: Mapping[str, str] | None = None,
    iou_thresholds: Sequence[float] | None = None,
    confusion_iou: float = 0.5,
    confusion_score: float = 0.25,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    workers: int = bs.DEFAULT_WORKERS,
) -> EvalResult:
    """Evaluate predictions against ground truth.

    Args:
        preds: detections (``image_id, class_id, score, boxes``) or top-1 labels.
        gts: ground-truth boxes or labels, with the image list.
        task: ``"detection"`` or ``"classification"``.
        class_names: names indexed by class id.
        slices: slice key -> image metadata column; each distinct value is one slice (bounded,
            see ``cvpilot.eval.slices``).
        iou_thresholds: strictly increasing IoU grid; the COCO grid by default.
        confusion_iou, confusion_score: matching IoU and score cut-off of the confusion matrix.
        n_resamples: bootstrap resamples over images for the headline intervals (0 turns them
            off; at most ``MAX_RESAMPLES``); ``seed`` seeds them, ``level`` is the coverage and
            ``workers`` the threads (the result does not depend on it).

    Raises:
        EvalInputError: any input fails validation (see ``cvpilot.eval.validate``) or the set
            has no ground-truth objects to score.
    """
    names = t.check_class_names(class_names)
    thresholds = _thresholds(iou_thresholds)
    if not (0.0 < confusion_iou <= 1.0) or not (0.0 <= confusion_score <= 1.0):
        raise EvalInputError("confusion_iou must be in (0, 1] and confusion_score in [0, 1]")
    needed = estimate_bytes(len(preds), len(gts), len(names), thresholds.size)
    if needed > MAX_ESTIMATED_BYTES:
        raise EvalInputError(
            f"estimated peak memory {needed} bytes exceeds the limit of {MAX_ESTIMATED_BYTES}"
        )
    workers = plan_workers(workers, len(preds))
    bs.check_seed(seed)
    if n_resamples != 0:
        bs.check_settings(n_resamples, level, seed)
    gt = prepare_ground_truth(gts, len(names), task)
    if gt.n_images == 0 or (task == "detection" and not bool((~gt.ann_crowd).any())):
        raise EvalInputError("no ground-truth objects to evaluate")
    plan = sl.plan_slices(slices, gt, len(preds) + len(gts))  # bounded before any metric
    pr = prepare_predictions(preds, gt, len(names), task)
    digest = set_hash(gt, names)
    ci: dict[str, tuple[float, float]] = {}
    slice_rows: dict[str, dict[str, float]] = {}
    if task == "classification":
        metrics, per_class, confusion, correct = _classification(gt, pr, names)
        for part in plan:
            slice_rows[part.label] = {
                "top1": float(correct[part.mask()].mean()),
                "n_images": float(part.mask().sum()),
            }
        if n_resamples and gt.n_images >= 2:
            ci["top1"] = bs.bootstrap_ci(
                correct.astype(np.float64), n_resamples=n_resamples, seed=seed, level=level
            )
    else:
        metrics, per_class, confusion, match = _detection(
            gt, pr, names, thresholds, confusion_iou, confusion_score
        )
        ap_label = "mAP50_95" if _is_coco_grid(thresholds) else "mAP"
        for part in plan:
            slice_rows[part.label] = _detection_slice(match, part.mask(), ap_label)
        if n_resamples and gt.n_images >= 2:
            raw = bs.bootstrap_detection(
                match, n_resamples=n_resamples, seed=seed, level=level, workers=workers
            )
            coco_grid = thresholds.size == 10 and bool(
                np.allclose(thresholds, m.coco_iou_thresholds())
            )
            rename = {"AP": "mAP50_95" if coco_grid else "mAP", "AP50": "mAP50", "AR100": "AR100"}
            ci = {rename[k]: v for k, v in raw.items() if rename[k] in metrics}
    info = (
        BootstrapInfo(n_resamples=n_resamples, seed=seed, level=level, numpy_version=np.__version__)
        if ci
        else None
    )
    status: CiStatus = "computed" if ci else "off"
    if not ci and n_resamples and gt.n_images < 2:
        status = "not_computed_too_few_images"
    params = EvalParams(
        iou_thresholds=[float(v) for v in thresholds],
        confusion_iou=confusion_iou,
        confusion_score=confusion_score,
    )
    return EvalResult(
        task, digest, names, gt.n_images, metrics, per_class, confusion, slice_rows, ci, info,
        status, params, dict(slices or {}),
    )  # fmt: skip


def _detection(
    gt: GroundTruth,
    pr: Predictions,
    names: list[str],
    thresholds: m.F64,
    confusion_iou: float,
    confusion_score: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], I64, m.MatchResult]:
    n_classes = len(names)
    match = m.match_detections(
        pr.image,
        pr.boxes,
        pr.score,
        pr.cls,
        gt.ann_image,
        gt.ann_boxes,
        gt.ann_cls,
        n_images=gt.n_images,
        n_classes=n_classes,
        iou_thresholds=thresholds,
        iscrowd=gt.ann_crowd,
    )
    curves = m.accumulate(match)
    summary = m.summarize(curves, thresholds)
    coco_grid = _is_coco_grid(thresholds)
    ap_name = "AP50_95" if coco_grid else "AP"
    metrics: dict[str, float] = {}
    if summary["AP"] > -1:
        metrics["mAP50_95" if coco_grid else "mAP"] = summary["AP"]
    for name, key in _DETECTION_HEADLINE:
        if summary.get(key, -1.0) > -1:
            metrics[name] = summary[key]
    support = np.bincount(gt.ann_cls[~gt.ann_crowd], minlength=n_classes)
    ap_all = m.per_class_ap(curves)
    has_50 = bool(np.isclose(thresholds, 0.5).any())
    ap_50 = m.per_class_ap(curves, iou=0.5) if has_50 else None
    recall = m.per_class_recall(curves)
    per_class: dict[str, dict[str, float]] = {}
    for k, name in enumerate(names):
        row = {"support": float(support[k])}
        if ap_all[k] > -1:
            row[ap_name] = float(ap_all[k])
            if ap_50 is not None:
                row["AP50"] = float(ap_50[k])
            row["AR100"] = float(recall[k])
        per_class[name] = row
    confusion = cf.confusion_matrix(
        pr.image,
        pr.boxes,
        pr.score,
        pr.cls,
        gt.ann_image,
        gt.ann_boxes,
        gt.ann_cls,
        n_images=gt.n_images,
        n_classes=n_classes,
        iscrowd=gt.ann_crowd,
        iou=confusion_iou,
        score=confusion_score,
    )
    return metrics, per_class, confusion, match


def _detection_slice(match: m.MatchResult, mask: BoolArray, ap_label: str) -> dict[str, float]:
    """Headline AP of the images in ``mask`` from the shared matching pass."""
    curves = m.accumulate(match, image_weights=mask.astype(np.float64), areas=[0], max_dets=[100])
    block = curves.precision[:, :, :, 0, 0]
    row = {"n_images": float(mask.sum())}
    if bool((block > -1).any()):
        row[ap_label] = float(block[block > -1].mean())
    return row


def _classification(
    gt: GroundTruth, pr: Predictions, names: list[str]
) -> tuple[dict[str, float], dict[str, dict[str, float]], I64, BoolArray]:
    n_classes = len(names)
    true = np.zeros(gt.n_images, dtype=np.int64)
    true[gt.ann_image] = gt.ann_cls
    predicted = np.full(gt.n_images, -1, dtype=np.int64)
    predicted[pr.image] = pr.cls
    result = cf.classification_metrics(true, predicted, n_classes=n_classes)
    present = result.support > 0
    metrics = {
        "top1": result.accuracy,
        "macro_precision": float(result.precision[present].mean()),
        "macro_recall": float(result.recall[present].mean()),
    }
    per_class: dict[str, dict[str, float]] = {}
    for k, name in enumerate(names):
        p, r = float(result.precision[k]), float(result.recall[k])
        per_class[name] = {
            "precision": p,
            "recall": r,
            "f1": 2 * p * r / (p + r) if p + r > 0 else 0.0,
            "support": float(result.support[k]),
        }
    return metrics, per_class, result.confusion, result.correct
