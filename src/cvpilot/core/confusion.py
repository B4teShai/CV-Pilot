"""Confusion matrices with a background class, and classification metrics.

Detection confusion is ``(C + 1) x (C + 1)``: rows are the true class and columns the predicted
class, with index ``C`` meaning *background*. A ground-truth box that no detection claims is
counted in ``[true, C]`` (missed); a detection that claims no box is counted in ``[C, pred]``
(spurious). Matching is class-agnostic so a wrong-class detection shows up off the diagonal:
candidate pairs with IoU >= ``iou`` are taken greedily from the highest IoU down, each box and
detection once. Detections below ``score`` are dropped before matching; detections that sit on
a crowd region (IoU against the detection area >= ``iou``) and match nothing are not spurious.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cvpilot.core.metrics import F64, I64, BoolArray, iou_matrix

CONFUSION_MAX_DETS = 300
"""Detections per image that take part in the confusion matrix (highest scores first)."""
CONFUSION_CELLS = 1 << 22
"""Largest ``detections x boxes`` IoU block computed at once (about 32 MB)."""
CONFUSION_CANDIDATES = 8
"""Ground-truth candidates kept per detection; more than 8 boxes competing for one detection is
rare, and the greedy matching only ever uses the best ones."""


def _split(sorted_image: I64, n_images: int) -> I64:
    return np.searchsorted(sorted_image, np.arange(n_images + 1)).astype(np.int64)


def _candidates(
    dets: F64, gts: F64, crowd: BoolArray, iou: float
) -> tuple[I64, I64, F64, BoolArray]:
    """Best matching candidates (per detection, at most ``CONFUSION_CANDIDATES`` boxes).

    Returns detection indices, box indices and IoUs of pairs with IoU >= ``iou`` against
    regular boxes, and a mask of detections lying on a crowd region. The IoU matrix is
    computed in blocks of at most ``CONFUSION_CELLS`` cells.
    """
    n_det, k = dets.shape[0], CONFUSION_CANDIDATES
    best: F64 = np.zeros((n_det, k))
    best_idx: I64 = np.full((n_det, k), -1, dtype=np.int64)
    on_crowd = np.zeros(n_det, dtype=bool)
    step = max(1, CONFUSION_CELLS // max(n_det, 1))
    for lo in range(0, gts.shape[0], step):
        block, flags = gts[lo : lo + step], crowd[lo : lo + step]
        ious = iou_matrix(dets, block)
        ious[:, flags] = 0.0
        if flags.any():
            ioa = iou_matrix(dets, block, flags)[:, flags]
            on_crowd |= (ioa >= iou).any(axis=1)
        ids = np.broadcast_to(np.arange(lo, lo + block.shape[0]), ious.shape)
        merged = np.concatenate([best, np.where(ious >= iou, ious, 0.0)], axis=1)
        merged_ids = np.concatenate([best_idx, ids], axis=1)
        keep = np.argpartition(-merged, min(k, merged.shape[1] - 1), axis=1)[:, :k]
        best = np.take_along_axis(merged, keep, axis=1)
        best_idx = np.take_along_axis(merged_ids, keep, axis=1)
    rows, cols = np.nonzero(best > 0)
    return rows.astype(np.int64), best_idx[rows, cols], best[rows, cols], on_crowd


def confusion_matrix(
    pred_image: I64,
    pred_boxes: F64,
    pred_scores: F64,
    pred_cls: I64,
    gt_image: I64,
    gt_boxes: F64,
    gt_cls: I64,
    *,
    n_images: int,
    n_classes: int,
    iscrowd: BoolArray | None = None,
    iou: float = 0.5,
    score: float = 0.25,
) -> I64:
    """``(C + 1, C + 1)`` detection confusion counts; see the module docstring.

    Only the ``CONFUSION_MAX_DETS`` best detections of each image are considered, and the
    matching keeps ``CONFUSION_CANDIDATES`` candidates per detection, so memory and time stay
    bounded on crowded images.
    """
    crowd = np.zeros(gt_cls.shape[0], dtype=bool) if iscrowd is None else iscrowd
    keep = pred_scores >= score
    pred_image, pred_boxes, pred_scores, pred_cls = (
        pred_image[keep],
        pred_boxes[keep],
        pred_scores[keep],
        pred_cls[keep],
    )
    psort = np.lexsort((-pred_scores, pred_image))
    pred_image, pred_boxes, pred_cls = pred_image[psort], pred_boxes[psort], pred_cls[psort]
    if pred_image.size:  # keep the best CONFUSION_MAX_DETS of every image
        first = np.searchsorted(pred_image, pred_image, side="left")
        under = (np.arange(pred_image.size) - first) < CONFUSION_MAX_DETS
        pred_image, pred_boxes, pred_cls = pred_image[under], pred_boxes[under], pred_cls[under]
    gsort = np.argsort(gt_image, kind="stable")
    gt_image, gt_boxes = gt_image[gsort], gt_boxes[gsort]
    gt_cls, crowd = gt_cls[gsort], crowd[gsort]
    pb, gb = _split(pred_image, n_images), _split(gt_image, n_images)
    side = n_classes + 1
    rows: list[I64] = []
    cols: list[I64] = []
    for img in range(n_images):
        p0, p1, g0, g1 = pb[img], pb[img + 1], gb[img], gb[img + 1]
        if p0 == p1 and g0 == g1:
            continue
        gcrowd = crowd[g0:g1]
        d_cls, g_cls = pred_cls[p0:p1], gt_cls[g0:g1]
        d_used = np.zeros(p1 - p0, dtype=bool)
        g_used = gcrowd.copy()  # crowd boxes never take part in one-to-one matching
        if p1 > p0 and g1 > g0:
            di, gi, ious, on_crowd = _candidates(pred_boxes[p0:p1], gt_boxes[g0:g1], gcrowd, iou)
            order = np.argsort(-ious, kind="stable")
            for d, g in zip(di[order].tolist(), gi[order].tolist(), strict=True):
                if d_used[d] or g_used[g]:
                    continue
                d_used[d] = g_used[g] = True
                rows.append(np.array([g_cls[g]]))
                cols.append(np.array([d_cls[d]]))
            d_used |= on_crowd
        missed = g_cls[~g_used]
        spurious = d_cls[~d_used]
        rows.append(missed)
        cols.append(np.full(missed.size, n_classes, dtype=np.int64))
        rows.append(np.full(spurious.size, n_classes, dtype=np.int64))
        cols.append(spurious)
    flat: I64 = np.zeros(side * side, dtype=np.int64)
    if rows:
        key = np.concatenate(rows).astype(np.int64) * side + np.concatenate(cols).astype(np.int64)
        flat = np.bincount(key, minlength=side * side).astype(np.int64)
    return flat.reshape(side, side)


@dataclass(frozen=True)
class ClassificationResult:
    """Top-1 classification scores; ``confusion`` has the background column for no prediction."""

    accuracy: float
    precision: F64
    recall: F64
    support: I64
    confusion: I64
    correct: BoolArray


def classification_metrics(true_cls: I64, pred_cls: I64, *, n_classes: int) -> ClassificationResult:
    """Accuracy, per-class precision and recall, and the confusion matrix.

    ``pred_cls == -1`` means the image has no prediction (counted wrong, column ``C``).
    Precision of a class never predicted, and recall of a class without examples, is 0.
    """
    side = n_classes + 1
    cols = np.where(pred_cls < 0, n_classes, pred_cls)
    confusion = np.bincount(true_cls * side + cols, minlength=side * side).reshape(side, side)
    correct = true_cls == pred_cls
    support = confusion[:n_classes].sum(axis=1)
    predicted = confusion[:n_classes, :n_classes].sum(axis=0)
    hits = np.diag(confusion)[:n_classes].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, hits / np.maximum(predicted, 1), 0.0)
        recall = np.where(support > 0, hits / np.maximum(support, 1), 0.0)
    accuracy = float(correct.mean()) if correct.size else 0.0
    return ClassificationResult(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        support=support.astype(np.int64),
        confusion=confusion.astype(np.int64),
        correct=correct,
    )
