"""Detection metrics on NumPy arrays: IoU, COCO-protocol matching, precision-recall, AP and AR.

Pure functions, no I/O and no framework types. The definitions follow the COCO evaluation
protocol so results can be checked against pycocotools (``tests/parity``):

* detections are ranked per (image, class) by score (ties keep input order) and only the best
  ``max_dets`` (100) of each group take part;
* matching is greedy per detection in rank order and per IoU threshold: a ground-truth box is
  taken by at most one detection (crowd boxes can absorb many), non-ignored boxes are preferred
  over ignored ones, and the highest IoU wins (the last of equals);
* crowd boxes measure IoU against the detection area; a box is *ignored* when it is a crowd or
  its area lies outside the area range, and a detection matched to an ignored box, or left
  unmatched outside the area range, counts as neither true nor false positive;
* AP samples the monotone precision envelope at 101 recall points; classes without
  (non-ignored) ground truth stay at -1 and are skipped by every mean.

Differences that are deliberate: boxes are ``x1, y1, x2, y2``; area is the box area (COCO
reads a stored ``area``); across images, tied scores are ordered by image index, which is the
order of sorted image ids. ``accumulate`` also takes per-image weights so that bootstrap
resampling and slices reuse one matching pass: a weight ``w`` counts an image ``w`` times, and
``0`` removes it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

RECALL_POINTS = 101
RECALL_THRESHOLDS: F64 = np.linspace(0.0, 1.0, RECALL_POINTS).astype(np.float64)
MAX_DETS: tuple[int, ...] = (1, 10, 100)
AREA_RANGES: tuple[tuple[str, float, float], ...] = (
    ("all", 0.0, 1e10),
    ("small", 0.0, 32.0**2),
    ("medium", 32.0**2, 96.0**2),
    ("large", 96.0**2, 1e10),
)
"""Inclusive COCO area ranges; a box on a boundary belongs to both neighbours."""

MATCH_BUDGET = 1 << 22
"""Largest padded ``groups x detections x ground-truth`` block matched at once. One float64
block is 32 MB; IoU, intersection, union and the gathered boxes are live together, so the
real peak of a block is roughly 200 MB, plus the match flags of the whole input."""

_EPS = float(np.spacing(1))


def max_int(values: I64) -> int:
    """Largest element of a non-empty integer array as ``int``.

    One typed entry point instead of ``ndarray.max()``: the numpy 2.5 stubs reject the
    argument-less method on integer arrays although it is valid at run time.
    """
    return int(np.max(values))


def coco_iou_thresholds() -> F64:
    """The ten COCO IoU thresholds 0.50, 0.55, ..., 0.95."""
    return np.linspace(0.5, 0.95, 10).astype(np.float64)


def box_areas(boxes: F64) -> F64:
    """Areas of ``(n, 4)`` boxes in ``x1, y1, x2, y2`` form."""
    return np.asarray((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]), dtype=np.float64)


def _pairwise(a: F64, b: F64) -> tuple[F64, F64]:
    """Intersection and area of ``a`` for broadcastable ``(..., 4)`` boxes."""
    iw = np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])
    ih = np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])
    inter: F64 = np.clip(iw, 0.0, None) * np.clip(ih, 0.0, None)
    area = np.asarray((a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1]), dtype=np.float64)
    return inter, area


def iou_matrix(dets: F64, gts: F64, iscrowd: BoolArray | None = None) -> F64:
    """``(n_det, n_gt)`` IoU; against a crowd box it is intersection over the detection area."""
    inter, det_area = _pairwise(dets[:, None, :], gts[None, :, :])
    gt_area = box_areas(gts)[None, :]
    union = det_area + gt_area - inter
    if iscrowd is not None:
        union = np.where(iscrowd[None, :], det_area, union)
    with np.errstate(divide="ignore", invalid="ignore"):
        result: F64 = inter / union
    return result


@dataclass(frozen=True)
class MatchResult:
    """Outcome of matching, for the detections that survived the per-group cap.

    Detections are stored grouped by (image, class) in descending score order; ``matched`` and
    ``ignored`` are indexed ``[area, iou_threshold, detection]``.
    """

    n_images: int
    n_classes: int
    iou_thresholds: F64
    area_ranges: tuple[tuple[str, float, float], ...]
    max_dets: tuple[int, ...]
    det_image: I64
    det_cls: I64
    det_score: F64
    det_rank: I64
    matched: BoolArray
    ignored: BoolArray
    gt_image: I64
    gt_cls: I64
    gt_counts: BoolArray  # [area, gt]: counts towards recall (not crowd, inside the range)
    order: I64  # detection indices sorted by class, -score, image, rank
    bounds: I64  # order[bounds[k]:bounds[k + 1]] are the detections of class k


@dataclass(frozen=True)
class Curves:
    """``precision[T, R, K, A, M]`` and ``recall[T, K, A, M]`` with -1 where undefined."""

    precision: F64
    recall: F64
    iou_thresholds: F64
    area_names: tuple[str, ...]
    max_dets: tuple[int, ...]


def _group_bounds(keys: I64) -> tuple[I64, I64]:
    """Start offsets and sizes of runs of equal values in sorted ``keys``."""
    if keys.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    starts = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]]).astype(np.int64)
    sizes = np.diff(np.r_[starts, keys.size]).astype(np.int64)
    return starts, sizes


def _chunks(dcount: I64, gcount: I64) -> list[I64]:
    """Index blocks of groups whose padded ``B x D x G`` size stays within ``MATCH_BUDGET``."""
    if dcount.size == 0:
        return []
    dpow = np.ceil(np.log2(np.maximum(dcount, 1))).astype(np.int64)
    gpow = np.ceil(np.log2(np.maximum(gcount, 1))).astype(np.int64)
    order = np.lexsort((np.arange(dcount.size), dpow, gpow))
    out: list[I64] = []
    keys = gpow[order] * 64 + dpow[order]
    starts, sizes = _group_bounds(keys)
    for start, size in zip(starts, sizes, strict=True):
        members = order[start : start + size]
        per = max(1, max_int(dcount[members]) * max_int(gcount[members]))
        step = max(1, MATCH_BUDGET // per)
        out.extend(members[i : i + step] for i in range(0, members.size, step))
    return out


def match_detections(
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
    iou_thresholds: F64,
    iscrowd: BoolArray | None = None,
    max_dets: int = MAX_DETS[-1],
    area_ranges: Sequence[tuple[str, float, float]] = AREA_RANGES,
) -> MatchResult:
    """Greedy COCO matching of detections to ground truth, per image, class and IoU threshold.

    Inputs must already be validated (finite, ordered boxes, indices in range); see
    ``cvpilot.eval``. ``pred_image`` and ``gt_image`` are image indices ``< n_images`` and
    classes are indices ``< n_classes``.
    """
    ranges = tuple(area_ranges)
    thresholds = np.asarray(iou_thresholds, dtype=np.float64)
    n_gt = gt_cls.shape[0]
    crowd = np.zeros(n_gt, dtype=bool) if iscrowd is None else iscrowd
    # rank detections inside their (image, class) group, best first, ties in input order
    pkey = pred_image * n_classes + pred_cls
    sort = np.lexsort((-pred_scores, pkey))
    pkey_s = pkey[sort]
    starts, sizes = _group_bounds(pkey_s)
    rank_all = np.arange(pkey_s.size, dtype=np.int64) - np.repeat(starts, sizes)
    keep = sort[rank_all < max_dets]
    rank = rank_all[rank_all < max_dets]
    det_image, det_cls = pred_image[keep], pred_cls[keep]
    det_score, det_boxes = pred_scores[keep], pred_boxes[keep]
    det_key = pkey[keep]
    det_area = box_areas(det_boxes)
    n_det = keep.size

    gsort = np.argsort(gt_image * n_classes + gt_cls, kind="stable")
    gkey = (gt_image * n_classes + gt_cls)[gsort]
    g_boxes, g_crowd = gt_boxes[gsort], crowd[gsort]
    g_area = box_areas(g_boxes)
    n_area, n_thr = len(ranges), thresholds.size
    lows = np.array([r[1] for r in ranges])[:, None]
    highs = np.array([r[2] for r in ranges])[:, None]
    g_ignore = g_crowd[None, :] | (g_area[None, :] < lows) | (g_area[None, :] > highs)
    matched = np.zeros((n_area, n_thr, n_det), dtype=bool)
    ignored = np.broadcast_to(
        ((det_area[None, :] < lows) | (det_area[None, :] > highs))[:, None, :],
        (n_area, n_thr, n_det),
    ).copy()

    dstart, dcount = _group_bounds(det_key)
    ukeys = det_key[dstart] if dstart.size else det_key[:0]
    gstart = np.searchsorted(gkey, ukeys, side="left")
    gcount = np.searchsorted(gkey, ukeys, side="right") - gstart
    has_gt = gcount > 0
    blocks = _chunks(dcount[has_gt], gcount[has_gt])
    group_ids = np.flatnonzero(has_gt)
    eff = np.minimum(thresholds, 1 - 1e-10)
    for block in blocks:
        gi = group_ids[block]
        _match_block(
            gi,
            dstart,
            dcount,
            gstart,
            gcount,
            det_boxes,
            det_area,
            g_boxes,
            g_area,
            g_crowd,
            g_ignore,
            eff,
            matched,
            ignored,
        )

    gt_image_s = gt_image[gsort]
    gt_counts = ~g_ignore
    gt_cls_s = gt_cls[gsort]
    order = np.lexsort((rank, det_image, -det_score, det_cls)).astype(np.int64)
    bounds = np.searchsorted(det_cls[order], np.arange(n_classes + 1)).astype(np.int64)
    return MatchResult(
        n_images=n_images,
        n_classes=n_classes,
        iou_thresholds=thresholds,
        area_ranges=ranges,
        max_dets=MAX_DETS,
        det_image=det_image,
        det_cls=det_cls,
        det_score=det_score,
        det_rank=rank.astype(np.int64),
        matched=matched,
        ignored=ignored,
        gt_image=gt_image_s,
        gt_cls=gt_cls_s,
        gt_counts=gt_counts,
        order=order,
        bounds=bounds,
    )


def _match_block(
    gi: I64,
    dstart: I64,
    dcount: I64,
    gstart: I64,
    gcount: I64,
    det_boxes: F64,
    det_area: F64,
    g_boxes: F64,
    g_area: F64,
    g_crowd: BoolArray,
    g_ignore: BoolArray,
    eff: F64,
    matched: BoolArray,
    ignored: BoolArray,
) -> None:
    """Match one padded block of (image, class) groups; writes into ``matched``/``ignored``."""
    nd, ng = dcount[gi], gcount[gi]
    dmax, gmax = max_int(nd), max_int(ng)
    dpos = dstart[gi][:, None] + np.arange(dmax)[None, :]
    gpos = gstart[gi][:, None] + np.arange(gmax)[None, :]
    dvalid = np.arange(dmax)[None, :] < nd[:, None]
    gvalid = np.arange(gmax)[None, :] < ng[:, None]
    dpos = np.where(dvalid, dpos, 0)
    gpos = np.where(gvalid, gpos, 0)
    gcrowd = g_crowd[gpos] & gvalid
    inter, _ = _pairwise(det_boxes[dpos][:, :, None, :], g_boxes[gpos][:, None, :, :])
    union = det_area[dpos][:, :, None] + g_area[gpos][:, None, :] - inter
    union = np.where(gcrowd[:, None, :], det_area[dpos][:, :, None], union)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = inter / union
    n_thr = eff.size
    for a in range(g_ignore.shape[0]):
        ign = g_ignore[a][gpos] & gvalid
        taken = np.zeros((gi.size, n_thr, gmax), dtype=bool)
        det_hit = np.zeros((gi.size, dmax, n_thr), dtype=bool)
        det_ign = np.zeros((gi.size, dmax, n_thr), dtype=bool)
        for d in range(dmax):
            iou_d = iou[:, d, :]
            elig = (
                gvalid[:, None, :]
                & (~taken | gcrowd[:, None, :])
                & (iou_d[:, None, :] >= eff[None, :, None])
                & dvalid[:, d][:, None, None]
            )
            free = elig & ~ign[:, None, :]
            pool = np.where(free.any(axis=2)[:, :, None], free, elig)
            score = np.where(pool, iou_d[:, None, :], -1.0)
            best = gmax - 1 - np.argmax(score[:, :, ::-1], axis=2)  # last of equal IoUs
            found = pool.any(axis=2)
            b_idx, t_idx = np.nonzero(found)
            g_idx = best[b_idx, t_idx]
            taken[b_idx, t_idx, g_idx] = True
            det_hit[b_idx, d, t_idx] = True
            det_ign[b_idx, d, t_idx] = ign[b_idx, g_idx]
        sel = np.nonzero(dvalid)
        flat = dpos[sel]
        hit = det_hit[sel]
        matched[a][:, flat] = hit.T
        ignored[a][:, flat] = np.where(hit, det_ign[sel], ignored[a][:, flat].T).T


def _class_curves(
    hit: BoolArray, ign: BoolArray, npig: float, weights: F64 | None
) -> tuple[F64, F64]:
    """Interpolated precision at the recall points ``(T, R)`` and final recall ``(T,)``."""
    n_thr, n = hit.shape
    tp_flags = hit & ~ign
    fp_flags = ~hit & ~ign
    if weights is None:
        tp_sum = np.cumsum(tp_flags, axis=1).astype(np.float64)
        fp_sum = np.cumsum(fp_flags, axis=1).astype(np.float64)
    else:
        tp_sum = np.cumsum(tp_flags * weights[None, :], axis=1)
        fp_sum = np.cumsum(fp_flags * weights[None, :], axis=1)
    q = np.zeros((n_thr, RECALL_POINTS))
    final: F64 = np.zeros(n_thr)
    if n == 0:
        return q, final
    rc = tp_sum / npig
    pr = tp_sum / (fp_sum + tp_sum + _EPS)
    pr = np.maximum.accumulate(pr[:, ::-1], axis=1)[:, ::-1]
    final = np.asarray(rc[:, -1], dtype=np.float64)
    for t in range(n_thr):
        idx = np.searchsorted(rc[t], RECALL_THRESHOLDS, side="left")
        ok = idx < n
        q[t, ok] = pr[t, idx[ok]]
    return q, final


def accumulate(
    match: MatchResult,
    *,
    image_weights: F64 | None = None,
    areas: Sequence[int] | None = None,
    max_dets: Sequence[int] | None = None,
) -> Curves:
    """Precision-recall curves per class, area range and detection cap.

    Args:
        match: result of :func:`match_detections`.
        image_weights: optional ``(n_images,)`` non-negative multiplicities; ``None`` is all ones.
        areas: indices into ``match.area_ranges`` to evaluate (default all).
        max_dets: detection caps to evaluate (default ``match.max_dets``).
    """
    area_idx = list(range(len(match.area_ranges))) if areas is None else list(areas)
    caps = list(match.max_dets if max_dets is None else max_dets)
    n_thr, n_cls = match.iou_thresholds.size, match.n_classes
    precision = -np.ones((n_thr, RECALL_POINTS, n_cls, len(area_idx), len(caps)))
    recall = -np.ones((n_thr, n_cls, len(area_idx), len(caps)))
    for ai, a in enumerate(area_idx):
        counts = match.gt_counts[a]
        w_gt = None if image_weights is None else image_weights[match.gt_image[counts]]
        npig = np.bincount(match.gt_cls[counts], weights=w_gt, minlength=n_cls)
        for k in np.flatnonzero(npig > 0):
            idx = match.order[match.bounds[k] : match.bounds[k + 1]]
            hit, ign = match.matched[a][:, idx], match.ignored[a][:, idx]
            rank = match.det_rank[idx]
            w_det = None if image_weights is None else image_weights[match.det_image[idx]]
            for mi, cap in enumerate(caps):
                sel = rank < cap
                if w_det is not None:
                    sel = sel & (w_det > 0)
                q, final = _class_curves(
                    hit[:, sel], ign[:, sel], float(npig[k]), None if w_det is None else w_det[sel]
                )
                precision[:, :, k, ai, mi] = q
                recall[:, k, ai, mi] = final
    return Curves(
        precision=precision,
        recall=recall,
        iou_thresholds=match.iou_thresholds,
        area_names=tuple(match.area_ranges[a][0] for a in area_idx),
        max_dets=tuple(caps),
    )


def _mean_defined(values: F64) -> float:
    defined = values[values > -1]
    return float(defined.mean()) if defined.size else -1.0


def summarize(curves: Curves, iou_thresholds: F64 | None = None) -> dict[str, float]:
    """The twelve COCO summary numbers; -1 where undefined.

    ``AP``, ``AP50``, ``AP75``, ``APs``/``APm``/``APl`` use 100 detections and ``AR1``,
    ``AR10``, ``AR100``, ``ARs``/``ARm``/``ARl`` follow COCO's layout. Entries whose area range
    or detection cap was not accumulated are omitted.
    """
    thr = curves.iou_thresholds if iou_thresholds is None else iou_thresholds

    def ap(iou: float | None, area: str) -> float | None:
        if area not in curves.area_names or 100 not in curves.max_dets:
            return None
        a, mi = curves.area_names.index(area), curves.max_dets.index(100)
        if iou is None:
            return _mean_defined(curves.precision[:, :, :, a, mi])
        hits = np.flatnonzero(np.isclose(thr, iou, atol=1e-9))
        return _mean_defined(curves.precision[hits[0], :, :, a, mi]) if hits.size else None

    def ar(cap: int, area: str) -> float | None:
        if area not in curves.area_names or cap not in curves.max_dets:
            return None
        return _mean_defined(
            curves.recall[:, :, curves.area_names.index(area), curves.max_dets.index(cap)]
        )

    entries = {
        "AP": ap(None, "all"),
        "AP50": ap(0.5, "all"),
        "AP75": ap(0.75, "all"),
        "APs": ap(None, "small"),
        "APm": ap(None, "medium"),
        "APl": ap(None, "large"),
        "AR1": ar(1, "all"),
        "AR10": ar(10, "all"),
        "AR100": ar(100, "all"),
        "ARs": ar(100, "small"),
        "ARm": ar(100, "medium"),
        "ARl": ar(100, "large"),
    }
    return {k: v for k, v in entries.items() if v is not None}


def per_class_ap(curves: Curves, *, iou: float | None = None, area: str = "all") -> F64:
    """AP of every class at ``iou`` (mean over thresholds when ``None``), -1 where undefined."""
    a, mi = curves.area_names.index(area), curves.max_dets.index(100)
    block = curves.precision[:, :, :, a, mi]
    if iou is not None:
        t = int(np.flatnonzero(np.isclose(curves.iou_thresholds, iou, atol=1e-9))[0])
        block = block[t : t + 1]
    out = np.full(block.shape[2], -1.0)
    for k in range(block.shape[2]):
        out[k] = _mean_defined(block[:, :, k])
    return out


def per_class_recall(curves: Curves, *, cap: int = 100, area: str = "all") -> F64:
    """Mean recall over thresholds for every class, -1 where undefined."""
    a, mi = curves.area_names.index(area), curves.max_dets.index(cap)
    block = curves.recall[:, :, a, mi]
    return np.array([_mean_defined(block[:, k]) for k in range(block.shape[1])])
