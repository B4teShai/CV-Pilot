"""Bootstrap confidence intervals over images, seeded and bounded.

Resampling draws images with replacement. For detection the matching of detections to ground
truth is per image and independent of the other images, so one matching pass serves every
resample: a resample is a vector of image multiplicities handed to ``accumulate`` as weights
(the same code path that evaluates slices). Up to the order in which tied scores from different
images are ranked, this equals evaluating the resampled set, at a fraction of the cost.

Determinism: the multiplicities of all resamples come from one ``numpy`` generator seeded with
``seed`` and drawn in the calling thread in a fixed order; worker threads only evaluate
already-drawn resamples, so the result does not depend on the number of workers. Memory is
bounded by ``MAX_RESAMPLES`` and by drawing at most ``BATCH_CELLS`` indices at a time.
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from cvpilot.core import metrics as m
from cvpilot.core.errors import EvalInputError
from cvpilot.core.metrics import F64, I64
from cvpilot.core.schema import MAX_RESAMPLES

BATCH_CELLS = 1 << 22
"""Most image indices drawn in one batch (about 32 MB of int64)."""
_CLASS_STRIDE = 1e12
"""Whole-number offset separating the classes in the global ``searchsorted`` (exact in float64)."""
DEFAULT_WORKERS = 4
"""Threads used for detection resamples; numpy releases the GIL in the heavy array calls."""


def check_seed(seed: int) -> None:
    """Raise :class:`EvalInputError` unless ``seed`` is a non-bool int in ``[0, 2**64)``."""
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise EvalInputError("seed must be an integer in [0, 2**64)")


def check_settings(n_resamples: int, level: float, seed: int = 0) -> None:
    """Raise :class:`EvalInputError` unless ``n_resamples``, ``level`` and ``seed`` are valid."""
    check_seed(seed)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int):
        raise EvalInputError("n_resamples must be an integer")
    if not 1 <= n_resamples <= MAX_RESAMPLES:
        raise EvalInputError(
            f"n_resamples must be between 1 and {MAX_RESAMPLES}, got {n_resamples}"
        )
    if not (isinstance(level, float) and 0.0 < level < 1.0):
        raise EvalInputError("level must be a float strictly between 0 and 1")


def _interval(samples: F64, level: float) -> tuple[float, float]:
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)


def bootstrap_ci(
    per_image: F64, *, n_resamples: int = 1000, seed: int = 0, level: float = 0.95
) -> tuple[float, float]:
    """Percentile interval of the mean of ``per_image`` over image resamples.

    Args:
        per_image: finite one-dimensional array, one value per image (for example correctness).
        n_resamples: number of resamples, ``1..MAX_RESAMPLES``.
        seed: seed of the generator; the same seed gives the same interval.
        level: coverage, strictly between 0 and 1.

    Raises:
        EvalInputError: bad shape, non-finite values or settings out of range.
    """
    check_settings(n_resamples, level, seed)
    values = np.asarray(per_image, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not bool(np.isfinite(values).all()):
        raise EvalInputError("per_image must be a non-empty finite one-dimensional array")
    rng = np.random.default_rng(seed)
    n = values.size
    rows = max(1, BATCH_CELLS // n)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, rows):
        count = min(rows, n_resamples - start)
        means[start : start + count] = values[rng.integers(0, n, size=(count, n))].mean(axis=1)
    return _interval(means, level)


def _weights(n_images: int, n_resamples: int, seed: int) -> Iterator[F64]:
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield np.bincount(rng.integers(0, n_images, n_images), minlength=n_images).astype(
            np.float64
        )


def _mean_or_nan(values: F64) -> float:
    return float(values.mean()) if values.size else math.nan


class _Resampler:
    """Weighted AP, AP at IoU 0.5 and AR100 of image resamples over precomputed arrays.

    Only true-positive detections move recall, and the monotone precision envelope is needed at
    those positions only, so a resample works on the (few) true positives of each IoU
    threshold instead of on every detection. Everything independent of the weights is prepared
    once for the area range ``all`` and 100 detections: detections sorted by class and rank,
    the positions of the true positives per threshold and their class segments.

    A resample then needs, per threshold, a cumulative sum over its true positives, one
    cumulative sum of the non-ignored detection weights (shared by all thresholds unless
    crowd regions make the ignore flags differ), a segmented cumulative maximum for the
    envelope and one ``searchsorted`` for all classes and recall points. The recall comparison
    is exact: counts and class sizes are integer-valued, so the smallest count that reaches each
    recall point is found with the same division COCO uses, and the per-class offset added for
    the global ``searchsorted`` is a whole number. Results agree with
    :func:`cvpilot.core.metrics.accumulate` to about 1e-12.
    """

    def __init__(self, match: m.MatchResult) -> None:
        order = match.order
        self.n_classes = k = match.n_classes
        self.n_thr = match.iou_thresholds.size
        self.det_image = match.det_image[order]
        hit = match.matched[0][:, order]
        ign = match.ignored[0][:, order]
        self.nonign = ~ign
        self.shared_ign = bool((ign == ign[0]).all()) if ign.size else True
        self.det_start = match.bounds[:-1]
        self.gt_cls = match.gt_cls[match.gt_counts[0]]
        self.gt_image = match.gt_image[match.gt_counts[0]]
        self.has_50 = np.flatnonzero(np.isclose(match.iou_thresholds, 0.5))
        det_cls = match.det_cls[order]
        self.tp_pos: list[I64] = []
        self.tp_len: list[I64] = []
        self.tp_end: list[I64] = []
        self.tp_start: list[I64] = []
        self.tp_cls_off: list[F64] = []
        self.tp_env_off: list[F64] = []
        classes = np.arange(k)
        for t in range(self.n_thr):
            pos = np.flatnonzero(hit[t] & ~ign[t]).astype(np.int64)
            counts = np.bincount(det_cls[pos], minlength=k).astype(np.int64)
            self.tp_pos.append(pos)
            self.tp_len.append(counts)
            self.tp_end.append(np.cumsum(counts))
            self.tp_start.append(np.cumsum(counts) - counts)
            self.tp_cls_off.append(np.repeat(_CLASS_STRIDE * classes.astype(np.float64), counts))
            self.tp_env_off.append(np.repeat(2.0 * (k - 1 - classes).astype(np.float64), counts))
        self.class_off = (_CLASS_STRIDE * classes.astype(np.float64))[:, None]

    def _targets(self, npig: F64) -> F64:
        """Smallest TP count per class and recall point with ``count / npig >= recall``."""
        size = np.maximum(npig, 1.0)[:, None]
        rec = m.RECALL_THRESHOLDS[None, :]
        count = np.ceil(rec * size)
        count = np.where(count / size < rec, count + 1.0, count)
        count = np.where((count - 1.0) / size >= rec, count - 1.0, count)
        result: F64 = count
        return result

    def run(self, weights: F64) -> tuple[float, float, float]:
        k = self.n_classes
        npig = np.bincount(self.gt_cls, weights=weights[self.gt_image], minlength=k).astype(
            np.float64
        )
        valid = npig > 0
        if not bool(valid.any()):
            return math.nan, math.nan, math.nan
        w = weights[self.det_image]
        wide_start = np.minimum(self.det_start, max(w.shape[0] - 1, 0))
        shared_cnt = np.cumsum(w * self.nonign[0]) if self.shared_ign else None
        queries = (self._targets(npig) + self.class_off).reshape(-1)
        ap = np.zeros((self.n_thr, k))
        final = np.zeros((self.n_thr, k))
        for t in range(self.n_thr):
            pos, length, end = self.tp_pos[t], self.tp_len[t], self.tp_end[t]
            if pos.size == 0:
                continue
            cnt = shared_cnt if shared_cnt is not None else np.cumsum(w * self.nonign[t])
            cum = np.cumsum(w[pos])
            tp = cum - np.repeat(np.concatenate(([0.0], cum))[self.tp_start[t]], length)
            before = np.where(self.det_start > 0, cnt[wide_start - 1], 0.0)
            seen = cnt[pos] - np.repeat(before, length)
            pr = tp / (seen + m._EPS)  # tp / (fp + tp + eps), as seen = tp + fp
            pr += self.tp_env_off[t]
            env = np.maximum.accumulate(pr[::-1])[::-1] - self.tp_env_off[t]
            idx = np.searchsorted(tp + self.tp_cls_off[t], queries, side="left").reshape(k, -1)
            inside = idx < end[:, None]
            picked = np.where(inside, env[np.minimum(idx, pos.size - 1)], 0.0)
            ap[t] = picked.mean(axis=1)
            final[t] = np.where(length > 0, tp[np.maximum(end - 1, 0)] / np.maximum(npig, 1.0), 0.0)
        ap_valid = ap[:, valid]
        ap50 = float(ap_valid[self.has_50[0]].mean()) if self.has_50.size else math.nan
        return float(ap_valid.mean()), ap50, float(final[:, valid].mean())


def bootstrap_detection(
    match: m.MatchResult,
    *,
    n_resamples: int,
    seed: int,
    level: float,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, tuple[float, float]]:
    """Intervals of ``AP`` (mean over IoU thresholds), ``AP50`` and ``AR100`` over images.

    Keys are the core names ``AP``, ``AP50`` (only when 0.5 is among the thresholds) and
    ``AR100``. Resamples where a metric is undefined are left out of its interval.
    """
    check_settings(n_resamples, level, seed)
    workers = max(1, min(int(workers), os.cpu_count() or 1, 32))
    rows: list[tuple[float, float, float]] = []
    resampler = _Resampler(match)
    stream = _weights(match.n_images, n_resamples, seed)
    if workers == 1:
        rows.extend(resampler.run(w) for w in stream)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                batch = [w for _, w in zip(range(workers * 4), stream, strict=False)]
                if not batch:
                    break
                rows.extend(pool.map(resampler.run, batch))
    table = np.array(rows, dtype=np.float64)
    out: dict[str, tuple[float, float]] = {}
    for column, name in enumerate(("AP", "AP50", "AR100")):
        values = table[:, column]
        values = values[np.isfinite(values)]
        if values.size:
            out[name] = _interval(values, level)
    return out
