"""Compare two eval snapshots: headline, per-class, slice and confusion deltas.

Units: snapshot metrics are fractions in [0, 1]; every delta here is in *metric points on the
0..100 scale* and signed ``candidate - baseline`` (negative is a regression). ``compare`` works
on two snapshots only: it does no resampling, so it is deterministic and cheap. The bootstrap
intervals shown are the ones stored in the snapshots (their seed and resample count are recorded
there); two intervals that do not overlap are a conservative sign of a real change, overlapping
intervals are not proof of none.

Snapshots are comparable only when ``eval_set_hash``, ``task``, ``class_names`` (order included)
and the evaluation parameters are equal; anything else raises ``CompareRefusedError`` and no
result exists. Missing parameters count as different: what cannot be checked is not comparable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from cvpilot.core.errors import CompareRefusedError, EvalSetMismatchError
from cvpilot.core.schema import EvalSnapshot

Overlap = Literal["overlap", "separated", "unknown"]
ClassStatus = Literal["ok", "missing", "not_applicable", "not_finite"]
SliceStatus = Literal[
    "ok",
    "missing_baseline",
    "missing_candidate",
    "missing_metric",
    "spec_mismatch",
    "size_mismatch",
    "not_finite",
]
PRIMARY_SLICE_METRICS = ("mAP50_95", "mAP", "top1")
"""Slice metric that is compared and gated, first present wins."""
POINTS = 100.0
"""Metric fractions are reported and gated in points on the 0..100 scale."""


@dataclass(frozen=True)
class SnapshotRef:
    run_id: str
    step: int
    eval_set_hash: str
    n_images: int
    task: str
    ci_status: str
    slice_hash: str | None = None
    provenance_commit: str | None = None


@dataclass(frozen=True)
class MetricDelta:
    name: str
    baseline: float | None
    candidate: float | None
    delta_points: float | None
    baseline_ci: tuple[float, float] | None
    candidate_ci: tuple[float, float] | None
    ci_overlap: Overlap


@dataclass(frozen=True)
class ClassDelta:
    name: str
    metric: str
    baseline: float | None
    candidate: float | None
    delta_points: float | None
    support_baseline: float | None
    support_candidate: float | None
    status: ClassStatus


@dataclass(frozen=True)
class SliceDelta:
    label: str
    metric: str
    baseline: float | None
    candidate: float | None
    delta_points: float | None
    n_images_baseline: float | None
    n_images_candidate: float | None
    status: SliceStatus


@dataclass(frozen=True)
class ConfusionCell:
    true_name: str
    pred_name: str
    baseline: int
    candidate: int
    delta: int


@dataclass(frozen=True)
class ConfusionDelta:
    """Changed cells of the ``(C+1) x (C+1)`` matrices; index ``C`` is the background."""

    cells: tuple[ConfusionCell, ...]
    n_changed_cells: int
    missed_delta: int
    false_positive_delta: int
    baseline_total: int
    candidate_total: int


@dataclass(frozen=True)
class Comparison:
    candidate: SnapshotRef
    baseline: SnapshotRef
    headline: tuple[MetricDelta, ...]
    per_class: Mapping[str, tuple[ClassDelta, ...]]
    slices: tuple[SliceDelta, ...]
    confusion: ConfusionDelta | None = None


def _ref(snapshot: EvalSnapshot) -> SnapshotRef:
    return SnapshotRef(
        snapshot.run_id,
        snapshot.step,
        snapshot.eval_set_hash,
        snapshot.n_images,
        snapshot.task,
        snapshot.ci_status,
        snapshot.slice_hash,
        None if snapshot.provenance is None else snapshot.provenance.git_commit,
    )


def _refuse(
    reason: str, message: str, candidate: EvalSnapshot, baseline: EvalSnapshot
) -> CompareRefusedError:
    cls = EvalSetMismatchError if reason == "eval_set_hash" else CompareRefusedError
    return cls(
        message,
        reason=reason,
        candidate_run_id=candidate.run_id,
        baseline_run_id=baseline.run_id,
        candidate_eval_set_hash=candidate.eval_set_hash,
        baseline_eval_set_hash=baseline.eval_set_hash,
    )


def check_comparable(candidate: EvalSnapshot, baseline: EvalSnapshot) -> None:
    """Raise ``CompareRefusedError`` unless the two snapshots measure the same thing."""
    if candidate.eval_set_hash != baseline.eval_set_hash:
        raise _refuse(
            "eval_set_hash",
            "evaluation sets differ: candidate "
            f"{candidate.eval_set_hash[:12]} vs baseline {baseline.eval_set_hash[:12]}",
            candidate,
            baseline,
        )
    if candidate.task != baseline.task:
        raise _refuse(
            "task", f"tasks differ: {candidate.task} vs {baseline.task}", candidate, baseline
        )
    if candidate.class_names != baseline.class_names:
        raise _refuse(
            "class_names",
            f"class names differ ({len(candidate.class_names)} vs {len(baseline.class_names)} "
            "classes, or another order)",
            candidate,
            baseline,
        )
    if candidate.params is None or baseline.params is None:
        raise _refuse(
            "params_missing",
            "a snapshot has no recorded evaluation parameters, so it cannot be compared",
            candidate,
            baseline,
        )
    if candidate.params != baseline.params:
        raise _refuse(
            "params",
            "evaluation parameters (IoU grid, confusion, max_dets) differ",
            candidate,
            baseline,
        )
    if candidate.slice_hash != baseline.slice_hash:
        raise _refuse(
            "slice_hash",
            "the images are sliced differently in the two snapshots (or only one records "
            "its slice membership); evaluate both with the same --slices",
            candidate,
            baseline,
        )
    for label, row in candidate.slices.items():
        other = baseline.slices.get(label)
        if other is not None and row.get("n_images") != other.get("n_images"):
            raise _refuse(
                "slice_n_images",
                "a slice holds a different number of images in the two snapshots",
                candidate,
                baseline,
            )


def _overlap(a: tuple[float, float] | None, b: tuple[float, float] | None) -> Overlap:
    if a is None or b is None:
        return "unknown"
    return "overlap" if a[0] <= b[1] and b[0] <= a[1] else "separated"


def _delta(base: float | None, cand: float | None) -> float | None:
    if base is None or cand is None:
        return None
    return (cand - base) * POINTS


def _finite(*values: float | None) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _headline(candidate: EvalSnapshot, baseline: EvalSnapshot) -> tuple[MetricDelta, ...]:
    names = [*baseline.metrics, *(n for n in candidate.metrics if n not in baseline.metrics)]
    rows = []
    for name in names:
        b, c = baseline.metrics.get(name), candidate.metrics.get(name)
        bi, ci = baseline.ci.get(name), candidate.ci.get(name)
        rows.append(MetricDelta(name, b, c, _delta(b, c), bi, ci, _overlap(bi, ci)))
    return tuple(rows)


def _class_row(
    name: str, metric: str, base: Mapping[str, float], cand: Mapping[str, float]
) -> ClassDelta:
    sb, sc = base.get("support"), cand.get("support")
    b, c = base.get(metric), cand.get(metric)
    if sb == 0.0 and sc == 0.0:
        status: ClassStatus = "not_applicable"
    elif b is None or c is None:
        status = "missing"
    elif not _finite(b, c):
        status = "not_finite"
    else:
        status = "ok"
    return ClassDelta(name, metric, b, c, _delta(b, c), sb, sc, status)


_CLASS_ORDER = {"missing": 0, "not_finite": 0, "ok": 1, "not_applicable": 2}


def _class_key(row: ClassDelta) -> tuple[int, float, str]:
    delta = row.delta_points if row.status == "ok" and row.delta_points is not None else 0.0
    return (_CLASS_ORDER[row.status], delta, row.name)


def _per_class(
    candidate: EvalSnapshot, baseline: EvalSnapshot
) -> dict[str, tuple[ClassDelta, ...]]:
    metrics = sorted(
        {
            m
            for snap in (candidate, baseline)
            for row in snap.per_class.values()
            for m in row
            if m != "support"
        }
    )
    out: dict[str, tuple[ClassDelta, ...]] = {}
    for metric in metrics:
        rows = [
            _class_row(
                name, metric, baseline.per_class.get(name, {}), candidate.per_class.get(name, {})
            )
            for name in baseline.class_names
        ]
        out[metric] = tuple(sorted(rows, key=_class_key))
    return out


def _primary(*rows: Mapping[str, float]) -> str:
    for name in PRIMARY_SLICE_METRICS:
        if any(name in row for row in rows):
            return name
    return ""


_SLICE_ORDER = {
    "spec_mismatch": 0,
    "size_mismatch": 0,
    "missing_metric": 0,
    "not_finite": 0,
    "missing_candidate": 0,
    "missing_baseline": 1,
    "ok": 2,
}


def _slices(candidate: EvalSnapshot, baseline: EvalSnapshot) -> tuple[SliceDelta, ...]:
    rows: list[SliceDelta] = []
    for label in sorted({*baseline.slices, *candidate.slices}):
        b_row, c_row = baseline.slices.get(label), candidate.slices.get(label)
        metric = _primary(b_row or {}, c_row or {})
        b = None if b_row is None else b_row.get(metric)
        c = None if c_row is None else c_row.get(metric)
        nb = None if b_row is None else b_row.get("n_images")
        nc = None if c_row is None else c_row.get("n_images")
        key = label.partition("=")[0]
        status: SliceStatus
        if b_row is None:
            status = "missing_baseline"
        elif c_row is None:
            status = "missing_candidate"
        elif baseline.slice_spec.get(key) != candidate.slice_spec.get(key):
            status = "spec_mismatch"
        elif nb != nc:
            status = "size_mismatch"
        elif b is None or c is None:
            status = "missing_metric"
        elif not _finite(b, c):
            status = "not_finite"
        else:
            status = "ok"
        delta = _delta(b, c) if status in ("ok", "not_finite") else None
        rows.append(SliceDelta(label, metric, b, c, delta, nb, nc, status))

    def order(row: SliceDelta) -> tuple[int, float, str]:
        delta = row.delta_points if row.status == "ok" and row.delta_points is not None else 0.0
        return (_SLICE_ORDER[row.status], delta, row.label)

    return tuple(sorted(rows, key=order))


def compare(
    candidate: EvalSnapshot, baseline: EvalSnapshot, *, confusion: ConfusionDelta | None = None
) -> Comparison:
    """Deltas of ``candidate`` against ``baseline``.

    Raises:
        CompareRefusedError: the snapshots are not comparable (``EvalSetMismatchError`` when
            the evaluation sets differ).
    """
    check_comparable(candidate, baseline)
    return Comparison(
        candidate=_ref(candidate),
        baseline=_ref(baseline),
        headline=_headline(candidate, baseline),
        per_class=_per_class(candidate, baseline),
        slices=_slices(candidate, baseline),
        confusion=confusion,
    )
