"""The regression gate: thresholds on a :class:`Comparison`, failing closed.

Units and sign: a *drop* is ``(baseline - candidate) * 100`` in metric points on the 0..100
scale (snapshots store fractions, so a drop of 0.5 points is 0.005). A limit is the largest drop
that still passes; a drop equal to the limit passes (drops are rounded to 1e-9 points first, so
float noise cannot fail an exact boundary). An improvement is a negative drop and never fails,
not even against a limit of 0.

Fail closed: the gate passes only when every requested check could be made and held. Anything
absent, non-finite, unmatched or too small is a violation, never a skip: a missing headline or
class metric, ``NaN`` or infinity on either side, a slice absent from either snapshot or
computed with another column, an evaluation set (or a slice) with fewer than ``min_images``
images, a per-class threshold that gated no class, a slice threshold that matched no slice.
Classes without ground truth in both snapshots are not applicable (they have no value to
regress) and are counted, not failed. ``skip_small_slices`` is the one explicit opt-in:
slices below ``min_images`` are then listed in ``GateResult.skipped`` instead of failing, but only
up to ``max_skipped`` of them (default 0, so the flag alone hides nothing), and a threshold whose
slices were all skipped still fails. Slice thresholds also need ``slice_hash`` on both snapshots:
without it slice membership is unverified. At least one threshold is required.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field

from cvpilot.compare.engine import Comparison, MetricDelta, SliceDelta
from cvpilot.compare.text import safe_text
from cvpilot.core.errors import CvpilotError

MAX_LIMIT_POINTS = 100.0
"""A drop is at most 100 points, so a limit of 100 or more could never fail and is refused."""
DEFAULT_MIN_IMAGES = 30
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_NUMBER = re.compile(r"(?:0|[1-9][0-9]{0,2})(?:\.[0-9]{1,6})?")
_ROUND = 9


class ThresholdError(CvpilotError, ValueError):
    """A threshold option is malformed or out of range (a usage error, exit 2)."""


def _number(text: str, what: str) -> float:
    if _NUMBER.fullmatch(text) is None:
        raise ThresholdError(
            f"{what}: expected a number from 0 to 100 such as 0.5, got {safe_text(text, 24)!r}"
        )
    value = float(text)
    if value >= MAX_LIMIT_POINTS:
        raise ThresholdError(
            f"{what}: a limit of {MAX_LIMIT_POINTS:g} points or more can never fail "
            "(a metric cannot drop by more than 100 points)"
        )
    return value


def parse_drop(text: str) -> tuple[str, float]:
    """``NAME=POINTS`` of ``--max-drop`` and ``--per-class-max-drop``."""
    name, sep, number = text.partition("=")
    if not sep or _NAME.fullmatch(name) is None:
        raise ThresholdError(
            f"expected NAME=POINTS such as mAP50_95=0.5, got {safe_text(text, 40)!r}"
        )
    return name, _number(number, name)


def _selector(selector: str) -> str:
    key, sep, value = selector.partition("=")
    if not sep or _NAME.fullmatch(key) is None:
        raise ThresholdError(
            f"slice selector {safe_text(selector, 40)!r} must be KEY=VALUE or KEY=*"
        )
    if value != "*" and (
        not value
        or len(value) > 128
        or "*" in value  # no globbing: only the whole value ``*`` is a wildcard
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
    ):
        raise ThresholdError(f"slice selector {safe_text(selector, 40)!r} has an invalid value")
    return selector


def parse_slice_drop(text: str) -> tuple[str, float]:
    """``KEY=VALUE:POINTS`` or ``KEY=*:POINTS`` of ``--slice-max-drop``, split at the last ``:``."""
    selector, sep, number = text.rpartition(":")
    if not sep:
        raise ThresholdError(
            f"expected KEY=VALUE:POINTS such as camera=*:1.5, got {safe_text(text, 40)!r}"
        )
    return _selector(selector), _number(number, selector)


def _check_limits(limits: Mapping[str, float], what: str) -> None:
    for name, value in limits.items():
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value < MAX_LIMIT_POINTS
        ):
            raise ThresholdError(
                f"{what} limit for {safe_text(name, 40)!r} must be a finite number from 0 up to "
                "(not including) 100"
            )


@dataclass(frozen=True)
class Thresholds:
    """Largest allowed drops in points; see the module docstring for units and fail-closed rules."""

    max_drop: Mapping[str, float] = field(default_factory=dict)
    per_class_max_drop: Mapping[str, float] = field(default_factory=dict)
    slice_max_drop: Mapping[str, float] = field(default_factory=dict)
    min_images: int = DEFAULT_MIN_IMAGES
    skip_small_slices: bool = False
    max_skipped: int = 0
    """Most slices ``skip_small_slices`` may skip before the gate fails (default 0: any skip
    fails, so the flag alone cannot hide a regressed small slice)."""

    def __post_init__(self) -> None:
        if not (self.max_drop or self.per_class_max_drop or self.slice_max_drop):
            raise ThresholdError(
                "at least one threshold is required; a gate with none checks nothing"
            )
        for names in (self.max_drop, self.per_class_max_drop):
            for name in names:
                if _NAME.fullmatch(name) is None:
                    raise ThresholdError(f"invalid metric name {safe_text(name, 40)!r}")
        for selector in self.slice_max_drop:
            _selector(selector)
        _check_limits(self.max_drop, "max-drop")
        _check_limits(self.per_class_max_drop, "per-class-max-drop")
        _check_limits(self.slice_max_drop, "slice-max-drop")
        if (
            isinstance(self.min_images, bool)
            or not isinstance(self.min_images, int)
            or self.min_images < 1
        ):
            raise ThresholdError("min_images must be an integer of at least 1")
        if (
            isinstance(self.max_skipped, bool)
            or not isinstance(self.max_skipped, int)
            or self.max_skipped < 0
        ):
            raise ThresholdError("max_skipped must be a non-negative integer")


@dataclass(frozen=True)
class Violation:
    kind: str
    subject: str
    message: str
    drop_points: float | None = None
    limit_points: float | None = None

    @property
    def severity(self) -> float:
        """Sort key: unverifiable checks first, then the larger drop."""
        return math.inf if self.drop_points is None else self.drop_points


@dataclass(frozen=True)
class GateResult:
    passed: bool
    violations: tuple[Violation, ...]
    skipped: tuple[str, ...] = ()
    n_checks: int = 0

    @property
    def worst(self) -> Violation | None:
        return self.violations[0] if self.violations else None


def _drop(delta_points: float) -> float:
    return round(-delta_points, _ROUND)


class _Checker:
    def __init__(self, cmp: Comparison, thresholds: Thresholds) -> None:
        self.cmp = cmp
        self.th = thresholds
        self.violations: list[Violation] = []
        self.skipped: list[str] = []
        self.n_checks = 0

    def fail(
        self,
        kind: str,
        subject: str,
        message: str,
        drop: float | None = None,
        limit: float | None = None,
    ) -> None:
        self.violations.append(Violation(kind, safe_text(subject), message, drop, limit))

    def _judge(
        self, kind: str, subject: str, label: str, delta: float | None, limit: float
    ) -> None:
        self.n_checks += 1
        if delta is None or not math.isfinite(delta):
            self.fail(
                f"{kind}_not_finite", subject, f"{label} is not a finite number on both sides"
            )
            return
        drop = _drop(delta)
        if drop > round(limit, _ROUND):
            self.fail(
                f"{kind}_drop",
                subject,
                f"{label} dropped {drop:.2f} points (limit {limit:g})",
                drop,
                limit,
            )

    # -- checks ---------------------------------------------------------------------------

    def images(self) -> None:
        low = min(self.cmp.candidate.n_images, self.cmp.baseline.n_images)
        if low < self.th.min_images:
            self.fail(
                "too_few_images",
                "evaluation set",
                f"the evaluation set has {low} images, fewer than "
                f"--min-images {self.th.min_images}",
            )

    def headline(self) -> None:
        by: dict[str, MetricDelta] = {d.name: d for d in self.cmp.headline}
        for name, limit in self.th.max_drop.items():
            row = by.get(name)
            if row is None or row.baseline is None or row.candidate is None:
                self.n_checks += 1
                self.fail("headline_missing", name, f"metric {name} is absent from a snapshot")
            else:
                self._judge("headline", name, name, row.delta_points, limit)

    def per_class(self) -> None:
        for metric, limit in self.th.per_class_max_drop.items():
            rows = self.cmp.per_class.get(metric, ())
            gated = 0
            for row in rows:
                if row.status == "not_applicable":
                    continue
                gated += 1
                if row.status == "missing":
                    self.n_checks += 1
                    self.fail(
                        "class_missing", row.name, f"{metric} of class is absent from a snapshot"
                    )
                else:
                    self._judge("class", row.name, f"{metric} of class", row.delta_points, limit)
            if gated == 0:
                self.n_checks += 1
                self.fail("no_class_checked", metric, f"no class had a {metric} value to gate")

    def _slice_rows(self, selector: str) -> list[SliceDelta]:
        key, _, value = selector.partition("=")
        return [
            row
            for row in self.cmp.slices
            if row.label.partition("=")[0] == key and (value == "*" or row.label == selector)
        ]

    def slices(self) -> None:
        if self.th.slice_max_drop and (
            self.cmp.candidate.slice_hash is None or self.cmp.baseline.slice_hash is None
        ):
            self.n_checks += 1
            self.fail(
                "slice_hash_missing",
                "slices",
                "a snapshot does not record its slice membership, so slice results cannot be "
                "trusted; evaluate again with --slices",
            )
            return
        for selector, limit in self.th.slice_max_drop.items():
            rows = self._slice_rows(selector)
            if not rows:
                kind = "slice_no_match" if selector.endswith("=*") else "slice_missing"
                self.n_checks += 1
                self.fail(
                    kind,
                    selector,
                    f"no slice matches {safe_text(selector, 64)!r} in either snapshot",
                )
                continue
            gated = 0
            for row in rows:
                if row.status == "size_mismatch":
                    self.n_checks += 1
                    self.fail(
                        "slice_size_mismatch",
                        row.label,
                        "the slice holds different images per side",
                    )
                elif row.status == "spec_mismatch":
                    self.n_checks += 1
                    self.fail(
                        "slice_spec_mismatch", row.label, "the slice was built from another column"
                    )
                elif row.status in ("missing_baseline", "missing_candidate", "missing_metric"):
                    self.n_checks += 1
                    self.fail(
                        "slice_missing",
                        row.label,
                        f"slice is absent or has no metric ({row.status})",
                    )
                else:
                    sizes = (row.n_images_baseline, row.n_images_candidate)
                    if any(n is None or n < self.th.min_images for n in sizes):
                        if self.th.skip_small_slices:
                            self.skipped.append(safe_text(row.label))
                        else:
                            self.n_checks += 1
                            self.fail(
                                "slice_too_small",
                                row.label,
                                f"slice has fewer than --min-images {self.th.min_images} images",
                            )
                        continue
                    gated += 1
                    self._judge(
                        "slice", row.label, f"{row.metric} of slice", row.delta_points, limit
                    )
            if self.th.skip_small_slices and gated == 0 and self.skipped:
                self.n_checks += 1
                self.fail(
                    "slice_none_gated", selector, "every matching slice was skipped as too small"
                )


def gate(cmp: Comparison, thresholds: Thresholds) -> GateResult:
    """Evaluate ``thresholds`` on ``cmp``; the result passes only if nothing could fail silently."""
    checker = _Checker(cmp, thresholds)
    checker.images()
    checker.headline()
    checker.per_class()
    checker.slices()
    if len(set(checker.skipped)) > thresholds.max_skipped:
        checker.n_checks += 1
        checker.fail(
            "too_many_skipped_slices",
            "slices",
            f"{len(set(checker.skipped))} slices were skipped as too small, more than "
            f"--max-skipped {thresholds.max_skipped}",
        )
    ordered = tuple(sorted(checker.violations, key=lambda v: (-v.severity, v.subject, v.kind)))
    return GateResult(not ordered, ordered, tuple(sorted(set(checker.skipped))), checker.n_checks)
