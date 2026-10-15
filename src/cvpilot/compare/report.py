"""Machine-readable comparison and gate results.

``report_dict`` builds a plain structure and ``report_json`` serialises it canonically (sorted
keys, compact separators, ``allow_nan=False``, no timestamps), so equal inputs give equal
bytes. Non-finite numbers are written as ``null``; the ``status`` fields say why. Names come
from data and are JSON-escaped by the serializer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any

from cvpilot.compare.engine import Comparison
from cvpilot.compare.gate import GateResult, Thresholds

SCHEMA = "cvpilot.compare/1"


def _clean(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if is_dataclass(value) and not isinstance(value, type):
        return _clean(asdict(value))
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    return value


def gate_dict(result: GateResult, thresholds: Thresholds | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {
        "passed": result.passed,
        "n_violations": len(result.violations),
        "n_checks": result.n_checks,
        "violations": _clean(list(result.violations)),
        "skipped_slices": list(result.skipped),
    }
    if thresholds is not None:
        out["thresholds"] = {
            "unit": "points on the 0..100 scale; a drop is (baseline - candidate) * 100",
            "max_drop": dict(sorted(thresholds.max_drop.items())),
            "per_class_max_drop": dict(sorted(thresholds.per_class_max_drop.items())),
            "slice_max_drop": dict(sorted(thresholds.slice_max_drop.items())),
            "min_images": thresholds.min_images,
            "skip_small_slices": thresholds.skip_small_slices,
        }
    return out


def report_dict(
    cmp: Comparison, result: GateResult | None = None, thresholds: Thresholds | None = None
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "unit": "delta_points = (candidate - baseline) * 100, metric points on the 0..100 scale",
        "candidate": _clean(cmp.candidate),
        "baseline": _clean(cmp.baseline),
        "headline": _clean(list(cmp.headline)),
        "per_class": {k: _clean(list(v)) for k, v in sorted(cmp.per_class.items())},
        "slices": _clean(list(cmp.slices)),
        "confusion": _clean(cmp.confusion),
        "gate": None if result is None else gate_dict(result, thresholds),
    }


def report_json(
    cmp: Comparison, result: GateResult | None = None, thresholds: Thresholds | None = None
) -> str:
    return json.dumps(
        report_dict(cmp, result, thresholds),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
