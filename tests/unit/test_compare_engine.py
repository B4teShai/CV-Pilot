"""``compare``: deltas, interval overlap, worst-first classes, slices and the refusal rules."""

from __future__ import annotations

import math
from typing import Any

import pytest
from support import H1, H2, PARAMS, build_snapshot

from cvpilot.compare.engine import compare
from cvpilot.core.errors import CompareRefusedError, EvalSetMismatchError
from cvpilot.core.schema import EvalSnapshot


def pair(base: dict[str, Any] | None = None, cand: dict[str, Any] | None = None) -> Any:
    b = build_snapshot(**(base or {}))
    c = build_snapshot(**(cand or {}))
    return compare(c, b)


def test_headline_deltas_are_in_points_and_signed_candidate_minus_baseline() -> None:
    cmp = pair(
        {"metrics": {"mAP50_95": 0.600, "mAP50": 0.800}},
        {"metrics": {"mAP50_95": 0.594, "mAP50": 0.81}},
    )
    by = {d.name: d for d in cmp.headline}
    assert by["mAP50_95"].delta_points == pytest.approx(-0.6)
    assert by["mAP50"].delta_points == pytest.approx(1.0)
    assert by["mAP50_95"].baseline == 0.6 and by["mAP50_95"].candidate == 0.594


def test_metric_only_on_one_side_is_listed_with_a_missing_side() -> None:
    cmp = pair({"metrics": {"mAP50_95": 0.6, "mAP_small": 0.3}}, {"metrics": {"mAP50_95": 0.6}})
    by = {d.name: d for d in cmp.headline}
    assert by["mAP_small"].candidate is None and by["mAP_small"].delta_points is None


def test_interval_overlap_is_reported() -> None:
    cmp = pair(
        {"metrics": {"mAP50_95": 0.6}, "ci": {"mAP50_95": (0.55, 0.65)}},
        {"metrics": {"mAP50_95": 0.5}, "ci": {"mAP50_95": (0.45, 0.52)}},
    )
    assert cmp.headline[0].ci_overlap == "separated"
    cmp = pair(
        {"metrics": {"mAP50_95": 0.6}, "ci": {"mAP50_95": (0.55, 0.65)}},
        {"metrics": {"mAP50_95": 0.58}, "ci": {"mAP50_95": (0.5, 0.62)}},
    )
    assert cmp.headline[0].ci_overlap == "overlap"
    assert pair().headline[0].ci_overlap == "unknown"


def test_per_class_deltas_are_sorted_worst_first_with_support() -> None:
    base = {
        n: {"AP50_95": v, "support": float(s)}
        for n, v, s in (("cat", 0.5, 10), ("dog", 0.5, 20), ("bird", 0.5, 5))
    }
    cand = {
        n: {"AP50_95": v, "support": float(s)}
        for n, v, s in (("cat", 0.6, 10), ("dog", 0.3, 20), ("bird", 0.5, 5))
    }
    cmp = pair({"per_class": base}, {"per_class": cand})
    rows = cmp.per_class["AP50_95"]
    assert [r.name for r in rows] == ["dog", "bird", "cat"]
    assert rows[0].delta_points == pytest.approx(-20.0) and rows[0].support_baseline == 20.0
    assert rows[2].delta_points == pytest.approx(10.0)


def test_class_without_ground_truth_is_not_applicable_and_a_lost_metric_is_missing() -> None:
    base = {
        "cat": {"AP50_95": 0.5, "support": 3.0},
        "dog": {"support": 0.0},
        "bird": {"AP50_95": 0.4, "support": 2.0},
    }
    cand = {
        "cat": {"AP50_95": 0.5, "support": 3.0},
        "dog": {"support": 0.0},
        "bird": {"support": 2.0},
    }
    rows = {r.name: r for r in pair({"per_class": base}, {"per_class": cand}).per_class["AP50_95"]}
    assert rows["dog"].status == "not_applicable"
    assert rows["bird"].status == "missing" and rows["bird"].delta_points is None
    assert rows["cat"].status == "ok"


def test_slice_deltas_and_absent_slices() -> None:
    base = {
        "camera=c1": {"mAP50_95": 0.6, "n_images": 40.0},
        "camera=c2": {"mAP50_95": 0.5, "n_images": 10.0},
    }
    cand = {
        "camera=c1": {"mAP50_95": 0.55, "n_images": 40.0},
        "camera=c3": {"mAP50_95": 0.5, "n_images": 7.0},
    }
    spec = {"camera": "camera"}
    cmp = pair({"slices": base, "slice_spec": spec}, {"slices": cand, "slice_spec": spec})
    rows = {r.label: r for r in cmp.slices}
    assert (
        rows["camera=c1"].delta_points == pytest.approx(-5.0) and rows["camera=c1"].status == "ok"
    )
    assert rows["camera=c2"].status == "missing_candidate"
    assert rows["camera=c3"].status == "missing_baseline"


def test_slice_with_another_column_is_flagged_not_compared() -> None:
    row = {"camera=c1": {"mAP50_95": 0.6, "n_images": 40.0}}
    cmp = pair(
        {"slices": row, "slice_spec": {"camera": "camera"}},
        {"slices": row, "slice_spec": {"camera": "cam_id"}},
    )
    assert cmp.slices[0].status == "spec_mismatch" and cmp.slices[0].delta_points is None


@pytest.mark.parametrize(
    ("cand", "reason"),
    [
        ({"h": H2}, "eval_set_hash"),
        (
            {
                "task": "classification",
                "metrics": {"top1": 0.5},
                "per_class": {
                    n: {"precision": 0.5, "support": 1.0} for n in ("cat", "dog", "bird")
                },
            },
            "task",
        ),
        ({"class_names": ["cat", "dog", "fish"], "per_class": {}}, "class_names"),
        ({"class_names": ["dog", "cat", "bird"], "per_class": {}}, "class_names"),
        ({"params": {**PARAMS, "confusion_iou": 0.6}}, "params"),
        ({"params": {**PARAMS, "iou_thresholds": [0.5, 0.75]}}, "params"),
        ({"params": {**PARAMS, "max_dets": 50}}, "params"),
        ({"params": None}, "params_missing"),
    ],
)
def test_incomparable_snapshots_are_refused_with_a_reason(
    cand: dict[str, Any], reason: str
) -> None:
    b = build_snapshot()
    c = build_snapshot(**cand)
    with pytest.raises(CompareRefusedError) as exc:
        compare(c, b)
    assert exc.value.reason == reason
    fields = exc.value.log_fields()
    assert fields["candidate_run_id"] == c.run_id and fields["baseline_run_id"] == b.run_id
    assert exc.value.event == "compare_refused"


def test_baseline_without_params_is_refused_too() -> None:
    with pytest.raises(CompareRefusedError) as exc:
        compare(build_snapshot(), build_snapshot(params=None))
    assert exc.value.reason == "params_missing"


def test_hash_mismatch_is_the_specific_error_with_both_hashes() -> None:
    with pytest.raises(EvalSetMismatchError) as exc:
        compare(build_snapshot(h=H2), build_snapshot(h=H1))
    f = exc.value.log_fields()
    assert f["candidate_eval_set_hash"] == H2 and f["baseline_eval_set_hash"] == H1


def test_comparison_is_deterministic_and_does_not_mutate_inputs() -> None:
    b, c = build_snapshot(metrics={"mAP50_95": 0.6}), build_snapshot(metrics={"mAP50_95": 0.5})
    before = (b.model_dump_json(), c.model_dump_json())
    assert compare(c, b) == compare(c, b)
    assert (b.model_dump_json(), c.model_dump_json()) == before


def test_non_finite_values_are_carried_so_the_gate_can_reject_them() -> None:
    c = EvalSnapshot.model_construct(
        **{**build_snapshot().__dict__, "metrics": {"mAP50_95": math.nan}}
    )
    cmp = compare(c, build_snapshot())
    assert math.isnan(cmp.headline[0].candidate or 0.0)
