"""Gate semantics: units, sign, boundaries and every fail-closed case."""

from __future__ import annotations

import math
from typing import Any

import pytest
from support import build_snapshot

from cvpilot.compare.engine import compare
from cvpilot.compare.gate import (
    GateResult,
    ThresholdError,
    Thresholds,
    gate,
    parse_drop,
    parse_slice_drop,
)
from cvpilot.core.schema import EvalSnapshot

SPEC = {"camera": "camera"}


def slice_row(value: float, n: float = 50.0) -> dict[str, float]:
    return {"mAP50_95": value, "n_images": n}


def run_gate(base: dict[str, Any], cand: dict[str, Any], **th: Any) -> GateResult:
    th.setdefault("min_images", 1)
    return gate(compare(build_snapshot(**cand), build_snapshot(**base)), Thresholds(**th))


def kinds(result: GateResult) -> list[str]:
    return [v.kind for v in result.violations]


# -- units, sign, boundary ---------------------------------------------------------------


def test_drops_are_measured_in_points_on_the_0_to_100_scale() -> None:
    base = {"metrics": {"mAP50_95": 0.600}}
    assert (
        run_gate(base, {"metrics": {"mAP50_95": 0.594}}, max_drop={"mAP50_95": 0.5}).passed is False
    )
    assert (
        run_gate(base, {"metrics": {"mAP50_95": 0.596}}, max_drop={"mAP50_95": 0.5}).passed is True
    )


def test_a_drop_exactly_at_the_limit_passes_despite_float_noise() -> None:
    for limit in (0.5, 1.0, 2.0):
        base = {"metrics": {"mAP50_95": 0.600}}
        cand = {"metrics": {"mAP50_95": 0.600 - limit / 100}}
        assert run_gate(base, cand, max_drop={"mAP50_95": limit}).passed, limit


def test_improvements_never_fail_even_with_a_zero_limit() -> None:
    base = {"metrics": {"mAP50_95": 0.5}}
    assert run_gate(base, {"metrics": {"mAP50_95": 0.9}}, max_drop={"mAP50_95": 0.0}).passed
    assert run_gate(base, base, max_drop={"mAP50_95": 0.0}).passed


def test_zero_limit_fails_on_any_real_drop() -> None:
    result = run_gate(
        {"metrics": {"mAP50_95": 0.5}},
        {"metrics": {"mAP50_95": 0.4999}},
        max_drop={"mAP50_95": 0.0},
    )
    assert kinds(result) == ["headline_drop"]
    v = result.violations[0]
    assert v.drop_points == pytest.approx(0.01) and v.limit_points == 0.0


# -- fail closed: headline -----------------------------------------------------------------


def test_absent_metric_on_either_side_fails() -> None:
    both = {"metrics": {"mAP50_95": 0.6}}
    assert kinds(run_gate(both, both, max_drop={"mAP75": 1.0})) == ["headline_missing"]
    assert kinds(run_gate(both, {"metrics": {"mAP50": 0.6}}, max_drop={"mAP50_95": 1.0})) == [
        "headline_missing"
    ]
    assert kinds(run_gate({"metrics": {"mAP50": 0.6}}, both, max_drop={"mAP50_95": 1.0})) == [
        "headline_missing"
    ]


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_fail_on_either_side(bad: float) -> None:
    good = build_snapshot(metrics={"mAP50_95": 0.6})
    broken = EvalSnapshot.model_construct(**{**good.__dict__, "metrics": {"mAP50_95": bad}})
    for cand, base in ((broken, good), (good, broken)):
        result = gate(compare(cand, base), Thresholds(max_drop={"mAP50_95": 5.0}, min_images=1))
        assert kinds(result) == ["headline_not_finite"] and not result.passed


def test_too_few_images_in_the_evaluation_set_fails() -> None:
    snap = {"n_images": 29, "metrics": {"mAP50_95": 0.6}}
    result = run_gate(snap, snap, max_drop={"mAP50_95": 1.0}, min_images=30)
    assert kinds(result) == ["too_few_images"]
    assert run_gate(
        {**snap, "n_images": 30},
        {**snap, "n_images": 30},
        max_drop={"mAP50_95": 1.0},
        min_images=30,
    ).passed


# -- per class -----------------------------------------------------------------------------


def classes(**ap: float) -> dict[str, dict[str, float]]:
    return {n: {"AP50_95": v, "support": 10.0} for n, v in ap.items()}


def test_per_class_drop_names_the_class_and_orders_worst_first() -> None:
    base = {"class_names": ["a", "b", "c"], "per_class": classes(a=0.5, b=0.5, c=0.5)}
    cand = {"class_names": ["a", "b", "c"], "per_class": classes(a=0.45, b=0.2, c=0.6)}
    result = run_gate(base, cand, per_class_max_drop={"AP50_95": 2.0})
    assert kinds(result) == ["class_drop", "class_drop"]
    assert [v.subject for v in result.violations] == ["b", "a"]
    assert result.violations[0].drop_points == pytest.approx(30.0)


def test_per_class_metric_lost_in_the_candidate_fails_but_no_ground_truth_does_not() -> None:
    base = {
        "per_class": {
            "cat": {"AP50_95": 0.5, "support": 3.0},
            "dog": {"support": 0.0},
            "bird": {"AP50_95": 0.5, "support": 2.0},
        }
    }
    cand = {
        "per_class": {
            "cat": {"AP50_95": 0.5, "support": 3.0},
            "dog": {"support": 0.0},
            "bird": {"support": 2.0},
        }
    }
    result = run_gate(base, cand, per_class_max_drop={"AP50_95": 2.0})
    assert kinds(result) == ["class_missing"] and result.violations[0].subject == "bird"


def test_a_per_class_threshold_that_gates_nothing_fails() -> None:
    empty = {"per_class": {n: {"support": 0.0} for n in ("cat", "dog", "bird")}}
    assert kinds(run_gate(empty, empty, per_class_max_drop={"AP50_95": 2.0})) == [
        "no_class_checked"
    ]


def test_non_finite_class_value_fails() -> None:
    good = build_snapshot()
    rows = {**good.per_class, "cat": {"AP50_95": math.nan, "support": 20.0}}
    broken = EvalSnapshot.model_construct(**{**good.__dict__, "per_class": rows})
    result = gate(
        compare(broken, good), Thresholds(per_class_max_drop={"AP50_95": 50.0}, min_images=1)
    )
    assert "class_not_finite" in kinds(result)


# -- slices --------------------------------------------------------------------------------


def with_slices(
    base_rows: dict[str, Any], cand_rows: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {"slices": base_rows, "slice_spec": SPEC, "slice_hash": "a" * 64},
        {"slices": cand_rows, "slice_spec": SPEC, "slice_hash": "a" * 64},
    )


def test_wildcard_slice_drop_reports_the_slice() -> None:
    base, cand = with_slices(
        {"camera=c1": slice_row(0.6), "camera=c2": slice_row(0.6)},
        {"camera=c1": slice_row(0.59), "camera=c2": slice_row(0.5)},
    )
    result = run_gate(base, cand, slice_max_drop={"camera=*": 1.5})
    assert kinds(result) == ["slice_drop"] and result.violations[0].subject == "camera=c2"


def test_named_slice_only_gates_that_slice() -> None:
    base, cand = with_slices(
        {"camera=c1": slice_row(0.6), "camera=c2": slice_row(0.6)},
        {"camera=c1": slice_row(0.1), "camera=c2": slice_row(0.6)},
    )
    assert run_gate(base, cand, slice_max_drop={"camera=c2": 1.0}).passed


def test_absent_slices_fail() -> None:
    base, cand = with_slices({"camera=c1": slice_row(0.6)}, {"camera=c2": slice_row(0.6)})
    assert "slice_missing" in kinds(run_gate(base, cand, slice_max_drop={"camera=*": 1.5}))
    assert kinds(run_gate(base, cand, slice_max_drop={"camera=c9": 1.5})) == ["slice_missing"]
    none = {"slices": {}, "slice_hash": "a" * 64}
    assert kinds(run_gate(none, none, slice_max_drop={"camera=*": 1.5})) == ["slice_no_match"]


def test_slice_with_another_column_fails() -> None:
    row = {"camera=c1": slice_row(0.6)}
    cand = {"slices": row, "slice_spec": {"camera": "cam_id"}, "slice_hash": "a" * 64}
    base = {"slices": row, "slice_spec": SPEC, "slice_hash": "a" * 64}
    assert kinds(run_gate(base, cand, slice_max_drop={"camera=*": 1.5})) == ["slice_spec_mismatch"]


def test_small_slices_fail_unless_skipping_is_asked_for() -> None:
    base, cand = with_slices(
        {"camera=c1": slice_row(0.6, 50), "camera=c2": slice_row(0.6, 5)},
        {"camera=c1": slice_row(0.6, 50), "camera=c2": slice_row(0.1, 5)},
    )
    strict = run_gate(base, cand, slice_max_drop={"camera=*": 1.5}, min_images=30)
    assert kinds(strict) == ["slice_too_small"] and strict.violations[0].subject == "camera=c2"
    lenient = run_gate(
        base,
        cand,
        slice_max_drop={"camera=*": 1.5},
        min_images=30,
        skip_small_slices=True,
        max_skipped=1,
    )
    assert lenient.passed and lenient.skipped == ("camera=c2",)


def test_skipping_every_slice_still_fails() -> None:
    base, cand = with_slices({"camera=c1": slice_row(0.6, 5)}, {"camera=c1": slice_row(0.1, 5)})
    result = run_gate(
        base,
        cand,
        slice_max_drop={"camera=*": 1.5},
        min_images=30,
        skip_small_slices=True,
        max_skipped=1,
    )
    assert kinds(result) == ["slice_none_gated"] and not result.passed


def test_non_finite_slice_fails() -> None:
    good = build_snapshot(
        slices={"camera=c1": slice_row(0.6)}, slice_spec=SPEC, slice_hash="a" * 64
    )
    broken = EvalSnapshot.model_construct(
        **{**good.__dict__, "slices": {"camera=c1": {"mAP50_95": math.nan, "n_images": 50.0}}}
    )
    result = gate(compare(broken, good), Thresholds(slice_max_drop={"camera=*": 9.0}, min_images=1))
    assert kinds(result) == ["slice_not_finite"]


# -- thresholds ------------------------------------------------------------------------------


def test_at_least_one_threshold_is_required() -> None:
    with pytest.raises(ThresholdError, match="at least one"):
        Thresholds()


@pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf, 100.0, 100.5])
def test_limits_must_be_finite_and_within_0_to_100(bad: float) -> None:
    with pytest.raises(ThresholdError):
        Thresholds(max_drop={"mAP50_95": bad})


def test_min_images_must_be_positive() -> None:
    with pytest.raises(ThresholdError):
        Thresholds(max_drop={"mAP50_95": 1.0}, min_images=0)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("mAP50_95=0.5", ("mAP50_95", 0.5)),
        ("AP50=2", ("AP50", 2.0)),
        ("x=0", ("x", 0.0)),
    ],
)
def test_parse_drop(text: str, expected: tuple[str, float]) -> None:
    assert parse_drop(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "mAP50_95",
        "=1",
        "a=",
        "a=-1",
        "a=nan",
        "a=inf",
        "a=1e3",
        "a=1_0",
        "a= 1",
        "a=1.5.2",
        "a=100",
        "a=101",
        "1a=1",
        "a b=1",
        "a=1\n",
        "a=०",
    ],
)
def test_parse_drop_rejects_bad_syntax(text: str) -> None:
    with pytest.raises(ThresholdError):
        parse_drop(text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("camera=*:1.5", ("camera=*", 1.5)),
        ("camera=c1:2", ("camera=c1", 2.0)),
        ("site=a:b:0.5", ("site=a:b", 0.5)),
    ],
)
def test_parse_slice_drop(text: str, expected: tuple[str, float]) -> None:
    assert parse_slice_drop(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "camera=*",
        "camera:1",
        "=c:1",
        "camera=:1",
        "camera=c1:x",
        "1cam=c:1",
        "camera=*x:1",
        "camera=c\n:1",
    ],
)
def test_parse_slice_drop_rejects_bad_syntax(text: str) -> None:
    with pytest.raises(ThresholdError):
        parse_slice_drop(text)


def test_hostile_text_parses_in_linear_time() -> None:
    import time

    start = time.perf_counter()
    for text in ("a" * 200_000 + "=1", "a=" + "1" * 200_000, "a=" + "1." * 100_000):
        with pytest.raises(ThresholdError):
            parse_drop(text)
    with pytest.raises(ThresholdError):
        parse_slice_drop("a=" + ":" * 200_000)
    assert time.perf_counter() - start < 1.0


def test_violations_are_ordered_worst_first_and_the_worst_is_exposed() -> None:
    base = {"metrics": {"mAP50_95": 0.6, "mAP50": 0.8}}
    cand = {"metrics": {"mAP50_95": 0.55, "mAP50": 0.5}}
    result = run_gate(base, cand, max_drop={"mAP50_95": 1.0, "mAP50": 1.0})
    assert [v.subject for v in result.violations] == ["mAP50", "mAP50_95"]
    assert result.worst is result.violations[0]
