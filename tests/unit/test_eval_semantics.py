"""Review fixes: parameters and provenance in snapshots, closed runs, ids, seeds, ci markers."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from cvpilot.core.errors import EvalInputError
from cvpilot.core.schema import EvalSnapshot
from cvpilot.eval import AnnotationTable, PredictionTable, evaluate
from cvpilot.eval import tables as t
from cvpilot.eval.bootstrap import check_settings

SHA = "a" * 64


def base(**over: object) -> dict[str, object]:
    fields: dict[str, object] = {
        "run_id": "0" * 26, "eval_set_hash": SHA, "step": 0, "task": "detection",
        "class_names": ["a"], "n_images": 3, "metrics": {"mAP50_95": 0.5}, "per_class": {},
        "slices": {}, "confusion_sha256": SHA, "predictions_sha256": SHA,
    }  # fmt: skip
    fields.update(over)
    return fields


GOOD_CI = {
    "ci": {"mAP50_95": (0.4, 0.6)},
    "bootstrap": {"n_resamples": 10, "seed": 0, "level": 0.95, "numpy_version": "2.2.6"},
    "ci_status": "computed",
}


def test_ci_and_bootstrap_must_appear_together() -> None:
    EvalSnapshot(**base(**GOOD_CI))
    with pytest.raises(ValidationError):
        EvalSnapshot(**base(ci={"mAP50_95": (0.4, 0.6)}, ci_status="computed"))
    with pytest.raises(ValidationError):  # bootstrap without intervals
        EvalSnapshot(**base(bootstrap=GOOD_CI["bootstrap"], ci_status="computed"))
    with pytest.raises(ValidationError):  # computed marker without intervals
        EvalSnapshot(**base(ci_status="computed"))
    with pytest.raises(ValidationError):  # intervals while the marker says none were computed
        EvalSnapshot(**{**base(**GOOD_CI), "ci_status": "off"})


def test_the_snapshot_records_why_no_intervals_exist() -> None:
    assert EvalSnapshot(**base()).ci_status == "off"
    marked = EvalSnapshot(**base(ci_status="not_computed_too_few_images"))
    assert marked.ci == {} and marked.bootstrap is None
    with pytest.raises(ValidationError):
        EvalSnapshot(**base(ci_status="bogus"))


def test_parameters_provenance_and_slice_spec_are_validated() -> None:
    snap = EvalSnapshot(
        **base(
            params={"iou_thresholds": [0.5, 0.75], "confusion_iou": 0.5, "confusion_score": 0.25},
            provenance={"checkpoint_sha256": SHA, "git_commit": "abc1234"},
            slice_spec={"camera": "camera_id"},
        )
    )
    assert snap.params is not None and snap.params.max_dets == 100
    for bad in (
        {"params": {"iou_thresholds": [], "confusion_iou": 0.5, "confusion_score": 0.2}},
        {"params": {"iou_thresholds": [0.5], "confusion_iou": 2.0, "confusion_score": 0.2}},
        {"provenance": {"checkpoint_sha256": "xyz"}},
        {"provenance": {"git_commit": "not hex!"}},
        {"slice_spec": {"bad key": "x"}},
    ):
        with pytest.raises(ValidationError):
            EvalSnapshot(**base(**bad))


@pytest.mark.parametrize("seed", [True, -1, 2**64, 1.5, "3"])
def test_seeds_must_be_integers_in_the_unsigned_64_bit_range(seed: object) -> None:
    with pytest.raises(EvalInputError, match="seed"):
        check_settings(10, 0.95, seed)  # type: ignore[arg-type]
    check_settings(10, 0.95, 2**64 - 1)


def test_ids_with_nuls_control_characters_or_lone_surrogates_are_rejected() -> None:
    for bad in ("a\x00", "a\nb", "\ud800", "x\x7f"):
        with pytest.raises(EvalInputError):
            t.as_ids(np.array(["ok", bad], dtype=object), "image_id")
    ok = t.as_ids(np.array(["ok", "fine"], dtype=object), "image_id")
    assert ok.tolist() == ["ok", "fine"]


def test_surrogates_in_fixed_width_ids_and_class_names_raise_the_domain_error() -> None:
    gts = AnnotationTable.from_columns(
        image_id=np.array(["\ud800"]), class_id=[0], boxes=[[0, 0, 5, 5]]
    )
    preds = PredictionTable.from_columns(
        image_id=np.array(["\ud800"]), class_id=[0], score=[0.5], boxes=[[0, 0, 5, 5]]
    )
    with pytest.raises(EvalInputError):
        evaluate(preds, gts, task="detection", class_names=["a"], n_resamples=0)
    with pytest.raises(EvalInputError):
        t.check_class_names(["a", "\ud800"])
