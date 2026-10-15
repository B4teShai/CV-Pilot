from __future__ import annotations

import math
from typing import Any

import pytest
from pydantic import ValidationError

from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import MAX_CLASSES, EvalSnapshot

SHA_A, SHA_B, SHA_C = "a" * 64, "b" * 64, "c" * 64


def fields(**over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "run_id": new_run_id(),
        "eval_set_hash": SHA_A,
        "step": 3,
        "task": "detection",
        "class_names": ["cat", "dog"],
        "n_images": 10,
        "metrics": {"mAP50_95": 0.5, "mAP50": 0.7},
        "per_class": {"cat": {"AP50_95": 0.4, "support": 3.0}},
        "slices": {"camera=cam03": {"mAP50_95": 0.3, "n_images": 4.0}},
        "ci": {"mAP50_95": (0.4, 0.6)},
        "bootstrap": {"n_resamples": 200, "seed": 0, "level": 0.95},
        "ci_status": "computed",
        "confusion_sha256": SHA_B,
        "predictions_sha256": SHA_C,
    }
    base.update(over)
    return base


def test_valid_snapshot_round_trips_through_json() -> None:
    snap = EvalSnapshot(**fields())
    again = EvalSnapshot.model_validate_json(snap.model_dump_json())
    assert again == snap and again.model_dump_json() == snap.model_dump_json()
    assert snap.ci["mAP50_95"] == (0.4, 0.6)


def test_ci_and_bootstrap_are_optional() -> None:
    snap = EvalSnapshot(**fields(ci={}, bootstrap=None, ci_status="off"))
    assert snap.ci == {} and snap.bootstrap is None


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_are_rejected_everywhere(bad: float) -> None:
    for over in (
        {"metrics": {"mAP50": bad}},
        {"per_class": {"cat": {"AP50": bad}}},
        {"slices": {"camera=a": {"mAP50_95": bad}}},
        {"ci": {"mAP50_95": (0.1, bad)}},
    ):
        with pytest.raises(ValidationError):
            EvalSnapshot(**fields(**over))


@pytest.mark.parametrize(
    "over",
    [
        {"eval_set_hash": "xyz"},
        {"confusion_sha256": "A" * 64},
        {"predictions_sha256": SHA_A[:-1]},
        {"run_id": "not-a-ulid"},
        {"step": -1},
        {"n_images": -1},
        {"class_names": ["cat", "cat"]},
        {"class_names": []},
        {"class_names": ["bad\nname"]},
        {"class_names": ["x" * 129]},
        {"class_names": [f"c{i}" for i in range(MAX_CLASSES + 1)], "per_class": {}},
        {"metrics": {}},
        {"metrics": {"bad key": 1.0}},
        {"per_class": {"bird": {"AP50": 0.1}}},
        {"per_class": {"cat": {"bad key": 0.1}}},
        {"slices": {"nokey": {"mAP50_95": 0.1}}},
        {"slices": {"=v": {"mAP50_95": 0.1}}},
        {"slices": {"camera=\n": {"mAP50_95": 0.1}}},
        {"ci": {"unknown": (0.1, 0.2)}},
        {"ci": {"mAP50": (0.9, 0.1)}},
        {"bootstrap": None},  # ci present without the bootstrap settings that produced it
        {"bootstrap": {"n_resamples": 0, "seed": 0, "level": 0.95}},
        {"bootstrap": {"n_resamples": 10, "seed": 0, "level": 1.0}},
        {"task": "segmentation"},
        {"unexpected": 1},
    ],
)
def test_invariants_reject_bad_snapshots(over: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        EvalSnapshot(**fields(**over))


def test_snapshots_are_frozen() -> None:
    snap = EvalSnapshot(**fields())
    with pytest.raises(ValidationError):
        snap.step = 4  # type: ignore[misc]
