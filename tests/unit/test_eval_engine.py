"""The eval engine: results on hand-built sets, canonical hashing, determinism and hostile input."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest

from cvpilot.core.errors import EvalInputError
from cvpilot.core.hashing import new_run_id
from cvpilot.eval import eval_set_hash, evaluate
from cvpilot.eval import tables as t

CLASSES = ["cat", "dog"]
SHA = "d" * 64


def gts_table(
    rows: list[tuple[str, int, tuple[float, float, float, float]]], **kw: Any
) -> t.AnnotationTable:
    return t.AnnotationTable.from_columns(
        image_id=[r[0] for r in rows],
        class_id=[r[1] for r in rows],
        boxes=np.array([r[2] for r in rows], dtype=float).reshape(-1, 4),
        **kw,
    )


def preds_table(
    rows: list[tuple[str, int, float, tuple[float, float, float, float]]],
) -> t.PredictionTable:
    return t.PredictionTable.from_columns(
        image_id=[r[0] for r in rows],
        class_id=[r[1] for r in rows],
        score=[r[2] for r in rows],
        boxes=np.array([r[3] for r in rows], dtype=float).reshape(-1, 4),
    )


A = (10.0, 10.0, 50.0, 50.0)
B = (60.0, 60.0, 120.0, 140.0)
GT = [("img1", 0, A), ("img1", 1, B), ("img2", 0, A)]
PRED_PERFECT = [("img1", 0, 0.9, A), ("img1", 1, 0.8, B), ("img2", 0, 0.7, A)]


def run_eval(preds: Any = None, gts: Any = None, **kw: Any) -> Any:
    kw.setdefault("task", "detection")
    kw.setdefault("class_names", CLASSES)
    kw.setdefault("n_resamples", 0)
    preds = preds_table(PRED_PERFECT) if preds is None else preds
    gts = gts_table(GT) if gts is None else gts
    return evaluate(preds, gts, **kw)


def test_perfect_predictions_score_one_everywhere() -> None:
    result = run_eval()
    assert result.metrics["mAP50_95"] == pytest.approx(1.0)
    assert result.metrics["mAP50"] == pytest.approx(1.0)
    assert result.metrics["AR100"] == 1.0
    assert result.per_class["cat"]["support"] == 2.0 and result.per_class["dog"]["support"] == 1.0
    assert result.per_class["cat"]["AP50_95"] == pytest.approx(1.0)
    assert result.n_images == 2 and result.class_names == CLASSES
    assert result.confusion.tolist() == [[2, 0, 0], [0, 1, 0], [0, 0, 0]]


def test_missing_and_extra_detections_lower_the_scores() -> None:
    preds = preds_table([("img1", 0, 0.9, A), ("img2", 1, 0.8, A)])  # dog is wrong, dog gt missed
    result = run_eval(preds)
    assert 0.0 < result.metrics["mAP50"] < 1.0
    assert result.per_class["dog"]["AP50"] == 0.0
    assert result.confusion[0, 1] == 1  # the cat at img2 was claimed by a dog detection
    assert result.confusion[1, 2] == 1  # the dog at img1 was missed


def test_images_without_annotations_count_their_detections_as_false_positives() -> None:
    images = t.ImageTable.from_columns(image_id=["img1", "img2", "empty"])
    gts = gts_table(GT, images=images)
    preds = preds_table(PRED_PERFECT + [("empty", 0, 0.99, A)])
    result = run_eval(preds, gts)
    assert result.n_images == 3
    assert result.per_class["cat"]["AP50"] < 1.0


def test_undefined_metrics_are_omitted() -> None:
    result = run_eval()
    assert "mAP_small" not in result.metrics  # no small objects in this set
    assert "mAP_medium" in result.metrics and "mAP_large" not in result.metrics


def test_a_class_without_ground_truth_reports_only_its_support() -> None:
    result = run_eval(gts=gts_table([g for g in GT if g[1] == 0]))
    assert result.per_class["dog"] == {"support": 0.0}


def test_no_countable_ground_truth_is_an_error() -> None:
    gts = gts_table(GT[:1], iscrowd=[True])
    with pytest.raises(EvalInputError, match="no ground-truth objects"):
        run_eval(gts=gts)
    with pytest.raises(EvalInputError, match="no ground-truth objects"):
        run_eval(gts=gts_table([]))


def test_snapshot_is_built_from_a_result() -> None:
    snap = run_eval().to_snapshot(
        run_id=new_run_id(), step=7, confusion_sha256=SHA, predictions_sha256=SHA
    )
    assert snap.step == 7 and snap.eval_set_hash == run_eval().eval_set_hash
    assert snap.metrics["mAP50_95"] == pytest.approx(1.0)
    assert snap.task == "detection" and snap.n_images == 2 and snap.ci == {}


def test_same_input_gives_identical_snapshot_bytes() -> None:
    def blob() -> str:
        return (
            run_eval()
            .to_snapshot(run_id="0" * 26, step=1, confusion_sha256=SHA, predictions_sha256=SHA)
            .model_dump_json()
        )

    assert blob() == blob()


def test_row_order_does_not_change_the_result_except_through_score_ties() -> None:
    shuffled = preds_table(list(reversed(PRED_PERFECT)))
    assert run_eval(shuffled).metrics == run_eval().metrics


# -- eval_set_hash ---------------------------------------------------------------------------


def canonical_hash(images: list[Any]) -> str:
    text = json.dumps(
        {"class_names": CLASSES, "images": images}, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(text.encode()).hexdigest()


def test_eval_set_hash_matches_the_documented_canonical_json() -> None:
    gts = gts_table(
        GT,
        images=t.ImageTable.from_columns(
            image_id=["img1", "img2"], width=[200, 300], height=[150, 100]
        ),
    )
    expected = canonical_hash(
        [
            ["img1", 200, 150, [[0, 10.0, 10.0, 50.0, 50.0, 0], [1, 60.0, 60.0, 120.0, 140.0, 0]]],
            ["img2", 300, 100, [[0, 10.0, 10.0, 50.0, 50.0, 0]]],
        ]
    )
    assert eval_set_hash(gts, CLASSES) == expected


def test_eval_set_hash_ignores_row_order_and_sub_cent_noise_but_sees_real_changes() -> None:
    base = eval_set_hash(gts_table(GT), CLASSES)
    assert eval_set_hash(gts_table(list(reversed(GT))), CLASSES) == base
    noisy = [(i, c, tuple(v + 0.0004 for v in b)) for i, c, b in GT]
    assert eval_set_hash(gts_table(noisy), CLASSES) == base
    moved = [(i, c, tuple(v + 0.02 for v in b)) for i, c, b in GT]
    assert eval_set_hash(gts_table(moved), CLASSES) != base
    assert eval_set_hash(gts_table(GT), ["cat", "puppy"]) != base
    assert eval_set_hash(gts_table(GT[:2]), CLASSES) != base
    flagged = eval_set_hash(gts_table(GT, iscrowd=[False, False, True]), CLASSES)
    assert flagged != base


def test_result_carries_the_set_hash() -> None:
    assert run_eval().eval_set_hash == eval_set_hash(gts_table(GT), CLASSES)


# -- classification --------------------------------------------------------------------------


def test_classification_metrics_and_confusion() -> None:
    gts = t.AnnotationTable.from_columns(image_id=["a", "b", "c", "d"], class_id=[0, 0, 1, 1])
    preds = t.PredictionTable.from_columns(
        image_id=["a", "b", "c"], class_id=[0, 1, 1], score=[0.9, 0.8, 0.7]
    )
    result = evaluate(preds, gts, task="classification", class_names=CLASSES)
    assert result.metrics["top1"] == pytest.approx(0.5)
    assert result.per_class["cat"]["recall"] == 0.5 and result.per_class["cat"]["precision"] == 1.0
    assert result.per_class["dog"]["support"] == 2.0
    assert result.confusion.tolist() == [[1, 1, 0], [0, 1, 1], [0, 0, 0]]


def test_classification_rejects_two_predictions_for_one_image() -> None:
    gts = t.AnnotationTable.from_columns(image_id=["a"], class_id=[0])
    preds = t.PredictionTable.from_columns(image_id=["a", "a"], class_id=[0, 1], score=[0.9, 0.8])
    with pytest.raises(EvalInputError, match="at most one prediction"):
        evaluate(preds, gts, task="classification", class_names=CLASSES)


def test_classification_needs_exactly_one_label_per_image() -> None:
    gts = t.AnnotationTable.from_columns(image_id=["a", "a"], class_id=[0, 1])
    preds = t.PredictionTable.from_columns(image_id=["a"], class_id=[0], score=[0.9])
    with pytest.raises(EvalInputError, match="exactly one label"):
        evaluate(preds, gts, task="classification", class_names=CLASSES)


# -- hostile input ----------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_scores_are_rejected(bad: float) -> None:
    preds = preds_table([("img1", 0, bad, A)])
    with pytest.raises(EvalInputError, match="non-finite scores"):
        run_eval(preds)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_boxes_are_rejected(bad: float) -> None:
    with pytest.raises(EvalInputError, match="non-finite box"):
        run_eval(preds_table([("img1", 0, 0.5, (0.0, 0.0, bad, 5.0))]))
    with pytest.raises(EvalInputError, match="non-finite box"):
        run_eval(gts=gts_table([("img1", 0, (0.0, bad, 5.0, 5.0))]))


@pytest.mark.parametrize("cls", [-1, 2, 2**40])
def test_class_ids_outside_the_range_are_rejected(cls: int) -> None:
    with pytest.raises(EvalInputError, match="class id"):
        run_eval(preds_table([("img1", cls, 0.5, A)]))
    with pytest.raises(EvalInputError, match="class id"):
        run_eval(gts=gts_table([("img1", cls, A)]))


def test_predictions_for_images_outside_the_evaluation_set_are_rejected() -> None:
    with pytest.raises(EvalInputError, match="not in the evaluation set"):
        run_eval(preds_table([("ghost", 0, 0.5, A)]))


def test_ground_truth_images_missing_from_the_image_list_are_rejected() -> None:
    images = t.ImageTable.from_columns(image_id=["img1"])
    with pytest.raises(EvalInputError, match="not in the evaluation set"):
        run_eval(gts=gts_table(GT, images=images))


def test_duplicate_image_ids_in_the_image_list_are_rejected() -> None:
    images = t.ImageTable.from_columns(image_id=["img1", "img1", "img2"])
    with pytest.raises(EvalInputError, match="repeats image id"):
        run_eval(gts=gts_table(GT, images=images))


@pytest.mark.parametrize("box", [(50.0, 10.0, 10.0, 50.0), (10.0, 50.0, 50.0, 10.0)])
def test_reversed_boxes_are_rejected(box: tuple[float, float, float, float]) -> None:
    with pytest.raises(EvalInputError, match="x2 < x1"):
        run_eval(preds_table([("img1", 0, 0.5, box)]))


def test_ground_truth_outside_the_image_is_rejected() -> None:
    images = t.ImageTable.from_columns(
        image_id=["img1", "img2"], width=[100, 100], height=[100, 100]
    )
    run_eval(gts=gts_table([("img1", 0, (0, 0, 100, 100)), ("img2", 0, A)], images=images))
    with pytest.raises(EvalInputError, match="outside the image"):
        run_eval(gts=gts_table([("img1", 0, (0, 0, 100.5, 100)), ("img2", 0, A)], images=images))
    with pytest.raises(EvalInputError, match="outside"):
        run_eval(gts=gts_table([("img1", 0, (-5, 0, 10, 10)), ("img2", 0, A)], images=images))


def test_absurd_coordinates_and_image_sizes_are_rejected() -> None:
    with pytest.raises(EvalInputError, match="outside"):
        run_eval(preds_table([("img1", 0, 0.5, (0.0, 0.0, 1e12, 5.0))]))
    bad = t.ImageTable.from_columns(image_id=["img1", "img2"], width=[0, 5], height=[5, 5])
    with pytest.raises(EvalInputError, match="width and height"):
        run_eval(gts=gts_table(GT, images=bad))


def test_class_names_are_validated() -> None:
    for names in (
        [],
        ["a", "a"],
        ["a", ""],
        ["bad\nname", "b"],
        ["x" * 129, "b"],
        [f"c{i}" for i in range(2049)],
    ):
        with pytest.raises(EvalInputError):
            run_eval(class_names=names)


@pytest.mark.parametrize(
    "ids", [["ok", ""], ["ok", "a\x00b"], ["ok", "x" * 300], ["ok", "line\nbreak"]]
)
def test_hostile_image_ids_are_rejected(ids: list[str]) -> None:
    with pytest.raises(EvalInputError):
        gts = t.AnnotationTable.from_columns(image_id=ids, class_id=[0, 0], boxes=np.array([A, A]))
        run_eval(gts=gts)


def test_wrong_dtypes_and_shapes_are_rejected_at_construction() -> None:
    with pytest.raises(EvalInputError, match="integers"):
        t.PredictionTable.from_columns(image_id=["a"], class_id=[0.5], score=[0.1])
    with pytest.raises(EvalInputError, match="integers"):
        t.PredictionTable.from_columns(image_id=["a"], class_id=[True], score=[0.1])
    with pytest.raises(EvalInputError, match="numbers"):
        t.PredictionTable.from_columns(image_id=["a"], class_id=[0], score=["x"])
    with pytest.raises(EvalInputError, match="strings or integers"):
        t.PredictionTable.from_columns(image_id=[None], class_id=[0], score=[0.1])
    with pytest.raises(EvalInputError, match="rows"):
        t.PredictionTable.from_columns(image_id=["a", "b"], class_id=[0], score=[0.1, 0.2])
    with pytest.raises(EvalInputError, match="shape"):
        t.PredictionTable.from_columns(image_id=["a"], class_id=[0], score=[0.1], boxes=[[1, 2, 3]])
    with pytest.raises(EvalInputError, match="beyond"):
        t.PredictionTable.from_columns(
            image_id=["a"], class_id=np.array([2**63], dtype=np.uint64), score=[0.1]
        )


def test_integer_image_ids_are_accepted_as_strings() -> None:
    gts = t.AnnotationTable.from_columns(image_id=[1, 2], class_id=[0, 1], boxes=[A, B])
    preds = t.PredictionTable.from_columns(
        image_id=["1", "2"], class_id=[0, 1], score=[0.9, 0.9], boxes=[A, B]
    )
    assert evaluate(preds, gts, task="detection", class_names=CLASSES).metrics[
        "mAP50"
    ] == pytest.approx(1.0)


# -- resource limits: each bound passes at the limit and fails one past it ---------------------


@pytest.mark.parametrize(
    ("constant", "build"),
    [
        ("MAX_PREDICTIONS", "preds"),
        ("MAX_ANNOTATIONS", "gts"),
        ("MAX_IMAGES", "images"),
        ("MAX_ANNOTATIONS_PER_IMAGE", "per_image"),
    ],
)
def test_resource_limits_have_boundary_tests(
    monkeypatch: pytest.MonkeyPatch, constant: str, build: str
) -> None:
    monkeypatch.setattr(t, constant, 3)

    def attempt(n: int) -> None:
        names = [f"i{i}" for i in range(n)]
        if build == "preds":
            run_eval(preds_table([("img1", 0, 0.5 + i * 0.01, A) for i in range(n)]))
        elif build == "gts":
            run_eval(gts=gts_table([(f"i{i % 2}", 0, A) for i in range(n)]), preds=preds_table([]))
        elif build == "images":
            gts = gts_table([(names[0], 0, A)], images=t.ImageTable.from_columns(image_id=names))
            run_eval(gts=gts, preds=preds_table([]))
        else:
            run_eval(gts=gts_table([("img1", 0, A)] * n), preds=preds_table([]))

    attempt(3)
    with pytest.raises(EvalInputError, match="limit|more than"):
        attempt(4)


# -- hostile text must be rejected quickly (no regex backtracking on untrusted strings) -------


def test_very_long_hostile_strings_are_rejected_in_linear_time() -> None:
    import time

    from pydantic import ValidationError

    from cvpilot.core.schema import EvalSnapshot
    from cvpilot.eval import slices as sl

    hostile = "a" * 2_000_000 + "!"
    started = time.perf_counter()
    for call in (
        lambda: run_eval(class_names=[hostile, "b"]),
        lambda: t.as_ids(np.array([hostile]), "image_id"),
        lambda: sl._KEY.fullmatch(hostile),
    ):
        try:
            call()
        except EvalInputError:
            pass
    with pytest.raises(ValidationError):
        EvalSnapshot(
            run_id="0" * 26, eval_set_hash=SHA, step=0, task="detection", class_names=["a"],
            n_images=1, metrics={hostile: 1.0}, per_class={}, slices={}, confusion_sha256=SHA,
            predictions_sha256=SHA,
        )  # fmt: skip
    assert time.perf_counter() - started < 2.0


def test_a_crowded_image_stays_within_the_matching_budget() -> None:
    import time

    n = 6000
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 900, (n, 2))
    boxes = np.column_stack([xy, xy + rng.uniform(5, 60, (n, 2))])
    gts = t.AnnotationTable.from_columns(image_id=["only"] * n, class_id=[0] * n, boxes=boxes)
    preds = t.PredictionTable.from_columns(
        image_id=["only"] * 300,
        class_id=[0] * 300,
        score=rng.random(300),
        boxes=boxes[:300] + 1.0,
    )
    started = time.perf_counter()
    result = evaluate(preds, gts, task="detection", class_names=["a"], n_resamples=0)
    assert time.perf_counter() - started < 10 and result.metrics["mAP50"] > 0.0


@pytest.mark.parametrize("step", [2**63, 2**70])
def test_steps_beyond_the_integer_range_are_rejected(step: int) -> None:
    from pydantic import ValidationError

    from cvpilot.core.schema import EvalSnapshot

    with pytest.raises(ValidationError):
        EvalSnapshot(
            run_id="0" * 26, eval_set_hash=SHA, step=step, task="detection", class_names=["a"],
            n_images=1, metrics={"m": 1.0}, per_class={}, slices={}, confusion_sha256=SHA,
            predictions_sha256=SHA,
        )  # fmt: skip


# -- memory budget ---------------------------------------------------------------------------


def test_the_estimated_peak_memory_is_checked_before_any_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cvpilot.eval import engine

    estimate = engine.estimate_bytes(len(PRED_PERFECT), len(GT), len(CLASSES), 10)
    assert estimate > 0
    monkeypatch.setattr(engine, "MAX_ESTIMATED_BYTES", estimate)
    run_eval()
    monkeypatch.setattr(engine, "MAX_ESTIMATED_BYTES", estimate - 1)
    with pytest.raises(EvalInputError, match="estimated peak memory"):
        run_eval()


def test_workers_shrink_with_the_dataset_instead_of_exhausting_memory() -> None:
    from cvpilot.eval import engine

    assert engine.plan_workers(4, n_pred=1_000) == 4
    assert engine.plan_workers(4, n_pred=10**7) in (1, 2, 3, 4)
    assert engine.plan_workers(8, n_pred=10**9) == 1
    assert engine.plan_workers(0, n_pred=10) == 1
