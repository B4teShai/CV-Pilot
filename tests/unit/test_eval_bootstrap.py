from __future__ import annotations

import numpy as np
import pytest

from cvpilot.core.errors import EvalInputError
from cvpilot.core.schema import MAX_RESAMPLES
from cvpilot.eval import AnnotationTable, PredictionTable, evaluate
from cvpilot.eval.bootstrap import bootstrap_ci

CLASSES = ["a", "b", "c"]


def random_set(seed: int = 0, n_images: int = 40) -> tuple[PredictionTable, AnnotationTable]:
    rng = np.random.default_rng(seed)
    g_img, g_cls, g_box, p_img, p_cls, p_score, p_box = [], [], [], [], [], [], []
    for i in range(n_images):
        for _ in range(int(rng.integers(1, 4))):
            x, y = rng.uniform(0, 300, 2)
            w, h = rng.uniform(20, 120, 2)
            box = [x, y, x + w, y + h]
            cls = int(rng.integers(0, 3))
            g_img.append(f"i{i:04d}")
            g_cls.append(cls)
            g_box.append(box)
            if rng.random() < 0.8:
                p_img.append(f"i{i:04d}")
                p_cls.append(cls if rng.random() < 0.9 else int(rng.integers(0, 3)))
                p_score.append(float(rng.random()))
                p_box.append(list(np.array(box) + rng.normal(0, 3, 4)))
    gts = AnnotationTable.from_columns(image_id=g_img, class_id=g_cls, boxes=g_box)
    preds = PredictionTable.from_columns(image_id=p_img, class_id=p_cls, score=p_score, boxes=p_box)
    return preds, gts


def test_bootstrap_ci_is_seeded_and_brackets_the_mean() -> None:
    values = np.random.default_rng(0).normal(0.0, 1.0, 400)
    lo, hi = bootstrap_ci(values, n_resamples=500, seed=7)
    assert (lo, hi) == bootstrap_ci(values, n_resamples=500, seed=7)
    assert (lo, hi) != bootstrap_ci(values, n_resamples=500, seed=8)
    assert lo < values.mean() < hi
    assert 0.1 < hi - lo < 0.4  # about 2 * 1.96 / sqrt(400)


def test_bootstrap_ci_of_a_constant_is_a_point_and_level_widens_the_interval() -> None:
    assert bootstrap_ci(np.full(10, 0.5), n_resamples=50) == (0.5, 0.5)
    values = np.arange(30, dtype=float)
    narrow = bootstrap_ci(values, n_resamples=300, level=0.5)
    wide = bootstrap_ci(values, n_resamples=300, level=0.99)
    assert wide[0] < narrow[0] and wide[1] > narrow[1]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"per_image": np.array([])},
        {"per_image": np.array([0.1, np.nan])},
        {"per_image": np.array([0.1, np.inf])},
        {"per_image": np.zeros((2, 2))},
        {"per_image": np.ones(3), "n_resamples": 0},
        {"per_image": np.ones(3), "n_resamples": MAX_RESAMPLES + 1},
        {"per_image": np.ones(3), "level": 0.0},
        {"per_image": np.ones(3), "level": 1.0},
        {"per_image": np.ones(3), "level": float("nan")},
    ],
)
def test_bootstrap_ci_rejects_bad_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(EvalInputError):
        bootstrap_ci(**kwargs)  # type: ignore[arg-type]


def test_resample_bound_is_inclusive() -> None:
    bootstrap_ci(np.arange(5.0), n_resamples=MAX_RESAMPLES)


def test_large_inputs_are_resampled_in_bounded_chunks() -> None:
    lo, hi = bootstrap_ci(np.random.default_rng(1).random(200_000), n_resamples=20)
    assert 0.49 < lo <= hi < 0.51


def test_evaluate_adds_seeded_intervals_for_the_headline_metrics() -> None:
    preds, gts = random_set()
    first = evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=100, seed=3)
    again = evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=100, seed=3)
    other = evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=100, seed=4)
    assert set(first.ci) == {"mAP50_95", "mAP50", "AR100"}
    assert first.ci == again.ci and first.ci != other.ci
    for name, (lo, hi) in first.ci.items():
        assert lo <= first.metrics[name] <= hi, name  # the 95% interval contains the estimate
        assert 0.0 <= lo <= hi <= 1.0
    assert first.bootstrap is not None
    assert (first.bootstrap.n_resamples, first.bootstrap.seed, first.bootstrap.level) == (
        100,
        3,
        0.95,
    )


def test_worker_count_does_not_change_the_intervals() -> None:
    preds, gts = random_set(1)
    kw = {"task": "detection", "class_names": CLASSES, "n_resamples": 60, "seed": 1}
    serial = evaluate(preds, gts, workers=1, **kw)  # type: ignore[arg-type]
    parallel = evaluate(preds, gts, workers=4, **kw)  # type: ignore[arg-type]
    assert serial.ci == parallel.ci


def test_snapshot_carries_the_intervals_and_bootstrap_settings() -> None:
    preds, gts = random_set()
    result = evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=50, seed=2)
    snap = result.to_snapshot(
        run_id="0" * 26, step=0, confusion_sha256="a" * 64, predictions_sha256="b" * 64
    )
    assert snap.ci == result.ci and snap.bootstrap is not None and snap.bootstrap.seed == 2


def test_zero_resamples_or_a_single_image_skip_the_intervals() -> None:
    preds, gts = random_set(n_images=5)
    off = evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=0)
    assert off.ci == {} and off.bootstrap is None
    preds1, gts1 = random_set(n_images=1)
    one = evaluate(preds1, gts1, task="detection", class_names=CLASSES, n_resamples=20)
    assert one.ci == {} and one.bootstrap is None


def test_classification_interval_covers_the_accuracy() -> None:
    n = 200
    rng = np.random.default_rng(0)
    ids = [f"i{i:04d}" for i in range(n)]
    truth = rng.integers(0, 3, n)
    guess = np.where(rng.random(n) < 0.8, truth, (truth + 1) % 3)
    gts = AnnotationTable.from_columns(image_id=ids, class_id=truth)
    preds = PredictionTable.from_columns(image_id=ids, class_id=guess, score=np.ones(n))
    result = evaluate(
        preds, gts, task="classification", class_names=CLASSES, n_resamples=300, seed=0
    )
    lo, hi = result.ci["top1"]
    assert lo <= result.metrics["top1"] <= hi and hi - lo < 0.2


@pytest.mark.parametrize("bad", [-1, MAX_RESAMPLES + 1])
def test_evaluate_rejects_out_of_range_resample_counts(bad: int) -> None:
    preds, gts = random_set(n_images=5)
    with pytest.raises(EvalInputError, match="n_resamples"):
        evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=bad)


def test_vectorised_resampler_agrees_with_the_weighted_accumulate() -> None:
    from cvpilot.core import metrics as m
    from cvpilot.eval import bootstrap as bs
    from cvpilot.eval.validate import prepare_ground_truth, prepare_predictions

    preds, gts = random_set(2, n_images=30)
    gt = prepare_ground_truth(gts, 3, "detection")
    pr = prepare_predictions(preds, gt, 3, "detection")
    match = m.match_detections(
        pr.image, pr.boxes, pr.score, pr.cls, gt.ann_image, gt.ann_boxes, gt.ann_cls,
        n_images=gt.n_images, n_classes=3, iou_thresholds=m.coco_iou_thresholds(),
    )  # fmt: skip
    resampler = bs._Resampler(match)
    rng = np.random.default_rng(0)
    for _ in range(25):
        weights = np.bincount(rng.integers(0, 30, 30), minlength=30).astype(float)
        curves = m.accumulate(match, image_weights=weights, areas=[0], max_dets=[100])
        block = curves.precision[:, :, :, 0, 0]
        ap, ap50, ar = resampler.run(weights)
        if not (block > -1).any():
            assert np.isnan(ap)
            continue
        assert ap == pytest.approx(block[block > -1].mean(), abs=1e-9)
        assert ap50 == pytest.approx(block[0][block[0] > -1].mean(), abs=1e-9)
        recall = curves.recall[:, :, 0, 0]
        assert ar == pytest.approx(recall[recall > -1].mean(), abs=1e-9)
