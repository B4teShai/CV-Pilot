"""Detection metrics on hand-built cases (definitions follow the COCO protocol)."""

from __future__ import annotations

import numpy as np
import pytest

from cvpilot.core import confusion as cf
from cvpilot.core import metrics as m

IOUS = m.coco_iou_thresholds()


def run(
    preds: list[tuple[int, int, float, tuple[float, float, float, float]]],
    gts: list[tuple[int, int, tuple[float, float, float, float]]],
    *,
    n_images: int = 1,
    n_classes: int = 1,
    crowd: list[bool] | None = None,
    **kw: object,
) -> tuple[m.MatchResult, m.Curves]:
    """preds: (image, cls, score, box); gts: (image, cls, box)."""
    match = m.match_detections(
        np.array([p[0] for p in preds], dtype=np.int64),
        np.array([p[3] for p in preds], dtype=np.float64).reshape(-1, 4),
        np.array([p[2] for p in preds], dtype=np.float64),
        np.array([p[1] for p in preds], dtype=np.int64),
        np.array([g[0] for g in gts], dtype=np.int64),
        np.array([g[2] for g in gts], dtype=np.float64).reshape(-1, 4),
        np.array([g[1] for g in gts], dtype=np.int64),
        n_images=n_images,
        n_classes=n_classes,
        iou_thresholds=IOUS,
        iscrowd=None if crowd is None else np.array(crowd, dtype=bool),
        **kw,  # type: ignore[arg-type]
    )
    return match, m.accumulate(match)


BOX = (10.0, 10.0, 50.0, 50.0)


def test_thresholds_are_the_coco_grid() -> None:
    assert len(IOUS) == 10 and IOUS[0] == 0.5 and IOUS[-1] == pytest.approx(0.95)
    assert len(m.RECALL_THRESHOLDS) == 101


def test_iou_matrix_and_crowd_uses_detection_area() -> None:
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    np.testing.assert_allclose(m.iou_matrix(a, b), [[50 / 150, 0.0]])
    crowd = m.iou_matrix(a, b, iscrowd=np.array([True, False]))
    np.testing.assert_allclose(crowd, [[50 / 100, 0.0]])


def test_perfect_detections_have_ap_one() -> None:
    _, curves = run([(0, 0, 0.9, BOX)], [(0, 0, BOX)])
    summary = m.summarize(curves, IOUS)
    assert summary["AP"] == pytest.approx(1.0) and summary["AP50"] == pytest.approx(1.0)
    assert summary["AR100"] == 1.0


def test_false_positive_ahead_of_true_positive_halves_precision() -> None:
    far = (200.0, 200.0, 240.0, 240.0)
    _, curves = run([(0, 0, 0.9, far), (0, 0, 0.8, BOX)], [(0, 0, BOX)])
    assert m.summarize(curves, IOUS)["AP50"] == pytest.approx(0.5)


def test_second_detection_of_the_same_object_is_a_false_positive() -> None:
    _, curves = run([(0, 0, 0.9, BOX), (0, 0, 0.8, BOX)], [(0, 0, BOX)])
    # recall reaches 1 at the first detection, so precision there is 1 and AP stays 1
    assert m.summarize(curves, IOUS)["AP50"] == pytest.approx(1.0)
    match, _ = run([(0, 0, 0.9, BOX), (0, 0, 0.8, BOX)], [(0, 0, BOX)])
    assert match.matched[0, 0].tolist() == [True, False]


def test_crowd_region_absorbs_detections_without_counting_them() -> None:
    inside = (12.0, 12.0, 30.0, 30.0)
    _, curves = run(
        [(0, 0, 0.9, BOX), (0, 0, 0.8, inside)],
        [(0, 0, BOX), (0, 0, (0.0, 0.0, 100.0, 100.0))],
        crowd=[False, True],
    )
    assert m.summarize(curves, IOUS)["AP50"] == pytest.approx(1.0)


def test_class_with_ground_truth_but_no_detections_scores_zero() -> None:
    _, curves = run([], [(0, 0, BOX)])
    summary = m.summarize(curves, IOUS)
    assert summary["AP"] == 0.0 and summary["AR100"] == 0.0


def test_no_ground_truth_leaves_metrics_undefined() -> None:
    _, curves = run([(0, 0, 0.9, BOX)], [])
    assert set(m.summarize(curves, IOUS).values()) == {-1.0}


def test_size_ranges_ignore_objects_outside_the_range() -> None:
    small = (0.0, 0.0, 10.0, 10.0)
    big = (0.0, 100.0, 200.0, 300.0)
    _, curves = run([(0, 0, 0.9, small)], [(0, 0, small), (0, 0, big)])
    summary = m.summarize(curves, IOUS)
    assert summary["APs"] == pytest.approx(1.0) and summary["APl"] == 0.0 and summary["APm"] == -1.0
    assert summary["AP"] == pytest.approx(51 / 101)  # recall tops out at 0.5


def test_per_image_detection_cap_applies_before_ranking() -> None:
    far = (200.0, 200.0, 240.0, 240.0)
    preds = [(0, 0, 0.9 - i * 0.001, far) for i in range(3)] + [(0, 0, 0.1, BOX)]
    match, curves = run(preds, [(0, 0, BOX)], max_dets=3)
    assert len(match.det_score) == 3
    assert m.summarize(curves, IOUS)["AP"] == 0.0


def test_ties_break_by_image_order_then_input_order() -> None:
    preds = [(1, 0, 0.5, BOX), (0, 0, 0.5, BOX)]
    gts = [(0, 0, BOX), (1, 0, BOX)]
    match, _ = run(preds, gts, n_images=2)
    assert match.det_image[match.order].tolist() == [0, 1]


def test_unit_weights_equal_the_unweighted_result_and_zero_weights_drop_images() -> None:
    far = (200.0, 200.0, 240.0, 240.0)
    preds = [(0, 0, 0.9, far), (0, 0, 0.8, BOX), (1, 0, 0.7, BOX)]
    gts = [(0, 0, BOX), (1, 0, BOX)]
    match, full = run(preds, gts, n_images=2)
    ones = m.accumulate(match, image_weights=np.ones(2))
    np.testing.assert_array_equal(ones.precision, full.precision)
    only_second = m.accumulate(match, image_weights=np.array([0.0, 1.0]))
    assert m.summarize(only_second, IOUS)["AP"] == pytest.approx(1.0)
    doubled = m.accumulate(match, image_weights=np.array([2.0, 1.0]))
    assert m.summarize(doubled, IOUS)["AP50"] == pytest.approx(
        m.summarize(
            run(
                preds + [(2, 0, 0.9, far), (2, 0, 0.8, BOX)],
                gts + [(2, 0, BOX)],
                n_images=3,
            )[1],
            IOUS,
        )["AP50"]
    )


def test_confusion_counts_wrong_class_missed_and_spurious() -> None:
    conf = cf.confusion_matrix(
        np.array([0, 0, 0]),
        np.array([BOX, (100.0, 100.0, 140.0, 140.0), (300.0, 300.0, 330.0, 330.0)]),
        np.array([0.9, 0.9, 0.9]),
        np.array([1, 0, 2]),
        np.array([0, 0]),
        np.array([BOX, (200.0, 10.0, 240.0, 50.0)]),
        np.array([0, 2]),
        n_images=1,
        n_classes=3,
    )
    assert conf.shape == (4, 4)
    assert conf[0, 1] == 1  # class 0 object predicted as class 1
    assert conf[2, 3] == 1  # class 2 object missed
    assert conf[3, 0] == 1 and conf[3, 2] == 1  # spurious detections
    assert conf.sum() == 4


def test_confusion_skips_low_scores_and_detections_on_crowd_regions() -> None:
    conf = cf.confusion_matrix(
        np.array([0, 0]),
        np.array([BOX, (12.0, 12.0, 30.0, 30.0)]),
        np.array([0.1, 0.9]),
        np.array([0, 0]),
        np.array([0, 0]),
        np.array([BOX, (0.0, 0.0, 100.0, 100.0)]),
        np.array([0, 0]),
        iscrowd=np.array([False, True]),
        n_images=1,
        n_classes=1,
    )
    assert conf.tolist() == [[0, 1], [0, 0]]  # the low-score box is dropped: object missed


def test_classification_metrics() -> None:
    result = cf.classification_metrics(
        np.array([0, 0, 1, 1, 2]), np.array([0, 1, 1, -1, 2]), n_classes=3
    )
    assert result.accuracy == pytest.approx(3 / 5)
    np.testing.assert_allclose(result.recall, [0.5, 0.5, 1.0])
    np.testing.assert_allclose(result.precision, [1.0, 0.5, 1.0])
    assert result.support.tolist() == [2, 2, 1]
    assert result.confusion[1, 3] == 1 and result.confusion.sum() == 5
    assert result.correct.tolist() == [True, False, True, False, True]
