"""The class-agnostic confusion matrix stays bounded on crowded images."""

from __future__ import annotations

import time

import numpy as np

from cvpilot.core import confusion as cf

I64 = np.int64


def test_one_image_with_many_predictions_and_boxes_stays_fast_and_bounded() -> None:
    rng = np.random.default_rng(0)
    n_pred, n_gt = 60_000, 4_000
    xy = rng.uniform(0, 900, (n_gt, 2))
    gt = np.column_stack([xy, xy + rng.uniform(5, 60, (n_gt, 2))])
    pxy = rng.uniform(0, 900, (n_pred, 2))
    pred = np.column_stack([pxy, pxy + rng.uniform(5, 60, (n_pred, 2))])
    start = time.perf_counter()
    conf = cf.confusion_matrix(
        np.zeros(n_pred, I64), pred, rng.random(n_pred), rng.integers(0, 3, n_pred).astype(I64),
        np.zeros(n_gt, I64), gt, rng.integers(0, 3, n_gt).astype(I64),
        n_images=1, n_classes=3, score=0.0,
    )  # fmt: skip
    assert time.perf_counter() - start < 10
    assert conf.shape == (4, 4) and conf.sum() >= n_gt  # every box is matched or missed


def test_only_the_highest_scoring_detections_per_image_are_considered() -> None:
    n = cf.CONFUSION_MAX_DETS + 1
    boxes = np.tile(np.array([[200.0, 200.0, 220.0, 220.0]]), (n, 1))
    boxes[0] = [10.0, 10.0, 50.0, 50.0]  # the lowest score is the only one on the object
    scores = np.linspace(0.3, 0.9, n)
    scores[0] = 0.26
    conf = cf.confusion_matrix(
        np.zeros(n, I64), boxes, scores, np.zeros(n, I64),
        np.zeros(1, I64), np.array([[10.0, 10.0, 50.0, 50.0]]), np.zeros(1, I64),
        n_images=1, n_classes=1,
    )  # fmt: skip
    assert conf[0, 1] == 1  # the object counts as missed: its detection fell below the cap
    assert conf[1, 0] == cf.CONFUSION_MAX_DETS  # the others are spurious


def test_identical_stacked_boxes_do_not_explode_the_pair_list() -> None:
    n = 3_000
    box = np.tile(np.array([[10.0, 10.0, 50.0, 50.0]]), (n, 1))
    start = time.perf_counter()
    conf = cf.confusion_matrix(
        np.zeros(n, I64), box, np.linspace(0.3, 0.9, n), np.zeros(n, I64),
        np.zeros(n, I64), box, np.zeros(n, I64), n_images=1, n_classes=1,
    )  # fmt: skip
    assert time.perf_counter() - start < 10 and conf[0, 0] > 0
