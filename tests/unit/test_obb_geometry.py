"""``cvpilot.core.obb``: conversions, canonical corner order, polygon IoU and probabilistic IoU.

The polygon IoU is checked against the axis-aligned IoU of ``cvpilot.core.metrics`` (an
independent formula) on boxes that happen to be axis-aligned, and against closed forms for
rotated ones; the parity suite compares it with an independent polygon library.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cvpilot.core import metrics as m
from cvpilot.core import obb


def rect(cx: float, cy: float, w: float, h: float, r: float = 0.0) -> np.ndarray:
    return obb.xywhr_to_corners(np.array([[cx, cy, w, h, r]]))[0]


def xyxy_corners(boxes: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = boxes.T
    return np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)


def test_corners_follow_the_yolo_obb_order_and_rotation() -> None:
    c = rect(10, 20, 4, 2, math.pi / 2)  # width axis points down (+y) in pixel coordinates
    assert np.allclose(c, [9, 22, 11, 22, 11, 18, 9, 18])


def test_xywhr_round_trip_is_regularised_to_w_ge_h_and_r_in_0_pi() -> None:
    rng = np.random.default_rng(1)
    boxes = np.c_[rng.uniform(0, 100, (50, 2)), rng.uniform(1, 40, (50, 2)), rng.uniform(-7, 7, 50)]
    back = obb.corners_to_xywhr(obb.xywhr_to_corners(boxes))
    assert (back[:, 2] >= back[:, 3]).all() and ((back[:, 4] >= 0) & (back[:, 4] < math.pi)).all()
    assert np.allclose(back[:, :2], boxes[:, :2])
    assert np.allclose(np.sort(back[:, 2:4], axis=1), np.sort(boxes[:, 2:4], axis=1))
    same = obb.polygon_iou_pairs(
        obb.canonical(obb.xywhr_to_corners(back)), obb.canonical(obb.xywhr_to_corners(boxes))
    )
    assert np.allclose(same, 1.0)


def test_min_area_rectangle_of_a_general_quadrilateral_and_of_a_point() -> None:
    kite = np.array([[0, 0, 4, 1, 5, 5, 1, 4.0]])
    w, h = obb.corners_to_xywhr(kite)[0, 2:4]
    assert w * h <= 5 * 5 + 1e-9 and w * h >= obb.polygon_areas(kite)[0]
    point = obb.corners_to_xywhr(np.array([[3, 4, 3, 4, 3, 4, 3, 4.0]]))
    assert np.allclose(point, [[3, 4, 0, 0, 0]])


def test_canonical_order_ignores_starting_corner_and_winding() -> None:
    c = rect(50, 40, 30, 10, 0.3).reshape(4, 2)
    variants = [np.roll(c, k, axis=0) for k in range(4)] + [c[::-1], np.roll(c[::-1], 2, axis=0)]
    out = obb.canonical(np.stack([v.reshape(-1) for v in variants]))
    assert (out == out[0]).all()
    assert obb.signed_areas(out[:1])[0] > 0


def test_convexity_accepts_rectangles_degenerate_boxes_and_rejects_bow_ties() -> None:
    boxes = np.array(
        [
            rect(10, 10, 4, 2, 0.7),
            [0, 0, 1, 1, 1, 0, 0, 1],  # bow tie
            [0, 0, 2, 0, 1, 1, 1, 3],  # concave dart
            [5, 5, 5, 5, 5, 5, 5, 5],  # a point
            [0, 0, 1, 0, 2, 0, 3, 0],  # a line
        ],
        dtype=np.float64,
    )
    assert obb.convex(boxes).tolist() == [True, False, False, True, True]


def test_polygon_iou_equals_the_axis_aligned_iou_on_axis_aligned_boxes() -> None:
    rng = np.random.default_rng(7)
    xy = rng.uniform(0, 100, (400, 2))
    wh = rng.uniform(0, 60, (400, 2))
    a = np.c_[xy, xy + wh]
    b = a + rng.normal(0, 10, a.shape)
    b[:, 2:] = np.maximum(b[:, 2:], b[:, :2])
    b[:50] = a[:50]
    expected = np.diag(m.iou_matrix(a, b))
    got = obb.polygon_iou_pairs(obb.canonical(xyxy_corners(a)), obb.canonical(xyxy_corners(b)))
    finite = np.isfinite(expected)
    assert np.allclose(got[finite], expected[finite], atol=1e-12, rtol=0)


def test_polygon_iou_closed_forms_for_rotated_boxes() -> None:
    square = rect(0, 0, 2, 2)
    diamond = rect(0, 0, 2, 2, math.pi / 4)
    inner = rect(0, 0, 1, 1, 0.4)
    far = rect(10, 10, 2, 2, 0.4)
    a = obb.canonical(np.stack([square, square, square, square]))
    b = obb.canonical(np.stack([diamond, inner, far, square]))
    got = obb.polygon_iou_pairs(a, b)
    octagon = 8 * (math.sqrt(2) - 1)  # square and the same square turned 45 degrees
    assert got[0] == pytest.approx(octagon / (8 - octagon), abs=1e-12)
    assert got[1] == pytest.approx(0.25, abs=1e-12)
    assert got[2] == 0.0 and got[3] == pytest.approx(1.0, abs=1e-12)


def test_polygon_iou_against_a_crowd_region_is_intersection_over_the_first_box() -> None:
    a = obb.canonical(np.stack([rect(0, 0, 1, 1)]))
    b = obb.canonical(np.stack([rect(0, 0, 4, 4)]))
    assert obb.polygon_iou_pairs(a, b, np.array([True]))[0] == pytest.approx(1.0)
    assert obb.polygon_iou_pairs(a, b, np.array([False]))[0] == pytest.approx(1 / 16)


def test_polygon_iou_of_two_empty_boxes_is_nan_and_never_a_match() -> None:
    point = np.array([[1, 1, 1, 1, 1, 1, 1, 1.0]])
    assert math.isnan(obb.polygon_iou_pairs(point, point)[0])


def test_polygon_iou_is_computed_in_chunks(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(3)
    boxes = np.c_[rng.uniform(0, 50, (30, 2)), rng.uniform(5, 20, (30, 2)), rng.uniform(0, 3, 30)]
    a = obb.canonical(obb.xywhr_to_corners(boxes))
    b = obb.canonical(obb.xywhr_to_corners(boxes + [2, 1, 0, 0, 0.1]))
    whole = obb.polygon_iou_pairs(a, b)
    monkeypatch.setattr(obb, "PAIR_CHUNK", 7)
    assert np.array_equal(obb.polygon_iou_pairs(a, b), whole)


def test_probiou_is_symmetric_near_one_for_equal_boxes_and_zero_far_apart() -> None:
    a = np.array([[10, 10, 8, 4, 0.3], [10, 10, 8, 4, 0.3], [0, 0, 4, 4, 0.0]])
    b = np.array([[10, 10, 8, 4, 0.3 + math.pi], [500, 500, 8, 4, 0.3], [1, 0, 4, 4, 0.0]])
    ab, ba = obb.probiou_pairs(a, b), obb.probiou_pairs(b, a)
    assert np.allclose(ab, ba, atol=1e-15)
    assert ab[0] > 0.999 and ab[1] < 1e-6 and 0.5 < ab[2] < 1.0


def test_probiou_ignores_which_side_is_called_width() -> None:
    a = np.array([[10, 10, 8, 4, 0.3]])
    b = np.array([[10, 10, 4, 8, 0.3 + math.pi / 2]])
    c = np.array([[11, 9, 6, 3, 1.0]])
    assert obb.probiou_pairs(a, c)[0] == pytest.approx(obb.probiou_pairs(b, c)[0], abs=1e-12)


def test_oriented_geometry_grid_shapes_and_refuses_crowd_regions() -> None:
    dets = obb.canonical(np.stack([rect(10, 10, 4, 2, 0.2), rect(30, 30, 4, 2)]))
    gts = obb.canonical(np.stack([rect(10, 10, 4, 2, 0.2), rect(12, 10, 4, 2), rect(0, 0, 1, 1)]))
    for geometry in (obb.POLYGON, obb.PROBIOU):
        grid = geometry.grid(dets[None], gts[None], np.zeros((1, 3), dtype=bool))
        assert grid.shape == (1, 2, 3) and grid[0, 0, 0] > 0.99 and grid[0, 1, 2] < 1e-6
        flat = geometry.grid(dets, gts, np.zeros(3, dtype=bool))
        assert np.array_equal(flat, grid[0])
    with pytest.raises(ValueError, match="crowd"):
        obb.POLYGON.grid(dets, gts, np.array([True, False, False]))
    assert np.allclose(obb.POLYGON.areas(gts), [8, 8, 1])
