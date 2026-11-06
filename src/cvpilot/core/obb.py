"""Oriented bounding boxes on NumPy arrays: conversions, areas and two IoU definitions.

A box is four corners ``x1, y1, x2, y2, x3, y3, x4, y4`` in pixels, in order around the box
(either winding), forming a convex quadrilateral; an oriented rectangle is the usual case. The
other common form, ``xywhr`` (centre, width, height, rotation in radians), converts at the edges
with :func:`xywhr_to_corners` and :func:`corners_to_xywhr`. The rotation follows the YOLO-OBB
convention in pixel coordinates (``y`` points down): the width axis is ``(cos r, sin r)``, so a
positive ``r`` turns clockwise on screen.

Two IoU definitions (ADR-0013):

* ``polygon``: exact area of the intersection polygon over the area of the union (DOTA style).
  The intersection of two convex quadrilaterals is the convex hull of the corners of each that
  lie inside the other and of the edge crossings; its area follows from the shoelace formula.
* ``probiou``: one minus the Hellinger distance between the Gaussians of the two boxes (Llerena
  et al., 2021, arXiv:2106.06072), the definition the YOLO-OBB validator matches with. A box
  becomes a Gaussian through its minimum-area enclosing rectangle (exact for rectangles).

Everything here is vectorised over box pairs and evaluated in bounded chunks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cvpilot.core.metrics import F64, MATCH_BUDGET, BoolArray

OBB_WIDTH = 8
"""Coordinates per oriented box (four corners)."""
PAIR_CHUNK = 1 << 15
"""Box pairs whose IoU is computed at once; bounds the temporary arrays (about 64 MB)."""
OBB_BUDGET = MATCH_BUDGET >> 4
"""Padded ``groups x detections x boxes`` cells matched at once for oriented boxes: each cell
carries both boxes' corners while it is evaluated, so the block is smaller than for axis-aligned
boxes."""
PROBIOU_EPS = 1e-7
"""The epsilon of the reference probabilistic IoU; kept equal so the values agree."""
_SIDE_TOLERANCE = 1e-9
"""Relative tolerance of the inside and on-edge tests in the polygon intersection."""


def _corners(boxes: F64) -> F64:
    return np.asarray(boxes, dtype=np.float64).reshape(-1, 4, 2)


def _cross(ax: F64, ay: F64, bx: F64, by: F64) -> F64:
    out: F64 = ax * by - ay * bx
    return out


def signed_areas(boxes: F64) -> F64:
    """Signed shoelace area of ``(n, 8)`` corner boxes; positive for one winding, negative for
    the other."""
    p = _corners(boxes)
    x, y = p[:, :, 0], p[:, :, 1]
    out: F64 = 0.5 * np.sum(x * np.roll(y, -1, axis=1) - np.roll(x, -1, axis=1) * y, axis=1)
    return out


def polygon_areas(boxes: F64) -> F64:
    """Area of ``(n, 8)`` corner boxes."""
    return np.abs(signed_areas(boxes))


def convex(boxes: F64) -> BoolArray:
    """Which ``(n, 8)`` corner boxes are convex quadrilaterals in order around the box (either
    winding; degenerate boxes with zero area included). A self-crossing ("bow-tie") order is not."""
    p = _corners(boxes)
    edge = np.roll(p, -1, axis=1) - p
    nxt = np.roll(edge, -1, axis=1)
    turn = _cross(edge[:, :, 0], edge[:, :, 1], nxt[:, :, 0], nxt[:, :, 1])
    span = np.max(np.abs(edge), axis=(1, 2))
    tol = _SIDE_TOLERANCE * span * span
    same_turn = ((turn >= -tol[:, None]).all(axis=1)) | ((turn <= tol[:, None]).all(axis=1))
    # a convex quadrilateral turns once around: the edge directions cover 360 degrees, which a
    # bow-tie (turns of one sign that wind twice) cannot do with four edges, so the sign test is
    # sufficient for four corners
    return np.asarray(same_turn, dtype=bool)


def canonical(boxes: F64) -> F64:
    """The same boxes with positive winding, starting at the corner with the smallest ``x``
    (then ``y``): one representation per box, whatever order the corners came in."""
    p = _corners(boxes).copy()
    flip = signed_areas(boxes) < 0
    p[flip] = p[flip][:, ::-1, :]
    xs, ys = p[:, :, 0], p[:, :, 1]
    first_x = xs == xs.min(axis=1, keepdims=True)
    start = np.argmin(np.where(first_x, ys, np.inf), axis=1)
    idx = (start[:, None] + np.arange(4)[None, :]) % 4
    out: F64 = np.take_along_axis(p, idx[:, :, None], axis=1).reshape(-1, OBB_WIDTH)
    return out


def xywhr_to_corners(xywhr: F64) -> F64:
    """``(n, 5)`` boxes ``cx, cy, w, h, r`` to ``(n, 8)`` corners (same corner order as the
    YOLO-OBB tools)."""
    b = np.asarray(xywhr, dtype=np.float64).reshape(-1, 5)
    cx, cy, w, h, r = (b[:, i] for i in range(5))
    cos, sin = np.cos(r), np.sin(r)
    v1 = np.stack([w / 2 * cos, w / 2 * sin], axis=1)
    v2 = np.stack([-h / 2 * sin, h / 2 * cos], axis=1)
    ctr = np.stack([cx, cy], axis=1)
    out: F64 = np.concatenate([ctr + v1 + v2, ctr + v1 - v2, ctr - v1 - v2, ctr - v1 + v2], axis=1)
    return out


def corners_to_xywhr(boxes: F64) -> F64:
    """``(n, 8)`` convex corner boxes to ``(n, 5)`` ``cx, cy, w, h, r`` of their minimum-area
    enclosing rectangle (exact for rectangles), with ``w >= h`` and ``r`` in ``[0, pi)``.

    The minimum-area rectangle of a convex polygon has a side along one of its edges, so each of
    the four edge directions is tried and the smallest rectangle wins (ties: the first edge).
    """
    p = _corners(boxes)
    edge = np.roll(p, -1, axis=1) - p  # (n, 4 edges, 2)
    length = np.hypot(edge[:, :, 0], edge[:, :, 1])
    safe = np.where(length > 0, length, 1.0)
    u = edge / safe[:, :, None]  # (n, 4, 2) unit direction per candidate edge
    v = np.stack([-u[:, :, 1], u[:, :, 0]], axis=2)
    pu = np.einsum("nkc,npc->nkp", u, p)  # (n, candidate, corner)
    pv = np.einsum("nkc,npc->nkp", v, p)
    w = pu.max(axis=2) - pu.min(axis=2)
    h = pv.max(axis=2) - pv.min(axis=2)
    area = np.where(length > 0, w * h, np.inf)
    best = np.argmin(area, axis=1)
    rows = np.arange(p.shape[0])
    has_edge = np.isfinite(area[rows, best])
    uu, vv = u[rows, best], v[rows, best]
    mu = (pu.max(axis=2) + pu.min(axis=2))[rows, best] / 2
    mv = (pv.max(axis=2) + pv.min(axis=2))[rows, best] / 2
    centre = uu * mu[:, None] + vv * mv[:, None]
    centre = np.where(has_edge[:, None], centre, p[:, 0, :])
    ww = np.where(has_edge, w[rows, best], 0.0)
    hh = np.where(has_edge, h[rows, best], 0.0)
    r = np.where(has_edge, np.arctan2(uu[:, 1], uu[:, 0]), 0.0)
    wide = ww >= hh
    out_w, out_h = np.where(wide, ww, hh), np.where(wide, hh, ww)
    out_r = np.mod(r + np.where(wide, 0.0, np.pi / 2), np.pi)
    out: F64 = np.stack([centre[:, 0], centre[:, 1], out_w, out_h, out_r], axis=1)
    return out


def _inside(points: F64, poly: F64, scale: F64) -> BoolArray:
    """``points`` ``(P, m, 2)`` inside or on the convex, positively wound ``poly`` ``(P, 4, 2)``."""
    edge = np.roll(poly, -1, axis=1) - poly  # (P, 4, 2)
    rel = points[:, :, None, :] - poly[:, None, :, :]  # (P, m, 4, 2)
    side = _cross(edge[:, None, :, 0], edge[:, None, :, 1], rel[..., 0], rel[..., 1])
    tol = _SIDE_TOLERANCE * scale[:, None, None]
    out: BoolArray = (side >= -tol).all(axis=2)
    return out


def _intersection_areas(a: F64, b: F64) -> F64:
    """Intersection area of positively wound convex quadrilaterals ``(P, 4, 2)``, pairwise."""
    n = a.shape[0]
    centre = (a.mean(axis=1) + b.mean(axis=1)) / 2  # work near the origin: less cancellation
    a = a - centre[:, None, :]
    b = b - centre[:, None, :]
    span = np.maximum(np.abs(a).max(axis=(1, 2)), np.abs(b).max(axis=(1, 2)))
    scale = np.maximum(span * span, np.finfo(np.float64).tiny)
    ra = np.roll(a, -1, axis=1) - a  # edges (P, 4, 2)
    rb = np.roll(b, -1, axis=1) - b
    denom = _cross(ra[:, :, None, 0], ra[:, :, None, 1], rb[:, None, :, 0], rb[:, None, :, 1])
    qp = b[:, None, :, :] - a[:, :, None, :]  # (P, i, j, 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = _cross(qp[..., 0], qp[..., 1], rb[:, None, :, 0], rb[:, None, :, 1]) / denom
        s = _cross(qp[..., 0], qp[..., 1], ra[:, :, None, 0], ra[:, :, None, 1]) / denom
    eps = _SIDE_TOLERANCE
    crossing = (
        (np.abs(denom) > eps * scale[:, None, None])
        & (t >= -eps)
        & (t <= 1 + eps)
        & (s >= -eps)
        & (s <= 1 + eps)
    )
    t = np.where(crossing, t, 0.0)
    hits = (a[:, :, None, :] + t[..., None] * ra[:, :, None, :]).reshape(n, 16, 2)
    points = np.concatenate([a, b, hits], axis=1)  # (P, 24, 2)
    valid = np.concatenate(
        [_inside(a, b, scale), _inside(b, a, scale), crossing.reshape(n, 16)], axis=1
    )
    count = valid.sum(axis=1)
    mid = np.where(
        count[:, None] > 0,
        (points * valid[:, :, None]).sum(axis=1) / np.maximum(count, 1)[:, None],
        0.0,
    )
    rel = points - mid[:, None, :]
    angle = np.where(valid, np.arctan2(rel[:, :, 1], rel[:, :, 0]), np.inf)
    order = np.argsort(angle, axis=1, kind="stable")
    ring = np.take_along_axis(rel, order[:, :, None], axis=1)
    ok = np.take_along_axis(valid, order, axis=1)
    ring = np.where(ok[:, :, None], ring, ring[:, :1, :])  # pad with the first point: closes it
    nxt = np.roll(ring, -1, axis=1)
    twice = _cross(ring[:, :, 0], ring[:, :, 1], nxt[:, :, 0], nxt[:, :, 1]).sum(axis=1)
    out: F64 = np.where(count >= 3, np.abs(twice) / 2, 0.0)
    return out


def polygon_iou_pairs(a: F64, b: F64, crowd: BoolArray | None = None) -> F64:
    """Exact IoU of ``(P, 8)`` box pairs (positively wound, see :func:`canonical`); against a
    crowd region it is the intersection over the area of ``a``. ``nan`` when both are empty."""
    p = a.shape[0]
    out: F64 = np.empty(p, dtype=np.float64)
    for lo in range(0, p, PAIR_CHUNK):
        ca, cb = _corners(a[lo : lo + PAIR_CHUNK]), _corners(b[lo : lo + PAIR_CHUNK])
        area_a = polygon_areas(a[lo : lo + PAIR_CHUNK])
        area_b = polygon_areas(b[lo : lo + PAIR_CHUNK])
        # an empty box overlaps nothing (its zero-length edges would pass every inside test),
        # and the intersection never exceeds either box (rounding near shared edges)
        inter = np.where(
            (area_a > 0) & (area_b > 0),
            np.minimum(_intersection_areas(ca, cb), np.minimum(area_a, area_b)),
            0.0,
        )
        union = area_a + area_b - inter
        if crowd is not None:
            union = np.where(crowd[lo : lo + PAIR_CHUNK], area_a, union)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[lo : lo + PAIR_CHUNK] = inter / union
    return out


def _covariance(xywhr: F64) -> tuple[F64, F64, F64]:
    a = xywhr[:, 2] ** 2 / 12
    b = xywhr[:, 3] ** 2 / 12
    cos, sin = np.cos(xywhr[:, 4]), np.sin(xywhr[:, 4])
    return a * cos**2 + b * sin**2, a * sin**2 + b * cos**2, (a - b) * cos * sin


def probiou_pairs(a: F64, b: F64) -> F64:
    """Probabilistic IoU of ``(P, 5)`` ``xywhr`` box pairs, as the YOLO-OBB validator computes
    it (same epsilon and clamping), in float64."""
    eps = PROBIOU_EPS
    a1, b1, c1 = _covariance(a)
    a2, b2, c2 = _covariance(b)
    dx, dy = a[:, 0] - b[:, 0], a[:, 1] - b[:, 1]
    sa, sb, sc = a1 + a2, b1 + b2, c1 + c2
    det = sa * sb - sc**2
    t1 = (sa * dy**2 + sb * dx**2) / (det + eps) * 0.25
    t2 = (sc * (-dx) * dy) / (det + eps) * 0.5
    own = np.sqrt(np.clip(a1 * b1 - c1**2, 0, None) * np.clip(a2 * b2 - c2**2, 0, None))
    t3 = np.log(det / (4 * own + eps) + eps) * 0.5
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    out: F64 = 1.0 - hd
    return out


@dataclass(frozen=True)
class OrientedGeometry:
    """Box geometry for ``cvpilot.core.metrics``: oriented boxes under one IoU definition."""

    iou_type: str
    width: int = OBB_WIDTH
    budget: int = OBB_BUDGET

    def areas(self, boxes: F64) -> F64:
        return polygon_areas(boxes)

    def grid(self, dets: F64, gts: F64, crowd: BoolArray) -> F64:
        """IoU of ``(..., D, 8)`` detections against ``(..., G, 8)`` boxes: ``(..., D, G)``."""
        if bool(np.any(crowd)):
            raise ValueError("crowd regions are not supported for oriented boxes")
        lead = dets.shape[:-2]
        n_det, n_gt = dets.shape[-2], gts.shape[-2]
        shape = (*lead, n_det, n_gt)
        if self.iou_type == "probiou":
            da = corners_to_xywhr(dets.reshape(-1, OBB_WIDTH)).reshape(*lead, n_det, 5)
            ga = corners_to_xywhr(gts.reshape(-1, OBB_WIDTH)).reshape(*lead, n_gt, 5)
            pair: Callable[[F64, F64], F64] = probiou_pairs
            width = 5
        else:
            da, ga, pair, width = dets, gts, polygon_iou_pairs, OBB_WIDTH
        left = np.broadcast_to(da[..., :, None, :], (*shape, width)).reshape(-1, width)
        right = np.broadcast_to(ga[..., None, :, :], (*shape, width)).reshape(-1, width)
        out: F64 = pair(left, right).reshape(shape)
        return out


POLYGON = OrientedGeometry("polygon")
PROBIOU = OrientedGeometry("probiou")
