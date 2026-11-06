"""Task ``obb``: oriented boxes through ``evaluate``, ``log_eval``, recompute and compare."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import structlog
from support import build_snapshot

from cvpilot import api
from cvpilot.compare.engine import check_comparable
from cvpilot.compare.recompute import RecomputeSpec, verify
from cvpilot.core import obb
from cvpilot.core.errors import CompareRefusedError, EvalInputError, RecomputeMismatchError
from cvpilot.core.schema import EvalParams, EvalSnapshot
from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable, eval_set_hash, evaluate
from cvpilot.eval.tables import OBB_COLUMNS
from cvpilot.logging import clear_request_context
from cvpilot.store.local import open_store

CLASSES = ["can", "bottle", "box"]


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_LOG_LEVEL", "CVPILOT_PROJECT"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    yield
    clear_request_context()
    structlog.reset_defaults()


def problem(
    seed: int = 0, n_images: int = 12, *, axis_aligned: bool = False
) -> tuple[dict[str, Any], dict[str, Any], ImageTable]:
    """Random oriented boxes on 640 x 480 images, with jittered, missed and extra detections."""
    rng = np.random.default_rng(seed)
    g_img, g_cls, g_box, p_img, p_cls, p_score, p_box = [], [], [], [], [], [], []
    for img in range(n_images):
        for _ in range(int(rng.integers(0, 7))):
            w, h = np.exp(rng.uniform(np.log(6), np.log(150), 2))
            r = 0.0 if axis_aligned else float(rng.uniform(0, math.pi))
            reach = 0.5 * math.hypot(w, h) + 1
            cx, cy = rng.uniform(reach, 640 - reach), rng.uniform(reach, 480 - reach)
            c = int(rng.integers(0, len(CLASSES)))
            g_img.append(f"img{img:03d}")
            g_cls.append(c)
            g_box.append([cx, cy, w, h, r])
            if rng.random() < 0.8:
                jit = rng.normal(0, 0.06, 5) * [w, h, w, h, 0.0 if axis_aligned else 1.0]
                p_img.append(f"img{img:03d}")
                p_cls.append(c if rng.random() < 0.85 else int(rng.integers(0, len(CLASSES))))
                p_score.append(float(rng.random()))
                p_box.append(list(np.array([cx, cy, w, h, r]) + jit))
        for _ in range(int(rng.integers(0, 3))):
            p_img.append(f"img{img:03d}")
            p_cls.append(int(rng.integers(0, len(CLASSES))))
            p_score.append(float(rng.random()))
            p_box.append([*rng.uniform(40, 400, 2), *rng.uniform(5, 60, 2), 0.0])
    images = ImageTable.from_columns(
        image_id=[f"img{i:03d}" for i in range(n_images)],
        width=[640] * n_images,
        height=[480] * n_images,
        meta={"shelf": [f"s{i % 3}" for i in range(n_images)]},
    )
    gts = {"image_id": g_img, "class_id": g_cls, "boxes": obb.xywhr_to_corners(np.array(g_box))}
    preds = {
        "image_id": p_img,
        "class_id": p_cls,
        "score": p_score,
        "boxes": obb.xywhr_to_corners(np.array(p_box)),
    }
    return preds, gts, images


def tables(
    preds: dict[str, Any], gts: dict[str, Any], images: ImageTable
) -> tuple[PredictionTable, AnnotationTable]:
    return PredictionTable.from_columns(**preds), AnnotationTable.from_columns(**gts, images=images)


def to_xyxy(corners: np.ndarray) -> np.ndarray:
    xs, ys = corners[:, 0::2], corners[:, 1::2]
    return np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)


def write_obb_gts(path: Path, gts: dict[str, Any], images: ImageTable) -> Path:
    """The ground truth as ``cvpilot eval`` reads it; a null ``cls`` row per image without boxes."""
    empty = sorted(set(images.image_id.tolist()) - set(gts["image_id"]))
    n = len(gts["image_id"])
    columns: dict[str, Any] = {
        "image_id": list(gts["image_id"]) + empty,
        "cls": pa.array(list(gts["class_id"]) + [None] * len(empty), type=pa.int64()),
        "width": [640] * (n + len(empty)),
        "height": [480] * (n + len(empty)),
    }
    for i, name in enumerate(OBB_COLUMNS):
        columns[name] = pa.array(gts["boxes"][:, i].tolist() + [None] * len(empty))
    pq.write_table(pa.table(columns), path)
    return path


def test_perfect_oriented_predictions_score_one_everywhere() -> None:
    _, gts, images = problem(1)
    preds = {**gts, "score": [0.9] * len(gts["image_id"])}
    p, g = tables(preds, gts, images)
    for kind in ("polygon", "probiou"):
        res = evaluate(p, g, task="obb", class_names=CLASSES, n_resamples=0, iou_type=kind)
        assert res.metrics["mAP50_95"] == pytest.approx(1.0) and res.metrics["mAP50"] == 1.0
        assert all(row["AP50_95"] == pytest.approx(1.0) for row in res.per_class.values())
        assert res.params is not None and res.params.iou_type == kind
        assert np.trace(res.confusion) == len(gts["image_id"])


def test_polygon_mode_on_axis_aligned_boxes_equals_detection_metric_for_metric() -> None:
    preds, gts, images = problem(2, 30, axis_aligned=True)
    po, go = tables(preds, gts, images)
    pd = PredictionTable.from_columns(**{**preds, "boxes": to_xyxy(preds["boxes"])})
    gd = AnnotationTable.from_columns(**{**gts, "boxes": to_xyxy(gts["boxes"])}, images=images)
    kw: dict[str, Any] = {"class_names": CLASSES, "slices": {"shelf": "shelf"}, "n_resamples": 50}
    o = evaluate(po, go, task="obb", **kw)
    d = evaluate(pd, gd, task="detection", **kw)
    assert set(o.metrics) == set(d.metrics) and len(o.metrics) >= 8
    for name in d.metrics:
        assert o.metrics[name] == pytest.approx(d.metrics[name], abs=1e-9), name
    for cls in CLASSES:
        assert o.per_class[cls] == pytest.approx(d.per_class[cls], abs=1e-9)
    assert set(o.slices) == set(d.slices)
    for label, row in d.slices.items():
        assert o.slices[label] == pytest.approx(row, abs=1e-9)
    assert np.array_equal(o.confusion, d.confusion)
    assert o.ci == pytest.approx(d.ci, abs=1e-9)
    assert o.params is not None and o.params.iou_type == "polygon"
    assert d.params is not None and d.params.iou_type == "box"


def test_probiou_and_polygon_disagree_on_rotated_boxes_and_record_their_type() -> None:
    p, g = tables(*problem(3, 30))
    poly = evaluate(p, g, task="obb", class_names=CLASSES, n_resamples=0)
    prob = evaluate(p, g, task="obb", class_names=CLASSES, n_resamples=0, iou_type="probiou")
    assert poly.eval_set_hash == prob.eval_set_hash
    assert poly.metrics["mAP50_95"] != prob.metrics["mAP50_95"]
    assert 0.0 < poly.metrics["mAP50_95"] < 1.0 and 0.0 < prob.metrics["mAP50_95"] < 1.0


def test_corner_order_and_winding_do_not_change_the_hash_or_the_metrics() -> None:
    preds, gts, images = problem(4)
    p, g = tables(preds, gts, images)
    corners = gts["boxes"].reshape(-1, 4, 2)
    shuffled = corners[:, [2, 1, 0, 3], :].reshape(-1, 8)  # reversed winding, another start
    g2 = AnnotationTable.from_columns(**{**gts, "boxes": shuffled}, images=images)
    a = evaluate(p, g, task="obb", class_names=CLASSES, n_resamples=0)
    b = evaluate(p, g2, task="obb", class_names=CLASSES, n_resamples=0)
    assert a.eval_set_hash == b.eval_set_hash == eval_set_hash(g2, CLASSES, task="obb")
    assert a.metrics == b.metrics


@pytest.mark.parametrize(
    ("task", "iou_type", "match"),
    [
        ("detection", "polygon", "does not apply to task detection"),
        ("obb", "box", "does not apply to task obb"),
        ("classification", "probiou", "does not apply"),
        ("segmentation", None, "unknown task"),
    ],
)
def test_iou_type_must_suit_the_task(task: str, iou_type: str | None, match: str) -> None:
    p, g = tables(*problem(5))
    with pytest.raises(EvalInputError, match=match):
        evaluate(p, g, task=task, class_names=CLASSES, iou_type=iou_type)  # type: ignore[arg-type]


def test_invalid_oriented_inputs_are_refused() -> None:
    preds, gts, images = problem(6)
    p, g = tables(preds, gts, images)
    detection_boxes = PredictionTable.from_columns(**{**preds, "boxes": to_xyxy(preds["boxes"])})
    with pytest.raises(EvalInputError, match="need 8 coordinates"):
        evaluate(detection_boxes, g, task="obb", class_names=CLASSES)
    bow = preds["boxes"].copy()
    bow[0] = [0, 0, 10, 10, 10, 0, 0, 10]
    with pytest.raises(EvalInputError, match="not a convex quadrilateral"):
        evaluate(PredictionTable.from_columns(**{**preds, "boxes": bow}), g, task="obb",
                 class_names=CLASSES)  # fmt: skip
    crowd = AnnotationTable.from_columns(
        **gts, iscrowd=[True] + [False] * (len(gts["image_id"]) - 1), images=images
    )
    with pytest.raises(EvalInputError, match="iscrowd is not supported"):
        evaluate(p, crowd, task="obb", class_names=CLASSES)
    outside = gts["boxes"].copy()
    outside[0, 0::2] += 700.0
    with pytest.raises(EvalInputError, match="outside the image"):
        evaluate(p, AnnotationTable.from_columns(**{**gts, "boxes": outside}, images=images),
                 task="obb", class_names=CLASSES)  # fmt: skip
    nan = preds["boxes"].copy()
    nan[0, 3] = np.nan
    with pytest.raises(EvalInputError, match="non-finite"):
        evaluate(PredictionTable.from_columns(**{**preds, "boxes": nan}), g, task="obb",
                 class_names=CLASSES)  # fmt: skip
    far = preds["boxes"].copy()
    far[0, 2] = 1e9
    with pytest.raises(EvalInputError, match="outside"):
        evaluate(PredictionTable.from_columns(**{**preds, "boxes": far}), g, task="obb",
                 class_names=CLASSES)  # fmt: skip


def test_an_empty_prediction_table_is_valid_for_obb() -> None:
    _, gts, images = problem(7)
    empty = PredictionTable.from_columns(image_id=[], class_id=[], score=[])
    res = evaluate(empty, AnnotationTable.from_columns(**gts, images=images), task="obb",
                   class_names=CLASSES, n_resamples=0)  # fmt: skip
    assert res.metrics["mAP50_95"] == 0.0


def test_log_eval_stores_an_obb_snapshot_that_recomputes_and_compares(tmp_path: Path) -> None:
    preds, gts, images = problem(8, 20)
    p, g = tables(preds, gts, images)
    run = api.init("proj", task="obb", name="r")
    snap = run.log_eval(p, g, step=1, class_names=CLASSES, n_resamples=20, iou_type="probiou")
    other = run.log_eval(p, g, step=2, class_names=CLASSES, n_resamples=20)
    run.finish()
    assert snap.task == "obb" and snap.params is not None and snap.params.iou_type == "probiou"
    assert other.params is not None and other.params.iou_type == "polygon"
    with pytest.raises(CompareRefusedError) as refused:
        check_comparable(other, snap)
    assert refused.value.reason == "iou_type"
    store = open_store("proj", tmp_path / "store")
    try:
        with store.open_artifact(snap.predictions_sha256) as handle:
            table = pq.read_table(handle)
        assert [c for c in table.column_names if c.startswith(("x", "y"))] == list(OBB_COLUMNS)
        gt_path = write_obb_gts(tmp_path / "g.parquet", gts, images)
        verify(store, snap, RecomputeSpec(gt_path, CLASSES))
        verify(store, other, RecomputeSpec(gt_path, CLASSES))
        forged = EvalSnapshot.model_validate(
            {**json.loads(snap.model_dump_json()), "params": {
                **json.loads(snap.params.model_dump_json()), "iou_type": "polygon"}}
        )  # fmt: skip
        with pytest.raises(RecomputeMismatchError):
            verify(store, forged, RecomputeSpec(gt_path, CLASSES))
    finally:
        store.close()


def test_a_run_refuses_an_iou_type_its_task_does_not_take() -> None:
    p, g = tables(*problem(9))
    run = api.init("proj", task="detection", name="r")
    try:
        with pytest.raises(EvalInputError, match="does not apply"):
            run.log_eval(p, g, step=0, iou_type="probiou")
    finally:
        run.finish()


def test_snapshot_params_and_task_must_agree_and_old_params_read_as_box() -> None:
    assert (
        EvalParams.model_validate(
            {"iou_thresholds": [0.5], "confusion_iou": 0.5, "confusion_score": 0.25}
        ).iou_type
        == "box"
    )
    params = {"iou_thresholds": [0.5], "confusion_iou": 0.5, "confusion_score": 0.25}
    with pytest.raises(ValueError, match="does not apply to obb"):
        build_snapshot(task="obb", params=params)
    with pytest.raises(ValueError, match="does not apply to detection"):
        build_snapshot(task="detection", params={**params, "iou_type": "polygon"})
    a = build_snapshot(task="obb", params={**params, "iou_type": "polygon"})
    b = build_snapshot(task="obb", params={**params, "iou_type": "probiou"}, run_id=None)
    with pytest.raises(CompareRefusedError, match="IoU definitions differ") as refused:
        check_comparable(a, b)
    assert refused.value.reason == "iou_type"
    check_comparable(a, build_snapshot(task="obb", params={**params, "iou_type": "polygon"}))
