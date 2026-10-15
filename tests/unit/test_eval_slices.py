from __future__ import annotations

import numpy as np
import pytest

from cvpilot.core.errors import EvalInputError, SliceCardinalityError
from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable, evaluate
from cvpilot.eval import slices as sl

CLASSES = ["a", "b"]


def dataset(
    cameras: list[str], seed: int = 0
) -> tuple[PredictionTable, AnnotationTable, list[str]]:
    rng = np.random.default_rng(seed)
    ids = [f"i{i:04d}" for i in range(len(cameras))]
    g_img, g_cls, g_box, p_img, p_cls, p_score, p_box = [], [], [], [], [], [], []
    for img in ids:
        for _ in range(2):
            x, y = rng.uniform(0, 200, 2)
            box = [x, y, x + rng.uniform(30, 90), y + rng.uniform(30, 90)]
            cls = int(rng.integers(0, 2))
            g_img.append(img)
            g_cls.append(cls)
            g_box.append(box)
            if rng.random() < 0.75:
                p_img.append(img)
                p_cls.append(cls)
                p_score.append(float(rng.random()))
                p_box.append(list(np.array(box) + rng.normal(0, 2, 4)))
    images = ImageTable.from_columns(image_id=ids, meta={"camera": cameras})
    gts = AnnotationTable.from_columns(image_id=g_img, class_id=g_cls, boxes=g_box, images=images)
    preds = PredictionTable.from_columns(image_id=p_img, class_id=p_cls, score=p_score, boxes=p_box)
    return preds, gts, ids


def run(cameras: list[str], **kw: object) -> object:
    preds, gts, _ = dataset(cameras)
    kw.setdefault("slices", {"camera": "camera"})
    return evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=0, **kw)  # type: ignore[arg-type]


def test_slices_report_metrics_and_image_counts_per_value() -> None:
    cameras = ["cam01"] * 6 + ["cam02"] * 4
    result = run(cameras)
    assert set(result.slices) == {"camera=cam01", "camera=cam02"}  # type: ignore[attr-defined]
    assert result.slices["camera=cam01"]["n_images"] == 6.0  # type: ignore[attr-defined]
    assert result.slices["camera=cam02"]["n_images"] == 4.0  # type: ignore[attr-defined]
    for row in result.slices.values():  # type: ignore[attr-defined]
        assert 0.0 <= row["mAP50_95"] <= 1.0


def test_a_slice_scores_like_evaluating_its_images_alone() -> None:
    cameras = ["cam01"] * 6 + ["cam02"] * 4
    preds, gts, ids = dataset(cameras)
    result = evaluate(
        preds,
        gts,
        task="detection",
        class_names=CLASSES,
        n_resamples=0,
        slices={"camera": "camera"},
    )
    keep = {i for i, c in zip(ids, cameras, strict=True) if c == "cam02"}
    sub_gts = AnnotationTable.from_columns(
        image_id=[i for i in gts.image_id if i in keep],
        class_id=gts.class_id[np.isin(gts.image_id, list(keep))],
        boxes=gts.boxes[np.isin(gts.image_id, list(keep))],
    )
    mask = np.isin(preds.image_id, list(keep))
    sub_preds = PredictionTable.from_columns(
        image_id=preds.image_id[mask],
        class_id=preds.class_id[mask],
        score=preds.score[mask],
        boxes=preds.boxes[mask],
    )
    alone = evaluate(sub_preds, sub_gts, task="detection", class_names=CLASSES, n_resamples=0)
    assert result.slices["camera=cam02"]["mAP50_95"] == pytest.approx(alone.metrics["mAP50_95"])


def test_several_slice_keys_and_the_prefix_form() -> None:
    preds, gts, ids = dataset(["x"] * 4 + ["y"] * 4)
    images = ImageTable.from_columns(
        image_id=ids, meta={"camera": ["x"] * 4 + ["y"] * 4, "site": ["n", "s"] * 4}
    )
    gts = AnnotationTable(gts.image_id, gts.class_id, gts.boxes, gts.iscrowd, images)
    result = evaluate(
        preds,
        gts,
        task="detection",
        class_names=CLASSES,
        n_resamples=0,
        slices={"cam": "meta.camera", "site": "site"},
    )
    assert set(result.slices) == {"cam=x", "cam=y", "site=n", "site=s"}


def test_classification_slices_report_top1() -> None:
    ids = [f"i{i}" for i in range(6)]
    images = ImageTable.from_columns(image_id=ids, meta={"cam": ["a", "a", "a", "b", "b", "b"]})
    gts = AnnotationTable.from_columns(image_id=ids, class_id=[0, 0, 1, 1, 1, 0], images=images)
    preds = PredictionTable.from_columns(image_id=ids, class_id=[0, 0, 0, 1, 0, 0], score=[0.5] * 6)
    result = evaluate(
        preds, gts, task="classification", class_names=CLASSES, n_resamples=0, slices={"cam": "cam"}
    )
    assert result.slices["cam=a"] == {"top1": pytest.approx(2 / 3), "n_images": 3.0}
    assert result.slices["cam=b"] == {"top1": pytest.approx(2 / 3), "n_images": 3.0}


def test_a_slice_without_countable_ground_truth_reports_only_its_size() -> None:
    preds, gts, ids = dataset(["x"] * 3 + ["y"] * 3)
    crowd = np.isin(gts.image_id, ids[3:])
    gts = AnnotationTable(gts.image_id, gts.class_id, gts.boxes, crowd, gts.images)
    result = evaluate(
        preds,
        gts,
        task="detection",
        class_names=CLASSES,
        n_resamples=0,
        slices={"camera": "camera"},
    )
    assert result.slices["camera=y"] == {"n_images": 3.0}
    assert "mAP50_95" in result.slices["camera=x"]


def test_slices_flow_into_a_valid_snapshot() -> None:
    result = run(["x"] * 3 + ["y"] * 3)
    snap = result.to_snapshot(  # type: ignore[attr-defined]
        run_id="0" * 26, step=0, confusion_sha256="a" * 64, predictions_sha256="b" * 64
    )
    assert set(snap.slices) == {"camera=x", "camera=y"}


# -- hostile input ----------------------------------------------------------------------------


def test_unbounded_slice_cardinality_is_rejected_before_any_metric_is_computed() -> None:
    cameras = [f"cam{i:04d}" for i in range(sl.MAX_SLICE_VALUES + 1)]
    with pytest.raises(SliceCardinalityError, match=r"camera.*257.*256"):
        run(cameras)


def test_the_cardinality_limit_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "MAX_SLICE_VALUES", 3)
    run(["a", "b", "c", "a"])
    with pytest.raises(SliceCardinalityError):
        run(["a", "b", "c", "d"])


def test_the_number_of_slice_keys_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sl, "MAX_SLICE_KEYS", 1)
    preds, gts, ids = dataset(["x", "y"])
    images = ImageTable.from_columns(image_id=ids, meta={"c": ["x", "y"], "d": ["x", "y"]})
    gts = AnnotationTable(gts.image_id, gts.class_id, gts.boxes, gts.iscrowd, images)
    with pytest.raises(EvalInputError, match="slice keys"):
        evaluate(
            preds, gts, task="detection", class_names=CLASSES, n_resamples=0,
            slices={"c": "c", "d": "d"},
        )  # fmt: skip


@pytest.mark.parametrize(
    "slices", [{"bad key": "camera"}, {"1abc": "camera"}, {"a=b": "camera"}, {"": "camera"}]
)
def test_slice_key_names_are_restricted(slices: dict[str, str]) -> None:
    with pytest.raises(EvalInputError, match="slice key"):
        run(["x", "y"], slices=slices)


def test_unknown_metadata_columns_are_rejected() -> None:
    with pytest.raises(EvalInputError, match="no metadata column 'ghost'"):
        run(["x", "y"], slices={"camera": "ghost"})


@pytest.mark.parametrize("value", ["", "a\nb", "x" * 129, "\x00"])
def test_slice_values_are_validated(value: str) -> None:
    with pytest.raises(EvalInputError, match="slice value"):
        run(["ok", value])


# -- review fixes: lazy masks and a work budget ----------------------------------------------


def test_slices_hold_no_per_slice_arrays_and_build_masks_on_demand() -> None:
    from cvpilot.eval.validate import prepare_ground_truth

    preds, gts, _ = dataset(["a", "b", "a", "c"])
    plan = sl.plan_slices({"camera": "camera"}, prepare_ground_truth(gts, 2, "detection"), 10)
    assert [p.label for p in plan] == ["camera=a", "camera=b", "camera=c"]
    assert plan[0].mask().tolist() == [True, False, True, False]
    assert all(not hasattr(p, "masks") and callable(p.mask) for p in plan)
    assert plan[0].codes is plan[1].codes  # one code array per key, shared by its slices


def test_the_slice_work_budget_is_enforced_and_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    preds, gts, _ = dataset(["a", "b", "a", "c"])
    work = len(preds) + len(gts)
    monkeypatch.setattr(sl, "MAX_SLICE_WORK", 3 * work)  # three slices at exactly the limit
    evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=0,
             slices={"camera": "camera"})  # fmt: skip
    monkeypatch.setattr(sl, "MAX_SLICE_WORK", 3 * work - 1)
    with pytest.raises(EvalInputError, match="slice work"):
        evaluate(preds, gts, task="detection", class_names=CLASSES, n_resamples=0,
                 slices={"camera": "camera"})  # fmt: skip
