"""Parity of ``cvpilot.core.metrics`` with pycocotools COCOeval.

Tolerance: 1e-12 absolute (the documented requirement is 1e-4) on every COCO summary number
and on per-class AP, AP50 and AR100. Reference versions
are pinned; a different pycocotools fails the test instead of silently loosening the check.
Boxes here carry ``area = w * h``, which is what ``cvpilot`` computes.
"""

from __future__ import annotations

import contextlib
import io
from importlib import metadata
from typing import Any

import numpy as np
import pytest

from cvpilot.core import metrics as m

pytestmark = pytest.mark.parity

PINNED = {"pycocotools": "2.0.11"}
TOLERANCE = 1e-12
"""Absolute tolerance. The protocol is implemented exactly, so differences are float noise;
the documented requirement of the architecture is 1e-4."""
RANDOM_CASES = 300
STAT_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)

Case = dict[str, Any]


def _require_pinned_versions() -> None:
    for name, version in PINNED.items():
        assert metadata.version(name) == version, f"{name} must be pinned to {version}"


def make_case(
    rng: np.random.Generator,
    *,
    n_images: int,
    n_classes: int,
    max_gt: int = 8,
    det_rate: float = 0.7,
    crowd_rate: float = 0.1,
    tie_scores: bool = False,
    extra_dets: int = 4,
) -> Case:
    """A random detection problem; boxes span all three COCO size ranges."""

    def box() -> list[float]:
        w, h = np.exp(rng.uniform(np.log(4), np.log(300), 2))
        x, y = rng.uniform(0, 400, 2)
        return [float(x), float(y), float(x + w), float(y + h)]

    gts: list[tuple[int, int, list[float], bool]] = []
    dets: list[tuple[int, int, float, list[float]]] = []
    for img in range(n_images):
        for _ in range(int(rng.integers(0, max_gt + 1))):
            b, c = box(), int(rng.integers(0, n_classes))
            crowd = bool(rng.random() < crowd_rate)
            gts.append((img, c, b, crowd))
            if rng.random() < det_rate:
                jitter = rng.normal(0, 0.05, 4) * np.array([b[2] - b[0], b[3] - b[1]] * 2)
                cls = c if rng.random() < 0.85 else int(rng.integers(0, n_classes))
                dets.append(
                    (img, cls, float(rng.random()), [float(v) for v in np.array(b) + jitter])
                )
        for _ in range(int(rng.integers(0, extra_dets + 1))):
            dets.append((img, int(rng.integers(0, n_classes)), float(rng.random()), box()))
    if tie_scores:
        dets = [(i, c, round(s, 1), b) for i, c, s, b in dets]
    # sorted() of zero-padded ids equals numeric order, which is what COCOeval uses for ties
    order = rng.permutation(len(dets))
    return {
        "n_images": n_images,
        "n_classes": n_classes,
        "gts": gts,
        "dets": [dets[i] for i in order],
    }


def coco_eval(case: Case) -> Any:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images = [{"id": i, "width": 1000, "height": 1000} for i in range(case["n_images"])]
    cats = [{"id": k} for k in range(case["n_classes"])]
    anns = []
    for n, (img, c, b, crowd) in enumerate(case["gts"], start=1):
        w, h = b[2] - b[0], b[3] - b[1]
        anns.append(
            {
                "id": n,
                "image_id": img,
                "category_id": c,
                "bbox": [b[0], b[1], w, h],
                "area": w * h,
                "iscrowd": int(crowd),
            }
        )
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "annotations": anns, "categories": cats}
        gt.createIndex()
        res = [
            {
                "image_id": i,
                "category_id": c,
                "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                "score": s,
            }
            for i, c, s, b in case["dets"]
        ]
        if res:
            dt = gt.loadRes(res)
        else:
            dt = COCO()
            dt.dataset = {"images": images, "annotations": [], "categories": cats}
            dt.createIndex()
        ev = COCOeval(gt, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def per_class_reference(ev: Any, n_classes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-class AP (all IoUs), AP50 and AR100 from a finished COCOeval; -1 where undefined."""
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    recall = ev.eval["recall"]  # [T, K, A, M]

    def mean(block: np.ndarray) -> float:
        return float(block[block > -1].mean()) if (block > -1).any() else -1.0

    ap = np.array([mean(precision[:, :, k, 0, 2]) for k in range(n_classes)])
    ap50 = np.array([mean(precision[0, :, k, 0, 2]) for k in range(n_classes)])
    ar = np.array([mean(recall[:, k, 0, 2]) for k in range(n_classes)])
    return ap, ap50, ar


def coco_reference(case: Case) -> tuple[dict[str, float], np.ndarray]:
    ev = coco_eval(case)
    stats = dict(zip(STAT_NAMES, (float(v) for v in ev.stats), strict=True))
    precision = ev.eval["precision"]  # [T, R, K, A, M]
    per_class = np.array(
        [
            float(p[p > -1].mean()) if (p := precision[:, :, k, 0, 2])[p > -1].size else -1.0
            for k in range(case["n_classes"])
        ]
    )
    return stats, per_class


def ours(case: Case) -> tuple[dict[str, float], np.ndarray]:
    gts, dets = case["gts"], case["dets"]
    ious = m.coco_iou_thresholds()
    match = m.match_detections(
        np.array([d[0] for d in dets], dtype=np.int64),
        np.array([d[3] for d in dets], dtype=np.float64).reshape(-1, 4),
        np.array([d[2] for d in dets], dtype=np.float64),
        np.array([d[1] for d in dets], dtype=np.int64),
        np.array([g[0] for g in gts], dtype=np.int64),
        np.array([g[2] for g in gts], dtype=np.float64).reshape(-1, 4),
        np.array([g[1] for g in gts], dtype=np.int64),
        n_images=case["n_images"],
        n_classes=case["n_classes"],
        iou_thresholds=ious,
        iscrowd=np.array([g[3] for g in gts], dtype=bool),
    )
    curves = m.accumulate(match)
    return m.summarize(curves, ious), m.per_class_ap(curves)


def ours_curves(case: Case) -> m.Curves:
    gts, dets = case["gts"], case["dets"]
    match = m.match_detections(
        np.array([d[0] for d in dets], dtype=np.int64),
        np.array([d[3] for d in dets], dtype=np.float64).reshape(-1, 4),
        np.array([d[2] for d in dets], dtype=np.float64),
        np.array([d[1] for d in dets], dtype=np.int64),
        np.array([g[0] for g in gts], dtype=np.int64),
        np.array([g[2] for g in gts], dtype=np.float64).reshape(-1, 4),
        np.array([g[1] for g in gts], dtype=np.int64),
        n_images=case["n_images"],
        n_classes=case["n_classes"],
        iou_thresholds=m.coco_iou_thresholds(),
        iscrowd=np.array([g[3] for g in gts], dtype=bool),
    )
    return m.accumulate(match)


def check(case: Case) -> float:
    """Assert parity on one case and return the largest absolute difference."""
    ref_stats, ref_cls = coco_reference(case)
    our_stats, our_cls = ours(case)
    ref_ap, ref_ap50, ref_ar = per_class_reference(coco_eval(case), case["n_classes"])
    curves = ours_curves(case)
    for ref_arr, our_arr in (
        (ref_ap50, m.per_class_ap(curves, iou=0.5)),
        (ref_ar, m.per_class_recall(curves)),
    ):
        assert np.abs(ref_arr - our_arr).max(initial=0.0) <= TOLERANCE
    worst = 0.0
    for name in STAT_NAMES:
        diff = abs(ref_stats[name] - our_stats[name])
        assert diff <= TOLERANCE, f"{name}: pycocotools {ref_stats[name]} vs {our_stats[name]}"
        worst = max(worst, diff)
    assert np.abs(ref_cls - our_cls).max(initial=0.0) <= TOLERANCE
    return max(worst, float(np.abs(ref_cls - our_cls).max(initial=0.0)))


def test_reference_versions_are_pinned() -> None:
    _require_pinned_versions()


@pytest.mark.parametrize("seed", range(RANDOM_CASES))
def test_randomized_cases_match_pycocotools(seed: int) -> None:
    rng = np.random.default_rng(seed)
    case = make_case(
        rng,
        n_images=int(rng.integers(1, 25)),
        n_classes=int(rng.integers(1, 7)),
        tie_scores=seed % 3 == 0,
        det_rate=float(rng.uniform(0.3, 1.0)),
        extra_dets=int(rng.integers(0, 8)),
    )
    check(case)


def test_no_detections() -> None:
    case = make_case(np.random.default_rng(1), n_images=5, n_classes=3)
    case["dets"] = []
    check(case)


def test_no_ground_truth() -> None:
    case = make_case(np.random.default_rng(2), n_images=5, n_classes=3)
    case["gts"] = []
    check(case)


def test_no_ground_truth_and_no_detections() -> None:
    check({"n_images": 3, "n_classes": 2, "gts": [], "dets": []})


def test_only_crowd_ground_truth() -> None:
    case = make_case(np.random.default_rng(3), n_images=8, n_classes=2, crowd_rate=1.0)
    check(case)


def test_crowd_heavy_and_tied_scores() -> None:
    case = make_case(
        np.random.default_rng(4), n_images=12, n_classes=3, crowd_rate=0.5, tie_scores=True
    )
    check(case)


def test_all_scores_identical() -> None:
    case = make_case(np.random.default_rng(5), n_images=10, n_classes=2)
    case["dets"] = [(i, c, 0.5, b) for i, c, _, b in case["dets"]]
    check(case)


def test_single_class_and_single_image() -> None:
    check(make_case(np.random.default_rng(6), n_images=1, n_classes=1, max_gt=12))


def test_images_without_any_annotation_or_detection() -> None:
    case = make_case(np.random.default_rng(7), n_images=6, n_classes=2)
    case["n_images"] = 20  # 14 empty images; detections on them stay false positives
    case["dets"] += [(15, 0, 0.9, [10.0, 10.0, 60.0, 60.0]), (19, 1, 0.4, [5.0, 5.0, 20.0, 20.0])]
    check(case)


def test_class_without_ground_truth_but_with_detections() -> None:
    case = make_case(np.random.default_rng(8), n_images=6, n_classes=2)
    case["n_classes"] = 4
    case["dets"] += [(0, 3, 0.9, [10.0, 10.0, 60.0, 60.0])]
    check(case)


def test_more_than_one_hundred_detections_in_one_image_and_class() -> None:
    rng = np.random.default_rng(9)
    case = make_case(rng, n_images=2, n_classes=1)
    for _ in range(150):
        x, y = rng.uniform(0, 300, 2)
        w, h = rng.uniform(5, 80, 2)
        case["dets"].append(
            (0, 0, float(rng.random()), [float(x), float(y), float(x + w), float(y + h)])
        )
    check(case)


def test_boundary_areas_and_exact_iou_thresholds() -> None:
    # 32x32 and 96x96 boxes sit on the range boundaries; the shifted box has IoU exactly 0.5
    gts = [(0, 0, [0.0, 0.0, 32.0, 32.0], False), (0, 0, [100.0, 0.0, 196.0, 96.0], False)]
    dets = [
        (0, 0, 0.9, [0.0, 0.0, 32.0, 32.0]),
        (0, 0, 0.8, [100.0, 0.0, 196.0, 48.0]),
    ]
    check({"n_images": 1, "n_classes": 1, "gts": gts, "dets": dets})


def test_recorded_maximum_difference_is_small(capsys: pytest.CaptureFixture[str]) -> None:
    worst = 0.0
    for seed in range(RANDOM_CASES):
        rng = np.random.default_rng(seed)
        case = make_case(
            rng,
            n_images=int(rng.integers(1, 25)),
            n_classes=int(rng.integers(1, 7)),
            tie_scores=seed % 3 == 0,
        )
        worst = max(worst, check(case))
    with capsys.disabled():
        print(f"\nparity: {RANDOM_CASES} randomized cases, max abs difference {worst:.3e}")
    assert worst <= TOLERANCE


@pytest.mark.parametrize("seed", range(40))
def test_engine_headline_metrics_match_pycocotools(seed: int) -> None:
    """The same check through the public tables and ``evaluate`` (names, undefined metrics)."""
    from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable, evaluate

    rng = np.random.default_rng(1000 + seed)
    case = make_case(rng, n_images=int(rng.integers(2, 20)), n_classes=int(rng.integers(1, 5)))
    if not any(not g[3] for g in case["gts"]):
        pytest.skip("no countable ground truth in this draw")
    ref, ref_cls = coco_reference(case)
    ids = [f"{i:06d}" for i in range(case["n_images"])]
    gts = AnnotationTable.from_columns(
        image_id=[ids[g[0]] for g in case["gts"]],
        class_id=[g[1] for g in case["gts"]],
        boxes=np.array([g[2] for g in case["gts"]], dtype=float).reshape(-1, 4),
        iscrowd=[g[3] for g in case["gts"]],
        images=ImageTable.from_columns(
            image_id=ids, width=[1000] * len(ids), height=[1000] * len(ids)
        ),
    )
    preds = PredictionTable.from_columns(
        image_id=[ids[d[0]] for d in case["dets"]],
        class_id=[d[1] for d in case["dets"]],
        score=[d[2] for d in case["dets"]],
        boxes=np.array([d[3] for d in case["dets"]], dtype=float).reshape(-1, 4),
    )
    names = [f"c{k}" for k in range(case["n_classes"])]
    result = evaluate(preds, gts, task="detection", class_names=names)
    expected = {
        "mAP50_95": "AP", "mAP50": "AP50", "mAP75": "AP75", "mAP_small": "APs",
        "mAP_medium": "APm", "mAP_large": "APl", "AR1": "AR1", "AR10": "AR10",
        "AR100": "AR100", "AR_small": "ARs", "AR_medium": "ARm", "AR_large": "ARl",
    }  # fmt: skip
    for ours_name, coco_name in expected.items():
        if ref[coco_name] == -1:
            assert ours_name not in result.metrics
        else:
            assert abs(result.metrics[ours_name] - ref[coco_name]) <= TOLERANCE
    for k, name in enumerate(names):
        row = result.per_class[name]
        if ref_cls[k] == -1:
            assert "AP50_95" not in row
        else:
            assert abs(row["AP50_95"] - ref_cls[k]) <= TOLERANCE


# -- cross-image tie order follows the numeric image id (pycocotools sorts ids as numbers) -------


def engine_metrics(case: Case, ids: list[str]) -> dict[str, float]:
    from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable, evaluate

    gts = AnnotationTable.from_columns(
        image_id=[ids[g[0]] for g in case["gts"]],
        class_id=[g[1] for g in case["gts"]],
        boxes=np.array([g[2] for g in case["gts"]], dtype=float).reshape(-1, 4),
        iscrowd=[g[3] for g in case["gts"]],
        images=ImageTable.from_columns(image_id=ids),
    )
    preds = PredictionTable.from_columns(
        image_id=[ids[d[0]] for d in case["dets"]],
        class_id=[d[1] for d in case["dets"]],
        score=[d[2] for d in case["dets"]],
        boxes=np.array([d[3] for d in case["dets"]], dtype=float).reshape(-1, 4),
    )
    names = [f"c{k}" for k in range(case["n_classes"])]
    return evaluate(preds, gts, task="detection", class_names=names, n_resamples=0).metrics


def test_tied_scores_across_images_are_ranked_by_numeric_id_not_string_order() -> None:
    box = [10.0, 10.0, 50.0, 50.0]
    case = {
        "n_images": 11,
        "n_classes": 1,
        "gts": [(9, 0, box, False)],
        "dets": [(10, 0, 0.9, box), (9, 0, 0.9, box)],  # a false positive on image 10, a hit on 9
    }
    ref, _ = coco_reference(case)
    assert ref["AP"] == pytest.approx(1.0, abs=1e-9)
    ours_metrics = engine_metrics(case, [str(i) for i in range(11)])
    assert ours_metrics["mAP50_95"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("seed", range(40))
def test_unpadded_integer_ids_with_ties_match_pycocotools(seed: int) -> None:
    rng = np.random.default_rng(5000 + seed)
    n_images = int(rng.choice([12, 25, 101, 130]))
    case = make_case(rng, n_images=n_images, n_classes=int(rng.integers(1, 4)), max_gt=3,
                     tie_scores=True, extra_dets=3)  # fmt: skip
    case["dets"] = [(i, c, round(s, 1) if seed % 2 else 0.5, b) for i, c, s, b in case["dets"]]
    if not any(not g[3] for g in case["gts"]):
        pytest.skip("no countable ground truth")
    ref, _ = coco_reference(case)
    got = engine_metrics(case, [str(i) for i in range(n_images)])
    for ours_name, coco_name in (("mAP50_95", "AP"), ("mAP50", "AP50"), ("AR100", "AR100")):
        assert abs(got[ours_name] - ref[coco_name]) <= TOLERANCE


def test_non_integer_ids_keep_string_order_and_say_so() -> None:
    from cvpilot.eval.validate import canonical_order

    assert canonical_order(np.array(["10", "9", "100"])).tolist() == [1, 0, 2]  # numeric
    assert canonical_order(np.array(["b", "a", "10"])).tolist() == [2, 1, 0]  # string order
    assert canonical_order(np.array(["01", "1"])).tolist() == [0, 1]  # not canonical: strings
    assert canonical_order(np.array(["-1", "5"])).tolist() == [0, 1]


# -- K3: harder corpora ------------------------------------------------------------------------


def test_identical_and_duplicated_ground_truth_boxes_follow_the_last_of_equal_ious_rule() -> None:
    box = [10.0, 10.0, 60.0, 60.0]
    gts = [(0, 0, box, False)] * 3 + [(0, 0, [10.0, 10.0, 60.0, 59.0], False), (1, 0, box, False)]
    dets = [(0, 0, 0.9 - 0.1 * i, box) for i in range(5)] + [(1, 0, 0.3, box)]
    check({"n_images": 2, "n_classes": 1, "gts": gts, "dets": dets})


@pytest.mark.parametrize("budget", [None, 64, 1])
def test_large_groups_across_the_multi_chunk_path(budget: int | None, monkeypatch) -> None:  # noqa: ANN001
    if budget is not None:
        monkeypatch.setattr(m, "MATCH_BUDGET", budget)
    rng = np.random.default_rng(77)
    case = make_case(rng, n_images=3, n_classes=2, max_gt=30, extra_dets=0)
    case["gts"] += [(0, 0, [float(v) for v in (x, y, x + 40, y + 40)], False)
                    for x, y in rng.uniform(0, 300, (40, 2))]  # fmt: skip
    for _ in range(160):  # more than 100 detections in one image and class
        x, y = rng.uniform(0, 300, 2)
        case["dets"].append(
            (0, 0, float(rng.random()), [float(x), float(y), float(x + 40), float(y + 40)])
        )
    check(case)


def test_ious_exactly_equal_to_each_threshold() -> None:
    # unit-height strips: IoU of [0, 0, a, 1] against [0, 0, b, 1] is b / a exactly
    gts, dets = [], []
    ratios = [(5, 10), (11, 20), (6, 10), (13, 20), (7, 10), (3, 4)]
    ratios += [(4, 5), (17, 20), (9, 10), (19, 20), (3, 5), (7, 8)]
    for k, (num, den) in enumerate(ratios):
        gts.append((k, 0, [0.0, 0.0, float(den), 1.0], False))
        dets.append((k, 0, 0.9 - 0.01 * k, [0.0, 0.0, float(num), 1.0]))
    check({"n_images": 12, "n_classes": 1, "gts": gts, "dets": dets})


@pytest.mark.parametrize("seed", range(40))
def test_up_to_twenty_classes(seed: int) -> None:
    rng = np.random.default_rng(9000 + seed)
    check(make_case(rng, n_images=int(rng.integers(5, 40)), n_classes=int(rng.integers(8, 21)),
                    tie_scores=seed % 4 == 0))  # fmt: skip


@pytest.mark.parametrize("seed", range(25))
def test_weighted_resampling_equals_a_literally_duplicated_dataset(seed: int) -> None:
    """A bootstrap resample is evaluated once with weights and once as real copies of the
    images by pycocotools; continuous scores keep tie order out of the comparison."""
    from cvpilot.eval import bootstrap as bs

    rng = np.random.default_rng(12000 + seed)
    n_images = int(rng.integers(4, 16))
    case = make_case(rng, n_images=n_images, n_classes=int(rng.integers(1, 5)), crowd_rate=0.05)
    if not any(not g[3] for g in case["gts"]):
        pytest.skip("no countable ground truth")
    counts = np.bincount(rng.integers(0, n_images, n_images), minlength=n_images)
    copies: list[int] = [i for i in range(n_images) for _ in range(int(counts[i]))]
    dup = {
        "n_images": len(copies),
        "n_classes": case["n_classes"],
        "gts": [(new, g[1], g[2], g[3]) for new, old in enumerate(copies)
                for g in case["gts"] if g[0] == old],
        "dets": [(new, d[1], d[2], d[3]) for new, old in enumerate(copies)
                 for d in case["dets"] if d[0] == old],
    }  # fmt: skip
    if not any(not g[3] for g in dup["gts"]):
        pytest.skip("the resample has no countable ground truth")
    ref, _ = coco_reference(dup)
    gts, dets = case["gts"], case["dets"]
    match = m.match_detections(
        np.array([d[0] for d in dets], dtype=np.int64),
        np.array([d[3] for d in dets], dtype=np.float64).reshape(-1, 4),
        np.array([d[2] for d in dets], dtype=np.float64),
        np.array([d[1] for d in dets], dtype=np.int64),
        np.array([g[0] for g in gts], dtype=np.int64),
        np.array([g[2] for g in gts], dtype=np.float64).reshape(-1, 4),
        np.array([g[1] for g in gts], dtype=np.int64),
        n_images=n_images, n_classes=case["n_classes"],
        iou_thresholds=m.coco_iou_thresholds(), iscrowd=np.array([g[3] for g in gts], dtype=bool),
    )  # fmt: skip
    ap, ap50, ar = bs._Resampler(match).run(counts.astype(float))
    assert abs(ap - ref["AP"]) <= 1e-9 and abs(ap50 - ref["AP50"]) <= 1e-9
    assert abs(ar - ref["AR100"]) <= 1e-9
