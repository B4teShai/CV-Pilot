"""Semantic validation of evaluation inputs; everything the metric code assumes is checked here.

Checks, in the order they run: row counts, image ids (non-empty, short, no control characters,
unique in the image list), every annotation and prediction image is part of the evaluation set,
class ids inside ``[0, C)``, finite scores, finite ordered boxes, ground-truth boxes inside the
image (when sizes are known), bounded boxes per image, and one label per image for
classification. Each failure raises :class:`EvalInputError` naming the column and the first
offending value (truncated).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from cvpilot.core.errors import EvalInputError
from cvpilot.core.metrics import F64, I64, BoolArray, max_int
from cvpilot.core.schema import Task
from cvpilot.eval import tables as t


@dataclass(frozen=True)
class GroundTruth:
    """Validated ground truth indexed by position in ``image_ids`` (sorted, unique)."""

    image_ids: t.StrArray  # ranked: numeric for canonical integer ids, else string order
    width: I64  # 0 when unknown
    height: I64
    meta: dict[str, t.StrArray]  # in image order
    ann_image: I64
    ann_cls: I64
    ann_boxes: F64
    ann_crowd: BoolArray
    sizes_known: bool = field(default=False)

    @property
    def n_images(self) -> int:
        return int(self.image_ids.shape[0])


@dataclass(frozen=True)
class Predictions:
    """Validated predictions with image indices into ``GroundTruth.image_ids``."""

    image: I64
    cls: I64
    score: F64
    boxes: F64


def _fail(message: str) -> EvalInputError:
    return EvalInputError(message)


def _show(value: object) -> str:
    """A reference to a value for error messages that does not echo its text (ids can be
    file names or personal data): its length and a short digest."""
    text = str(value)
    digest = hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:8]
    return f"<{len(text)} characters, sha256:{digest}>"


def _check_ids(ids: t.StrArray, what: str) -> None:
    if ids.size == 0:
        return
    lengths = np.char.str_len(ids)
    if int(lengths.min()) < 1:
        raise _fail(f"{what} contains an empty image id")
    if int(lengths.max()) > t.MAX_ID_LENGTH:
        raise _fail(f"{what} contains an image id longer than {t.MAX_ID_LENGTH} characters")
    for value in ids.tolist():
        t.check_text(value, f"{what} image id")


def canonical_order(ids: t.StrArray) -> I64:
    """Permutation that ranks unique image ids: numeric when every id is a canonical
    non-negative decimal integer (``0`` or no leading zero, at most 18 digits), else string
    order. The rank is the image index, and ties between equal scores of different images
    are broken by it, which is what pycocotools does with integer ids. Ids that are not
    canonical integers follow string order (``"10"`` before ``"9"``); parity with COCO tools
    then needs ids that sort the same way, for example zero-padded."""
    if ids.size and bool(np.char.isdigit(ids).all()):
        lengths = np.char.str_len(ids)
        leading_zero = (np.char.startswith(ids, "0")) & (lengths > 1)
        if int(np.max(lengths)) <= 18 and not bool(leading_zero.any()):
            return np.argsort(ids.astype(np.int64), kind="stable").astype(np.int64)
    return np.argsort(ids, kind="stable").astype(np.int64)


def _locate(ranked_ids: t.StrArray, ids: t.StrArray, what: str) -> I64:
    """Rank (image index) of each of ``ids`` in ``ranked_ids``; unknown ids are an error."""
    by_string = np.argsort(ranked_ids, kind="stable")
    sorted_ids = ranked_ids[by_string]
    if ids.size == 0:
        return np.zeros(0, dtype=np.int64)
    if sorted_ids.size == 0:
        raise _fail(
            f"{what} refers to image id {_show(str(ids[0]))} but the evaluation set is empty"
        )
    pos = np.searchsorted(sorted_ids, ids)
    known = (pos < sorted_ids.shape[0]) & (
        sorted_ids[np.minimum(pos, sorted_ids.shape[0] - 1)] == ids
    )
    if not bool(np.all(known)):
        first = ids[int(np.flatnonzero(~known)[0])]
        raise _fail(
            f"{what} refers to image id {_show(str(first))} that is not in the evaluation set "
            f"({int((~known).sum())} rows affected)"
        )
    ranks: I64 = by_string[np.minimum(pos, sorted_ids.shape[0] - 1)].astype(np.int64)
    return ranks


def _check_classes(cls: I64, n_classes: int, what: str) -> None:
    bad = (cls < 0) | (cls >= n_classes)
    if bool(bad.any()):
        value = int(cls[int(np.flatnonzero(bad)[0])])
        raise _fail(f"{what} has class id {value} outside [0, {n_classes})")


def _check_boxes(boxes: F64, what: str, *, low: float, high: float) -> None:
    if not bool(np.isfinite(boxes).all()):
        raise _fail(f"{what} has non-finite box coordinates")
    if bool((boxes < low).any()) or bool((boxes > high).any()):
        raise _fail(f"{what} has box coordinates outside [{low:g}, {high:g}]")
    if bool((boxes[:, 2] < boxes[:, 0]).any()) or bool((boxes[:, 3] < boxes[:, 1]).any()):
        raise _fail(f"{what} has boxes with x2 < x1 or y2 < y1")


def prepare_ground_truth(gts: t.AnnotationTable, n_classes: int, task: Task) -> GroundTruth:
    """Validate ground truth and index its images (sorted by id)."""
    if len(gts) > t.MAX_ANNOTATIONS:
        raise _fail(f"ground truth has {len(gts)} rows; the limit is {t.MAX_ANNOTATIONS}")
    images = gts.images
    meta: dict[str, t.StrArray] = {}
    width: I64
    height: I64
    if images is None:
        unique = np.unique(gts.image_id)
        ids = unique[canonical_order(unique)]
        width = np.zeros(ids.shape[0], dtype=np.int64)
        height = width.copy()
        _check_ids(ids, "ground truth")
    else:
        if images.image_id.shape[0] > t.MAX_IMAGES:
            raise _fail(
                f"image list has {images.image_id.shape[0]} images; the limit is {t.MAX_IMAGES}"
            )
        _check_ids(images.image_id, "image list")
        ids, counts = np.unique(images.image_id, return_counts=True)
        if bool((counts > 1).any()):
            dup = ids[int(np.flatnonzero(counts > 1)[0])]
            raise _fail(f"image list repeats image id {_show(str(dup))}")
        order = canonical_order(images.image_id)
        ids = images.image_id[order]
        if images.width is None or images.height is None:
            width = np.zeros(ids.shape[0], dtype=np.int64)
            height = width.copy()
        else:
            width, height = images.width[order], images.height[order]
            if bool(((width < 1) | (width > t.MAX_IMAGE_SIDE)).any()) or bool(
                ((height < 1) | (height > t.MAX_IMAGE_SIDE)).any()
            ):
                raise _fail(f"image width and height must be in [1, {t.MAX_IMAGE_SIDE}]")
        meta = {k: v[order] for k, v in images.meta.items()}
    if ids.shape[0] > t.MAX_IMAGES:
        raise _fail(f"ground truth has {ids.shape[0]} images; the limit is {t.MAX_IMAGES}")
    ann_image = _locate(ids, gts.image_id, "ground truth")
    _check_classes(gts.class_id, n_classes, "ground truth")
    counts = np.bincount(ann_image, minlength=ids.shape[0])
    if counts.size and max_int(counts) > t.MAX_ANNOTATIONS_PER_IMAGE:
        raise _fail(f"an image has more than {t.MAX_ANNOTATIONS_PER_IMAGE} ground-truth boxes")
    sizes_known = bool(width.size and (width > 0).all())
    if task == "classification":
        if bool((counts != 1).any()):
            raise _fail("classification ground truth needs exactly one label per image")
        boxes: F64 = np.zeros((len(gts), 4))
        crowd: BoolArray = np.zeros(len(gts), dtype=bool)
    else:
        boxes, crowd = gts.boxes, gts.iscrowd
        _check_boxes(boxes, "ground truth", low=-t.BOUNDS_TOLERANCE, high=t.MAX_COORDINATE)
        if sizes_known:
            w, h = width[ann_image], height[ann_image]
            if bool((boxes[:, 2] > w + t.BOUNDS_TOLERANCE).any()) or bool(
                (boxes[:, 3] > h + t.BOUNDS_TOLERANCE).any()
            ):
                raise _fail("ground truth has boxes outside the image")
    return GroundTruth(ids, width, height, meta, ann_image, gts.class_id, boxes, crowd, sizes_known)


def prepare_predictions(
    preds: t.PredictionTable, gt: GroundTruth, n_classes: int, task: Task
) -> Predictions:
    """Validate predictions against the evaluation set of ``gt``."""
    if len(preds) > t.MAX_PREDICTIONS:
        raise _fail(f"predictions have {len(preds)} rows; the limit is {t.MAX_PREDICTIONS}")
    _check_ids(np.unique(preds.image_id), "predictions")
    image = _locate(gt.image_ids, preds.image_id, "predictions")
    _check_classes(preds.class_id, n_classes, "predictions")
    if not bool(np.isfinite(preds.score).all()):
        raise _fail("predictions have non-finite scores")
    if task == "classification":
        if image.size and max_int(np.bincount(image)) > 1:
            raise _fail("classification needs at most one prediction per image (top-1)")
        boxes = preds.boxes
    else:
        boxes = preds.boxes
        _check_boxes(boxes, "predictions", low=-t.MAX_COORDINATE, high=t.MAX_COORDINATE)
    return Predictions(image, preds.class_id, preds.score, boxes)
