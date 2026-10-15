"""Typed evaluation inputs: predictions, ground-truth annotations and the image list.

Tables are plain NumPy columns. ``from_columns`` accepts lists, arrays and Arrow arrays;
``from_table`` accepts a mapping of columns or an Arrow table (duck-typed, ``pyarrow`` is not
imported here). Construction only fixes dtypes and shapes; the semantic checks (finite scores,
box bounds, id and class ranges, matching image sets, bounded counts) run in
``cvpilot.eval.validate`` so a table can be built cheaply and rejected with a precise message.

Image ids are strings (integer ids are converted). Class ids are integers indexing
``class_names``. Boxes are ``x1, y1, x2, y2`` in pixels.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from cvpilot.core.errors import EvalInputError
from cvpilot.core.metrics import F64, I64, BoolArray

StrArray = npt.NDArray[np.str_]

BOX_COLUMNS = ("x1", "y1", "x2", "y2")

MAX_IMAGES = 1_000_000
"""Most images in one evaluation set."""
MAX_PREDICTIONS = 5_000_000
"""Most prediction rows in one evaluation (a few hundred MB of working memory at the limit)."""
MAX_ANNOTATIONS = 5_000_000
"""Most ground-truth rows in one evaluation."""
MAX_ANNOTATIONS_PER_IMAGE = 100_000
"""Most ground-truth boxes on one image; bounds the padded matching block."""
MAX_ID_LENGTH = 256
"""Longest image id in characters."""
MAX_COORDINATE = 1e7
"""Largest absolute box coordinate accepted (guards the area arithmetic against overflow)."""
MAX_IMAGE_SIDE = 1_000_000
"""Largest image width or height in pixels."""
BOUNDS_TOLERANCE = 1e-3
"""Pixels a ground-truth box may exceed the image by (rounding in annotation tools)."""
MAX_STRING_COLUMN_BYTES = 512 * 1024 * 1024
"""Most memory a string column may take as a fixed-width array (rows x longest id x 4 bytes)."""


def _fail(message: str) -> EvalInputError:
    return EvalInputError(message)


def _to_array(values: object, what: str) -> np.ndarray[Any, Any]:
    to_numpy = getattr(values, "to_numpy", None)
    try:
        if callable(to_numpy):
            try:
                return np.asarray(to_numpy(zero_copy_only=False))
            except TypeError:
                return np.asarray(to_numpy())
        return np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise _fail(f"column {what!r} cannot be read as an array: {exc}") from None


_BAD_TEXT = re.compile(r"[\x00-\x1f\x7f]")


def check_text(value: str, what: str) -> str:
    """Reject control characters (NUL included) and lone surrogates; never a raw codec error."""
    if _BAD_TEXT.search(value):
        raise _fail(f"{what} contains control characters")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _fail(f"{what} contains invalid Unicode (a lone surrogate)") from None
    return value


def as_ids(values: object, what: str) -> StrArray:
    """String array of image ids; integers are converted, anything else is rejected."""
    arr = _to_array(values, what)
    if arr.ndim != 1:
        raise _fail(f"column {what!r} must be one-dimensional")
    if arr.size == 0:
        return np.zeros(0, dtype="<U1")
    if arr.dtype.kind in "iu":
        return arr.astype(str)
    if arr.dtype.kind == "O":
        if not all(type(v) is str for v in arr):
            raise _fail(f"column {what!r} must hold strings or integers")
        longest = max(map(len, arr), default=0)
        check_string_footprint(arr.shape[0], longest, what)
        for value in set(arr.tolist()):  # checked on the Python strings, before astype(str)
            check_text(value, what)
        return arr.astype(str)
    if arr.dtype.kind == "U":
        check_string_footprint(arr.shape[0], arr.dtype.itemsize // 4, what)
        return arr
    raise _fail(f"column {what!r} must hold strings or integers")


def check_string_footprint(rows: int, longest: int, what: str) -> None:
    """Reject a string column whose fixed-width array would exceed ``MAX_STRING_COLUMN_BYTES``."""
    if longest > MAX_ID_LENGTH:
        raise _fail(f"column {what!r} has a value longer than {MAX_ID_LENGTH} characters")
    if rows * max(longest, 1) * 4 > MAX_STRING_COLUMN_BYTES:
        raise _fail(f"column {what!r} is too large: {rows} rows of up to {longest} characters")


def as_ints(values: object, what: str) -> I64:
    arr = _to_array(values, what)
    if arr.ndim != 1:
        raise _fail(f"column {what!r} must be one-dimensional")
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)
    if arr.dtype.kind == "u" and arr.size and int(arr.max()) > np.iinfo(np.int64).max:
        raise _fail(f"column {what!r} has values beyond the int64 range")
    if arr.dtype.kind not in "iu":
        raise _fail(f"column {what!r} must hold integers, got dtype {arr.dtype}")
    return arr.astype(np.int64)


def as_floats(values: object, what: str, *, width: int | None = None) -> F64:
    arr = _to_array(values, what)
    if arr.dtype.kind not in "iuf":
        raise _fail(f"column {what!r} must hold numbers, got dtype {arr.dtype}")
    out = arr.astype(np.float64)
    expected_ndim = 1 if width is None else 2
    if out.ndim != expected_ndim or (width is not None and out.shape[1] != width):
        raise _fail(f"column {what!r} has the wrong shape {out.shape}")
    return out


def as_bools(values: object, what: str) -> BoolArray:
    arr = _to_array(values, what)
    if arr.size == 0 and arr.ndim == 1:
        return np.zeros(0, dtype=bool)
    if arr.ndim != 1 or arr.dtype.kind not in "biu":
        raise _fail(f"column {what!r} must hold booleans or 0/1 integers")
    if arr.dtype.kind != "b" and arr.size and not np.isin(arr, (0, 1)).all():
        raise _fail(f"column {what!r} must hold booleans or 0/1 integers")
    return arr.astype(bool)


def column(source: object, name: str) -> object:
    """One column of a mapping or an Arrow-like table."""
    if isinstance(source, Mapping):
        if name not in source:
            raise _fail(f"missing column {name!r}")
        return source[name]
    names = getattr(source, "column_names", None)
    getter = getattr(source, "column", None)
    if names is None or getter is None:
        raise _fail("expected a mapping of columns or an Arrow table")
    if name not in names:
        raise _fail(f"missing column {name!r}")
    return getter(name)


def has_column(source: object, name: str) -> bool:
    if isinstance(source, Mapping):
        return name in source
    return name in (getattr(source, "column_names", None) or ())


def _boxes(source: object | None, n: int, what: str) -> F64:
    if source is None:
        return np.zeros((n, 4), dtype=np.float64)
    return as_floats(source, what, width=4)


def _box_columns(source: object) -> F64:
    return np.column_stack([as_floats(column(source, c), c) for c in BOX_COLUMNS])


def _check_lengths(n: int, **arrays: np.ndarray[Any, Any]) -> None:
    for name, arr in arrays.items():
        if arr.shape[0] != n:
            raise _fail(f"column {name!r} has {arr.shape[0]} rows, expected {n}")


@dataclass(frozen=True)
class ImageTable:
    """The images of an evaluation set: ids, optional pixel sizes and per-image metadata.

    ``meta`` columns (strings) feed slices, for example ``{"camera": ["cam01", ...]}``.
    """

    image_id: StrArray
    width: I64 | None = None
    height: I64 | None = None
    meta: Mapping[str, StrArray] = field(default_factory=dict)

    @classmethod
    def from_columns(
        cls,
        *,
        image_id: object,
        width: object | None = None,
        height: object | None = None,
        meta: Mapping[str, object] | None = None,
    ) -> ImageTable:
        ids = as_ids(image_id, "image_id")
        n = ids.shape[0]
        w = None if width is None else as_ints(width, "width")
        h = None if height is None else as_ints(height, "height")
        if (w is None) != (h is None):
            raise _fail("width and height must be given together")
        columns = {k: as_ids(v, f"meta.{k}") for k, v in (meta or {}).items()}
        _check_lengths(
            n, **{k: v for k, v in {"width": w, "height": h}.items() if v is not None}, **columns
        )
        return cls(ids, w, h, columns)


@dataclass(frozen=True)
class PredictionTable:
    """Model outputs: one row per detection (or per image for classification)."""

    image_id: StrArray
    class_id: I64
    score: F64
    boxes: F64  # (N, 4); zeros for classification

    @classmethod
    def from_columns(
        cls, *, image_id: object, class_id: object, score: object, boxes: object | None = None
    ) -> PredictionTable:
        ids = as_ids(image_id, "image_id")
        n = ids.shape[0]
        table = cls(
            ids, as_ints(class_id, "class_id"), as_floats(score, "score"), _boxes(boxes, n, "boxes")
        )
        _check_lengths(n, class_id=table.class_id, score=table.score, boxes=table.boxes)
        return table

    @classmethod
    def from_table(cls, source: object, *, task: str = "detection") -> PredictionTable:
        """Columns ``image_id, cls, score`` and, for detection, ``x1, y1, x2, y2``."""
        boxes = _box_columns(source) if task == "detection" else None
        return cls.from_columns(
            image_id=column(source, "image_id"),
            class_id=column(source, "cls"),
            score=column(source, "score"),
            boxes=boxes,
        )

    def __len__(self) -> int:
        return int(self.image_id.shape[0])


@dataclass(frozen=True)
class AnnotationTable:
    """Ground truth: one row per box (or per image label for classification).

    ``images`` lists every image of the set, including those without annotations; when it is
    ``None`` the set is the images that have at least one row.
    """

    image_id: StrArray
    class_id: I64
    boxes: F64
    iscrowd: BoolArray
    images: ImageTable | None = None

    @classmethod
    def from_columns(
        cls,
        *,
        image_id: object,
        class_id: object,
        boxes: object | None = None,
        iscrowd: object | None = None,
        images: ImageTable | None = None,
    ) -> AnnotationTable:
        ids = as_ids(image_id, "image_id")
        n = ids.shape[0]
        crowd = np.zeros(n, dtype=bool) if iscrowd is None else as_bools(iscrowd, "iscrowd")
        table = cls(ids, as_ints(class_id, "class_id"), _boxes(boxes, n, "boxes"), crowd, images)
        _check_lengths(n, class_id=table.class_id, boxes=table.boxes, iscrowd=table.iscrowd)
        return table

    @classmethod
    def from_table(
        cls, source: object, *, images: ImageTable | None = None, task: str = "detection"
    ) -> AnnotationTable:
        """Columns ``image_id, cls``, for detection ``x1..y2`` and optionally ``iscrowd``."""
        boxes = _box_columns(source) if task == "detection" else None
        crowd = column(source, "iscrowd") if has_column(source, "iscrowd") else None
        return cls.from_columns(
            image_id=column(source, "image_id"),
            class_id=column(source, "cls"),
            boxes=boxes,
            iscrowd=crowd,
            images=images,
        )

    def __len__(self) -> int:
        return int(self.image_id.shape[0])


def check_class_names(class_names: Sequence[str]) -> list[str]:
    """Validated copy of the class list: 1 to ``MAX_CLASSES`` unique, printable, short names."""
    from cvpilot.core.schema import MAX_CLASSES, MAX_NAME_LENGTH

    names = list(class_names)
    if not 1 <= len(names) <= MAX_CLASSES:
        raise _fail(f"class_names needs between 1 and {MAX_CLASSES} entries, got {len(names)}")
    for name in names:
        if not isinstance(name, str) or not name or len(name) > MAX_NAME_LENGTH:
            raise _fail(
                f"class names must be non-empty strings of at most {MAX_NAME_LENGTH} characters"
            )
        check_text(name, "class name")
    if len(set(names)) != len(names):
        raise _fail("class_names must be unique")
    return names
