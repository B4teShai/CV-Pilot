"""Slices: metrics per value of an image metadata column (for example ``camera=cam03``).

A slice spec maps a slice key to a metadata column of the image list. Every distinct value of
the column is one slice holding the images with that value. Cardinality is bounded before any
metric is computed, because a column like an image path would otherwise produce one slice per
image: at most ``MAX_SLICE_KEYS`` keys with at most ``MAX_SLICE_VALUES`` values each (the
product also fits ``cvpilot.core.schema.MAX_SLICE_ENTRIES``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from cvpilot.core.errors import EvalInputError, SliceCardinalityError
from cvpilot.core.metrics import BoolArray
from cvpilot.core.schema import MAX_NAME_LENGTH, MAX_SLICE_ENTRIES
from cvpilot.eval import tables as t
from cvpilot.eval.validate import GroundTruth

MAX_SLICE_KEYS = 16
"""Most slice keys (metadata columns) one evaluation may slice on."""
MAX_SLICE_VALUES = 256
"""Most distinct values per slice key; more raises :class:`SliceCardinalityError`."""

_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_PREFIX = "meta."


MAX_SLICE_WORK = 300_000_000
"""Most ``slices x (predictions + ground-truth boxes)`` an evaluation may process; every slice
is one pass over the matched detections, so this bounds the time spent on slices."""


def _printable(value: str) -> bool:
    try:
        t.check_text(value, "slice value")
    except EvalInputError:
        return False
    return True


@dataclass(frozen=True)
class Slice:
    """One ``key=value`` slice: a label and a view on the shared per-image code array."""

    label: str
    codes: np.ndarray[Any, Any]  # shared by all slices of a key: value index per image
    index: int

    def mask(self) -> BoolArray:
        """Boolean mask of the images in this slice, built when asked for."""
        result: BoolArray = self.codes == self.index
        return result


def plan_slices(
    spec: Mapping[str, str] | None, gt: GroundTruth, work_per_slice: int = 0
) -> list[Slice]:
    """Validate ``spec`` against the image metadata and return the slices, sorted by label.

    Raises:
        EvalInputError: a bad key, an unknown column, an invalid value or too many keys.
        SliceCardinalityError: a column has more than ``MAX_SLICE_VALUES`` distinct values.
    """
    if not spec:
        return []
    if len(spec) > MAX_SLICE_KEYS:
        raise EvalInputError(f"at most {MAX_SLICE_KEYS} slice keys are allowed, got {len(spec)}")
    planned: list[Slice] = []
    for key, column in spec.items():
        if not isinstance(key, str) or _KEY.fullmatch(key) is None:
            raise EvalInputError(f"slice key {key[:40]!r} must match [A-Za-z][A-Za-z0-9_]{{0,63}}")
        name = column[len(_PREFIX) :] if column.startswith(_PREFIX) else column
        values = gt.meta.get(name)
        if values is None:
            known = ", ".join(sorted(gt.meta)) or "none"
            raise EvalInputError(
                f"slice {key!r}: the image list has no metadata column {name[:40]!r} (has: {known})"
            )
        uniques, inverse = np.unique(values, return_inverse=True)
        if uniques.shape[0] > MAX_SLICE_VALUES:
            raise SliceCardinalityError(
                f"slice key {key!r} has {uniques.shape[0]} distinct values; "
                f"the limit is {MAX_SLICE_VALUES}"
            )
        for value in uniques.tolist():
            if not value or len(value) > MAX_NAME_LENGTH or not _printable(value):
                raise EvalInputError(
                    f"slice value {value[:40]!r} of {key!r} must be 1 to {MAX_NAME_LENGTH} "
                    "printable characters"
                )
        codes = inverse.reshape(-1)
        planned.extend(
            Slice(f"{key}={value}", codes, j) for j, value in enumerate(uniques.tolist())
        )
    if len(planned) > MAX_SLICE_ENTRIES:
        raise SliceCardinalityError(
            f"{len(planned)} slices exceed the limit of {MAX_SLICE_ENTRIES}"
        )
    if len(planned) * work_per_slice > MAX_SLICE_WORK:
        raise EvalInputError(
            f"{len(planned)} slices over {work_per_slice} rows exceed the slice work budget "
            f"of {MAX_SLICE_WORK}; slice on fewer values"
        )
    return sorted(planned, key=lambda s: s.label)
