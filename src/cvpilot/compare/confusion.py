"""Confusion matrices of two snapshots and their difference, background class included.

The matrices live in content-addressed Parquet artifacts (sparse ``true_class, pred_class,
count`` rows). The loader reads them through ``open_artifact`` (SHA-256 verified before use),
bounds size, rows and values, and builds the dense ``(C+1) x (C+1)`` matrix. Index ``C`` is the
background: ``[true, C]`` is a missed ground-truth box, ``[C, pred]`` a spurious detection.
"""

from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cvpilot.compare.engine import ConfusionCell, ConfusionDelta
from cvpilot.core.errors import EvalInputError, StoreError
from cvpilot.core.schema import RunStore
from cvpilot.registry import tables

MAX_CONFUSION_BYTES = 64 * 1024 * 1024
"""Largest confusion artifact read; a full 2 049 x 2 049 matrix is far smaller than this."""
MAX_CELL_COUNT = 2**40
"""Largest count accepted in one cell; keeps sums far from int64 overflow."""
MAX_REPORTED_CELLS = 50
"""Changed cells kept in a ``ConfusionDelta``, largest absolute change first."""
BACKGROUND = "background"

Matrix = np.ndarray[Any, Any]


def load_confusion(store: RunStore, sha256: str, n_classes: int) -> Matrix:
    """The verified confusion matrix stored under ``sha256`` as a dense int64 array.

    Raises:
        SnapshotIntegrityError: the stored bytes do not hash to the address.
        StoreError: the artifact is missing or larger than ``MAX_CONFUSION_BYTES``.
        EvalInputError: malformed Parquet, too many rows, indices outside ``[0, C]`` or counts
            outside ``[0, MAX_CELL_COUNT]``.
    """
    with store.open_artifact(sha256) as handle:
        data = handle.read(MAX_CONFUSION_BYTES + 1)
    if len(data) > MAX_CONFUSION_BYTES:
        raise StoreError(f"confusion artifact {sha256[:12]} is too large")
    size = n_classes + 1
    with tempfile.TemporaryDirectory(prefix="cvpilot-cmp-") as scratch:
        path = Path(scratch) / "confusion.parquet"
        path.write_bytes(data)
        columns = tables.get("parquet")().read_columns(
            path,
            required=["true_class", "pred_class", "count"],
            max_rows=size * size,
            max_file_bytes=MAX_CONFUSION_BYTES,
        )
    rows, cols, counts = columns["true_class"], columns["pred_class"], columns["count"]
    for name, values in (("true_class", rows), ("pred_class", cols), ("count", counts)):
        if values.dtype.kind not in "iu":
            raise EvalInputError(f"confusion column {name!r} must hold integers")
    if rows.size and (
        int(rows.min()) < 0
        or int(cols.min()) < 0
        or int(rows.max()) >= size
        or int(cols.max()) >= size
    ):
        raise EvalInputError("confusion cell index outside the class range")
    if counts.size and (int(counts.min()) < 0 or int(counts.max()) > MAX_CELL_COUNT):
        raise EvalInputError("confusion cell count outside the supported range")
    dense = np.zeros((size, size), dtype=np.int64)
    np.add.at(dense, (rows.astype(np.int64), cols.astype(np.int64)), counts.astype(np.int64))
    return dense


def confusion_delta(
    baseline: Matrix,
    candidate: Matrix,
    class_names: Sequence[str],
    *,
    max_cells: int = MAX_REPORTED_CELLS,
) -> ConfusionDelta:
    """Difference ``candidate - baseline`` of two ``(C+1) x (C+1)`` matrices.

    Raises:
        EvalInputError: a matrix does not have the shape implied by ``class_names``.
    """
    size = len(class_names) + 1
    if baseline.shape != (size, size) or candidate.shape != (size, size):
        raise EvalInputError("confusion matrices do not match the class list")
    delta = candidate - baseline
    names = [*class_names, BACKGROUND]
    rows, cols = np.nonzero(delta)
    order = sorted(
        range(rows.size),
        key=lambda i: (-abs(int(delta[rows[i], cols[i]])), int(rows[i]), int(cols[i])),
    )
    cells = tuple(
        ConfusionCell(
            names[int(rows[i])],
            names[int(cols[i])],
            int(baseline[rows[i], cols[i]]),
            int(candidate[rows[i], cols[i]]),
            int(delta[rows[i], cols[i]]),
        )
        for i in order[:max_cells]
    )
    background = size - 1
    return ConfusionDelta(
        cells=cells,
        n_changed_cells=int(rows.size),
        missed_delta=int(delta[:background, background].sum()),
        false_positive_delta=int(delta[background, :background].sum()),
        baseline_total=int(baseline.sum()),
        candidate_total=int(candidate.sum()),
    )
