"""Evaluate Parquet files into an existing run (``cvpilot eval``).

Predictions: ``image_id, cls, score`` and, for detection, ``x1, y1, x2, y2``.

Ground truth: ``image_id, cls`` and, for detection, ``x1, y1, x2, y2``; optional ``iscrowd``,
``width`` and ``height`` (constant per image) and any metadata columns named by ``--slices``. A
row whose ``cls`` is null describes an image without objects (its boxes are null too), so empty
images take part in the evaluation without a separate image list. Files and class lists are
bounded before they are read; see ``cvpilot.store.tables`` and ``MAX_CLASS_FILE_BYTES``.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from cvpilot.core.errors import EvalInputError
from cvpilot.core.schema import MAX_CLASSES, EvalSnapshot, TableIO, Task
from cvpilot.eval import bootstrap as bs
from cvpilot.eval import tables as t
from cvpilot.eval.record import record_eval
from cvpilot.logging import get_logger
from cvpilot.registry import tables as table_registry
from cvpilot.runs.queries import open_existing

Array = np.ndarray[Any, Any]

MAX_CLASS_FILE_BYTES = 512 * 1024
"""Largest class-name file; 2 048 names of 128 characters would fit in 264 KB."""


def _read_regular_file(path: Path, limit: int) -> bytes:
    """At most ``limit + 1`` bytes of a regular file, opened without following symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise EvalInputError(f"cannot read the class file {path.name}: {exc.strerror}") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise EvalInputError(f"class file {path.name} is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read(limit + 1)
    finally:
        os.close(fd)


def read_class_names(path: Path) -> list[str]:
    """Class names from a JSON list of strings or a text file with one name per line."""
    raw = _read_regular_file(path, MAX_CLASS_FILE_BYTES)
    if len(raw) > MAX_CLASS_FILE_BYTES:
        raise EvalInputError(f"class file {path.name} is too large")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise EvalInputError(f"class file {path.name} is not valid UTF-8") from None
    stripped = text.strip()
    if stripped.startswith("["):
        try:
            names = json.loads(stripped)
        except (ValueError, RecursionError):
            raise EvalInputError(f"class file {path.name} is not valid JSON") from None
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise EvalInputError(f"class file {path.name} must hold a JSON list of strings")
    else:
        names = [line.strip() for line in text.splitlines() if line.strip()]
    if len(names) > MAX_CLASSES:
        raise EvalInputError(f"class file {path.name} has more than {MAX_CLASSES} classes")
    return t.check_class_names(names)


def _integral(values: Array, what: str) -> Array:
    if values.dtype.kind == "f":
        if not bool(np.isfinite(values).all()) or bool((values != np.floor(values)).any()):
            raise EvalInputError(f"column {what!r} must hold whole numbers")
        if bool((np.abs(values) > 2**53).any()):
            raise EvalInputError(f"column {what!r} has values beyond the supported range")
        return values.astype(np.int64)
    return values


def load_predictions(path: Path, task: Task, io: TableIO) -> t.PredictionTable:
    """Read a predictions file into a table (bounded, validated on use)."""
    boxes = list(t.BOX_COLUMNS) if task == "detection" else []
    cols = io.read_columns(
        path,
        required=["image_id", "cls", "score", *boxes],
        max_rows=t.MAX_PREDICTIONS,
        max_string_length=t.MAX_ID_LENGTH,
        max_string_bytes=t.MAX_STRING_COLUMN_BYTES,
    )
    cols["cls"] = _integral(cols["cls"], "cls")
    return t.PredictionTable.from_table(cols, task=task)


def _per_image(values: Array, first: Array, inverse: Array, what: str) -> Array:
    per: Array = values[first]
    if not bool(np.all(per[inverse] == values)):
        raise EvalInputError(f"column {what!r} has conflicting values for the same image id")
    return per


def load_ground_truth(
    path: Path, task: Task, io: TableIO, slice_columns: Mapping[str, str] | None = None
) -> t.AnnotationTable:
    """Read a ground-truth file; rows with a null ``cls`` declare images without objects."""
    boxes = list(t.BOX_COLUMNS) if task == "detection" else []
    meta_names = sorted({c.removeprefix("meta.") for c in (slice_columns or {}).values()})
    cols = io.read_columns(
        path,
        required=["image_id", "cls", *boxes],
        optional=["iscrowd", "width", "height", *meta_names],
        nullable=["cls", *boxes],
        max_rows=t.MAX_ANNOTATIONS,
        max_string_length=t.MAX_ID_LENGTH,
        max_string_bytes=t.MAX_STRING_COLUMN_BYTES,
    )
    ids = cols["image_id"]
    uniq, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    inverse = inverse.reshape(-1)
    raw = cols["cls"]
    has = ~cols.get("cls.null", np.zeros(raw.shape[0], dtype=bool))
    cls = _integral(raw[has], "cls")  # a real NaN here is rejected, only nulls mark empty rows
    for box in boxes:
        box_null = cols.get(f"{box}.null", np.zeros(raw.shape[0], dtype=bool))
        if bool((~box_null[~has]).any()):
            raise EvalInputError("a row with null cls has box values; boxes must be null too")
        if bool(box_null[has].any()):
            raise EvalInputError(f"a row with a class has a null {box} value")
    width = height = None
    if ("width" in cols) != ("height" in cols):
        raise EvalInputError("ground truth needs both width and height columns or neither")
    if "width" in cols:
        width = _integral(_per_image(cols["width"], first, inverse, "width"), "width")
        height = _integral(_per_image(cols["height"], first, inverse, "height"), "height")
    meta = {
        name: _per_image(cols[name], first, inverse, name) for name in meta_names if name in cols
    }
    images = t.ImageTable.from_columns(image_id=uniq, width=width, height=height, meta=meta)
    box_array = np.column_stack([cols[c][has].astype(np.float64) for c in boxes]) if boxes else None
    crowd = cols["iscrowd"][has] if "iscrowd" in cols else None
    return t.AnnotationTable.from_columns(
        image_id=ids[has], class_id=cls, boxes=box_array, iscrowd=crowd, images=images
    )


def eval_file(
    project: str,
    run_id: str,
    *,
    predictions: Path,
    ground_truth: Path,
    class_names: list[str],
    step: int,
    allow_closed: bool = False,
    provenance: Mapping[str, str | None] | None = None,
    slices: Mapping[str, str] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    store_dir: Path | None = None,
) -> EvalSnapshot:
    """Evaluate two Parquet files into ``run_id`` and store the snapshot (``eval_completed``).

    Raises:
        InvalidRunIdError, RunNotFoundError: the run id is malformed or unknown.
        EvalInputError, SnapshotExistsError, StoreError: see ``record_eval``.
    """
    store = open_existing(project, run_id, store_dir)
    try:
        meta = store.get_run(run_id)
        io = table_registry.get("parquet")()
        preds = load_predictions(predictions, meta.task, io)
        gts = load_ground_truth(ground_truth, meta.task, io, slices)
        log = get_logger("cvpilot.eval")

        def emit(level_name: str, event: str, **fields: object) -> None:
            getattr(log, level_name)(event, run_id=run_id, **fields)

        return record_eval(
            store,
            meta,
            preds,
            gts,
            step=step,
            class_names=class_names,
            slices=slices,
            n_resamples=n_resamples,
            seed=seed,
            level=level,
            workers=bs.DEFAULT_WORKERS,
            emit=emit,
            io=io,
            allow_closed=allow_closed,
            provenance=provenance,
        )
    finally:
        store.close()
