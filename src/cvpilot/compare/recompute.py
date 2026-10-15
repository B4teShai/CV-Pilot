"""Recompute a snapshot's results from its stored predictions on trusted ground truth.

A snapshot is only as trustworthy as whoever wrote it: ``put_snapshot`` and ``record_eval`` are
public API, so anyone who can write the store can store any metric values with any
``eval_set_hash`` and valid digests. ``verify`` closes that gap for the *metrics*: it evaluates
the predictions artifact of the snapshot (read through ``open_artifact``, SHA-256 verified)
against ground truth supplied by the caller with the parameters the snapshot recorded, and the
stored results must equal the recomputed ones: headline metrics, per-class rows, slice rows,
image count, slice membership hash and the confusion matrix, each within ``TOLERANCE`` (the same
code on the same inputs gives bit-identical numbers; the tolerance only absorbs platform
differences in floating point). Bootstrap intervals are not recomputed (they are not gate
inputs; their resampling stream depends on the numpy version).

What it does not prove: that the stored *predictions* came from the model under test. A forged
predictions file recomputes consistently. Bind predictions to the commit with ``--require-commit``
and, where contributors are adversaries, produce them in the trusted job itself.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cvpilot.compare import confusion as cf
from cvpilot.compare.text import safe_text
from cvpilot.core.errors import RecomputeMismatchError, StoreError
from cvpilot.core.schema import EvalSnapshot, RunStore
from cvpilot.eval import engine
from cvpilot.eval.files import load_ground_truth, load_predictions
from cvpilot.registry import tables

TOLERANCE = 1e-9
"""Largest absolute difference between a stored and a recomputed metric (fractions in [0, 1])."""
MAX_PREDICTIONS_BYTES = 1 << 30
"""Largest predictions artifact read for a recomputation (1 GiB, the Parquet reader's limit)."""
SUPPORTED_MAX_DETS = 100
"""``EvalParams.max_dets`` the evaluator uses; another recorded value cannot be checked."""


@dataclass(frozen=True)
class RecomputeSpec:
    """Trusted inputs of a recomputation: the ground-truth Parquet and the class names."""

    ground_truth: Path
    class_names: list[str]


def _fail(snapshot: EvalSnapshot, what: str, message: str) -> RecomputeMismatchError:
    return RecomputeMismatchError(
        f"snapshot of run {snapshot.run_id} does not match its recomputation: {message}",
        run_id=snapshot.run_id,
        what=safe_text(what, 64),
    )


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= TOLERANCE


def _same_rows(
    snapshot: EvalSnapshot,
    what: str,
    stored: Mapping[str, Mapping[str, float]],
    fresh: Mapping[str, Mapping[str, float]],
) -> None:
    if set(stored) != set(fresh):
        raise _fail(snapshot, what, f"{what} cover different entries")
    for name, row in stored.items():
        other = fresh[name]
        if set(row) != set(other) or not all(_close(row[k], other[k]) for k in row):
            raise _fail(snapshot, f"{what}:{name}", f"{what} entry {safe_text(name, 40)!r} differs")


def verify(store: RunStore, snapshot: EvalSnapshot, spec: RecomputeSpec) -> None:
    """Fail closed unless ``snapshot`` equals a fresh evaluation on ``spec``'s ground truth.

    Raises:
        RecomputeMismatchError: any stored result differs, or the snapshot cannot be checked
            (other class names, other ground truth, unsupported parameters).
        SnapshotIntegrityError, StoreError: the predictions artifact is missing or fails its
            digest.
        EvalInputError: the supplied ground truth or the stored predictions are invalid.
    """
    names = list(snapshot.class_names)
    if list(spec.class_names) != names:
        raise _fail(snapshot, "class_names", "the supplied class names differ from the snapshot's")
    params = snapshot.params
    if params is None or params.max_dets != SUPPORTED_MAX_DETS:
        raise _fail(snapshot, "params", "evaluation parameters are missing or unsupported")
    io = tables.get("parquet")()
    with store.open_artifact(snapshot.predictions_sha256) as handle:
        data = handle.read(MAX_PREDICTIONS_BYTES + 1)
    if len(data) > MAX_PREDICTIONS_BYTES:
        raise StoreError("the predictions artifact is too large to recompute")
    with tempfile.TemporaryDirectory(prefix="cvpilot-recompute-") as scratch:
        path = Path(scratch) / "predictions.parquet"
        path.write_bytes(data)
        preds = load_predictions(path, snapshot.task, io)
    gts = load_ground_truth(spec.ground_truth, snapshot.task, io, snapshot.slice_spec or None)
    if engine.eval_set_hash(gts, names, task=snapshot.task) != snapshot.eval_set_hash:
        raise _fail(
            snapshot, "eval_set_hash", "the supplied ground truth is not the evaluation set"
        )
    fresh = engine.evaluate(
        preds,
        gts,
        task=snapshot.task,
        class_names=names,
        slices=snapshot.slice_spec or None,
        iou_thresholds=params.iou_thresholds,
        confusion_iou=params.confusion_iou,
        confusion_score=params.confusion_score,
        n_resamples=0,
    )
    if fresh.n_images != snapshot.n_images:
        raise _fail(snapshot, "n_images", "the image count differs")
    if fresh.slice_hash != snapshot.slice_hash:
        raise _fail(snapshot, "slice_hash", "the slice membership differs")
    if set(fresh.metrics) != set(snapshot.metrics) or not all(
        _close(snapshot.metrics[k], fresh.metrics[k]) for k in snapshot.metrics
    ):
        bad = sorted(k for k in snapshot.metrics if k not in fresh.metrics)
        bad += sorted(
            k
            for k in snapshot.metrics
            if k in fresh.metrics and not _close(snapshot.metrics[k], fresh.metrics[k])
        )
        raise _fail(snapshot, f"metrics:{(bad or ['keys'])[0]}", "a headline metric differs")
    _same_rows(snapshot, "per_class", snapshot.per_class, fresh.per_class)
    _same_rows(snapshot, "slices", snapshot.slices, fresh.slices)
    stored = cf.load_confusion(store, snapshot.confusion_sha256, len(names))
    if not np.array_equal(stored, fresh.confusion):
        raise _fail(snapshot, "confusion", "the confusion matrix differs")
