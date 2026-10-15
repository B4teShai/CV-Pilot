"""Record an evaluation in a run: compute it, store its artifacts, then its snapshot.

Order of effects: validate and hash the ground truth, refuse if the snapshot key exists, run
the evaluation, write ``predictions`` and ``confusion`` Parquet files to a private temporary
directory and register them as content-addressed artifacts, and finally insert the snapshot.
A failure before the last step leaves at most content-addressed artifacts that identical
inputs would produce again; the snapshot itself is all-or-nothing and immutable.

The event ``eval_completed`` (INFO) carries counts and the set hash, never image ids, paths or
scores.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from cvpilot.core.errors import EvalInputError, RunClosedError, SnapshotExistsError
from cvpilot.core.metrics import max_int
from cvpilot.core.schema import (
    ACTIVE_STATUSES,
    MAX_CLASSES,
    MAX_STEP,
    EvalSnapshot,
    Provenance,
    RunMeta,
    RunStore,
    TableIO,
)
from cvpilot.eval import bootstrap as bs
from cvpilot.eval import tables as t
from cvpilot.eval.engine import EvalResult, eval_set_hash, evaluate
from cvpilot.registry import tables as table_registry

Emit = Callable[..., None]


def default_class_names(gts: t.AnnotationTable) -> list[str]:
    """``["0", "1", ...]`` up to the largest class id of the *ground truth*.

    Predictions never influence the names: the same ground truth hashes the same at every
    epoch, and a prediction with a class id above the ground truth's range is rejected by the
    class range check. Pass ``class_names`` to evaluate classes that have no ground truth.
    """
    top = max_int(gts.class_id) if len(gts) else -1
    if top < 0:
        raise EvalInputError("class_names are required when the ground truth has no labels")
    if top >= MAX_CLASSES:
        raise EvalInputError(f"class id {top} is outside the supported range; pass class_names")
    return [str(i) for i in range(top + 1)]


def _check_step(step: object) -> int:
    if isinstance(step, bool) or not isinstance(step, (int, np.integer)):
        raise EvalInputError("step must be a non-negative integer")
    if not 0 <= int(step) <= MAX_STEP:
        raise EvalInputError("step must be a non-negative integer")
    return int(step)


def _prediction_columns(preds: t.PredictionTable, task: str) -> dict[str, np.ndarray[Any, Any]]:
    columns: dict[str, np.ndarray[Any, Any]] = {
        "image_id": preds.image_id,
        "cls": preds.class_id,
        "score": preds.score,
    }
    if task == "detection":
        for i, name in enumerate(t.BOX_COLUMNS):
            columns[name] = preds.boxes[:, i]
    return columns


def _confusion_columns(confusion: np.ndarray[Any, Any]) -> dict[str, np.ndarray[Any, Any]]:
    rows, cols = np.nonzero(confusion)
    return {
        "true_class": rows.astype(np.int64),
        "pred_class": cols.astype(np.int64),
        "count": confusion[rows, cols].astype(np.int64),
    }


def record_eval(
    store: RunStore,
    meta: RunMeta,
    preds: t.PredictionTable,
    gts: t.AnnotationTable,
    *,
    step: int,
    class_names: Sequence[str] | None = None,
    slices: Mapping[str, str] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
    workers: int = bs.DEFAULT_WORKERS,
    emit: Emit | None = None,
    io: TableIO | None = None,
    allow_closed: bool = False,
    provenance: Mapping[str, str | None] | None = None,
) -> EvalSnapshot:
    """Evaluate ``preds`` against ``gts`` for the run ``meta`` and store the result.

    Raises:
        RunClosedError: the run is finished, failed or preempted and ``allow_closed`` is false.
        EvalInputError: invalid inputs (see ``cvpilot.eval.validate``).
        SnapshotExistsError: a snapshot for ``(run, eval_set_hash, step)`` already exists.
        StoreError: an artifact or the snapshot could not be written.
    """
    started = time.perf_counter()
    status = store.get_run(meta.run_id).status
    if status not in ACTIVE_STATUSES and not allow_closed:
        raise RunClosedError(
            f"run {meta.run_id} is {status}; evaluating into it needs allow_closed (--allow-closed)"
        )
    if provenance:
        Provenance.model_validate(dict(provenance))  # fail before any work
    step = _check_step(step)
    names = list(class_names) if class_names is not None else default_class_names(gts)
    digest = eval_set_hash(gts, names, task=meta.task)
    if store.has_snapshot(meta.run_id, digest, step):
        raise SnapshotExistsError(
            f"snapshot ({digest[:12]}, step {step}) of run {meta.run_id} exists; "
            "snapshots are immutable"
        )
    result = evaluate(
        preds,
        gts,
        task=meta.task,
        class_names=names,
        slices=slices,
        n_resamples=n_resamples,
        seed=seed,
        level=level,
        workers=workers,
    )
    snapshot = _store_result(store, meta, result, preds, step, io, provenance)
    if emit is not None:
        emit(
            "info",
            "eval_completed",
            step=step,
            task=meta.task,
            eval_set_hash=snapshot.eval_set_hash,
            n_images=snapshot.n_images,
            n_classes=len(names),
            n_predictions=len(preds),
            duration_ms=round((time.perf_counter() - started) * 1000, 1),
        )
    return snapshot


def _store_result(
    store: RunStore,
    meta: RunMeta,
    result: EvalResult,
    preds: t.PredictionTable,
    step: int,
    io: TableIO | None,
    provenance: Mapping[str, str | None] | None = None,
) -> EvalSnapshot:
    writer = io if io is not None else table_registry.get("parquet")()
    with tempfile.TemporaryDirectory(prefix="cvpilot-eval-") as scratch:
        pred_path = Path(scratch) / "predictions.parquet"
        conf_path = Path(scratch) / "confusion.parquet"
        writer.write_columns(pred_path, _prediction_columns(preds, meta.task))
        writer.write_columns(
            conf_path,
            _confusion_columns(result.confusion),
            metadata={
                "cvpilot.layout": "sparse (true_class, pred_class, count); index C is background",
                "cvpilot.n_classes": str(len(result.class_names)),
            },
        )
        predictions = store.put_artifact(
            meta.run_id, pred_path, "predictions", {"n_rows": len(preds)}
        )
        confusion = store.put_artifact(
            meta.run_id, conf_path, "confusion", {"n_classes": len(result.class_names)}
        )
    snapshot = result.to_snapshot(
        run_id=meta.run_id,
        step=step,
        confusion_sha256=confusion.sha256,
        predictions_sha256=predictions.sha256,
        provenance=provenance,
    )
    store.put_snapshot(snapshot)
    return snapshot
