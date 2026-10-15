"""Builders for runs and eval snapshots with chosen metric values (no model, no evaluation)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, EvalSnapshot, Repro, RunMeta
from cvpilot.registry import tables
from cvpilot.store.local import LocalRunStore, open_store

H1 = "1" * 64
H2 = "2" * 64
SHA = "b" * 64
CLASSES = ("cat", "dog", "bird")
PARAMS: dict[str, Any] = {
    "iou_thresholds": [round(0.5 + 0.05 * i, 2) for i in range(10)],
    "confusion_iou": 0.5,
    "confusion_score": 0.25,
    "max_dets": 100,
}


def open_project(root: Path, project: str = "proj") -> LocalRunStore:
    return open_store(project, root)


def make_run(
    store: LocalRunStore,
    *,
    project: str = "proj",
    task: str = "detection",
    status: str = "finished",
) -> str:
    rid = new_run_id()
    started = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
    meta = RunMeta(
        run_id=rid,
        project=project,
        name="n",
        status=status,  # type: ignore[arg-type]
        task=task,  # type: ignore[arg-type]
        framework="python==3.12.1",
        host="h",
        started_at=started,
        ended_at=None if status in ("running", "resumed") else started,
        schema_version=SCHEMA_VERSION,
    )
    repro = Repro(
        run_id=rid,
        git_dirty=False,
        packages_lock_sha256=SHA,
        python_version="3.12.1",
        gpu_names=[],
        dataset_uri="/d",
        dataset_hash="merkle:" + SHA,
        config={},
        command_line="train.py",
    )
    store.put_run(meta, repro)
    return rid


def confusion_matrix(n_classes: int, diagonal: float = 10.0) -> np.ndarray[Any, Any]:
    matrix = np.zeros((n_classes + 1, n_classes + 1), dtype=np.int64)
    for k in range(n_classes):
        matrix[k, k] = int(diagonal)
    return matrix


def add_snapshot(
    store: LocalRunStore,
    run_id: str,
    *,
    h: str = H1,
    step: int = 0,
    task: str = "detection",
    class_names: Sequence[str] = CLASSES,
    metrics: Mapping[str, float] | None = None,
    per_class: Mapping[str, Mapping[str, float]] | None = None,
    slices: Mapping[str, Mapping[str, float]] | None = None,
    slice_spec: Mapping[str, str] | None = None,
    ci: Mapping[str, tuple[float, float]] | None = None,
    n_images: int = 100,
    params: Mapping[str, Any] | None = PARAMS,
    confusion: np.ndarray[Any, Any] | None = None,
) -> EvalSnapshot:
    names = list(class_names)
    if metrics is None:
        metrics = {"mAP50_95": 0.6, "mAP50": 0.8}
    if per_class is None:
        per_class = {n: {"AP50_95": 0.6, "AP50": 0.8, "AR100": 0.7, "support": 20.0} for n in names}
    matrix = confusion if confusion is not None else confusion_matrix(len(names))
    rows, cols = np.nonzero(matrix)
    scratch = store.path / f"scratch-{new_run_id()}.parquet"
    tables.get("parquet")().write_columns(
        scratch,
        {
            "true_class": rows.astype(np.int64),
            "pred_class": cols.astype(np.int64),
            "count": matrix[rows, cols].astype(np.int64),
        },
        metadata={"cvpilot.n_classes": str(len(names))},
    )
    conf = store.put_artifact(run_id, scratch, "confusion", {"n_classes": len(names)})
    scratch.unlink()
    pred = store.put_artifact_bytes(run_id, f"p{step}{h[:4]}".encode(), "predictions", {})
    snap = EvalSnapshot.model_validate(
        {
            "run_id": run_id,
            "eval_set_hash": h,
            "step": step,
            "task": task,
            "class_names": names,
            "n_images": n_images,
            "metrics": dict(metrics),
            "per_class": {k: dict(v) for k, v in per_class.items()},
            "slices": {k: dict(v) for k, v in (slices or {}).items()},
            "ci": {k: tuple(v) for k, v in (ci or {}).items()},
            "bootstrap": {"n_resamples": 100, "seed": 0, "level": 0.95} if ci else None,
            "ci_status": "computed" if ci else "off",
            "params": None if params is None else dict(params),
            "slice_spec": dict(slice_spec or {}),
            "confusion_sha256": conf.sha256,
            "predictions_sha256": pred.sha256,
        }
    )
    store.put_snapshot(snap)
    return snap
