"""Builders for runs and eval snapshots with chosen metric values (no model, no evaluation)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from cvpilot.core.hashing import new_run_id
from cvpilot.core.schema import SCHEMA_VERSION, EvalSnapshot, MetricPoint, Repro, RunMeta
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


def build_snapshot(
    run_id: str | None = None,
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
    confusion_sha256: str = SHA,
    predictions_sha256: str = SHA,
) -> EvalSnapshot:
    """A snapshot object with chosen values; nothing is stored."""
    names = list(class_names)
    if metrics is None:
        metrics = {"mAP50_95": 0.6, "mAP50": 0.8}
    if per_class is None:
        per_class = {n: {"AP50_95": 0.6, "AP50": 0.8, "AR100": 0.7, "support": 20.0} for n in names}
    return EvalSnapshot.model_validate(
        {
            "run_id": run_id or new_run_id(),
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
            "confusion_sha256": confusion_sha256,
            "predictions_sha256": predictions_sha256,
        }
    )


def add_snapshot(
    store: LocalRunStore,
    run_id: str,
    *,
    confusion: np.ndarray[Any, Any] | None = None,
    **kw: Any,
) -> EvalSnapshot:
    """Build a snapshot with real confusion and predictions artifacts and store it."""
    step = kw.get("step", 0)
    h = kw.get("h", H1)
    names = list(kw.get("class_names", CLASSES))
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
    snap = build_snapshot(
        run_id, confusion_sha256=conf.sha256, predictions_sha256=pred.sha256, **kw
    )
    store.put_snapshot(snap)
    return snap


def write_eval_files(
    folder: Path, *, n: int = 40, broken: int = 0, shift_gt: bool = False
) -> tuple[Path, Path, Path]:
    """Parquet predictions, ground truth and class names for ``cvpilot eval``.

    ``n`` images with one box each (alternating two classes and two cameras); the predictions
    of the first ``broken`` images sit away from the ground truth, so they are misses plus
    false positives. ``shift_gt`` moves every ground-truth box, which changes the eval set hash.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder.mkdir(parents=True, exist_ok=True)
    ids = [f"img{i:03d}" for i in range(n)]
    gt_box = (15.0, 15.0, 55.0, 55.0) if shift_gt else (10.0, 10.0, 50.0, 50.0)
    far = (120.0, 120.0, 180.0, 180.0)
    pq.write_table(
        pa.table(
            {
                "image_id": ids,
                "cls": pa.array([i % 2 for i in range(n)], type=pa.int64()),
                "x1": [gt_box[0]] * n,
                "y1": [gt_box[1]] * n,
                "x2": [gt_box[2]] * n,
                "y2": [gt_box[3]] * n,
                "width": [200] * n,
                "height": [200] * n,
                "camera": [f"c{i % 2}" for i in range(n)],
            }
        ),
        folder / "g.parquet",
    )
    boxes = [far if i < broken else (10.0, 10.0, 50.0, 50.0) for i in range(n)]
    pq.write_table(
        pa.table(
            {
                "image_id": ids,
                "cls": [i % 2 for i in range(n)],
                "score": [0.9] * n,
                "x1": [b[0] for b in boxes],
                "y1": [b[1] for b in boxes],
                "x2": [b[2] for b in boxes],
                "y2": [b[3] for b in boxes],
            }
        ),
        folder / "p.parquet",
    )
    (folder / "c.txt").write_text("cat\ndog\n")
    return folder / "p.parquet", folder / "g.parquet", folder / "c.txt"


def meta_repro() -> tuple[RunMeta, Repro]:
    rid = new_run_id()
    meta = RunMeta(
        run_id=rid,
        project="p",
        name="n",
        status="running",
        task="detection",
        framework="python==3.12.1",
        host="h",
        started_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
        schema_version=SCHEMA_VERSION,
    )
    repro = Repro(
        run_id=rid,
        git_dirty=False,
        packages_lock_sha256=SHA,
        python_version="3.12",
        gpu_names=[],
        dataset_uri="/d",
        dataset_hash="merkle:" + SHA,
        config={},
        command_line="train.py",
    )
    return meta, repro


def batch(meta: RunMeta, step: int, n: int = 3) -> list[MetricPoint]:
    return [
        MetricPoint(run_id=meta.run_id, key=f"k{i}", step=step, value=step + i / 10, wall_time=1.0)
        for i in range(n)
    ]
