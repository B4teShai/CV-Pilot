"""``log_eval``: snapshot, predictions and confusion artifacts, immutability, events, CLI-free."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
import structlog

from cvpilot import api, runs
from cvpilot.core.errors import (
    EvalInputError,
    NoActiveRunError,
    RunClosedError,
    SnapshotExistsError,
)
from cvpilot.eval import AnnotationTable, ImageTable, PredictionTable
from cvpilot.logging import clear_request_context, configure_logging
from cvpilot.settings import Settings

CLASSES = ["cat", "dog"]
A, B = (10.0, 10.0, 50.0, 50.0), (60.0, 60.0, 120.0, 140.0)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in (
        "CVPILOT_ENV",
        "CVPILOT_LOG_LEVEL",
        "CVPILOT_PROJECT",
        "CVPILOT_STORE_COMMAND_LINE",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    yield
    clear_request_context()
    structlog.reset_defaults()


@pytest.fixture
def logs() -> Any:
    buf = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="dev", log_level="DEBUG"), stream=buf)
    return lambda: [json.loads(x) for x in buf.getvalue().splitlines() if x.startswith("{")]


def tables(extra_pred: bool = False) -> tuple[PredictionTable, AnnotationTable]:
    images = ImageTable.from_columns(
        image_id=["i1", "i2", "i3"],
        width=[200, 200, 200],
        height=[200, 200, 200],
        meta={"camera": ["c1", "c1", "c2"]},
    )
    gts = AnnotationTable.from_columns(
        image_id=["i1", "i1", "i2", "i3"],
        class_id=[0, 1, 0, 1],
        boxes=[A, B, A, B],
        images=images,
    )
    rows = [("i1", 0, 0.9, A), ("i1", 1, 0.8, B), ("i2", 0, 0.7, A), ("i3", 1, 0.6, B)]
    if extra_pred:
        rows.append(("i3", 0, 0.5, A))
    preds = PredictionTable.from_columns(
        image_id=[r[0] for r in rows],
        class_id=[r[1] for r in rows],
        score=[r[2] for r in rows],
        boxes=[r[3] for r in rows],
    )
    return preds, gts


def start(task: str = "detection", name: str = "r") -> api.Run:
    return api.init("proj", task=task, name=name)  # type: ignore[arg-type]


def test_log_eval_persists_snapshot_predictions_and_confusion() -> None:
    preds, gts = tables()
    run = start()
    snap = run.log_eval(preds, gts, step=3, class_names=CLASSES, slices={"camera": "camera"})
    assert snap.step == 3 and snap.metrics["mAP50"] == pytest.approx(1.0)
    assert set(snap.slices) == {"camera=c1", "camera=c2"} and snap.bootstrap is not None
    run.finish()
    view = runs.get_run_view("proj", run.run_id, store_dir=Path("store"))
    assert view.snapshots == [snap]
    kinds = {a.sha256: a.kind for a in view.artifacts}
    assert kinds[snap.predictions_sha256] == "predictions"
    assert kinds[snap.confusion_sha256] == "confusion"
    store_dir = Path("store") / "proj" / "artifacts"
    pred_file = store_dir / snap.predictions_sha256[:2] / snap.predictions_sha256
    table = pq.read_table(pred_file)
    assert table.column_names == ["image_id", "cls", "score", "x1", "y1", "x2", "y2"]
    assert table.num_rows == 4 and table.column("image_id").to_pylist() == ["i1", "i1", "i2", "i3"]
    conf_file = store_dir / snap.confusion_sha256[:2] / snap.confusion_sha256
    conf = pq.read_table(conf_file)
    assert conf.column_names == ["true_class", "pred_class", "count"]
    assert conf.schema.metadata[b"cvpilot.n_classes"] == b"2"
    assert sum(conf.column("count").to_pylist()) == 4


def test_the_module_level_api_logs_to_the_active_run() -> None:
    preds, gts = tables()
    with pytest.raises(NoActiveRunError):
        api.log_eval(preds, gts, step=0)
    run = start()
    snap = api.log_eval(preds, gts, step=0, class_names=CLASSES, n_resamples=0)
    assert snap.run_id == run.run_id and snap.ci == {}
    run.finish()


def test_class_names_default_to_their_ids() -> None:
    preds, gts = tables()
    run = start()
    snap = run.log_eval(preds, gts, step=0, n_resamples=0)
    assert snap.class_names == ["0", "1"]
    run.finish()


def test_snapshots_are_immutable_and_keyed_by_set_hash_and_step() -> None:
    preds, gts = tables()
    run = start()
    run.log_eval(preds, gts, step=1, class_names=CLASSES, n_resamples=0)
    before = len(runs.get_run_view("proj", run.run_id, store_dir=Path("store")).artifacts)
    worse, _ = tables(extra_pred=True)
    with pytest.raises(SnapshotExistsError, match="immutable"):
        run.log_eval(worse, gts, step=1, class_names=CLASSES, n_resamples=0)
    run.log_eval(worse, gts, step=2, class_names=CLASSES, n_resamples=0)
    other_classes = run.log_eval(preds, gts, step=1, class_names=["x", "y"], n_resamples=0)
    assert (
        other_classes.eval_set_hash
        != run.log_eval(preds, gts, step=9, class_names=CLASSES, n_resamples=0).eval_set_hash
    )
    run.finish()
    view = runs.get_run_view("proj", run.run_id, store_dir=Path("store"))
    assert len(view.snapshots) == 4 and before == 2  # the refused call stored no artifacts


def test_identical_inputs_give_identical_artifacts_and_numbers_in_two_runs() -> None:
    preds, gts = tables()
    first, second = start(name="a"), None
    s1 = first.log_eval(preds, gts, step=0, class_names=CLASSES, seed=5, n_resamples=40)
    first.finish()
    second = start(name="b")
    s2 = second.log_eval(preds, gts, step=0, class_names=CLASSES, seed=5, n_resamples=40)
    second.finish()
    assert s1.predictions_sha256 == s2.predictions_sha256
    assert s1.confusion_sha256 == s2.confusion_sha256
    assert s1.model_copy(update={"run_id": s2.run_id}).model_dump_json() == s2.model_dump_json()


def test_finished_runs_refuse_log_eval_and_bad_steps_are_rejected() -> None:
    preds, gts = tables()
    run = start()
    for bad in (-1, True, 1.5):
        with pytest.raises(EvalInputError, match="step"):
            run.log_eval(preds, gts, step=bad, class_names=CLASSES)  # type: ignore[arg-type]
    run.finish()
    with pytest.raises(RunClosedError):
        run.log_eval(preds, gts, step=0, class_names=CLASSES)


def test_invalid_inputs_leave_nothing_behind() -> None:
    preds, gts = tables()
    bad = PredictionTable.from_columns(
        image_id=["i1"], class_id=[0], score=[float("nan")], boxes=[A]
    )
    run = start()
    with pytest.raises(EvalInputError):
        run.log_eval(bad, gts, step=0, class_names=CLASSES)
    run.finish()
    view = runs.get_run_view("proj", run.run_id, store_dir=Path("store"))
    assert view.snapshots == [] and [a.kind for a in view.artifacts] == []


def test_classification_runs_store_label_predictions(tmp_path: Path) -> None:
    ids = ["a", "b", "c"]
    gts = AnnotationTable.from_columns(image_id=ids, class_id=[0, 1, 1])
    preds = PredictionTable.from_columns(image_id=ids, class_id=[0, 1, 0], score=[0.9, 0.8, 0.7])
    run = start("classification")
    snap = run.log_eval(preds, gts, step=0, class_names=CLASSES, n_resamples=0)
    run.finish()
    assert snap.metrics["top1"] == pytest.approx(2 / 3)
    sha = snap.predictions_sha256
    table = pq.read_table(Path("store") / "proj" / "artifacts" / sha[:2] / sha)
    assert table.column_names == ["image_id", "cls", "score"]


def test_eval_completed_event_carries_counts_and_no_data(logs: Any) -> None:
    preds, gts = tables()
    run = start()
    snap = run.log_eval(preds, gts, step=4, class_names=CLASSES, n_resamples=0)
    run.finish()
    event = next(e for e in logs() if e["message"] == "eval_completed")
    assert event["run_id"] == run.run_id and event["step"] == 4
    assert event["eval_set_hash"] == snap.eval_set_hash and event["n_images"] == 3
    assert event["n_classes"] == 2 and event["level"] == "INFO" and event["duration_ms"] >= 0
    assert "i1" not in json.dumps(event) and event["request_id"].startswith(run.run_id)


def test_steps_outside_the_signed_64_bit_range_are_refused() -> None:
    preds, gts = tables()
    run = start()
    with pytest.raises(EvalInputError, match="step"):
        run.log_eval(preds, gts, step=2**63, class_names=CLASSES)
    run.log_eval(preds, gts, step=2**63 - 1, class_names=CLASSES, n_resamples=0)
    run.finish()


CHILD = """
import sys
sys.path.insert(0, sys.argv[4])
from pathlib import Path
from cvpilot.core.errors import SnapshotExistsError
from cvpilot.eval.record import record_eval
from cvpilot.runs.queries import open_existing
from test_eval_record import tables
store = open_existing("proj", sys.argv[1], Path(sys.argv[2]))
preds, gts = tables()
try:
    record_eval(store, store.get_run(sys.argv[1]), preds, gts, step=0, class_names=["cat", "dog"],
                n_resamples=30, seed=int(sys.argv[3]), allow_closed=True)
    print("ok")
except SnapshotExistsError:
    print("exists")
"""


def test_two_processes_with_different_seeds_leave_exactly_one_snapshot(tmp_path: Path) -> None:
    import subprocess
    import sys

    run = start()
    run.finish()
    script = tmp_path / "child.py"
    script.write_text(CHILD)
    here = str(Path(__file__).parent)
    procs = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, str(script), run.run_id, str(tmp_path / "store"), str(seed), here],
            stdout=subprocess.PIPE,
            text=True,
            env={**__import__("os").environ, "CVPILOT_STORE_DIR": str(tmp_path / "store")},
        )
        for seed in (1, 2)
    ]
    outputs = sorted(p.communicate()[0].strip() for p in procs)
    assert outputs == ["exists", "ok"]
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert len(view.snapshots) == 1


def test_default_class_names_come_from_the_ground_truth_only() -> None:
    preds, gts = tables()
    run = start()
    first = run.log_eval(preds, gts, step=0, n_resamples=0)
    wide = PredictionTable.from_columns(
        image_id=["i1"], class_id=[1], score=[0.5], boxes=[A]
    )  # a subset of the classes: same ground truth, same hash
    second = run.log_eval(wide, gts, step=1, n_resamples=0)
    assert first.eval_set_hash == second.eval_set_hash and first.class_names == ["0", "1"]
    bigger = PredictionTable.from_columns(image_id=["i1"], class_id=[7], score=[0.5], boxes=[A])
    with pytest.raises(EvalInputError, match="class id 7"):
        run.log_eval(bigger, gts, step=2, n_resamples=0)  # a new class no longer renames the set
    run.finish()
