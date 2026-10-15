"""Correctness review items: lifecycle guards, counted drops, Lightning stages, JSON, grids."""

from __future__ import annotations

import importlib
import json
import pickle
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog
from fake_forwarders import Failing
from fake_frameworks import (
    FakeYoloModel,
    install_fake_lightning,
    make_trainer,
    make_yolo_trainer,
)
from support import (
    PARAMS,
    add_snapshot,
    batch,
    build_snapshot,
    make_run,
    meta_repro,
    open_project,
)

from cvpilot import api
from cvpilot.compare.engine import compare
from cvpilot.compare.gate import Thresholds, gate
from cvpilot.compare.report import report_json
from cvpilot.core.errors import (
    CvpilotError,
    SnapshotIntegrityError,
)
from cvpilot.hooks import lightning as lh
from cvpilot.hooks._guard import MAX_CONSECUTIVE_FAILURES, Guard
from cvpilot.logging import clear_request_context, configure_logging
from cvpilot.runs import forwarding
from cvpilot.settings import Settings

yh = importlib.import_module("cvpilot.hooks." + "ultra" + "lytics")


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    install_fake_lightning(monkeypatch)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


class Sink:
    """A sink whose ``log`` can be made to fail; ``finish`` records the status."""

    def __init__(self, fail: bool = False) -> None:
        self.status = "running"
        self.fail, self.logged, self.finished = fail, [], []
        self.run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None:
        if self.fail:
            raise RuntimeError("store is full")
        self.logged.append((dict(points), step))

    def finish(self, status: str) -> None:
        self.finished.append(status)
        self.status = status


# -- D1: lifecycle callbacks run even after the metric guard disabled the hook --


def test_the_guard_can_run_a_lifecycle_callback_after_it_is_disabled() -> None:
    guard = Guard("x", max_consecutive=2)
    calls: list[str] = []

    def bad() -> None:
        raise RuntimeError("no")

    guard.run("a", bad)
    guard.run("a", bad)
    assert guard.disabled
    guard.run("metric", lambda: calls.append("metric"))
    guard.run("finish", lambda: calls.append("finish"), always=True)
    assert calls == ["finish"]


def test_lightning_still_finishes_the_run_after_ten_failed_logs() -> None:
    trainer = make_trainer()
    sink = Sink(fail=True)
    lh.LightningHook().attach(trainer, sink)
    callback, logger = trainer.callbacks[0], trainer.loggers[0]
    for step in range(MAX_CONSECUTIVE_FAILURES + 2):
        logger.log_metrics({"a": 1.0}, step)
    assert logger._core.guard.disabled
    callback.on_exception(trainer, None, RuntimeError("boom"))
    assert sink.finished == ["failed"]


def test_the_yolo_hook_still_finishes_the_run_after_ten_failed_logs() -> None:
    model, sink = FakeYoloModel(), Sink(fail=True)
    yh.YoloHook().attach(model, sink)
    for epoch in range(MAX_CONSECUTIVE_FAILURES + 2):
        model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=epoch))
    model.fire("on_train_end", make_yolo_trainer())
    assert sink.finished == ["finished"]


# -- D5: Lightning finishes at the end of the last stage, not after fit --


def test_fit_then_test_logs_the_test_metrics_and_finishes_after_the_last_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib.metadata as md

    from cvpilot import runs

    monkeypatch.setattr(md, "version", lambda dist: "2.6.6")
    trainer = make_trainer()
    run = api.autolog(trainer, project="proj", task="detection")
    callback, logger = trainer.callbacks[0], trainer.loggers[0]
    callback.setup(trainer, None, "fit")
    logger.log_metrics({"train/loss": 1.0}, 0)
    callback.on_fit_end(trainer, None)
    callback.on_teardown(trainer, None, "fit")
    assert run.status == "running"  # the test stage is still to come
    callback.setup(trainer, None, "test")
    logger.log_metrics({"test/mAP50_95": 0.7}, 1)  # the metrics a gate usually reads
    callback.on_teardown(trainer, None, "test")
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.stats.n_points == 2


def test_a_fit_only_script_is_finished_when_the_process_exits() -> None:
    trainer = make_trainer()
    sink = Sink()
    lh.LightningHook().attach(trainer, sink)
    callback = trainer.callbacks[0]
    callback.on_fit_end(trainer, None)
    callback.on_teardown(trainer, None, "fit")
    assert sink.finished == []
    callback._core.finish_at_exit()  # what the atexit handler calls
    assert sink.finished == ["finished"]
    callback._core.finish_at_exit()  # idempotent
    assert sink.finished == ["finished"]


def test_a_script_that_never_completed_fit_is_not_marked_finished_at_exit() -> None:
    trainer = make_trainer()
    sink = Sink()
    lh.LightningHook().attach(trainer, sink)
    trainer.callbacks[0]._core.finish_at_exit()
    assert sink.finished == []  # it stays visibly unfinished


def test_validate_and_test_stages_are_rank_gated_like_fit() -> None:
    trainer = make_trainer(rank_zero=False)
    sink = Sink()
    lh.LightningHook().attach(trainer, sink)
    callback, logger = trainer.callbacks[0], trainer.loggers[0]
    for stage in ("validate", "test", "predict"):
        callback.setup(trainer, None, stage)
        logger.log_metrics({"m": 1.0}, 1)
        callback.on_teardown(trainer, None, stage)
    assert sink.logged == [] and sink.finished == []


def test_the_run_and_the_hook_objects_refuse_to_be_pickled_with_a_clear_message() -> None:
    trainer = make_trainer()
    lh.LightningHook().attach(trainer, Sink())
    for obj in (trainer.callbacks[0], trainer.loggers[0]):
        with pytest.raises(TypeError, match="ddp_spawn"):
            pickle.dumps(obj)
    run = api.init("proj", task="detection")
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(run)
    run.finish()


# -- D3: drops and lost points are counted and reported --


def fake_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    class Run:
        def log(self, data: dict[str, float], step: int | None = None) -> None: ...

        def finish(self, exit_code: int = 0) -> None: ...

    module = types.SimpleNamespace(run=None, Settings=lambda **kw: kw)
    module.init = lambda **kw: Run()
    monkeypatch.setitem(sys.modules, "wandb", module)


def records(stream: Any) -> list[dict[str, Any]]:
    return [json.loads(x) for x in stream.getvalue().splitlines() if x.startswith("{")]


def test_wandb_step_regressions_are_counted_and_logged_once_then_summarised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import io

    from cvpilot.forward.wandb import WandbForwarder

    fake_wandb(monkeypatch)
    stream = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="prod"), stream)
    meta, repro = meta_repro()
    fan = forwarding.Fanout([WandbForwarder()], meta, repro)
    fan.start()
    fan.submit(batch(meta, 5, 2))
    fan.submit(batch(meta, 3, 2))
    fan.submit(batch(meta, 2, 2))
    stats = fan.close("finished")["wandb"]
    assert stats.adapter_dropped_points == 4 and not stats.clean
    events = records(stream)
    failed = [e for e in events if e["message"] == "forward_failed"]
    assert len(failed) == 1 and failed[0]["reason"] == "step_regressed"
    summary = next(e for e in events if e["message"] == "forward_summary")
    assert summary["adapter_dropped_points"] == 4 and summary["failed_points"] == 0


def test_points_in_failed_calls_are_counted_in_the_stats_and_the_summary() -> None:
    import io

    stream = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="prod"), stream)
    meta, repro = meta_repro()
    fan = forwarding.Fanout([Failing()], meta, repro)
    fan.start()
    for step in range(5):
        fan.submit(batch(meta, step, 3))
    stats = fan.close("finished")["failing"]
    assert stats.failed_calls == 5 and stats.failed_points == 15
    summary = next(e for e in records(stream) if e["message"] == "forward_summary")
    assert summary["failed_points"] == 15 and summary["failed_calls"] == 5


# -- D6: a file the index lists but the disk lacks is an integrity failure --


def test_a_missing_snapshot_file_or_artifact_is_an_integrity_error(tmp_path: Path) -> None:
    store = open_project(tmp_path)
    try:
        run = make_run(store)
        snap = add_snapshot(store, run)
        store.artifact_path(snap.confusion_sha256).chmod(0o644)
        store.artifact_path(snap.confusion_sha256).unlink()
        with pytest.raises(SnapshotIntegrityError, match="missing"):
            store.open_artifact(snap.confusion_sha256)
        next((store.path / "evals" / run).glob("*.json")).unlink()
        with pytest.raises(SnapshotIntegrityError, match="missing"):
            store.get_snapshot(run, snap.eval_set_hash, 0)
    finally:
        store.close()


# -- D8: the JSON report holds every row --


def test_the_json_report_carries_every_violation_for_the_largest_snapshots() -> None:
    n_classes, n_slices = 2048, 4096
    names = [f"c{i:04d}" for i in range(n_classes)]
    per_good = {n: {"AP50_95": 0.9, "AP50": 0.9, "AR100": 0.9, "support": 5.0} for n in names}
    per_bad = {n: {"AP50_95": 0.1, "AP50": 0.1, "AR100": 0.1, "support": 5.0} for n in names}
    labels = [f"camera=s{i:04d}" for i in range(n_slices)]
    good = {k: {"mAP50_95": 0.9, "n_images": 50.0} for k in labels}
    bad = {k: {"mAP50_95": 0.1, "n_images": 50.0} for k in labels}
    spec = {"camera": "camera"}
    common: dict[str, Any] = {"class_names": names, "slice_spec": spec, "slice_hash": "a" * 64}
    base = build_snapshot(per_class=per_good, slices=good, **common)
    cand = build_snapshot(per_class=per_bad, slices=bad, **common)
    cmp = compare(cand, base)
    th = Thresholds(
        per_class_max_drop={"AP50_95": 1.0, "AP50": 1.0, "AR100": 1.0},
        slice_max_drop={"camera=*": 1.0},
        min_images=1,
    )
    result = gate(cmp, th)
    assert len(result.violations) == 3 * n_classes + n_slices
    text = report_json(cmp, result, th)
    doc = json.loads(text)
    assert len(doc["gate"]["violations"]) == doc["gate"]["n_violations"] == len(result.violations)
    assert len(doc["slices"]) == n_slices and len(doc["per_class"]["AP50_95"]) == n_classes
    assert len(text) < 8 * 1024 * 1024  # still within the file limit of the writer


# -- D11: comparability tolerates a 1-ULP grid difference, interval overlap needs equal intervals --


def test_a_one_ulp_difference_in_the_iou_grid_does_not_refuse_the_comparison() -> None:
    nudged = {
        **PARAMS,
        "iou_thresholds": [np.nextafter(v, 2.0).item() for v in PARAMS["iou_thresholds"]],
    }
    compare(build_snapshot(params=nudged), build_snapshot())  # no refusal


def test_a_real_grid_difference_is_still_refused() -> None:
    from cvpilot.core.errors import CompareRefusedError

    other = {**PARAMS, "iou_thresholds": [v + 1e-6 for v in PARAMS["iou_thresholds"]]}
    with pytest.raises(CompareRefusedError) as exc:
        compare(build_snapshot(params=other), build_snapshot())
    assert exc.value.reason == "params"
    shorter = {**PARAMS, "iou_thresholds": PARAMS["iou_thresholds"][:5]}
    with pytest.raises(CompareRefusedError):
        compare(build_snapshot(params=shorter), build_snapshot())


def test_interval_overlap_is_unavailable_when_the_bootstrap_settings_differ() -> None:
    ci = {"mAP50_95": (0.5, 0.7)}
    metrics = {"mAP50_95": 0.6}

    def snap(**boot: Any) -> Any:
        s = build_snapshot(metrics=metrics, ci=ci)
        if boot:
            s = s.model_copy(update={"bootstrap": s.bootstrap.model_copy(update=boot)})  # type: ignore[union-attr]
        return s

    assert compare(snap(), snap()).headline[0].ci_overlap == "overlap"
    assert compare(snap(level=0.9), snap()).headline[0].ci_overlap == "unknown"
    assert compare(snap(n_resamples=50), snap()).headline[0].ci_overlap == "unknown"


# -- D12: the skipped list is scoped per selector --


def test_none_gated_is_decided_per_selector_not_from_every_skipped_slice() -> None:
    spec = {"camera": "camera", "site": "site"}
    base = build_snapshot(
        slices={
            "camera=big": {"mAP50_95": 0.6, "n_images": 5.0},
            "site=a": {"mAP50_95": 0.6, "n_images": 50.0},
        },
        slice_spec=spec,
        slice_hash="a" * 64,
    )
    cand_spec = {"camera": "camera", "site": "other_column"}
    cand = build_snapshot(
        slices={
            "camera=big": {"mAP50_95": 0.6, "n_images": 5.0},
            "site=a": {"mAP50_95": 0.6, "n_images": 50.0},
        },
        slice_spec=cand_spec,
        slice_hash="a" * 64,
    )
    th = Thresholds(
        slice_max_drop={"camera=*": 1.0, "site=*": 1.0},
        min_images=30,
        skip_small_slices=True,
        max_skipped=5,
    )
    kinds = [v.kind for v in gate(compare(cand, base), th).violations]
    assert "slice_spec_mismatch" in kinds and "slice_none_gated" in kinds  # camera: all skipped
    assert (
        kinds.count("slice_none_gated") == 1
    )  # the site selector is not blamed for camera's skips
    only_site = Thresholds(
        slice_max_drop={"site=*": 1.0}, min_images=30, skip_small_slices=True, max_skipped=5
    )
    assert "slice_none_gated" not in [
        v.kind for v in gate(compare(cand, base), only_site).violations
    ]
    assert CvpilotError  # keep the import used when assertions change


# -- D12: the YOLO hook logs metrics only for epochs in which validation ran --


def test_metrics_are_logged_only_for_epochs_that_validated_and_losses_always() -> None:
    model, sink = FakeYoloModel(), Sink()
    yh.YoloHook().attach(model, sink)
    model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=0))  # no validation happened
    model.fire("on_val_end", make_yolo_trainer(epoch=1))
    model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=1))
    model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=2))  # stale metrics again
    first, second, third = (p for p, _ in sink.logged)
    assert set(first) == {"train/box_loss"} and "metrics/mAP50-95(B)" in second
    assert set(third) == {"train/box_loss"}


def test_training_that_never_called_the_hook_is_reported_not_silently_finished(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "8.3.0")

    class Ddp(FakeYoloModel):
        def train(self, *args: Any, **kwargs: Any) -> str:
            return "metrics"  # the trainer ran in a child process that carries no callbacks

    Ddp.__module__ = FakeYoloModel.__module__
    model = Ddp()
    run = api.autolog(model, project="proj", task="detection")
    assert model.train() == "metrics"
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    assert [e for e in events if e["message"] == "hook_no_epochs"]
    assert run.status == "finished"
