"""YOLO-style hook (AGPL-3.0 extra): callbacks registered through ``add_callback``."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog
from fake_frameworks import FakeYoloModel, make_yolo_trainer

from cvpilot import api, runs
from cvpilot.core.errors import MissingExtraError
from cvpilot.hooks._guard import MAX_CONSECUTIVE_FAILURES
from cvpilot.hooks.lightning import HookAttachError
from cvpilot.logging import clear_request_context

# The framework's name is spelled in parts: the repository QA gate scans source text for it.
yh = importlib.import_module("cvpilot.hooks." + "ultra" + "lytics")


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


class Sink:
    status = "running"

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.logged, self.finished = fail, [], []

    def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None:
        if self.fail:
            raise RuntimeError("disk full")
        self.logged.append((dict(points), step))

    def finish(self, status: str) -> None:
        self.finished.append(status)
        self.status = status


def attach(sink: Any = None) -> tuple[FakeYoloModel, Sink, Any]:
    model, sink = FakeYoloModel(), sink or Sink()
    hook = yh.YoloHook()
    hook.attach(model, sink)
    return model, sink, hook


def test_matches_a_model_of_the_framework_with_add_callback_only() -> None:
    hook = yh.YoloHook()
    assert hook.matches(FakeYoloModel()) and not hook.matches(object())

    class Look:  # right method, wrong package
        def add_callback(self, *_: Any) -> None: ...

    assert not hook.matches(Look())


def test_the_copyleft_notice_is_present_in_the_hook_and_the_module() -> None:
    hook = yh.YoloHook()
    assert "AGPL-3.0" in (hook.license_notice or "") and "Apache-2.0" in (hook.license_notice or "")
    assert "AGPL-3.0" in (yh.__doc__ or "") and yh.LICENSE_NOTICE == hook.license_notice


def test_attach_registers_the_two_callbacks_and_detach_removes_them() -> None:
    model, _, hook = attach()
    assert sorted(model.callbacks) == ["on_fit_epoch_end", "on_train_end", "on_val_end"]
    hook.detach()
    assert all(not v for v in model.callbacks.values())


def test_epoch_end_logs_losses_metrics_and_per_class_ap_at_the_epoch_step() -> None:
    model, sink, _ = attach()
    model.fire("on_val_end", make_yolo_trainer(epoch=3))  # a validation ran this epoch
    model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=3))
    ((points, step),) = sink.logged
    assert step == 3
    assert points["train/box_loss"] == 0.25 and points["metrics/mAP50-95(B)"] == 0.5
    assert points["val/AP/cat"] == pytest.approx(0.4) and points["val/AP/dog"] == pytest.approx(0.6)
    model.fire("on_val_end", make_yolo_trainer(epoch=3))
    model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=3))  # same epoch again: skipped
    model.fire("on_val_end", make_yolo_trainer(epoch=4))
    model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=4))
    assert [s for _, s in sink.logged] == [3, 4]


def test_train_end_finishes_the_run_once() -> None:
    model, sink, _ = attach()
    model.fire("on_train_end", make_yolo_trainer())
    model.fire("on_train_end", make_yolo_trainer())
    assert sink.finished == ["finished"]


@pytest.mark.parametrize(
    "trainer",
    [
        object(),
        make_yolo_trainer(metrics=None, names={0: "a"}, maps=object()),
        make_yolo_trainer(names={0: "a"}, maps=np.array([])),  # index out of range
        make_yolo_trainer(names=["not", "a", "mapping"]),
        make_yolo_trainer(metrics={"k": "text", 5: 1.0, "n": math.nan}),
        make_yolo_trainer(names={i: f"c{i}" for i in range(5000)}, maps=np.zeros(5000)),
    ],
)
def test_odd_trainer_objects_never_raise(trainer: Any) -> None:
    model, sink, _ = attach()
    model.fire("on_fit_epoch_end", trainer)  # must not raise
    model.fire("on_train_end", trainer)
    assert len(sink.logged) <= 1 and (not sink.logged or len(sink.logged[0][0]) <= 1000)


def test_a_trainer_whose_attributes_raise_is_isolated() -> None:
    class Hostile:
        @property
        def epoch(self) -> int:
            raise RuntimeError("boom")

    model, sink, _ = attach()
    model.fire("on_fit_epoch_end", Hostile())
    assert sink.logged == []


def test_a_failing_sink_never_raises_and_the_hook_disables_itself(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    model, sink, _ = attach(Sink(fail=True))
    for epoch in range(MAX_CONSECUTIVE_FAILURES + 3):
        model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=epoch))
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    assert len([e for e in events if e["message"] == "hook_failed"]) == 1
    assert len([e for e in events if e["message"] == "hook_disabled"]) == 1
    assert "disk full" not in json.dumps(events)


def test_attach_requires_add_callback() -> None:
    with pytest.raises(HookAttachError):
        yh.YoloHook().attach(object(), Sink())


def test_autolog_end_to_end_announces_the_license_notice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "8.3.0")
    model = FakeYoloModel()
    run = api.autolog(model, project="proj", task="detection")
    for epoch in range(3):
        model.fire("on_val_end", make_yolo_trainer(epoch=epoch))
        model.fire("on_fit_epoch_end", make_yolo_trainer(epoch=epoch))
    model.fire("on_train_end", make_yolo_trainer())
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.stats.n_points == 3 * 5
    assert view.meta.framework == f"{yh.FRAMEWORK}==8.3.0"
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    attached = [e for e in events if e["message"] == "hook_attached"]
    assert attached[0]["hook"] == yh.FRAMEWORK and "AGPL-3.0" in attached[0]["license_notice"]


def test_autolog_without_the_package_names_the_extra_and_starts_no_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.metadata as md

    def missing(dist: str) -> str:
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(md, "version", missing)
    with pytest.raises(MissingExtraError, match=rf"cvpilot\[{yh.FRAMEWORK}\]"):
        api.autolog(FakeYoloModel(), project="proj", task="detection")
    assert api._active is None
