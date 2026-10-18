"""MMEngine hook: a Hook against a fake runner; isolation from the host, rank and stage rules."""

from __future__ import annotations

import json
import math
import pickle
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_frameworks import FakeMMRunner, install_fake_mmengine

from cvpilot import api, runs
from cvpilot.core.errors import HookNotFoundError, MissingExtraError
from cvpilot.hooks import mmengine as mm
from cvpilot.hooks._guard import MAX_CONSECUTIVE_FAILURES, MAX_POINTS_PER_CALL
from cvpilot.hooks.lightning import HookAttachError
from cvpilot.logging import clear_request_context


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "CVPILOT_ALERTS", "RANK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    install_fake_mmengine(monkeypatch)
    structlog.reset_defaults()
    yield
    clear_request_context()
    structlog.reset_defaults()


class RecordingSink:
    status = "running"

    def __init__(self, fail: bool = False) -> None:
        self.fail, self.logged, self.finished = fail, [], []
        self.run_id = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    def log(self, points: dict[str, float], *, step: int, epoch: int | None = None) -> None:
        if self.fail:
            raise RuntimeError("store is full")
        self.logged.append((dict(points), step))

    def finish(self, status: str) -> None:
        self.finished.append(status)
        self.status = status


def attached(
    sink: Any = None, *, interval: int = 1, **kw: Any
) -> tuple[FakeMMRunner, Any, Any, mm.MMEngineHook]:
    runner = FakeMMRunner(**kw)
    sink = sink or RecordingSink()
    hook = mm.MMEngineHook(interval=interval)
    hook.attach(runner, sink)
    return runner, runner.hooks[0], sink, hook


def test_matches_a_runner_of_the_framework_and_its_subclasses_only() -> None:
    hook = mm.MMEngineHook()
    assert hook.matches(FakeMMRunner())

    class FlexibleRunner(FakeMMRunner):
        pass

    FlexibleRunner.__module__ = "mmengine.runner.flexible_runner"
    assert hook.matches(FlexibleRunner())

    class MyRunner(FakeMMRunner):
        pass

    MyRunner.__module__ = "my_project.runner"
    assert hook.matches(MyRunner())
    assert not hook.matches(object()) and not hook.matches("x")

    class Imposter:
        def register_hook(self, *a: Any) -> None: ...

    Imposter.__name__ = "Runner"
    assert not hook.matches(Imposter())  # the name alone is not enough


def test_the_hook_is_named_for_its_extra_and_has_no_resume_helper() -> None:
    assert mm.MMEngineHook.name == "mmengine" and mm.MMEngineHook.requires == ("mmengine",)
    assert mm.MMEngineHook.license_notice is None
    assert not hasattr(mm, "resume_arguments")  # its checkpoints pickle objects the scan refuses


def test_attach_registers_a_subclassed_hook_late_and_detach_restores_the_runner() -> None:
    runner, hook_object, _, hook = attached()
    assert type(hook_object).__mro__[1].__name__ == "Hook"
    assert runner.priorities == ["VERY_LOW"]
    assert all(name in vars(runner) for name in ("train", "val", "test"))
    hook.detach()
    assert runner.hooks == [] and not any(n in vars(runner) for n in ("train", "val", "test"))


@pytest.mark.parametrize("interval", [0, -1, True, 2.5, "3", mm.MAX_INTERVAL + 1])
def test_a_bad_interval_is_refused(interval: Any) -> None:
    with pytest.raises(ValueError, match="interval"):
        mm.MMEngineHook(interval=interval)


def test_training_scalars_every_interval_and_validation_flow_into_the_sink() -> None:
    runner, _, sink, _ = attached(interval=2)
    assert runner.train() == "model" and runner.calls == ["train"]
    assert sink.logged == [
        ({"train/loss": 0.5, "train/loss_cls": 0.5}, 1),
        ({"train/loss": 0.25, "train/loss_cls": 0.5}, 3),
        ({"val/accuracy/top1": 0.75}, 4),
    ]
    assert sink.finished == []  # train leaves the run open for the stages that follow


def test_the_run_finishes_when_test_returns_and_not_twice() -> None:
    runner, _, sink, _ = attached()
    runner.train()
    runner.test()
    runner.test()
    assert sink.finished == ["finished"]
    assert ({"test/accuracy/top1": 0.7}, 4) in sink.logged


def test_a_script_that_stops_after_train_is_finished_at_exit() -> None:
    runner, hook_object, sink, _ = attached()
    runner.train()
    hook_object._core.finish_at_exit()
    assert sink.finished == ["finished"]


def test_a_failed_stage_is_not_finished_at_exit() -> None:
    runner, hook_object, sink, _ = attached(fail_with=ValueError("boom"))
    with pytest.raises(ValueError):
        runner.train()
    hook_object._core.finish_at_exit()
    assert sink.finished == ["failed"]


def test_repeats_at_one_step_are_skipped_not_raised() -> None:
    runner, _, sink, _ = attached()
    runner.at(3).fire("after_train_iter", 3, outputs={"loss": 1.0})
    runner.at(3).fire("after_train_iter", 3, outputs={"loss": 2.0, "aux": 0.1})
    runner.at(1).fire("after_train_iter", 1, outputs={"loss": 9.0})
    assert sink.logged == [({"train/loss": 1.0}, 3), ({"train/aux": 0.1}, 3)]


def test_only_rank_zero_logs_and_finishes() -> None:
    runner, _, sink, _ = attached(rank=1)
    runner.train()
    runner.test()
    assert sink.logged == [] and sink.finished == []


def test_a_non_zero_rank_in_the_environment_logs_nothing_even_before_the_first_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    runner, _, sink, _ = attached()  # the runner object itself claims rank zero
    runner.at(1).fire("after_train_iter", 1, outputs={"loss": 1.0})
    runner.at(1).fire("after_val_epoch", metrics={"acc": 1.0})
    assert sink.logged == []


@pytest.mark.parametrize("stage", ["train", "val", "test"])
def test_an_exception_in_any_stage_marks_the_run_failed_and_propagates_unchanged(
    stage: str,
) -> None:
    boom = ValueError("boom")
    runner, _, sink, _ = attached(fail_with=boom)
    with pytest.raises(ValueError) as info:
        getattr(runner, stage)()
    assert info.value is boom and sink.finished == ["failed"]


@pytest.mark.parametrize("stop", [KeyboardInterrupt(), SystemExit(3)])
def test_an_interrupt_or_exit_marks_the_run_preempted_and_is_never_swallowed(
    stop: BaseException,
) -> None:
    runner, _, sink, _ = attached(fail_with=stop)
    with pytest.raises(type(stop)):
        runner.train()
    assert sink.finished == ["preempted"]


def test_arguments_and_results_pass_through_the_wrapped_stages() -> None:
    runner, _, _, _ = attached()
    assert runner.val() == {"accuracy/top1": 0.8}


def test_metrics_after_the_run_finished_are_skipped_quietly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    runner, hook_object, sink, _ = attached()
    runner.test()
    count = len(sink.logged)
    runner.at(9).fire("after_val_epoch", metrics={"acc": 0.1})
    assert len(sink.logged) == count
    assert hook_object._core.guard.failures == 0
    assert "hook_failed" not in capsys.readouterr().err


def test_hostile_outputs_never_raise_and_are_counted() -> None:
    class Bad:
        def __float__(self) -> float:
            raise RuntimeError("tensor on a dead device")

    runner, hook_object, sink, _ = attached()
    outputs: dict[Any, Any] = {
        "loss": math.nan,
        "ok": 1.0,
        3: 1.0,
        None: 1.0,
        "text": "abc",
        "none": None,
        "bool": True,
        "bad": Bad(),
        "ctrl\x00key\n": 2.0,
        "k" * 5000: 3.0,
        "": 1.0,
    }
    runner.at(1).fire("after_train_iter", 1, outputs=outputs)
    ((points, step),) = sink.logged
    assert step == 1 and math.isnan(points["train/loss"]) and points["train/ok"] == 1.0
    assert "train/ctrl_key_" in points and max(len(k) for k in points) <= 256
    assert not hook_object._core.guard.failures and hook_object._core.guard.skipped >= 6


@pytest.mark.parametrize("outputs", [None, [], "x", 5, object()])
def test_outputs_that_are_not_a_mapping_are_ignored(outputs: Any) -> None:
    runner, hook_object, sink, _ = attached()
    runner.at(1).fire("after_train_iter", 1, outputs=outputs)
    runner.at(1).fire("after_val_epoch", metrics=outputs)
    assert sink.logged == [] and hook_object._core.guard.failures == 0


def test_an_ordered_mapping_of_values_with_item_style_floats_is_read() -> None:
    class Scalar:
        def __float__(self) -> float:
            return 0.25

    runner, _, sink, _ = attached()
    runner.at(0).fire("after_train_iter", 0, outputs=OrderedDict(loss=Scalar()))
    assert sink.logged == [({"train/loss": 0.25}, 0)]


@pytest.mark.parametrize("step", [None, -1, True, "x", 1.5, math.nan, math.inf])
def test_bad_steps_are_skipped_without_raising(step: Any) -> None:
    runner, _, sink, _ = attached()
    runner.iter = step
    runner.fire("after_train_iter", 0, outputs={"loss": 1.0})
    runner.fire("after_val_epoch", metrics={"acc": 1.0})
    assert sink.logged == []


def test_points_per_call_are_bounded() -> None:
    runner, hook_object, sink, _ = attached()
    runner.at(1).fire(
        "after_train_iter", 1, outputs={f"k{i}": 1.0 for i in range(MAX_POINTS_PER_CALL + 50)}
    )
    assert len(sink.logged[0][0]) == MAX_POINTS_PER_CALL
    assert hook_object._core.guard.skipped == 50


def test_a_failing_sink_never_raises_into_the_host_and_is_logged_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    runner, hook_object, sink, _ = attached(RecordingSink(fail=True))
    for step in range(1, MAX_CONSECUTIVE_FAILURES + 6):
        runner.at(step).fire("after_train_iter", step, outputs={"loss": 1.0})  # must not raise
    runner.test()  # a disabled hook still finishes the run
    assert hook_object._core.guard.disabled and sink.finished == ["finished"]
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    failed = [e for e in events if e["message"] == "hook_failed"]
    assert len(failed) == 1 and failed[0]["hook"] == "mmengine"
    assert failed[0]["error_type"] == "RuntimeError" and "store is full" not in json.dumps(events)


def test_a_sink_that_raises_on_finish_does_not_mask_the_training_exception() -> None:
    class BadFinish(RecordingSink):
        def finish(self, status: str) -> None:
            raise RuntimeError("cannot close the store")

    runner, _, _, _ = attached(BadFinish(), fail_with=ValueError("the real failure"))
    with pytest.raises(ValueError, match="the real failure"):
        runner.train()


def test_the_hook_cannot_be_pickled() -> None:
    _, hook_object, _, _ = attached()
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(hook_object)


def test_attach_to_something_that_is_not_a_runner_is_an_error() -> None:
    class Odd:
        register_hook = None

    with pytest.raises(HookAttachError):
        mm.MMEngineHook().attach(Odd(), RecordingSink())
    with pytest.raises(HookAttachError):
        mm.MMEngineHook().attach(object(), RecordingSink())


def test_without_the_framework_attach_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "mmengine.hooks", None)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[mmengine\]"):
        mm.MMEngineHook().attach(FakeMMRunner(), RecordingSink())


def test_autolog_end_to_end_records_a_real_run_and_announces_the_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "0.99.0")
    runner = FakeMMRunner()
    run = api.autolog(runner, project="proj", task="classification")
    runner.train()
    runner.test()
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert (
        view.meta.status == "finished" and view.stats.n_points == 2
    )  # the default interval is 10: no training iteration is a multiple
    assert view.meta.framework.startswith("mmengine==0.99.0")
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    (announce,) = [e for e in events if e["message"] == "hook_attached"]
    assert announce["hook"] == "mmengine" and announce["license_notice"] is None


def test_autolog_without_the_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    def missing(dist: str) -> str:
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(md, "version", missing)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[mmengine\]"):
        api.autolog(FakeMMRunner(), project="proj", task="classification")
    assert api._active is None


def test_an_unknown_object_is_still_hook_not_found() -> None:
    with pytest.raises(HookNotFoundError, match="mmengine"):
        api.autolog(object(), project="proj", task="classification")
    assert api._active is None


def test_alerts_work_through_the_hook_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    from fake_channels import RecordingChannel, use_channels

    monkeypatch.setattr(md, "version", lambda dist: "0.99.0")
    use_channels(monkeypatch, recording=RecordingChannel)
    runner = FakeMMRunner()
    api.autolog(
        runner,
        project="proj",
        task="classification",
        alerts={"channels": ["recording"], "rules": [{"rule": "nan_loss"}]},
    )
    runner.fire("before_run")
    runner.at(9).fire("after_train_iter", 9, outputs={"loss": float("nan")})
    runner.test()
    (chan,) = RecordingChannel.instances
    assert [(r, s) for r, s, _ in chan.sent] == [("nan_loss", 9)]
