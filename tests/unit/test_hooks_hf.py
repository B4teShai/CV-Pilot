"""Hugging Face ``Trainer`` hook: a callback against a fake trainer; isolation from the host."""

from __future__ import annotations

import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_frameworks import FakeHFTrainer, install_fake_transformers

from cvpilot import api, runs
from cvpilot.core.errors import HookNotFoundError, MissingExtraError
from cvpilot.hooks import hf_trainer as hf
from cvpilot.hooks._guard import MAX_CONSECUTIVE_FAILURES, MAX_POINTS_PER_CALL
from cvpilot.hooks.lightning import HookAttachError
from cvpilot.logging import clear_request_context


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "CVPILOT_ALERTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    install_fake_transformers(monkeypatch)
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


def attached(sink: Any = None, **kw: Any) -> tuple[FakeHFTrainer, Any, Any, hf.HFHook]:
    trainer = FakeHFTrainer(**kw)
    sink = sink or RecordingSink()
    hook = hf.HFHook()
    hook.attach(trainer, sink)
    return trainer, trainer.callbacks[0], sink, hook


def test_matches_a_trainer_of_the_framework_and_its_subclasses_only() -> None:
    hook = hf.HFHook()
    assert hook.matches(FakeHFTrainer())

    class Seq2SeqTrainer(FakeHFTrainer):
        pass

    Seq2SeqTrainer.__module__ = "transformers.trainer_seq2seq"
    assert hook.matches(Seq2SeqTrainer())

    class MyTrainer(FakeHFTrainer):  # a user's subclass lives in the user's module
        pass

    MyTrainer.__module__ = "my_project.training"
    assert hook.matches(MyTrainer())
    assert not hook.matches(object()) and not hook.matches("x")

    class Imposter:
        def add_callback(self, cb: Any) -> None: ...

    Imposter.__name__ = "Trainer"
    assert not hook.matches(Imposter())  # the name alone is not enough


def test_the_hook_is_named_for_its_extra() -> None:
    assert hf.HFHook.name == "hf" and hf.HFHook.requires == ("transformers",)
    assert hf.HFHook.license_notice is None


def test_attach_adds_a_subclassed_callback_and_detach_restores_the_trainer() -> None:
    trainer, callback, _, hook = attached()
    assert type(callback).__mro__[1].__name__ == "TrainerCallback"
    original = FakeHFTrainer.train
    assert trainer.train is not original and "train" in vars(trainer)
    hook.detach()
    assert trainer.callbacks == [] and "train" not in vars(trainer)


def test_train_scalars_evaluation_and_the_end_flow_into_the_sink() -> None:
    trainer, _, sink, _ = attached()
    assert trainer.train("a", b=1) == "output"
    assert trainer.train_calls == [(("a",), {"b": 1})]  # arguments and result pass through
    assert sink.logged == [
        ({"train/loss": 1.0, "train/epoch": 1.0}, 1),
        ({"train/loss": 0.5, "train/epoch": 2.0}, 2),
        ({"eval/loss": 0.5, "eval/accuracy": 0.9}, 2),
    ]
    assert sink.finished == ["finished"]


def test_evaluation_values_in_the_log_stream_are_not_recorded_twice() -> None:
    trainer, _, sink, _ = attached()
    trainer.fire("on_train_begin")
    trainer.at(5).fire("on_log", logs={"eval_loss": 0.4, "epoch": 1.0})  # Trainer.evaluate logs too
    trainer.at(5).fire("on_evaluate", metrics={"eval_loss": 0.4})
    assert sink.logged == [({"train/epoch": 1.0}, 5), ({"eval/loss": 0.4}, 5)]


def test_repeats_at_one_step_are_skipped_not_raised() -> None:
    trainer, _, sink, _ = attached()
    trainer.fire("on_train_begin")
    trainer.at(3).fire("on_log", logs={"loss": 1.0})
    trainer.at(3).fire("on_log", logs={"loss": 2.0, "learning_rate": 0.1})
    trainer.at(1).fire("on_log", logs={"loss": 9.0})
    assert sink.logged == [({"train/loss": 1.0}, 3), ({"train/learning_rate": 0.1}, 3)]


def test_a_save_is_an_event_with_the_step_and_no_path(capsys: pytest.CaptureFixture[str]) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    trainer, _, _, _ = attached()
    trainer.fire("on_train_begin")
    trainer.at(40).fire("on_save")
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    (saved,) = [e for e in events if e["message"] == "hook_checkpoint_saved"]
    assert saved["hook"] == "hf" and saved["step"] == 40 and saved["level"] == "INFO"
    assert set(saved) <= {
        "message", "hook", "step", "level", "logger", "service", "env", "request_id", "@timestamp",
    }  # fmt: skip


def test_only_rank_zero_logs_and_finishes() -> None:
    trainer, _, sink, _ = attached(rank_zero=False)
    trainer.train()
    assert sink.logged == [] and sink.finished == []


def test_an_exception_marks_the_run_failed_and_propagates_unchanged() -> None:
    boom = ValueError("boom")
    trainer, _, sink, _ = attached(fail_with=boom)
    with pytest.raises(ValueError) as info:
        trainer.train()
    assert info.value is boom and sink.finished == ["failed"]


@pytest.mark.parametrize("stop", [KeyboardInterrupt(), SystemExit(3)])
def test_an_interrupt_or_exit_marks_the_run_preempted_and_is_never_swallowed(
    stop: BaseException,
) -> None:
    trainer, _, sink, _ = attached(fail_with=stop)
    with pytest.raises(type(stop)):
        trainer.train()
    assert sink.finished == ["preempted"]  # stopped from outside, not a failure of the training


def test_a_second_train_end_or_a_failure_after_the_end_does_not_finish_twice() -> None:
    trainer, callback, sink, _ = attached()
    trainer.train()
    trainer.fire("on_train_end")
    assert sink.finished == ["finished"]


def test_metrics_after_the_run_finished_are_skipped_quietly(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    trainer, callback, sink, _ = attached()
    trainer.train()
    trainer.at(9).fire("on_evaluate", metrics={"eval_loss": 0.1})  # a manual evaluate() afterwards
    assert len(sink.logged) == 3
    assert callback._core.guard.failures == 0 and callback._core.guard.skipped >= 1
    assert "hook_failed" not in capsys.readouterr().err


def test_hostile_logs_never_raise_and_are_counted() -> None:
    class Bad:
        def __float__(self) -> float:
            raise RuntimeError("tensor on a dead device")

    trainer, callback, sink, _ = attached()
    trainer.fire("on_train_begin")
    logs: dict[Any, Any] = {
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
    trainer.at(1).fire("on_log", logs=logs)
    ((points, step),) = sink.logged
    assert step == 1 and math.isnan(points["train/loss"]) and points["train/ok"] == 1.0
    assert "train/ctrl_key_" in points and max(len(k) for k in points) <= 256
    assert not callback._core.guard.failures and callback._core.guard.skipped >= 6


@pytest.mark.parametrize("logs", [None, [], "x", 5, object()])
def test_logs_that_are_not_a_mapping_are_ignored(logs: Any) -> None:
    trainer, callback, sink, _ = attached()
    trainer.fire("on_train_begin")
    trainer.at(1).fire("on_log", logs=logs)
    trainer.at(1).fire("on_evaluate", metrics=logs)
    assert sink.logged == [] and callback._core.guard.failures == 0


@pytest.mark.parametrize("step", [None, -1, True, "x", 1.5, math.nan, math.inf])
def test_bad_steps_are_skipped_without_raising(step: Any) -> None:
    trainer, _, sink, _ = attached()
    trainer.fire("on_train_begin")
    trainer.state.global_step = step
    trainer.fire("on_log", logs={"loss": 1.0})
    assert sink.logged == []


def test_points_per_call_are_bounded() -> None:
    trainer, callback, sink, _ = attached()
    trainer.fire("on_train_begin")
    trainer.at(1).fire("on_log", logs={f"k{i}": 1.0 for i in range(MAX_POINTS_PER_CALL + 50)})
    assert len(sink.logged[0][0]) == MAX_POINTS_PER_CALL
    assert callback._core.guard.skipped == 50


def test_a_failing_sink_never_raises_into_the_host_and_is_logged_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    configure_logging("cvpilot-client", Settings(env="prod"))
    trainer, callback, sink, _ = attached(RecordingSink(fail=True))
    trainer.fire("on_train_begin")
    for step in range(1, MAX_CONSECUTIVE_FAILURES + 6):
        trainer.at(step).fire("on_log", logs={"loss": 1.0})  # must not raise
    trainer.fire("on_train_end")  # a disabled hook still finishes the run
    core = callback._core
    assert core.guard.disabled and sink.finished == ["finished"]
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    failed = [e for e in events if e["message"] == "hook_failed"]
    assert len(failed) == 1 and failed[0]["hook"] == "hf"
    assert failed[0]["error_type"] == "RuntimeError" and "store is full" not in json.dumps(events)


def test_a_sink_that_raises_on_finish_does_not_mask_the_training_exception() -> None:
    class BadFinish(RecordingSink):
        def finish(self, status: str) -> None:
            raise RuntimeError("cannot close the store")

    trainer, _, _, _ = attached(BadFinish(), fail_with=ValueError("the real failure"))
    with pytest.raises(ValueError, match="the real failure"):
        trainer.train()


def test_the_callback_cannot_be_pickled() -> None:
    _, callback, _, _ = attached()
    with pytest.raises(TypeError, match="pickled"):
        pickle.dumps(callback)


def test_attach_to_something_that_is_not_a_trainer_is_an_error() -> None:
    class Odd:
        add_callback = None

    with pytest.raises(HookAttachError):
        hf.HFHook().attach(Odd(), RecordingSink())
    with pytest.raises(HookAttachError):
        hf.HFHook().attach(object(), RecordingSink())


def test_without_the_framework_attach_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, hf.PACKAGE, None)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[hf\]"):
        hf.HFHook().attach(FakeHFTrainer(), RecordingSink())


def test_autolog_end_to_end_records_a_real_run_and_announces_the_hook(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import importlib.metadata as md

    monkeypatch.setattr(md, "version", lambda dist: "4.99.0")
    trainer = FakeHFTrainer()
    run = api.autolog(trainer, project="proj", task="classification")
    trainer.train()
    view = runs.get_run_view("proj", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.stats.n_points == 6
    assert view.meta.framework.startswith("transformers==4.99.0")
    events = [json.loads(x) for x in capsys.readouterr().err.splitlines() if x.startswith("{")]
    (announce,) = [e for e in events if e["message"] == "hook_attached"]
    assert announce["hook"] == "hf" and announce["license_notice"] is None


def test_autolog_without_the_package_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    def missing(dist: str) -> str:
        raise md.PackageNotFoundError(dist)

    monkeypatch.setattr(md, "version", missing)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[hf\]"):
        api.autolog(FakeHFTrainer(), project="proj", task="classification")
    assert api._active is None


def test_an_unknown_object_is_still_hook_not_found(tmp_path: Path) -> None:
    with pytest.raises(HookNotFoundError, match="hf"):
        api.autolog(object(), project="proj", task="classification")
    assert api._active is None


def test_alerts_work_through_the_hook_path(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib.metadata as md

    from fake_channels import RecordingChannel, use_channels

    monkeypatch.setattr(md, "version", lambda dist: "4.99.0")
    use_channels(monkeypatch, recording=RecordingChannel)
    trainer = FakeHFTrainer()
    api.autolog(
        trainer,
        project="proj",
        task="classification",
        alerts={"channels": ["recording"], "rules": [{"rule": "nan_loss"}]},
    )
    trainer.fire("on_train_begin")
    trainer.at(7).fire("on_log", logs={"loss": float("nan")})
    trainer.fire("on_train_end")
    (chan,) = RecordingChannel.instances
    assert [(r, s) for r, s, _ in chan.sent] == [("nan_loss", 7)]
    assert "train/loss is NaN at step 7" in chan.sent[0][2]


def test_a_custom_evaluation_prefix_is_recorded_once_under_eval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, _, sink, _ = attached()
    trainer.fire("on_train_begin")
    trainer.at(5).fire("on_log", logs={"val_loss": 0.4, "val_accuracy": 0.8, "loss": 1.0})
    trainer.at(5).fire("on_evaluate", metrics={"val_loss": 0.4, "val_accuracy": 0.8})
    keys = [k for points, _ in sink.logged for k in points]
    assert sorted(keys) == ["eval/val_accuracy", "eval/val_loss", "train/loss"]
    trainer.at(6).fire("on_log", logs={"test_loss": 0.3, "eval_loss": 0.3})
    trainer.at(6).fire("on_evaluate", metrics={"test_loss": 0.3})
    keys = [k for points, step in sink.logged for k in points if step == 6]
    assert keys == ["eval/test_loss"]


def test_a_non_zero_rank_logs_nothing_even_before_train_begin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RANK", "1")
    trainer, _, sink, _ = attached()  # the trainer object itself claims to be rank zero
    trainer.at(1).fire("on_log", logs={"loss": 1.0})
    trainer.at(1).fire("on_evaluate", metrics={"eval_loss": 1.0})
    assert sink.logged == []
