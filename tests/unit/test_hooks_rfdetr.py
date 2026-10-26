"""RF-DETR hook: a logger and a callback on the trainer that ``RFDETR.train`` builds, against a
fake model and a fake legacy Lightning package; isolation from the host, rank and status rules."""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path
from typing import Any

import pytest
import structlog
from fake_frameworks import RFDETR, RFDETRNano, install_fake_rfdetr

from cvpilot import api, runs
from cvpilot.core.errors import MissingExtraError
from cvpilot.hooks import rfdetr as rf
from cvpilot.hooks._guard import MAX_CONSECUTIVE_FAILURES
from cvpilot.hooks.lightning import HookAttachError
from cvpilot.logging import clear_request_context


@pytest.fixture(autouse=True)
def env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    for var in ("CVPILOT_ENV", "CVPILOT_PROJECT", "CVPILOT_FORWARD_TO", "CVPILOT_ALERTS", "RANK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield
    if api._active is not None:  # noqa: SLF001
        api._active.abandon()  # noqa: SLF001
        api._active = None  # noqa: SLF001
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
    monkeypatch: pytest.MonkeyPatch, model: Any = None, sink: Any = None, **fake: Any
) -> tuple[Any, RecordingSink, rf.RFDETRHook, Any]:
    training = install_fake_rfdetr(monkeypatch, **fake)
    model = model if model is not None else RFDETRNano()
    sink = sink if sink is not None else RecordingSink()
    hook = rf.RFDETRHook()
    hook.attach(model, sink)
    return model, sink, hook, training


def test_matches_the_frameworks_models_and_their_subclasses_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hook = rf.RFDETRHook()
    assert hook.matches(RFDETRNano()) and hook.matches(RFDETR())

    class MyModel(RFDETRNano):
        pass

    MyModel.__module__ = "my_project.models"
    assert hook.matches(MyModel())

    class Imposter:
        def train(self) -> None: ...

    Imposter.__name__ = "RFDETR"
    assert not hook.matches(Imposter())  # the name alone is not enough
    assert not hook.matches(object()) and not hook.matches("RFDETR")


def test_the_hook_is_named_for_its_extra_and_has_no_resume_helper() -> None:
    assert rf.RFDETRHook.name == "rfdetr" and rf.RFDETRHook.requires == ("rfdetr",)
    assert rf.RFDETRHook.license_notice is None  # Apache-2.0
    assert not hasattr(rf, "resume_arguments")  # its resume falls back to full unpickling


def test_training_and_validation_metrics_flow_into_the_sink_and_train_finishes_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sink, _, training = attached(monkeypatch)
    model.train(epochs=2, batch_size=2)
    assert model.train_calls == [{"epochs": 2, "batch_size": 2}]  # arguments pass through
    assert sink.logged == [
        ({"train/loss": 2.0, "epoch": 0.0}, 49),
        ({"train/loss": 1.5, "epoch": 0.0}, 99),
        ({"val/mAP_50_95": 0.25, "val/mAP_50": 0.5}, 99),  # "epoch" was logged at 99 already
    ]
    assert sink.finished == ["finished"]
    (trainer,) = training.built
    assert len(trainer.callbacks) == 2 and len(trainer.loggers) == 2  # one of each added


def test_the_trainer_factory_is_wrapped_only_while_train_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sink, hook, training = attached(monkeypatch)
    original = training.build_trainer
    model.train()
    assert training.build_trainer is original
    other = training.build_trainer()  # evaluate() and other code paths are left alone
    assert other.callbacks == ["rfdetr-callback"] and other.loggers == ["rfdetr-csv-logger"]
    assert "train" not in vars(RFDETRNano)  # the class is untouched, only this instance
    hook.detach()
    assert "train" not in vars(model)


def test_an_exception_marks_the_run_failed_and_propagates_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = RuntimeError("CUDA out of memory")
    model, sink, _, training = attached(monkeypatch, model=RFDETRNano(fail_with=error))
    original = training.build_trainer
    with pytest.raises(RuntimeError) as info:
        model.train()
    assert info.value is error
    assert sink.finished == ["failed"]
    assert training.build_trainer is original  # restored on the way out too


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(1)])
def test_an_interrupt_marks_the_run_preempted(
    monkeypatch: pytest.MonkeyPatch, interrupt: BaseException
) -> None:
    model, sink, _, _ = attached(monkeypatch, model=RFDETRNano(fail_with=interrupt))
    with pytest.raises(type(interrupt)):
        model.train()
    assert sink.finished == ["preempted"]


def test_ranks_other_than_zero_log_nothing_and_finish_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sink, _, _ = attached(monkeypatch, rank_zero=False)
    model.train()
    assert sink.logged == [] and sink.finished == []


def test_the_rank_environment_variable_also_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RANK", "1")
    model, sink, _, _ = attached(monkeypatch)
    model.train()
    assert sink.logged == [] and sink.finished == []


def test_a_failing_store_never_raises_into_training_and_the_hook_stands_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = RFDETRNano(steps=MAX_CONSECUTIVE_FAILURES + 5)
    model, sink, hook, _ = attached(monkeypatch, model=model, sink=RecordingSink(fail=True))
    model.train()  # every log call fails; none of them reaches the training loop
    core = hook._cores[0]  # noqa: SLF001
    assert core.guard.disabled and core.guard.failures == MAX_CONSECUTIVE_FAILURES
    assert sink.finished == ["finished"]  # the lifecycle still runs after the metrics stood down


def test_a_model_whose_training_extras_are_missing_still_fails_the_framework_way(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, sink, _, _ = attached(monkeypatch, training=False)
    with pytest.raises(ImportError):
        model.train()
    assert sink.finished == ["failed"]


def test_without_the_lightning_package_attach_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_rfdetr(monkeypatch)
    monkeypatch.setitem(sys.modules, "py" + "to" + "rch_lightning", None)
    with pytest.raises(MissingExtraError, match=r"cvpilot\[rfdetr\]"):
        rf.RFDETRHook().attach(RFDETRNano(), RecordingSink())


def test_attach_refuses_an_object_without_train(monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_rfdetr(monkeypatch)
    model = RFDETRNano()
    model.train = None  # type: ignore[assignment,method-assign]
    with pytest.raises(HookAttachError):
        rf.RFDETRHook().attach(model, RecordingSink())


def test_the_added_callback_and_logger_refuse_to_be_pickled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _, _, training = attached(monkeypatch)
    model.train()
    (trainer,) = training.built
    for item in (trainer.callbacks[-1], trainer.loggers[-1]):
        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(item)


def test_autolog_attaches_to_a_model_and_the_run_ends_finished(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import importlib.metadata as md

    install_fake_rfdetr(monkeypatch)
    monkeypatch.setattr(md, "version", lambda dist: "1.11.2")
    model = RFDETRNano()
    run = api.autolog(model, project="rf", task="detection")
    model.train(epochs=1)
    view = runs.get_run_view("rf", run.run_id, store_dir=tmp_path / "store")
    assert view.meta.status == "finished" and view.meta.framework.startswith("rfdetr==1.11.2")
    assert view.stats.n_points == 6


def test_the_hook_module_imports_no_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("rfdetr", "rfdetr.training", "py" + "to" + "rch_lightning"):
        monkeypatch.setitem(sys.modules, name, None)  # any import of them would fail
    importlib.reload(rf)
    assert rf.RFDETRHook().matches(object()) is False


def test_a_constructor_that_fails_never_escapes_into_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Broken:
        def __init__(self, core: Any) -> None:
            raise RuntimeError("constructor failed")

    monkeypatch.setattr(rf, "build_classes", lambda *a, **k: (Broken, Broken))
    model, sink, _, training = attached(monkeypatch)
    model.train()  # the factory wrapper's failure is isolated: training goes on without the hook
    (trainer,) = training.built
    assert trainer.callbacks == ["rfdetr-callback"] and trainer.loggers == ["rfdetr-csv-logger"]
    assert sink.logged == []


def test_a_renamed_trainer_factory_is_reported_instead_of_silently_recording_nothing(
    monkeypatch: pytest.MonkeyPatch, events: Any
) -> None:
    model, sink, _, training = attached(monkeypatch)
    del training.build_trainer  # a framework release that renamed the patch target
    with pytest.raises(AttributeError):  # the fake framework looks the name up and fails, as the
        model.train()  # real one would; the hook has already said why it recorded nothing
    names = [e["message"] for e in events()]
    assert names.count("hook_target_missing") == 1 and "hook_not_attached" not in names
    assert sink.logged == [] and sink.finished == ["failed"]


def test_a_train_that_never_built_a_trainer_through_the_factory_is_reported(
    monkeypatch: pytest.MonkeyPatch, events: Any
) -> None:
    class QuietModel(RFDETRNano):
        def train(self, **kwargs: Any) -> None:  # a framework whose train() skips the factory
            return None

    QuietModel.__module__ = "my_project.models"
    model, sink, _, _ = attached(monkeypatch, model=QuietModel())
    model.train()
    names = [e["message"] for e in events()]
    assert names.count("hook_not_attached") == 1
    assert sink.logged == [] and sink.finished == ["finished"]


def test_the_compatibility_check_reads_the_framework_source_without_importing_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = tmp_path / "rfdetr" / "training"
    package.mkdir(parents=True)
    (tmp_path / "rfdetr" / "__init__.py").write_text("raise SystemExit('imported')\n")
    (package / "__init__.py").write_text("raise SystemExit('imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("rfdetr", "rfdetr.training"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    problem = rf.compatibility()
    assert problem is not None and "build_trainer" in problem
    (package / "__init__.py").write_text("from .trainer import build_trainer\n")
    assert rf.compatibility() is None
    monkeypatch.setattr(sys, "path", [p for p in sys.path if str(tmp_path) not in p])
    for name in ("rfdetr", "rfdetr.training"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    assert rf.compatibility() is None  # not installed: nothing to report
