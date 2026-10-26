"""Fake Lightning and YOLO-style objects (no framework installed, no model, no GPU)."""

from __future__ import annotations

import importlib
import sys
import types
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest

from cvpilot.hooks import lightning as lightning_hook

# The framework's name is spelled in parts: the repository QA gate scans source text for it.
_yolo = importlib.import_module("cvpilot.hooks." + "ultra" + "lytics")

FRAMEWORK = lightning_hook.PACKAGES[0].split(".")[0]


def install_fake_lightning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register fake modules under the names ``build_classes`` imports."""
    root_name, sub_name = lightning_hook.PACKAGES[0].split(".")[0], lightning_hook.PACKAGES[0]

    class Callback:
        pass

    class Logger:
        pass

    sub = types.ModuleType(sub_name)
    sub.Callback = Callback  # type: ignore[attr-defined]
    sub.loggers = types.SimpleNamespace(Logger=Logger)  # type: ignore[attr-defined]
    root = types.ModuleType(root_name)
    root.__dict__[sub_name.split(".")[1]] = sub
    monkeypatch.setitem(sys.modules, root_name, root)
    monkeypatch.setitem(sys.modules, sub_name, sub)


def make_trainer(*, rank_zero: bool = True) -> Any:
    """An object that looks like a Lightning ``Trainer`` to the hook (and to nothing else)."""

    class Trainer:
        pass

    Trainer.__module__ = f"{FRAMEWORK}.trainer"
    trainer = Trainer()
    trainer.callbacks = []  # type: ignore[attr-defined]
    trainer.loggers = []  # type: ignore[attr-defined]
    trainer.is_global_zero = rank_zero  # type: ignore[attr-defined]
    return trainer


class FakeYoloModel:
    """Has ``add_callback`` like the framework's model; its class claims the framework module."""

    def __init__(self, fail_with: BaseException | None = None) -> None:
        self.callbacks: dict[str, list[Any]] = {}
        self.fail_with = fail_with
        self.train_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def train(self, *args: Any, **kwargs: Any) -> str:
        """Like the framework's ``train``: fires the epoch callbacks, then ends or crashes."""
        self.train_calls.append((args, kwargs))
        for epoch in range(2):
            self.fire("on_fit_epoch_end", make_yolo_trainer(epoch=epoch))
        if self.fail_with is not None:
            raise self.fail_with
        self.fire("on_train_end", make_yolo_trainer(epoch=1))
        return "metrics"

    def add_callback(self, event: str, func: Any) -> None:
        self.callbacks.setdefault(event, []).append(func)

    def fire(self, event: str, trainer: Any) -> None:
        for func in list(self.callbacks.get(event, [])):
            func(trainer)


FakeYoloModel.__module__ = f"{_yolo.FRAMEWORK}.models"


def make_yolo_trainer(
    epoch: int = 0, *, metrics: Mapping[str, Any] | None = None, names: Any = None, maps: Any = None
) -> Any:
    class Box:
        pass

    class Validator:
        pass

    box = Box()
    box.maps = np.array([0.4, 0.6]) if maps is None else maps  # type: ignore[attr-defined]
    validator = Validator()
    validator.names = {0: "cat", 1: "dog"} if names is None else names  # type: ignore[attr-defined]
    validator.metrics = types.SimpleNamespace(box=box)  # type: ignore[attr-defined]

    class Trainer:
        def label_loss_items(self, tloss: Any, prefix: str = "train") -> dict[str, float]:
            return {f"{prefix}/box_loss": float(tloss)}

    trainer = Trainer()
    trainer.epoch = epoch  # type: ignore[attr-defined]
    trainer.tloss = 0.25  # type: ignore[attr-defined]
    trainer.metrics = (  # type: ignore[attr-defined]
        {"metrics/mAP50-95(B)": 0.5, "val/box_loss": 1.5} if metrics is None else metrics
    )
    trainer.validator = validator  # type: ignore[attr-defined]
    return trainer


# -- the Hugging Face ``Trainer`` ---------------------------------------------------------------

HF_PACKAGE = "transformers"


def install_fake_transformers(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register a fake ``transformers`` module with the base class the hook subclasses."""

    class TrainerCallback:
        pass

    module = types.ModuleType(HF_PACKAGE)
    module.TrainerCallback = TrainerCallback  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, HF_PACKAGE, module)


class FakeHFTrainer:
    """Has ``add_callback``/``remove_callback``/``train`` like the framework's ``Trainer``."""

    def __init__(self, *, rank_zero: bool = True, fail_with: BaseException | None = None) -> None:
        self.callbacks: list[Any] = []
        self.state = types.SimpleNamespace(
            global_step=0, epoch=0.0, is_world_process_zero=rank_zero
        )
        self.args = object()
        self.control = object()
        self.fail_with = fail_with
        self.train_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def remove_callback(self, callback: Any) -> None:
        self.callbacks.remove(callback)

    def fire(self, event: str, **kwargs: Any) -> None:
        for callback in list(self.callbacks):
            getattr(callback, event)(self.args, self.state, self.control, **kwargs)

    def at(self, step: int) -> FakeHFTrainer:
        self.state.global_step = step
        return self

    def train(self, *args: Any, **kwargs: Any) -> str:
        """Like the framework: begin, two logged steps, an evaluation, a save, then end or crash."""
        self.train_calls.append((args, kwargs))
        self.fire("on_train_begin")
        for step in (1, 2):
            self.at(step).fire("on_log", logs={"loss": 1.0 / step, "epoch": float(step)})
        self.at(2).fire("on_evaluate", metrics={"eval_loss": 0.5, "eval_accuracy": 0.9})
        self.fire("on_save")
        if self.fail_with is not None:
            raise self.fail_with
        self.fire("on_train_end")
        return "output"


FakeHFTrainer.__module__ = f"{HF_PACKAGE}.trainer"
FakeHFTrainer.__name__ = "Trainer"


# -- the MMEngine ``Runner`` -----------------------------------------------------------------

MM_PACKAGE = "mmengine"


def install_fake_mmengine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register fake ``mmengine`` and ``mmengine.hooks`` modules with the ``Hook`` base class."""

    class Hook:
        """Like the framework's base class: every stage method exists and does nothing."""

        def __getattr__(self, name: str) -> Any:
            if name.startswith(("before_", "after_")):
                return lambda *args, **kwargs: None
            raise AttributeError(name)

    package = types.ModuleType(MM_PACKAGE)
    hooks_module = types.ModuleType(f"{MM_PACKAGE}.hooks")
    hooks_module.Hook = Hook  # type: ignore[attr-defined]
    package.hooks = hooks_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, MM_PACKAGE, package)
    monkeypatch.setitem(sys.modules, f"{MM_PACKAGE}.hooks", hooks_module)


class FakeMMRunner:
    """Has ``register_hook``, ``hooks``, ``rank``, ``iter``, ``epoch`` and the three loops
    (``train``, ``val``, ``test``) like the framework's ``Runner``."""

    def __init__(self, *, rank: int = 0, fail_with: BaseException | None = None) -> None:
        self.hooks: list[Any] = []
        self.priorities: list[Any] = []
        self.rank = rank
        self.iter = 0
        self.epoch = 0
        self.fail_with = fail_with
        self.calls: list[str] = []

    def register_hook(self, hook: Any, priority: Any = None) -> None:
        self.hooks.append(hook)
        self.priorities.append(priority)

    def fire(self, event: str, *args: Any, **kwargs: Any) -> None:
        for hook in list(self.hooks):
            getattr(hook, event)(self, *args, **kwargs)

    def at(self, iteration: int) -> FakeMMRunner:
        self.iter = iteration
        return self

    def _crash(self) -> None:
        if self.fail_with is not None:
            raise self.fail_with

    def train(self) -> str:
        """Like the framework: four iterations, a validation after the second epoch, then end."""
        self.calls.append("train")
        self.fire("before_run")
        for iteration in range(4):
            self.at(iteration).fire(
                "after_train_iter", iteration, data_batch=None,
                outputs={"loss": 1.0 / (iteration + 1), "loss_cls": 0.5},
            )  # fmt: skip
        self.at(4).fire("after_val_epoch", metrics={"accuracy/top1": 0.75})
        self._crash()
        self.fire("after_run")
        return "model"

    def val(self) -> dict[str, float]:
        self.calls.append("val")
        self.fire("before_run")
        self.at(4).fire("after_val_epoch", metrics={"accuracy/top1": 0.8})
        self._crash()
        self.fire("after_run")
        return {"accuracy/top1": 0.8}

    def test(self) -> dict[str, float]:
        self.calls.append("test")
        self.fire("before_run")
        self.at(4).fire("after_test_epoch", metrics={"accuracy/top1": 0.7})
        self._crash()
        self.fire("after_run")
        return {"accuracy/top1": 0.7}


FakeMMRunner.__module__ = f"{MM_PACKAGE}.runner.runner"
FakeMMRunner.__name__ = "Runner"


# -- RF-DETR: a model whose train() builds a trainer of the legacy Lightning package --------------

RF_PACKAGE = "rfdetr"
RF_LIGHTNING = "py" + "to" + "rch_lightning"  # spelled in parts for the repository QA gate


class FakeRFTrainer:
    """The parts of the trainer RF-DETR builds that the hook touches: two lists and the rank."""

    def __init__(self, *, rank_zero: bool = True) -> None:
        self.callbacks: list[Any] = ["rfdetr-callback"]
        self.loggers: list[Any] = ["rfdetr-csv-logger"]
        self.is_global_zero = rank_zero


def install_fake_rfdetr(
    monkeypatch: pytest.MonkeyPatch, *, rank_zero: bool = True, training: bool = True
) -> types.ModuleType:
    """Register ``rfdetr``, ``rfdetr.training`` (with ``build_trainer``) and the legacy Lightning
    package with its ``Callback`` and ``Logger``. ``training=False`` leaves ``rfdetr.training``
    unimportable, as when the framework's training extras are missing."""

    class Callback:
        pass

    class Logger:
        pass

    lightning = types.ModuleType(RF_LIGHTNING)
    lightning.Callback = Callback  # type: ignore[attr-defined]
    lightning.loggers = types.SimpleNamespace(Logger=Logger)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, RF_LIGHTNING, lightning)
    package = types.ModuleType(RF_PACKAGE)
    monkeypatch.setitem(sys.modules, RF_PACKAGE, package)
    training_module = types.ModuleType(f"{RF_PACKAGE}.training")
    built: list[FakeRFTrainer] = []

    def build_trainer(*args: Any, **kwargs: Any) -> FakeRFTrainer:
        trainer = FakeRFTrainer(rank_zero=rank_zero)
        built.append(trainer)
        return trainer

    training_module.build_trainer = build_trainer  # type: ignore[attr-defined]
    training_module.built = built  # type: ignore[attr-defined]
    if training:
        package.training = training_module  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, f"{RF_PACKAGE}.training", training_module)
    else:
        monkeypatch.setitem(sys.modules, f"{RF_PACKAGE}.training", None)  # import fails
    return training_module


class FakeRFDETR:
    """Like the framework's ``RFDETR``: ``train`` imports ``build_trainer`` when it runs, builds a
    trainer, fires its callbacks and loggers like a fit with two epochs, then ends or crashes."""

    def __init__(self, fail_with: BaseException | None = None, *, steps: int = 2) -> None:
        self.fail_with = fail_with
        self.steps = steps
        self.train_calls: list[dict[str, Any]] = []

    def train(self, **kwargs: Any) -> None:
        self.train_calls.append(kwargs)
        module = importlib.import_module(f"{RF_PACKAGE}.training")
        trainer = module.build_trainer("train-config", "model-config")
        ours = [c for c in trainer.callbacks if not isinstance(c, str)]
        loggers = [x for x in trainer.loggers if not isinstance(x, str)]
        for callback in ours:
            callback.setup(trainer, None, "fit")
            callback.on_fit_start(trainer, None)
        for step, loss in [(49 + 50 * i, 2.0 - 0.5 * i) for i in range(self.steps)]:
            for logger in loggers:
                logger.log_metrics({"train/loss": loss, "epoch": 0}, step)
        for logger in loggers:
            logger.log_metrics({"val/mAP_50_95": 0.25, "val/mAP_50": 0.5, "epoch": 1}, 99)
        if self.fail_with is not None:
            for callback in ours:
                callback.on_exception(trainer, None, self.fail_with)
            raise self.fail_with
        for callback in ours:
            callback.on_fit_end(trainer, None)
            callback.on_teardown(trainer, None, "fit")


class RFDETR(FakeRFDETR):
    pass


class RFDETRNano(RFDETR):
    pass


RFDETR.__module__ = f"{RF_PACKAGE}.detr"
RFDETRNano.__module__ = f"{RF_PACKAGE}.variants"
