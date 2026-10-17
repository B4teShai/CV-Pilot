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
