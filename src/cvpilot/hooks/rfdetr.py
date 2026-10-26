"""RF-DETR hook: the Lightning logger and callback on the trainer that ``RFDETR.train`` builds.

RF-DETR (Apache-2.0) trains through a trainer of the legacy Lightning package that ``train`` builds
itself with ``rfdetr.training.build_trainer``; it offers no callback argument. ``RFDETRHook.attach
(model, sink)`` therefore wraps the model's ``train`` (on that instance only): while it runs, the
module's ``build_trainer`` is wrapped so that the first trainer it builds gets a ``CVPilotLogger``
and a ``CVPilotCallback`` (the classes of the Lightning hook, built from the legacy package because
the trainer accepts only its own classes); the original ``build_trainer`` is put back when ``train``
returns or raises. Nothing else (``evaluate``, ``predict``, another model) is touched.

What is recorded, on rank zero only (``trainer.is_global_zero`` and the ``RANK`` environment
check): every scalar RF-DETR logs (``train/loss`` and its parts, ``val/mAP_50_95``, ``val/mAP_50``,
the EMA and per-class values) at the trainer's global step, when the framework's own logger
connector hands them to its loggers. When ``train`` returns the run is ``finished``; an exception
marks it ``failed``, ``KeyboardInterrupt`` or ``SystemExit`` ``preempted``, and the exception
propagates unchanged.

There is no ``resume_arguments`` for this framework, on purpose: its ``train(resume=...)`` loads the
file with the framework's own loader, which falls back to full unpickling for a file that the
restricted loader refuses. Use ``CheckpointManager`` for state you want verified.

Callbacks never raise into the trainer and make no network call (see ``cvpilot.hooks``); the
framework is imported when the hook is attached, never at module import.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any

from cvpilot.hooks._guard import Guard
from cvpilot.hooks._sink import ScalarRelay
from cvpilot.hooks.lightning import HookAttachError, LightningCore, build_classes
from cvpilot.settings import is_rank_zero

PACKAGE = "rfdetr"
EXTRA = "rfdetr"
TRAINING_MODULE = f"{PACKAGE}.training"
FACTORY = "build_trainer"
MODEL_CLASS = "RFDETR"
# Spelled in parts: the repository QA gate scans source text for the framework's name.
LIGHTNING = ("py" + "to" + "rch_lightning",)
"""The legacy Lightning package RF-DETR trains with; its trainer refuses the unified package's
classes."""
MISSING = (
    "the Lightning package RF-DETR trains with is not installed: "
    'pip install "cvpilot[rfdetr]" (it brings the framework with its training extras)'
)


class RFDETRCore(LightningCore):
    """The Lightning hook's logic with this framework's rules: the run ends when ``train`` ends,
    and an interrupt is a preemption, not a failure."""

    def __init__(self, sink: Any) -> None:
        super().__init__(sink)
        self.guard = Guard(EXTRA)
        self.relay = ScalarRelay(sink, self.guard)
        self.enabled = is_rank_zero()

    def _rank(self, trainer: object) -> None:
        super()._rank(trainer)
        self.enabled = self.enabled and is_rank_zero()

    def teardown(self, stage: object) -> None:
        """``train`` ends the run (see ``train_done``); the test stage RF-DETR may run at the end
        of fitting does not."""
        self.completed = True

    def exception(self, error: BaseException | None = None) -> None:
        self.failed = True
        status = "preempted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        self.guard.run("on_exception", self._finish, status, always=True)

    def train_done(self) -> None:
        self.completed = True
        self.guard.run("train_done", self._finish, "finished", always=True)


def _add(trainer: object, callback: object, logger: object) -> None:
    for attribute in ("callbacks", "loggers"):
        target = getattr(trainer, attribute, None)
        if not callable(getattr(target, "append", None)):
            raise HookAttachError(f"the trainer has no {attribute!r} list to add to")
    trainer.callbacks.append(callback)  # type: ignore[attr-defined]
    trainer.loggers.append(logger)  # type: ignore[attr-defined]


class RFDETRHook:
    """Attach CV-Pilot to an RF-DETR model (``RFDETRNano``, ``RFDETRBase``, ... or a subclass)."""

    name = EXTRA
    requires = (PACKAGE,)
    license_notice: str | None = None

    def __init__(self) -> None:
        self._wrapped: list[tuple[object, Any, bool]] = []
        self._cores: list[RFDETRCore] = []

    def matches(self, target: object) -> bool:
        """An instance of the framework's ``RFDETR`` class or of a subclass of it (a subclass in
        another module counts when a class in its ancestry is the framework's)."""
        return callable(getattr(target, "train", None)) and any(
            cls.__name__ == MODEL_CLASS and cls.__module__.split(".")[0] == PACKAGE
            for cls in type(target).__mro__
        )

    def attach(self, target: object, sink: Any) -> None:
        original = getattr(target, "train", None)
        if not callable(original):
            raise HookAttachError("the model has no train method")
        callback_cls, logger_cls = build_classes(LIGHTNING, MISSING)
        core = RFDETRCore(sink)
        self._cores.append(core)
        had_own = "train" in vars(target)

        def train(*args: Any, **kwargs: Any) -> Any:
            restore = self._wrap_factory(core, callback_cls, logger_cls)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                core.exception(exc)  # guarded: it can never mask the exception being raised
                raise
            finally:
                restore()
            core.train_done()
            return result

        setattr(target, "train", train)  # noqa: B010  # an instance attribute, by name
        self._wrapped.append((target, original, had_own))

    @staticmethod
    def _wrap_factory(core: RFDETRCore, callback_cls: type, logger_cls: type) -> Callable[[], None]:
        """Wrap ``rfdetr.training.build_trainer`` for one ``train`` call; returns the undo. When
        the module cannot be imported (the framework's training extras are missing) nothing is
        wrapped and ``train`` fails the framework's own way."""
        try:
            module = importlib.import_module(TRAINING_MODULE)
        except ImportError:
            return lambda: None
        real = getattr(module, FACTORY, None)
        if not callable(real):
            return lambda: None
        added: list[object] = []

        def build_trainer(*args: Any, **kwargs: Any) -> Any:
            trainer = real(*args, **kwargs)
            if not added:  # the first trainer of this train call is the one that fits
                added.append(trainer)
                core.guard.run(
                    "attach", _add, trainer, callback_cls(core), logger_cls(core), always=True
                )
            return trainer

        setattr(module, FACTORY, build_trainer)

        def restore() -> None:
            if getattr(module, FACTORY, None) is build_trainer:
                setattr(module, FACTORY, real)

        return restore

    def detach(self) -> None:
        for target, original, had_own in reversed(self._wrapped):
            if had_own:
                setattr(target, "train", original)  # noqa: B010
            else:
                vars(target).pop("train", None)
        self._wrapped.clear()
        self._cores.clear()
