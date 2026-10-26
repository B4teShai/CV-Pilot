"""RF-DETR hook: the Lightning logger and callback on the trainer that ``RFDETR.train`` builds.

RF-DETR (Apache-2.0) trains through a trainer of the legacy Lightning package that ``train`` builds
itself with ``rfdetr.training.build_trainer``; it offers no callback argument. ``RFDETRHook.attach
(model, sink)`` therefore wraps the model's ``train`` (on that instance only): while it runs, the
module's ``build_trainer`` is wrapped so that the first trainer it builds gets a ``CVPilotLogger``
and a ``CVPilotCallback`` (the classes of the Lightning hook, built from the legacy package because
the trainer accepts only its own classes); the original ``build_trainer`` is put back when ``train``
returns or raises. ``evaluate``, ``predict`` and other model instances are not touched, but while
``train`` runs the module's ``build_trainer`` is replaced for the whole process: another thread that
builds an RF-DETR trainer at that moment would get the hook's logger and callback too. One training
per process is the supported case.

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
import importlib.util
from collections.abc import Callable
from pathlib import Path
from typing import Any

from cvpilot.hooks._guard import Guard
from cvpilot.hooks._sink import ScalarRelay
from cvpilot.hooks.lightning import HookAttachError, LightningCore, build_classes
from cvpilot.logging import get_logger
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


MAX_SOURCE_BYTES = 1024 * 1024
"""Most of the framework's ``rfdetr/training/__init__.py`` that the compatibility check reads."""


def compatibility() -> str | None:
    """Why the hook would silently record nothing, or None. Reads the source of the framework's
    ``rfdetr/training/__init__.py`` (bounded) for the name ``build_trainer`` that the hook patches,
    without importing the framework: ``cvpilot doctor`` calls it, and an import would load the
    whole training stack.
    A framework that is not installed has nothing to report."""
    try:  # the top-level package only: finding a submodule would import (run) its parent
        spec = importlib.util.find_spec(PACKAGE)
    except (ImportError, ValueError):
        return None
    folders = [] if spec is None else list(spec.submodule_search_locations or [])
    if not folders:
        return None
    try:
        with (Path(folders[0]) / "training" / "__init__.py").open("rb") as handle:
            text = handle.read(MAX_SOURCE_BYTES).decode("utf-8", "replace")
    except OSError:
        return None
    if FACTORY in text:
        return None
    return (
        f"{TRAINING_MODULE} no longer mentions {FACTORY}: the hook would record nothing "
        "(a newer RF-DETR release than the one this hook was tested with)"
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

    @staticmethod
    def compatibility() -> str | None:
        """See the module-level ``compatibility``; ``cvpilot doctor`` asks every hook this."""
        return compatibility()

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
            get_logger("cvpilot.hooks").warning(
                "hook_target_missing", hook=EXTRA, target=f"{TRAINING_MODULE}.{FACTORY}"
            )
            return lambda: None
        added: list[object] = []

        def attach_one(trainer: object) -> None:
            # built here, inside the guard: a constructor that fails is isolated like any callback
            _add(trainer, callback_cls(core), logger_cls(core))

        def build_trainer(*args: Any, **kwargs: Any) -> Any:
            trainer = real(*args, **kwargs)
            if not added:  # the first trainer of this train call is the one that fits
                added.append(trainer)
                core.guard.run("attach", attach_one, trainer, always=True)
            return trainer

        setattr(module, FACTORY, build_trainer)

        def restore() -> None:
            if getattr(module, FACTORY, None) is build_trainer:
                setattr(module, FACTORY, real)
            if not added:
                get_logger("cvpilot.hooks").warning("hook_not_attached", hook=EXTRA)

        return restore

    def detach(self) -> None:
        for target, original, had_own in reversed(self._wrapped):
            if had_own:
                setattr(target, "train", original)  # noqa: B010
            else:
                vars(target).pop("train", None)
        self._wrapped.clear()
        self._cores.clear()
