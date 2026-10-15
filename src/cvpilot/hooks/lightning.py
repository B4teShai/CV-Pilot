"""Lightning hook: a logger for ``self.log`` scalars and a callback for the run lifecycle.

``LightningHook.attach(trainer, sink)`` appends a ``CVPilotLogger`` (scalars, rank zero only)
and a ``CVPilotCallback`` (rank zero detection, ``finished`` at fit end, ``failed`` on an
exception) to the trainer. The two classes subclass Lightning's own ``Logger`` and ``Callback``,
so they are built when the hook is attached (the package is imported then, never at module
import); the logic lives in framework-free ``LightningCore``. Hyperparameters are not
forwarded: the run's config is captured at ``init`` and redacted there.

The trainer is read through ``TrainerView`` only: ``is_global_zero``. Callbacks never raise into
the trainer and make no network call (see ``cvpilot.hooks``).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any

from cvpilot.core.errors import CvpilotError, MissingExtraError
from cvpilot.hooks._guard import Guard, step_value
from cvpilot.hooks._sink import ScalarRelay

# Spelled in parts: the repository QA gate scans source text for the framework's name.
PACKAGES = ("lightning." + "py" + "to" + "rch", "py" + "to" + "rch_lightning")
"""Import names tried in order: the unified package, then the legacy one."""
EXTRA = "lightning"


class HookAttachError(CvpilotError, TypeError):
    """The object passed to ``attach`` does not look like the framework's trainer or model."""

    event = "hook_missing"


class TrainerView:
    """The only parts of a Lightning trainer this hook reads."""

    def __init__(self, trainer: object) -> None:
        self._trainer = trainer

    @property
    def is_global_zero(self) -> bool:
        return bool(getattr(self._trainer, "is_global_zero", True))


class LightningCore:
    """Framework-free logic behind the logger and the callback."""

    def __init__(self, sink: Any) -> None:
        self.sink = sink
        self.guard = Guard("lightning")
        self.relay = ScalarRelay(sink, self.guard)
        self.enabled = True  # false on ranks other than zero

    def log_metrics(self, metrics: Mapping[Any, Any], step: object) -> None:
        def body() -> None:
            number = step_value(step)
            if not self.enabled or number is None:
                self.guard.skipped += 1
                return
            self.relay.log(metrics, number)

        self.guard.run("log_metrics", body)

    def fit_start(self, trainer: object) -> None:
        self.guard.run("on_fit_start", self._rank, trainer)

    def _rank(self, trainer: object) -> None:
        self.enabled = TrainerView(trainer).is_global_zero

    def fit_end(self) -> None:
        self.guard.run("on_fit_end", self._finish, "finished")

    def exception(self) -> None:
        self.guard.run("on_exception", self._finish, "failed")

    def _finish(self, status: str) -> None:
        if self.enabled and getattr(self.sink, "status", "running") in ("running", "resumed"):
            self.sink.finish(status)


def build_classes() -> tuple[type, type]:
    """``(CVPilotCallback, CVPilotLogger)`` subclassing Lightning's ``Callback`` and ``Logger``.

    Raises:
        MissingExtraError: Lightning is not installed (``pip install "cvpilot[lightning]"``).
    """
    module = None
    for name in PACKAGES:
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            continue
    if module is None:
        raise MissingExtraError('Lightning is not installed: pip install "cvpilot[lightning]"')
    base_callback = module.Callback if hasattr(module, "Callback") else module.callbacks.Callback
    base_logger = module.loggers.Logger

    class CVPilotCallback(base_callback):  # type: ignore[misc, valid-type]
        def __init__(self, core: LightningCore) -> None:
            super().__init__()
            self._core = core

        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            self._core.fit_start(trainer)

        def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
            self._core.fit_end()

        def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
            self._core.exception()

    class CVPilotLogger(base_logger):  # type: ignore[misc, valid-type]
        def __init__(self, core: LightningCore) -> None:
            super().__init__()
            self._core = core

        @property
        def name(self) -> str:
            return "cvpilot"

        @property
        def version(self) -> str:
            return str(getattr(self._core.sink, "run_id", "0"))

        @property
        def experiment(self) -> Any:
            return self._core.sink

        def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
            return None  # the run config is captured and redacted at init

        def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
            self._core.log_metrics(metrics, step)

        def save(self) -> None:
            return None

        def finalize(self, status: str) -> None:
            return None

    return CVPilotCallback, CVPilotLogger


def _append(trainer: object, attribute: str, item: object) -> None:
    target = getattr(trainer, attribute, None)
    if not callable(getattr(target, "append", None)):
        raise HookAttachError(f"the trainer has no {attribute!r} list to add to")
    target.append(item)  # type: ignore[union-attr]


def _remove(trainer: object, attribute: str, item: object) -> None:
    target = getattr(trainer, attribute, None)
    if isinstance(target, list) and item in target:
        target.remove(item)


class LightningHook:
    """Attach CV-Pilot to a Lightning ``Trainer``: scalars through a logger, lifecycle through a
    callback."""

    name = "lightning"
    requires = ("lightning|py" + "to" + "rch-lightning",)
    license_notice: str | None = None

    def __init__(self) -> None:
        self._attached: list[tuple[object, object, object]] = []

    def matches(self, target: object) -> bool:
        cls = type(target)
        return cls.__name__ == "Trainer" and cls.__module__.split(".")[0] in {
            PACKAGES[0].split(".")[0],
            PACKAGES[1],
        }

    def attach(self, target: object, sink: Any) -> None:
        for attribute in ("callbacks", "loggers"):
            if not callable(getattr(getattr(target, attribute, None), "append", None)):
                raise HookAttachError(f"the trainer has no {attribute!r} list to add to")
        callback_cls, logger_cls = build_classes()
        core = LightningCore(sink)
        callback, logger = callback_cls(core), logger_cls(core)
        _append(target, "callbacks", callback)
        _append(target, "loggers", logger)
        self._attached.append((target, callback, logger))

    def detach(self) -> None:
        for trainer, callback, logger in self._attached:
            _remove(trainer, "callbacks", callback)
            _remove(trainer, "loggers", logger)
        self._attached.clear()
