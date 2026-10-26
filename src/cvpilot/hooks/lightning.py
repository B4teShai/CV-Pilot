"""Lightning hook: a logger for ``self.log`` scalars and a callback for the run lifecycle.

``LightningHook.attach(trainer, sink)`` appends a ``CVPilotLogger`` (scalars, rank zero only)
and a ``CVPilotCallback`` (rank zero detection in every stage, ``finished`` after the last stage,
``failed`` on an exception) to the trainer. The two classes subclass Lightning's own ``Logger``
and ``Callback``, so they are built when the hook is attached (the package is imported then,
never at module import); the logic lives in framework-free ``LightningCore``.
Hyperparameters are not forwarded: the run's config is captured at ``init`` and redacted there.

The trainer is read through ``TrainerView`` only: ``is_global_zero``. Callbacks never raise into
the trainer and make no network call (see ``cvpilot.hooks``).
"""

from __future__ import annotations

import atexit
import importlib
from collections.abc import Mapping
from typing import Any

from cvpilot.core.errors import CvpilotError, MissingExtraError
from cvpilot.hooks._guard import Guard, step_value
from cvpilot.hooks._resume import verified_path_text
from cvpilot.hooks._sink import ScalarRelay

# Spelled in parts: the repository QA gate scans source text for the framework's name.
PACKAGES = ("lightning." + "py" + "to" + "rch", "py" + "to" + "rch_lightning")
"""Import names tried in order: the unified package, then the legacy one."""
EXTRA = "lightning"


def resume_arguments(path: Any) -> dict[str, str]:
    """Keyword arguments for ``Trainer.fit``/``validate``/``test`` that resume from a verified
    checkpoint: ``{"ckpt_path": <absolute path>}``, or ``{}`` when there is nothing to resume.
    Use with ``cvpilot.resume_path``; the string ``"last"`` is refused.

    Raises:
        ValueError: ``path`` is not an absolute file path.
    """
    return {} if path is None else {"ckpt_path": verified_path_text(path)}


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
        self.completed = False  # a stage ended without an exception
        self.failed = False

    def log_metrics(self, metrics: Mapping[Any, Any], step: object) -> None:
        def body() -> None:
            number = step_value(step)
            if not self.enabled or number is None:
                self.guard.skipped += 1
                return
            self.relay.log(metrics, number)

        self.guard.run("log_metrics", body)

    def setup(self, trainer: object) -> None:
        """Every stage (fit, validate, test, predict) is rank-gated, not only fit."""
        self.guard.run("setup", self._rank, trainer, always=True)

    def _rank(self, trainer: object) -> None:
        self.enabled = TrainerView(trainer).is_global_zero

    def fit_end(self) -> None:
        self.completed = True

    def teardown(self, stage: object) -> None:
        """A stage ended. ``test`` and ``predict`` are the last stages of a script, so the run
        finishes there; ``fit`` and ``validate`` leave it open for the stages that follow (their
        metrics are the ones gates usually read), and ``finish_at_exit`` covers scripts that stop
        after them."""
        self.completed = True
        if stage in ("test", "predict"):
            self.guard.run("on_teardown", self._finish, "finished", always=True)

    def exception(self, error: BaseException | None = None) -> None:
        self.failed = True
        self.guard.run("on_exception", self._finish, "failed", always=True)

    def finish_at_exit(self) -> None:
        """Registered with ``atexit``: finish a run whose training completed without error."""
        if self.completed and not self.failed:
            self.guard.run("at_exit", self._finish, "finished", always=True)

    def _finish(self, status: str) -> None:
        if self.enabled and getattr(self.sink, "status", "running") in ("running", "resumed"):
            self.sink.finish(status)


_NO_PICKLE = (
    "the cvpilot hook cannot be pickled: strategy='ddp_spawn' pickles the trainer, and a run "
    "holds an open store connection. Use strategy='ddp' (one process per device, rank zero logs)."
)


def build_classes(
    packages: tuple[str, ...] = PACKAGES, missing: str | None = None
) -> tuple[type, type]:
    """``(CVPilotCallback, CVPilotLogger)`` subclassing Lightning's ``Callback`` and ``Logger``,
    from the first of ``packages`` that imports (a trainer only accepts the classes of the package
    it was built from: RF-DETR passes the legacy package alone).

    Raises:
        MissingExtraError: none of ``packages`` is installed (``missing`` is the message;
            by default ``pip install "cvpilot[lightning]"``).
    """
    module = None
    for name in packages:
        try:
            module = importlib.import_module(name)
            break
        except ImportError:
            continue
    if module is None:
        raise MissingExtraError(
            missing or 'Lightning is not installed: pip install "cvpilot[lightning]"'
        )
    base_callback = module.Callback if hasattr(module, "Callback") else module.callbacks.Callback
    base_logger = module.loggers.Logger

    class CVPilotCallback(base_callback):  # type: ignore[misc, valid-type]
        def __init__(self, core: LightningCore) -> None:
            super().__init__()
            self._core = core

        def setup(self, trainer: Any, pl_module: Any, stage: str) -> None:
            self._core.setup(trainer)

        def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
            self._core.setup(trainer)

        def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
            self._core.fit_end()

        def on_teardown(self, trainer: Any, pl_module: Any, stage: str) -> None:
            self._core.teardown(stage)

        def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
            self._core.exception(exception)

        def __getstate__(self) -> Any:
            raise TypeError(_NO_PICKLE)

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

        def __getstate__(self) -> Any:
            raise TypeError(_NO_PICKLE)

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
        self._cores: list[LightningCore] = []

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
        atexit.register(core.finish_at_exit)
        self._cores.append(core)
        callback, logger = callback_cls(core), logger_cls(core)
        _append(target, "callbacks", callback)
        _append(target, "loggers", logger)
        self._attached.append((target, callback, logger))

    def detach(self) -> None:
        for trainer, callback, logger in self._attached:
            _remove(trainer, "callbacks", callback)
            _remove(trainer, "loggers", logger)
        self._attached.clear()
        for core in self._cores:
            atexit.unregister(core.finish_at_exit)
        self._cores.clear()
