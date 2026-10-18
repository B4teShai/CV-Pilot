"""MMEngine hook: one ``Hook`` for scalars, validation and test metrics, and the run's end.

``MMEngineHook.attach(runner, sink)`` registers a ``CVPilotHook`` (last in the hook order) with the
runner and wraps its ``train``, ``val`` and ``test`` so that an exception marks the run failed
(an interrupt or exit, preempted) before it propagates unchanged: the framework has no failure
event of its own. The hook subclasses the framework's ``Hook``, so it is built when the hook is
attached (the package is imported then, never at module import); the logic lives in
framework-free ``MMEngineCore``.

What is recorded, always at ``step = runner.iter`` and on rank zero only (``runner.rank``, on top
of the ``RANK`` environment check):

* ``after_train_iter``: the scalars the model's ``train_step`` returned, under ``train/`` (``loss``
  and each named loss), every ``interval`` iterations (default 10, the framework's own logging
  interval: reading a value from an accelerator waits for it).
* ``after_val_epoch`` and ``after_test_epoch``: the evaluation metrics, under ``val/`` and
  ``test/``.
* The run finishes when ``test`` returns (the last stage of a script, as in the Lightning hook);
  a script that stops after ``train`` or ``val`` is finished at exit. ``train`` does not finish
  it, so the ``test`` that usually follows is recorded.

There is no ``resume_arguments`` for this framework, on purpose: its checkpoint files pickle the
logging buffers by class (``HistoryBuffer``), which the strict checkpoint scan refuses, so
CV-Pilot cannot vouch for them. Use ``CheckpointManager`` for weights and optimizer state you want
verified, and the framework's own resume for everything else, knowing that it is not verified.

Every value is converted defensively (``cvpilot.hooks._guard``): NaN and infinity pass because they
are real signals, anything that is not a number is skipped and counted. Callbacks never raise into
the runner and make no network call (see ``cvpilot.hooks``).
"""

from __future__ import annotations

import atexit
import importlib
from collections.abc import Mapping
from typing import Any

from cvpilot.core.errors import MissingExtraError
from cvpilot.hooks._guard import Guard, step_value
from cvpilot.hooks._sink import ScalarRelay
from cvpilot.hooks.lightning import HookAttachError
from cvpilot.settings import is_rank_zero

PACKAGE = "mmengine"
EXTRA = "mmengine"
DEFAULT_INTERVAL = 10
MAX_INTERVAL = 100_000
STAGES = ("train", "val", "test")
RUNNER_CLASSES = ("Runner", "FlexibleRunner")
"""The framework's two runner classes: the second does not derive from the first."""
PRIORITY = "VERY_LOW"
"""The framework's name for the last position in the hook order."""

_NO_PICKLE = (
    "the cvpilot hook cannot be pickled: it holds the run, which owns an open store connection"
)


def check_interval(interval: object) -> int:
    """``interval`` as an int in ``[1, MAX_INTERVAL]``.

    Raises:
        ValueError: anything else (a zero would divide by zero in every callback).
    """
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise ValueError("interval must be an integer number of iterations")
    if not 1 <= interval <= MAX_INTERVAL:
        raise ValueError(f"interval must be between 1 and {MAX_INTERVAL}")
    return interval


class MMEngineCore:
    """Framework-free logic behind the hook."""

    def __init__(self, sink: Any, interval: int = DEFAULT_INTERVAL) -> None:
        interval = check_interval(interval)
        self.sink = sink
        self.interval = interval
        self.guard = Guard(EXTRA)
        self.relay = ScalarRelay(sink, self.guard)
        self.enabled = is_rank_zero()  # false on ranks other than zero, from the first callback
        self.completed = False
        self.failed = False

    def _active(self) -> bool:
        return self.enabled and getattr(self.sink, "status", "running") in ("running", "resumed")

    def before_run(self, runner: object) -> None:
        self.guard.run("before_run", self._rank, runner, always=True)

    def _rank(self, runner: object) -> None:
        rank = getattr(runner, "rank", 0)
        self.enabled = is_rank_zero() and not isinstance(rank, bool) and rank == 0

    def _record(self, raw: object, runner: object, prefix: str) -> None:
        step = step_value(getattr(runner, "iter", None))
        if not self._active() or step is None or not isinstance(raw, Mapping):
            self.guard.skipped += 1
            return
        self.relay.log(raw, step, prefix)

    def train_iter(self, runner: object, outputs: object) -> None:
        self.guard.run("after_train_iter", self._train_iter, runner, outputs)

    def _train_iter(self, runner: object, outputs: object) -> None:
        step = step_value(getattr(runner, "iter", None))
        if step is not None and (step + 1) % self.interval != 0:
            return  # not an iteration to record: neither a value read nor a skip counted
        self._record(outputs, runner, "train/")

    def val_epoch(self, runner: object, metrics: object) -> None:
        self.guard.run("after_val_epoch", self._record, metrics, runner, "val/")

    def test_epoch(self, runner: object, metrics: object) -> None:
        self.guard.run("after_test_epoch", self._record, metrics, runner, "test/")

    def stage_done(self, stage: str, runner: object = None) -> None:
        """A stage returned. ``test`` is the last stage of a script, so the run finishes there;
        ``train`` and ``val`` leave it open for the stages that follow, and ``finish_at_exit``
        covers scripts that stop after them."""
        self.completed = True
        if stage == "test":
            self.guard.run("stage_done", self._ranked_finish, "finished", runner, always=True)

    def _ranked_finish(self, status: str, runner: object) -> None:
        if runner is not None:
            self._rank(runner)  # the rank gate also holds when no hook callback ran yet
        self._finish(status)

    def stage_failed(self, runner: object = None) -> None:
        self.failed = True
        self.guard.run("stage_failed", self._ranked_finish, "failed", runner, always=True)

    def stage_interrupted(self, runner: object = None) -> None:
        """``KeyboardInterrupt`` or ``SystemExit`` ended a stage: stopped from outside (a signal, a
        scheduler, a person), which is ``preempted``, not a failure of the training."""
        self.failed = True
        self.guard.run("stage_interrupted", self._ranked_finish, "preempted", runner, always=True)

    def _finish(self, status: str) -> None:
        if self.enabled and getattr(self.sink, "status", "running") in ("running", "resumed"):
            self.sink.finish(status)

    def finish_at_exit(self) -> None:
        """Registered with ``atexit``: finish a run whose stages completed without error."""
        if self.completed and not self.failed:
            self.guard.run("at_exit", self._finish, "finished", always=True)


def build_hook_class() -> type:
    """``CVPilotHook``, subclassing the framework's ``Hook``.

    Raises:
        MissingExtraError: ``mmengine`` is not installed (``pip install "cvpilot[mmengine]"``).
    """
    try:
        base = importlib.import_module(f"{PACKAGE}.hooks").Hook
    except (ImportError, AttributeError):
        raise MissingExtraError(
            'the MMEngine package is not installed: pip install "cvpilot[mmengine]"'
        ) from None

    class CVPilotHook(base):  # type: ignore[misc, valid-type]
        def __init__(self, core: MMEngineCore) -> None:
            super().__init__()
            self._core = core

        def before_run(self, runner: Any) -> None:
            self._core.before_run(runner)

        def after_train_iter(
            self, runner: Any, batch_idx: int, data_batch: Any = None, outputs: Any = None
        ) -> None:
            self._core.train_iter(runner, outputs)

        def after_val_epoch(self, runner: Any, metrics: Any = None) -> None:
            self._core.val_epoch(runner, metrics)

        def after_test_epoch(self, runner: Any, metrics: Any = None) -> None:
            self._core.test_epoch(runner, metrics)

        def __getstate__(self) -> Any:
            raise TypeError(_NO_PICKLE)

    return CVPilotHook


class MMEngineHook:
    """Attach CV-Pilot to an MMEngine ``Runner`` (or a subclass of it) through its hook list.

    ``interval`` is how many training iterations pass between recorded values (default 10).
    ``cvpilot.autolog`` builds the hook with the default; pass another interval by constructing
    ``MMEngineHook(interval=...)`` and calling ``attach(runner, run)`` yourself.
    """

    name = EXTRA
    requires = (PACKAGE,)
    license_notice: str | None = None

    def __init__(self, interval: int = DEFAULT_INTERVAL) -> None:
        self.interval = check_interval(interval)
        self._attached: list[tuple[object, object]] = []
        self._wrapped: list[tuple[object, str, Any, bool]] = []
        self._cores: list[MMEngineCore] = []

    def matches(self, target: object) -> bool:
        """A ``Runner`` of the framework, or an instance of a subclass of one (a subclass in
        another module counts when a class in its ancestry is the framework's ``Runner``)."""
        return callable(getattr(target, "register_hook", None)) and any(
            cls.__name__ in RUNNER_CLASSES and cls.__module__.split(".")[0] == PACKAGE
            for cls in type(target).__mro__
        )

    def attach(self, target: object, sink: Any) -> None:
        register = getattr(target, "register_hook", None)
        if not callable(register):
            raise HookAttachError("the runner has no register_hook method")
        core = MMEngineCore(sink, self.interval)
        hook = build_hook_class()(core)
        register(hook, priority=PRIORITY)
        atexit.register(core.finish_at_exit)
        self._cores.append(core)
        self._attached.append((target, hook))
        for stage in STAGES:
            self._wrap_stage(target, stage, core)

    def _wrap_stage(self, target: object, stage: str, core: MMEngineCore) -> None:
        """The framework has no failure event, so each stage is wrapped: an exception (or an
        interrupt) marks the run failed (or preempted), then propagates unchanged."""
        original = getattr(target, stage, None)
        if not callable(original):
            return
        had_own = stage in vars(target)

        def run_stage(*args: Any, **kwargs: Any) -> Any:
            try:
                result = original(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                core.stage_interrupted(target)  # guarded: it can never mask the exception
                raise
            except BaseException:
                core.stage_failed(target)  # guarded: it can never mask the exception being raised
                raise
            core.stage_done(stage, target)
            return result

        setattr(target, stage, run_stage)
        self._wrapped.append((target, stage, original, had_own))

    def detach(self) -> None:
        for runner, hook in self._attached:
            registered = getattr(runner, "hooks", None)
            if isinstance(registered, list) and hook in registered:
                registered.remove(hook)
        self._attached.clear()
        for target, stage, original, had_own in reversed(self._wrapped):  # last wrapper off first
            if had_own:
                setattr(target, stage, original)
            else:
                vars(target).pop(stage, None)
        self._wrapped.clear()
        for core in self._cores:
            atexit.unregister(core.finish_at_exit)
        self._cores.clear()
