"""Hugging Face ``Trainer`` hook: one ``TrainerCallback`` for scalars, evaluations, saves, the end.

``HFHook.attach(trainer, sink)`` adds a ``CVPilotCallback`` to the trainer and wraps its ``train``
so that an exception marks the run failed (an interrupt or exit, preempted) before it propagates
unchanged: the framework has no failure event of its own. The callback subclasses the framework's
``TrainerCallback``, so it is built when the hook is attached (the package is imported then, never
at module import); the logic lives in framework-free ``HFCore``.

What is recorded, always at ``step = state.global_step`` and on rank zero only
(``state.is_world_process_zero``):

* ``on_log``: the scalars of the training log under ``train/`` (``loss``, ``learning_rate``,
  ``grad_norm``, ``epoch``, ...). Keys that start like an evaluation metric (``eval_``, ``val_``,
  ``valid_``, ``validation_``, ``test_``: the framework's default prefix and the usual custom
  ones) are left to ``on_evaluate``: the framework logs them through ``on_log`` too, and one copy
  is enough.
* ``on_evaluate``: the evaluation metrics under ``eval/`` (``eval_loss`` becomes ``eval/loss``).
* ``on_save``: one ``hook_checkpoint_saved`` event with the step and nothing else. The
  checkpoint directory is not hashed or adopted: see "What is not built" in ADR-0009.
* ``on_train_end``: finishes the run. An exception inside ``train`` marks it ``failed``;
  ``KeyboardInterrupt`` and ``SystemExit`` mark it ``preempted``. Metrics reported after that
  (a manual ``trainer.evaluate()`` after ``train()``) are skipped and counted, not errors.

Every value is converted defensively (``cvpilot.hooks._guard``): NaN and infinity pass because
they are real signals, anything that is not a number is skipped and counted. Callbacks never raise
into the trainer and make no network call (see ``cvpilot.hooks``).

There is no ``resume_arguments`` for this framework, on purpose. The trainer resumes from a
*directory* of files that includes pickled optimizer and random-number state, and CV-Pilot's
verification covers single files (a size, a SHA-256 and a format), not a directory bundle.
Restore a ``CheckpointManager`` checkpoint into the model with ``CheckpointManager.restore`` and
pass the weights to the trainer yourself.
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
from cvpilot.logging import get_logger
from cvpilot.settings import is_rank_zero

PACKAGE = "transformers"
EXTRA = "hf"
EVAL_PREFIX = "eval_"
EVAL_LIKE = ("eval_", "val_", "valid_", "validation_", "test_")
"""Key prefixes of evaluation metrics (``Trainer.evaluate(metric_key_prefix=...)`` defaults to
``eval``): such keys are recorded from ``on_evaluate`` only, never also from ``on_log``."""

_NO_PICKLE = (
    "the cvpilot hook cannot be pickled: it holds the run, which owns an open store connection"
)


class HFCore:
    """Framework-free logic behind the callback."""

    def __init__(self, sink: Any) -> None:
        self.sink = sink
        self.guard = Guard(EXTRA)
        self.relay = ScalarRelay(sink, self.guard)
        self.enabled = is_rank_zero()  # false on ranks other than zero, from the first callback
        self.completed = False
        self.failed = False

    # -- helpers -----------------------------------------------------------------------------

    def _active(self) -> bool:
        return self.enabled and getattr(self.sink, "status", "running") in ("running", "resumed")

    def _record(self, raw: object, state: object, prefix: str, evaluation: bool) -> None:
        step = step_value(getattr(state, "global_step", None))
        if not self._active() or step is None or not isinstance(raw, Mapping):
            self.guard.skipped += 1
            return
        picked: dict[Any, Any] = {}
        for key, value in raw.items():
            if isinstance(key, str):
                if not key.strip():
                    self.guard.skipped += 1
                    continue
                if key.startswith(EVAL_LIKE):
                    if not evaluation:
                        continue  # evaluation values reach on_evaluate; one copy is enough
                    if key.startswith(EVAL_PREFIX):
                        key = key[len(EVAL_PREFIX) :]
            picked[key] = value
        self.relay.log(picked, step, prefix)

    # -- callbacks ---------------------------------------------------------------------------

    def train_begin(self, state: object) -> None:
        """Rank gate: the trainer's own view (``state.is_world_process_zero``) is read when
        training starts, on top of the ``RANK`` check made when the hook was created."""
        self.guard.run("on_train_begin", self._rank, state, always=True)

    def _rank(self, state: object) -> None:
        self.enabled = bool(getattr(state, "is_world_process_zero", True))

    def log(self, state: object, logs: object) -> None:
        self.guard.run("on_log", self._record, logs, state, "train/", False)

    def evaluate(self, state: object, metrics: object) -> None:
        self.guard.run("on_evaluate", self._record, metrics, state, "eval/", True)

    def save(self, state: object) -> None:
        self.guard.run("on_save", self._saved, state)

    def _saved(self, state: object) -> None:
        step = step_value(getattr(state, "global_step", None))
        if self._active() and step is not None:
            get_logger("cvpilot.hooks").info("hook_checkpoint_saved", hook=EXTRA, step=step)

    def train_end(self) -> None:
        self.completed = True
        self.guard.run("on_train_end", self._finish, "finished", always=True)

    def train_failed(self) -> None:
        self.failed = True
        self.guard.run("train_failed", self._finish, "failed", always=True)

    def train_interrupted(self) -> None:
        """``KeyboardInterrupt`` or ``SystemExit`` ended ``train``: stopped from outside (a
        signal, a scheduler, a person), which is ``preempted``, not a failure of the training."""
        self.failed = True
        self.guard.run("train_interrupted", self._finish, "preempted", always=True)

    def _finish(self, status: str) -> None:
        if self.enabled and getattr(self.sink, "status", "running") in ("running", "resumed"):
            self.sink.finish(status)

    def finish_at_exit(self) -> None:
        """Registered with ``atexit``: finish a run whose training ended without ``on_train_end``
        reaching the sink (the hook was disabled by repeated failures)."""
        if self.completed and not self.failed:
            self.guard.run("at_exit", self._finish, "finished", always=True)


def build_callback_class() -> type:
    """``CVPilotCallback``, subclassing the framework's ``TrainerCallback``.

    Raises:
        MissingExtraError: ``transformers`` is not installed (``pip install "cvpilot[hf]"``).
    """
    try:
        base = importlib.import_module(PACKAGE).TrainerCallback
    except (ImportError, AttributeError):
        raise MissingExtraError(
            'the Hugging Face transformers package is not installed: pip install "cvpilot[hf]"'
        ) from None

    class CVPilotCallback(base):  # type: ignore[misc, valid-type]
        def __init__(self, core: HFCore) -> None:
            super().__init__()
            self._core = core

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._core.train_begin(state)

        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            self._core.log(state, logs)

        def on_evaluate(
            self, args: Any, state: Any, control: Any, metrics: Any = None, **kwargs: Any
        ) -> None:
            self._core.evaluate(state, metrics)

        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._core.save(state)

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            self._core.train_end()

        def __getstate__(self) -> Any:
            raise TypeError(_NO_PICKLE)

    return CVPilotCallback


class HFHook:
    """Attach CV-Pilot to a Hugging Face ``Trainer`` (or a subclass of it) through its callback
    list."""

    name = EXTRA
    requires = (PACKAGE,)
    license_notice: str | None = None

    def __init__(self) -> None:
        self._attached: list[tuple[object, object]] = []
        self._wrapped: list[tuple[object, Any, bool]] = []
        self._cores: list[HFCore] = []

    def matches(self, target: object) -> bool:
        """A ``Trainer`` of the framework, or an instance of a subclass of one (a subclass in
        another module counts when a class in its ancestry is the framework's ``Trainer``)."""
        return callable(getattr(target, "add_callback", None)) and any(
            cls.__name__ == "Trainer" and cls.__module__.split(".")[0] == PACKAGE
            for cls in type(target).__mro__
        )

    def attach(self, target: object, sink: Any) -> None:
        add = getattr(target, "add_callback", None)
        if not callable(add) or not callable(getattr(target, "remove_callback", None)):
            raise HookAttachError("the trainer has no add_callback and remove_callback methods")
        core = HFCore(sink)
        callback = build_callback_class()(core)
        add(callback)
        atexit.register(core.finish_at_exit)
        self._cores.append(core)
        self._attached.append((target, callback))
        self._wrap_train(target, core)

    def _wrap_train(self, target: object, core: HFCore) -> None:
        """The framework has no failure event, so ``train`` is wrapped: an exception (or an
        interrupt) marks the run failed, then propagates unchanged."""
        original = getattr(target, "train", None)
        if not callable(original):
            return
        had_own = "train" in vars(target)

        def train(*args: Any, **kwargs: Any) -> Any:
            try:
                return original(*args, **kwargs)
            except (KeyboardInterrupt, SystemExit):
                core.train_interrupted()  # guarded: it can never mask the exception
                raise
            except BaseException:
                core.train_failed()  # guarded: it can never mask the exception being raised
                raise

        target.train = train  # type: ignore[attr-defined]
        self._wrapped.append((target, original, had_own))

    def detach(self) -> None:
        for trainer, callback in self._attached:
            remove = getattr(trainer, "remove_callback", None)
            if callable(remove):
                try:
                    remove(callback)
                except ValueError:
                    pass  # already removed by the user
        self._attached.clear()
        for target, original, had_own in self._wrapped:
            if had_own:
                target.train = original  # type: ignore[attr-defined]
            else:
                vars(target).pop("train", None)
        self._wrapped.clear()
        for core in self._cores:
            atexit.unregister(core.finish_at_exit)
        self._cores.clear()
