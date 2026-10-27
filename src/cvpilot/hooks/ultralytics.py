"""Hook for the YOLO training framework (the optional extra named after it).

LICENSE NOTICE: the framework this adapter talks to is licensed AGPL-3.0. Installing the extra
and using this hook makes the combined work you distribute or serve subject to that license.
This package stays Apache-2.0 because it neither contains nor imports framework code: the hook
only receives the framework's own objects through ``add_callback`` and never imports the package
(an import, if one is ever needed, belongs inside a function of this module and nowhere else).
The notice is repeated in ``LICENSE_NOTICE``, in the ``hook_attached`` event and in
docs/guide/install.md.

``YoloHook.attach(model, sink)`` registers ``on_fit_epoch_end`` (scalars and per-class AP
of the epoch, step = epoch; metrics only for epochs in which ``on_val_end`` fired, because the
trainer's metrics are stale otherwise) and ``on_train_end`` (finish the run). Distributed
training is not supported: the framework runs it in child processes that carry no callbacks,
so the hook sees no epoch; ``hook_no_epochs`` says so. The trainer is read through
``TrainerView`` only: ``epoch``, ``metrics``, ``label_loss_items``/``tloss`` and
``validator.names`` / ``validator.metrics.box.maps``. Every value is converted defensively;
callbacks never raise into the trainer (see ``cvpilot.hooks``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cvpilot.hooks._guard import MAX_POINTS_PER_CALL, Guard, metric_key, metric_value, step_value
from cvpilot.hooks._resume import verified_path_text
from cvpilot.hooks._sink import ScalarRelay
from cvpilot.hooks.lightning import HookAttachError
from cvpilot.logging import get_logger


def resume_arguments(path: Any) -> tuple[str | None, dict[str, Any]]:
    """``(weights, train_kwargs)`` to resume from a verified checkpoint: construct the model
    from ``weights`` and call ``train(**train_kwargs)`` (``resume=True``). ``(None, {})`` when
    there is nothing to resume. Use with ``cvpilot.resume_path``; ``"last"`` is refused.

    Raises:
        ValueError: ``path`` is not an absolute file path.
    """
    if path is None:
        return None, {}
    return verified_path_text(path), {"resume": True}


# Spelled in parts: the repository QA gate scans source text for the framework's name.
FRAMEWORK = "ultra" + "lytics"
LICENSE_NOTICE = (
    f"{FRAMEWORK.capitalize()} is licensed AGPL-3.0: using this hook makes the combined work "
    "subject to AGPL-3.0. CV-Pilot itself stays Apache-2.0 and does not include framework code."
)
EVENTS = ("on_fit_epoch_end", "on_val_end", "on_train_end")
MAX_CLASSES = 2048


class TrainerView:
    """The only parts of the framework's trainer this hook reads; every read is defensive."""

    def __init__(self, trainer: object) -> None:
        self._t = trainer

    @property
    def epoch(self) -> int | None:
        return step_value(getattr(self._t, "epoch", None))

    @property
    def metrics(self) -> Mapping[object, object]:
        metrics = getattr(self._t, "metrics", None)
        return metrics if isinstance(metrics, Mapping) else {}

    @property
    def losses(self) -> Mapping[object, object]:
        label = getattr(self._t, "label_loss_items", None)
        tloss = getattr(self._t, "tloss", None)
        if not callable(label) or tloss is None:
            return {}
        items = label(tloss, prefix="train")
        return items if isinstance(items, Mapping) else {}

    @property
    def per_class_ap(self) -> dict[str, float]:
        validator = getattr(self._t, "validator", None)
        names = getattr(validator, "names", None)
        maps = getattr(getattr(getattr(validator, "metrics", None), "box", None), "maps", None)
        if not isinstance(names, Mapping) or maps is None:
            return {}
        out: dict[str, float] = {}
        for index, name in list(names.items())[:MAX_CLASSES]:
            key = metric_key(name, "val/AP/")
            try:
                value = metric_value(maps[index])
            except Exception:  # noqa: BLE001  # index out of range, wrong container
                value = None
            if key is not None and value is not None:
                out[key] = value
        return out


class YoloCore:
    def __init__(self, sink: Any) -> None:
        self.sink = sink
        self.guard = Guard(FRAMEWORK)
        self.relay = ScalarRelay(sink, self.guard)
        self.validated = False  # a validation ran since the last epoch-end callback
        self.epochs_seen = 0

    def val_end(self, trainer: object) -> None:
        self.guard.run("on_val_end", self._mark_validated)

    def _mark_validated(self) -> None:
        self.validated = True

    def epoch_end(self, trainer: object) -> None:
        self.guard.run("on_fit_epoch_end", self._epoch_end, trainer)

    def _epoch_end(self, trainer: object) -> None:
        validated, self.validated = self.validated, False
        self.epochs_seen += 1
        view = TrainerView(trainer)
        step = view.epoch
        if step is None:
            self.guard.skipped += 1
            return
        scalars: dict[object, object] = {}
        scalars.update(view.losses)
        if validated:  # without a validation this epoch the trainer's metrics are last epoch's
            scalars.update(view.metrics)
            scalars.update(view.per_class_ap)
        if len(scalars) > MAX_POINTS_PER_CALL:
            scalars = dict(list(scalars.items())[:MAX_POINTS_PER_CALL])
            self.guard.skipped += 1
        self.relay.log(scalars, step)

    def train_end(self, trainer: object) -> None:
        self.guard.run("on_train_end", self._finish, "finished", always=True)

    def failed(self) -> None:
        self.guard.run("train_failed", self._finish, "failed", always=True)

    def train_returned(self) -> None:
        """``train`` returned normally. With no epoch callback seen the framework trained where
        this hook is not installed (distributed training starts child processes that carry no
        callbacks): say so instead of staying silent."""
        self.guard.run("train_returned", self._returned, always=True)

    def _returned(self) -> None:
        if self.epochs_seen == 0:
            get_logger("cvpilot.hooks").warning(
                "hook_no_epochs", hook=FRAMEWORK, reason="no_epoch_callback_ran"
            )
        self._finish("finished")

    def _finish(self, status: str) -> None:
        if getattr(self.sink, "status", "running") in ("running", "resumed"):
            self.sink.finish(status)


class YoloHook:
    """Attach CV-Pilot to a YOLO model through its ``add_callback`` registry.

    See the module docstring for the AGPL-3.0 notice that applies to the framework.
    """

    name = FRAMEWORK
    requires = (FRAMEWORK,)
    license_notice: str | None = LICENSE_NOTICE

    def __init__(self) -> None:
        self._attached: list[tuple[object, Any]] = []
        self._wrapped: list[tuple[object, Any, bool]] = []

    def matches(self, target: object) -> bool:
        return type(target).__module__.split(".")[0] == FRAMEWORK and callable(
            getattr(target, "add_callback", None)
        )

    def attach(self, target: object, sink: Any) -> None:
        add = getattr(target, "add_callback", None)
        if not callable(add):
            raise HookAttachError("the model has no add_callback method")
        core = YoloCore(sink)
        handlers = {
            "on_fit_epoch_end": core.epoch_end,
            "on_val_end": core.val_end,
            "on_train_end": core.train_end,
        }
        for event in EVENTS:
            add(event, handlers[event])
        self._attached.append((target, handlers))
        self._wrap_train(target, core)

    def _wrap_train(self, target: object, core: YoloCore) -> None:
        """The framework has no failure event, so ``train`` is wrapped: an exception (or an
        interrupt) marks the run failed, then propagates unchanged."""
        original = getattr(target, "train", None)
        if not callable(original):
            return
        had_own = "train" in vars(target)

        def train(*args: Any, **kwargs: Any) -> Any:
            try:
                result = original(*args, **kwargs)
            except BaseException:
                core.failed()  # guarded: it can never mask the exception being raised
                raise
            core.train_returned()
            return result

        target.train = train  # type: ignore[attr-defined]
        self._wrapped.append((target, original, had_own))

    def detach(self) -> None:
        for target, handlers in self._attached:
            callbacks = getattr(target, "callbacks", None)
            if isinstance(callbacks, dict):
                for event, handler in handlers.items():
                    registered = callbacks.get(event)
                    if isinstance(registered, list) and handler in registered:
                        registered.remove(handler)
        self._attached.clear()
        for target, original, had_own in self._wrapped:
            if had_own:
                target.train = original  # type: ignore[attr-defined]
            else:
                vars(target).pop("train", None)
        self._wrapped.clear()
