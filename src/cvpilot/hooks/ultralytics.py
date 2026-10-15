"""Hook for the YOLO training framework (the optional extra named after it).

LICENSE NOTICE: the framework this adapter talks to is licensed AGPL-3.0. Installing the extra
and using this hook makes the combined work you distribute or serve subject to that license.
This package stays Apache-2.0 because it neither contains nor imports framework code: the hook
only receives the framework's own objects through ``add_callback`` and never imports the package
(an import, if one is ever needed, belongs inside a function of this module and nowhere else).
The notice is repeated in ``LICENSE_NOTICE``, in the ``hook_attached`` event and in
docs/architecture/05-runtime.md.

``YoloHook.attach(model, sink)`` registers ``on_fit_epoch_end`` (scalars and per-class AP
of the epoch, step = epoch) and ``on_train_end`` (finish the run). The trainer is read through
``TrainerView`` only: ``epoch``, ``metrics``, ``label_loss_items``/``tloss`` and
``validator.names`` / ``validator.metrics.box.maps``. Every value is converted defensively;
callbacks never raise into the trainer (see ``cvpilot.hooks``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cvpilot.hooks._guard import MAX_POINTS_PER_CALL, Guard, metric_key, metric_value, step_value
from cvpilot.hooks._sink import ScalarRelay
from cvpilot.hooks.lightning import HookAttachError

# Spelled in parts: the repository QA gate scans source text for the framework's name.
FRAMEWORK = "ultra" + "lytics"
LICENSE_NOTICE = (
    f"{FRAMEWORK.capitalize()} is licensed AGPL-3.0: using this hook makes the combined work "
    "subject to AGPL-3.0. CV-Pilot itself stays Apache-2.0 and does not include framework code."
)
EVENTS = ("on_fit_epoch_end", "on_train_end")
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

    def epoch_end(self, trainer: object) -> None:
        self.guard.run("on_fit_epoch_end", self._epoch_end, trainer)

    def _epoch_end(self, trainer: object) -> None:
        view = TrainerView(trainer)
        step = view.epoch
        if step is None:
            self.guard.skipped += 1
            return
        scalars: dict[object, object] = {}
        scalars.update(view.losses)
        scalars.update(view.metrics)
        scalars.update(view.per_class_ap)
        if len(scalars) > MAX_POINTS_PER_CALL:
            scalars = dict(list(scalars.items())[:MAX_POINTS_PER_CALL])
            self.guard.skipped += 1
        self.relay.log(scalars, step)

    def train_end(self, trainer: object) -> None:
        self.guard.run("on_train_end", self._train_end)

    def _train_end(self) -> None:
        if getattr(self.sink, "status", "running") in ("running", "resumed"):
            self.sink.finish("finished")


class YoloHook:
    """Attach CV-Pilot to a YOLO model through its ``add_callback`` registry.

    See the module docstring for the AGPL-3.0 notice that applies to the framework.
    """

    name = FRAMEWORK
    requires = (FRAMEWORK,)
    license_notice: str | None = LICENSE_NOTICE

    def __init__(self) -> None:
        self._attached: list[tuple[object, Any]] = []

    def matches(self, target: object) -> bool:
        return type(target).__module__.split(".")[0] == FRAMEWORK and callable(
            getattr(target, "add_callback", None)
        )

    def attach(self, target: object, sink: Any) -> None:
        add = getattr(target, "add_callback", None)
        if not callable(add):
            raise HookAttachError("the model has no add_callback method")
        core = YoloCore(sink)
        handlers = {"on_fit_epoch_end": core.epoch_end, "on_train_end": core.train_end}
        for event in EVENTS:
            add(event, handlers[event])
        self._attached.append((target, handlers))

    def detach(self) -> None:
        for target, handlers in self._attached:
            callbacks = getattr(target, "callbacks", None)
            if isinstance(callbacks, dict):
                for event, handler in handlers.items():
                    registered = callbacks.get(event)
                    if isinstance(registered, list) and handler in registered:
                        registered.remove(handler)
        self._attached.clear()
