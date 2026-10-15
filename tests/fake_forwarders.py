"""Forwarder doubles registered through a fake entry-point lookup (no tracker, no network)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from typing import Any

import pytest

from cvpilot import registry
from cvpilot.core.schema import MetricPoint, Repro, RunMeta, RunStatus

LEAK_MARKER = "tok-9f8e7d6c5b4a-do-not-log"


class Recording:
    name = "recording"
    requires: tuple[str, ...] = ()
    instances: list[Recording] = []

    def __init__(self, *, allow_insecure: bool = False) -> None:
        self.allow_insecure = allow_insecure
        self.calls: list[tuple[str, Any]] = []
        self.release = threading.Event()
        self.release.set()
        self.entered = threading.Event()
        Recording.instances.append(self)

    def open(self, meta: RunMeta, repro: Repro) -> None:
        self.calls.append(("open", meta.run_id))

    def log_points(self, points: Sequence[MetricPoint]) -> None:
        self.entered.set()
        self.release.wait(30)
        self.calls.append(("points", [(p.key, p.step, p.value) for p in points]))

    def close(self, status: RunStatus) -> None:
        self.calls.append(("close", status))

    @property
    def logged(self) -> list[tuple[str, int, float]]:
        return [row for kind, rows in self.calls if kind == "points" for row in rows]


class Failing(Recording):
    name = "failing"

    def log_points(self, points: Sequence[MetricPoint]) -> None:
        self.calls.append(("attempt", len(points)))
        raise RuntimeError(f"upstream said no: Authorization: Bearer {LEAK_MARKER}")


class FailsToOpen(Recording):
    name = "failsopen"

    def open(self, meta: RunMeta, repro: Repro) -> None:
        raise ConnectionError(f"cannot reach https://user:{LEAK_MARKER}@tracker.invalid")


class FailsToConstruct(Recording):
    name = "failsctor"

    def __init__(self, *, allow_insecure: bool = False) -> None:
        raise RuntimeError("plugin bug")


class Blocking(Recording):
    name = "blocking"

    def __init__(self, *, allow_insecure: bool = False) -> None:
        super().__init__(allow_insecure=allow_insecure)
        self.release.clear()


class NeedsSdk(Recording):
    name = "needssdk"
    requires = ("definitely-not-installed-tracker",)


def use_forwarders(monkeypatch: pytest.MonkeyPatch, **classes: type) -> None:
    """Make ``cvpilot.registry.forwarders`` resolve exactly ``classes`` (name -> class)."""
    Recording.instances.clear()
    points = {
        name: EntryPoint(name, f"fake_forwarders:{cls.__name__}", "cvpilot.forwarders")
        for name, cls in classes.items()
    }

    def lookup(*, group: str) -> list[EntryPoint]:
        return list(points.values()) if group == "cvpilot.forwarders" else []

    monkeypatch.setattr(registry.forwarders, "_lookup", lookup)
    monkeypatch.setattr(registry.forwarders, "_loaded", {})
