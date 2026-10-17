"""Alert channel doubles registered through a fake entry-point lookup (no network)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from importlib.metadata import EntryPoint
from typing import Any

import pytest
from fake_forwarders import LEAK_MARKER

from cvpilot import registry
from cvpilot.core.schema import Alert, Repro, RunMeta, RunStatus


class RecordingChannel:
    name = "recording"
    requires: tuple[str, ...] = ()
    instances: list[RecordingChannel] = []

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        self.allow_insecure = allow_insecure
        self.allowed_hosts = tuple(allowed_hosts)
        self.calls: list[tuple[str, Any]] = []
        self.release = threading.Event()
        self.release.set()
        RecordingChannel.instances.append(self)

    def open(self, meta: RunMeta, repro: Repro) -> None:
        self.calls.append(("open", meta.run_id))

    def send(self, alerts: Sequence[Alert]) -> None:
        self.release.wait(30)
        self.calls.append(("send", [(a.rule, a.step, a.text) for a in alerts]))

    def close(self, status: RunStatus) -> None:
        self.calls.append(("close", status))

    @property
    def sent(self) -> list[tuple[str, int | None, str]]:
        return [row for kind, rows in self.calls if kind == "send" for row in rows]


class FailingChannel(RecordingChannel):
    name = "failing"

    def send(self, alerts: Sequence[Alert]) -> None:
        raise OSError(f"https://hooks.invalid/{LEAK_MARKER}")


class BlockingChannel(RecordingChannel):
    name = "blocking"

    def __init__(self, *, allow_insecure: bool = False, allowed_hosts: Sequence[str] = ()) -> None:
        super().__init__(allow_insecure=allow_insecure, allowed_hosts=allowed_hosts)
        self.release.clear()


def use_channels(monkeypatch: pytest.MonkeyPatch, **classes: type) -> None:
    """Make ``cvpilot.registry.channels`` resolve exactly ``classes`` (name -> class)."""
    RecordingChannel.instances.clear()
    points = {
        name: EntryPoint(name, f"fake_channels:{cls.__name__}", "cvpilot.channels")
        for name, cls in classes.items()
    }

    def lookup(*, group: str) -> list[EntryPoint]:
        return list(points.values()) if group == "cvpilot.channels" else []

    monkeypatch.setattr(registry.channels, "_lookup", lookup)
    monkeypatch.setattr(registry.channels, "_loaded", {})
