"""Lazy name-to-adapter registries populated from entry points (ADR-0004).

Application code selects infrastructure by name through a registry and never imports an
adapter module by path. Listing names never imports a plugin; ``get`` imports one.
First-party adapters are declared in ``pyproject.toml`` under the same groups as third-party
ones.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Generic, TypeVar

from cvpilot.core.errors import CvpilotError
from cvpilot.core.schema import ForwarderFactory, StoreFactory, TableIOFactory

T = TypeVar("T")
EntryPointLookup = Callable[..., Iterable[EntryPoint]]
Capturer = Callable[..., Mapping[str, Any]]  # optional keyword parameters, never settings


class UnknownPluginError(CvpilotError, LookupError):
    """A registry has no usable plugin of this name."""


class Registry(Generic[T]):
    """Name to plugin mapping for one entry-point group."""

    def __init__(self, group: str, lookup: EntryPointLookup = entry_points) -> None:
        self.group = group
        self._lookup = lookup
        self._loaded: dict[str, T] = {}

    def _points(self) -> dict[str, EntryPoint]:
        return {ep.name: ep for ep in self._lookup(group=self.group)}

    def names(self) -> list[str]:
        """Registered names, sorted; imports nothing."""
        return sorted(self._points())

    def get(self, name: str) -> T:
        """Import and return the plugin ``name``."""
        if name in self._loaded:
            return self._loaded[name]
        points = self._points()
        if name not in points:
            available = ", ".join(sorted(points)) or "none"
            raise UnknownPluginError(
                f"no {self.group} plugin named {name!r} (available: {available})"
            )
        try:
            plugin: T = points[name].load()
        except Exception as exc:  # noqa: BLE001  # plugin boundary: name the plugin, keep cause
            raise UnknownPluginError(f"{self.group} plugin {name!r} failed to load: {exc}") from exc
        self._loaded[name] = plugin
        return plugin

    def find(self, predicate: Callable[[T], bool]) -> T | None:
        """First plugin (by name order) satisfying ``predicate``, else None."""
        for name in self.names():
            plugin = self.get(name)
            if predicate(plugin):
                return plugin
        return None


stores: Registry[StoreFactory] = Registry("cvpilot.stores")
capturers: Registry[Capturer] = Registry("cvpilot.capturers")
tables: Registry[TableIOFactory] = Registry("cvpilot.tables")
forwarders: Registry[ForwarderFactory] = Registry("cvpilot.forwarders")
