from __future__ import annotations

from importlib.metadata import EntryPoint

import pytest

from cvpilot.core.errors import CvpilotError
from cvpilot.registry import Registry, UnknownPluginError, stores


def fake_entry_points(**mapping: str):  # type: ignore[no-untyped-def]
    def lookup(*, group: str) -> list[EntryPoint]:
        return [EntryPoint(name, value, group) for name, value in mapping.items()]

    return lookup


def test_get_loads_lazily_and_caches() -> None:
    reg: Registry[object] = Registry(
        "x.group", fake_entry_points(sep="os.path:sep", other="does.not:exist")
    )
    assert reg.names() == ["other", "sep"]  # listing never imports
    first = reg.get("sep")
    assert first == "/" or first == "\\"
    assert reg.get("sep") is first


def test_unknown_name_lists_available() -> None:
    reg: Registry[object] = Registry("x.group", fake_entry_points(a="os:sep"))
    with pytest.raises(UnknownPluginError, match="available: a") as exc:
        reg.get("b")
    assert isinstance(exc.value, CvpilotError) and isinstance(exc.value, LookupError)


def test_broken_plugin_is_reported_with_its_name() -> None:
    reg: Registry[object] = Registry("x.group", fake_entry_points(bad="no.such.module:thing"))
    with pytest.raises(UnknownPluginError, match="bad"):
        reg.get("bad")


def test_find_returns_first_match_or_none() -> None:
    reg: Registry[object] = Registry("x.group", fake_entry_points(a="os:sep", b="os:linesep"))
    assert reg.find(lambda v: v == "\n" or v == "\r\n") in ("\n", "\r\n")
    assert reg.find(lambda v: v == "nope") is None


def test_first_party_local_store_is_registered() -> None:
    assert "local" in stores.names()
    assert callable(stores.get("local"))
