"""The public API surface is a snapshot: a change shows up as a failing test and a diff.

Regenerate after an intentional change with ``CVPILOT_UPDATE_API_SNAPSHOT=1 pytest
tests/unit/test_api_surface.py`` and review the diff of ``api_surface.json``: a removed or
re-signed name needs a deprecation cycle (``cvpilot.deprecation``) before it ships.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pytest

import cvpilot

SNAPSHOT = Path(__file__).with_name("api_surface.json")
MODULES = ("cvpilot", "cvpilot.ckpt", "cvpilot.eval", "cvpilot.deprecation", "cvpilot.core.errors")


def _signature(obj: Any) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def describe(obj: Any) -> Any:
    if inspect.isclass(obj):
        entry: dict[str, Any] = {
            "kind": "class",
            "bases": [b.__name__ for b in obj.__bases__ if b is not object],
        }
        if dataclasses.is_dataclass(obj):
            entry["fields"] = {f.name: str(f.type) for f in dataclasses.fields(obj)}
        else:
            entry["init"] = _signature(obj)
        members: dict[str, str] = {}
        for name, value in vars(obj).items():
            if name.startswith("_"):
                continue
            if isinstance(value, property):
                members[name] = "property"
            elif isinstance(value, (staticmethod, classmethod)):
                members[name] = _signature(value.__func__)
            elif callable(value):
                members[name] = _signature(value)
        entry["members"] = dict(sorted(members.items()))
        return entry
    if callable(obj):
        return {"kind": "function", "signature": _signature(obj)}
    return {"kind": type(obj).__name__}


def surface() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in MODULES:
        module = importlib.import_module(name)
        names = list(getattr(module, "__all__", None) or [])
        if name == "cvpilot.core.errors":
            names = sorted(
                n for n, v in vars(module).items() if inspect.isclass(v) and n.endswith("Error")
            )
        out[name] = {n: describe(getattr(module, n)) for n in sorted(names)}
    return out


def test_every_public_module_declares_all() -> None:
    for name in MODULES[:-1]:
        module = importlib.import_module(name)
        exported = getattr(module, "__all__", None)
        assert exported, f"{name} has no __all__"
        assert list(exported) == sorted(exported, key=str.lower) or list(exported) == sorted(
            exported
        ), f"{name}.__all__ is not sorted"
        for item in exported:
            assert hasattr(module, item), f"{name}.__all__ names a missing {item}"


def test_nothing_public_is_exported_without_being_listed() -> None:
    for name in ("cvpilot", "cvpilot.ckpt"):
        module = importlib.import_module(name)
        extras = [
            n
            for n in dir(module)
            if not n.startswith("_")
            and not inspect.ismodule(getattr(module, n))
            and n not in module.__all__
            and n not in {"annotations", "Any", "TYPE_CHECKING"}
        ]
        assert extras == [], f"{name} exposes {extras} outside __all__"


def test_the_public_surface_matches_the_snapshot() -> None:
    current = surface()
    if os.environ.get("CVPILOT_UPDATE_API_SNAPSHOT") == "1":
        SNAPSHOT.write_text(json.dumps(current, indent=2, sort_keys=True) + "\n")
    expected = json.loads(SNAPSHOT.read_text())
    assert current == expected, (
        "the public API changed; if that is intended, regenerate the snapshot "
        "(CVPILOT_UPDATE_API_SNAPSHOT=1) and review the diff"
    )


def test_the_documented_entry_points_are_in_the_snapshot() -> None:
    expected = {"init", "log", "log_artifact", "log_eval", "finish", "autolog", "resume_path"}
    assert expected <= set(cvpilot.__all__)
    assert {"CheckpointManager", "Run", "__version__"} <= set(cvpilot.__all__)


def test_lazy_names_resolve_to_the_real_objects() -> None:
    from cvpilot import api
    from cvpilot.ckpt.manager import CheckpointManager

    assert cvpilot.init is api.init and cvpilot.CheckpointManager is CheckpointManager
    with pytest.raises(AttributeError):
        cvpilot.does_not_exist  # noqa: B018
