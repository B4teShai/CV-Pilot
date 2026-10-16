"""The build backend and its dependencies are pinned exactly and agree with the lock file."""

from __future__ import annotations

import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
REQUIRES = tomllib.loads((ROOT / "pyproject.toml").read_text())["build-system"]["requires"]


def test_every_build_requirement_is_pinned_exactly() -> None:
    assert REQUIRES and all(
        re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9_.+-]*", r) for r in REQUIRES
    )
    assert any(r.startswith("hatchling==") for r in REQUIRES)


def test_the_pins_equal_the_locked_versions() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    locked = {p["name"]: p.get("version") for p in lock["package"]}
    for requirement in REQUIRES:
        name, _, version = requirement.partition("==")
        assert locked.get(name) == version, f"{name}: pinned {version}, locked {locked.get(name)}"
