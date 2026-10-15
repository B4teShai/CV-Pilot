"""Forward-only schema migrations, one module per version: ``NNNN_<name>.py``.

Each module defines ``VERSION: int``, ``NAME: str`` and ``upgrade(conn)``; ``upgrade`` runs
inside a transaction opened by the runner and must not commit.
"""

from __future__ import annotations

import importlib
import pkgutil
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    upgrade: Callable[[sqlite3.Connection], object]


def discover() -> list[Migration]:
    """All bundled migrations ordered by version; versions must be 1..N without gaps."""
    found = []
    for info in pkgutil.iter_modules(__path__):
        if not info.name[:4].isdigit():
            continue
        module = importlib.import_module(f"{__name__}.{info.name}")
        found.append(Migration(module.VERSION, module.NAME, module.upgrade))
    found.sort(key=lambda m: m.version)
    if [m.version for m in found] != list(range(1, len(found) + 1)):
        raise RuntimeError("migration versions must be consecutive starting at 1")
    return found
