"""Convert arbitrary user config values into plain JSON types."""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date, datetime
from enum import Enum
from pathlib import PurePath
from typing import Any

_MAX_DEPTH = 8


def to_jsonable(value: Any, _depth: int = 1) -> Any:
    """Return ``value`` as dict/list/str/int/float/bool/None; unknown objects become ``str``.

    Non-finite floats become strings (JSON has no NaN). Cycles and structures deeper than 8
    levels are truncated to the string ``"[TRUNCATED]"``.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, Enum):
        return to_jsonable(value.value, _depth)
    if _depth > _MAX_DEPTH:
        return "[TRUNCATED]"
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=repr) if isinstance(value, (set, frozenset)) else value
        return [to_jsonable(v, _depth + 1) for v in items]
    return str(value)
