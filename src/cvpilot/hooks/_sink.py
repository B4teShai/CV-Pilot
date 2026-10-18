"""Shared scalar plumbing: bounded, validated, strictly increasing points into a sink."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cvpilot.hooks._guard import MAX_POINTS_PER_CALL, Guard, metric_key, metric_value

MAX_TRACKED_KEYS = 10_000
"""Most keys a relay remembers a last step for; any key beyond that is skipped and counted."""


class ScalarRelay:
    """Turns raw framework scalars into one ``sink.log`` call per step.

    Keys must be strings and values numbers (NaN and infinity pass: they are real signals);
    anything else is skipped and counted on the guard. A key is only logged at a step greater than
    the last step logged for it, because the store requires strictly increasing steps per key and
    a framework may report the same metric twice at one step; skipped repeats are counted too. At
    most ``MAX_POINTS_PER_CALL`` points go to the sink per call.
    """

    def __init__(self, sink: Any, guard: Guard) -> None:
        self._sink = sink
        self._guard = guard
        self._last: dict[str, int] = {}

    def log(self, raw: Mapping[Any, Any], step: int, prefix: str = "") -> None:
        points: dict[str, float] = {}
        for raw_key, raw_value in raw.items():
            key, value = metric_key(raw_key, prefix), metric_value(raw_value)
            if key is None or value is None or self._last.get(key, -1) >= step:
                self._guard.skipped += 1
                continue
            if key not in self._last and len(self._last) + len(points) >= MAX_TRACKED_KEYS:
                self._guard.skipped += 1
                continue
            if len(points) >= MAX_POINTS_PER_CALL:
                self._guard.skipped += 1
                continue
            points[key] = value
        if not points:
            return
        self._sink.log(points, step=step)
        for key in points:
            self._last[key] = step
