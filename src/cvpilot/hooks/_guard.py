"""Isolation of hook callbacks from the host training loop, and defensive value conversion."""

from __future__ import annotations

import hashlib
import math
import operator
from collections.abc import Callable
from typing import Any

from cvpilot.core.schema import MAX_STEP
from cvpilot.logging import get_logger

MAX_CONSECUTIVE_FAILURES = 10
"""A hook whose callbacks failed this many times in a row is disabled for the run."""
MAX_POINTS_PER_CALL = 1000
"""Most metrics one callback hands to the sink; the rest are skipped and counted."""
MAX_KEY_CHARS = 256
"""Longest metric key (the ``MetricPoint`` limit)."""


class Guard:
    """Runs callback bodies so that no ``Exception`` reaches the framework."""

    def __init__(self, hook: str, *, max_consecutive: int = MAX_CONSECUTIVE_FAILURES) -> None:
        self.hook = hook
        self.max_consecutive = max_consecutive
        self.failures = 0
        self.consecutive = 0
        self.skipped = 0
        self.disabled = False

    def run(self, callback: str, fn: Callable[..., Any], *args: Any, always: bool = False) -> None:
        """Run ``fn`` isolated from the host. ``always`` marks a lifecycle callback (finish,
        mark failed): it still runs after the metric callbacks disabled the hook, because a run
        that is never finished stays ``running`` forever."""
        if self.disabled and not always:
            return
        try:
            fn(*args)
        except Exception as exc:  # noqa: BLE001  # host boundary: a callback must never raise
            self.failures += 1
            self.consecutive += 1
            if self.failures == 1:
                get_logger("cvpilot.hooks").warning(
                    "hook_failed", hook=self.hook, callback=callback, error_type=type(exc).__name__
                )
            if self.consecutive >= self.max_consecutive and not self.disabled:
                self.disabled = True
                get_logger("cvpilot.hooks").warning(
                    "hook_disabled", hook=self.hook, failures=self.failures
                )
        else:
            self.consecutive = 0


def metric_key(raw: object, prefix: str = "") -> str | None:
    """A valid metric key from ``raw`` (control characters become ``_``), or None."""
    if not isinstance(raw, str):
        return None
    text = "".join("_" if ord(c) < 32 or ord(c) == 127 else c for c in prefix + raw).strip()
    if len(text) > MAX_KEY_CHARS:
        # cut, but keep two different long keys different: a digest of the whole key ends the name
        tag = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:8]
        text = f"{text[: MAX_KEY_CHARS - 9]}~{tag}"
    return text or None


def metric_value(raw: object) -> float | None:
    """``float(raw)`` for numbers and 1-element tensors; None for anything else. NaN is kept."""
    if isinstance(raw, (bool, str, bytes)) or raw is None:
        return None
    try:
        return float(raw)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001  # a tensor on a dead device, a ragged array, ...
        return None


def step_value(raw: object) -> int | None:
    """An integer step in ``[0, MAX_STEP]``, or None (larger values are skipped, not clamped)."""
    if isinstance(raw, bool) or raw is None:
        return None
    try:
        number = operator.index(raw)  # type: ignore[arg-type]  # exact for ints and numpy integers
    except TypeError:
        try:
            as_float = float(raw)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            return None
        if not math.isfinite(as_float) or as_float != int(as_float):
            return None
        number = int(as_float)
    return number if 0 <= number <= MAX_STEP else None
