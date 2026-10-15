"""Helpers shared by the tracker adapters."""

from __future__ import annotations

import importlib
import os
from collections.abc import Sequence
from typing import Any
from urllib.parse import urlsplit

from cvpilot.core.errors import CvpilotError, MissingExtraError
from cvpilot.core.schema import MetricPoint

MAX_URL_CHARS = 2048
"""Longest endpoint accepted from the environment."""
_TRUE = {"1", "true", "yes", "on"}


class ForwardConfigError(CvpilotError, ValueError):
    """A tracker endpoint or TLS setting is unsafe; the message never contains the value."""


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUE


def check_endpoint(url: str | None, *, allow_insecure: bool, what: str) -> None:
    """Refuse an endpoint with credentials, control characters, or plain http (unless dev).

    ``what`` names the environment variable; the URL itself is never put into the message,
    because it may carry the very credentials this check exists to reject.

    Raises:
        ForwardConfigError: the endpoint is unsafe.
    """
    if not url:
        return
    if len(url) > MAX_URL_CHARS or any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise ForwardConfigError(f"{what} is too long or contains control characters")
    try:
        parts = urlsplit(url)
        userinfo = parts.username is not None or parts.password is not None or "@" in parts.netloc
    except ValueError:
        raise ForwardConfigError(f"{what} is not a valid URL") from None
    if userinfo:
        raise ForwardConfigError(f"{what} contains credentials; pass them in the SDK's environment")
    if parts.scheme.lower() == "http" and not allow_insecure:
        raise ForwardConfigError(
            f"{what} uses plain http; use https (CVPILOT_ENV=dev allows http for a local server)"
        )


def refuse_insecure_flag(name: str, *, allow_insecure: bool) -> None:
    """Refuse an SDK environment switch that turns TLS verification off, unless dev."""
    if env_flag(name) and not allow_insecure:
        raise ForwardConfigError(
            f"{name} disables TLS verification; refused outside CVPILOT_ENV=dev"
        )


def load_sdk(module: str, extra: str) -> Any:
    """Import a tracker SDK lazily; ``MissingExtraError`` names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingExtraError(
            f"forwarder {extra!r} needs the {module} package: install cvpilot[{extra}]"
        ) from None


def group_by_step(points: Sequence[MetricPoint]) -> list[tuple[int, dict[str, float]]]:
    """Points grouped by step, ascending; a key logged twice at one step keeps the last value."""
    steps: dict[int, dict[str, float]] = {}
    for p in points:
        steps.setdefault(p.step, {})[p.key] = p.value
    return sorted(steps.items())
