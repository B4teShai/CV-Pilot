"""Deprecation policy as code: a decorator, a warning function and a registry.

Policy (documented in ``docs/reference/api.md``): a public name is removed only after at least
one release that warned. ``deprecated(since="0.2.0", removal="0.4.0")`` refuses a ``removal``
that is not in a later minor version than ``since``; the warning names the version it started
in, the version that removes it and the replacement, and points at the caller's line. Every
decorated name is registered, and a test fails when something registered is due for removal in
the version being released (``expired``), so a deprecation cannot be forgotten.
"""

from __future__ import annotations

import functools
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])
_VERSION = re.compile(r"([0-9]+)\.([0-9]+)\.([0-9]+)")


class CvpilotDeprecationWarning(DeprecationWarning):
    """Warns about use of a deprecated cvpilot API; shown by default, unlike the base class."""


warnings.simplefilter("default", CvpilotDeprecationWarning)


@dataclass(frozen=True)
class Entry:
    """One registered deprecation."""

    name: str
    since: str
    removal: str
    alternative: str | None


_REGISTRY: list[Entry] = []


def _parse(version: str) -> tuple[int, int, int]:
    match = _VERSION.match(version) if isinstance(version, str) else None
    if match is None:
        raise ValueError(f"{version!r} is not a MAJOR.MINOR.PATCH version")
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _message(what: str, since: str, removal: str, alternative: str | None) -> str:
    text = f"{what} is deprecated since cvpilot {since} and will be removed in {removal}"
    return f"{text}; use {alternative} instead" if alternative else text


def deprecated(*, since: str, removal: str, alternative: str | None = None) -> Callable[[F], F]:
    """Decorator for a function or method that is going away.

    Raises:
        ValueError: a version is not ``MAJOR.MINOR.PATCH``, or ``removal`` is not in a later
            minor version than ``since`` (the policy needs at least one release that warns).
    """
    start, end = _parse(since), _parse(removal)
    if end[:2] <= start[:2]:
        raise ValueError(
            "removal must be in a later minor version than since: users get at least one "
            "release that warns"
        )

    def decorate(func: F) -> F:
        name = f"{func.__module__}.{func.__qualname__}"
        _REGISTRY.append(Entry(name, since, removal, alternative))
        message = _message(f"{func.__qualname__}()", since, removal, alternative)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(message, CvpilotDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        note = f"Deprecated since {since}, removed in {removal}."
        if alternative:
            note += f" Use {alternative}."
        wrapper.__doc__ = f"{func.__doc__ or ''}\n\n    .. deprecated:: {since}\n        {note}\n"
        return wrapper  # type: ignore[return-value]

    return decorate


def warn_deprecated(
    what: str,
    *,
    since: str,
    removal: str,
    alternative: str | None = None,
    stacklevel: int = 2,
) -> None:
    """Warn about a deprecated parameter, value or module (``what`` names it)."""
    _parse(since)
    _parse(removal)
    warnings.warn(
        _message(what, since, removal, alternative),
        CvpilotDeprecationWarning,
        stacklevel=stacklevel,
    )


def registered() -> list[Entry]:
    """Every deprecation registered so far."""
    return list(_REGISTRY)


def expired(current: str) -> list[Entry]:
    """Registered deprecations whose removal version is ``current`` or older."""
    now = _parse(current)
    return [e for e in _REGISTRY if _parse(e.removal) <= now]


__all__ = [
    "CvpilotDeprecationWarning",
    "Entry",
    "deprecated",
    "expired",
    "registered",
    "warn_deprecated",
]
