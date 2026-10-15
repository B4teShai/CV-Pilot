"""CV-Pilot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.0.1.dev0"

_LAZY = {"init", "log", "log_artifact", "log_eval", "finish", "Run"}

if TYPE_CHECKING:
    from cvpilot.api import Run, finish, init, log, log_artifact, log_eval

__all__ = ["Run", "__version__", "finish", "init", "log", "log_artifact", "log_eval"]


def __getattr__(name: str) -> Any:
    """Load the public API on first use so ``cvpilot --version`` stays fast."""
    if name in _LAZY:
        from cvpilot import api

        return getattr(api, name)
    raise AttributeError(f"module 'cvpilot' has no attribute {name!r}")
