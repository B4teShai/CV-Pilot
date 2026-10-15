"""CV-Pilot: run capture, regression gates and resume-safe checkpoints for vision models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_API = {"init", "log", "log_artifact", "log_eval", "finish", "autolog", "Run", "resume_path"}

if TYPE_CHECKING:
    from cvpilot.api import Run, autolog, finish, init, log, log_artifact, log_eval, resume_path
    from cvpilot.ckpt.manager import CheckpointManager

__all__ = [
    "CheckpointManager",
    "Run",
    "__version__",
    "autolog",
    "finish",
    "init",
    "log",
    "log_artifact",
    "log_eval",
    "resume_path",
]


def __getattr__(name: str) -> Any:
    """Load the public API on first use so ``cvpilot --version`` stays fast."""
    if name in _API:
        from cvpilot import api

        return getattr(api, name)
    if name == "CheckpointManager":
        from cvpilot.ckpt.manager import CheckpointManager

        return CheckpointManager
    raise AttributeError(f"module 'cvpilot' has no attribute {name!r}")
