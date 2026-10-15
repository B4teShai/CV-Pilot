"""Application layer: the checkpoint manager (ADR-0006).

Public names load on first use so that importing a submodule (``cvpilot.ckpt.fsops``) does not
pull in the manager, the run store registry or NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

_LAZY = {
    "CheckpointManager": "cvpilot.ckpt.manager",
    "Restored": "cvpilot.ckpt.manager",
    "ResumePoint": "cvpilot.ckpt.resume",
    "find_resume_point": "cvpilot.ckpt.resume",
    "private_copy": "cvpilot.ckpt.resume",
    "resume_path": "cvpilot.ckpt.resume",
    "capture_rng": "cvpilot.ckpt.rng",
    "apply_rng": "cvpilot.ckpt.rng",
}

if TYPE_CHECKING:
    from cvpilot.ckpt.manager import CheckpointManager, Restored
    from cvpilot.ckpt.resume import ResumePoint, find_resume_point, private_copy, resume_path
    from cvpilot.ckpt.rng import apply_rng, capture_rng

__all__ = [
    "CheckpointManager",
    "Restored",
    "ResumePoint",
    "apply_rng",
    "capture_rng",
    "find_resume_point",
    "private_copy",
    "resume_path",
]


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'cvpilot.ckpt' has no attribute {name!r}")
