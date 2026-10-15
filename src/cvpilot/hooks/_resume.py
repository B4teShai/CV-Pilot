"""Validation shared by the hooks' resume helpers (see ``cvpilot.core.verified``)."""

from __future__ import annotations

from typing import Any

from cvpilot.core.verified import framework_file


def verified_path_text(path: Any) -> str:
    """``str(path)`` for a ``pt`` checkpoint that ``cvpilot.resume_path`` returned in this process
    (the path or the ``ResumePoint``).

    Raises:
        ValueError: the value is a name, a relative or unverified path, or a safetensors file.
    """
    return str(framework_file(path))
