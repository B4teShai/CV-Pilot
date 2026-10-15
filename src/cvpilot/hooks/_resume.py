"""Validation shared by the hooks' resume helpers: only an absolute path of a file that was
verified elsewhere, never the string ``"last"`` that frameworks would resolve by name."""

from __future__ import annotations

from pathlib import Path


def verified_path_text(path: str | Path) -> str:
    """``str(path)`` for an absolute path; anything else (``"last"``, a relative name, an empty
    string) is refused.

    Raises:
        ValueError: the value is not an absolute path or names a checkpoint by keyword.
    """
    text = str(path)
    candidate = Path(text)
    if not text or not candidate.is_absolute() or candidate.name in ("last", ""):
        raise ValueError(
            "pass the absolute path returned by cvpilot.resume_path, never a name like 'last'"
        )
    return text
