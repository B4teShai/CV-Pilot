"""Which checkpoint files this process has verified, for the hooks that hand a path to a framework.

A training framework opens the file it is given by name, so a hook must never pass on the string
``last``, a relative name or an arbitrary path: only a file that ``cvpilot.resume_path`` returned
here, in a format the framework can read. The checkpoint code records what it verified
(``remember``); the hooks ask (``framework_file``). Both live in ``core`` so neither layer imports
the other.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

MAX_REMEMBERED = 256
_VERIFIED: OrderedDict[str, str] = OrderedDict()  # path text -> format


def remember(path: Path, fmt: str) -> None:
    """Record that ``path`` was verified in format ``fmt`` (the newest 256 are kept)."""
    _VERIFIED[str(path)] = fmt
    _VERIFIED.move_to_end(str(path))
    while len(_VERIFIED) > MAX_REMEMBERED:
        _VERIFIED.popitem(last=False)


def framework_file(value: Any) -> Path:
    """The path a framework may load: a verified ``pt`` checkpoint, given as the path or as the
    resume point (anything with ``.path`` and ``.manifest.format``) that ``resume_path`` returned.

    The safetensors files ``CheckpointManager`` writes by default are its own format, not one
    Lightning or Ultralytics can resume from: restore them with ``CheckpointManager.restore``.

    Raises:
        ValueError: a name like ``last``, a relative or unverified path, or a safetensors file.
    """
    manifest = getattr(value, "manifest", None)
    if manifest is not None and hasattr(value, "path"):
        path, fmt = Path(value.path), str(manifest.format)
        if _VERIFIED.get(str(path)) != fmt:
            raise ValueError("this resume point was not returned by cvpilot.resume_path here")
    else:
        text = str(value)
        path = Path(text)
        if not text or not path.is_absolute() or path.name in ("last", ""):
            raise ValueError(
                "pass the absolute path returned by cvpilot.resume_path, never a name like 'last'"
            )
        found = _VERIFIED.get(text)
        if found is None:
            raise ValueError(
                "this path was not verified by cvpilot.resume_path in this process: "
                "pass what it returned"
            )
        fmt = found
    if fmt != "pt":
        raise ValueError(
            f"a {fmt} checkpoint is CheckpointManager's own format and not a file Lightning or "
            "Ultralytics can resume from; restore it with CheckpointManager.restore() into your "
            "model, or keep framework-written files with format='pt' (opt-in) or adopt()"
        )
    return path
