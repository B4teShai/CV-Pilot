"""Which checkpoint files this process has verified, for the hooks that hand a path to a framework.

A training framework opens the file it is given by name, so a hook must never pass on the string
``last``, a relative name or an arbitrary path: only a file that ``cvpilot.resume_path`` returned
here, in a format the framework can read. The checkpoint code records what it verified
(``remember``); the hooks ask (``framework_file``). Both live in ``core`` so neither layer imports
the other.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAX_REMEMBERED = 256


@dataclass(frozen=True)
class _Record:
    fmt: str
    ino: int
    size: int
    mtime_ns: int
    sha256: str


_VERIFIED: OrderedDict[str, _Record] = OrderedDict()


def remember(path: Path, fmt: str, sha256: str, size: int) -> None:
    """Record that ``path`` was verified: its format, the digest it matched and the identity of the
    file (inode, size, modification time) at that moment. The newest 256 are kept."""
    try:
        st = os.stat(path, follow_symlinks=False)
    except OSError:
        return  # nothing to vouch for
    _VERIFIED[str(path)] = _Record(fmt, st.st_ino, st.st_size, st.st_mtime_ns, sha256)
    _VERIFIED.move_to_end(str(path))
    while len(_VERIFIED) > MAX_REMEMBERED:
        _VERIFIED.popitem(last=False)


def format_of(path: Path) -> str | None:
    """The format recorded for a verified path, or None when it was never remembered."""
    found = _VERIFIED.get(str(path))
    return None if found is None else found.fmt


def _still_the_same(path: Path, record: _Record) -> bool:
    """The file is what was verified: same inode, size and time, and the same SHA-256."""
    try:
        st = os.stat(path, follow_symlinks=False)
        if (st.st_ino, st.st_size, st.st_mtime_ns) != (record.ino, record.size, record.mtime_ns):
            return False
        digest = hashlib.sha256()
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            if (os.fstat(fd).st_ino, os.fstat(fd).st_size) != (record.ino, record.size):
                return False  # the name now points at another file than the one checked
            while chunk := os.read(fd, 1 << 20):
                digest.update(chunk)
        finally:
            os.close(fd)
    except OSError:
        return False
    return digest.hexdigest() == record.sha256


def framework_file(value: Any) -> Path:
    """The path a framework may load: a verified ``pt`` checkpoint, given as the path or as the
    resume point (anything with ``.path`` and ``.manifest.format``) that ``resume_path`` returned.

    The safetensors files ``CheckpointManager`` writes by default are its own format, not one
    a training framework can resume from: restore them with ``CheckpointManager.restore``.

    Raises:
        ValueError: a name like ``last``, a relative or unverified path, or a safetensors file.
    """
    manifest = getattr(value, "manifest", None)
    if manifest is not None and hasattr(value, "path"):
        path, fmt = Path(value.path), str(manifest.format)
        record = _VERIFIED.get(str(path))
        if record is None or record.fmt != fmt:
            raise ValueError("this resume point was not returned by cvpilot.resume_path here")
    else:
        text = str(value)
        path = Path(text)
        if not text or not path.is_absolute() or path.name in ("last", ""):
            raise ValueError(
                "pass the absolute path returned by cvpilot.resume_path, never a name like 'last'"
            )
        record = _VERIFIED.get(text)
        if record is None:
            raise ValueError(
                "this path was not verified by cvpilot.resume_path in this process: "
                "pass what it returned"
            )
        fmt = record.fmt
    if fmt != "pt":
        raise ValueError(
            f"a {fmt} checkpoint is CheckpointManager's own format and not a file a training "
            "framework can resume from; restore it with CheckpointManager.restore() into your "
            "model, or keep framework-written files with format='pt' (opt-in) or adopt()"
        )

    if not _still_the_same(path, record):
        raise ValueError(
            "the file changed after it was verified (another writer replaced or rewrote it); "
            "resolve the resume point again, or use cvpilot.ckpt.private_copy"
        )
    return path
