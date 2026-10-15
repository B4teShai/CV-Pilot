"""Store path resolution and validation. Every path built from input goes through here."""

from __future__ import annotations

import re
from pathlib import Path

from cvpilot.core.errors import InvalidRunIdError, StoreError
from cvpilot.core.schema import is_valid_project
from cvpilot.core.ulid import is_valid_ulid

_SHA256 = re.compile(r"[0-9a-f]{64}")


def default_root() -> Path:
    try:
        return Path.home() / ".cvpilot"
    except RuntimeError as exc:
        raise StoreError("cannot locate the home directory; set CVPILOT_STORE_DIR") from exc


def resolve_project_dir(root: Path | None, project: str) -> Path:
    """Resolved ``<root>/<project>``, guaranteed to lie inside the resolved root."""
    if not is_valid_project(project):
        raise StoreError(f"invalid project name {project!r}: use [a-z0-9][a-z0-9_-]{{0,63}}")
    base = (root if root is not None else default_root()).expanduser().resolve()
    if base.exists() and not base.is_dir():
        raise StoreError(f"store root {base} is not a directory")
    target = (base / project).resolve()
    if not target.is_relative_to(base) or target == base:
        raise StoreError(f"project directory for {project!r} resolves outside the store root")
    return target


def require_run_id(run_id: str) -> str:
    """Return ``run_id`` if it is a valid ULID, else raise before it can reach a path."""
    if not is_valid_ulid(run_id):
        raise InvalidRunIdError(f"{run_id[:40]!r} is not a valid run id (26-character ULID)")
    return run_id


def require_sha256(value: str) -> str:
    if not _SHA256.fullmatch(value):
        raise StoreError("artifact address must be 64 lower-case hex characters")
    return value
