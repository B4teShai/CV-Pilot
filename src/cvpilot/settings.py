"""Runtime settings read from the environment."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError, field_validator

Env = Literal["dev", "staging", "prod"]
Level = Literal["DEBUG", "INFO", "WARNING", "ERROR"]

_PREFIX = "CVPILOT_"
# Duplicated from cvpilot.core.schema on purpose: settings imports no other cvpilot module.
# A unit test asserts both patterns stay equal.
PROJECT_PATTERN = r"[a-z0-9][a-z0-9_-]{0,63}"
_PROJECT = re.compile(PROJECT_PATTERN)
_TRUE = {"true", "1", "yes", "on"}
_FALSE = {"false", "0", "no", "off"}


def _bool_text(value: str) -> str:
    """Map a boolean spelling to ``"true"``/``"false"``; anything else is left to fail."""
    lowered = value.lower()
    return "true" if lowered in _TRUE else "false" if lowered in _FALSE else value


_FIELDS: dict[str, tuple[str, Callable[[str], str]]] = {
    "env": ("ENV", str.lower),
    "log_level": ("LOG_LEVEL", str.upper),
    "store_command_line": ("STORE_COMMAND_LINE", _bool_text),
    "project": ("PROJECT", str),
    "store_dir": ("STORE_DIR", str),
}


class SettingsError(ValueError):
    """An environment variable holds an invalid value."""


class Settings(BaseModel, frozen=True):
    env: Env = "prod"
    log_level: Level = "INFO"
    store_command_line: bool = True
    project: str | None = None
    store_dir: Path | None = None

    @field_validator("project")
    @classmethod
    def _project(cls, value: str | None) -> str | None:
        if value is not None and not _PROJECT.fullmatch(value):
            raise ValueError("invalid project name")
        return value

    @field_validator("store_dir", mode="before")
    @classmethod
    def _store_dir(cls, value: Any) -> Path | None:
        if value is None:
            return None
        text = str(value)
        if not text.strip() or "\x00" in text:
            raise ValueError("must be a non-empty path")
        try:
            path = Path(text).expanduser().resolve()
        except (RuntimeError, OSError) as exc:
            raise ValueError("path cannot be resolved") from exc
        if path == Path(path.anchor):
            raise ValueError("must not be the filesystem root")
        if path.exists() and not path.is_dir():
            raise ValueError("must be a directory")
        return path

    @property
    def effective_log_level(self) -> Level:
        """DEBUG is honoured only in dev; elsewhere it is capped to INFO."""
        if self.log_level == "DEBUG" and self.env != "dev":
            return "INFO"
        return self.log_level


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    source = os.environ if environ is None else environ
    raw: dict[str, str] = {}
    for field, (suffix, normalise) in _FIELDS.items():
        value = source.get(_PREFIX + suffix)
        if value is not None:
            raw[field] = normalise(value.strip())
    try:
        return Settings(**raw)  # type: ignore[arg-type]
    except ValidationError as exc:
        names = sorted({_PREFIX + _FIELDS[str(err["loc"][0])][0] for err in exc.errors()})
        raise SettingsError(f"invalid value for {', '.join(names)}") from None
