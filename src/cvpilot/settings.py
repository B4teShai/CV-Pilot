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
FORWARDER_PATTERN = r"[a-z][a-z0-9_-]{0,31}"
_FORWARDER = re.compile(FORWARDER_PATTERN)
MAX_FORWARDERS = 8
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
    "forward_to": ("FORWARD_TO", str),
    "forward_hosts": ("FORWARD_HOSTS", str),
}


def parse_forwarders(value: Any) -> tuple[str, ...]:
    """Forwarder names from a comma-separated string or a sequence; validated, no duplicates.

    Raises:
        ValueError: a name is empty or does not match ``[a-z][a-z0-9_-]{0,31}``, a name is
            repeated, or there are more than ``MAX_FORWARDERS``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",")] if value.strip() else []
    elif isinstance(value, (list, tuple)):
        names = list(value)
    else:
        raise ValueError("must be a comma-separated list of names")
    if len(names) > MAX_FORWARDERS or len(set(names)) != len(names):
        raise ValueError("too many or repeated forwarder names")
    if not all(isinstance(n, str) and _FORWARDER.fullmatch(n) for n in names):
        raise ValueError("invalid forwarder name")
    return tuple(names)


_HOST = re.compile(r"[a-z0-9](?:[a-z0-9.-]{0,251}[a-z0-9])?")
MAX_FORWARD_HOSTS = 32


def parse_hosts(value: Any) -> tuple[str, ...]:
    """Lower-case host names from a comma-separated string or a sequence (no wildcards).

    Raises:
        ValueError: a name is empty, has characters outside ``[a-z0-9.-]``, or there are more
            than ``MAX_FORWARD_HOSTS``.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        names = [part.strip().lower() for part in value.split(",")] if value.strip() else []
    elif isinstance(value, (list, tuple)):
        names = [str(v).strip().lower() for v in value]
    else:
        raise ValueError("must be a comma-separated list of host names")
    if len(names) > MAX_FORWARD_HOSTS or len(set(names)) != len(names):
        raise ValueError("too many or repeated hosts")
    if not all(_HOST.fullmatch(n) for n in names):
        raise ValueError("invalid host name")
    return tuple(names)


def is_rank_zero(environ: Mapping[str, str] | None = None) -> bool:
    """False when ``RANK`` or ``LOCAL_RANK`` names a process other than rank zero.

    Distributed launchers set these; only rank zero starts forwarders and other background
    threads, so a multi-process job does not log every point once per process. A missing or
    non-integer value counts as rank zero.
    """
    source = os.environ if environ is None else environ
    for name in ("RANK", "LOCAL_RANK"):
        raw = source.get(name, "").strip()
        if raw.lstrip("-").isdigit() and int(raw) != 0:
            return False
    return True


class SettingsError(ValueError):
    """An environment variable holds an invalid value."""


class Settings(BaseModel, frozen=True):
    env: Env = "prod"
    log_level: Level = "INFO"
    store_command_line: bool = True
    project: str | None = None
    store_dir: Path | None = None
    forward_to: tuple[str, ...] = ()
    forward_hosts: tuple[str, ...] = ()

    @field_validator("forward_hosts", mode="before")
    @classmethod
    def _forward_hosts(cls, value: Any) -> tuple[str, ...]:
        return parse_hosts(value)

    @field_validator("forward_to", mode="before")
    @classmethod
    def _forward_to(cls, value: Any) -> tuple[str, ...]:
        return parse_forwarders(value)

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
