"""Shared fixtures: an isolated CLI environment and an ``invoke`` helper."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import structlog

from cvpilot.cli import app as cli
from cvpilot.logging import clear_request_context

_ENV_VARS = (
    "CVPILOT_ENV",
    "CVPILOT_LOG_LEVEL",
    "CVPILOT_PROJECT",
    "CVPILOT_STORE_COMMAND_LINE",
    "CVPILOT_FORWARD_TO",
)


@dataclass
class CliResult:
    code: int
    out: str
    err: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def event(self, name: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["message"] == name]

    @property
    def human(self) -> str:
        """The stderr lines meant for people: everything that is not a JSON log record."""
        return "\n".join(x for x in self.err.splitlines() if not x.startswith("{"))


@pytest.fixture
def store_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """An empty store root wired through ``CVPILOT_STORE_DIR`` with logging reset."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "store"
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(root))
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)
    structlog.reset_defaults()
    yield root
    clear_request_context()
    structlog.reset_defaults()


@pytest.fixture
def invoke(
    monkeypatch: pytest.MonkeyPatch, store_root: Path, capsys: pytest.CaptureFixture[str]
) -> Callable[..., CliResult]:
    def run(*argv: str) -> CliResult:
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["cvpilot", *argv])
        with pytest.raises(SystemExit) as exc:
            cli.main()
        out, err = capsys.readouterr()
        records = [json.loads(x) for x in err.splitlines() if x.startswith("{")]
        return CliResult(int(exc.value.code or 0), out, err, records)

    return run


@pytest.fixture
def events() -> Iterator[Callable[[], list[dict[str, Any]]]]:
    """Structured log records emitted so far (dev settings, DEBUG kept), as dictionaries."""
    import io

    from cvpilot.logging import configure_logging
    from cvpilot.settings import Settings

    buffer = io.StringIO()
    configure_logging("cvpilot-client", Settings(env="dev", log_level="DEBUG"), stream=buffer)
    yield lambda: [
        json.loads(line) for line in buffer.getvalue().splitlines() if line.startswith("{")
    ]
    clear_request_context()
    structlog.reset_defaults()
