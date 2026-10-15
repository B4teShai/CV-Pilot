"""The generated command-line reference matches the Typer application."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "gen_cli_reference.py"
PAGE = ROOT / "docs" / "reference" / "cli.md"


def generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gen_cli_reference", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_reference_is_what_the_application_renders() -> None:
    rendered = generator().render()
    if os.environ.get("CVPILOT_UPDATE_CLI_REFERENCE") == "1":
        PAGE.write_text(rendered, encoding="utf-8")
    assert PAGE.read_text(encoding="utf-8") == rendered, (
        "docs/reference/cli.md is out of date: run `python scripts/gen_cli_reference.py`"
    )


def test_every_visible_command_has_a_section() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for command in (
        "cvpilot doctor",
        "cvpilot eval",
        "cvpilot compare",
        "cvpilot gate",
        "cvpilot repro",
        "cvpilot runs list",
        "cvpilot runs show",
        "cvpilot baseline set",
        "cvpilot ckpt list",
        "cvpilot ckpt verify",
        "cvpilot ckpt prune",
    ):
        assert f"`{command}`" in text, command
    assert "eval-file" not in text  # hidden aliases are not documented


def test_the_reference_does_not_depend_on_the_terminal() -> None:
    first = generator().render()
    os.environ["COLUMNS"] = "40"
    try:
        assert generator().render() == first
    finally:
        del os.environ["COLUMNS"]
