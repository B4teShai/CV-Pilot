"""Render the command-line reference (docs/reference/cli.md) from the Typer application.

Run ``python scripts/gen_cli_reference.py`` after changing a command, an option or a help text;
``tests/unit/test_docs_cli_reference.py`` fails when the committed page differs from what this
renders, so the reference cannot drift from the code. The script reads the click command tree
only: nothing is executed and the output does not depend on the terminal width.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path
from typing import Any

import typer

OUT = Path(__file__).resolve().parents[1] / "docs" / "reference" / "cli.md"
HEADER = """# Command line

This page is generated from the Typer application by `scripts/gen_cli_reference.py`; a test
fails when it is out of date. Exit codes: `0` success, `1` failure (a regression, a failed
verification, a missing or unreadable input), `2` usage or configuration error, `130`
interrupted. Every command logs structured JSON events on `stderr`; results go to `stdout`.
"""


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _kind(param: Any) -> str:
    """ "Argument" or "Option" (Typer's subclasses are named TyperArgument and TyperOption)."""
    name = type(param).__name__
    return "Argument" if "Argument" in name else "Option" if "Option" in name else name


def _default(param: Any) -> str:
    default = param.default
    if default is None or callable(default) or default == () or default is False:
        return ""
    return f"`{getattr(default, 'value', default)}`"


def _type_name(param: Any) -> str:
    kind = param.type
    if hasattr(kind, "choices"):
        return "one of " + ", ".join(f"`{c}`" for c in kind.choices)
    if getattr(param, "is_flag", False) or str(kind.name) == "boolean":
        return "flag"
    return str(kind.name)


def _params(command: Any) -> list[str]:
    arguments = [p for p in command.params if _kind(p) == "Argument"]
    options = [
        p
        for p in command.params
        if _kind(p) == "Option" and not p.hidden and "--help" not in p.opts
    ]
    lines: list[str] = []
    if arguments:
        lines += ["", "| Argument | Type | Description |", "|---|---|---|"]
        for p in arguments:
            lines.append(
                f"| `{p.name}` | {_type_name(p)} | {_cell(getattr(p, 'help', '') or '')} |"
            )
    if options:
        lines += ["", "| Option | Type | Default | Description |", "|---|---|---|---|"]
        for p in options:
            names = ", ".join(f"`{o}`" for o in [*p.opts, *p.secondary_opts])
            note = " Required." if p.required else ""
            note += " Repeatable." if p.multiple else ""
            lines.append(
                f"| {names} | {_type_name(p)} | {_default(p)} | {_cell((p.help or '') + note)} |"
            )
    return lines


def _walk(command: Any, path: list[str], depth: int, out: list[str]) -> None:
    if command.hidden:
        return
    name = " ".join(path)
    ctx = command.make_context(path[-1], [], resilient_parsing=True)
    if path[1:]:
        out += ["", f"{'#' * min(depth, 4)} `{name}`"]
        usage = command.get_usage(ctx).replace("Usage: ", "", 1).strip()
        out += ["", "```text", usage.replace(path[-1], name, 1), "```"]
        if command.help:
            out += ["", inspect.cleandoc(command.help)]
        out += _params(command)
    if hasattr(command, "list_commands"):
        for sub_name in command.list_commands(ctx):
            sub = command.get_command(ctx, sub_name)
            if sub is not None:
                _walk(sub, [*path, sub_name], depth + 1, out)


def render() -> str:
    """The whole reference page as text."""
    from cvpilot.cli.app import app

    root = typer.main.get_command(app)
    out: list[str] = [HEADER]
    _walk(root, ["cvpilot"], 1, out)
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    OUT.write_text(render(), encoding="utf-8")
    sys.stdout.write(f"wrote {OUT}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
