"""Command-line entry point."""

from __future__ import annotations

import platform
import sys
from typing import Annotated, NoReturn

import typer

from cvpilot import __version__
from cvpilot.cli.baseline_cmds import baseline_app
from cvpilot.cli.eval_cmds import eval_command
from cvpilot.cli.gate_cmds import compare_command, gate_command
from cvpilot.cli.runs_cmds import repro_command, runs_app
from cvpilot.core.errors import CvpilotError
from cvpilot.logging import clear_request_context, configure_logging, get_logger, scrub
from cvpilot.settings import SettingsError, load_settings

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False, help="CV-Pilot")


def _version(value: bool) -> None:
    if value:
        typer.echo(f"cvpilot {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    ctx: typer.Context,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version, is_eager=True, help="Show version and exit."),
    ] = False,
) -> None:
    try:
        settings = load_settings()
    except SettingsError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(2) from None
    configure_logging("cvpilot", settings)
    ctx.obj = settings


app.add_typer(runs_app, name="runs")
app.add_typer(baseline_app, name="baseline")
app.command("repro")(repro_command)
app.command("eval")(eval_command)
app.command("compare")(compare_command)
app.command("gate")(gate_command)
app.command("eval-file", hidden=True)(eval_command)


def _run_doctor() -> None:
    get_logger("cvpilot.cli").info(
        "doctor_completed",
        status="ok",
        package_version=__version__,
        python_version=platform.python_version(),
    )


@app.command()
def doctor() -> None:
    """Check that the installation works."""
    _run_doctor()
    typer.echo("ok")


def _usage_exit_code(exc: Exception) -> int | None:
    """Exit code for Click-style usage errors, else None.

    Duck-typed because recent Typer vendors its own Click exception classes, so
    ``isinstance(exc, click.ClickException)`` misses them.
    """
    code = getattr(exc, "exit_code", None)
    if isinstance(code, int) and code != 0 and callable(getattr(exc, "show", None)):
        return code
    return None


def main() -> NoReturn:
    argv = sys.argv[1:]
    command = argv[0] if argv and not argv[0].startswith("-") else None
    code = 0
    try:
        result = app(standalone_mode=False)
        code = result if isinstance(result, int) else 0
    except typer.Abort:
        code = 130
    except CvpilotError as exc:  # expected failure: one line for the human, one event for machines
        typer.echo(f"error: {scrub(str(exc))}", err=True)
        get_logger("cvpilot.cli").error(
            exc.event, command=command, exc_info=exc, **exc.log_fields()
        )
        code = 1
    except Exception as exc:  # error boundary: logged once here, nowhere else
        usage_code = _usage_exit_code(exc)
        if usage_code is None:
            get_logger("cvpilot.cli").error("command_failed", command=command, exc_info=exc)
            code = 1
        else:
            exc.show()  # type: ignore[attr-defined]
            code = usage_code
    finally:
        clear_request_context()
    raise SystemExit(code)
