"""``cvpilot alerts check``: validate an alert configuration and replay a run against it.

Exit codes: 0 the configuration is valid, its channels can be built from the environment and
(with ``--run``) the replay ran; 1 the run cannot be found, read or replayed; 2 usage or
configuration error (an invalid configuration file, an unknown channel or one whose endpoint or
credential is missing or unsafe, no project, a malformed run id). It never sends anything and
never fails because an alert *would* fire.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cvpilot.cli.runs_cmds import ProjectOpt, _project, run_id_argument
from cvpilot.core.errors import AlertConfigError, CvpilotError
from cvpilot.core.text import safe_text
from cvpilot.settings import Settings

alerts_app = typer.Typer(help="Alert rules.", no_args_is_help=True)


def _optional_run_id(value: str | None) -> str | None:
    return None if value is None else run_id_argument(value)


def _fail(message: str) -> typer.Exit:
    typer.echo(f"error: {safe_text(message, 300)}", err=True)
    return typer.Exit(2)


@alerts_app.command("check")
def alerts_check(
    ctx: typer.Context,
    config: Annotated[Path, typer.Argument(dir_okay=False, help="The alert configuration (JSON).")],
    project: ProjectOpt = None,
    run: Annotated[
        str | None,
        typer.Option("--run", callback=_optional_run_id, help="Replay this recorded run."),
    ] = None,
) -> None:
    """Validate CONFIG, check its channels and replay a run against it; nothing is sent."""
    from cvpilot import runs
    from cvpilot.alerts.config import load_alert_config
    from cvpilot.alerts.replay import dry_run

    settings: Settings = ctx.obj
    try:
        cfg = load_alert_config(config)
    except AlertConfigError as exc:
        raise _fail(str(exc)) from None
    typer.echo(f"config: ok ({len(cfg.rules)} rules, rate limit {cfg.rate_limit_s:g} s)")
    for spec in cfg.rules:
        fields = {k: v for k, v in spec.model_dump().items() if k != "rule" and v is not None}
        detail = ", ".join(f"{k}={safe_text(v, 60)}" for k, v in fields.items())
        typer.echo(f"rule {spec.rule}" + (f" ({detail})" if detail else ""))
    if not cfg.channels:
        typer.echo("channels: none (alerts are only logged)")
    else:
        try:
            built = runs.resolve_channels(
                cfg.channels,
                allow_insecure=settings.env == "dev",
                allowed_hosts=settings.forward_hosts,
            )
        except CvpilotError as exc:
            raise _fail(str(exc)) from None
        for name in cfg.channels:
            ready = any(getattr(channel, "name", None) == name for channel in built)
            typer.echo(f"channel {name}: {'ready' if ready else 'could not be built'}")
            if not ready:
                raise _fail(f"channel {name} could not be built")
    if run is not None:
        found = dry_run(cfg, _project(ctx, project), run, store_dir=settings.store_dir)
        typer.echo(
            f"replay: run {found.run_id} (project {_project(ctx, project)}): "
            f"{found.n_points} points, {found.n_snapshots} evaluation snapshots, "
            f"status {found.status}"
        )
        for alert in found.alerts:
            where = "" if alert.step is None else f" at step {alert.step}"
            typer.echo(f"would fire: {alert.rule}{where}: {alert.text}")
        if not found.alerts:
            typer.echo("would fire: nothing")
        if found.suppressed:
            typer.echo(f"would suppress: {found.suppressed} (rate limit or per-run cap)")
        if found.rule_failures:
            typer.echo(f"rule failures: {found.rule_failures}")
    typer.echo("nothing was sent")
