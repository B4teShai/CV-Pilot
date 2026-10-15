"""``cvpilot runs list|show`` and ``cvpilot repro``: read-only views of the run store.

Exit codes: 0 ok; 1 the run is not found or the store is unreadable or too new; 2 usage or
configuration error (no project, invalid project or run id, invalid status).
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Any

import typer

from cvpilot.core.schema import PROJECT_PATTERN, RunStatus, is_valid_project
from cvpilot.core.ulid import is_valid_ulid
from cvpilot.settings import Settings

runs_app = typer.Typer(help="Inspect recorded runs.", no_args_is_help=True)


class StatusOption(str, Enum):
    running = "running"
    resumed = "resumed"
    finished = "finished"
    failed = "failed"
    preempted = "preempted"


def _project(ctx: typer.Context, option: str | None) -> str:
    settings: Settings = ctx.obj
    project = option or settings.project
    if project is None:
        typer.echo("error: no project: pass --project or set CVPILOT_PROJECT", err=True)
        raise typer.Exit(2)
    if not is_valid_project(project):
        raise typer.BadParameter(f"must match {PROJECT_PATTERN}", param_hint="--project")
    return project


def run_id_argument(value: str) -> str:
    """Normalise to upper case and require a ULID before the value can reach any path."""
    normalised = value.strip().upper()
    if not is_valid_ulid(normalised):
        raise typer.BadParameter("not a valid run id (26-character ULID)")
    return normalised


ProjectOpt = Annotated[
    str | None, typer.Option("--project", help="Project name (default: $CVPILOT_PROJECT).")
]
RunIdArg = Annotated[str, typer.Argument(callback=run_id_argument, help="Run id (ULID).")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")]


def _dump(model: Any) -> Any:
    return json.loads(model.model_dump_json())


@runs_app.command("list")
def runs_list(
    ctx: typer.Context,
    project: ProjectOpt = None,
    status: Annotated[
        StatusOption | None, typer.Option("--status", help="Only this status.")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """List runs of a project, newest first. Exit 0 (an empty project is not an error)."""
    from cvpilot import runs

    found = runs.list_runs(
        _project(ctx, project),
        status=None if status is None else _status(status),
        store_dir=ctx.obj.store_dir,
    )
    if as_json:
        typer.echo(json.dumps([_dump(r) for r in found]))
        return
    typer.echo(f"{'RUN_ID':<26}  {'STATUS':<9}  {'STARTED':<25}  NAME")
    for r in found:
        typer.echo(f"{r.run_id:<26}  {r.status:<9}  {r.started_at.isoformat():<25}  {r.name}")


def _status(option: StatusOption) -> RunStatus:
    value: RunStatus = option.value
    return value


runs_app.command("ls", hidden=True)(runs_list)


@runs_app.command("show")
def runs_show(
    ctx: typer.Context, run_id: RunIdArg, project: ProjectOpt = None, as_json: JsonOpt = False
) -> None:
    """Show one stored run. Exit 1 if the run does not exist."""
    from cvpilot import runs

    view = runs.get_run_view(_project(ctx, project), run_id, store_dir=ctx.obj.store_dir)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "meta": _dump(view.meta),
                    "repro": _dump(view.repro),
                    "metrics": _dump(view.stats),
                    "artifacts": [_dump(a) for a in view.artifacts],
                    "snapshots": [_dump(s) for s in view.snapshots],
                }
            )
        )
        return
    m, r, s = view.meta, view.repro, view.stats
    ended = m.ended_at.isoformat() if m.ended_at else "-"
    rows = [
        ("run_id", m.run_id),
        ("project", m.project),
        ("name", m.name),
        ("status", m.status),
        ("parent_run_id", m.parent_run_id or "-"),
        ("task", m.task),
        ("framework", m.framework),
        ("host", m.host),
        ("started_at", m.started_at.isoformat()),
        ("ended_at", ended),
        ("git_commit", r.git_commit or "-"),
        ("git_dirty", str(r.git_dirty).lower()),
        ("python_version", r.python_version),
        ("cuda", r.cuda or "-"),
        ("gpu_names", ", ".join(r.gpu_names) or "-"),
        ("dataset_uri", r.dataset_uri or "-"),
        ("dataset_hash", r.dataset_hash),
        ("packages_lock_sha256", r.packages_lock_sha256),
        ("seed", str(r.seed) if r.seed is not None else "-"),
        ("command_line", r.command_line),
        ("config", json.dumps(r.config, sort_keys=True)),
        ("n_points", str(s.n_points)),
        ("n_files", str(s.n_files)),
        ("steps", f"{s.min_step}..{s.max_step}" if s.n_points else "-"),
    ]
    for key, value in rows:
        typer.echo(f"{key}: {value}")
    for a in view.artifacts:
        typer.echo(f"artifact: {a.kind} {a.sha256[:12]} {a.size_bytes} bytes")
    for snap in view.snapshots:
        shown = " ".join(f"{k}={v:.4f}" for k, v in snap.metrics.items())
        typer.echo(
            f"snapshot: {snap.eval_set_hash[:12]} step={snap.step} n_images={snap.n_images} {shown}"
        )


def repro_command(ctx: typer.Context, run_id: RunIdArg, project: ProjectOpt = None) -> None:
    """Print a bash script that reproduces a run (stdout). Exit 1 if the run does not exist."""
    from cvpilot import repro, runs

    view = runs.get_run_view(_project(ctx, project), run_id, store_dir=ctx.obj.store_dir)
    typer.echo(repro.script(view), nl=False)
