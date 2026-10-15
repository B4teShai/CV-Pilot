"""``cvpilot baseline set|list|show``: pin and inspect the snapshot a gate compares against.

Exit codes: 0 ok; 1 refused (no such snapshot or eval set, tag already pinned without
``--replace``), unknown tag, run not found, or a snapshot or baseline that fails integrity
verification; 2 usage or configuration error (no project, malformed tag, run id or eval set,
step out of range). Pins are append-only: ``--replace`` adds a newer pin and keeps the old one
in the history.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from cvpilot.cli.runs_cmds import JsonOpt, ProjectOpt, RunIdArg, _project
from cvpilot.core.schema import BASELINE_TAG_PATTERN, MAX_STEP, is_valid_tag

baseline_app = typer.Typer(help="Pin and inspect baselines.", no_args_is_help=True)


def tag_argument(value: str) -> str:
    if not is_valid_tag(value):
        raise typer.BadParameter(f"a tag must match {BASELINE_TAG_PATTERN}")
    return value


def eval_set_option(value: str | None) -> str | None:
    if value is None:
        return None
    from cvpilot.compare.baselines import EVAL_SET_ARGUMENT

    if EVAL_SET_ARGUMENT.fullmatch(value) is None:
        raise typer.BadParameter("expected 8 to 64 lower-case hex characters")
    return value


TagArg = Annotated[str, typer.Argument(callback=tag_argument, help="Baseline tag.")]


def _dump(model: Any) -> Any:
    return json.loads(model.model_dump_json())


@baseline_app.command("set")
def baseline_set(
    ctx: typer.Context,
    tag: TagArg,
    run_id: RunIdArg,
    project: ProjectOpt = None,
    eval_set: Annotated[
        str | None,
        typer.Option(
            "--eval-set",
            callback=eval_set_option,
            help="Eval set hash or prefix (8+ hex characters); optional when the run has one.",
        ),
    ] = None,
    step: Annotated[
        int | None,
        typer.Option("--step", min=0, max=MAX_STEP, help="Snapshot step (default: the latest)."),
    ] = None,
    replace: Annotated[
        bool, typer.Option("--replace", help="Pin a tag that is already pinned (keeps history).")
    ] = False,
) -> None:
    """Pin a verified snapshot of RUN_ID under TAG. Exit 1 when refused."""
    from cvpilot.compare import baselines

    pinned = baselines.set_baseline(
        _project(ctx, project),
        tag,
        run_id,
        eval_set=eval_set,
        step=step,
        replace=replace,
        store_dir=ctx.obj.store_dir,
    )
    typer.echo(
        f"pinned {pinned.tag}: run {pinned.run_id} eval_set {pinned.eval_set_hash[:12]} "
        f"step {pinned.step}"
    )


@baseline_app.command("list")
def baseline_list(
    ctx: typer.Context,
    project: ProjectOpt = None,
    history: Annotated[bool, typer.Option("--history", help="Include replaced pins.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """List the verified baselines of a project (exit 0 when there are none)."""
    from cvpilot.compare import baselines

    found = baselines.list_baselines(
        _project(ctx, project), history=history, store_dir=ctx.obj.store_dir
    )
    if as_json:
        typer.echo(json.dumps([_dump(b) for b in found]))
        return
    typer.echo(f"{'TAG':<32}  {'RUN_ID':<26}  {'STEP':>9}  {'EVAL_SET':<12}  PINNED_AT")
    for b in found:
        typer.echo(
            f"{b.tag:<32}  {b.run_id:<26}  {b.step:>9}  {b.eval_set_hash[:12]:<12}  "
            f"{b.pinned_at.isoformat()}"
        )


@baseline_app.command("show")
def baseline_show(
    ctx: typer.Context, tag: TagArg, project: ProjectOpt = None, as_json: JsonOpt = False
) -> None:
    """Show a baseline and its headline metrics, verifying both. Exit 1 for an unknown tag."""
    from cvpilot.compare import baselines

    view = baselines.show_baseline(_project(ctx, project), tag, store_dir=ctx.obj.store_dir)
    if as_json:
        typer.echo(
            json.dumps(
                {
                    "baseline": _dump(view.baseline),
                    "metrics": view.snapshot.metrics,
                    "n_images": view.snapshot.n_images,
                    "history": view.history,
                    "head": {"seq": view.head_seq, "chain_sha256": view.head_chain_sha256},
                }
            )
        )
        return
    b = view.baseline
    for key, value in (
        ("tag", b.tag),
        ("run_id", b.run_id),
        ("eval_set_hash", b.eval_set_hash),
        ("step", str(b.step)),
        ("snapshot_sha256", b.snapshot_sha256),
        ("pinned_at", b.pinned_at.isoformat()),
        ("n_images", str(view.snapshot.n_images)),
        ("history", str(view.history)),
        ("head_seq", str(view.head_seq)),
        ("head_chain_sha256", view.head_chain_sha256),
    ):
        typer.echo(f"{key}: {value}")
    for name, score in view.snapshot.metrics.items():
        typer.echo(f"{name}: {score:.4f}")
