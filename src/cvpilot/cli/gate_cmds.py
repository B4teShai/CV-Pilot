"""``cvpilot compare`` and ``cvpilot gate``: candidate run against a baseline snapshot.

Exit codes of ``gate``: 0 every requested check ran and held; 1 a threshold was exceeded, a
check could not be made (absent or non-finite metric, absent or too small slice, too few
images), the comparison was refused (different ``eval_set_hash``, task, class names or
evaluation parameters, or the candidate has no snapshot for the baseline's evaluation set), the
baseline tag or run is unknown, or any snapshot or baseline failed integrity verification; 2
usage error (no threshold given, bad threshold syntax, malformed tag, run id or number, no
project). CI should treat any non-zero code as blocking; exit 1 with ``gate_failed`` is the
regression signal. ``compare`` has the same refusals and failures but never fails on a
difference.

Units: limits are in points on the 0..100 scale. ``--max-drop mAP50_95=0.5`` fails when the
candidate's mAP50_95 is more than 0.5 points (0.005 as a fraction) below the baseline's;
improvements never fail. ``--slice-max-drop "camera=*:1.5"`` is ``KEY=VALUE:POINTS`` (``*`` is
every value of KEY). ``--min-images`` (default 30) is the smallest evaluation set, and slice,
the gate accepts.
"""

from __future__ import annotations

from typing import Annotated

import typer

from cvpilot.cli.baseline_cmds import eval_set_option
from cvpilot.cli.runs_cmds import JsonOpt, ProjectOpt, RunIdArg, _project
from cvpilot.core.schema import MAX_STEP

BaselineOpt = Annotated[
    str, typer.Option("--baseline", help="tag:NAME of a pinned baseline, or a baseline run id.")
]
EvalSetOpt = Annotated[
    str | None,
    typer.Option("--eval-set", callback=eval_set_option, help="Eval set hash or prefix (8+ hex)."),
]
StepOpt = Annotated[
    int | None,
    typer.Option("--step", min=0, max=MAX_STEP, help="Candidate snapshot step (default: latest)."),
]


def _baseline_ref(value: str) -> str:
    from cvpilot.compare.service import parse_baseline_ref

    try:
        parse_baseline_ref(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--baseline") from None
    return value


def _drops(values: list[str] | None, parse: object, hint: str) -> dict[str, float]:
    from cvpilot.compare.gate import ThresholdError

    out: dict[str, float] = {}
    for text in values or []:
        try:
            name, limit = parse(text)  # type: ignore[operator]
        except ThresholdError as exc:
            raise typer.BadParameter(str(exc), param_hint=hint) from None
        if name in out:
            raise typer.BadParameter(f"{name!r} is given twice", param_hint=hint)
        out[name] = limit
    return out


def compare_command(
    ctx: typer.Context,
    run_id: RunIdArg,
    baseline: BaselineOpt,
    project: ProjectOpt = None,
    eval_set: EvalSetOpt = None,
    step: StepOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Compare a run with a baseline (deltas, intervals, worst classes, slices, confusion)."""
    from cvpilot.compare import service
    from cvpilot.compare.report import report_json

    _baseline_ref(baseline)
    cmp = service.run_compare(
        _project(ctx, project), run_id, baseline, eval_set=eval_set, step=step,
        store_dir=ctx.obj.store_dir,
    )  # fmt: skip
    if as_json:
        typer.echo(report_json(cmp))
        return
    from cvpilot.compare.render_text import compare_lines

    for line in compare_lines(cmp):
        typer.echo(line)


def gate_command(
    ctx: typer.Context,
    run_id: RunIdArg,
    baseline: BaselineOpt,
    project: ProjectOpt = None,
    eval_set: EvalSetOpt = None,
    step: StepOpt = None,
    max_drop: Annotated[
        list[str] | None,
        typer.Option("--max-drop", help="NAME=POINTS: largest allowed drop of a headline metric."),
    ] = None,
    per_class_max_drop: Annotated[
        list[str] | None,
        typer.Option("--per-class-max-drop", help="NAME=POINTS: largest drop of any class."),
    ] = None,
    slice_max_drop: Annotated[
        list[str] | None,
        typer.Option(
            "--slice-max-drop", help='"KEY=VALUE:POINTS" or "KEY=*:POINTS" (every value).'
        ),
    ] = None,
    min_images: Annotated[
        int,
        typer.Option("--min-images", min=1, help="Fewest images in the eval set and in a slice."),
    ] = 30,
    skip_small_slices: Annotated[
        bool,
        typer.Option(
            "--skip-small-slices",
            help="List slices below --min-images as skipped instead of failing them.",
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Gate a run against a baseline; exit 1 on a regression or any unverifiable check."""
    from cvpilot.compare import service
    from cvpilot.compare.gate import ThresholdError, Thresholds, parse_drop, parse_slice_drop
    from cvpilot.compare.render_text import gate_lines
    from cvpilot.compare.report import report_json

    _baseline_ref(baseline)
    try:
        thresholds = Thresholds(
            max_drop=_drops(max_drop, parse_drop, "--max-drop"),
            per_class_max_drop=_drops(per_class_max_drop, parse_drop, "--per-class-max-drop"),
            slice_max_drop=_drops(slice_max_drop, parse_slice_drop, "--slice-max-drop"),
            min_images=min_images,
            skip_small_slices=skip_small_slices,
        )
    except ThresholdError as exc:
        raise typer.BadParameter(str(exc)) from None
    outcome = service.run_gate(
        _project(ctx, project), run_id, baseline, thresholds, eval_set=eval_set, step=step,
        store_dir=ctx.obj.store_dir,
    )  # fmt: skip
    if as_json:
        typer.echo(report_json(outcome.comparison, outcome.result, thresholds))
    else:
        for line in gate_lines(outcome):
            typer.echo(line)
    if not outcome.result.passed:
        raise typer.Exit(1)
