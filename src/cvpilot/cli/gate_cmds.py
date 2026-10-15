"""``cvpilot compare`` and ``cvpilot gate``: candidate run against a baseline snapshot.

Exit codes of ``gate``: 0 every requested check ran and held; 1 a threshold was exceeded, a
check could not be made (absent or non-finite metric, absent, too small or differently sliced
slice, too few images), the comparison was refused (different ``eval_set_hash``, task, class
names, evaluation parameters or slice membership, the candidate has no snapshot for the
baseline's evaluation set, the baseline is the candidate or an unfinished run, the candidate was
not produced from ``--require-commit``), a recomputation on trusted ground truth disagrees, the
baseline tag or run is unknown, or any snapshot, artifact or baseline failed integrity
verification (including ``--expect-baseline-chain``); 2 usage error (no threshold given, bad
threshold syntax, malformed tag, run id, hash or number, no project, ``--recompute`` without
``--gt`` and ``--classes``). CI should treat any non-zero code as blocking; exit 1 with
``gate_failed`` is the regression signal. ``compare`` has the same refusals and failures but
never fails on a difference.

Units: limits are in points on the 0..100 scale. ``--max-drop mAP50_95=0.5`` fails when the
candidate's mAP50_95 is more than 0.5 points (0.005 as a fraction) below the baseline's;
improvements never fail. ``--slice-max-drop "camera=*:1.5"`` is ``KEY=VALUE:POINTS`` (``*`` is
every value of KEY). ``--min-images`` (default 30) is the smallest evaluation set and slice the
gate accepts.

Trust: without ``--recompute`` the gate trusts the metrics stored in the snapshots, so it only
protects against accidents. A CI gate should always pass ``--recompute --gt GT --classes FILE``
(ground truth from a trusted place), ``--require-commit`` (the commit under test) and
``--expect-baseline-chain`` (an anchor from a protected variable); see "Trust model for CI gates"
in docs/architecture/06-cross-cutting.md.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated

import typer

from cvpilot.cli.baseline_cmds import eval_set_option
from cvpilot.cli.runs_cmds import JsonOpt, ProjectOpt, RunIdArg, _project
from cvpilot.core.schema import MAX_STEP

_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{7,64}")


def _commit_option(value: str | None) -> str | None:
    if value is not None and _COMMIT.fullmatch(value) is None:
        raise typer.BadParameter("expected 7 to 64 lower-case hex characters")
    return value


def _chain_option(value: str | None) -> str | None:
    if value is not None and _SHA256.fullmatch(value) is None:
        raise typer.BadParameter("expected 64 lower-case hex characters")
    return value


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
RequireCommitOpt = Annotated[
    str | None,
    typer.Option(
        "--require-commit",
        callback=_commit_option,
        help="Refuse a candidate not produced from this commit (7+ hex); eval needs --git-commit.",
    ),
]
ExpectChainOpt = Annotated[
    str | None,
    typer.Option(
        "--expect-baseline-chain",
        callback=_chain_option,
        help="Chain digest of the baseline table kept outside the store (see baseline show).",
    ),
]
ReportOpt = Annotated[
    Path | None,
    typer.Option(
        "--report", dir_okay=False, help="Write a static HTML report (no script, strict CSP)."
    ),
]
ReportJsonOpt = Annotated[
    Path | None,
    typer.Option("--report-json", dir_okay=False, help="Write the canonical JSON report."),
]


def _baseline_ref(value: str) -> tuple[str | None, str | None]:
    from cvpilot.compare.service import parse_baseline_ref

    try:
        return parse_baseline_ref(value)
    except ValueError as exc:
        raise typer.BadParameter(str(exc), param_hint="--baseline") from None


def _anchor_needs_tag(baseline: str, expect_chain: str | None) -> None:
    if expect_chain is not None and _baseline_ref(baseline)[0] is None:
        raise typer.BadParameter(
            "--expect-baseline-chain needs --baseline tag:NAME",
            param_hint="--expect-baseline-chain",
        )


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
    require_commit: RequireCommitOpt = None,
    expect_baseline_chain: ExpectChainOpt = None,
    report: ReportOpt = None,
    report_json_path: ReportJsonOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Compare a run with a baseline (deltas, intervals, worst classes, slices, confusion)."""
    from cvpilot.compare import service
    from cvpilot.compare.html_report import render_html, write_report
    from cvpilot.compare.report import report_json

    _baseline_ref(baseline)
    cmp = service.run_compare(
        _project(ctx, project), run_id, baseline, eval_set=eval_set, step=step,
        store_dir=ctx.obj.store_dir, require_commit=require_commit,
        expect_chain=expect_baseline_chain,
    )  # fmt: skip
    if report is not None:
        write_report(report, render_html(cmp))
    if report_json_path is not None:
        write_report(report_json_path, report_json(cmp))
    if as_json:
        typer.echo(report_json(cmp))
    else:
        from cvpilot.compare.render_text import compare_lines

        for line in compare_lines(cmp):
            typer.echo(line)
    service.log_compare(cmp)


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
            help="List slices below --min-images as skipped instead of failing them (see "
            "--max-skipped).",
        ),
    ] = False,
    max_skipped: Annotated[
        int,
        typer.Option(
            "--max-skipped", min=0, help="Most slices --skip-small-slices may skip (default 0)."
        ),
    ] = 0,
    recompute: Annotated[
        bool,
        typer.Option(
            "--recompute",
            help="Re-evaluate the stored predictions on --gt and fail unless every stored metric "
            "matches. Use it in CI: without it the gate trusts whatever the store says.",
        ),
    ] = False,
    gt: Annotated[
        Path | None,
        typer.Option("--gt", dir_okay=False, help="Trusted ground-truth Parquet for --recompute."),
    ] = None,
    classes: Annotated[
        Path | None,
        typer.Option("--classes", dir_okay=False, help="Class names file for --recompute."),
    ] = None,
    require_commit: RequireCommitOpt = None,
    expect_baseline_chain: ExpectChainOpt = None,
    report: ReportOpt = None,
    report_json_path: ReportJsonOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """Gate a run against a baseline; exit 1 on a regression or any unverifiable check."""
    from cvpilot.compare import service
    from cvpilot.compare.gate import ThresholdError, Thresholds, parse_drop, parse_slice_drop
    from cvpilot.compare.html_report import render_html, write_report
    from cvpilot.compare.render_text import gate_lines
    from cvpilot.compare.report import report_json

    _baseline_ref(baseline)
    _anchor_needs_tag(baseline, expect_baseline_chain)
    if recompute != (gt is not None) or recompute != (classes is not None):
        raise typer.BadParameter(
            "--recompute, --gt and --classes must be given together", param_hint="--recompute"
        )
    try:
        thresholds = Thresholds(
            max_drop=_drops(max_drop, parse_drop, "--max-drop"),
            per_class_max_drop=_drops(per_class_max_drop, parse_drop, "--per-class-max-drop"),
            slice_max_drop=_drops(slice_max_drop, parse_slice_drop, "--slice-max-drop"),
            min_images=min_images,
            skip_small_slices=skip_small_slices,
            max_skipped=max_skipped,
        )
    except ThresholdError as exc:
        raise typer.BadParameter(str(exc)) from None
    spec = None
    if recompute and gt is not None and classes is not None:
        from cvpilot.cli.eval_cmds import DEFAULT_MAX_MEMORY_MB, apply_memory_limit
        from cvpilot.compare.recompute import RecomputeSpec
        from cvpilot.eval.files import read_class_names

        apply_memory_limit(DEFAULT_MAX_MEMORY_MB)
        spec = RecomputeSpec(gt, read_class_names(classes))
    outcome = service.run_gate(
        _project(ctx, project), run_id, baseline, thresholds, eval_set=eval_set, step=step,
        store_dir=ctx.obj.store_dir, require_commit=require_commit,
        expect_chain=expect_baseline_chain, recompute=spec,
    )  # fmt: skip
    if report is not None:  # a report that cannot be written fails the command, pass or not
        write_report(report, render_html(outcome.comparison, outcome.result, thresholds))
    if report_json_path is not None:
        write_report(report_json_path, report_json(outcome.comparison, outcome.result, thresholds))
    if as_json:
        typer.echo(report_json(outcome.comparison, outcome.result, thresholds))
    else:
        for line in gate_lines(outcome):
            typer.echo(line)
    service.log_gate(outcome)  # the verdict is logged only after everything above succeeded
    if not outcome.result.passed:
        raise typer.Exit(1)
