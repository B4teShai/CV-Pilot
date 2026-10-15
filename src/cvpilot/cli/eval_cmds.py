"""``cvpilot eval``: evaluate Parquet predictions against ground truth into a run.

Exit codes: 0 ok; 1 the run is not found, the inputs fail validation (non-finite scores, class
ids out of range, mismatched image sets, boxes outside the image, slice cardinality, malformed
or oversized Parquet), the snapshot already exists or the store cannot be written; 2 usage or
configuration error (no project, malformed run id, missing file, bad ``--slices`` syntax or a
number out of range). Snapshots are immutable: repeating an evaluation of the same set at the
same ``--step`` fails. File formats: see ``cvpilot.eval.files``.
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from cvpilot.cli.runs_cmds import JsonOpt, ProjectOpt, RunIdArg, _project
from cvpilot.core.schema import MAX_RESAMPLES, MAX_STEP, EvalSnapshot

DEFAULT_MAX_MEMORY_MB = 8192
"""Default address-space ceiling of ``cvpilot eval`` (8 GiB); ``0`` turns it off."""


def apply_memory_limit(
    megabytes: int,
    *,
    platform: str = sys.platform,
    setrlimit: Callable[[int, tuple[int, int]], None] | None = None,
    getrlimit: Callable[[int], tuple[int, int]] | None = None,
) -> bool:
    """Lower ``RLIMIT_AS`` of this process to ``megabytes`` where the OS enforces it (Linux).

    Returns True when a limit was set. On other platforms, and for ``0``, nothing is set:
    inputs are still bounded by the file, row and decoded-size limits, but the process is not
    capped. A hard limit that is already lower stays; a failure to set the limit is ignored.
    """
    if megabytes <= 0 or not platform.startswith("linux"):
        return False
    import resource

    get = getrlimit or resource.getrlimit
    put = setrlimit or resource.setrlimit
    wanted = megabytes * 1024 * 1024
    soft, hard = get(resource.RLIMIT_AS)
    if soft != resource.RLIM_INFINITY and soft <= wanted:
        return False
    try:
        put(resource.RLIMIT_AS, (wanted, hard))
    except (ValueError, OSError):
        return False
    return True


_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{7,64}")


def _parse_slice(value: str) -> str:
    key, sep, column = value.partition("=")
    if not sep or not key or not column or "=" in column:
        raise typer.BadParameter("expected KEY=COLUMN, for example camera=camera_id")
    if not key.isascii() or _KEY.fullmatch(key) is None:
        raise typer.BadParameter("KEY must be ASCII: a letter, then letters, digits or _")
    return value


def _slices(values: list[str] | None) -> list[str]:
    parsed = [_parse_slice(v) for v in values or []]
    keys = [v.split("=", 1)[0] for v in parsed]
    if len(set(keys)) != len(keys):
        raise typer.BadParameter("each slice key may be given once")
    return parsed


def _sha(value: str | None) -> str | None:
    if value is not None and _SHA256.fullmatch(value) is None:
        raise typer.BadParameter("expected 64 lower-case hex characters")
    return value


def _commit(value: str | None) -> str | None:
    if value is not None and _COMMIT.fullmatch(value) is None:
        raise typer.BadParameter("expected 7 to 64 lower-case hex characters")
    return value


def _headline(snapshot: EvalSnapshot) -> list[str]:
    lines = [
        f"run_id: {snapshot.run_id}",
        f"eval_set_hash: {snapshot.eval_set_hash}",
        f"step: {snapshot.step}",
        f"n_images: {snapshot.n_images}",
    ]
    for name, value in snapshot.metrics.items():
        interval = snapshot.ci.get(name)
        suffix = f" [{interval[0]:.4f}, {interval[1]:.4f}]" if interval else ""
        lines.append(f"{name}: {value:.4f}{suffix}")
    for label, row in snapshot.slices.items():
        shown = " ".join(
            f"{k}={v:.4f}" if k != "n_images" else f"{k}={int(v)}" for k, v in row.items()
        )
        lines.append(f"slice {label}: {shown}")
    return lines


def eval_command(
    ctx: typer.Context,
    run_id: RunIdArg,
    preds: Annotated[
        Path,
        typer.Option(
            "--preds", "--predictions", exists=True, dir_okay=False, help="Predictions Parquet."
        ),
    ],
    gts: Annotated[
        Path,
        typer.Option(
            "--gts", "--ground-truth", exists=True, dir_okay=False, help="Ground-truth Parquet."
        ),
    ],
    classes: Annotated[
        Path,
        typer.Option(
            "--classes", exists=True, dir_okay=False, help="Class names: JSON list or lines."
        ),
    ],
    project: ProjectOpt = None,
    step: Annotated[
        int,
        typer.Option("--step", min=0, max=MAX_STEP, help="Training step of the predictions."),
    ] = 0,
    slices: Annotated[
        list[str] | None,
        typer.Option(
            "--slices",
            callback=_slices,
            help="KEY=COLUMN of a ground-truth metadata column; repeatable.",
        ),
    ] = None,
    seed: Annotated[int, typer.Option("--seed", min=0, max=2**64 - 1, help="Bootstrap seed.")] = 0,
    resamples: Annotated[
        int,
        typer.Option(
            "--resamples", min=0, max=MAX_RESAMPLES, help="Bootstrap resamples (0 = off)."
        ),
    ] = 1000,
    max_memory_mb: Annotated[
        int,
        typer.Option(
            "--max-memory-mb", min=0, help="Address-space ceiling in MB (Linux; 0 = off)."
        ),
    ] = DEFAULT_MAX_MEMORY_MB,
    allow_closed: Annotated[
        bool, typer.Option("--allow-closed", help="Allow finished, failed or preempted runs.")
    ] = False,
    checkpoint_sha256: Annotated[
        str | None,
        typer.Option("--checkpoint-sha256", callback=_sha, help="SHA-256 of the checkpoint."),
    ] = None,
    git_commit: Annotated[
        str | None, typer.Option("--git-commit", callback=_commit, help="Commit of the code.")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Evaluate Parquet files into a run and store an immutable snapshot."""
    apply_memory_limit(max_memory_mb)
    from cvpilot.eval.files import eval_file, read_class_names

    spec = dict(s.split("=", 1) for s in slices or [])
    snapshot = eval_file(
        _project(ctx, project),
        run_id,
        predictions=preds,
        ground_truth=gts,
        class_names=read_class_names(classes),
        step=step,
        slices=spec or None,
        n_resamples=resamples,
        seed=seed,
        allow_closed=allow_closed,
        provenance={"checkpoint_sha256": checkpoint_sha256, "git_commit": git_commit},
        store_dir=ctx.obj.store_dir,
    )
    if as_json:
        typer.echo(json.dumps(json.loads(snapshot.model_dump_json())))
        return
    for line in _headline(snapshot):
        typer.echo(line)
