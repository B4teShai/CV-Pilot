"""``cvpilot ckpt list|verify|prune``: look at, check and trim a checkpoint directory.

Exit codes: 0 ok; 1 a checkpoint failed verification (or none verifies), the directory is locked
by a running saver, or a file could not be read; 2 usage or configuration error (the path is not
a checkpoint directory, an invalid run id, ``--require-trust authenticated`` without
``CVPILOT_CKPT_KEY``).

``--project`` (default ``CVPILOT_PROJECT``) also checks each manifest against the digest the run
store recorded; ``--require-trust`` (default ``CVPILOT_CKPT_REQUIRE_TRUST``, ``any``) decides how
much proof a checkpoint needs: ``authenticated`` (HMAC tag, key from ``CVPILOT_CKPT_KEY``),
``anchored`` (run-store digest) or ``any`` (the directory's own chain: fine for a private
directory, not for one other people can write).
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Any

import typer

from cvpilot.cli.runs_cmds import run_id_argument
from cvpilot.core.errors import CheckpointError
from cvpilot.core.text import safe_text
from cvpilot.logging import get_logger
from cvpilot.settings import Settings

ckpt_app = typer.Typer(help="List, verify and prune checkpoints.", no_args_is_help=True)


class KindOption(str, Enum):
    last = "last"
    best = "best"
    all = "all"


class TrustOption(str, Enum):
    any = "any"
    anchored = "anchored"
    authenticated = "authenticated"


class ModeOption(str, Enum):
    max = "max"
    min = "min"


DirArg = Annotated[Path, typer.Argument(help="Checkpoint directory (holds one folder per run).")]
RunOpt = Annotated[
    str | None,
    typer.Option("--run", callback=lambda v: None if v is None else run_id_argument(v)),
]
KindOpt = Annotated[KindOption, typer.Option("--kind", help="Which checkpoints.")]
ProjectOpt = Annotated[
    str | None, typer.Option("--project", help="Project whose run store anchors the digests.")
]
TrustOpt = Annotated[
    TrustOption | None,
    typer.Option("--require-trust", help="any, anchored or authenticated (see the help above)."),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")]


def _usage(message: str) -> typer.Exit:
    typer.echo(f"error: {safe_text(message, 300)}", err=True)
    return typer.Exit(2)


def _root(directory: Path) -> Path:
    path = directory.expanduser()
    if not path.is_dir():
        raise _usage(f"{directory} is not a checkpoint directory")
    return path


def _kinds(kind: KindOption) -> list[Any]:
    return ["last", "best"] if kind is KindOption.all else [kind.value]


class _Session:
    """Opens one ``RunCheckpoints`` per run with the key, policy and store anchor in force."""

    def __init__(
        self,
        ctx: typer.Context,
        root: Path,
        project: str | None,
        trust: TrustOption | None,
        min_step: int | None = None,
        accept_rollback_to: int | None = None,
    ) -> None:
        from cvpilot import runs
        from cvpilot.ckpt import auth
        from cvpilot.ckpt.directory import AdmitContext

        settings: Settings = ctx.obj
        self.root = root
        try:
            self.key = auth.load_key()
        except CheckpointError as exc:
            raise _usage(str(exc)) from None
        weak = None if self.key is None else auth.weakness(self.key)
        if weak is not None:
            get_logger("cvpilot.cli").warning("checkpoint_key_weak", reason=weak)
        asked = trust.value if trust else None
        self.policy: Any = auth.resolve_policy(asked, settings.ckpt_require_trust, self.key)
        if asked is not None and auth.is_weaker(asked, self.policy):
            get_logger("cvpilot.cli").warning(
                "checkpoint_trust_downgrade_ignored", requested=asked, effective=self.policy
            )
        floors = [m for m in (min_step, settings.ckpt_min_step) if m is not None]
        self.admit = AdmitContext(
            min_step=max(floors) if floors else None, accept_rollback_to=accept_rollback_to
        )  # both floors bind; accepting a rollback cannot lower the operator's
        name = project or settings.project
        self._opened = runs.open_anchor(name, settings.store_dir) if name else None
        self.anchor = self._opened[0] if self._opened else None

    def close(self) -> None:
        if self._opened is not None:
            self._opened[1]()

    def open(self, run_id: str) -> Any:
        from cvpilot.ckpt.directory import RunCheckpoints

        try:
            return RunCheckpoints(
                self.root, run_id, anchor=self.anchor, key=self.key, require_trust=self.policy
            )
        except CheckpointError as exc:
            raise _usage(str(exc)) from None


def _run_ids(root: Path, run: str | None) -> list[str]:
    from cvpilot.ckpt.directory import list_runs

    return [run] if run else list_runs(root)


def _row(run_id: str, entry: Any) -> dict[str, Any]:
    manifest = entry.manifest
    return {
        "run_id": run_id,
        "kind": entry.kind,
        "step": entry.step,
        "name": safe_text(entry.name, 80),
        "status": entry.status,
        "reason": entry.reason,
        "trust": entry.trust,
        "size_bytes": None if manifest is None else manifest.size_bytes,
        "sha256": None if manifest is None else manifest.sha256,
        "format": None if manifest is None else manifest.format,
        "created_at": None if manifest is None else manifest.created_at.isoformat(),
        "chain_reset": False if manifest is None else manifest.chain_reset,
    }


def _failed_row(run_id: str, kind: str, reason: str, policy: str) -> dict[str, Any]:
    return {
        "run_id": run_id, "kind": kind, "step": None, "name": run_id, "status": "corrupt",
        "reason": reason, "trust": None, "size_bytes": None, "sha256": None, "format": None,
        "created_at": None, "policy": policy, "chain_reset": False,
    }  # fmt: skip


def _line(row: dict[str, Any]) -> str:
    step = "-" if row["step"] is None else f"{row['step']:09d}"
    detail = row["trust"] or row["reason"] or ""
    return f"{row['run_id']}  {row['kind']:<4}  {step:>9}  {row['status']:<7}  {detail}"


def _collect(
    ctx: typer.Context,
    directory: Path,
    run: str | None,
    kind: KindOption,
    project: str | None,
    trust: TrustOption | None,
    *,
    check_data: bool,
    min_step: int | None = None,
    accept_rollback_to: int | None = None,
) -> list[dict[str, Any]]:
    root = _root(directory)
    session = _Session(ctx, root, project, trust, min_step, accept_rollback_to)
    try:
        rows: list[dict[str, Any]] = []
        try:
            run_ids = _run_ids(root, run)
        except (
            CheckpointError
        ) as exc:  # the root of the file system, a directory that cannot be read
            raise _usage(str(exc)) from None
        for run_id in run_ids:
            found = session.open(run_id)
            for one in _kinds(kind):
                try:
                    entries = found.inspect(one, check_data=check_data, ctx=session.admit)
                except CheckpointError as exc:
                    # a hostile or unreadable entry (a symlink or a file named like a run, a
                    # directory nobody may read) is a corrupt row; the rest is still examined
                    reason = getattr(exc, "reason", None) or "unreadable"
                    rows.append(_failed_row(run_id, one, reason, session.policy))
                    continue
                rows += [{**_row(run_id, e), "policy": session.policy} for e in entries]
        return rows
    finally:
        session.close()


@ckpt_app.command("list")
def ckpt_list(
    ctx: typer.Context,
    directory: DirArg,
    run: RunOpt = None,
    kind: KindOpt = KindOption.all,
    project: ProjectOpt = None,
    require_trust: TrustOpt = None,
    as_json: JsonOpt = False,
) -> None:
    """List the checkpoints of every run (or one run), newest first, judging manifests only.

    Nothing is hashed: use ``verify`` for that. Exit 0 (an empty directory is not an error);
    exit 2 when the path is not a directory.
    """
    rows = _collect(ctx, directory, run, kind, project, require_trust, check_data=False)
    if as_json:
        typer.echo(json.dumps(rows))
        return
    for row in rows:
        typer.echo(_line(row))


@ckpt_app.command("verify")
def ckpt_verify(
    ctx: typer.Context,
    directory: DirArg,
    run: RunOpt = None,
    kind: KindOpt = KindOption.all,
    project: ProjectOpt = None,
    require_trust: TrustOpt = None,
    expect_min_step: Annotated[
        int | None,
        typer.Option(
            "--expect-min-step",
            min=0,
            help="A floor you know from elsewhere: older checkpoints fail (rolled_back).",
        ),
    ] = None,
    accept_rollback_to: Annotated[
        int | None,
        typer.Option(
            "--accept-rollback-to",
            min=0,
            help="Accept a rollback from the run store or pointer down to this step, "
            "on purpose (logged); it never lowers --expect-min-step or CVPILOT_CKPT_MIN_STEP.",
        ),
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Verify every checkpoint: size, structure, SHA-256, chain and the trust policy.

    The newest step known to have existed (run store, a tag-checked pointer,
    ``CVPILOT_CKPT_MIN_STEP``, ``--expect-min-step``) is a floor: a checkpoint older than it
    is reported as ``rolled_back`` unless you accept that with ``--accept-rollback-to``.

    Prints one line per checkpoint on stdout. Exit 0 when everything verifies; exit 1 when a
    checkpoint is corrupt, truncated, forged or untrusted, or when nothing verifies at all;
    exit 2 when the path is not a directory or the options are invalid.
    """
    from cvpilot.ckpt.directory import TRUST_REASONS

    rows = _collect(
        ctx, directory, run, kind, project, require_trust, check_data=True,
        min_step=expect_min_step, accept_rollback_to=accept_rollback_to,
    )  # fmt: skip
    log = get_logger("cvpilot.cli")
    bad = 0
    for row in rows:
        fields = {k: row[k] for k in ("run_id", "kind", "step")}
        if row["status"] == "valid":
            log.info("checkpoint_verified", trust_level=row["trust"], **fields)
        else:
            bad += 1
            event = (
                "checkpoint_untrusted"
                if row["reason"] in TRUST_REASONS
                else "checkpoint_verify_failed"
            )
            log.warning(event, reason=row["reason"], **fields)
    if as_json:
        typer.echo(json.dumps(rows))
    else:
        for row in rows:
            typer.echo(_line(row))
    if not rows:
        typer.echo("error: no checkpoints found", err=True)
        raise typer.Exit(1)
    if bad or not any(r["status"] == "valid" for r in rows):
        raise typer.Exit(1)


@ckpt_app.command("quarantine")
def ckpt_quarantine(
    ctx: typer.Context,
    directory: DirArg,
    run: Annotated[
        str, typer.Option("--run", callback=run_id_argument, help="Run whose quarantine to show.")
    ],
    clear: Annotated[
        bool,
        typer.Option(
            "--clear",
            help="Delete the quarantined entries (you are saying they are not yours); the step "
            "floor they imposed falls away.",
        ),
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Show what a save set aside, or discard it with ``--clear``.

    A save quarantines a manifest it does not trust whose step is above every trusted one (a
    planted high step would otherwise block every later save). Quarantined entries keep counting
    toward the step floor until you clear them. Exit 0; exit 2 on a usage error.
    """
    root = _root(directory)
    session = _Session(ctx, root, None, None)
    try:
        found = session.open(run)
        try:
            entries = found.quarantined()
            cleared = found.clear_quarantine() if clear else 0
        except CheckpointError as exc:
            raise _usage(str(exc)) from None
        rows = [{"run_id": run, "kind": e.kind, "step": e.step, "name": e.name} for e in entries]
        if as_json:
            typer.echo(json.dumps({"entries": rows, "cleared": cleared} if clear else rows))
        else:
            for row in rows:
                step = "-" if row["step"] is None else f"{row['step']:09d}"
                typer.echo(f"{run}  {row['kind']:<4}  {step:>9}  quarantined")
            if clear:
                typer.echo(f"cleared {cleared} quarantined manifest(s)")
    finally:
        session.close()


@ckpt_app.command("prune")
def ckpt_prune(
    ctx: typer.Context,
    directory: DirArg,
    run: Annotated[
        str, typer.Option("--run", callback=run_id_argument, help="Run to prune (required).")
    ],
    keep_last: Annotated[int, typer.Option("--keep-last", min=1)] = 3,
    keep_best: Annotated[int, typer.Option("--keep-best", min=1)] = 2,
    mode: Annotated[
        ModeOption | None,
        typer.Option(
            "--mode",
            help="Best means max or min metric; default: the mode the checkpoints were saved with.",
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Only say what would go.")] = False,
    project: ProjectOpt = None,
    require_trust: TrustOpt = None,
) -> None:
    """Delete checkpoints beyond ``--keep-last`` and ``--keep-best``.

    Only checkpoints that verify are counted or deleted, the newest valid checkpoint of each
    kind is never deleted and at least one is always kept (``--keep-*`` of 0 is refused). Exit
    0; exit 1 when a saver holds the directory lock; exit 2 on a usage error.
    """
    root = _root(directory)
    session = _Session(ctx, root, project, require_trust)
    try:
        found = session.open(run)
        verb = "would delete" if dry_run else "deleted"
        for one, keep in (("last", keep_last), ("best", keep_best)):
            try:
                result = found.prune(
                    one, keep=keep, mode=mode.value if mode else None, dry_run=dry_run
                )
            except CheckpointError as exc:
                raise _usage(str(exc)) from None
            for manifest in result.deleted:
                typer.echo(f"{verb} {one} {manifest.step:09d}")
    finally:
        session.close()
