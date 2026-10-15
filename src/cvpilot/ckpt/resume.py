"""Find the checkpoint a restarted job should resume from: the newest one that verifies.

``resume_path`` returns a path to a verified ``last`` checkpoint file (or ``None``), so a
framework is never handed the string ``"last"`` or a file that merely exists. The search covers
the project's runs that can be resumed (``preempted`` and ``failed``, plus ``running`` ones whose
process died; those need ``cvpilot.runs.recover_run`` before ``init(resume_from=...)``), newest
first, and takes the first run that has a checkpoint passing every check of ``ckpt/directory``
at the trust level the policy requires. A ``pt`` file is returned only when that format was opted
into, the framework is new enough and the archive passed the strict scan; safetensors files
verify by size, structure and SHA-256. The returned path is read-only inside a private directory,
but a file can still be replaced between this check and a framework opening it: restrict write
access to the checkpoint directory (see the threat model).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from cvpilot import runs
from cvpilot.ckpt import auth, serializers
from cvpilot.ckpt.directory import AdmitContext, RunCheckpoints, Skipped, gate_failure
from cvpilot.core.errors import CheckpointCorruptError, CheckpointUntrustedError
from cvpilot.core.schema import (
    CheckpointKind,
    CheckpointManifest,
    RequireTrust,
    RunMeta,
    TrustLevel,
)
from cvpilot.logging import get_logger
from cvpilot.settings import load_settings

RESUMABLE = ("preempted", "failed", "running", "resumed")
LIVE = ("running", "resumed")
DEFAULT_ALIVE_WITHIN_S = 900.0


@dataclass(frozen=True)
class ResumePoint:
    """A verified checkpoint to resume from and where it came from."""

    run_id: str
    """The run to pass as ``resume_from``: the newest run of the lineage."""
    path: Path
    manifest: CheckpointManifest
    trust: TrustLevel
    status: str
    needs_recover: bool
    """True when the run is still ``running`` or ``resumed`` (its process died): mark it failed
    with ``cvpilot.runs.recover_run`` before passing it as ``resume_from``."""
    checkpoint_run_id: str = ""
    """The run whose directory holds the file (an ancestor when ``run_id`` crashed before it
    saved anything)."""
    reason: str = ""


def _alive(
    project: str, root: Path, store_root: Path | None, run_id: str, alive_within_s: float
) -> bool:
    """Is a process still working on this run? The liveness lease is exact (the kernel drops it
    when the process dies); without one, activity within ``alive_within_s`` seconds counts."""
    probe = RunCheckpoints(root, run_id)
    state = probe.lease_state()
    if state is not None:
        return state
    seen = [probe.last_activity(), runs.last_metric_activity(project, run_id, store_root)]
    latest = max((t for t in seen if t is not None), default=None)
    return latest is not None and time.time() - latest < alive_within_s


def _lineage(leaf: RunMeta, by_id: dict[str, RunMeta]) -> list[RunMeta]:
    chain = [leaf]
    seen = {leaf.run_id}
    while chain[-1].parent_run_id is not None and len(chain) < 64:
        parent = by_id.get(chain[-1].parent_run_id or "")
        if parent is None or parent.run_id in seen:
            break
        chain.append(parent)
        seen.add(parent.run_id)
    return chain


def find_resume_point(
    project: str,
    *,
    checkpoint_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
    require_trust: RequireTrust | None = None,
    kind: CheckpointKind = "last",
    min_step: int | None = None,
    accept_rollback_to: int | None = None,
    name: str | None = None,
    alive_within_s: float = DEFAULT_ALIVE_WITHIN_S,
    force: bool = False,
) -> ResumePoint | None:
    """The newest verified checkpoint of ``kind`` among the project's resumable lineages.

    A lineage is a run and the runs it resumed. Only its newest run (the one nobody resumed) is a
    starting point, so a crashed resumed run (status ``resumed``) is chosen over its parent and a
    lineage whose newest run finished is done. The checkpoint is searched in the newest run
    first and then in its ancestors; the returned ``run_id`` is the run to ``resume_from`` even
    when the file belongs to an ancestor.

    **The search is project-wide.** ``name`` restricts it to runs of one job (``init(name=...)``);
    without it, any resumable run of the project qualifies, so two different jobs in one project
    can pick up each other's weights. A run that a live process is still working on is never
    chosen (and so never recovered) unless ``force``: liveness is the run's lease in the
    checkpoint directory, or, for runs without one, activity within ``alive_within_s``.

    The policy is ``require_trust``, else ``CVPILOT_CKPT_REQUIRE_TRUST``, else ``authenticated``
    when ``CVPILOT_CKPT_KEY`` is set, else ``any``. Under the strict policies a point older than
    the tag-checked pointer, the run store or ``min_step`` (``CVPILOT_CKPT_MIN_STEP``) is a
    rollback and raises; ``accept_rollback_to`` accepts it on purpose (see
    ``CheckpointManager.restore``).

    Raises:
        ValueError: no checkpoint directory is given or configured.
        CheckpointUntrustedError: checkpoints exist but none reaches the required trust level, or
            the newest is a rollback or the run store cannot be consulted.
        CheckpointError: ``require_trust='authenticated'`` without ``CVPILOT_CKPT_KEY``.
    """
    settings = load_settings()
    root = Path(checkpoint_dir) if checkpoint_dir is not None else settings.checkpoint_dir
    if root is None:
        raise ValueError(
            "a checkpoint directory is required: pass it or set CVPILOT_CHECKPOINT_DIR"
        )
    store_root = Path(store_dir) if store_dir is not None else settings.store_dir
    key = auth.load_key()
    policy: RequireTrust = auth.resolve_policy(  # type: ignore[assignment]
        require_trust, settings.ckpt_require_trust, key
    )
    mark = min_step if min_step is not None else settings.ckpt_min_step
    ctx = AdmitContext(accept_rollback_to=accept_rollback_to, min_step=mark)
    log = get_logger("cvpilot.ckpt")
    opened = runs.open_anchor(project, store_root)
    anchor = opened[0] if opened is not None else None
    rejected: list[Skipped] = []
    try:
        metas = runs.list_runs(project, store_dir=store_root)
        by_id = {m.run_id: m for m in metas}
        parents = {m.parent_run_id for m in metas if m.parent_run_id is not None}
        leaves = sorted(
            (m for m in metas if m.run_id not in parents), key=lambda m: m.run_id, reverse=True
        )
        for leaf in leaves:
            if leaf.status not in RESUMABLE or (name is not None and leaf.name != name):
                continue
            if (
                leaf.status in LIVE
                and not force
                and _alive(project, root, store_root, leaf.run_id, alive_within_s)
            ):
                log.info("resume_point_skipped_alive", run_id=leaf.run_id, status=leaf.status)
                continue
            for owner in _lineage(leaf, by_id):
                found = RunCheckpoints(
                    root, owner.run_id, anchor=anchor, key=key, require_trust=policy,
                    allow_pt=settings.ckpt_allow_pt,
                )  # fmt: skip
                point = _first_usable(found, owner.run_id, leaf, kind, root, ctx)
                rejected += found.skipped
                if point is not None:
                    log.info(
                        "resume_point_selected", run_id=point.run_id,
                        checkpoint_run_id=point.checkpoint_run_id, step=point.manifest.step,
                        status=point.status, trust_level=point.trust, reason=point.reason,
                    )  # fmt: skip
                    return point
                failure = gate_failure(found.skipped, policy)
                if failure is not None and failure[0] in ("rolled_back", "anchor_error"):
                    raise CheckpointUntrustedError(failure[1], reason=failure[0])
    finally:
        if opened is not None:
            opened[1]()
    if mark is not None and accept_rollback_to != 0:
        rejected.append(Skipped(None, kind, kind, "rolled_back"))
    failure = gate_failure(rejected, policy)
    if failure is not None:
        raise CheckpointUntrustedError(failure[1], reason=failure[0])
    return None


def _first_usable(
    found: RunCheckpoints,
    owner_id: str,
    leaf: RunMeta,
    kind: CheckpointKind,
    root: Path,
    ctx: AdmitContext,
) -> ResumePoint | None:
    for verified in found.iter_valid(kind, ctx=ctx):
        manifest = verified.manifest
        if manifest.format == "pt":
            try:
                _check_pt(found, manifest)
            except CheckpointCorruptError as exc:
                found.skipped.append(Skipped(manifest.step, kind, manifest.path, exc.reason))
                continue
        path = Path(root).expanduser().resolve() / owner_id / manifest.path
        inherited = owner_id != leaf.run_id
        return ResumePoint(
            leaf.run_id, path, manifest, verified.trust, leaf.status, leaf.status in LIVE,
            owner_id,
            "checkpoint inherited from an ancestor run" if inherited else "newest resumable run",
        )  # fmt: skip
    return None


def _check_pt(found: RunCheckpoints, manifest: CheckpointManifest) -> None:
    """A pt file goes to a framework's own loader: require a safe version and a clean archive."""
    version = serializers.framework_version()
    if version is None or version < serializers.MIN_PT_VERSION:
        raise CheckpointCorruptError(
            "the framework is too old for pt files", reason="unsafe_loader"
        )
    data = found.read_range(manifest, 0, manifest.size_bytes)
    if len(data) != manifest.size_bytes:
        raise CheckpointCorruptError("the file changed while read", reason="size_mismatch")
    serializers.scan_pt_archive(data)


@overload
def resume_path(
    project: str,
    *,
    checkpoint_dir: str | Path | None = ...,
    store_dir: str | Path | None = ...,
    require_trust: RequireTrust | None = ...,
    accept_rollback_to: int | None = ...,
    with_trust: Literal[False] = ...,
) -> Path | None: ...


@overload
def resume_path(
    project: str,
    *,
    checkpoint_dir: str | Path | None = ...,
    store_dir: str | Path | None = ...,
    require_trust: RequireTrust | None = ...,
    accept_rollback_to: int | None = ...,
    with_trust: Literal[True],
) -> ResumePoint | None: ...


def resume_path(
    project: str,
    *,
    checkpoint_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
    require_trust: RequireTrust | None = None,
    accept_rollback_to: int | None = None,
    with_trust: bool = False,
) -> Path | ResumePoint | None:
    """The path of the newest verified ``last`` checkpoint to resume from, or ``None``.

    The return type is unchanged by default (a ``Path``); with ``with_trust=True`` the whole
    ``ResumePoint`` comes back, so the caller sees the trust level that was achieved
    (``ResumePoint.trust``), the run and whether it needs recovery. See ``find_resume_point`` for
    the search, the trust policy and the errors.
    """
    point = find_resume_point(
        project,
        checkpoint_dir=checkpoint_dir,
        store_dir=store_dir,
        require_trust=require_trust,
        accept_rollback_to=accept_rollback_to,
    )
    if point is None:
        return None
    return point if with_trust else point.path
