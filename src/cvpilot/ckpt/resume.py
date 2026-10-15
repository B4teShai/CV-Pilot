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

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

from cvpilot import runs
from cvpilot.ckpt import auth, serializers
from cvpilot.ckpt.directory import TRUST_REASONS, RunCheckpoints, Skipped
from cvpilot.core.errors import CheckpointCorruptError, CheckpointUntrustedError
from cvpilot.core.schema import CheckpointKind, CheckpointManifest, RequireTrust, TrustLevel
from cvpilot.settings import load_settings

RESUMABLE = ("preempted", "failed", "running")


@dataclass(frozen=True)
class ResumePoint:
    """A verified checkpoint to resume from and where it came from."""

    run_id: str
    path: Path
    manifest: CheckpointManifest
    trust: TrustLevel
    status: str
    needs_recover: bool
    """True when the run is still ``running`` (its process died): mark it failed with
    ``cvpilot.runs.recover_run`` before passing it as ``resume_from``."""


def find_resume_point(
    project: str,
    *,
    checkpoint_dir: str | Path | None = None,
    store_dir: str | Path | None = None,
    require_trust: RequireTrust | None = None,
    kind: CheckpointKind = "last",
    min_step: int | None = None,
    accept_rollback_to: int | None = None,
) -> ResumePoint | None:
    """The newest verified checkpoint of ``kind`` among the project's resumable runs.

    The policy is ``require_trust``, else ``CVPILOT_CKPT_REQUIRE_TRUST``, else ``authenticated``
    when ``CVPILOT_CKPT_KEY`` is set, else ``any``. Under the strict policies a point older than
    the tag-checked pointer, the run store or ``min_step`` (``CVPILOT_CKPT_MIN_STEP``) is a
    rollback and raises; ``accept_rollback_to`` accepts it on purpose (see
    ``CheckpointManager.restore``).

    Raises:
        ValueError: no checkpoint directory is given or configured.
        CheckpointUntrustedError: checkpoints exist but none reaches the required trust level.
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
    floor_mark = min_step if min_step is not None else settings.ckpt_min_step
    opened = runs.open_anchor(project, store_root)
    anchor = opened[0] if opened is not None else None
    rejected: list[str] = []
    try:
        for meta in runs.list_runs(project, store_dir=store_root):
            if meta.status not in RESUMABLE:
                continue
            found = RunCheckpoints(
                root, meta.run_id, anchor=anchor, key=key, require_trust=policy,
                allow_pt=settings.ckpt_allow_pt,
            )  # fmt: skip
            point = _first_usable(found, meta.run_id, meta.status, kind, root)
            rejected += [s.reason for s in found.skipped]
            if point is None:  # a newer run whose checkpoints are all gone is not skipped silently
                found.check_fresh(kind, None, accept_rollback_to=accept_rollback_to)
                continue
            found.check_fresh(
                kind,
                point.manifest.step,
                accept_rollback_to=accept_rollback_to,
                min_step=floor_mark,
            )
            return point
    finally:
        if opened is not None:
            opened[1]()
    if policy != "any" and any(r in TRUST_REASONS for r in rejected):
        raise CheckpointUntrustedError(
            f"checkpoints exist but none reaches the required trust level {policy!r}"
        )
    return None


def _first_usable(
    found: RunCheckpoints, run_id: str, status: str, kind: CheckpointKind, root: Path
) -> ResumePoint | None:
    for verified in found.iter_valid(kind):
        manifest = verified.manifest
        if manifest.format == "pt":
            try:
                _check_pt(found, manifest)
            except CheckpointCorruptError as exc:
                found.skipped.append(Skipped(manifest.step, kind, manifest.path, exc.reason))
                continue
        path = Path(root).expanduser().resolve() / run_id / manifest.path
        return ResumePoint(run_id, path, manifest, verified.trust, status, status == "running")
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
