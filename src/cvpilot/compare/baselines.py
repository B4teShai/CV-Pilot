"""Baseline use cases: pin a verified snapshot under a tag, list, show.

Pinning verifies the snapshot (file digest, the run's snapshot chain and the baseline chain)
before anything is written, so a baseline never points at a snapshot that fails verification.
Events (constant messages): ``baseline_set`` (INFO) and, from the CLI boundary,
``baseline_refused`` and the integrity events.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cvpilot.core.errors import (
    BaselineNotFoundError,
    BaselineRefusedError,
    StoreNotFoundError,
)
from cvpilot.core.schema import Baseline, EvalSnapshot, is_valid_tag
from cvpilot.logging import get_logger
from cvpilot.runs.queries import open_existing, open_store

EVAL_SET_ARGUMENT = re.compile(r"[0-9a-f]{8,64}")
"""An eval set hash or a prefix of at least 8 lower-case hex characters."""


@dataclass(frozen=True)
class BaselineView:
    """A verified pin with the verified snapshot it points at."""

    baseline: Baseline
    snapshot: EvalSnapshot
    history: int
    head_seq: int
    head_chain_sha256: str


def _resolve_eval_set(
    snapshots: list[EvalSnapshot], project: str, tag: str, run_id: str, wanted: str | None
) -> str:
    hashes = sorted({s.eval_set_hash for s in snapshots})

    def refuse(message: str, eval_set_hash: str | None = None) -> BaselineRefusedError:
        return BaselineRefusedError(
            message, project=project, tag=tag, run_id=run_id, eval_set_hash=eval_set_hash
        )

    if wanted is None:
        if not hashes:
            raise refuse(f"run {run_id} has no eval snapshot to pin")
        if len(hashes) > 1:
            shown = ", ".join(h[:12] for h in hashes[:8])
            raise refuse(f"run {run_id} has several eval sets ({shown}); pass --eval-set")
        return hashes[0]
    if EVAL_SET_ARGUMENT.fullmatch(wanted) is None:
        raise refuse("eval set must be 8 to 64 lower-case hex characters")
    matches = [h for h in hashes if h.startswith(wanted)]
    if not matches:
        raise refuse(f"run {run_id} has no snapshot for eval set {wanted[:12]}", wanted)
    if len(matches) > 1:
        raise refuse(f"eval set prefix {wanted} is ambiguous in run {run_id}")
    return matches[0]


def set_baseline(
    project: str,
    tag: str,
    run_id: str,
    *,
    eval_set: str | None = None,
    step: int | None = None,
    replace: bool = False,
    store_dir: Path | None = None,
) -> Baseline:
    """Pin the snapshot of ``run_id`` (latest step of the eval set unless ``step`` is given).

    Raises:
        BaselineRefusedError: the run is not finished, no such snapshot, ambiguous eval set,
            wrong project.
        BaselineExistsError: the tag is pinned and ``replace`` is false.
        SnapshotIntegrityError, BaselineIntegrityError: something fails verification.
        RunNotFoundError, InvalidRunIdError: the run is unknown or the id malformed.
    """
    if not is_valid_tag(tag):
        raise BaselineRefusedError("invalid baseline tag", project=project, run_id=run_id)
    store = open_existing(project, run_id, store_dir)
    try:
        status = store.get_run(run_id).status
        if status != "finished":  # same rule as --baseline RUN_ID; recover_run, then pin
            raise BaselineRefusedError(
                f"run {run_id} is {status}; a baseline run must be finished",
                project=project,
                tag=tag,
                run_id=run_id,
            )
        snapshots = store.list_snapshots(run_id)  # verified: files and the run's chain
        eval_set_hash = _resolve_eval_set(snapshots, project, tag, run_id, eval_set)
        steps = sorted(s.step for s in snapshots if s.eval_set_hash == eval_set_hash)
        chosen = steps[-1] if step is None else step
        if chosen not in steps:
            raise BaselineRefusedError(
                f"run {run_id} has no snapshot at step {chosen} for eval set {eval_set_hash[:12]}",
                project=project,
                tag=tag,
                run_id=run_id,
                eval_set_hash=eval_set_hash,
            )
        digest = store.snapshot_digest(run_id, eval_set_hash, chosen)  # verifies the snapshot
        replaced = any(b.tag == tag for b in store.list_baselines())  # verifies the chain
        baseline = Baseline(
            project=project,
            tag=tag,
            run_id=run_id,
            eval_set_hash=eval_set_hash,
            step=chosen,
            snapshot_sha256=digest,
            pinned_at=datetime.now(timezone.utc),
        )
        store.put_baseline(baseline, replace=replace)
    finally:
        store.close()
    get_logger("cvpilot.compare").info(
        "baseline_set",
        project=project,
        tag=tag,
        run_id=run_id,
        eval_set_hash=eval_set_hash,
        step=chosen,
        replaced=replaced,
    )
    return baseline


def list_baselines(
    project: str, *, history: bool = False, store_dir: Path | None = None
) -> list[Baseline]:
    """Verified pins of a project (an absent store is an empty list)."""
    try:
        store = open_store(project, store_dir, create=False)
    except StoreNotFoundError:
        return []
    try:
        return store.list_baselines(history=history)
    finally:
        store.close()


def show_baseline(project: str, tag: str, *, store_dir: Path | None = None) -> BaselineView:
    """The newest pin of ``tag`` and its snapshot, both verified.

    Raises:
        BaselineNotFoundError: unknown tag (or the project has no store).
        BaselineIntegrityError, SnapshotIntegrityError: verification failed.
    """
    try:
        store = open_store(project, store_dir, create=False)
    except StoreNotFoundError:
        raise BaselineNotFoundError(f"project {project!r} has no store, so no baseline") from None
    try:
        baseline = store.get_baseline(tag)
        snapshot = store.get_snapshot(baseline.run_id, baseline.eval_set_hash, baseline.step)
        count = sum(1 for b in store.list_baselines(history=True) if b.tag == tag)
        head = store.baseline_head()
    finally:
        store.close()
    if head is None:  # a verified tag implies at least one row
        raise BaselineNotFoundError(f"no baseline is pinned under tag {tag!r}")
    return BaselineView(baseline, snapshot, count, head[0], head[1])
