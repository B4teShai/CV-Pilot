"""Compare and gate use cases: resolve two verified snapshots, compare, apply thresholds.

Trust: nothing is read from a previous comparison, report or gate result. Both snapshots come
from ``get_snapshot`` (file digest and the run's chain verified), a tag baseline additionally
from ``get_baseline`` (record, baseline chain and pinned snapshot digest verified), and a
confusion matrix from a SHA-256 verified artifact. Any verification failure raises and ends in
exit 1; a missing candidate snapshot or a different evaluation set is a refusal
(``compare_refused``), never a pass.

Events (constant messages): ``compare_completed`` (INFO), ``gate_passed`` (INFO),
``gate_failed`` (ERROR), ``slice_skipped`` (WARN); refusals are logged by the CLI boundary as
``compare_refused``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cvpilot.compare import confusion as cf
from cvpilot.compare.engine import Comparison, ConfusionDelta, compare
from cvpilot.compare.gate import GateResult, Thresholds, gate
from cvpilot.core.errors import (
    BaselineIntegrityError,
    CompareRefusedError,
    EvalSetMismatchError,
    SnapshotNotFoundError,
)
from cvpilot.core.schema import EvalSnapshot, RunStore, is_valid_tag
from cvpilot.core.ulid import is_valid_ulid
from cvpilot.logging import get_logger
from cvpilot.runs.queries import open_existing

TAG_PREFIX = "tag:"


def parse_baseline_ref(text: str) -> tuple[str | None, str | None]:
    """``tag:NAME`` gives ``(NAME, None)``; a run id (any case) gives ``(None, RUN_ID)``.

    Raises:
        ValueError: neither a valid tag reference nor a run id.
    """
    if text.startswith(TAG_PREFIX):
        tag = text[len(TAG_PREFIX) :]
        if not is_valid_tag(tag):
            raise ValueError("invalid baseline tag after 'tag:'")
        return tag, None
    run_id = text.strip().upper()
    if not is_valid_ulid(run_id):
        raise ValueError("expected tag:NAME or a run id (26-character ULID)")
    return None, run_id


@dataclass(frozen=True)
class Resolved:
    candidate: EvalSnapshot
    baseline: EvalSnapshot
    tag: str | None
    confusion: ConfusionDelta | None


@dataclass(frozen=True)
class GateOutcome:
    comparison: Comparison
    result: GateResult
    tag: str | None
    baseline_run_id: str


def _mismatch(
    reason: str,
    message: str,
    *,
    candidate_run_id: str,
    baseline_run_id: str,
    candidate_hash: str | None,
    baseline_hash: str | None,
) -> EvalSetMismatchError:
    return EvalSetMismatchError(
        message,
        reason=reason,
        candidate_run_id=candidate_run_id,
        baseline_run_id=baseline_run_id,
        candidate_eval_set_hash=candidate_hash,
        baseline_eval_set_hash=baseline_hash,
    )


def _candidate_hash(store: RunStore, run_id: str, eval_set: str | None) -> str:
    snapshots = store.list_snapshots(run_id)
    hashes = sorted({s.eval_set_hash for s in snapshots})
    if not hashes:
        raise CompareRefusedError(
            f"run {run_id} has no eval snapshot to compare",
            reason="no_candidate_snapshot",
            candidate_run_id=run_id,
        )
    if eval_set is not None:
        found = [h for h in hashes if h.startswith(eval_set)]
        if len(found) == 1:
            return found[0]
        raise CompareRefusedError(
            f"run {run_id} has no unique snapshot for eval set {eval_set[:12]}",
            reason="eval_set_hash",
            candidate_run_id=run_id,
        )
    if len(hashes) > 1:
        raise CompareRefusedError(
            f"run {run_id} has several eval sets; pass --eval-set",
            reason="eval_set_hash",
            candidate_run_id=run_id,
        )
    return hashes[0]


def _load(
    store: RunStore,
    run_id: str,
    eval_set_hash: str,
    step: int | None,
    *,
    other_run_id: str,
    candidate: bool,
) -> EvalSnapshot:
    try:
        return store.get_snapshot(run_id, eval_set_hash, step)
    except SnapshotNotFoundError:
        pass
    own = sorted({s.eval_set_hash for s in store.list_snapshots(run_id)})
    if not own:
        raise CompareRefusedError(
            f"run {run_id} has no eval snapshot to compare",
            reason="no_candidate_snapshot" if candidate else "no_baseline_snapshot",
            candidate_run_id=run_id if candidate else other_run_id,
            baseline_run_id=other_run_id if candidate else run_id,
        ) from None
    mine = own[0] if len(own) == 1 else None
    raise _mismatch(
        "eval_set_hash",
        f"run {run_id} has no snapshot for eval set {eval_set_hash[:12]}"
        + ("" if step is None else f" at step {step}"),
        candidate_run_id=run_id if candidate else other_run_id,
        baseline_run_id=other_run_id if candidate else run_id,
        candidate_hash=mine if candidate else eval_set_hash,
        baseline_hash=eval_set_hash if candidate else mine,
    ) from None


def resolve(
    project: str,
    run_id: str,
    baseline_ref: str,
    *,
    eval_set: str | None = None,
    step: int | None = None,
    with_confusion: bool = False,
    store_dir: Path | None = None,
) -> Resolved:
    """The verified candidate and baseline snapshots (and the confusion delta when asked).

    Raises:
        CompareRefusedError / EvalSetMismatchError: no comparable snapshot exists.
        BaselineNotFoundError, BaselineIntegrityError, SnapshotIntegrityError, StoreError,
        RunNotFoundError: unknown tag or run, or a verification failure.
    """
    tag, base_run = parse_baseline_ref(baseline_ref)
    store = open_existing(project, run_id, store_dir)
    try:
        store.get_run(run_id)
        if tag is not None:
            pin = store.get_baseline(tag)
            base = store.get_snapshot(pin.run_id, pin.eval_set_hash, pin.step)
            if (
                store.snapshot_digest(pin.run_id, pin.eval_set_hash, pin.step)
                != pin.snapshot_sha256
            ):
                raise BaselineIntegrityError(f"baseline {tag!r}: the pinned snapshot changed")
            target = pin.eval_set_hash
            if eval_set is not None and not target.startswith(eval_set):
                raise _mismatch(
                    "eval_set_hash",
                    f"--eval-set {eval_set[:12]} is not the evaluation set of baseline {tag!r}",
                    candidate_run_id=run_id,
                    baseline_run_id=pin.run_id,
                    candidate_hash=None,
                    baseline_hash=target,
                )
            base_id = pin.run_id
        else:
            if base_run is None:  # parse_baseline_ref returns exactly one of the two
                raise CompareRefusedError("no baseline given", reason="no_baseline")
            store.get_run(base_run)
            target = _candidate_hash(store, run_id, eval_set)
            base = _load(store, base_run, target, None, other_run_id=run_id, candidate=False)
            base_id = base_run
        cand = _load(store, run_id, target, step, other_run_id=base_id, candidate=True)
        delta = None
        if with_confusion:
            names = len(cand.class_names)
            delta = cf.confusion_delta(
                cf.load_confusion(store, base.confusion_sha256, names),
                cf.load_confusion(store, cand.confusion_sha256, names),
                cand.class_names,
            )
    finally:
        store.close()
    return Resolved(cand, base, tag, delta)


def run_compare(
    project: str,
    run_id: str,
    baseline_ref: str,
    *,
    eval_set: str | None = None,
    step: int | None = None,
    store_dir: Path | None = None,
) -> Comparison:
    """Compare a run against a baseline; logs ``compare_completed``."""
    found = resolve(
        project, run_id, baseline_ref, eval_set=eval_set, step=step, with_confusion=True,
        store_dir=store_dir,
    )  # fmt: skip
    comparison = compare(found.candidate, found.baseline, confusion=found.confusion)
    get_logger("cvpilot.compare").info(
        "compare_completed",
        candidate_run_id=found.candidate.run_id,
        baseline_run_id=found.baseline.run_id,
        candidate_eval_set_hash=found.candidate.eval_set_hash,
        baseline_eval_set_hash=found.baseline.eval_set_hash,
        candidate_step=found.candidate.step,
        baseline_step=found.baseline.step,
    )
    return comparison


def run_gate(
    project: str,
    run_id: str,
    baseline_ref: str,
    thresholds: Thresholds,
    *,
    eval_set: str | None = None,
    step: int | None = None,
    with_confusion: bool = False,
    store_dir: Path | None = None,
) -> GateOutcome:
    """Gate a run against a baseline; logs ``gate_passed`` or ``gate_failed``."""
    found = resolve(
        project, run_id, baseline_ref, eval_set=eval_set, step=step,
        with_confusion=with_confusion, store_dir=store_dir,
    )  # fmt: skip
    comparison = compare(found.candidate, found.baseline, confusion=found.confusion)
    result = gate(comparison, thresholds)
    log = get_logger("cvpilot.compare")
    for label in result.skipped:
        log.warning("slice_skipped", run_id=run_id, slice=label)
    fields = {"run_id": run_id, "baseline_tag": found.tag, "n_violations": len(result.violations)}
    if result.passed:
        log.info("gate_passed", **fields)
    else:
        worst = result.violations[0]
        log.error(
            "gate_failed",
            **fields,
            worst_metric=worst.subject,
            worst_kind=worst.kind,
            worst_delta=None if worst.drop_points is None else -worst.drop_points,
        )
    return GateOutcome(comparison, result, found.tag, found.baseline.run_id)
