"""Dry run: show what a configuration would have done to a recorded run, sending nothing.

The stored scalars are shown to a fresh engine in step order with a clock taken from their
recording times (so the rate limit behaves as it would have), every evaluation snapshot of the
run goes through the real baseline comparison, and the final status is applied last. The engine's
delivery function only collects: no channel is built here, no event named ``alert_fired`` is
logged, nothing leaves the process.

An evaluation snapshot has no recording time. Snapshots are replayed in step order, interleaved
with the scalars of the steps before them, and each is taken to be logged one rate-limit window
after the previous event, so two snapshots are never muted by each other in the replay; in a live
run two evaluations less than a window apart are, and the second sends no alert.

At most ``MAX_REPLAY_POINTS`` points are replayed; a larger run is refused rather than
cut short, because a partial replay would look like a clean one.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

from cvpilot.alerts.config import AlertConfig
from cvpilot.alerts.engine import AlertEngine
from cvpilot.core.errors import CvpilotError
from cvpilot.core.schema import Alert
from cvpilot.runs.queries import open_existing

MAX_REPLAY_POINTS = 1_000_000


class ReplayTooLargeError(CvpilotError, ValueError):
    """The run has more points than a dry run replays."""


@dataclass(frozen=True)
class Replay:
    run_id: str
    status: str
    n_points: int
    n_snapshots: int
    alerts: tuple[Alert, ...]
    suppressed: int
    rule_failures: int


def dry_run(
    config: AlertConfig, project: str, run_id: str, *, store_dir: Path | None = None
) -> Replay:
    """Replay a run of ``project`` through ``config``.

    Raises:
        InvalidRunIdError, RunNotFoundError, StoreError: the run cannot be read.
        ReplayTooLargeError: the run has more than ``MAX_REPLAY_POINTS`` points.
    """
    store = open_existing(project, run_id, store_dir)
    try:
        meta = store.get_run(run_id)
        if store.metric_stats(run_id).n_points > MAX_REPLAY_POINTS:
            raise ReplayTooLargeError(
                f"run {run_id} has more than {MAX_REPLAY_POINTS} points; a dry run replays "
                "smaller runs only"
            )
        points = sorted(store.read_points(run_id), key=lambda p: (p.step, p.key))
        snapshots = sorted(store.list_snapshots(run_id), key=lambda s: (s.step, s.eval_set_hash))
    finally:
        store.close()
    now = [0.0]
    found: list[Alert] = []
    engine = AlertEngine(
        config,
        project=project,
        run_id=run_id,
        deliver=found.append,
        emit=lambda level, event, **fields: None,
        clock=lambda: now[0],
        store_dir=store_dir,
    )
    pending = list(snapshots)

    def evaluate_due(up_to: float) -> None:
        while pending and pending[0].step <= up_to:
            now[0] += config.rate_limit_s  # a snapshot carries no time of its own
            engine.on_eval(pending.pop(0))

    for step, group in groupby(points, key=lambda p: p.step):
        evaluate_due(step - 1)
        batch = list(group)
        now[0] = max(now[0], min(p.wall_time for p in batch))
        engine.on_points({p.key: p.value for p in batch}, step)
    evaluate_due(float("inf"))
    engine.on_finish(meta.status)
    return Replay(
        run_id,
        meta.status,
        len(points),
        len(snapshots),
        tuple(found),
        engine.stats.suppressed,
        engine.stats.rule_failures,
    )
