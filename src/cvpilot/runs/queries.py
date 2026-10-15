"""Read-only use cases used by the CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from cvpilot.core.errors import InvalidRunIdError, RunNotFoundError, StoreNotFoundError
from cvpilot.core.schema import (
    Artifact,
    EvalSnapshot,
    MetricStats,
    Repro,
    RunMeta,
    RunStatus,
    RunStore,
)
from cvpilot.core.ulid import is_valid_ulid
from cvpilot.registry import stores


def open_store(project: str, store_dir: Path | None, *, create: bool = True) -> RunStore:
    """Open a project's store through the ``stores`` registry (default adapter ``local``)."""
    return stores.get("local")(project, store_dir, create=create)


def store_location(project: str, store_dir: Path | None) -> Path | None:
    """The directory of a project's store, or None when it does not exist (nothing is created)."""
    try:
        store = open_store(project, store_dir, create=False)
    except StoreNotFoundError:
        return None
    try:
        return store.path
    finally:
        store.close()


@dataclass(frozen=True)
class RunView:
    """Everything stored about one run, for ``runs show``."""

    meta: RunMeta
    repro: Repro
    stats: MetricStats
    artifacts: list[Artifact]
    store_path: Path
    snapshots: list[EvalSnapshot] = field(default_factory=list)


def list_runs(
    project: str, *, status: RunStatus | None = None, store_dir: Path | None = None
) -> list[RunMeta]:
    """Runs of a project, newest first; an absent store is an empty list."""
    try:
        store = open_store(project, store_dir, create=False)
    except StoreNotFoundError:
        return []
    try:
        return store.list_runs(status=status)
    finally:
        store.close()


def open_existing(project: str, run_id: str, store_dir: Path | None) -> RunStore:
    """Open a project's store to act on ``run_id``; an absent store means the run is unknown."""
    if not is_valid_ulid(run_id):
        raise InvalidRunIdError(f"{run_id[:40]!r} is not a valid run id (26-character ULID)")
    try:
        return open_store(project, store_dir, create=False)
    except StoreNotFoundError:
        raise RunNotFoundError(
            f"run {run_id} not found: project {project!r} has no store"
        ) from None


def get_run_view(project: str, run_id: str, *, store_dir: Path | None = None) -> RunView:
    """Load one run completely.

    Raises:
        InvalidRunIdError: ``run_id`` is not a ULID.
        RunNotFoundError: no such run in the project (including an absent store).
    """
    store = open_existing(project, run_id, store_dir)
    try:
        return RunView(
            meta=store.get_run(run_id),
            repro=store.get_repro(run_id),
            stats=store.metric_stats(run_id),
            artifacts=store.list_artifacts(run_id),
            snapshots=store.list_snapshots(run_id),
            store_path=store.path,
        )
    finally:
        store.close()
