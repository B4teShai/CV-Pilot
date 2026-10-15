"""Specific exceptions raised by the domain, application and store layers.

Every exception names the offending input. They are never swallowed by the library; the CLI
error boundary turns a ``CvpilotError`` into one line on stderr and one structured log event.
"""

from __future__ import annotations

from typing import Any, ClassVar


class CvpilotError(Exception):
    """Base class for expected, user-actionable failures."""

    event: ClassVar[str] = "command_failed"

    def log_fields(self) -> dict[str, Any]:
        """Extra structured fields for the log event emitted at the boundary."""
        return {}


class InvalidRunIdError(CvpilotError, ValueError):
    """A run id is not a valid ULID."""


class RunNotFoundError(CvpilotError, LookupError):
    """No run with this id exists in the project store."""


class InvalidTransitionError(CvpilotError, ValueError):
    """A run status change is not allowed by the state machine."""


class DuplicateMetricError(CvpilotError, ValueError):
    """A metric point repeats an existing ``(run_id, key, step)``."""


class MetricOrderError(CvpilotError, ValueError):
    """A metric step is not greater than the last step logged for the same key."""


class StoreError(CvpilotError, OSError):
    """The run store could not be read or written."""

    event = "store_write_failed"


class StoreNotFoundError(StoreError):
    """The project has no run store yet."""


class StoreVersionError(StoreError):
    """The store was written by a newer library than this one."""

    event = "store_version_unsupported"

    def __init__(self, store_version: int, library_version: int) -> None:
        super().__init__(
            f"store schema version {store_version} is newer than the supported "
            f"{library_version}; upgrade cvpilot"
        )
        self.store_version = store_version
        self.library_version = library_version

    def log_fields(self) -> dict[str, Any]:
        return {"store_version": self.store_version, "library_version": self.library_version}


class RunClosedError(CvpilotError, RuntimeError):
    """A run was used after it finished."""


class NoActiveRunError(CvpilotError, RuntimeError):
    """``log`` or ``finish`` was called before ``init``."""


class RunAlreadyActiveError(CvpilotError, RuntimeError):
    """``init`` was called while another run of this process is still active."""


class InvalidResumeError(CvpilotError, ValueError):
    """A run cannot be resumed because its status or task does not allow it."""


class EvalInputError(CvpilotError, ValueError):
    """Evaluation inputs (tables, class names, slices, files) failed validation."""


class SliceCardinalityError(EvalInputError):
    """A slice key has more distinct values than the evaluation allows."""


class SnapshotExistsError(CvpilotError, ValueError):
    """A snapshot for ``(run_id, eval_set_hash, step)`` exists; snapshots are immutable."""


class SnapshotNotFoundError(CvpilotError, LookupError):
    """No eval snapshot with this ``(run_id, eval_set_hash, step)`` exists."""


class SnapshotIntegrityError(StoreError):
    """A stored snapshot, artifact or the snapshot hash chain does not match its digest."""

    event = "snapshot_integrity_failed"
