"""Domain models and the invariants of docs/architecture/03-data-model.md.

All models are frozen and reject unknown fields. Timestamps are timezone-aware and stored in
UTC. Invariants that need other rows (parent status, store uniqueness) are enforced by the
application and store layers; everything checkable on one record is checked here.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, BinaryIO, Literal, Protocol

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    field_validator,
    model_validator,
)

from cvpilot.core.errors import InvalidTransitionError
from cvpilot.core.redact import redact_command_line, redact_mapping
from cvpilot.core.ulid import is_valid_ulid

if TYPE_CHECKING:
    import numpy as np

SCHEMA_VERSION = 1

RunStatus = Literal["running", "resumed", "finished", "failed", "preempted"]
Task = Literal["detection", "classification"]
ArtifactKind = Literal[
    "checkpoint", "report", "predictions", "confusion", "config", "env", "image", "diff"
]

ACTIVE_STATUSES: frozenset[str] = frozenset({"running", "resumed"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"finished", "failed", "preempted"})

PROJECT_PATTERN = r"[a-z0-9][a-z0-9_-]{0,63}"
BASELINE_TAG_PATTERN = r"[a-z0-9][a-z0-9_-]{0,31}"
_TAG = re.compile(BASELINE_TAG_PATTERN)
_PROJECT = re.compile(PROJECT_PATTERN)
_FRAMEWORK = re.compile(r"[A-Za-z0-9_.-]+==[^\s=]+")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_KEY = 256
_METRIC_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")

MAX_CLASSES = 2048
"""Most classes an evaluation may have; bounds the confusion matrix and the AP tables."""
MAX_NAME_LENGTH = 128
"""Longest class name, slice key or slice value."""
MAX_SLICE_ENTRIES = 4096
"""Most ``key=value`` slices one snapshot may carry."""
MAX_RESAMPLES = 10_000
"""Most bootstrap resamples; bounds time and memory of the confidence intervals."""
_MAX_METRICS_PER_ROW = 64
MAX_STEP = 2**63 - 1
"""Largest step; steps are stored as signed 64-bit integers."""


def is_valid_project(value: object) -> bool:
    """True for a name allowed as a store directory (``[a-z0-9][a-z0-9_-]{0,63}``)."""
    return isinstance(value, str) and _PROJECT.fullmatch(value) is not None


def is_terminal(status: RunStatus) -> bool:
    return status in TERMINAL_STATUSES


def check_transition(old: RunStatus, new: RunStatus) -> None:
    """Allow ``running|resumed -> finished|failed|preempted`` only (invariant 9)."""
    if old in ACTIVE_STATUSES and new in TERMINAL_STATUSES:
        return
    raise InvalidTransitionError(f"run status cannot change from {old} to {new}")


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


def _no_control_chars(value: str, what: str) -> str:
    if not value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"{what} must be non-empty and contain no control characters")
    return value


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RunMeta(_Model):
    """One training run. A resumed run has ``parent_run_id`` set."""

    run_id: str
    project: str
    name: str = Field(max_length=200)
    status: RunStatus
    parent_run_id: str | None = None
    task: Task
    framework: str
    host: str = Field(max_length=255)
    started_at: AwareDatetime
    ended_at: AwareDatetime | None = None
    schema_version: StrictInt = Field(ge=1)

    @field_validator("run_id", "parent_run_id")
    @classmethod
    def _ulid(cls, value: str | None) -> str | None:
        if value is not None and not is_valid_ulid(value):
            raise ValueError("must be a valid ULID (26 Crockford base32 characters)")
        return value

    @field_validator("project")
    @classmethod
    def _project(cls, value: str) -> str:
        if not is_valid_project(value):
            raise ValueError(f"must match {PROJECT_PATTERN}")
        return value

    @field_validator("name", "host")
    @classmethod
    def _text(cls, value: str) -> str:
        return _no_control_chars(value, "value")

    @field_validator("framework")
    @classmethod
    def _framework(cls, value: str) -> str:
        if not _FRAMEWORK.fullmatch(value):
            raise ValueError('must look like "<name>==<version>"')
        return value

    @field_validator("started_at", "ended_at")
    @classmethod
    def _to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def _consistent(self) -> RunMeta:
        if self.status == "resumed" and self.parent_run_id is None:
            raise ValueError("a resumed run needs parent_run_id")
        if self.status == "running" and self.parent_run_id is not None:
            raise ValueError("parent_run_id is only set on resumed runs")
        if self.parent_run_id == self.run_id:
            raise ValueError("parent_run_id must differ from run_id")
        if is_terminal(self.status) != (self.ended_at is not None):
            raise ValueError("ended_at is set exactly when the run is in a terminal status")
        if self.ended_at is not None and self.ended_at < self.started_at:
            raise ValueError("ended_at is before started_at")
        return self


class Repro(_Model):
    """Reproducibility capture taken once at ``init``; secrets are redacted on construction."""

    run_id: str
    git_commit: str | None = None
    git_dirty: bool
    git_diff_sha256: str | None = None
    packages_lock_sha256: str
    python_version: str
    cuda: str | None = None
    gpu_names: list[str]
    dataset_uri: str
    dataset_hash: str
    config: dict[str, Any]
    seed: StrictInt | None = None
    command_line: str

    @field_validator("run_id")
    @classmethod
    def _ulid(cls, value: str) -> str:
        if not is_valid_ulid(value):
            raise ValueError("must be a valid ULID")
        return value

    @field_validator("packages_lock_sha256", "git_diff_sha256")
    @classmethod
    def _sha(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("must be 64 lower-case hex characters")
        return value

    @field_validator("dataset_hash")
    @classmethod
    def _dataset_hash(cls, value: str) -> str:
        prefix, _, rest = value.partition(":")
        if prefix not in ("dcp", "merkle") or not rest:
            raise ValueError('must be "dcp:<version>" or "merkle:<hex>"')
        return value

    @field_validator("config", mode="before")
    @classmethod
    def _config(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            raise ValueError("config must be a mapping")
        redacted = redact_mapping(value)
        try:
            json.dumps(redacted, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"config must be JSON-serialisable: {exc}") from None
        return redacted

    @field_validator("command_line", mode="before")
    @classmethod
    def _redact_command(cls, value: Any) -> Any:
        return redact_command_line(value) if isinstance(value, str) else value

    @model_validator(mode="after")
    def _git(self) -> Repro:
        if self.git_diff_sha256 is not None and not self.git_dirty:
            raise ValueError("git_diff_sha256 requires git_dirty")
        if self.git_dirty and self.git_commit is None:
            raise ValueError("git_dirty requires git_commit")
        return self


class MetricPoint(_Model):
    """One scalar at one step. NaN and +-inf are permitted values."""

    run_id: str
    key: str
    step: StrictInt = Field(ge=0, le=MAX_STEP)
    epoch: StrictInt | None = Field(default=None, ge=0, le=MAX_STEP)
    value: float
    wall_time: float

    @field_validator("run_id")
    @classmethod
    def _ulid(cls, value: str) -> str:
        if not is_valid_ulid(value):
            raise ValueError("must be a valid ULID")
        return value

    @field_validator("key")
    @classmethod
    def _key(cls, value: str) -> str:
        if len(value) > _MAX_KEY:
            raise ValueError(f"must be at most {_MAX_KEY} characters")
        return _no_control_chars(value, "key")

    @field_validator("value", "wall_time", mode="before")
    @classmethod
    def _number(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("must be a number")
        return value

    @field_validator("wall_time")
    @classmethod
    def _finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("wall_time must be finite")
        return value


class Artifact(_Model):
    """A file registered with a run, stored by content address."""

    run_id: str
    sha256: str
    kind: ArtifactKind
    uri: str
    size_bytes: StrictInt = Field(ge=0)
    meta: dict[str, Any]
    created_at: AwareDatetime

    @field_validator("run_id")
    @classmethod
    def _ulid(cls, value: str) -> str:
        if not is_valid_ulid(value):
            raise ValueError("must be a valid ULID")
        return value

    @field_validator("sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("must be 64 lower-case hex characters")
        return value

    @field_validator("created_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return _utc(value)


def _finite_mapping(values: Mapping[str, float], what: str) -> None:
    if len(values) > _MAX_METRICS_PER_ROW:
        raise ValueError(f"{what} has more than {_MAX_METRICS_PER_ROW} entries")
    for name, value in values.items():
        if _METRIC_NAME.fullmatch(name) is None:
            raise ValueError(f"{what} name {name[:40]!r} must match [A-Za-z][A-Za-z0-9_]{{0,63}}")
        if not math.isfinite(value):
            raise ValueError(f"{what} {name!r} must be finite")


class BootstrapInfo(_Model):
    """How the confidence intervals of a snapshot were produced (they are reproducible)."""

    n_resamples: StrictInt = Field(ge=1, le=MAX_RESAMPLES)
    seed: StrictInt = Field(ge=0)
    level: float = Field(gt=0.0, lt=1.0)
    numpy_version: str | None = Field(default=None, max_length=32)
    """The resampling stream is only reproducible within one numpy version."""


CiStatus = Literal["off", "computed", "not_computed_too_few_images"]
_GIT_COMMIT = re.compile(r"[0-9a-f]{7,64}")


class EvalParams(_Model):
    """Evaluation parameters; later comparisons can refuse snapshots that differ."""

    iou_thresholds: list[float] = Field(min_length=1, max_length=20)
    confusion_iou: float = Field(gt=0.0, le=1.0)
    confusion_score: float = Field(ge=0.0, le=1.0)
    max_dets: StrictInt = Field(default=100, ge=1, le=10_000)

    @field_validator("iou_thresholds")
    @classmethod
    def _grid(cls, value: list[float]) -> list[float]:
        if any(not (0.0 < v <= 1.0) for v in value) or any(
            b <= a for a, b in zip(value, value[1:], strict=False)
        ):
            raise ValueError("must be strictly increasing values in (0, 1]")
        return value


class Provenance(_Model):
    """Where the predictions came from, when the caller says so."""

    checkpoint_sha256: str | None = None
    git_commit: str | None = None

    @field_validator("checkpoint_sha256")
    @classmethod
    def _sha(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("must be 64 lower-case hex characters")
        return value

    @field_validator("git_commit")
    @classmethod
    def _commit(cls, value: str | None) -> str | None:
        if value is not None and not _GIT_COMMIT.fullmatch(value):
            raise ValueError("must be 7 to 64 lower-case hex characters")
        return value


class EvalSnapshot(_Model):
    """One evaluation of one run on one evaluation set at one step (invariants 3 and 6).

    Metric values are fractions in ``[0, 1]`` (COCO convention) unless the name says otherwise.
    A metric that is undefined for the data (for example ``mAP_small`` without small objects)
    is absent. The JSON is canonical: same inputs, same bytes.
    """

    run_id: str
    eval_set_hash: str
    step: StrictInt = Field(ge=0, le=MAX_STEP)
    task: Task
    class_names: list[str]
    n_images: StrictInt = Field(ge=0)
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]]
    slices: dict[str, dict[str, float]]
    ci: dict[str, tuple[float, float]] = Field(default_factory=dict)
    bootstrap: BootstrapInfo | None = None
    ci_status: CiStatus = "off"
    params: EvalParams | None = None
    provenance: Provenance | None = None
    slice_spec: dict[str, str] = Field(default_factory=dict)
    slice_hash: str | None = None
    """Digest of every slice column's per-image values (``core.hashing.slice_hash``); None for
    snapshots without slices and for snapshots written before it existed."""
    confusion_sha256: str
    predictions_sha256: str

    @field_validator("slice_hash")
    @classmethod
    def _slice_hash(cls, value: str | None) -> str | None:
        if value is not None and not _SHA256.fullmatch(value):
            raise ValueError("must be 64 lower-case hex characters")
        return value

    @field_validator("run_id")
    @classmethod
    def _ulid(cls, value: str) -> str:
        if not is_valid_ulid(value):
            raise ValueError("must be a valid ULID")
        return value

    @field_validator("eval_set_hash", "confusion_sha256", "predictions_sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("must be 64 lower-case hex characters")
        return value

    @field_validator("class_names")
    @classmethod
    def _classes(cls, value: list[str]) -> list[str]:
        if not 1 <= len(value) <= MAX_CLASSES:
            raise ValueError(f"needs between 1 and {MAX_CLASSES} class names")
        if len(set(value)) != len(value):
            raise ValueError("class names must be unique")
        for name in value:
            _no_control_chars(name, "class name")
            if len(name) > MAX_NAME_LENGTH:
                raise ValueError(f"class name longer than {MAX_NAME_LENGTH} characters")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> EvalSnapshot:
        if not self.metrics:
            raise ValueError("metrics must not be empty")
        _finite_mapping(self.metrics, "metric")
        known = set(self.class_names)
        if len(self.per_class) > MAX_CLASSES or len(self.slices) > MAX_SLICE_ENTRIES:
            raise ValueError("too many per_class or slices entries")
        for name, row in self.per_class.items():
            if name not in known:
                raise ValueError(f"per_class entry {name[:40]!r} is not in class_names")
            _finite_mapping(row, "per_class metric")
        for label, row in self.slices.items():
            key, sep, value = label.partition("=")
            if not sep or _METRIC_NAME.fullmatch(key) is None:
                raise ValueError(f"slice {label[:40]!r} must look like key=value")
            _no_control_chars(value, "slice value")
            if len(value) > MAX_NAME_LENGTH:
                raise ValueError(f"slice value longer than {MAX_NAME_LENGTH} characters")
            _finite_mapping(row, "slice metric")
        for name, (low, high) in self.ci.items():
            if name not in self.metrics:
                raise ValueError(f"ci entry {name!r} is not a metric")
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(f"ci of {name!r} must be a finite interval with low <= high")
        if bool(self.ci) != (self.bootstrap is not None):
            raise ValueError("bootstrap is set exactly when ci is non-empty")
        if bool(self.ci) != (self.ci_status == "computed"):
            raise ValueError('ci_status is "computed" exactly when ci is non-empty')
        if len(self.slice_spec) > MAX_SLICE_ENTRIES:
            raise ValueError("too many slice_spec entries")
        for key, column in self.slice_spec.items():
            if _METRIC_NAME.fullmatch(key) is None:
                raise ValueError(f"slice_spec key {key[:40]!r} must match [A-Za-z][A-Za-z0-9_]*")
            _no_control_chars(column, "slice_spec column")
            if len(column) > MAX_NAME_LENGTH:
                raise ValueError("slice_spec column is too long")
        return self


def is_valid_tag(value: object) -> bool:
    """True for a baseline tag (``[a-z0-9][a-z0-9_-]{0,31}``); never starts with ``-``."""
    return isinstance(value, str) and _TAG.fullmatch(value) is not None


class Baseline(_Model):
    """A named pin of one snapshot, used by ``compare`` and ``gate`` (invariant 11).

    ``snapshot_sha256`` is the digest of the pinned snapshot file at pin time; every read
    compares it with the stored snapshot, so a snapshot rewritten after pinning is detected.
    """

    project: str
    tag: str
    run_id: str
    eval_set_hash: str
    step: StrictInt = Field(ge=0, le=MAX_STEP)
    snapshot_sha256: str
    pinned_at: AwareDatetime

    @field_validator("project")
    @classmethod
    def _project(cls, value: str) -> str:
        if not is_valid_project(value):
            raise ValueError(f"must match {PROJECT_PATTERN}")
        return value

    @field_validator("tag")
    @classmethod
    def _tag_pattern(cls, value: str) -> str:
        if not is_valid_tag(value):
            raise ValueError(f"must match {BASELINE_TAG_PATTERN}")
        return value

    @field_validator("run_id")
    @classmethod
    def _ulid(cls, value: str) -> str:
        if not is_valid_ulid(value):
            raise ValueError("must be a valid ULID")
        return value

    @field_validator("eval_set_hash", "snapshot_sha256")
    @classmethod
    def _sha(cls, value: str) -> str:
        if not _SHA256.fullmatch(value):
            raise ValueError("must be 64 lower-case hex characters")
        return value

    @field_validator("pinned_at")
    @classmethod
    def _to_utc(cls, value: datetime) -> datetime:
        return _utc(value)


class MetricSink(Protocol):
    """What framework hooks talk to; implemented by ``runs``.

    ``log_eval`` and ``checkpoint_saved`` join this protocol when the eval engine and the
    checkpoint manager exist.
    """

    def log(self, points: Mapping[str, float], *, step: int, epoch: int | None = None) -> None: ...

    def log_artifact(
        self, path: Path, *, kind: ArtifactKind, meta: Mapping[str, Any] | None = None
    ) -> Artifact: ...


class Forwarder(Protocol):
    """Pushes a run's scalars to a tracker; adapters register under ``forwarders``.

    An adapter is constructed with ``allow_insecure`` (true only when ``CVPILOT_ENV=dev``) and
    ``allowed_hosts`` (``CVPILOT_FORWARD_HOSTS``; empty means no restriction), reads the few
    variables that decide where it connects once at construction, and takes its credentials from
    the tracker SDK's own environment variables: it takes no token argument, stores none and logs
    none. ``open``, ``log_points`` and ``close``
    run on a worker thread owned by the fan-out, never on the training thread.
    """

    name: str
    requires: tuple[str, ...]

    def open(self, meta: RunMeta, repro: Repro) -> None: ...
    def log_points(self, points: Sequence[MetricPoint]) -> None: ...
    def close(self, status: RunStatus) -> None: ...


ForwarderFactory = Callable[..., Forwarder]


class Hook(Protocol):
    """Adapter between a training framework and a ``MetricSink``; registers under ``hooks``.

    ``requires`` lists distribution names (``"a|b"``: either); ``license_notice`` is set for a
    copyleft framework. Callbacks the hook installs never raise into the framework.
    """

    name: str
    requires: tuple[str, ...]
    license_notice: str | None

    def matches(self, target: object) -> bool: ...
    def attach(self, target: object, sink: MetricSink) -> None: ...
    def detach(self) -> None: ...


HookFactory = Callable[[], Hook]


class MetricStats(_Model):
    """Index-level summary of a run's metric files; ``min_step``/``max_step`` are None if empty."""

    n_points: int
    n_files: int
    min_step: int | None = None
    max_step: int | None = None


class RunStore(Protocol):
    """Persistence port used by the application layer; adapters register under ``stores``."""

    @property
    def path(self) -> Path: ...

    def put_run(self, meta: RunMeta, repro: Repro) -> None: ...
    def get_run(self, run_id: str) -> RunMeta: ...
    def get_repro(self, run_id: str) -> Repro: ...
    def list_runs(self, *, status: RunStatus | None = None) -> list[RunMeta]: ...
    def set_status(self, run_id: str, status: RunStatus, ended_at: datetime) -> None: ...
    def append_points(self, points: Sequence[MetricPoint]) -> str: ...
    def read_points(self, run_id: str) -> list[MetricPoint]: ...
    def metric_stats(self, run_id: str) -> MetricStats: ...
    def compact_metrics(self, run_id: str) -> None: ...
    def put_artifact(
        self, run_id: str, path: Path, kind: ArtifactKind, meta: Mapping[str, Any]
    ) -> Artifact: ...
    def put_artifact_bytes(
        self, run_id: str, data: bytes, kind: ArtifactKind, meta: Mapping[str, Any]
    ) -> Artifact: ...
    def list_artifacts(self, run_id: str) -> list[Artifact]: ...
    def put_snapshot(self, snapshot: EvalSnapshot) -> None: ...
    def get_snapshot(
        self, run_id: str, eval_set_hash: str, step: int | None = None
    ) -> EvalSnapshot: ...
    def list_snapshots(self, run_id: str) -> list[EvalSnapshot]: ...
    def has_snapshot(self, run_id: str, eval_set_hash: str, step: int) -> bool: ...
    def snapshot_digest(self, run_id: str, eval_set_hash: str, step: int) -> str: ...
    def put_baseline(self, baseline: Baseline, *, replace: bool = False) -> None: ...
    def get_baseline(self, tag: str) -> Baseline: ...
    def list_baselines(self, *, history: bool = False) -> list[Baseline]: ...
    def baseline_head(self) -> tuple[int, str] | None: ...
    def artifact_path(self, sha256: str) -> Path: ...
    def open_artifact(self, sha256: str) -> BinaryIO: ...
    def close(self) -> None: ...


class TableIO(Protocol):
    """Columnar file adapter used by the eval engine; registered under ``tables``."""

    def read_columns(
        self,
        path: Path,
        *,
        required: Sequence[str],
        optional: Sequence[str] = (),
        nullable: Sequence[str] = (),
        max_rows: int,
        max_file_bytes: int = ...,
        max_uncompressed_bytes: int = ...,
        max_row_groups: int = ...,
        max_string_length: int = ...,
        max_string_bytes: int = ...,
    ) -> dict[str, np.ndarray[Any, Any]]: ...

    def write_columns(
        self,
        path: Path,
        columns: Mapping[str, np.ndarray[Any, Any]],
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> None: ...


StoreFactory = Callable[..., RunStore]
TableIOFactory = Callable[[], TableIO]
