import math
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from cvpilot.core.errors import InvalidTransitionError
from cvpilot.core.hashing import new_run_id
from cvpilot.core.redact import REDACTED
from cvpilot.core.schema import (
    Artifact,
    MetricPoint,
    MetricSink,
    Repro,
    RunMeta,
    check_transition,
    is_terminal,
)

NOW = datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc)
SHA = "a" * 64


def meta(**over: Any) -> RunMeta:
    base: dict[str, Any] = {
        "run_id": new_run_id(),
        "project": "fridge-sku",
        "name": "python-2026-10-14",
        "status": "running",
        "task": "detection",
        "framework": "python==3.12.1",
        "host": "box",
        "started_at": NOW,
        "schema_version": 1,
    }
    base.update(over)
    return RunMeta(**base)


def repro(run_id: str | None = None, **over: Any) -> Repro:
    base: dict[str, Any] = {
        "run_id": run_id or new_run_id(),
        "git_commit": None,
        "git_dirty": False,
        "packages_lock_sha256": SHA,
        "python_version": "3.12.1",
        "gpu_names": [],
        "dataset_uri": "/data/x",
        "dataset_hash": "merkle:" + SHA,
        "config": {"lr": 0.1},
        "command_line": "train.py --epochs 2",
    }
    base.update(over)
    return Repro(**base)


# --- RunMeta -------------------------------------------------------------


def test_runmeta_accepts_a_valid_run() -> None:
    m = meta()
    assert m.parent_run_id is None and m.ended_at is None and m.status == "running"


@pytest.mark.parametrize("bad", ["", "abc", "01ARZ3NDEKTSV4RRFFQ69G5FAI", "../x"])
def test_runmeta_rejects_invalid_ulid(bad: str) -> None:
    with pytest.raises(ValidationError):
        meta(run_id=bad)


@pytest.mark.parametrize(
    "project", ["", "Upper", "-lead", "_lead", "a/b", "a b", "..", "a" * 65, "é"]
)
def test_runmeta_rejects_invalid_project(project: str) -> None:
    with pytest.raises(ValidationError):
        meta(project=project)


@pytest.mark.parametrize("project", ["a", "0", "fridge-sku", "a_b-c9", "a" * 64])
def test_runmeta_accepts_valid_project(project: str) -> None:
    assert meta(project=project).project == project


def test_resumed_requires_parent_and_running_forbids_it() -> None:
    parent = new_run_id()
    assert meta(status="resumed", parent_run_id=parent).parent_run_id == parent
    with pytest.raises(ValidationError):
        meta(status="resumed")
    with pytest.raises(ValidationError):
        meta(status="running", parent_run_id=parent)


def test_parent_cannot_be_self() -> None:
    rid = new_run_id()
    with pytest.raises(ValidationError):
        meta(run_id=rid, status="resumed", parent_run_id=rid)


def test_terminal_status_requires_ended_at_and_active_forbids_it() -> None:
    with pytest.raises(ValidationError):
        meta(status="finished")
    with pytest.raises(ValidationError):
        meta(status="running", ended_at=NOW)
    assert meta(status="failed", ended_at=NOW).ended_at == NOW


def test_ended_before_started_is_rejected() -> None:
    with pytest.raises(ValidationError):
        meta(status="finished", ended_at=NOW - timedelta(seconds=1))


def test_naive_datetime_rejected_and_other_zone_normalised_to_utc() -> None:
    with pytest.raises(ValidationError):
        meta(started_at=datetime(2026, 1, 1))
    tz = timezone(timedelta(hours=9))
    m = meta(started_at=datetime(2026, 1, 1, 9, 0, tzinfo=tz))
    assert m.started_at.utcoffset() == timedelta(0)
    assert m.started_at.hour == 0


def test_models_are_frozen() -> None:
    m = meta()
    with pytest.raises(ValidationError):
        m.status = "finished"  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        meta(user="me")


def test_framework_shape() -> None:
    assert meta(framework="lightning==2.6.6")
    with pytest.raises(ValidationError):
        meta(framework="lightning")


def test_transitions() -> None:
    for start in ("running", "resumed"):
        for end in ("finished", "failed", "preempted"):
            check_transition(start, end)  # type: ignore[arg-type]
    for terminal in ("finished", "failed", "preempted"):
        assert is_terminal(terminal)  # type: ignore[arg-type]
        for anything in ("running", "resumed", "finished", "failed", "preempted"):
            with pytest.raises(InvalidTransitionError):
                check_transition(terminal, anything)  # type: ignore[arg-type]
    for bad in (("running", "running"), ("running", "resumed"), ("resumed", "running")):
        with pytest.raises(InvalidTransitionError):
            check_transition(*bad)  # type: ignore[arg-type]


# --- Repro ---------------------------------------------------------------


def test_repro_valid_and_defaults() -> None:
    r = repro()
    assert r.cuda is None and r.seed is None and r.git_diff_sha256 is None


@pytest.mark.parametrize("h", ["", "merkle:", "dcp:", "sha:abc", "abc", "MERKLE:abc"])
def test_repro_dataset_hash_prefix(h: str) -> None:
    with pytest.raises(ValidationError):
        repro(dataset_hash=h)


def test_repro_accepts_dcp_hash() -> None:
    assert repro(dataset_hash="dcp:1a2b3c").dataset_hash == "dcp:1a2b3c"


def test_repro_diff_hash_requires_dirty_tree() -> None:
    with pytest.raises(ValidationError):
        repro(git_dirty=False, git_diff_sha256=SHA)
    assert repro(git_commit="abc", git_dirty=True, git_diff_sha256=SHA)


def test_repro_dirty_requires_commit() -> None:
    with pytest.raises(ValidationError):
        repro(git_commit=None, git_dirty=True)


def test_repro_sha_shape() -> None:
    with pytest.raises(ValidationError):
        repro(packages_lock_sha256="xyz")


def test_repro_redacts_command_line_and_config_before_storage() -> None:
    r = repro(
        command_line="train.py --token=abc123 --lr 0.1",
        config={"api_key": "k", "opt": {"password": "p", "lr": 1}},
    )
    assert "abc123" not in r.command_line and REDACTED in r.command_line
    assert r.config["api_key"] == REDACTED
    assert r.config["opt"] == {"password": REDACTED, "lr": 1}


def test_repro_config_objects_and_bytes_are_stored_as_safe_strings() -> None:
    r = repro(config={"x": object(), "b": b"secret-bytes"})
    assert isinstance(r.config["x"], str) and r.config["b"] == REDACTED


def test_repro_config_with_nan_rejected() -> None:
    with pytest.raises(ValidationError):
        repro(config={"x": math.nan})


# --- MetricPoint ---------------------------------------------------------


def point(**over: Any) -> MetricPoint:
    base: dict[str, Any] = {
        "run_id": new_run_id(),
        "key": "train/loss",
        "step": 0,
        "value": 1.5,
        "wall_time": 1.7e9,
    }
    base.update(over)
    return MetricPoint(**base)


def test_metric_point_permits_nan_and_inf() -> None:
    assert math.isnan(point(value=math.nan).value)
    assert point(value=math.inf).value == math.inf
    assert point(value=-math.inf).value == -math.inf


@pytest.mark.parametrize("bad", [{"step": -1}, {"epoch": -1}, {"key": ""}, {"key": "a\nb"}])
def test_metric_point_rejects_bad_fields(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        point(**bad)


def test_metric_point_rejects_non_finite_wall_time_and_bool_step() -> None:
    with pytest.raises(ValidationError):
        point(wall_time=math.nan)
    with pytest.raises(ValidationError):
        point(step=True)


def test_metric_point_key_length_cap() -> None:
    with pytest.raises(ValidationError):
        point(key="k" * 257)


# --- Artifact ------------------------------------------------------------


def test_artifact_valid_and_checks() -> None:
    a = Artifact(
        run_id=new_run_id(),
        sha256=SHA,
        kind="diff",
        uri=f"artifacts/aa/{SHA}",
        size_bytes=3,
        meta={},
        created_at=NOW,
    )
    assert a.kind == "diff"
    with pytest.raises(ValidationError):
        Artifact(**{**a.model_dump(), "sha256": "ZZ"})
    with pytest.raises(ValidationError):
        Artifact(**{**a.model_dump(), "kind": "movie"})
    with pytest.raises(ValidationError):
        Artifact(**{**a.model_dump(), "size_bytes": -1})


# --- MetricSink ----------------------------------------------------------


def test_metric_sink_is_a_structural_protocol() -> None:
    class Sink:
        def log(self, points: Any, *, step: int, epoch: int | None = None) -> None: ...
        def log_artifact(self, path: Any, *, kind: Any, meta: Any = None) -> Any: ...

    sink: MetricSink = Sink()
    sink.log({"a": 1.0}, step=1)
