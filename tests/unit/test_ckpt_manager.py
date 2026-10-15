"""CheckpointManager: when to save, what a checkpoint holds, verified restore, retention."""

from __future__ import annotations

import hashlib
import os
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import blob
from test_runs import make

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError
from cvpilot.core.hashing import new_run_id

Events = Callable[[], list[dict[str, Any]]]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Model:
    def __init__(self, scale: float = 1.0) -> None:
        self.scale = scale

    def state_dict(self) -> dict[str, Any]:
        return {"layer.weight": np.full((2, 3), self.scale, dtype=np.float32), "bias": np.zeros(3)}


class Optimizer:
    def state_dict(self) -> dict[str, Any]:
        return {"state": {0: {"exp_avg": np.ones(3)}}, "param_groups": [{"lr": 0.1, "params": [0]}]}


def start(tmp_path: Path, **kw: Any) -> runs.RunHandle:
    meta, repro = make(**{k: kw.pop(k) for k in ("status", "parent_run_id") if k in kw})
    return runs.start_run(meta, repro, store_dir=tmp_path / "store", **kw)


def manager(run: runs.RunHandle, tmp_path: Path, **kw: Any) -> CheckpointManager:
    kw.setdefault("install_signal_handlers", False)
    kw.setdefault("every_n_steps", 10)
    return CheckpointManager(run, dir=tmp_path / "ckpt", **kw)


def names(events: Events) -> list[str]:
    return [e["message"] for e in events()]


def test_save_now_writes_a_verified_checkpoint_and_records_it(
    tmp_path: Path, events: Events
) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    manifest = mgr.save_now(5, model=Model(), optimizer=Optimizer(), rng=True)
    assert manifest.step == 5 and manifest.kind == "last" and manifest.format == "safetensors"
    assert manifest.dataset_hash == "merkle:" + "c" * 64
    assert manifest.versions["python"] and "cvpilot" in manifest.versions
    path = tmp_path / "ckpt" / run.run_id / manifest.path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest.sha256
    # the digest of the manifest file is anchored in the run store
    digest = run.checkpoint_digest(run.run_id, "last", 5)
    raw = (tmp_path / "ckpt" / run.run_id / "last" / "step_000000005.manifest.json").read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest()
    saved = [e for e in events() if e["message"] == "checkpoint_saved"]
    assert len(saved) == 1
    assert saved[0]["step"] == 5 and saved[0]["kind"] == "last"
    assert saved[0]["sha256_prefix"] == manifest.sha256[:12]
    assert saved[0]["size_bytes"] == manifest.size_bytes and "duration_ms" in saved[0]
    assert "checkpoint_verified" in names(events)
    assert saved[0]["run_id"] == run.run_id
    run.finish()


def test_restore_returns_the_saved_state_and_logs_it(tmp_path: Path, events: Events) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    random.seed(3)
    np.random.seed(3)
    mgr.save_now(
        7, model=Model(2.0), optimizer=Optimizer(), rng=True, dataloader_state={"offset": 9}
    )
    state = mgr.restore_latest_valid()
    assert state is not None
    np.testing.assert_array_equal(state["model"]["layer.weight"], np.full((2, 3), 2.0, np.float32))
    assert state["optimizer"]["param_groups"][0]["lr"] == 0.1
    assert set(state["optimizer"]["state"]) == {0}
    assert state["dataloader_state"] == {"offset": 9}
    assert "rng" in state
    assert mgr.last_restore is not None and mgr.last_restore.manifest.step == 7
    restored = [e for e in events() if e["message"] == "checkpoint_restored"]
    assert restored and restored[0]["step"] == 7
    run.finish()


def test_restore_with_nothing_saved_returns_none_and_says_so(
    tmp_path: Path, events: Events
) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    assert mgr.restore_latest_valid() is None
    assert "checkpoint_restore_empty" in names(events)
    run.finish()


def test_maybe_save_follows_the_step_interval(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path, every_n_steps=10)
    assert mgr.maybe_save(3, model=Model()) is None
    assert mgr.maybe_save(10, model=Model()) is not None
    assert mgr.maybe_save(15, model=Model()) is None
    assert mgr.maybe_save(20, model=Model()) is not None
    assert mgr.maybe_save(20, model=Model()) is None  # never twice at one step
    run.finish()


def test_maybe_save_follows_the_time_interval(tmp_path: Path) -> None:
    run = start(tmp_path)
    clock = Clock()
    mgr = manager(run, tmp_path, every_n_steps=None, every_n_minutes=2.0, clock=clock)
    assert mgr.maybe_save(1, model=Model()) is None
    clock.now += 119
    assert mgr.maybe_save(2, model=Model()) is None
    clock.now += 2
    assert mgr.maybe_save(3, model=Model()) is not None
    assert mgr.maybe_save(4, model=Model()) is None
    run.finish()


def test_without_any_interval_the_default_is_ten_minutes(tmp_path: Path) -> None:
    run = start(tmp_path)
    clock = Clock()
    mgr = manager(run, tmp_path, every_n_steps=None, clock=clock)
    assert mgr.maybe_save(1, model=Model()) is None
    clock.now += 601
    assert mgr.maybe_save(2, model=Model()) is not None
    run.finish()


def test_best_checkpoints_are_written_only_when_the_metric_improves(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path, every_n_steps=1000, monitor="mAP50")
    assert mgr.maybe_save(1, model=Model(), metrics={"loss": 1.0}) is None  # monitor absent
    first = mgr.maybe_save(2, model=Model(1.0), metrics={"mAP50": 0.4})
    assert first is not None and first.kind == "best" and first.metric_value == 0.4
    assert mgr.maybe_save(3, model=Model(), metrics={"mAP50": 0.3}) is None
    better = mgr.maybe_save(4, model=Model(2.0), metrics={"mAP50": 0.6}, ema=Model(3.0))
    assert better is not None and better.metric_value == 0.6
    restored = mgr.restore(kind="best")
    assert restored is not None and restored.manifest.step == 4
    assert set(restored.state) == {"model", "ema"}  # weights and EMA only
    run.finish()


def test_min_mode_improves_downwards_and_ignores_non_finite_values(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path, every_n_steps=1000, monitor="loss", mode="min")
    assert mgr.maybe_save(1, model=Model(), metrics={"loss": 2.0}) is not None
    assert mgr.maybe_save(2, model=Model(), metrics={"loss": float("nan")}) is None
    assert mgr.maybe_save(3, model=Model(), metrics={"loss": 3.0}) is None
    assert mgr.maybe_save(4, model=Model(), metrics={"loss": 1.0}) is not None
    run.finish()


def test_last_and_best_can_be_saved_by_one_call(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path, every_n_steps=1, monitor="m")
    primary = mgr.maybe_save(1, model=Model(), metrics={"m": 1.0})
    assert primary is not None and primary.kind == "last"
    kinds = sorted(e.manifest.kind for e in mgr.list() if e.manifest is not None)
    assert kinds == ["best", "last"]
    run.finish()


def test_a_corrupt_newest_checkpoint_is_skipped_counted_and_logged(
    tmp_path: Path, events: Events
) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    mgr.save_now(1, model=Model(1.0))
    mgr.save_now(2, model=Model(2.0))
    newest = tmp_path / "ckpt" / run.run_id / "last" / "step_000000002.safetensors"
    os.chmod(newest, 0o644)
    newest.write_bytes(newest.read_bytes()[:20])
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 1
    assert [(s.step, s.reason) for s in restored.skipped] == [(2, "size_mismatch")]
    skipped = [e for e in events() if e["message"] == "checkpoint_skipped_corrupt"]
    assert skipped and skipped[0]["reason"] == "size_mismatch" and skipped[0]["step"] == 2
    run.finish()


def test_a_manifest_that_verifies_but_whose_file_does_not_load_is_skipped(
    tmp_path: Path,
) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    mgr.save_now(1, model=Model(1.0))
    # a valid safetensors file that was not written by cvpilot (no state skeleton)
    from safetensors.numpy import save as raw_save

    mgr._own.save(
        kind="last",
        step=2,
        data=raw_save({"w": np.zeros(2, dtype=np.float32)}),
        format="safetensors",
        versions={},
        dataset_hash="merkle:" + "c" * 64,
    )
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 1
    assert [(s.step, s.reason) for s in restored.skipped] == [(2, "unloadable")]
    run.finish()


def test_a_resumed_run_restores_from_its_parent_and_links_the_lineage(tmp_path: Path) -> None:
    parent = start(tmp_path)
    mgr = manager(parent, tmp_path)
    mgr.save_now(4, model=Model(4.0))
    parent_id = parent.run_id
    parent.finish("preempted")
    meta = runs.get_run_view("p", parent_id, store_dir=tmp_path / "store").meta
    child_id = new_run_id()
    child_meta = runs.resume_from(meta, run_id=child_id, host="h", name=None)
    _, repro = make(run_id=child_id)
    child = runs.start_run(child_meta, repro, store_dir=tmp_path / "store")
    child_mgr = manager(child, tmp_path)
    restored = child_mgr.restore()
    assert restored is not None
    assert restored.run_id == parent_id and restored.manifest.step == 4
    assert child.meta.parent_run_id == parent_id
    np.testing.assert_array_equal(
        restored.state["model"]["layer.weight"], np.full((2, 3), 4.0, np.float32)
    )
    child_mgr.save_now(9, model=Model(9.0))  # the step timeline continues in the child's own dir
    assert (tmp_path / "ckpt" / child_id / "last" / "step_000000009.safetensors").exists()
    child.finish()


def test_keep_last_applies_after_every_save(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path, keep_last=2)
    for step in range(1, 6):
        mgr.save_now(step, model=Model(float(step)))
    steps = sorted(e.step for e in mgr.list() if e.status == "valid" and e.step is not None)
    assert steps == [4, 5]
    run.finish()


def test_keep_best_applies_after_every_best_save(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path, every_n_steps=1000, monitor="m", keep_best=2)
    for step, value in enumerate([0.1, 0.2, 0.3, 0.4], start=1):
        mgr.maybe_save(step, model=Model(), metrics={"m": value})
    best = sorted(e.step for e in mgr.list() if e.manifest and e.manifest.kind == "best")
    assert best == [3, 4]
    run.finish()


def test_verify_accepts_good_manifests_and_rejects_damaged_files(
    tmp_path: Path, events: Events
) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    good = mgr.save_now(1, model=Model())
    assert mgr.verify(good) is True
    path = tmp_path / "ckpt" / run.run_id / good.path
    os.chmod(path, 0o644)
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")
    assert mgr.verify(good) is False
    assert "checkpoint_verify_failed" in names(events)
    run.finish()


def test_a_disk_error_is_raised_to_the_trainer_and_logged(
    tmp_path: Path, events: Events, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    mgr.save_now(1, model=Model())

    def full(fd: int, data: Any) -> int:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "write", full)
    with pytest.raises(CheckpointError):
        mgr.save_now(2, model=Model())
    monkeypatch.undo()
    failed = [e for e in events() if e["message"] == "checkpoint_save_failed"]
    assert failed and failed[0]["level"] == "ERROR" and failed[0]["step"] == 2
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 1
    run.finish()


@pytest.mark.parametrize(
    "kw",
    [
        {"keep_last": 0},
        {"keep_best": 0},
        {"mode": "up"},
        {"every_n_steps": 0},
        {"every_n_steps": True},
        {"every_n_minutes": 0},
        {"every_n_minutes": float("nan")},
        {"grace_seconds": 0},
        {"grace_seconds": 100_000},
        {"format": "pickle"},
        {"monitor": ""},
        {"max_checkpoint_bytes": 0},
    ],
)
def test_invalid_settings_are_refused_before_anything_is_written(
    tmp_path: Path, kw: dict[str, Any]
) -> None:
    run = start(tmp_path)
    with pytest.raises(ValueError):
        manager(run, tmp_path, **kw)
    assert not (tmp_path / "ckpt").exists()
    run.finish()


def test_a_run_like_object_without_the_needed_methods_is_refused(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        CheckpointManager(object(), dir=tmp_path / "ckpt")  # type: ignore[arg-type]


def test_a_model_without_state_dict_is_refused(tmp_path: Path) -> None:
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    with pytest.raises(CheckpointError):
        mgr.save_now(1, model=object())
    with pytest.raises(CheckpointError):
        mgr.save_now(2)  # a last checkpoint needs a model
    run.finish()


def test_the_blob_helper_produces_files_the_manager_can_restore(tmp_path: Path) -> None:
    """Sanity: files written by the low-level layer restore through the manager."""
    run = start(tmp_path)
    mgr = manager(run, tmp_path)
    mgr._own.save(
        kind="last",
        step=3,
        data=blob(3),
        format="safetensors",
        versions={},
        dataset_hash="merkle:" + "c" * 64,
    )
    restored = mgr.restore()
    assert restored is not None and restored.state["step"] == 3
    run.finish()
