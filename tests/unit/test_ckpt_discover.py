"""A restarted script has a new run id and, on a fresh node, no run store: restore must be able to
find the previous run's checkpoints on the remote by itself, through the same gate."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError
from cvpilot.core.hashing import new_run_id

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
URI = "s3://bucket-name/discover"


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(2, value, dtype=np.float32)}


def writer(
    tmp: Path, fake: FakeRemote, *, steps: tuple[int, ...] = (1, 2, 3), tag: str = "w", **kw: Any
) -> str:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp / f"store-{tag}")
    mgr = CheckpointManager(
        run, dir=tmp / f"ckpt-{tag}", remote=URI, remote_store=fake, every_n_steps=1,
        keep_last=20, install_signal_handlers=False, remote_sleep=lambda s: None, **kw,
    )  # fmt: skip
    for step in steps:
        mgr.save_now(step, model=weights(float(step)))
    mgr.close()
    run.finish("preempted")
    return run.run_id


def fresh(tmp: Path, fake: FakeRemote, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager]:
    """A new run in a store and a checkpoint directory that have never seen the earlier run."""
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp / "store-new")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt-new", remote=URI, remote_store=fake, every_n_steps=1,
        install_signal_handlers=False, remote_sleep=lambda s: None, **kw,
    )  # fmt: skip
    return run, mgr


def test_discovery_finds_the_previous_run_on_a_fresh_node(tmp_path: Path) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake)
    run, mgr = fresh(tmp_path, fake)
    assert mgr.restore_latest_valid() is None  # no lineage, nothing local: the old behaviour
    restored = mgr.restore(discover=True)
    assert restored is not None and restored.run_id == old and restored.manifest.step == 3
    assert np.all(restored.state["model"]["w"] == 3.0)
    state = mgr.restore_latest_valid(discover=True)
    assert isinstance(state, dict) and "model" in state
    run.finish()


def test_the_newest_step_across_runs_wins_and_ties_go_to_the_newer_run(tmp_path: Path) -> None:
    fake = FakeRemote()
    writer(tmp_path, fake, steps=(1, 2, 3), tag="a")
    newer = writer(tmp_path, fake, steps=(1, 2, 3), tag="b")
    run, mgr = fresh(tmp_path, fake)
    restored = mgr.restore(discover=True)
    assert restored is not None and restored.manifest.step == 3 and restored.run_id == newer
    run.finish()


def test_discovery_is_bounded(tmp_path: Path, events: Events) -> None:
    fake = FakeRemote()
    for _ in range(1100):
        fake.objects[f"{new_run_id()}/last/x"] = b"x"
    run, mgr = fresh(tmp_path, fake)
    assert mgr.restore(discover=True) is None
    assert any(e["message"] == "checkpoint_download_failed" for e in events())
    run.finish()


def test_discovery_without_a_remote_is_a_usage_error(tmp_path: Path) -> None:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "s")
    mgr = CheckpointManager(run, dir=tmp_path / "c", every_n_steps=1, install_signal_handlers=False)
    with pytest.raises(ValueError, match="remote"):
        mgr.restore(discover=True)
    run.finish()


def test_discovery_goes_through_the_gate_before_any_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    writer(tmp_path, fake)  # written without a key: untagged
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "ab" * 32)
    run, mgr = fresh(tmp_path, fake, require_trust="authenticated")
    fake.reads.clear()
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(discover=True)
    data_reads = [k for k in fake.reads if k.endswith(".safetensors")]
    assert data_reads == []  # only manifests were looked at
    run.finish()


def test_run_ids_can_be_given_to_restore_latest_valid(tmp_path: Path) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake)
    run, mgr = fresh(tmp_path, fake)
    state = mgr.restore_latest_valid(run_ids=[old])
    assert isinstance(state, dict) and np.all(state["model"]["w"] == 3.0)
    with_level = mgr.restore_latest_valid(run_ids=[old], with_trust=True)
    assert with_level is not None and with_level.run_id == old
    run.finish()
