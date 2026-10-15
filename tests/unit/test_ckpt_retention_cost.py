"""Retention must not re-hash every kept checkpoint after every save."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")


def make(tmp: Path, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    return run, CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


def count_data_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real = RunCheckpoints._check_data

    def spy(self: RunCheckpoints, kind_fd: int, manifest: Any, *, load: bool) -> Any:
        calls.append(manifest.path)
        return real(self, kind_fd, manifest, load=load)

    monkeypatch.setattr(RunCheckpoints, "_check_data", spy)
    return calls


def test_steady_state_retention_hashes_nothing_it_already_verified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, keep_last=3)
    calls = count_data_checks(monkeypatch)
    for step in range(1, 11):
        mgr.save_now(step, model={"w": np.full(4, float(step))})
    assert calls == []  # every file was verified when it was saved; retention reused that
    assert [e.step for e in mgr.list()] == [10, 9, 8]
    run.finish()


def test_a_file_that_changed_is_hashed_again_before_it_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, keep_last=2)
    for step in (1, 2):
        mgr.save_now(step, model={"w": np.full(4, float(step))})
    path = tmp_path / "ckpt" / run.run_id / "last" / "step_000000002.safetensors"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes()[:-1] + b"\x00")  # same size, other content, newer ctime
    os.utime(path, ns=(1, 1))
    calls = count_data_checks(monkeypatch)
    mgr.save_now(3, model={"w": np.full(4, 3.0)})
    assert any(c.endswith("step_000000002.safetensors") for c in calls)
    # the damaged newest-but-one no longer counts: step 3 and step 1 are what was kept
    assert [e.step for e in mgr.list() if e.status == "valid"] == [3, 1]
    run.finish()


def test_a_corrupt_older_file_is_left_alone_and_never_counted(tmp_path: Path) -> None:
    run, mgr = make(tmp_path, keep_last=1)
    mgr.save_now(1, model={"w": np.ones(4)})
    mgr.save_now(2, model={"w": np.ones(4)})
    mgr.save_now(3, model={"w": np.ones(4)})
    mgr.save_now(4, model={"w": np.ones(4)})
    assert [e.step for e in mgr.list()] == [4]
    run.finish()


def test_a_first_prune_in_a_new_process_verifies_then_caches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, keep_last=5)
    for step in (1, 2, 3):
        mgr.save_now(step, model={"w": np.full(4, float(step))})
    fresh = RunCheckpoints(tmp_path / "ckpt", run.run_id)
    calls = count_data_checks(monkeypatch)
    fresh.prune("last", keep=2)  # a new object has verified nothing yet
    first = len(calls)
    assert first >= 2
    fresh.prune("last", keep=2)
    assert len(calls) == first  # the second pass reuses what the first verified
    run.finish()
