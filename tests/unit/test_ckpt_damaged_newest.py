"""Damage to the newest checkpoint is not a licence to serve an older one: falling back below a
known floor needs an explicit, logged acceptance, and never lowers the operator's floor."""

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

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
KEY = "5a" * 32
URI = "s3://bucket-name/damaged"


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def world(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, **kw: Any
) -> tuple[runs.RunHandle, CheckpointManager]:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False,
        remote_sleep=lambda s: None, **kw,
    )  # fmt: skip
    for step in (1, 2, 3):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def truncate(tmp: Path, run: runs.RunHandle, step: int = 3) -> None:
    path = tmp / "ckpt" / run.run_id / "last" / f"step_{step:09d}.safetensors"
    path.chmod(0o600)
    path.write_bytes(path.read_bytes()[:40])


def test_a_truncated_newest_data_file_is_not_silently_replaced_by_an_older_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    truncate(tmp_path, run)
    with pytest.raises(CheckpointUntrustedError) as caught:
        mgr.restore()
    assert caught.value.reason == "rolled_back" and "step 3 is damaged" in str(caught.value)
    run.finish()


def test_the_older_checkpoint_is_available_on_explicit_acceptance_and_it_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    truncate(tmp_path, run)
    restored = mgr.restore(accept_rollback_to=2)
    assert restored is not None and restored.manifest.step == 2
    assert [e for e in events() if e["message"] == "checkpoint_rollback_accepted"]
    run.finish()


def test_a_deleted_remote_data_object_with_its_manifest_kept_does_not_serve_an_older_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    run, mgr = world(tmp_path, monkeypatch, remote=URI, remote_store=fake)
    del fake.objects[f"{run.run_id}/last/step_000000003.safetensors"]
    import shutil

    shutil.rmtree(tmp_path / "ckpt" / run.run_id)  # a fresh node: only the bucket and the store
    with pytest.raises(CheckpointUntrustedError) as caught:
        mgr.restore()
    assert caught.value.reason in {"rolled_back", "checkpoint_missing"}
    run.finish()


def test_min_step_is_not_bypassed_by_a_damaged_newer_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    mgr.min_step = 3
    truncate(tmp_path, run)
    for accept in (None, 2, 1):
        with pytest.raises(CheckpointUntrustedError):
            mgr.restore(accept_rollback_to=accept)
    run.finish()


def test_a_directory_with_no_known_floor_still_walks_back_past_damage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    from cvpilot.ckpt.directory import RunCheckpoints

    run_id = run_meta("p")[0].run_id
    rc = RunCheckpoints(tmp_path / "ckpt", run_id, create=True)
    from ckpt_support import blob

    for step in (1, 2):
        rc.save(
            kind="last", step=step, data=blob(step), format="safetensors", versions={},
            dataset_hash="merkle:" + "c" * 64,
        )  # fmt: skip
    path = tmp_path / "ckpt" / run_id / "last" / "step_000000002.safetensors"
    path.chmod(0o600)
    path.write_bytes(b"x")
    # no store, no key, no min_step: nothing says step 2 mattered, so the damage is just skipped
    assert [v.manifest.step for v in rc.iter_valid("last")] == [1]
