"""Rollback and replay: an older, validly tagged checkpoint must not be served silently once newer
ones are known to have existed (the tag-checked pointer, the run store, an external high-water
mark)."""

from __future__ import annotations

import os
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
KEY = "ab" * 32  # 64 hex characters: 32 bytes


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(4, value, dtype=np.float32)}


def start(tmp_path: Path, *, store: str = "store") -> runs.RunHandle:
    meta, repro = run_meta()
    return runs.start_run(meta, repro, store_dir=tmp_path / store)


def manager(run: runs.RunHandle, tmp_path: Path, **kw: Any) -> CheckpointManager:
    kw.setdefault("dir", tmp_path / "ckpt")
    kw.setdefault("keep_last", 10)
    return CheckpointManager(run, every_n_steps=1, install_signal_handlers=False, **kw)


def saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int = 3, **kw: Any
) -> tuple[Any, Any]:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    run = start(tmp_path)
    mgr = manager(run, tmp_path, require_trust="authenticated", **kw)
    for step in range(1, n + 1):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def remove_step(tmp_path: Path, run_id: str, step: int, kind: str = "last") -> None:
    folder = tmp_path / "ckpt" / run_id / kind
    for name in (f"step_{step:09d}.manifest.json", f"step_{step:09d}.safetensors"):
        os.unlink(folder / name)


def remove_pointer(tmp_path: Path, run_id: str) -> None:
    os.unlink(tmp_path / "ckpt" / run_id / "LATEST")


def test_deleting_the_newest_manifests_is_a_rollback_not_a_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    remove_step(tmp_path, run.run_id, 3)
    remove_step(tmp_path, run.run_id, 2)  # LATEST still names step 3
    with pytest.raises(CheckpointUntrustedError) as info:
        mgr.restore()
    assert info.value.reason == "rolled_back"
    logged = [e for e in events() if e["message"] == "checkpoint_untrusted"]
    assert any(e.get("reason") == "rolled_back" and e["level"] == "ERROR" for e in logged)
    run.finish()


def test_the_run_store_remembers_what_the_pointer_forgot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    remove_step(tmp_path, run.run_id, 3)
    remove_step(tmp_path, run.run_id, 2)
    remove_pointer(tmp_path, run.run_id)
    with pytest.raises(CheckpointUntrustedError, match="rolled back"):
        mgr.restore()
    run.finish()


def test_a_fresh_machine_with_only_the_key_cannot_see_pure_deletion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The documented limit: with no pointer, store or external mark there is nothing to compare."""
    run, mgr = saved(tmp_path, monkeypatch)
    remove_step(tmp_path, run.run_id, 3)
    remove_step(tmp_path, run.run_id, 2)
    remove_pointer(tmp_path, run.run_id)
    run.finish("preempted")
    fresh = start(tmp_path, store="elsewhere")  # a store that has never heard of that run
    reader = manager(fresh, tmp_path, require_trust="authenticated")
    restored = reader.restore(run_ids=[run.run_id])
    assert restored is not None and restored.manifest.step == 1
    assert restored.trust == "authenticated"
    fresh.finish()


def test_a_damaged_newest_checkpoint_needs_an_explicit_acceptance_to_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    newest = tmp_path / "ckpt" / run.run_id / "last" / "step_000000003.safetensors"
    newest.chmod(0o644)
    newest.write_bytes(newest.read_bytes()[:30])
    with pytest.raises(CheckpointUntrustedError, match="step 3 is damaged"):
        mgr.restore()
    restored = mgr.restore(accept_rollback_to=2)
    assert restored is not None and restored.manifest.step == 2
    assert [s.reason for s in restored.skipped][:1] == ["size_mismatch"]
    run.finish()


def test_a_store_that_is_ahead_of_the_directory_is_a_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    ahead = mgr._own.existing("last", 3)
    assert ahead is not None
    five = ahead.model_copy(update={"step": 5, "path": "last/step_000000005.safetensors"})
    from cvpilot.core.hashing import checkpoint_chain

    five = five.model_copy(
        update={
            "chain_sha256": checkpoint_chain(
                five.prev_chain_sha256, five.run_id, "last", 5, five.sha256, five.size_bytes
            )
        }
    )
    run.record_checkpoint(five, "d" * 64)  # the store saw step 5; the directory never has it
    with pytest.raises(CheckpointUntrustedError) as info:
        mgr.restore()
    assert info.value.reason == "rolled_back"
    run.finish()


def test_a_deliberate_rollback_can_be_accepted_and_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    remove_step(tmp_path, run.run_id, 3)
    remove_step(tmp_path, run.run_id, 2)
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(accept_rollback_to=2)  # step 1 is below even the accepted floor
    restored = mgr.restore(accept_rollback_to=1)
    assert restored is not None and restored.manifest.step == 1
    accepted = [e for e in events() if e["message"] == "checkpoint_rollback_accepted"]
    assert accepted and accepted[0]["step"] == 1 and accepted[0]["floor"] == 3
    assert accepted[0]["level"] == "WARNING"
    run.finish()


def test_an_external_high_water_mark_anchors_a_fresh_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    run.finish("preempted")
    fresh = start(tmp_path, store="elsewhere")
    behind = manager(fresh, tmp_path, require_trust="authenticated", min_step=5)
    with pytest.raises(CheckpointUntrustedError) as info:
        behind.restore(run_ids=[run.run_id])
    assert info.value.reason == "rolled_back"
    fine = manager(fresh, tmp_path, require_trust="authenticated", min_step=3)
    restored = fine.restore(run_ids=[run.run_id])
    assert restored is not None and restored.manifest.step == 3
    fresh.finish()


def test_the_high_water_mark_comes_from_the_environment_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "9")
    other = manager(run, tmp_path, require_trust="authenticated")
    with pytest.raises(CheckpointUntrustedError):
        other.restore()
    run.finish()


def test_nothing_restored_when_the_store_knows_checkpoints_existed_is_a_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch, n=2)
    for step in (1, 2):
        remove_step(tmp_path, run.run_id, step)
    remove_pointer(tmp_path, run.run_id)
    with pytest.raises(CheckpointUntrustedError) as info:
        mgr.restore()
    assert info.value.reason == "rolled_back"
    assert mgr.restore(accept_rollback_to=0) is None  # start from scratch, on purpose
    run.finish()


def test_a_stale_remote_listing_is_a_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    fake = FakeRemote()
    run = start(tmp_path)
    writer = CheckpointManager(
        run, dir=tmp_path / "a", remote="s3://bucket-name/x", remote_store=fake, every_n_steps=1,
        keep_last=10, require_trust="authenticated", install_signal_handlers=False,
        remote_sleep=lambda s: None,
    )  # fmt: skip
    for step in (1, 2, 3):
        writer.save_now(step, model=weights(float(step)))
    run.finish("preempted")
    for suffix in (".manifest.json", ".safetensors"):
        del fake.objects[f"{run.run_id}/last/step_000000003{suffix}"]  # a stale or truncated bucket
    again = start(tmp_path)  # the same store: it knows step 3 was saved
    reader = CheckpointManager(
        again, dir=tmp_path / "b", remote="s3://bucket-name/x", remote_store=fake,
        every_n_steps=1, require_trust="authenticated", install_signal_handlers=False,
        remote_sleep=lambda s: None,
    )  # fmt: skip
    with pytest.raises(CheckpointUntrustedError) as info:
        reader.restore(run_ids=[run.run_id])
    assert info.value.reason == "rolled_back"
    assert reader.restore(run_ids=[run.run_id], accept_rollback_to=2) is not None
    again.finish()


def test_the_default_policy_any_does_not_look_for_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    run = start(tmp_path)
    mgr = manager(run, tmp_path, require_trust="any")
    for step in (1, 2, 3):
        mgr.save_now(step, model=weights(float(step)))
    remove_step(tmp_path, run.run_id, 3)
    remove_step(tmp_path, run.run_id, 2)
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 1  # documented: use a strict policy
    run.finish()


def test_anchored_mode_ignores_the_unauthenticated_pointer_but_uses_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    run = start(tmp_path)
    mgr = manager(run, tmp_path, require_trust="anchored")
    for step in (1, 2, 3):
        mgr.save_now(step, model=weights(float(step)))
    remove_step(tmp_path, run.run_id, 3)
    remove_step(tmp_path, run.run_id, 2)
    with pytest.raises(CheckpointUntrustedError) as info:
        mgr.restore()
    assert info.value.reason == "rolled_back"
    run.finish()


def test_the_floor_is_reported_by_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = saved(tmp_path, monkeypatch)
    assert mgr._own.freshness_floor("last") == 3
    assert mgr._own.freshness_floor("best") == 0
    run.finish()
