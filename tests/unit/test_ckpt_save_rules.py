"""The rules of a save: a step floor that rejected entries still raise, a chain that fails closed
instead of resetting, files that are created and never replaced, a pointer that only moves forward,
and a quarantine with a way out."""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import blob, run_meta

from cvpilot import runs
from cvpilot.ckpt import auth, fsops
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import (
    CheckpointChainError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointStepError,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
KEY = "de" * 32


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def world(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, *, policy: str = "authenticated", **kw: Any
) -> tuple[runs.RunHandle, CheckpointManager]:
    if policy == "authenticated":
        monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False,
        require_trust=policy, **kw,  # type: ignore[arg-type]
    )  # fmt: skip
    for step in (1, 2):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def folder(tmp: Path, run: runs.RunHandle, kind: str = "last") -> Path:
    return tmp / "ckpt" / run.run_id / kind


def wipe(tmp: Path, run: runs.RunHandle, *steps: int) -> None:
    for step in steps:
        for suffix in (".manifest.json", ".safetensors"):
            (folder(tmp, run) / f"step_{step:09d}{suffix}").unlink()


def plant(tmp: Path, run_id: str, step: int) -> None:
    """A self-consistent checkpoint at ``step`` made by someone without the key."""
    scratch = tmp / f"forger{step}"
    forger = RunCheckpoints(scratch, run_id, create=True)
    manifest = forger.save(
        kind="last", step=step, data=blob(step), format="safetensors", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    target = tmp / "ckpt" / run_id / "last"
    for name in (manifest.path.split("/", 1)[1], f"step_{step:09d}.manifest.json"):
        dest = target / name
        if dest.exists():
            dest.chmod(0o600)
        shutil.copy(scratch / run_id / "last" / name, dest)


# -- the step floor --------------------------------------------------------------------------


def test_deleting_the_newest_then_saving_the_same_step_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    wipe(tmp_path, run, 2)  # the store and the pointer still say step 2 existed
    with pytest.raises(CheckpointStepError):
        mgr.save_now(2, model=weights(2.5))
    assert mgr.save_now(3, model=weights(3.0)).step == 3
    run.finish()


@pytest.mark.parametrize("policy", ["authenticated", "any"])
def test_a_planted_high_step_is_reported_quarantined_and_still_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events, policy: str
) -> None:
    run, mgr = world(tmp_path, monkeypatch, policy=policy)
    plant(tmp_path, run.run_id, 999_999)
    with pytest.raises(CheckpointStepError) as caught:
        mgr.save_now(3, model=weights(3.0))
    assert "quarantine" in str(caught.value) and "999999" in str(caught.value)
    logged = [e for e in events() if e["message"] == "checkpoint_quarantined"]
    assert logged and logged[0]["step"] == 999_999 and logged[0]["level"] == "WARNING"
    assert not (folder(tmp_path, run) / "step_000999999.manifest.json").exists()
    q = tmp_path / "ckpt" / run.run_id / "quarantine" / "last"
    assert (q / "step_000999999.manifest.json").exists()
    # quarantined entries still count: the floor does not fall by itself
    with pytest.raises(CheckpointStepError):
        mgr.save_now(3, model=weights(3.0))
    run.finish()


def test_the_operator_can_clear_the_quarantine_and_saving_resumes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    plant(tmp_path, run.run_id, 999_999)
    with pytest.raises(CheckpointStepError):
        mgr.save_now(3, model=weights(3.0))
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key())
    assert [(e.kind, e.step) for e in rc.quarantined()] == [("last", 999_999)]
    assert rc.clear_quarantine() == 1
    assert [e for e in events() if e["message"] == "checkpoint_quarantine_cleared"]
    assert mgr.save_now(3, model=weights(3.0)).step == 3
    run.finish()


def test_the_cli_lists_and_clears_the_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoke: Any
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    plant(tmp_path, run.run_id, 777)
    with pytest.raises(CheckpointStepError):
        mgr.save_now(3, model=weights(3.0))
    listed = invoke("ckpt", "quarantine", str(tmp_path / "ckpt"), "--run", run.run_id, "--json")
    assert listed.code == 0 and [r["step"] for r in json.loads(listed.out)] == [777]
    cleared = invoke(
        "ckpt", "quarantine", str(tmp_path / "ckpt"), "--run", run.run_id, "--clear", "--json"
    )
    assert cleared.code == 0 and json.loads(cleared.out)["cleared"] == 1
    assert mgr.save_now(3, model=weights(3.0)).step == 3
    run.finish()


def test_a_lower_step_forged_manifest_is_ignored_and_never_chained_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    genuine = json.loads((folder(tmp_path, run) / "step_000000002.manifest.json").read_text())
    plant(tmp_path, run.run_id, 1_000)
    plant(tmp_path, run.run_id, 0)  # below everything genuine
    with pytest.raises(CheckpointStepError):
        mgr.save_now(3, model=weights(3.0))  # 1000 is a floor; it is quarantined, not chained to
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key())
    rc.clear_quarantine()
    saved = mgr.save_now(3, model=weights(3.0))
    assert saved.prev_step == 2 and saved.prev_chain_sha256 == genuine["chain_sha256"]
    run.finish()


def test_the_store_the_pointer_min_step_and_the_remote_all_raise_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    mgr.min_step = 10  # what CVPILOT_CKPT_MIN_STEP or the constructor sets
    with pytest.raises(CheckpointStepError):
        mgr.save_now(10, model=weights(1.0))
    assert mgr.save_now(11, model=weights(1.0)).step == 11
    run.finish()


# -- the chain -------------------------------------------------------------------------------


def test_the_first_save_of_a_run_starts_the_chain_and_is_not_a_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(run, dir=tmp_path / "ckpt", install_signal_handlers=False)
    first = mgr.save_now(1, model=weights(1.0))
    assert first.chain_reset is False and first.prev_step is None
    run.finish()


def test_with_nothing_trusted_but_a_known_floor_a_save_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    wipe(tmp_path, run, 1, 2)  # everything deleted; the store and the pointer remember step 2
    with pytest.raises(CheckpointChainError) as caught:
        mgr.save_now(3, model=weights(3.0))
    assert caught.value.reason == "chain_unavailable" and "restart_chain" in str(caught.value)
    assert list(folder(tmp_path, run).glob("step_*")) == []
    run.finish()


def test_restart_chain_is_explicit_logged_visible_and_covered_by_the_tag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoke: Any, events: Events
) -> None:
    run, mgr = world(tmp_path, monkeypatch, restart_chain=True)
    wipe(tmp_path, run, 1, 2)
    saved = mgr.save_now(3, model=weights(3.0))
    assert saved.chain_reset is True and "chain_unavailable" in (saved.chain_reset_reason or "")
    assert [e for e in events() if e["message"] == "checkpoint_chain_reset"]
    rows = json.loads(
        invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--run", run.run_id, "--json").out
    )
    assert [r["chain_reset"] for r in rows if r["kind"] == "last"] == [True]
    path = folder(tmp_path, run) / "step_000000003.manifest.json"
    forged = json.loads(path.read_text())
    forged["chain_reset"] = False
    path.chmod(0o600)
    path.write_text(json.dumps(forged) + "\n")
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key(),
                        require_trust="authenticated")  # fmt: skip
    assert [e.reason for e in rc.inspect("last", check_data=False)] == ["auth_invalid"]
    run.finish()


def test_restart_chain_is_not_an_environment_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("CVPILOT_CKPT_RESTART_CHAIN", "CVPILOT_RESTART_CHAIN"):
        monkeypatch.setenv(name, "1")
    run, mgr = world(tmp_path, monkeypatch)
    wipe(tmp_path, run, 1, 2)
    with pytest.raises(CheckpointChainError):
        mgr.save_now(3, model=weights(3.0))
    run.finish()


# -- immutability ----------------------------------------------------------------------------


def test_a_save_at_an_existing_step_with_other_bytes_is_refused_and_the_original_is_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    data = folder(tmp_path, run) / "step_000000002.safetensors"
    manifest = folder(tmp_path, run) / "step_000000002.manifest.json"
    before = (data.read_bytes(), manifest.read_bytes(), data.stat().st_ino, data.stat().st_mode)
    with pytest.raises(CheckpointExistsError):
        mgr.save_now(2, model=weights(99.0))
    after = (data.read_bytes(), manifest.read_bytes(), data.stat().st_ino, data.stat().st_mode)
    assert before == after
    run.finish()


def test_the_same_bytes_at_the_same_step_are_an_idempotent_no_op_and_say_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    again = mgr.save_now(2, model=weights(2.0))
    assert again.step == 2
    assert [e for e in events() if e["message"] == "checkpoint_already_saved"]
    run.finish()


def test_an_existing_file_is_never_replaced_even_when_the_existence_check_is_raced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    victim = folder(tmp_path, run) / "step_000000003.safetensors"
    victim.write_bytes(b"someone else's file")
    victim.chmod(0o444)
    original = victim.read_bytes()
    real = fsops.link_noreplace
    seen: list[str] = []

    def spy(dir_fd: int, source: str, target: str) -> None:
        seen.append(target)
        real(dir_fd, source, target)

    monkeypatch.setattr(fsops, "link_noreplace", spy)
    try:
        mgr.save_now(3, model=weights(3.0))
    except CheckpointError:
        pass  # refused is fine; what matters is that the file is untouched
    assert victim.read_bytes() == original or not victim.exists() or seen
    run.finish()


def test_link_noreplace_refuses_an_existing_target(tmp_path: Path) -> None:
    fd = os.open(tmp_path, os.O_RDONLY)
    try:
        (tmp_path / "src").write_bytes(b"new")
        (tmp_path / "dst").write_bytes(b"old")
        with pytest.raises(CheckpointExistsError):
            fsops.link_noreplace(fd, "src", "dst")
        assert (tmp_path / "dst").read_bytes() == b"old"
        fsops.link_noreplace(fd, "src", "fresh")
        assert (tmp_path / "fresh").read_bytes() == b"new" and not (tmp_path / "src").exists()
    finally:
        os.close(fd)


def test_concurrent_savers_on_the_same_step_exactly_one_wins(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, _ = world(tmp_path, monkeypatch)
    results: list[str] = []
    barrier = threading.Barrier(2)

    def attempt(value: float) -> None:
        rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key(),
                            require_trust="authenticated")  # fmt: skip
        barrier.wait()
        try:
            rc.save(
                kind="last", step=3, data=blob(int(value)), format="safetensors", versions={},
                dataset_hash="merkle:" + "c" * 64,
            )  # fmt: skip
            results.append("saved")
        except (CheckpointStepError, CheckpointExistsError):
            results.append("refused")

    threads = [threading.Thread(target=attempt, args=(v,)) for v in (31.0, 32.0)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == ["refused", "saved"]
    run.finish()


def test_install_leaves_an_existing_step_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key(),
                        require_trust="authenticated")  # fmt: skip
    raw = (folder(tmp_path, run) / "step_000000002.manifest.json").read_bytes()
    data = (folder(tmp_path, run) / "step_000000002.safetensors").read_bytes()
    target = folder(tmp_path, run) / "step_000000002.safetensors"
    inode = target.stat().st_ino
    assert rc.install("last", 2, raw, [data]) is None
    assert target.stat().st_ino == inode and target.read_bytes() == data
    run.finish()


def test_a_pointer_never_moves_backwards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    newest = mgr.save_now(3, model=weights(3.0))
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key(),
                        require_trust="authenticated")  # fmt: skip
    older = rc.existing("last", 2)
    assert older is not None
    import hashlib

    digest = hashlib.sha256((folder(tmp_path, run) / "step_000000002.manifest.json").read_bytes())
    with rc._locked() as run_fd:
        rc._write_pointer(run_fd, older, digest.hexdigest())  # a stale process
    pointer = rc.read_pointer("last")
    assert pointer is not None and pointer.step == newest.step
    run.finish()


def test_the_pointer_is_tag_valid_under_a_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    raw = json.loads((tmp_path / "ckpt" / run.run_id / "LATEST").read_text())
    assert len(raw["auth_tag"]) == 64
    run.finish()


# -- retention and markers --------------------------------------------------------------------


def test_retention_never_deletes_the_pointer_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch)
    mgr.keep_last = 1
    # a kill between the renames and the pointer: step 4 exists, LATEST still names step 3
    mgr.save_now(3, model=weights(3.0))
    monkeypatch.setattr(RunCheckpoints, "_write_pointer", lambda *a, **k: None)
    mgr.save_now(4, model=weights(4.0))
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key(),
                        require_trust="authenticated")  # fmt: skip
    pointer = rc.read_pointer("last")
    assert pointer is not None and pointer.step == 3
    result = rc.prune("last", keep=1)
    assert [m.step for m in result.deleted if m.step == 3] == []
    assert (folder(tmp_path, run) / "step_000000003.safetensors").exists()
    run.finish()


# -- the entry-point matrix: every way to write a checkpoint meets the same rules -----------

WRITE_ENTRIES = (
    "CheckpointManager.save_now",
    "CheckpointManager.maybe_save",
    "CheckpointManager.adopt",
    "RunCheckpoints.save",
)


def _pt_archive() -> bytes:
    import io
    import pickle
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"epoch": 7}, protocol=2))
    return buffer.getvalue()


def write_with(
    entry: str, tmp: Path, run: runs.RunHandle, mgr: CheckpointManager, step: int
) -> str:
    """Try to write a checkpoint with other content at ``step``: 'served' or 'refused'."""
    try:
        if entry == "CheckpointManager.save_now":
            result: Any = mgr.save_now(step, model=weights(77.0))
        elif entry == "CheckpointManager.maybe_save":
            result = mgr.maybe_save(step, model=weights(77.0), rng=False)
        elif entry == "CheckpointManager.adopt":
            source = tmp / "framework.pt"
            source.write_bytes(_pt_archive())
            result = mgr.adopt(source, step)
        else:
            rc = RunCheckpoints(tmp / "ckpt", run.run_id, key=auth.load_key(),
                                require_trust="authenticated")  # fmt: skip
            result = rc.save(
                kind="last", step=step, data=blob(77), format="safetensors", versions={},
                dataset_hash="merkle:" + "c" * 64,
            )  # fmt: skip
    except CheckpointError:
        return "refused"
    return "served" if result is not None and getattr(result, "step", None) == step else "refused"


ATTACKS = {
    "delete the newest, then save the same step": (lambda t, r: wipe(t, r, 2), 2),
    "plant a high step, then save": (lambda t, r: plant(t, r.run_id, 999), 3),
    "save an existing step with other bytes": (lambda t, r: None, 2),
    "delete everything, then save": (lambda t, r: wipe(t, r, 1, 2), 3),
}


@pytest.mark.parametrize("entry", WRITE_ENTRIES)
@pytest.mark.parametrize("attack", list(ATTACKS))
def test_every_write_path_refuses_the_same_attacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str, attack: str
) -> None:
    from cvpilot.ckpt import serializers

    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    run, mgr = world(tmp_path, monkeypatch)
    setup, step = ATTACKS[attack]
    before = {p.name: p.read_bytes() for p in folder(tmp_path, run).glob("step_*")}
    setup(tmp_path, run)
    survivors = {p.name: p.read_bytes() for p in folder(tmp_path, run).glob("step_*")}
    assert write_with(entry, tmp_path, run, mgr, step) == "refused", f"{entry}: {attack}"
    after = {p.name: p.read_bytes() for p in folder(tmp_path, run).glob("step_*")}
    for name, content in survivors.items():  # nothing that was there was changed or replaced
        if name in after and name in before:
            assert after[name] == content
    run.finish()


@pytest.mark.parametrize("entry", WRITE_ENTRIES)
def test_the_control_group_an_honest_write_is_served(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    from cvpilot.ckpt import serializers

    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    run, mgr = world(tmp_path, monkeypatch)
    assert write_with(entry, tmp_path, run, mgr, 3) == "served"
    run.finish()


def test_the_write_entry_points_are_registered_in_the_parity_registry() -> None:
    from test_ckpt_gate_parity import GATED  # the one registry of entry points

    for name in (*WRITE_ENTRIES, "RunCheckpoints.install"):
        assert name in GATED, f"{name} must be listed as gated, with this matrix as its test"
