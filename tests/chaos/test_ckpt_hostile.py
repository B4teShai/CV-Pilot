"""Hostile and unlucky conditions around checkpoint files, with real processes where it matters."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import blob, data_file, forge_manifest, forge_pointer, kind_dir, make_writable
from ckpt_support import run_meta as make_meta

from cvpilot import runs
from cvpilot.ckpt import fsops
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.errors import CheckpointUntrustedError
from cvpilot.core.hashing import new_run_id

pytestmark = [
    pytest.mark.chaos,
    pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX"),
]
HERE = Path(__file__).parent


def started(tmp_path: Path, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = make_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    return run, CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(32, value, dtype=np.float32)}


def test_a_full_disk_fails_the_save_cleanly_and_the_next_one_succeeds(tmp_path: Path) -> None:
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(HERE / "_ckpt_limit.py"), str(tmp_path), str(tmp_path / "store")],
        capture_output=True,
        text=True,
        timeout=120,
        check=True,
    )
    lines = [x for x in proc.stdout.splitlines() if x.split(" ")[0] in ("RUN", "SAVED", "FAILED")]
    run_id = lines[0].split()[1]
    assert lines[1:] == ["SAVED 1", "FAILED 2 CheckpointError", "SAVED 3"]
    rc = RunCheckpoints(tmp_path / "ckpt", run_id)
    assert [v.manifest.step for v in rc.iter_valid("last")] == [3, 1]
    assert not [p for p in kind_dir(rc).iterdir() if p.name.endswith(".tmp")]


def test_concurrent_saver_processes_keep_one_ordered_valid_chain(tmp_path: Path) -> None:
    run_id = new_run_id()
    procs = [
        subprocess.Popen(  # noqa: S603
            [sys.executable, str(HERE / "_ckpt_concurrent.py"), str(tmp_path), run_id,
             str(offset), "3", "30"],
            stdout=subprocess.PIPE,
            text=True,
        )
        for offset in (1, 2, 3)
    ]  # fmt: skip
    outputs = [p.communicate(timeout=180)[0] for p in procs]
    assert [p.returncode for p in procs] == [0, 0, 0], outputs
    rc = RunCheckpoints(tmp_path, run_id)
    entries = rc.inspect("last")
    assert entries and all(e.status == "valid" for e in entries), entries
    steps = [e.step for e in entries]
    assert steps == sorted(steps, reverse=True) and len(set(steps)) == len(steps)
    saved_total = sum(int(o.split()[1]) for o in outputs)
    assert saved_total == len(steps)
    chain = [e.manifest for e in entries if e.manifest is not None]
    for newer, older in zip(chain, chain[1:], strict=False):
        assert newer.prev_chain_sha256 == older.chain_sha256
    assert not [p for p in kind_dir(rc).iterdir() if p.name.endswith(".tmp")]
    codec = get_serializer("safetensors")
    for verified in rc.iter_valid("last", load=True):
        assert verified.data is not None
        state = codec.loads(verified.data)
        assert np.all(state["model"]["w"] == float(verified.manifest.step))


def test_a_file_replaced_after_it_was_verified_is_not_what_gets_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, manager = started(tmp_path)
    manager.save_now(1, model=weights(1.0))
    path = data_file(manager._own, 1)
    real = fsops.read_regular
    evil = blob(666)

    def read_then_swap(dir_fd: int, name: str, *, max_bytes: int) -> bytes:
        data = real(dir_fd, name, max_bytes=max_bytes)
        if name.endswith(".safetensors"):  # the attacker wins the race right after the read
            make_writable(path)
            path.write_bytes(evil)
        return data

    monkeypatch.setattr(fsops, "read_regular", read_then_swap)
    restored = manager.restore()
    monkeypatch.undo()
    assert restored is not None
    assert np.all(restored.state["model"]["w"] == 1.0)  # the bytes that were hashed were loaded
    assert hashlib.sha256(path.read_bytes()).hexdigest() != restored.manifest.sha256
    assert manager.restore() is None  # and from now on the replaced file is refused
    run.finish()


def test_symlink_swaps_between_restores_are_never_followed(tmp_path: Path) -> None:
    run, manager = started(tmp_path)
    manager.save_now(1, model=weights(1.0))
    manager.save_now(2, model=weights(2.0))
    target = data_file(manager._own, 2)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(target.read_bytes())
    os.unlink(target)
    os.symlink(outside, target)
    restored = manager.restore()
    assert restored is not None and restored.manifest.step == 1
    assert [s.reason for s in restored.skipped] == ["not_regular_file"]
    run.finish()


def test_orphan_temp_files_do_not_block_or_confuse_restore_and_are_cleaned(
    tmp_path: Path,
) -> None:
    run, manager = started(tmp_path)
    manager.save_now(1, model=weights(1.0))
    folder = kind_dir(manager._own)
    (folder / ".step_000000002.safetensors.99999.deadbeef.tmp").write_bytes(b"half a checkpoint")
    (folder / ".step_000000002.manifest.json.99999.deadbeef.tmp").write_bytes(b"{")
    restored = manager.restore()
    assert restored is not None and restored.manifest.step == 1 and restored.skipped == []
    manager.save_now(2, model=weights(2.0))
    assert not [p for p in folder.iterdir() if p.name.endswith(".tmp")]
    run.finish()


def test_clock_skew_in_manifests_does_not_change_which_checkpoint_is_newest(
    tmp_path: Path,
) -> None:
    run, manager = started(tmp_path)
    now = datetime.now(timezone.utc)
    for step, skew in ((1, 3650), (2, -3650), (3, 0)):
        manager._own.save(
            kind="last", step=step, data=blob(step), format="safetensors", versions={},
            dataset_hash="merkle:" + "0" * 64, created_at=now + timedelta(days=skew),
        )  # fmt: skip
    restored = manager.restore()
    assert restored is not None and restored.manifest.step == 3
    run.finish()


def test_a_forged_coherent_directory_is_refused_when_authentication_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "s" * 48)
    run, manager = started(tmp_path, require_trust="authenticated")
    manager.save_now(1, model=weights(1.0))
    manager.save_now(2, model=weights(2.0))
    rc = manager._own
    evil = blob(666)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(evil)
    forged = forge_manifest(rc, 2, sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    forge_pointer(rc, forged, 2)
    restored = manager.restore()
    assert restored is not None and restored.manifest.step == 1
    assert restored.trust == "authenticated"
    # delete the genuine step 1 as well: nothing left that is authentic, so nothing is loaded
    for name in ("step_000000001.safetensors", "step_000000001.manifest.json"):
        os.unlink(kind_dir(rc) / name)
    with pytest.raises(CheckpointUntrustedError):
        manager.restore()
    run.finish()
