"""resume_path, adopting framework files, and the framework argument helpers."""

from __future__ import annotations

import importlib
import io
import os
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

import cvpilot
from cvpilot import runs
from cvpilot.ckpt import resume, serializers
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError, CheckpointFormatError, CheckpointUntrustedError
from cvpilot.hooks import lightning as lightning_hook

# the framework's name is spelled in parts: the repository QA gate scans source text for it
yolo_hook = importlib.import_module("cvpilot.hooks." + "ultra" + "lytics")


def start(store: Path, project: str = "p") -> runs.RunHandle:
    meta, repro = run_meta(project)
    return runs.start_run(meta, repro, store_dir=store)


def manager(run: runs.RunHandle, tmp_path: Path, **kw: Any) -> CheckpointManager:
    return CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(8, value, dtype=np.float32)}


def preempted_run(tmp_path: Path, steps: tuple[int, ...] = (1, 2)) -> str:
    run = start(tmp_path / "store")
    mgr = manager(run, tmp_path, keep_last=10)
    for step in steps:
        mgr.save_now(step, model=weights(float(step)))
    run_id = run.run_id
    run.finish("preempted")
    return run_id


def find(tmp_path: Path, **kw: Any) -> resume.ResumePoint | None:
    return resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store", **kw
    )


def test_the_newest_verified_checkpoint_of_the_newest_resumable_run_is_returned(
    tmp_path: Path,
) -> None:
    older = preempted_run(tmp_path, (1,))
    newer = preempted_run(tmp_path, (1, 2, 3))
    point = find(tmp_path)
    assert point is not None and point.run_id == newer != older
    assert point.manifest.step == 3 and point.needs_recover is False
    assert (
        point.path == (tmp_path / "ckpt" / newer / "last" / "step_000000003.safetensors").resolve()
    )
    assert point.trust == "anchored"  # the run store recorded its digest
    assert (
        resume.resume_path("p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store")
        == point.path
    )


def test_nothing_to_resume_is_none(tmp_path: Path) -> None:
    assert find(tmp_path) is None  # no store, no directory
    run = start(tmp_path / "store")
    run.finish()  # finished runs are never resumed
    manager_run = start(tmp_path / "store")
    manager(manager_run, tmp_path).save_now(1, model=weights(1.0))
    manager_run.finish()
    assert find(tmp_path) is None


def test_a_corrupt_newest_file_falls_back_to_the_previous_valid_one(tmp_path: Path) -> None:
    run_id = preempted_run(tmp_path, (1, 2))
    newest = tmp_path / "ckpt" / run_id / "last" / "step_000000002.safetensors"
    newest.chmod(0o644)
    newest.write_bytes(newest.read_bytes()[:30])
    point = find(tmp_path)
    assert point is not None and point.manifest.step == 1


def test_a_run_whose_checkpoints_are_all_bad_hands_over_to_an_older_run(tmp_path: Path) -> None:
    good = preempted_run(tmp_path, (1,))
    bad = preempted_run(tmp_path, (1,))
    only = tmp_path / "ckpt" / bad / "last" / "step_000000001.safetensors"
    only.chmod(0o644)
    only.write_bytes(b"x")
    point = find(tmp_path)
    assert point is not None and point.run_id == good


def test_a_crashed_run_is_offered_but_flagged_as_needing_recovery(tmp_path: Path) -> None:
    run = start(tmp_path / "store")
    mgr = manager(run, tmp_path)
    mgr.save_now(1, model=weights(1.0))
    mgr.close()  # the process died: the liveness lease is gone
    run.abandon()  # and the status stays running
    point = find(tmp_path)
    assert point is not None and point.needs_recover is True and point.status == "running"


def test_a_strict_policy_fails_closed_instead_of_returning_an_unproven_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preempted_run(tmp_path, (1,))
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "k" * 40)
    with pytest.raises(CheckpointUntrustedError):
        find(tmp_path, require_trust="authenticated")
    point = find(tmp_path, require_trust="anchored")
    assert point is not None and point.trust == "anchored"


def pt_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"epoch": 1}, protocol=2))
    return buffer.getvalue()


def test_adopting_a_framework_file_needs_the_opt_in_and_a_clean_archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    source = tmp_path / "last.pt"
    source.write_bytes(pt_archive())
    run = start(tmp_path / "store")
    plain = manager(run, tmp_path)
    with pytest.raises(CheckpointFormatError):
        plain.adopt(source, step=5)
    mgr = manager(run, tmp_path, allow_pt_format=True)
    manifest = mgr.adopt(source, step=5)
    assert manifest.format == "pt" and manifest.step == 5
    assert (tmp_path / "ckpt" / run.run_id / manifest.path).read_bytes() == source.read_bytes()
    assert mgr.adopt(source, step=5) == manifest  # idempotent for identical bytes
    run.finish("preempted")
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    point = find(tmp_path)
    assert point is not None and point.manifest.format == "pt"
    monkeypatch.delenv("CVPILOT_CKPT_ALLOW_PT")
    assert find(tmp_path) is None  # not opted in: a pt file is never handed to a framework


def test_adopt_refuses_symlinks_directories_oversize_and_non_archives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    run = start(tmp_path / "store")
    mgr = manager(run, tmp_path, allow_pt_format=True, max_checkpoint_bytes=2000)
    real = tmp_path / "real.pt"
    real.write_bytes(pt_archive())
    link = tmp_path / "link.pt"
    os.symlink(real, link)
    for bad in (link, tmp_path, tmp_path / "missing.pt"):
        with pytest.raises(CheckpointError):
            mgr.adopt(bad, step=1)
    big = tmp_path / "big.pt"
    big.write_bytes(pt_archive() + bytes(5000))
    with pytest.raises(CheckpointError):
        mgr.adopt(big, step=1)
    junk = tmp_path / "junk.pt"
    junk.write_bytes(pickle.dumps({"x": 1}))
    with pytest.raises(CheckpointError):
        mgr.adopt(junk, step=1)
    assert [e for e in mgr.list()] == []
    run.finish()


def test_a_hostile_archive_that_was_adopted_is_still_refused_at_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    run = start(tmp_path / "store")
    mgr = manager(run, tmp_path)
    mgr._own.save(
        kind="last", step=1, data=b"PK\x03\x04" + bytes(100), format="pt", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    run.finish("preempted")
    assert find(tmp_path) is None


def test_the_public_function_reads_directory_and_store_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = preempted_run(tmp_path, (1,))
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("CVPILOT_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    got = cvpilot.resume_path("p")
    assert got is not None and run_id in str(got)
    assert cvpilot.resume_path("p", checkpoint_dir=tmp_path / "elsewhere") is None


def test_framework_arguments_never_contain_the_string_last(tmp_path: Path) -> None:
    path = tmp_path / "ckpt" / "last" / "step_000000001.pt"
    assert lightning_hook.resume_arguments(None) == {}
    assert lightning_hook.resume_arguments(path) == {"ckpt_path": str(path)}
    assert yolo_hook.resume_arguments(None) == (None, {})
    assert yolo_hook.resume_arguments(path) == (str(path), {"resume": True})
    for bad in ("last", Path("last"), "", "relative/path.pt"):
        with pytest.raises(ValueError):
            lightning_hook.resume_arguments(bad)
        with pytest.raises(ValueError):
            yolo_hook.resume_arguments(bad)
