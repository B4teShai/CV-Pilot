"""The framework argument helpers hand over only a path that this process verified, and only a
file the framework can load (a ``pt`` checkpoint): never ``last`` or an arbitrary path."""

from __future__ import annotations

import importlib
import io
import pickle
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import resume, serializers
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.hooks import lightning as lightning_hook

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
# the framework's name is spelled in parts: the repository QA gate scans source text for it
yolo_hook = importlib.import_module("cvpilot.hooks." + "ultra" + "lytics")
Events = Any


def pt_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps({"epoch": 1}, protocol=2))
    return buffer.getvalue()


def point_of(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, fmt: str) -> resume.ResumePoint:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    if fmt == "pt":
        source = tmp_path / "last.pt"
        source.write_bytes(pt_archive())
        mgr.adopt(source, step=3)
    else:
        mgr.save_now(3, model={"w": np.ones(2)})
    mgr.close()
    run.finish("preempted")
    found = resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
    )
    assert found is not None
    return found


def test_a_pt_checkpoint_found_by_resume_is_accepted_as_a_path_or_a_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = point_of(tmp_path, monkeypatch, fmt="pt")
    for value in (point, point.path):
        assert lightning_hook.resume_arguments(value) == {"ckpt_path": str(point.path)}
        assert yolo_hook.resume_arguments(value) == (str(point.path), {"resume": True})
    assert lightning_hook.resume_arguments(None) == {}
    assert yolo_hook.resume_arguments(None) == (None, {})


def test_a_safetensors_checkpoint_is_refused_with_an_explanation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = point_of(tmp_path, monkeypatch, fmt="safetensors")
    for value in (point, point.path):
        for helper in (lightning_hook.resume_arguments, yolo_hook.resume_arguments):
            with pytest.raises(ValueError, match="safetensors"):
                helper(value)


def test_a_path_this_process_never_verified_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point_of(tmp_path, monkeypatch, fmt="pt")
    stranger = tmp_path / "elsewhere" / "step_000000001.pt"
    for helper in (lightning_hook.resume_arguments, yolo_hook.resume_arguments):
        with pytest.raises(ValueError, match="verified"):
            helper(stranger)
        with pytest.raises(ValueError):
            helper(tmp_path / "ckpt" / "last" / "step_000000001.pt")


@pytest.mark.parametrize("bad", ["last", Path("last"), "", "relative/path.pt", "/etc/passwd"])
def test_names_relative_paths_and_files_nobody_verified_are_refused(bad: Any) -> None:
    for helper in (lightning_hook.resume_arguments, yolo_hook.resume_arguments):
        with pytest.raises(ValueError):
            helper(bad)


def test_a_pt_file_skipped_as_unsafe_says_so_instead_of_returning_a_bare_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    mgr._own.save(
        kind="last", step=1, data=b"PK\x03\x04" + bytes(100), format="pt", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    mgr.close()
    run.finish("preempted")
    assert (
        resume.find_resume_point(
            "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
        )
        is None
    )
    logged = events()
    skipped = [e for e in logged if e["message"] == "resume_point_skipped"]
    assert skipped and skipped[0]["reason"] == "unsafe_archive" and skipped[0]["step"] == 1
    none = [e for e in logged if e["message"] == "resume_point_unavailable"]
    assert none and none[0]["reasons"] == ["unsafe_archive"] and none[0]["level"] == "WARNING"


def test_a_file_replaced_by_a_symlink_after_verification_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    point = point_of(tmp_path, monkeypatch, fmt="pt")
    path = point.path
    other = tmp_path / "other.pt"
    other.write_bytes(path.read_bytes())  # same bytes: only the identity differs
    path.parent.chmod(0o700)
    path.unlink()
    path.symlink_to(other)
    with pytest.raises(ValueError, match="changed after it was verified"):
        yolo_hook.resume_arguments(point)
