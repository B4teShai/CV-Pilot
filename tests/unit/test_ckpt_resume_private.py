"""A pt checkpoint is read once on the way to a resume point, and a private copy closes the gap
between 'verified' and 'opened by the framework'."""

from __future__ import annotations

import hashlib
import io
import os
import pickle
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import fsops, resume, serializers
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointCorruptError
from cvpilot.core.verified import framework_file

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")


def archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("archive/data.pkl", pickle.dumps({"epoch": 1}, protocol=2))
    return buffer.getvalue()


def point(tmp: Path, monkeypatch: pytest.MonkeyPatch) -> resume.ResumePoint:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(run, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False)
    (tmp / "last.pt").write_bytes(archive())
    mgr.adopt(tmp / "last.pt", 3)
    mgr.close()
    run.finish("preempted")
    found = resume.find_resume_point("p", checkpoint_dir=tmp / "ckpt", store_dir=tmp / "store")
    assert found is not None
    return found


def test_a_pt_file_is_read_once_on_the_way_to_a_resume_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    (tmp_path / "last.pt").write_bytes(archive())
    mgr.adopt(tmp_path / "last.pt", 3)
    mgr.close()
    run.finish("preempted")
    reads: list[str] = []
    for name in ("read_regular", "hash_regular"):
        real = getattr(fsops, name)

        def spy(dir_fd: int, file: str, *a: Any, _real: Any = real, **kw: Any) -> Any:
            reads.append(file)
            return _real(dir_fd, file, *a, **kw)

        monkeypatch.setattr(fsops, name, spy)
    from cvpilot.ckpt.directory import RunCheckpoints

    ranges: list[int] = []
    real_range = RunCheckpoints.read_range

    def counting(self: RunCheckpoints, manifest: Any, start: int, length: int) -> bytes:
        ranges.append(length)
        return real_range(self, manifest, start, length)

    monkeypatch.setattr(RunCheckpoints, "read_range", counting)
    found = resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
    )
    assert found is not None and found.manifest.format == "pt"
    assert [r for r in reads if r.endswith(".pt")] == ["step_000000003.pt"]  # hashed once
    assert ranges == []  # and not read a second time for the archive scan


def test_a_private_copy_is_verified_bytes_in_a_directory_only_the_owner_can_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = point(tmp_path, monkeypatch)
    copy = resume.private_copy(found)
    assert copy.read_bytes() == found.path.read_bytes()
    assert stat.S_IMODE(copy.stat().st_mode) == 0o600
    assert stat.S_IMODE(copy.parent.stat().st_mode) == 0o700
    assert copy.parent != found.path.parent and copy.name == found.path.name
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == found.manifest.sha256
    assert framework_file(copy) == copy  # the hooks accept it: it is a verified pt file
    assert framework_file(found) == found.path


def test_a_file_that_changes_while_it_is_copied_is_refused_and_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = point(tmp_path, monkeypatch)
    real = os.read
    flipped = {"done": False}

    def tamper(fd: int, n: int) -> bytes:
        data = real(fd, n)
        if data and not flipped["done"]:
            flipped["done"] = True
            return bytes([data[0] ^ 0xFF]) + data[1:]
        return data

    monkeypatch.setattr(os, "read", tamper)
    tmp_root = Path(tempfile.gettempdir())
    before = set(tmp_root.glob("cvpilot-resume-*"))
    with pytest.raises(CheckpointCorruptError):
        resume.private_copy(found)
    monkeypatch.undo()
    after = set(tmp_root.glob("cvpilot-resume-*"))
    assert after - before == set()


def test_a_point_that_was_never_verified_cannot_be_copied(tmp_path: Path) -> None:
    from cvpilot.core.schema import CheckpointManifest  # noqa: F401

    class Stranger:
        path = tmp_path / "x.pt"
        manifest = None

    with pytest.raises((ValueError, CheckpointCorruptError, AttributeError)):
        resume.private_copy(Stranger())  # type: ignore[arg-type]


def test_the_stat_of_the_original_is_not_what_the_copy_vouches_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    found = point(tmp_path, monkeypatch)
    copy = resume.private_copy(found)
    original = found.path
    original.chmod(0o600)
    original.write_bytes(b"swapped after verification")  # the framework opens the copy instead
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == found.manifest.sha256
