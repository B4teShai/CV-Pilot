"""Directory-descriptor file operations: no symlinks, regular files only, bounded, exclusive."""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from cvpilot.ckpt import fsops
from cvpilot.core.errors import (
    CheckpointBusyError,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointPathError,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")


@pytest.fixture
def root_fd(tmp_path: Path):  # type: ignore[no-untyped-def]
    fd = fsops.open_root(tmp_path / "ckpt", create=True)
    yield fd
    os.close(fd)


def test_root_must_not_be_the_filesystem_root(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        fsops.open_root(Path("/"), create=False)


def test_missing_root_without_create_fails(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        fsops.open_root(tmp_path / "nope", create=False)


def test_root_that_is_a_file_fails(tmp_path: Path) -> None:
    (tmp_path / "f").write_text("x")
    with pytest.raises(CheckpointError):
        fsops.open_root(tmp_path / "f", create=True)


def test_exclusive_write_sets_mode_and_returns_digest(root_fd: int) -> None:
    sha, size = fsops.write_exclusive(root_fd, "a.bin", b"hello", final_mode=0o444)
    assert size == 5
    assert sha == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    st = os.stat("a.bin", dir_fd=root_fd)
    assert stat.S_IMODE(st.st_mode) == 0o444


def test_exclusive_write_refuses_an_existing_name(root_fd: int) -> None:
    fsops.write_exclusive(root_fd, "a.bin", b"1")
    with pytest.raises(FileExistsError):
        fsops.write_exclusive(root_fd, "a.bin", b"2")


def test_write_never_follows_a_planted_symlink(root_fd: int, tmp_path: Path) -> None:
    victim = tmp_path / "victim"
    victim.write_text("keep")
    os.symlink(victim, tmp_path / "ckpt" / "a.bin")
    with pytest.raises(FileExistsError):
        fsops.write_exclusive(root_fd, "a.bin", b"overwrite")
    assert victim.read_text() == "keep"


def test_write_accepts_chunks_and_bounds_them(root_fd: int) -> None:
    sha, size = fsops.write_exclusive(root_fd, "c.bin", iter([b"ab", b"cd"]), max_bytes=4)
    assert size == 4 and sha
    with pytest.raises(CheckpointError):
        fsops.write_exclusive(root_fd, "d.bin", iter([b"ab", b"cde"]), max_bytes=4)
    assert not [n for n in os.listdir(root_fd) if n.endswith(".tmp") or n == "d.bin"]


def test_write_cleans_its_temp_file_when_the_disk_is_full(
    root_fd: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full(fd: int, data: bytes) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(fsops.os, "write", full)
    with pytest.raises(CheckpointError) as info:
        fsops.write_exclusive(root_fd, "e.bin", b"data")
    assert "space" in str(info.value).lower()
    monkeypatch.undo()
    assert os.listdir(root_fd) == []


def test_read_regular_returns_bytes_once(root_fd: int) -> None:
    fsops.write_exclusive(root_fd, "r.bin", b"payload")
    assert fsops.read_regular(root_fd, "r.bin", max_bytes=100) == b"payload"


def test_read_refuses_symlinks(root_fd: int, tmp_path: Path) -> None:
    (tmp_path / "secret").write_bytes(b"s")
    os.symlink(tmp_path / "secret", tmp_path / "ckpt" / "l.bin")
    with pytest.raises(CheckpointCorruptError) as info:
        fsops.read_regular(root_fd, "l.bin", max_bytes=100)
    assert info.value.reason == "not_regular_file"


def test_read_refuses_fifos_without_blocking(root_fd: int, tmp_path: Path) -> None:
    os.mkfifo(tmp_path / "ckpt" / "p.bin")
    with pytest.raises(CheckpointCorruptError) as info:
        fsops.read_regular(root_fd, "p.bin", max_bytes=100)
    assert info.value.reason == "not_regular_file"


def test_read_refuses_directories_and_oversize_and_missing(root_fd: int) -> None:
    os.mkdir("d", dir_fd=root_fd)
    with pytest.raises(CheckpointCorruptError):
        fsops.read_regular(root_fd, "d", max_bytes=100)
    fsops.write_exclusive(root_fd, "big.bin", b"x" * 50)
    with pytest.raises(CheckpointCorruptError) as info:
        fsops.read_regular(root_fd, "big.bin", max_bytes=10)
    assert info.value.reason == "too_large"
    with pytest.raises(CheckpointCorruptError) as info:
        fsops.read_regular(root_fd, "missing.bin", max_bytes=10)
    assert info.value.reason == "missing"


def test_hash_regular_streams_and_matches(root_fd: int) -> None:
    data = os.urandom(3 * (1 << 20) + 17)
    sha, size = fsops.write_exclusive(root_fd, "h.bin", data)
    assert fsops.hash_regular(root_fd, "h.bin", max_bytes=len(data)) == (sha, size)
    with pytest.raises(CheckpointCorruptError):
        fsops.hash_regular(root_fd, "h.bin", max_bytes=10)


def test_subdir_refuses_symlink_swap(root_fd: int, tmp_path: Path) -> None:
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    os.symlink(elsewhere, tmp_path / "ckpt" / "last")
    with pytest.raises(CheckpointCorruptError) as info:
        fsops.open_subdir(root_fd, "last", create=True)
    assert info.value.reason == "not_a_directory"
    assert list(elsewhere.iterdir()) == []


def test_subdir_is_created_private(root_fd: int) -> None:
    fd = fsops.open_subdir(root_fd, "run1", create=True)
    try:
        assert stat.S_IMODE(os.fstat(fd).st_mode) == 0o700
    finally:
        os.close(fd)


def test_listdir_is_bounded(root_fd: int) -> None:
    for i in range(5):
        fsops.write_exclusive(root_fd, f"f{i}", b"x")
    assert len(fsops.listdir(root_fd, limit=10)) == 5
    with pytest.raises(CheckpointError):
        fsops.listdir(root_fd, limit=3)


def test_replace_and_unlink(root_fd: int) -> None:
    fsops.write_exclusive(root_fd, "a", b"1")
    fsops.replace(root_fd, "a", "b")
    assert fsops.listdir(root_fd, limit=10) == ["b"]
    fsops.unlink_quiet(root_fd, "b")
    fsops.unlink_quiet(root_fd, "b")  # idempotent
    assert fsops.listdir(root_fd, limit=10) == []


def test_lock_excludes_a_second_holder(root_fd: int, tmp_path: Path) -> None:
    other = fsops.open_root(tmp_path / "ckpt", create=False)
    try:
        with fsops.lock(root_fd, timeout=1.0):
            with pytest.raises(CheckpointBusyError):
                with fsops.lock(other, timeout=0.2):
                    pass
        with fsops.lock(other, timeout=0.2):
            pass
    finally:
        os.close(other)


def test_temp_names_are_unpredictable_and_recognised() -> None:
    a, b = (
        fsops.temp_name("step_000000001.safetensors"),
        fsops.temp_name("step_000000001.safetensors"),
    )
    assert a != b and fsops.is_temp_name(a)
    assert not fsops.is_temp_name("step_000000001.safetensors")


def test_the_lock_file_survives_a_creation_race(
    root_fd: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Some filesystems answer ENOENT to a plain O_CREAT while another process creates the file."""
    real = os.open
    calls = {"n": 0}

    def flaky(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        if path == fsops.LOCK_NAME and calls["n"] < 3:
            calls["n"] += 1
            raise FileNotFoundError(errno.ENOENT, "raced")
        return real(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(fsops, "require_posix", lambda: None)  # the stand-in is not in the set
    monkeypatch.setattr(fsops.os, "open", flaky)
    with fsops.lock(root_fd, timeout=1.0):
        pass
    assert calls["n"] == 3


def test_the_lock_file_gives_up_after_bounded_retries(
    root_fd: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always(path: Any, flags: int, mode: int = 0o777, *, dir_fd: int | None = None) -> int:
        raise FileNotFoundError(errno.ENOENT, "gone")

    monkeypatch.setattr(fsops, "require_posix", lambda: None)
    monkeypatch.setattr(fsops.os, "open", always)
    monkeypatch.setattr(fsops.time, "sleep", lambda s: None)
    with pytest.raises(CheckpointError):
        with fsops.lock(root_fd, timeout=1.0):
            pass


@pytest.mark.parametrize("mode", [0o770, 0o707, 0o755, 0o777, 0o750])
def test_a_root_that_others_can_enter_or_write_is_refused(tmp_path: Path, mode: int) -> None:
    root = tmp_path / "shared"
    root.mkdir()
    root.chmod(mode)
    with pytest.raises(CheckpointPathError, match="chmod 700"):
        fsops.open_root(root, create=False)


def test_a_root_the_caller_owns_with_no_group_or_world_access_is_opened(tmp_path: Path) -> None:
    root = tmp_path / "mine"
    root.mkdir()
    root.chmod(0o700)
    os.close(fsops.open_root(root, create=False))
    os.close(fsops.open_root(tmp_path / "created", create=True))
    assert stat.S_IMODE((tmp_path / "created").stat().st_mode) == 0o700


def test_a_root_owned_by_someone_else_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "theirs"
    root.mkdir(mode=0o700)
    mine = os.getuid()
    monkeypatch.setattr(fsops.os, "getuid", lambda: mine + 1)
    with pytest.raises(CheckpointPathError, match="owned by"):
        fsops.open_root(root, create=False)


def test_a_root_is_checked_through_its_descriptor_not_its_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # the mode is read with fstat on the opened descriptor, so a swap after the open changes nothing
    root = tmp_path / "swap"
    root.mkdir(mode=0o700)
    calls: list[int] = []
    real = os.fstat
    monkeypatch.setattr(fsops.os, "fstat", lambda fd: calls.append(fd) or real(fd))
    os.close(fsops.open_root(root, create=False))
    assert calls
