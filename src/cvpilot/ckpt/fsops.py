"""File operations for checkpoint directories, all relative to directory descriptors.

Every name is opened relative to a directory descriptor that was itself opened without
following symlinks, so a symlink planted (or swapped in) between two steps can neither redirect
a write nor make a read leave the directory. Files are opened with ``O_NOFOLLOW`` and
``O_NONBLOCK`` (a FIFO cannot hang a reader), must be regular, are bounded in size, and are read
once from one descriptor. New files are created with ``O_EXCL`` and mode 0600, fsynced, and made
read-only (0444) before they get their final name. POSIX only: Windows has no directory
descriptors, so ``open_root`` refuses there rather than fall back to path-based operations.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import secrets
import stat
import time
from collections.abc import Iterable, Iterator
from pathlib import Path

from cvpilot.core.errors import (
    CheckpointBusyError,
    CheckpointCorruptError,
    CheckpointError,
    CheckpointPathError,
)

try:
    import fcntl
except ImportError:  # pragma: no cover  # Windows
    fcntl = None  # type: ignore[assignment]

CHUNK = 1 << 20
LOCK_NAME = ".lock"
_DIR_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0) | os.O_CLOEXEC
)


def require_posix() -> None:
    """Raise unless directory-descriptor operations and ``O_NOFOLLOW`` exist."""
    ok = (
        os.name == "posix"
        and fcntl is not None
        and os.open in os.supports_dir_fd
        and os.unlink in os.supports_dir_fd
        and os.stat in os.supports_dir_fd
        and os.mkdir in os.supports_dir_fd
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "O_DIRECTORY")
    )
    if not ok:
        raise CheckpointError("checkpoints need a POSIX platform (directory descriptors)")


def open_root(path: Path, *, create: bool) -> int:
    """A descriptor for the resolved checkpoint root; the caller closes it.

    The root itself may be reached through a symlink (a mounted bucket is usually one): it is
    resolved once, here. Everything below it is opened without following symlinks.
    """
    require_posix()
    try:
        resolved = Path(path).expanduser().resolve()
    except (OSError, RuntimeError) as exc:
        raise CheckpointPathError(f"cannot resolve the checkpoint directory: {exc}") from exc
    if resolved == Path(resolved.anchor):
        raise CheckpointPathError("the checkpoint directory must not be the filesystem root")
    try:
        if create:
            resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
        return os.open(resolved, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    except OSError as exc:
        raise CheckpointPathError(
            f"cannot open the checkpoint directory {resolved}: {exc}"
        ) from exc


def open_subdir(parent_fd: int, name: str, *, create: bool, mode: int = 0o700) -> int:
    """Open ``name`` below ``parent_fd`` as a directory, never through a symlink."""
    try:
        try:
            return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            with contextlib.suppress(FileExistsError):
                os.mkdir(name, mode, dir_fd=parent_fd)
            return os.open(name, _DIR_FLAGS, dir_fd=parent_fd)
    except FileNotFoundError:
        raise CheckpointCorruptError(f"directory {name} is missing", reason="missing") from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise CheckpointCorruptError(
                f"{name} is not a plain directory (a symlink or a file)", reason="not_a_directory"
            ) from None
        raise CheckpointError(f"cannot open directory {name}: {exc}") from exc


def listdir(dir_fd: int, *, limit: int) -> list[str]:
    """Names in the directory, sorted; more than ``limit`` entries is an error (bounded work)."""
    try:
        names = os.listdir(dir_fd)
    except OSError as exc:
        raise CheckpointError(f"cannot list the checkpoint directory: {exc}") from exc
    if len(names) > limit:
        raise CheckpointError(f"too many entries in the checkpoint directory (limit {limit})")
    return sorted(names)


def _open_regular(dir_fd: int, name: str, max_bytes: int) -> tuple[int, int]:
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except FileNotFoundError:
        raise CheckpointCorruptError(f"{name} is missing", reason="missing") from None
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENXIO, errno.EISDIR):
            raise CheckpointCorruptError(
                f"{name} is not a regular file", reason="not_regular_file"
            ) from None
        raise CheckpointError(f"cannot open {name}: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise CheckpointCorruptError(f"{name} is not a regular file", reason="not_regular_file")
        if info.st_size > max_bytes:
            raise CheckpointCorruptError(
                f"{name} is {info.st_size} bytes, over the limit of {max_bytes}", reason="too_large"
            )
        return fd, info.st_size
    except BaseException:
        os.close(fd)
        raise


def stat_regular(dir_fd: int, name: str, *, max_bytes: int) -> int:
    """The size of a regular file (not a symlink, FIFO or directory), without reading it."""
    fd, size = _open_regular(dir_fd, name, max_bytes)
    os.close(fd)
    return size


def read_regular(dir_fd: int, name: str, *, max_bytes: int) -> bytes:
    """The bytes of a regular file, read once from one descriptor and bounded by ``max_bytes``."""
    fd, _ = _open_regular(dir_fd, name, max_bytes)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(fd, min(CHUNK, max_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise CheckpointCorruptError(
                    f"{name} grew past {max_bytes} bytes", reason="too_large"
                )
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError as exc:
        raise CheckpointError(f"cannot read {name}: {exc}") from exc
    finally:
        os.close(fd)


def hash_regular(dir_fd: int, name: str, *, max_bytes: int) -> tuple[str, int]:
    """``(sha256, size)`` of a regular file, streamed from one descriptor."""
    fd, _ = _open_regular(dir_fd, name, max_bytes)
    digest = hashlib.sha256()
    total = 0
    try:
        while True:
            chunk = os.read(fd, CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise CheckpointCorruptError(
                    f"{name} grew past {max_bytes} bytes", reason="too_large"
                )
            digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(f"cannot read {name}: {exc}") from exc
    finally:
        os.close(fd)
    return digest.hexdigest(), total


def read_prefix(dir_fd: int, name: str, length: int, *, max_bytes: int) -> bytes:
    """The first ``length`` bytes of a regular file (used for header checks)."""
    fd, _ = _open_regular(dir_fd, name, max_bytes)
    try:
        return os.read(fd, length)
    except OSError as exc:
        raise CheckpointError(f"cannot read {name}: {exc}") from exc
    finally:
        os.close(fd)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def write_exclusive(
    dir_fd: int,
    name: str,
    data: bytes | Iterable[bytes],
    *,
    max_bytes: int | None = None,
    final_mode: int = 0o444,
) -> tuple[str, int]:
    """Create ``name`` (``O_EXCL``, mode 0600), write, fsync, chmod and return ``(sha256, size)``.

    Raises:
        FileExistsError: ``name`` exists, even as a dangling symlink; nothing is touched.
        CheckpointError: the write failed (disk full, over ``max_bytes``); the partial file is
            removed.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    fd = os.open(name, flags, 0o600, dir_fd=dir_fd)  # FileExistsError propagates untouched
    digest = hashlib.sha256()
    total = 0
    try:
        chunks = [data] if isinstance(data, (bytes, bytearray, memoryview)) else data
        for chunk in chunks:
            total += len(chunk)
            if max_bytes is not None and total > max_bytes:
                raise CheckpointError(f"checkpoint data exceeds the limit of {max_bytes} bytes")
            digest.update(chunk)
            _write_all(fd, bytes(chunk))
        os.fsync(fd)
        os.fchmod(fd, final_mode)
    except BaseException as exc:
        os.close(fd)
        with contextlib.suppress(OSError):
            os.unlink(name, dir_fd=dir_fd)
        if isinstance(exc, OSError) and not isinstance(exc, CheckpointError):
            reason = "no space left on device" if exc.errno == errno.ENOSPC else str(exc)
            raise CheckpointError(f"cannot write {name}: {reason}") from exc
        raise
    os.close(fd)
    return digest.hexdigest(), total


def replace(dir_fd: int, source: str, target: str) -> None:
    """Atomically rename ``source`` over ``target`` inside one directory."""
    try:
        os.replace(source, target, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except OSError as exc:
        raise CheckpointError(f"cannot rename {source} to {target}: {exc}") from exc


def unlink_quiet(dir_fd: int, name: str) -> bool:
    """Remove ``name`` if present (never following a symlink); True if something was removed."""
    try:
        os.unlink(name, dir_fd=dir_fd)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise CheckpointError(f"cannot remove {name}: {exc}") from exc
    return True


def exists(dir_fd: int, name: str) -> bool:
    """True when ``name`` exists as any kind of entry (a symlink counts, it is not followed)."""
    try:
        os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def fsync_dir(dir_fd: int) -> None:
    """Flush directory entries; filesystems that cannot do it are tolerated."""
    try:
        os.fsync(dir_fd)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP, errno.EBADF):
            raise CheckpointError(f"cannot fsync the checkpoint directory: {exc}") from exc


def _open_lock_file(dir_fd: int) -> int:
    """Open ``.lock``, creating it exclusively when absent. Several processes may race to create
    it (some filesystems answer ENOENT to a plain O_CREAT in that window), so the open and the
    exclusive create alternate a few times instead of trusting one call."""
    base = os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC
    last: OSError | None = None
    for attempt in range(20):
        try:
            return os.open(LOCK_NAME, base, dir_fd=dir_fd)
        except FileNotFoundError as exc:
            last = exc
        except OSError as exc:
            raise CheckpointError(f"cannot open the checkpoint lock: {exc}") from exc
        try:
            return os.open(LOCK_NAME, base | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=dir_fd)
        except (FileExistsError, FileNotFoundError) as exc:
            last = exc
        except OSError as exc:
            raise CheckpointError(f"cannot open the checkpoint lock: {exc}") from exc
        time.sleep(0.005 * (attempt + 1))
    raise CheckpointError(f"cannot open the checkpoint lock: {last}")


@contextlib.contextmanager
def lock(dir_fd: int, *, timeout: float) -> Iterator[None]:
    """Hold an exclusive ``flock`` on ``<dir>/.lock`` for the duration of a save or prune.

    The lock dies with the process, so a killed saver never leaves it held.

    Raises:
        CheckpointBusyError: it could not be taken within ``timeout`` seconds.
    """
    require_posix()
    fd = _open_lock_file(dir_fd)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as exc:
                if exc.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    raise CheckpointError(f"cannot lock the checkpoint directory: {exc}") from exc
                if time.monotonic() >= deadline:
                    raise CheckpointBusyError(
                        "another process is saving into this checkpoint directory"
                    ) from None
                time.sleep(0.02)
        try:
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def temp_name(final: str) -> str:
    """An unpredictable temporary name for ``final`` (``.final.<pid>.<token>.tmp``)."""
    return f".{final}.{os.getpid()}.{secrets.token_hex(6)}.tmp"


def is_temp_name(name: str) -> bool:
    return name.startswith(".") and name.endswith(".tmp")
