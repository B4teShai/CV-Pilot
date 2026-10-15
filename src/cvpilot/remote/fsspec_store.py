"""Remote checkpoint storage through ``fsspec`` (the optional ``[remote]`` extra).

fsspec and the backend package (s3fs, gcsfs, adlfs) are imported inside ``open_remote``. TLS is
requested explicitly where the backend has a switch, every call has a bounded timeout and the
backend's own retry loop is turned down to one attempt (the caller owns retries, so the total
time is bounded). No credential is passed or read here: the backends take theirs from their own
environment variables or instance roles. Keys are validated against a strict pattern so a key
cannot leave the remote root, and backend exceptions are replaced by ``OSError`` naming only the
exception type, because their text can carry URLs, tokens or account names.
"""

from __future__ import annotations

import importlib
import math
import re
from collections.abc import Iterable
from typing import Any

from cvpilot.core.errors import MissingExtraError
from cvpilot.core.uri import RemoteLocation

MAX_TIMEOUT_S = 300.0
_SEGMENT = r"[A-Za-z0-9][A-Za-z0-9._=-]{0,127}"
_KEY = re.compile(rf"{_SEGMENT}(?:/{_SEGMENT}){{0,7}}")
_BACKEND = {"s3": "s3fs", "gs": "gcsfs", "az": "adlfs", "memory": "fsspec"}


def fsspec_options(scheme: str, timeout: float) -> dict[str, Any]:
    """Filesystem options for ``scheme``: TLS on, bounded timeouts, no credentials."""
    if scheme == "s3":
        return {
            "use_ssl": True,
            "config_kwargs": {
                "connect_timeout": timeout,
                "read_timeout": timeout,
                "retries": {"max_attempts": 1},
            },
        }
    if scheme == "gs":
        return {"timeout": timeout}
    if scheme == "az":
        return {"connection_timeout": timeout, "read_timeout": timeout}
    if scheme == "memory":
        return {}
    raise ValueError(f"unsupported remote scheme {scheme[:20]!r}")


class FsspecStore:
    """A ``RemoteStore`` over one fsspec filesystem below a root path."""

    def __init__(self, fs: Any, root: str) -> None:
        self._fs = fs
        self._root = root

    def _path(self, key: str) -> str:
        if not isinstance(key, str) or not _KEY.fullmatch(key):
            raise ValueError("invalid remote key")
        return f"{self._root}/{key}"

    @staticmethod
    def _fail(exc: Exception) -> OSError:
        return OSError(f"remote operation failed: {type(exc).__name__}")

    def size(self, key: str) -> int | None:
        path = self._path(key)
        try:
            return int(self._fs.info(path)["size"])
        except FileNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001  # backend text may carry secrets: type only
            raise self._fail(exc) from None

    def read_range(self, key: str, start: int, length: int) -> bytes:
        path = self._path(key)
        try:
            return bytes(self._fs.cat_file(path, start=start, end=start + length))
        except Exception as exc:  # noqa: BLE001
            raise self._fail(exc) from None

    def read_bounded(self, key: str, max_bytes: int) -> bytes:
        size = self.size(key)
        if size is None:
            raise FileNotFoundError(key)
        if size > max_bytes:
            raise OSError("remote object is larger than the limit")
        return self.read_range(key, 0, size)

    def write(self, key: str, chunks: Iterable[bytes], *, append: bool = False) -> None:
        path = self._path(key)
        try:
            with self._fs.open(path, "ab" if append else "wb") as handle:
                for chunk in chunks:
                    handle.write(chunk)
        except Exception as exc:  # noqa: BLE001
            raise self._fail(exc) from None

    def move(self, source: str, target: str) -> None:
        src, dst = self._path(source), self._path(target)
        try:
            self._fs.mv(src, dst)
        except Exception as exc:  # noqa: BLE001
            raise self._fail(exc) from None

    def delete(self, key: str) -> None:
        path = self._path(key)
        try:
            self._fs.rm(path)
        except FileNotFoundError:
            return
        except Exception as exc:  # noqa: BLE001
            raise self._fail(exc) from None

    def list(self, prefix: str) -> list[str]:
        path = self._path(prefix)
        try:
            entries = self._fs.ls(path, detail=False)
        except FileNotFoundError:
            return []
        except Exception as exc:  # noqa: BLE001
            raise self._fail(exc) from None
        return sorted({str(e).rstrip("/").rsplit("/", 1)[-1] for e in entries})


def open_remote(location: RemoteLocation, *, timeout: float = 10.0) -> FsspecStore:
    """The store for ``location`` (registered under ``remotes`` for each scheme).

    Raises:
        ValueError: ``timeout`` is not in (0, 300].
        MissingExtraError: fsspec or the scheme's backend is not installed.
    """
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or math.isnan(timeout)
        or not 0 < timeout <= MAX_TIMEOUT_S
    ):
        raise ValueError(f"timeout must be in (0, {MAX_TIMEOUT_S}] seconds")
    options = fsspec_options(location.scheme, float(timeout))
    try:
        fsspec = importlib.import_module("fsspec")
        fs = fsspec.filesystem("gcs" if location.scheme == "gs" else location.scheme, **options)
    except ImportError:
        raise MissingExtraError(
            f"remote checkpoints need fsspec and {_BACKEND[location.scheme]}: "
            'pip install "cvpilot[remote]"'
        ) from None
    return FsspecStore(fs, location.key())
