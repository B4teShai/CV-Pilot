"""A fake remote object store with failure injection, for the upload and download tests."""

from __future__ import annotations

from collections.abc import Iterable


class FakeRemote:
    """In-memory ``RemoteStore``; ``fail_writes`` makes the next N writes raise OSError,
    ``corrupt`` flips the last byte of every finished write, ``down`` fails everything."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_writes = 0
        self.fail_reads = 0
        self.corrupt = False
        self.down = False
        self.writes: list[tuple[str, int, bool]] = []
        self.moves: list[tuple[str, str]] = []
        self.list_calls: list[str] = []
        self.reads: list[str] = []

    def _check(self) -> None:
        if self.down:
            raise OSError("remote down")

    def size(self, key: str) -> int | None:
        self._check()
        return len(self.objects[key]) if key in self.objects else None

    def read_range(self, key: str, start: int, length: int) -> bytes:
        self._check()
        self.reads.append(key)
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise OSError("transient read failure")
        return self.objects[key][start : start + length]

    def read_bounded(self, key: str, max_bytes: int) -> bytes:
        self._check()
        data = self.objects[key]
        if len(data) > max_bytes:
            raise OSError("too large")
        return data

    def write(self, key: str, chunks: Iterable[bytes], *, append: bool = False) -> None:
        self._check()
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise OSError("transient write failure")
        data = b"".join(chunks)
        self.writes.append((key, len(data), append))
        previous = self.objects.get(key, b"") if append else b""
        stored = previous + data
        if self.corrupt and stored:
            stored = stored[:-1] + bytes([stored[-1] ^ 0xFF])
        self.objects[key] = stored

    def move(self, source: str, target: str) -> None:
        self._check()
        self.moves.append((source, target))
        self.objects[target] = self.objects.pop(source)

    def delete(self, key: str) -> None:
        self._check()
        self.objects.pop(key, None)

    def list(self, prefix: str, limit: int = 10_000) -> list[str]:
        self._check()
        self.list_calls.append(prefix)
        head = prefix.rstrip("/") + "/" if prefix else ""
        names = sorted({k[len(head) :].split("/")[0] for k in self.objects if k.startswith(head)})
        if len(names) > limit:
            raise OSError("too many entries")
        return names
