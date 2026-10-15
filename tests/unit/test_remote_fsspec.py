"""The fsspec adapter: lazy import, TLS and bounded timeouts, no credentials, errors without
backend text."""

from __future__ import annotations

import sys
import uuid
from typing import Any

import pytest

from cvpilot import registry
from cvpilot.core.errors import MissingExtraError
from cvpilot.core.uri import parse_remote_uri
from cvpilot.remote import fsspec_store

fsspec = pytest.importorskip("fsspec")


def memory_store() -> Any:
    return fsspec_store.open_remote(parse_remote_uri(f"memory://t{uuid.uuid4().hex}/runs"))


def test_a_memory_remote_round_trips_every_operation() -> None:
    store = memory_store()
    assert store.size("a/x.bin") is None
    store.write("a/x.part", [b"hello ", b"wor"])
    store.write("a/x.part", [b"ld"], append=True)
    assert store.size("a/x.part") == 11
    assert store.read_range("a/x.part", 6, 5) == b"world"
    assert store.read_bounded("a/x.part", 100) == b"hello world"
    store.move("a/x.part", "a/x.bin")
    assert store.size("a/x.part") is None and store.size("a/x.bin") == 11
    assert store.list("a") == ["x.bin"]
    assert store.list("missing") == []
    store.delete("a/x.bin")
    store.delete("a/x.bin")  # idempotent
    assert store.size("a/x.bin") is None


def test_reads_are_bounded() -> None:
    store = memory_store()
    store.write("k", [b"x" * 100])
    with pytest.raises(OSError):
        store.read_bounded("k", 10)


@pytest.mark.parametrize("key", ["../x", "a/../b", "/abs", "a//b", "a\\b", "", "a/\x00", "a b"])
def test_keys_cannot_leave_the_remote_root(key: str) -> None:
    store = memory_store()
    with pytest.raises(ValueError):
        store.size(key)


def test_options_use_tls_bounded_timeouts_and_no_credentials() -> None:
    s3 = fsspec_store.fsspec_options("s3", 7.0)
    assert s3["use_ssl"] is True
    assert (
        s3["config_kwargs"]["connect_timeout"] == 7.0 and s3["config_kwargs"]["read_timeout"] == 7.0
    )
    assert s3["config_kwargs"]["retries"] == {"max_attempts": 1}
    assert fsspec_store.fsspec_options("gs", 7.0)["timeout"] == 7.0
    assert fsspec_store.fsspec_options("az", 7.0)["connection_timeout"] == 7.0
    assert fsspec_store.fsspec_options("memory", 7.0) == {}
    for scheme in ("s3", "gs", "az", "memory"):
        text = repr(fsspec_store.fsspec_options(scheme, 7.0)).lower()
        assert not any(w in text for w in ("secret", "password", "token", "key", "anon"))
    with pytest.raises(ValueError):
        fsspec_store.fsspec_options("ftp", 7.0)


def test_timeouts_are_bounded() -> None:
    loc = parse_remote_uri("memory://t/x")
    for bad in (0, -1, 301, float("nan")):
        with pytest.raises(ValueError):
            fsspec_store.open_remote(loc, timeout=bad)


def test_backend_errors_never_carry_the_backends_text() -> None:
    class Boom:
        def info(self, path: str) -> Any:
            raise RuntimeError("SECRET-TOKEN-abc123 for https://user:pw@host")

    store = memory_store()
    store._fs = Boom()
    with pytest.raises(OSError) as info:
        store.size("a/b")
    assert "SECRET" not in str(info.value) and "RuntimeError" in str(info.value)


def test_a_missing_fsspec_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "fsspec", None)
    with pytest.raises(MissingExtraError) as info:
        fsspec_store.open_remote(parse_remote_uri("memory://t/x"))
    assert "cvpilot[remote]" in str(info.value)


def test_the_schemes_are_registered() -> None:
    assert registry.remotes.names() == ["az", "gs", "memory", "s3"]
    assert registry.remotes.get("memory") is fsspec_store.open_remote
