"""The pt pickle scan must never accept a stream whose STACK_GLOBAL resolves to something it did
not allow, however the names reach the stack (pushed, memoized, fetched again)."""

from __future__ import annotations

import random
import struct

import pytest

from cvpilot.ckpt import serializers
from cvpilot.core.errors import CheckpointCorruptError

ALLOWED = serializers._ALLOWED_GLOBALS


def s(text: str) -> bytes:  # SHORT_BINUNICODE
    return b"\x8c" + bytes([len(text)]) + text.encode()


MEMOIZE, STACK_GLOBAL, REDUCE, STOP, EMPTY_TUPLE = b"\x94", b"\x93", b"R", b".", b")"


def binget(i: int) -> bytes:
    return b"h" + bytes([i])


def binput(i: int) -> bytes:
    return b"q" + bytes([i])


def stream(*ops: bytes) -> bytes:
    return b"\x80\x04" + b"".join(ops) + STOP


def refuses(data: bytes) -> bool:
    try:
        serializers._scan_pickle(data)
    except CheckpointCorruptError:
        return True
    return False


def test_the_reported_bypass_is_refused() -> None:
    """os, memoize, system, memoize, push collections.OrderedDict, then fetch the first two."""
    data = stream(
        s("os"), MEMOIZE, s("system"), MEMOIZE, s("collections"), s("OrderedDict"),
        binget(0), binget(1), STACK_GLOBAL, EMPTY_TUPLE, REDUCE,
    )  # fmt: skip
    assert refuses(data)


def test_a_plain_allowed_stack_global_still_passes() -> None:
    # a fixed pick: iterating a frozenset of strings changes order with PYTHONHASHSEED, and the
    # codec call is the one global with its own rule (see the codec tests)
    module, attr = sorted(ALLOWED - {("_codecs", "encode")})[0]
    assert not refuses(stream(s(module), MEMOIZE, s(attr), MEMOIZE, STACK_GLOBAL, MEMOIZE))


@pytest.mark.parametrize(
    "ops",
    [
        (s("os"), s("system"), binget(0), STACK_GLOBAL),
        (binget(0), binget(1), STACK_GLOBAL),
        (s("os"), binput(0), s("system"), binput(1), binget(1), binget(0), STACK_GLOBAL),
        (s("os"), MEMOIZE, MEMOIZE, STACK_GLOBAL),
    ],
)
def test_a_stack_global_not_built_from_two_plain_strings_is_refused(ops: tuple[bytes, ...]) -> None:
    assert refuses(stream(*ops))


def reference_globals(ops: list[tuple[str, int | str | None]]) -> list[tuple[str, str]] | None:
    """Emulate the stack and memo for the generated ops: the globals each STACK_GLOBAL resolves
    to, or None when the stream is not executable (underflow, a missing memo entry)."""
    stack: list[object] = []
    memo: dict[int, object] = {}
    memo_counter = 0
    found: list[tuple[str, str]] = []
    for kind, arg in ops:
        if kind == "str":
            stack.append(arg)
        elif kind == "memoize":
            if not stack:
                return None
            memo[memo_counter] = stack[-1]
            memo_counter += 1
        elif kind == "binput":
            if not stack:
                return None
            memo[int(arg or 0)] = stack[-1]
        elif kind == "binget":
            if int(arg or 0) not in memo:
                return None
            stack.append(memo[int(arg or 0)])
        elif kind == "stack_global":
            if len(stack) < 2:
                return None
            attr, module = stack.pop(), stack.pop()
            if not isinstance(module, str) or not isinstance(attr, str):
                return None
            found.append((module, attr))
            stack.append(("global", module, attr))
    return found


def encode(ops: list[tuple[str, int | str | None]]) -> bytes:
    out = []
    for kind, arg in ops:
        if kind == "str":
            out.append(s(str(arg)))
        elif kind == "memoize":
            out.append(MEMOIZE)
        elif kind == "binput":
            out.append(binput(int(arg or 0)))
        elif kind == "binget":
            out.append(binget(int(arg or 0)))
        else:
            out.append(STACK_GLOBAL)
    return stream(*out)


def test_random_streams_the_scan_accepts_never_resolve_to_a_disallowed_global() -> None:
    rng = random.Random(7)  # noqa: S311  # a reproducible fuzz seed
    names = ["os", "system", "collections", "OrderedDict", "builtins", "eval", "x"]
    names += [m for m, _ in list(ALLOWED)[:3]] + [a for _, a in list(ALLOWED)[:3]]
    accepted = refused = 0
    for _ in range(4000):
        ops: list[tuple[str, int | str | None]] = []
        for _ in range(rng.randrange(2, 12)):
            kind = rng.choice(["str", "str", "memoize", "binput", "binget", "stack_global"])
            arg: int | str | None = None
            if kind == "str":
                arg = rng.choice(names)
            elif kind in ("binput", "binget"):
                arg = rng.randrange(0, 4)
            ops.append((kind, arg))
        resolved = reference_globals(ops)
        if resolved is None:
            continue
        if refuses(encode(ops)):
            refused += 1
            continue
        accepted += 1
        assert all(g in ALLOWED for g in resolved), (ops, resolved)
    assert accepted > 50 and refused > 50  # both outcomes were exercised
    assert struct.calcsize("<I") == 4


def test_a_byte_string_pickled_by_protocol_2_still_passes() -> None:
    import pickle

    from zip_support import framework_archive

    data = pickle.dumps({"rng": b"\x00\xff" * 100}, protocol=2)
    assert b"_codecs" in data
    serializers.scan_pt_archive(framework_archive({"archive/data.pkl": data}))


def test_a_codec_call_that_could_multiply_memory_is_refused() -> None:
    """``_codecs.encode`` is reachable by the framework's loader with any arguments, so a nested
    hex or base64 encode could grow memory before the loader finishes; only latin1 is allowed."""
    import codecs
    import pickle

    from zip_support import framework_archive

    class Encode:
        def __init__(self, arg: object, codec: str) -> None:
            self.arg, self.codec = arg, codec

        def __reduce__(self) -> tuple[object, ...]:
            return (codecs.encode, (self.arg, self.codec))

    nested = "x" * 1000
    obj: object = nested
    for _ in range(6):
        obj = Encode(obj, "hex")
    for payload in (
        obj,
        Encode("abc", "hex"),
        Encode("abc", "base64"),
        Encode(Encode("a", "latin1"), "latin1"),
    ):
        data = pickle.dumps({"a": payload}, protocol=2)
        with pytest.raises(CheckpointCorruptError, match="codec") as info:
            serializers.scan_pt_archive(framework_archive({"archive/data.pkl": data}))
        assert info.value.reason == "unsafe_archive"
