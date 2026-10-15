"""Parser differentials: what the strict pre-check accepts is what the loader will see.

Every case here is a file that a lenient parser (the library, a fake that models one) could read
differently from Python's own: duplicate keys, NaN and huge numbers, overlapping or gapped
offsets, hidden bytes in an archive, local and central headers that disagree.
"""

from __future__ import annotations

import io
import json
import pickle
import random
import struct
import sys
import types
import zipfile
import zlib
from typing import Any

import numpy as np
import pytest

from cvpilot.ckpt import serializers, state
from cvpilot.core.errors import CheckpointCorruptError

FRAMEWORK = "to" + "rch"
safetensors_numpy = pytest.importorskip("safetensors.numpy")


# -- safetensors ---------------------------------------------------------------------------------


def st_file(header: bytes | dict[str, Any], body: bytes) -> bytes:
    raw = header if isinstance(header, bytes) else json.dumps(header).encode()
    return struct.pack("<Q", len(raw)) + raw + body


def entry(dtype: str, shape: list[int], start: int, end: int) -> dict[str, Any]:
    return {"dtype": dtype, "shape": shape, "data_offsets": [start, end]}


def meta() -> dict[str, Any]:
    return {"__metadata__": {serializers.STATE_KEY: "{}"}}


def codec() -> Any:
    return serializers.get_serializer("safetensors")


def refuses(data: bytes) -> None:
    with pytest.raises(CheckpointCorruptError) as info:
        codec().check(lambda n: data[:n], len(data))
    assert info.value.reason == "unloadable"
    with pytest.raises(CheckpointCorruptError) as info:
        codec().loads(data)
    assert info.value.reason == "unloadable"


GOOD_BODY = bytes(8)
HEADERS: dict[str, bytes] = {
    "duplicate_tensor_name": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        b'"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    ),
    "duplicate_metadata": (
        b'{"__metadata__":{"cvpilot_state":"{}"},"__metadata__":{"cvpilot_state":"{\\"a\\":1}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    ),
    "duplicate_field": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","dtype":"U8","shape":[2],"data_offsets":[0,8]}}'
    ),
    "nan_shape": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[NaN],"data_offsets":[0,8]}}'
    ),
    "infinity_offset": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[0,Infinity]}}'
    ),
    "float_shape": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2.0],"data_offsets":[0,8]}}'
    ),
    "exponent_shape": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2e0],"data_offsets":[0,8]}}'
    ),
    "huge_integer": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[99999999999999999999999999],"data_offsets":[0,8]}}'
    ),
    "negative_offset": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[-8,0]}}'
    ),
    "overlap": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        b'"b":{"dtype":"F32","shape":[1],"data_offsets":[4,8]}}'
    ),
    "gap": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    ),
    "same_span_twice": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"a":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},'
        b'"b":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    ),
    "reversed_offsets": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}'
    ),
    "lowercase_dtype": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"f32","shape":[2],"data_offsets":[0,8]}}'
    ),
    "numeric_dtype": (
        b'{"__metadata__":{"cvpilot_state":"{}"},"w":{"dtype":1,"shape":[2],"data_offsets":[0,8]}}'
    ),
    "extra_tensor_field_is_fine_but_extra_type_is_not": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":"0,8"}}'
    ),
    "numeric_metadata_value": (
        b'{"__metadata__":{"cvpilot_state":5},"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    ),
    "empty_tensor_name": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    ),
    "control_in_name": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"a\\u0000b":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    ),
    "not_an_object": b"[1, 2]",
    "leading_garbage": b' x{"__metadata__":{}}',
    "trailing_garbage": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}xx'
    ),
    "invalid_utf8": b'{"__metadata__":{"cvpilot_state":"\xff"}}',
    "rank_over_cap": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"U8","shape":[' + b",".join([b"1"] * 40) + b'],"data_offsets":[0,1]}}'
    ),
    "element_count_over_cap": (
        b'{"__metadata__":{"cvpilot_state":"{}"},'
        b'"w":{"dtype":"U8","shape":[4294967296,4294967296,4294967296],"data_offsets":[0,8]}}'
    ),
}


@pytest.mark.parametrize("name", list(HEADERS))
def test_the_strict_header_parse_refuses_what_a_lenient_parser_might_accept(name: str) -> None:
    refuses(st_file(HEADERS[name], GOOD_BODY))


def test_utf16_and_bom_headers_are_refused() -> None:
    text = json.dumps({**meta(), "w": entry("F32", [2], 0, 8)})
    refuses(st_file(text.encode("utf-16"), GOOD_BODY))
    refuses(st_file(b"\xef\xbb\xbf" + text.encode(), GOOD_BODY))


def test_too_many_tensors_are_refused() -> None:
    header = {**meta(), **{f"t{i}": entry("U8", [0], 0, 0) for i in range(state.MAX_TENSORS + 1)}}
    refuses(st_file(header, b""))


def test_a_tensor_count_dtype_mismatch_between_header_and_bytes_is_refused() -> None:
    refuses(st_file({**meta(), "w": entry("F32", [3], 0, 8)}, GOOD_BODY))  # 3 floats are 12 bytes
    refuses(
        st_file({**meta(), "w": entry("F32", [2], 0, 8)}, GOOD_BODY + b"extra")
    )  # trailing bytes


class Lenient:
    """Models a loader whose parser disagrees with the strict one: it returns other tensors."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def load(self, data: bytes) -> dict[str, Any]:
        return self.result

    def save(self, tensors: Any, metadata: Any = None) -> bytes:  # pragma: no cover
        raise AssertionError


def valid_file() -> bytes:
    return st_file(
        {"__metadata__": {serializers.STATE_KEY: '{"model":{"w":{"$tensor":"model/w"}}}'},
         "model/w": entry("F32", [2], 0, 8)},
        GOOD_BODY,
    )  # fmt: skip


@pytest.mark.parametrize(
    "result",
    [
        {"model/w": np.zeros(2, dtype=np.float32), "extra": np.zeros(1, dtype=np.float32)},
        {},
        {"model/other": np.zeros(2, dtype=np.float32)},
        {"model/w": np.zeros(3, dtype=np.float32)},  # other shape
        {"model/w": np.zeros((1, 2), dtype=np.float32)},
        {"model/w": np.zeros(2, dtype=np.float64)},  # other dtype
        {"model/w": np.zeros(2, dtype=np.uint8)},
    ],
    ids=["extra", "none", "renamed", "shape", "rank", "dtype-wide", "dtype-narrow"],
)
def test_what_the_library_loaded_must_equal_what_the_strict_parse_declared(
    monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]
) -> None:
    data = valid_file()
    assert codec().loads(data)["model"]["w"].shape == (
        2,
    )  # the real library agrees on the good file
    monkeypatch.setattr(
        serializers.SafetensorsSerializer, "_numpy_module", staticmethod(lambda: Lenient(result))
    )
    with pytest.raises(CheckpointCorruptError) as info:
        codec().loads(data)
    assert info.value.reason == "unloadable"


def test_the_real_library_rejects_or_the_strict_check_does_for_each_hostile_header() -> None:
    """For every hostile header the strict check refuses, whatever the library would do."""
    for name, header in HEADERS.items():
        data = st_file(header, GOOD_BODY)
        with pytest.raises(CheckpointCorruptError):
            codec().loads(data)
        try:
            safetensors_numpy.load(data)
            outcome = "library accepted"
        except Exception:  # noqa: BLE001
            outcome = "library refused"
        assert outcome in {"library accepted", "library refused"}, name


@pytest.mark.parametrize("seed", range(40))
def test_valid_states_round_trip_unchanged_and_re_serialise_to_the_same_bytes(seed: int) -> None:
    rng = random.Random(seed)  # noqa: S311
    dtypes = [np.float32, np.float64, np.float16, np.int64, np.int32, np.uint8, np.bool_]
    original: dict[str, Any] = {}
    for i in range(rng.randint(1, 6)):
        shape = tuple(rng.randint(0, 4) for _ in range(rng.randint(0, 3)))
        dtype = rng.choice(dtypes)
        values = np.asarray(np.random.default_rng(seed + i).integers(0, 100, size=shape)).astype(
            dtype
        )
        original[f"layer/{i}.w"] = values
    original["meta"] = {"epoch": rng.randint(0, 1000), "lr": rng.random(), "tags": ["a", "b"]}  # noqa: S311
    original["opt"] = {"state": {rng.randint(0, 5): {"m": np.ones(2)}}, "n": float("inf")}
    data = codec().dumps(original)
    back = codec().loads(data)
    assert set(back) == set(original)
    for key, value in original.items():
        if isinstance(value, np.ndarray):
            assert back[key].dtype == value.dtype and back[key].shape == value.shape
            np.testing.assert_array_equal(back[key], value)
    assert back["meta"] == original["meta"]
    assert codec().dumps(back) == data


# -- pt archives ---------------------------------------------------------------------------------

GOOD_PICKLE = pickle.dumps({"epoch": 3}, protocol=2)
PAD = struct.pack("<HH", 0x4246, 4) + b"\x00" * 4


def raw_zip(
    members: list[dict[str, Any]],
    *,
    comment: bytes = b"",
    prefix: bytes = b"",
    suffix: bytes = b"",
    total_override: int | None = None,
    cd_offset_delta: int = 0,
    cd_gap: bytes = b"",
) -> bytes:
    out = bytearray(prefix)
    central = bytearray()
    for m in members:
        name = m["name"] if isinstance(m["name"], bytes) else m["name"].encode()
        data: bytes = m.get("data", b"")
        method = m.get("method", 0)
        stored = zlib.compress(data)[2:-4] if method == 8 else data
        crc = m.get("crc", zlib.crc32(data))
        flags = m.get("flags", 0)
        csize, usize = len(stored), len(data)
        offset = len(out)
        local_extra = m.get("local_extra", b"")
        out += struct.pack(
            "<4sHHHHHIIIHH", b"PK\x03\x04", 20, m.get("local_flags", flags),
            m.get("local_method", method), 0, 0, crc,
            m.get("local_csize", csize), m.get("local_usize", usize),
            len(m.get("local_name", name)), len(local_extra),
        )  # fmt: skip
        out += m.get("local_name", name) + local_extra + stored + m.get("after_data", b"")
        central_extra = m.get("central_extra", b"")
        central += struct.pack(
            "<4sHHHHHHIIIHHHHHII", b"PK\x01\x02", 20, m.get("vneed", 20), flags, method, 0, 0,
            crc, m.get("central_csize", csize), m.get("central_usize", usize), len(name),
            len(central_extra), len(m.get("central_comment", b"")), 0, 0, 0, offset,
        )  # fmt: skip
        central += name + central_extra + m.get("central_comment", b"")
    cd_offset = len(out) + len(cd_gap)
    out += cd_gap + central
    out += struct.pack(
        "<4sHHHHIIH", b"PK\x05\x06", 0, 0,
        total_override if total_override is not None else len(members),
        total_override if total_override is not None else len(members),
        len(central), cd_offset + cd_offset_delta, len(comment),
    )  # fmt: skip
    return bytes(out) + comment + suffix


def pk() -> dict[str, Any]:
    return {"name": "archive/data.pkl", "data": GOOD_PICKLE}


def good_zip(**kw: Any) -> bytes:
    return raw_zip([pk(), {"name": "archive/data/0", "data": b"\x00" * 16}], **kw)


def pt_refuses(data: bytes) -> None:
    with pytest.raises(CheckpointCorruptError) as info:
        serializers.scan_pt_archive(data)
    assert info.value.reason == "unsafe_archive"


def test_the_raw_builder_produces_archives_the_scan_accepts() -> None:
    serializers.scan_pt_archive(good_zip())
    serializers.scan_pt_archive(raw_zip([{**pk(), "method": 8}]))
    serializers.scan_pt_archive(raw_zip([{**pk(), "local_extra": PAD}]))  # alignment padding
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:  # an archive written by the standard library
        archive.writestr("archive/data.pkl", GOOD_PICKLE)
    serializers.scan_pt_archive(buffer.getvalue())


PT_CASES: dict[str, bytes] = {
    "duplicate_names": raw_zip([pk(), pk()]),
    "duplicate_names_case": raw_zip([pk(), {"name": "ARCHIVE/DATA.PKL", "data": GOOD_PICKLE}]),
    "absolute": raw_zip([pk(), {"name": "/abs", "data": b"x"}]),
    "traversal": raw_zip([pk(), {"name": "a/../b", "data": b"x"}]),
    "leading_traversal": raw_zip([pk(), {"name": "../b", "data": b"x"}]),
    "backslash": raw_zip([pk(), {"name": "a\\b", "data": b"x"}]),
    "nul": raw_zip([pk(), {"name": b"a\x00b", "data": b"x"}]),
    "drive_letter": raw_zip([pk(), {"name": "C:/x", "data": b"x"}]),
    "non_ascii": raw_zip([pk(), {"name": "caf\u00e9".encode(), "data": b"x"}]),
    "method_bzip2": raw_zip([{**pk(), "method": 12, "local_method": 12}]),
    "encrypted": raw_zip([{**pk(), "flags": 1}]),
    "data_descriptor": raw_zip([{**pk(), "flags": 8}]),
    "zip64_sizes": raw_zip([{**pk(), "central_csize": 0xFFFFFFFF, "central_usize": 0xFFFFFFFF}]),
    "zip64_version": raw_zip([{**pk(), "vneed": 45}]),
    "central_extra": raw_zip([{**pk(), "central_extra": PAD}]),
    "local_extra_other": raw_zip([{**pk(), "local_extra": struct.pack("<HH", 1, 8) + bytes(8)}]),
    "local_extra_truncated": raw_zip([{**pk(), "local_extra": b"\x46\x42\x09"}]),
    "central_comment": raw_zip([{**pk(), "central_comment": b"hi"}]),
    "eocd_comment": good_zip(comment=b"hidden"),
    "prefix_bytes": good_zip(prefix=b"MZ-hidden-payload"),
    "suffix_bytes": good_zip(suffix=b"appended"),
    "gap_before_central_directory": good_zip(cd_gap=b"hidden"),
    "wrong_cd_offset": good_zip(cd_offset_delta=1),
    "wrong_entry_count": good_zip(total_override=5),
    "name_mismatch": raw_zip([{**pk(), "local_name": b"archive/data.pkx"}]),
    "size_mismatch": raw_zip([{**pk(), "central_usize": len(GOOD_PICKLE) + 1}]),
    "csize_mismatch": raw_zip([{**pk(), "local_csize": 1}]),
    "method_mismatch": raw_zip([{**pk(), "local_method": 8}]),
    "flags_mismatch": raw_zip([{**pk(), "local_flags": 0x800}]),
    "crc_mismatch": raw_zip([{**pk(), "crc": 12345}]),
    "bytes_between_members": raw_zip(
        [{**pk(), "after_data": b"junk"}, {"name": "b", "data": b"x"}]
    ),
    "no_pickle": raw_zip([{"name": "archive/version", "data": b"3"}]),
    "empty_archive": raw_zip([]),
    "too_many": raw_zip(
        [{"name": f"a/{i}", "data": b""} for i in range(serializers.MAX_ARCHIVE_MEMBERS + 1)]
    ),
    "ratio": raw_zip([pk(), {"name": "a/z", "data": bytes(5_000_000), "method": 8}]),
    "truncated": good_zip()[:-5],
    "garbage": b"PK\x03\x04" + bytes(60),
}


@pytest.mark.parametrize("name", list(PT_CASES))
def test_the_strict_archive_scan_refuses_hidden_or_ambiguous_structure(name: str) -> None:
    pt_refuses(PT_CASES[name])


def test_a_lenient_loader_that_reads_other_tensors_is_caught_after_the_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def numel(self) -> int:
            return 10_000_000

        def element_size(self) -> int:
            return 4

    module = types.ModuleType(FRAMEWORK)
    module.load = lambda handle, **kw: {"w": Tensor()}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, FRAMEWORK, module)
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    with pytest.raises(CheckpointCorruptError) as info:
        serializers.get_serializer("pt").loads(good_zip())
    assert info.value.reason == "unloadable"
    module.load = lambda handle, **kw: {"w": {1: 2}}  # type: ignore[attr-defined]
    with pytest.raises(CheckpointCorruptError):
        serializers.get_serializer("pt").loads(good_zip())  # non-string keys are not a state


def test_a_loaded_state_within_the_archives_bounds_is_returned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tensor:
        def numel(self) -> int:
            return 4

        def element_size(self) -> int:
            return 4

    module = types.ModuleType(FRAMEWORK)
    module.load = lambda handle, **kw: {"w": Tensor(), "n": {"x": 1}}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, FRAMEWORK, module)
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    assert set(serializers.get_serializer("pt").loads(good_zip())) == {"w", "n"}
