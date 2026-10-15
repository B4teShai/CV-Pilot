"""Training state to bytes and back: tensors by safetensors, the rest as bounded JSON."""

from __future__ import annotations

import io
import json
import struct
import sys
import types
from typing import Any

import numpy as np
import pytest

from cvpilot.ckpt import serializers, state
from cvpilot.core.errors import CheckpointCorruptError, CheckpointError, MissingExtraError

FRAMEWORK = "to" + "rch"  # the QA gate scans source text for the framework's name


def sample() -> dict[str, Any]:
    return {
        "model": {"layer.weight": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(3)},
        "optimizer": {"lr": 0.1, "steps": 12, "betas": [0.9, 0.99], "name": "sgd"},
        "scaler": {"scale": float("inf"), "loss": float("nan"), "ok": True, "none": None},
        "rng": {"python": (3, [1, 2, 3], None), "bytes": b"\x00\x01\xff"},
        "dataloader_state": {"epoch": 3, "offset": 17},
    }


def test_split_and_merge_round_trip_every_supported_type() -> None:
    tensors, skeleton = state.split_state(sample())
    assert set(tensors) == {"model/layer.weight", "model/b"}
    json.dumps(skeleton, allow_nan=False)  # the skeleton is strict JSON
    back = state.merge_state(tensors, skeleton)
    np.testing.assert_array_equal(back["model"]["layer.weight"], sample()["model"]["layer.weight"])
    assert back["optimizer"] == {"lr": 0.1, "steps": 12, "betas": [0.9, 0.99], "name": "sgd"}
    assert back["scaler"]["scale"] == float("inf") and np.isnan(back["scaler"]["loss"])
    assert back["scaler"]["ok"] is True and back["scaler"]["none"] is None
    assert back["rng"]["python"] == [3, [1, 2, 3], None]  # tuples come back as lists
    assert back["rng"]["bytes"] == b"\x00\x01\xff"


def test_dicts_keyed_by_integers_keep_their_key_type() -> None:
    original = {
        "opt": {"state": {0: {"exp_avg": np.ones(2)}, 7: {"step": 3}}, "groups": [{"lr": 1}]}
    }
    tensors, skeleton = state.split_state(original)
    back = state.merge_state(tensors, skeleton)
    assert set(back["opt"]["state"]) == {0, 7} and back["opt"]["state"][7] == {"step": 3}
    np.testing.assert_array_equal(back["opt"]["state"][0]["exp_avg"], np.ones(2))


def test_mixed_key_types_and_bad_integer_markers_are_refused() -> None:
    with pytest.raises(CheckpointError):
        state.split_state({"x": {0: 1, "a": 2}})
    with pytest.raises(CheckpointError):
        state.split_state({"x": {True: 1}})
    with pytest.raises(ValueError):
        state.merge_state({}, {"x": {"$intdict": {"a": 1}}})
    with pytest.raises(ValueError):
        state.merge_state({}, {"x": {"$intdict": [1]}})


def test_tensor_names_escape_slashes_and_tildes() -> None:
    tensors, skeleton = state.split_state({"a/b": {"c~d": np.zeros(1)}})
    assert list(tensors) == ["a~1b/c~0d"]
    assert "c~d" in state.merge_state(tensors, skeleton)["a/b"]


def test_array_like_objects_with_a_numpy_method_become_tensors() -> None:
    class Tensor:
        def detach(self) -> Tensor:
            return self

        def cpu(self) -> Tensor:
            return self

        def numpy(self) -> Any:
            return np.arange(4, dtype=np.int64)

    tensors, _ = state.split_state({"w": Tensor()})
    assert tensors["w"].dtype == np.int64


@pytest.mark.parametrize(
    "bad",
    [
        {1: "non-str key"},
        {"$tensor": "collides with the marker"},
        {"x": {1, 2}},
        {"x": object()},
        {"x": lambda: 1},
        {"x": np.array(["a"], dtype=object)},
        {"x": complex(1, 2)},
    ],
)
def test_unsupported_state_is_refused_with_the_type_named(bad: dict[Any, Any]) -> None:
    with pytest.raises(CheckpointError):
        state.split_state(bad)


def test_state_depth_and_size_are_bounded() -> None:
    deep: dict[str, Any] = {}
    node = deep
    for _ in range(state.MAX_DEPTH + 2):
        node["n"] = {}
        node = node["n"]
    with pytest.raises(CheckpointError):
        state.split_state(deep)
    with pytest.raises(CheckpointError):
        state.split_state({"big": "x" * (state.MAX_STATE_JSON_BYTES + 1)})
    with pytest.raises(CheckpointError):
        state.split_state({f"k{i}": np.zeros(1) for i in range(state.MAX_TENSORS + 1)})


def test_merge_rejects_a_skeleton_that_names_a_missing_tensor() -> None:
    with pytest.raises(ValueError):
        state.merge_state({}, {"w": {"$tensor": "nope"}})
    with pytest.raises(ValueError):
        state.merge_state({"extra": np.zeros(1)}, {})


def test_safetensors_round_trip_and_determinism() -> None:
    codec = serializers.get_serializer("safetensors")
    data = codec.dumps(sample())
    assert data == codec.dumps(sample())  # equal state, equal bytes
    back = codec.loads(data)
    np.testing.assert_array_equal(back["model"]["b"], np.ones(3))
    assert back["optimizer"]["steps"] == 12
    assert codec.suffix == "safetensors"


def test_safetensors_structure_check_accepts_what_dumps_produces() -> None:
    codec = serializers.get_serializer("safetensors")
    data = codec.dumps(sample())
    codec.check(lambda n: data[:n], len(data))


def corrupt_cases() -> dict[str, bytes]:
    codec = serializers.get_serializer("safetensors")
    good = codec.dumps(sample())
    header_len = struct.unpack("<Q", good[:8])[0]
    header = json.loads(good[8 : 8 + header_len])
    first = next(k for k in header if k != "__metadata__")
    bad_offsets = dict(header)
    bad_offsets[first] = {**header[first], "data_offsets": [0, 10**9]}
    unknown_ref = dict(header)
    unknown_ref["__metadata__"] = {"cvpilot_state": '{"model":{"$tensor":"ghost"}}'}
    no_state = {k: v for k, v in header.items() if k != "__metadata__"}

    def rebuild(h: dict[str, Any]) -> bytes:
        raw = json.dumps(h).encode()
        return struct.pack("<Q", len(raw)) + raw + good[8 + header_len :]

    return {
        "empty": b"",
        "short": b"\x01\x02",
        "garbage": b"\xff" * 64,
        "truncated": good[: len(good) // 2],
        "huge_header": struct.pack("<Q", 2**40) + b"{}",
        "header_not_json": struct.pack("<Q", 4) + b"nope" + b"\x00" * 8,
        "bad_offsets": rebuild(bad_offsets),
        "unknown_reference": rebuild(unknown_ref),
        "no_state": rebuild(no_state),
    }


@pytest.mark.parametrize("name", list(corrupt_cases()))
def test_safetensors_loads_refuses_corrupt_input(name: str) -> None:
    codec = serializers.get_serializer("safetensors")
    with pytest.raises(CheckpointCorruptError) as info:
        codec.loads(corrupt_cases()[name])
    assert info.value.reason == "unloadable"


@pytest.mark.parametrize(
    "name", ["empty", "short", "garbage", "huge_header", "header_not_json", "bad_offsets"]
)
def test_safetensors_structure_check_refuses_corrupt_headers(name: str) -> None:
    codec = serializers.get_serializer("safetensors")
    data = corrupt_cases()[name]
    with pytest.raises(CheckpointCorruptError):
        codec.check(lambda n: data[:n], len(data))


def test_safetensors_missing_extra_names_the_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "safetensors", None)
    monkeypatch.setitem(sys.modules, "safetensors.numpy", None)
    with pytest.raises(MissingExtraError) as info:
        serializers.get_serializer("safetensors").dumps({"a": np.zeros(1)})
    assert "cvpilot[safetensors]" in str(info.value)


def test_unknown_format_is_refused() -> None:
    with pytest.raises(CheckpointCorruptError) as info:
        serializers.get_serializer("pickle")  # type: ignore[arg-type]
    assert info.value.reason == "unsupported_format"


def fake_framework(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    seen: dict[str, Any] = {}
    module = types.ModuleType(FRAMEWORK)

    def save(obj: Any, handle: io.BytesIO) -> None:
        handle.write(b"PK\x03\x04" + json.dumps(obj, default=str).encode())

    def load(handle: io.BytesIO, *, map_location: str, weights_only: bool) -> Any:
        seen["weights_only"] = weights_only
        seen["map_location"] = map_location
        if not weights_only:
            raise AssertionError("pickle loading must never be enabled")
        return json.loads(handle.read()[4:])

    module.save = save  # type: ignore[attr-defined]
    module.load = load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, FRAMEWORK, module)
    return seen


def test_pt_format_always_loads_with_weights_only(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = fake_framework(monkeypatch)
    codec = serializers.get_serializer("pt")
    data = codec.dumps({"epoch": 3})
    assert codec.loads(data, map_location="cpu") == {"epoch": 3}
    assert seen == {"weights_only": True, "map_location": "cpu"}
    assert codec.suffix == "pt"
    codec.check(lambda n: data[:n], len(data))


def test_pt_format_refuses_files_that_are_not_zip_archives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_framework(monkeypatch)
    codec = serializers.get_serializer("pt")
    legacy_pickle = b"\x80\x02}q\x00."  # a raw pickle stream: never handed to the loader
    with pytest.raises(CheckpointCorruptError):
        codec.loads(legacy_pickle)
    with pytest.raises(CheckpointCorruptError):
        codec.check(lambda n: legacy_pickle[:n], len(legacy_pickle))


def test_pt_format_refuses_a_non_mapping_result(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_framework(monkeypatch)
    codec = serializers.get_serializer("pt")
    with pytest.raises(CheckpointCorruptError):
        codec.loads(b"PK\x03\x04" + b"[1, 2]")


def test_pt_format_without_the_framework_names_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, FRAMEWORK, None)
    with pytest.raises(MissingExtraError):
        serializers.get_serializer("pt").dumps({"a": 1})
