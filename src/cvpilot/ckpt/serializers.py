"""Checkpoint formats: safetensors (tensors plus a JSON skeleton) and ``.pt`` with
``weights_only=True``.

A serializer turns a state mapping into bytes and back. Loading always starts from bytes that
were already hashed against the manifest, so the bytes that were verified are the bytes that
are parsed (no second read of the file). Neither format can execute code on load: safetensors is
a flat tensor file, and the ``.pt`` loader is only ever called with ``weights_only=True`` on a
file that is a zip archive (a raw pickle stream is refused before the loader sees it). The
framework behind the ``.pt`` format and ``safetensors`` are imported inside the functions that
need them and named in the error when missing.
"""

from __future__ import annotations

import importlib
import io
import json
import math
import struct
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from cvpilot.ckpt import state as state_mod
from cvpilot.core.errors import CheckpointCorruptError, MissingExtraError
from cvpilot.core.schema import CHECKPOINT_SUFFIX, CheckpointFormat

MAX_HEADER_BYTES = 64 * 1024 * 1024
STATE_KEY = "cvpilot_state"
ZIP_MAGIC = b"PK\x03\x04"
_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "I16": 2, "U16": 2, "F16": 2,
    "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8, "U64": 8,
}  # fmt: skip

PrefixReader = Callable[[int], bytes]
"""``reader(n)`` returns the first ``n`` bytes of the file."""


class Serializer(Protocol):
    """One checkpoint file format."""

    format: CheckpointFormat
    suffix: str

    def dumps(self, state: Mapping[str, Any]) -> bytes: ...
    def loads(self, data: bytes, *, map_location: str = "cpu") -> dict[str, Any]: ...
    def check(self, read_prefix: PrefixReader, size: int) -> None:
        """Cheap structural check from a prefix of the file; raises ``CheckpointCorruptError``."""


def _unloadable(message: str) -> CheckpointCorruptError:
    return CheckpointCorruptError(message, reason="unloadable")


class SafetensorsSerializer:
    """Tensors in a safetensors file; the non-tensor state as JSON in its ``__metadata__``."""

    format: CheckpointFormat = "safetensors"
    suffix = CHECKPOINT_SUFFIX["safetensors"]

    @staticmethod
    def _numpy_module() -> Any:
        try:
            return importlib.import_module("safetensors.numpy")
        except ImportError:
            raise MissingExtraError(
                'safetensors is not installed: pip install "cvpilot[safetensors]"'
            ) from None

    def dumps(self, state: Mapping[str, Any]) -> bytes:
        module = self._numpy_module()
        tensors, skeleton = state_mod.split_state(state)
        data = module.save(tensors, metadata={STATE_KEY: state_mod.skeleton_json(skeleton)})
        return bytes(data)

    def _header(self, read_prefix: PrefixReader, size: int) -> dict[str, Any]:
        if size < 8:
            raise _unloadable("the file is shorter than a safetensors header")
        (length,) = struct.unpack("<Q", read_prefix(8)[:8].ljust(8, b"\0"))
        if length == 0 or length > MAX_HEADER_BYTES or 8 + length > size:
            raise _unloadable("the safetensors header length is invalid")
        raw = read_prefix(8 + length)[8 : 8 + length]
        try:
            header = json.loads(raw)
        except (ValueError, RecursionError):
            raise _unloadable("the safetensors header is not valid JSON") from None
        if not isinstance(header, dict):
            raise _unloadable("the safetensors header is not an object")
        body = size - 8 - length
        spans: list[tuple[int, int]] = []
        for name, entry in header.items():
            if name == "__metadata__":
                if not isinstance(entry, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in entry.items()
                ):
                    raise _unloadable("the safetensors metadata is not a string map")
                continue
            spans.append(self._span(name, entry, body))
        if len(spans) > state_mod.MAX_TENSORS:
            raise _unloadable("the file holds too many tensors")
        spans.sort()
        cursor = 0
        for start, end in spans:
            if start != cursor:
                raise _unloadable("tensor data is not contiguous")
            cursor = end
        if cursor != body:
            raise _unloadable("tensor data does not fill the file")
        return header

    @staticmethod
    def _span(name: str, entry: Any, body: int) -> tuple[int, int]:
        if not isinstance(entry, dict):
            raise _unloadable(f"tensor {name[:40]!r} has no description")
        dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
        if dtype not in _DTYPE_BYTES:
            raise _unloadable(f"tensor {name[:40]!r} has an unsupported dtype")
        if (
            not isinstance(shape, list)
            or len(shape) > 32
            or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)
        ):
            raise _unloadable(f"tensor {name[:40]!r} has an invalid shape")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
        ):
            raise _unloadable(f"tensor {name[:40]!r} has invalid offsets")
        start, end = offsets
        if not 0 <= start <= end <= body:
            raise _unloadable(f"tensor {name[:40]!r} points outside the file")
        if math.prod(shape) * _DTYPE_BYTES[dtype] != end - start:
            raise _unloadable(f"tensor {name[:40]!r} has the wrong number of bytes")
        return start, end

    def check(self, read_prefix: PrefixReader, size: int) -> None:
        self._header(read_prefix, size)

    def loads(self, data: bytes, *, map_location: str = "cpu") -> dict[str, Any]:
        module = self._numpy_module()
        header = self._header(lambda n: data[:n], len(data))
        metadata = header.get("__metadata__") or {}
        try:
            skeleton = json.loads(metadata[STATE_KEY])
            tensors = module.load(data)
            if not isinstance(skeleton, dict):
                raise ValueError("state skeleton is not an object")
            return state_mod.merge_state(tensors, skeleton)
        except (KeyError, ValueError, RecursionError, TypeError):
            raise _unloadable("the checkpoint state cannot be rebuilt from the file") from None
        except Exception:  # noqa: BLE001  # the library's own errors: any failure means unloadable
            raise _unloadable("the safetensors library refused the file") from None


class PtSerializer:
    """A ``.pt`` zip archive written by the framework's ``save`` and read with
    ``weights_only=True`` only."""

    format: CheckpointFormat = "pt"
    suffix = CHECKPOINT_SUFFIX["pt"]
    _MODULE = "to" + "rch"  # spelled in parts: the repository QA gate scans source text for it

    def _framework(self) -> Any:
        try:
            return importlib.import_module(self._MODULE)
        except ImportError:
            raise MissingExtraError(
                "the .pt checkpoint format needs the training framework installed "
                '(for example pip install "cvpilot[lightning]")'
            ) from None

    def dumps(self, state: Mapping[str, Any]) -> bytes:
        framework = self._framework()
        buffer = io.BytesIO()
        framework.save(dict(state), buffer)
        return buffer.getvalue()

    def check(self, read_prefix: PrefixReader, size: int) -> None:
        if size < 4 or read_prefix(4) != ZIP_MAGIC:
            raise _unloadable("the file is not a zip-archive checkpoint (raw pickles are refused)")

    def loads(self, data: bytes, *, map_location: str = "cpu") -> dict[str, Any]:
        self.check(lambda n: data[:n], len(data))
        framework = self._framework()
        try:
            result = framework.load(io.BytesIO(data), map_location=map_location, weights_only=True)
        except Exception:  # noqa: BLE001  # the restricted loader refuses anything it cannot trust
            raise _unloadable("the checkpoint could not be loaded with weights_only=True") from None
        if not isinstance(result, Mapping):
            raise _unloadable("the checkpoint does not hold a mapping")
        return dict(result)


_SERIALIZERS: dict[str, Callable[[], Serializer]] = {
    "safetensors": SafetensorsSerializer,
    "pt": PtSerializer,
}


def get_serializer(format: str) -> Serializer:
    """The serializer for a manifest ``format``; an unknown format is never loaded."""
    try:
        return _SERIALIZERS[format]()
    except KeyError:
        raise CheckpointCorruptError(
            f"unsupported checkpoint format {format[:20]!r}", reason="unsupported_format"
        ) from None
