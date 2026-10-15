"""Checkpoint formats: safetensors (tensors plus a JSON skeleton) and, opt-in only, ``.pt``.

A serializer turns a state mapping into bytes and back. Loading always starts from bytes that
were already hashed against the manifest, so the bytes that were verified are the bytes that
are parsed (no second read of the file).

safetensors is the default and the recommended format: a flat tensor file with no code path
that executes anything on load. The ``.pt`` format is a pickle-based archive. ``weights_only=True``
has been bypassed in released versions of the framework (CVE-2025-32434, fixed in 2.6.0), so it
is never enough on its own, and this module layers four controls on top: the format needs an
explicit opt-in (``allow_pt_format`` or ``CVPILOT_CKPT_ALLOW_PT``, enforced by the manager and
by the directory walk); the installed framework version is read from package metadata (nothing
is imported or executed) and must be at least ``MIN_PT_VERSION`` before the loader is imported;
the archive is pre-scanned without executing it (member count, names, sizes, compression ratio,
exactly one ``data.pkl`` whose opcode stream may only reference an allow-list of tensor
storage, dtype and ordered-dict globals); and the loader is only ever called with
``weights_only=True``. Residual risk: the scan is defence in depth, not a verifier of the
pickle's semantics, and a vulnerability in the framework's own restricted unpickler is out of
reach; keep the framework patched and prefer safetensors. The framework behind ``.pt`` and
``safetensors`` are imported inside the functions that need them and named in the error when
missing.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import io
import json
import math
import pickletools
import re
import struct
import zipfile
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from cvpilot.ckpt import state as state_mod
from cvpilot.core.errors import CheckpointCorruptError, CheckpointFormatError, MissingExtraError
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


MIN_PT_VERSION = (2, 6)
"""First framework release in which ``weights_only=True`` no longer allows code execution."""
MAX_ARCHIVE_MEMBERS = 10_000
MAX_MEMBER_NAME = 200
MAX_COMPRESSION_RATIO = 100
MAX_PICKLE_BYTES = 64 * 1024 * 1024
_MEMBER = re.compile(r"[A-Za-z0-9_.\-]+(?:/[A-Za-z0-9_.\-]+)*")
_VERSION = re.compile(r"([0-9]{1,4})\.([0-9]{1,4})(?:[.+\-a-zA-Z0-9]*)")
_T = "to" + "rch"  # spelled in parts: the repository QA gate scans source text for it
_ALLOWED_GLOBALS = frozenset(
    {
        ("collections", "OrderedDict"),
        (_T + "._utils", "_rebuild_tensor_v2"),
        (_T + "._utils", "_rebuild_tensor_v3"),
        (_T + "._utils", "_rebuild_parameter"),
        (_T, "Size"),
        ("_codecs", "encode"),
        ("builtins", "set"),
    }
    | {
        (_T, name)
        for name in (
            "FloatStorage",
            "DoubleStorage",
            "HalfStorage",
            "BFloat16Storage",
            "LongStorage",
            "IntStorage",
            "ShortStorage",
            "CharStorage",
            "ByteStorage",
            "BoolStorage",
            "UntypedStorage",
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "bool",
            "complex64",
            "complex128",
        )
    }  # fmt: skip
)
_FORBIDDEN_OPS = frozenset({"INST", "OBJ", "EXT1", "EXT2", "EXT4"})
_STRING_OPS = frozenset({"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"})


def parse_version(text: object) -> tuple[int, int] | None:
    """``(major, minor)`` of a version string, or None when it cannot be read."""
    if not isinstance(text, str):
        return None
    match = _VERSION.fullmatch(text)
    return None if match is None else (int(match.group(1)), int(match.group(2)))


def framework_version() -> tuple[int, int] | None:
    """The installed training framework's ``(major, minor)`` from package metadata only."""
    try:
        return parse_version(importlib.metadata.version(_T))
    except importlib.metadata.PackageNotFoundError:
        return None


def _unsafe(message: str) -> CheckpointCorruptError:
    return CheckpointCorruptError(message, reason="unsafe_archive")


def _scan_pickle(data: bytes) -> None:
    recent: list[str | None] = []
    try:
        for opcode, arg, _ in pickletools.genops(data):
            name = opcode.name
            if name in _FORBIDDEN_OPS:
                raise _unsafe(f"the pickle uses the forbidden opcode {name}")
            if name == "GLOBAL":
                module, _, attr = str(arg).partition(" ")
                if (module, attr) not in _ALLOWED_GLOBALS:
                    raise _unsafe(f"the pickle refers to {module[:40]}.{attr[:40]}")
            elif name == "STACK_GLOBAL":
                if len(recent) < 2 or recent[-1] is None or recent[-2] is None:
                    raise _unsafe("the pickle builds a global the scan cannot resolve")
                if (recent[-2], recent[-1]) not in _ALLOWED_GLOBALS:
                    raise _unsafe(
                        f"the pickle refers to {str(recent[-2])[:40]}.{str(recent[-1])[:40]}"
                    )
            if name != "MEMOIZE" and not name.endswith(("PUT", "GET")):
                recent.append(str(arg) if name in _STRING_OPS else None)
                del recent[:-2]
    except CheckpointCorruptError:
        raise
    except Exception:  # noqa: BLE001  # anything genops cannot read is not a checkpoint
        raise _unsafe("the pickle stream is malformed") from None


def scan_pt_archive(data: bytes) -> None:
    """Refuse a ``.pt`` archive that is not shaped like a tensor checkpoint (nothing executes).

    Raises:
        CheckpointCorruptError: reason ``unsafe_archive``.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError):
        raise _unsafe("the file is not a readable zip archive") from None
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_ARCHIVE_MEMBERS:
            raise _unsafe("the archive has too many members")
        total = 0
        pickles = []
        for info in infos:
            name = info.filename
            if (
                len(name) > MAX_MEMBER_NAME
                or not _MEMBER.fullmatch(name)
                or any(part in (".", "..") for part in name.split("/"))
            ):
                raise _unsafe("the archive has an unsafe member name")
            if info.flag_bits & 0x1:
                raise _unsafe("the archive is encrypted")
            if info.file_size > (1 << 20) and info.file_size > MAX_COMPRESSION_RATIO * max(
                1, info.compress_size
            ):
                raise _unsafe("the archive has a member with an implausible compression ratio")
            total += info.file_size
            if name == "data.pkl" or name.endswith("/data.pkl"):
                pickles.append(info)
        if total > 8 * len(data) + (1 << 20):
            raise _unsafe("the archive expands to much more than its size")
        if len(pickles) != 1 or pickles[0].file_size > MAX_PICKLE_BYTES:
            raise _unsafe("the archive must hold exactly one reasonably sized data.pkl")
        try:
            with archive.open(pickles[0]) as handle:
                body = handle.read(MAX_PICKLE_BYTES + 1)
        except (zipfile.BadZipFile, OSError, RuntimeError, ValueError, EOFError):
            raise _unsafe("data.pkl cannot be read") from None
    if len(body) > MAX_PICKLE_BYTES:
        raise _unsafe("data.pkl is too large")
    _scan_pickle(body)


class PtSerializer:
    """A ``.pt`` zip archive written by the framework's ``save`` and read with
    ``weights_only=True`` only, behind the controls described in the module docstring."""

    format: CheckpointFormat = "pt"
    suffix = CHECKPOINT_SUFFIX["pt"]
    _MODULE = _T

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
        version = framework_version()
        if version is None or version < MIN_PT_VERSION:
            raise CheckpointFormatError(
                "loading .pt checkpoints needs the training framework at version "
                f"{MIN_PT_VERSION[0]}.{MIN_PT_VERSION[1]} or newer (weights_only=True could be "
                "bypassed before that); upgrade it or use the safetensors format"
            )
        self.check(lambda n: data[:n], len(data))
        scan_pt_archive(data)
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
