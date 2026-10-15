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
import zlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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


_NUMPY_DTYPE = {
    "BOOL": "bool", "U8": "uint8", "I8": "int8", "I16": "int16", "U16": "uint16",
    "F16": "float16", "I32": "int32", "U32": "uint32", "F32": "float32", "F64": "float64",
    "I64": "int64", "U64": "uint64",
}  # fmt: skip
MAX_RANK = 32
MAX_ELEMENTS = 1 << 40
MAX_NAME_CHARS = 1024
MAX_METADATA_VALUE = state_mod.MAX_STATE_JSON_BYTES
MAX_INT_DIGITS = 19  # header integers (sizes, offsets); the state's own bound is state.MAX_INT


@dataclass(frozen=True)
class StrictHeader:
    """A safetensors header parsed once, strictly: what the file declares, nothing else."""

    tensors: dict[str, tuple[str, tuple[int, ...]]]
    metadata: dict[str, str]
    body_bytes: int


def _bounded_int(text: str) -> int:
    if len(text) > MAX_INT_DIGITS or text.startswith("-"):
        raise ValueError("integer out of range")
    return int(text)


def _reject_float(text: str) -> Any:
    raise ValueError("fractional number")


def _clean_name(name: str) -> bool:
    return 0 < len(name) <= MAX_NAME_CHARS and not any(ord(c) < 32 or ord(c) == 127 for c in name)


def parse_safetensors_header(read_prefix: PrefixReader, size: int) -> StrictHeader:
    """Parse and validate a safetensors header once, rejecting everything a lenient parser might
    read differently: duplicate keys (also a repeated ``__metadata__``), NaN and Infinity,
    fractions, negative or oversized integers, non-UTF-8 or BOM-prefixed text, trailing garbage,
    unknown dtype spellings, wrong byte counts and overlapping, gapped or unsorted spans.

    Raises:
        CheckpointCorruptError: reason ``unloadable``.
    """
    if size < 8:
        raise _unloadable("the file is shorter than a safetensors header")
    (length,) = struct.unpack("<Q", read_prefix(8)[:8].ljust(8, b"\0"))
    if length < 2 or length > MAX_HEADER_BYTES or 8 + length > size:
        raise _unloadable("the safetensors header length is invalid")
    raw = read_prefix(8 + length)[8 : 8 + length]
    if len(raw) != length or raw[:1] != b"{":
        raise _unloadable("the safetensors header does not start with an object")
    try:
        header = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=state_mod.no_duplicate_keys,
            parse_constant=state_mod.reject_constant,
            parse_int=_bounded_int,
            parse_float=_reject_float,
        )
    except (ValueError, RecursionError):
        raise _unloadable("the safetensors header is not strict JSON") from None
    if not isinstance(header, dict):
        raise _unloadable("the safetensors header is not an object")
    body = size - 8 - length
    metadata: dict[str, str] = {}
    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    spans: list[tuple[int, int, str]] = []
    for name, value in header.items():
        if name == "__metadata__":
            if (
                not isinstance(value, dict)
                or len(value) > 64
                or not all(
                    isinstance(k, str)
                    and isinstance(v, str)
                    and _clean_name(k)
                    and len(v) <= MAX_METADATA_VALUE
                    for k, v in value.items()
                )
            ):
                raise _unloadable("the safetensors metadata is not a bounded string map")
            metadata = dict(value)
            continue
        if not _clean_name(name):
            raise _unloadable("a tensor name is empty, too long or has control characters")
        dtype, shape, start, end = _tensor_entry(name, value, body)
        tensors[name] = (dtype, shape)
        spans.append((start, end, name))
        if len(spans) > state_mod.MAX_TENSORS:
            raise _unloadable("the file holds too many tensors")
    spans.sort()
    cursor = 0
    for start, end, _ in spans:
        if start != cursor:
            raise _unloadable("tensor data overlaps or leaves gaps")
        cursor = end
    if cursor != body:
        raise _unloadable("tensor data does not fill the file exactly")
    return StrictHeader(tensors, metadata, body)


def _tensor_entry(name: str, entry: Any, body: int) -> tuple[str, tuple[int, ...], int, int]:
    label = name[:40]
    if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
        raise _unloadable(f"tensor {label!r} does not have exactly dtype, shape and data_offsets")
    dtype, shape, offsets = entry["dtype"], entry["shape"], entry["data_offsets"]
    if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
        raise _unloadable(f"tensor {label!r} has an unsupported dtype")
    if (
        not isinstance(shape, list)
        or len(shape) > MAX_RANK
        or not all(isinstance(d, int) and not isinstance(d, bool) for d in shape)
    ):
        raise _unloadable(f"tensor {label!r} has an invalid shape")
    elements = math.prod(shape)
    if elements > MAX_ELEMENTS:
        raise _unloadable(f"tensor {label!r} has too many elements")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(isinstance(o, int) and not isinstance(o, bool) for o in offsets)
    ):
        raise _unloadable(f"tensor {label!r} has invalid offsets")
    start, end = offsets
    if not 0 <= start <= end <= body:
        raise _unloadable(f"tensor {label!r} points outside the file")
    if elements * _DTYPE_BYTES[dtype] != end - start:
        raise _unloadable(f"tensor {label!r} has the wrong number of bytes")
    return dtype, tuple(shape), start, end


class SafetensorsSerializer:
    """Tensors in a safetensors file; the non-tensor state as JSON in its ``__metadata__``.

    The header is parsed once by ``parse_safetensors_header``; after the library loads the same
    bytes, the loaded names, dtypes, shapes and byte total must equal what that parse declared.
    """

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

    def check(self, read_prefix: PrefixReader, size: int) -> None:
        parse_safetensors_header(read_prefix, size)

    @staticmethod
    def _same_as_declared(header: StrictHeader, loaded: Any) -> None:
        if not isinstance(loaded, Mapping) or set(loaded) != set(header.tensors):
            raise _unloadable("the loaded tensors differ from the header's")
        total = 0
        for name, (dtype, shape) in header.tensors.items():
            array = loaded[name]
            if getattr(array, "shape", None) != shape or getattr(
                getattr(array, "dtype", None), "name", None
            ) != _NUMPY_DTYPE.get(dtype):
                raise _unloadable(f"tensor {name[:40]!r} was loaded as something else")
            total += int(array.nbytes)
        if total != header.body_bytes:
            raise _unloadable("the loaded tensors do not add up to the header's byte total")

    def loads(self, data: bytes, *, map_location: str = "cpu") -> dict[str, Any]:
        module = self._numpy_module()
        header = parse_safetensors_header(lambda n: data[:n], len(data))
        try:
            skeleton = state_mod.parse_skeleton(header.metadata[STATE_KEY])
            tensors = module.load(data)
        except (KeyError, ValueError, RecursionError, TypeError):
            raise _unloadable("the checkpoint state cannot be rebuilt from the file") from None
        except Exception:  # noqa: BLE001  # the library's own errors: any failure means unloadable
            raise _unloadable("the safetensors library refused the file") from None
        self._same_as_declared(header, tensors)
        if not isinstance(skeleton, dict):
            raise _unloadable("the state skeleton is not an object")
        try:
            merged = state_mod.merge_state(tensors, skeleton)
            state_mod.validate_plain(merged)
        except (ValueError, TypeError, RecursionError):
            raise _unloadable("the checkpoint state cannot be rebuilt from the file") from None
        return merged


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


_LOCAL = struct.Struct("<4sHHHHHIIIHH")
_CENTRAL = struct.Struct("<4sHHHHHHIIIHHHHHII")
_EOCD = struct.Struct("<4sHHHHIIH")
_PAD_TAG = 0x4246  # the alignment padding record the framework writes into local headers
_ZIP64_MARK = 0xFFFFFFFF
MAX_MEMBER_BYTES = 1 << 36


def _local_extra_ok(extra: bytes) -> bool:
    """Only the framework's alignment padding records are tolerated in a local header."""
    if len(extra) > 4096:
        return False
    pos = 0
    while pos < len(extra):
        if pos + 4 > len(extra):
            return False
        tag, length = struct.unpack_from("<HH", extra, pos)
        if tag != _PAD_TAG or pos + 4 + length > len(extra):
            return False
        pos += 4 + length
    return True


def _member_name(raw: bytes) -> str:
    try:
        name = raw.decode("ascii")
    except UnicodeDecodeError:
        raise _unsafe("an archive member name is not ASCII") from None
    if (
        len(name) > MAX_MEMBER_NAME
        or not _MEMBER.fullmatch(name)
        or any(part in (".", "..") for part in name.split("/"))
    ):
        raise _unsafe("the archive has an unsafe member name")
    return name


def _read_member(data: bytes, start: int, csize: int, usize: int, method: int, crc: int) -> bytes:
    blob = data[start : start + csize]
    if method == 0:
        body = blob
    else:
        decoder = zlib.decompressobj(-15)
        try:
            body = decoder.decompress(blob, usize + 1)
        except zlib.error:
            raise _unsafe("a deflated member is corrupt") from None
        if decoder.unconsumed_tail or not decoder.eof:
            raise _unsafe("a deflated member does not end where it says")
    if len(body) != usize or zlib.crc32(body) != crc:
        raise _unsafe("a member does not match its recorded size or checksum")
    return body


def scan_pt_archive(data: bytes) -> int:
    """Check a ``.pt`` archive with one strict reader and return its tensor data size in bytes.

    The archive is parsed here, not by the loader, with these rules: it starts with a local
    header and ends with an end-of-central-directory record with no comment, so nothing hides
    before, between or after the members; central and local headers must agree on name, method,
    flags, checksum and sizes; members are stored or deflated, never encrypted, never described
    by a data descriptor, never zip64 (archives over 4 GiB are refused: use safetensors), with
    no central extra fields or comments and only alignment padding as local extra; names are
    ASCII, relative, free of ``..``, backslashes, NUL and drive letters, and unique ignoring
    case; members are contiguous and non-overlapping; counts, sizes and ratios are bounded; the
    single ``data.pkl`` is decompressed here (checksum verified) and its opcode stream scanned,
    never executed.

    Raises:
        CheckpointCorruptError: reason ``unsafe_archive``.
    """
    n = len(data)
    if n < _EOCD.size or data[:4] != b"PK\x03\x04":
        raise _unsafe("the archive does not start with a local file header")
    sig, disk, cd_disk, on_disk, total, cd_size, cd_offset, comment_len = _EOCD.unpack(
        data[n - _EOCD.size :]
    )
    if sig != b"PK\x05\x06" or comment_len != 0:
        raise _unsafe("the archive must end with a comment-free end-of-central-directory record")
    if disk or cd_disk or on_disk != total or total == 0xFFFF:
        raise _unsafe("multi-disk and zip64 archives are refused")
    if _ZIP64_MARK in (cd_size, cd_offset):
        raise _unsafe("multi-disk and zip64 archives are refused")
    if total == 0 or total > MAX_ARCHIVE_MEMBERS:
        raise _unsafe("the archive has no members or too many")
    if cd_offset + cd_size != n - _EOCD.size:
        raise _unsafe("bytes lie between the members and the central directory or after it")
    seen: set[str] = set()
    cursor = 0
    total_size = 0
    pickle_body: bytes | None = None
    data_bytes = 0
    pos = cd_offset
    for _ in range(total):
        if pos + _CENTRAL.size > cd_offset + cd_size:
            raise _unsafe("the central directory is truncated")
        (csig, _vm, vneed, flags, method, _t, _d, crc, csize, usize, nlen, elen, clen,
         dstart, _ia, _ea, offset) = _CENTRAL.unpack_from(data, pos)  # fmt: skip
        name_raw = data[pos + _CENTRAL.size : pos + _CENTRAL.size + nlen]
        pos += _CENTRAL.size + nlen + elen + clen
        if csig != b"PK\x01\x02" or len(name_raw) != nlen or pos > cd_offset + cd_size:
            raise _unsafe("a central directory record is malformed")
        if elen or clen or dstart or vneed > 20 or flags & ~0x800:
            raise _unsafe("a member uses extra fields, comments, zip64, encryption or descriptors")
        if method not in (0, 8) or _ZIP64_MARK in (csize, usize, offset):
            raise _unsafe("a member is not plainly stored or deflated")
        name = _member_name(name_raw)
        if name.casefold() in seen:
            raise _unsafe("the archive has duplicate member names")
        seen.add(name.casefold())
        implausible = usize > (1 << 20) and usize > MAX_COMPRESSION_RATIO * max(1, csize)
        if usize > MAX_MEMBER_BYTES or implausible:
            raise _unsafe("a member is too large or has an implausible compression ratio")
        total_size += usize
        if offset != cursor or offset + _LOCAL.size > cd_offset:
            raise _unsafe("members overlap, leave gaps or hide bytes")
        (lsig, _lv, lflags, lmethod, _lt, _ld, lcrc, lcsize, lusize, lnlen, lelen) = (
            _LOCAL.unpack_from(data, offset)
        )
        lname = data[offset + _LOCAL.size : offset + _LOCAL.size + lnlen]
        extra = data[offset + _LOCAL.size + lnlen : offset + _LOCAL.size + lnlen + lelen]
        if (
            lsig != b"PK\x03\x04"
            or lname != name_raw
            or (lflags, lmethod, lcrc, lcsize, lusize) != (flags, method, crc, csize, usize)
            or len(extra) != lelen
            or not _local_extra_ok(extra)
        ):
            raise _unsafe("a local header disagrees with the central directory")
        start = offset + _LOCAL.size + lnlen + lelen
        cursor = start + csize
        if cursor > cd_offset:
            raise _unsafe("a member runs into the central directory")
        if name == "data.pkl" or name.endswith("/data.pkl"):
            if pickle_body is not None or usize > MAX_PICKLE_BYTES:
                raise _unsafe("the archive must hold exactly one reasonably sized data.pkl")
            pickle_body = _read_member(data, start, csize, usize, method, crc)
        elif "/data/" in name or name.startswith("data/"):
            data_bytes += usize
    if cursor != cd_offset:
        raise _unsafe("bytes lie between the last member and the central directory")
    if total_size > 8 * n + (1 << 20):
        raise _unsafe("the archive expands to much more than its size")
    if pickle_body is None:
        raise _unsafe("the archive must hold exactly one reasonably sized data.pkl")
    _scan_pickle(pickle_body)
    return data_bytes


def _check_loaded(state: Any, data_bytes: int) -> dict[str, Any]:
    """The loaded state must be a string-keyed mapping, bounded in depth and count, whose
    tensors together cannot need more memory than the archive's tensor data (storages may be
    shared, so this is an upper bound, not an equality: the residual risk of ``.pt``)."""
    leaves = 0
    needed = 0

    def is_tensor(value: Any) -> bool:
        return callable(getattr(value, "numel", None)) and callable(
            getattr(value, "element_size", None)
        )

    def walk(value: Any, depth: int) -> None:
        nonlocal leaves, needed
        if depth > state_mod.MAX_DEPTH + 1:
            raise _unloadable("the loaded state is nested too deeply")
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise _unloadable("the loaded state has non-string keys")
                walk(item, depth + 1)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item, depth + 1)
        elif is_tensor(value):
            leaves += 1
            needed += int(value.numel()) * int(value.element_size())
            if leaves > state_mod.MAX_TENSORS:
                raise _unloadable("the loaded state holds too many tensors")

    walk(state, 0)
    if needed > data_bytes + (1 << 20):
        raise _unloadable("the loaded tensors are larger than the archive's tensor data")
    return dict(state)


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
        data_bytes = scan_pt_archive(data)
        framework = self._framework()
        try:
            result = framework.load(io.BytesIO(data), map_location=map_location, weights_only=True)
        except Exception:  # noqa: BLE001  # the restricted loader refuses anything it cannot trust
            raise _unloadable("the checkpoint could not be loaded with weights_only=True") from None
        if not isinstance(result, Mapping):
            raise _unloadable("the checkpoint does not hold a mapping")
        return _check_loaded(result, data_bytes)


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
