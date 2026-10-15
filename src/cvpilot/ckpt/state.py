"""Training state split into tensors and a bounded JSON skeleton, and back.

A checkpoint state is a nested mapping. Arrays (NumPy arrays and framework tensors, which are
converted through ``detach().cpu().numpy()``) go to the tensor file under a name derived from
their path; everything else (numbers, strings, bytes, lists, dicts, ``None``) stays in a JSON
skeleton in which each tensor is replaced by ``{"$tensor": "<name>"}``. The skeleton is bounded
to ``MAX_STATE_JSON_BYTES`` and ``MAX_DEPTH`` levels; tuples come back as lists; floats that JSON
cannot hold (inf, nan), bytes and dicts keyed by integers (optimizer state) use the markers
``$float``, ``$bytes``, ``$intdict`` and (0-d arrays) ``$scalar``. Nothing here executes
code on load: the skeleton is parsed as JSON and only these markers are interpreted.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from cvpilot.core.errors import CheckpointError

MAX_DEPTH = 32
MAX_TENSORS = 100_000
MAX_STATE_JSON_BYTES = 1 << 20
MAX_NODES = 100_000
MAX_KEY_CHARS = 4096
MAX_INT = 2**64
TENSOR = "$tensor"
FLOAT = "$float"
BYTES = "$bytes"
INTDICT = "$intdict"
SCALAR = "$scalar"
_MARKERS = {TENSOR, FLOAT, BYTES, INTDICT, SCALAR}
_FLOATS = {"inf": math.inf, "-inf": -math.inf, "nan": math.nan}


def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def reject_constant(name: str) -> Any:
    raise ValueError("non-finite number")


def _bounded_signed_int(text: str) -> int:
    value = int(text)  # at most 4300 digits are converted; longer text raises ValueError
    if abs(value) >= MAX_INT:
        raise ValueError("integer out of range")
    return value


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError("non-finite number")
    return value


def parse_skeleton(text: str) -> Any:
    """Parse skeleton JSON as restore does: no duplicate keys, no NaN or Infinity constants,
    integers below ``MAX_INT`` in magnitude, finite floats. Save runs the same parser over what it
    wrote, so nothing is saved that a restore would refuse.

    Raises:
        ValueError: any of the above is violated (``RecursionError`` for absurd nesting).
    """
    return json.loads(
        text,
        object_pairs_hook=no_duplicate_keys,
        parse_constant=reject_constant,
        parse_int=_bounded_signed_int,
        parse_float=_finite_float,
    )


def _escape(part: str) -> str:
    return part.replace("~", "~0").replace("/", "~1")


def _to_array(value: Any) -> np.ndarray[Any, Any] | None:
    if isinstance(value, np.ndarray):
        array = value
    elif all(callable(getattr(value, name, None)) for name in ("detach", "cpu", "numpy")):
        array = np.asarray(value.detach().cpu().numpy())
    else:
        return None
    if array.dtype.hasobject:
        raise CheckpointError("an array of Python objects cannot be stored in a checkpoint")
    # ascontiguousarray would turn a 0-d array into a 1-d one; 0-d arrays are contiguous already
    return array if array.flags.c_contiguous else np.ascontiguousarray(array)


def split_state(state: Mapping[str, Any]) -> tuple[dict[str, np.ndarray[Any, Any]], dict[str, Any]]:
    """``(tensors, skeleton)`` for ``state``.

    Raises:
        CheckpointError: an unsupported type, a non-string key, a key that collides with a
            marker, nesting deeper than ``MAX_DEPTH``, more than ``MAX_TENSORS`` tensors or a
            skeleton larger than ``MAX_STATE_JSON_BYTES``.
    """
    tensors: dict[str, np.ndarray[Any, Any]] = {}

    def walk(value: Any, path: tuple[str, ...], depth: int) -> Any:
        if depth > MAX_DEPTH:
            raise CheckpointError(f"checkpoint state is nested deeper than {MAX_DEPTH} levels")
        array = _to_array(value)
        if array is not None:
            name = "/".join(_escape(p) for p in path)
            if len(tensors) >= MAX_TENSORS:
                raise CheckpointError(f"checkpoint state has more than {MAX_TENSORS} tensors")
            if array.ndim == 0:  # the tensor file stores a scalar as one element: say so
                tensors[name] = array.reshape(1)
                return {SCALAR: name}
            tensors[name] = array
            return {TENSOR: name}
        if value is None or isinstance(value, (bool, str)):
            return value
        if isinstance(value, np.generic):
            return walk(value.item(), path, depth)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float):
            if math.isfinite(value):
                return float(value)
            return {FLOAT: "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")}
        if isinstance(value, (bytes, bytearray)):
            return {BYTES: base64.b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, Mapping):
            if len(value) == 1 and next(iter(value)) in _MARKERS:
                raise CheckpointError(
                    f"a state key may not be the reserved name {next(iter(value))!r}"
                )
            if value and all(isinstance(k, int) and not isinstance(k, bool) for k in value):
                # optimizer state is keyed by parameter index: keep the keys' type
                return {
                    INTDICT: {str(k): walk(v, (*path, str(k)), depth + 1) for k, v in value.items()}
                }
            out: dict[str, Any] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise CheckpointError(
                        f"state keys must be strings, found {type(key).__name__} at "
                        f"{'/'.join(path) or '<top>'}"
                    )
                out[key] = walk(item, (*path, key), depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            return [walk(item, (*path, str(i)), depth + 1) for i, item in enumerate(value)]
        raise CheckpointError(
            f"cannot store a {type(value).__name__} in a checkpoint "
            "(tensors, numbers, strings, bytes, lists and dicts only)"
        )

    if not all(isinstance(k, str) for k in state):
        raise CheckpointError("top-level state keys must be strings")
    skeleton = walk(dict(state), (), 0)
    text = json.dumps(skeleton, allow_nan=False, separators=(",", ":"), sort_keys=True)
    if len(text.encode("utf-8")) > MAX_STATE_JSON_BYTES:
        raise CheckpointError(
            f"the non-tensor state is larger than {MAX_STATE_JSON_BYTES} bytes of JSON; "
            "store large buffers as arrays"
        )
    try:  # the exact checks a restore applies: a save never writes what a restore would refuse
        validate_plain(merge_state(tensors, parse_skeleton(text)))
    except (ValueError, RecursionError) as exc:
        raise CheckpointError(
            f"this state could not be restored, so it is not saved: {exc}"
        ) from None
    return tensors, skeleton


def skeleton_json(skeleton: Mapping[str, Any]) -> str:
    """The canonical JSON text of a skeleton (sorted keys, compact)."""
    return json.dumps(skeleton, allow_nan=False, separators=(",", ":"), sort_keys=True)


def merge_state(
    tensors: Mapping[str, np.ndarray[Any, Any]], skeleton: Mapping[str, Any]
) -> dict[str, Any]:
    """Rebuild the nested state; every tensor must be referenced exactly once.

    Raises:
        ValueError: the skeleton is malformed, too deep, references a missing tensor or leaves
            one unreferenced.
    """
    remaining = dict(tensors)

    def walk(value: Any, depth: int) -> Any:
        if depth > MAX_DEPTH + 1:
            raise ValueError("state skeleton is nested too deeply")
        if isinstance(value, dict):
            if len(value) == 1:
                (key,) = value
                if key == TENSOR:
                    name = value[key]
                    if not isinstance(name, str) or name not in remaining:
                        raise ValueError("state refers to a tensor that is not in the file")
                    return remaining.pop(name)
                if key == SCALAR:
                    name = value[key]
                    if not isinstance(name, str) or name not in remaining:
                        raise ValueError("state refers to a tensor that is not in the file")
                    scalar = remaining.pop(name)
                    if scalar.shape != (1,):
                        raise ValueError("a scalar tensor must hold exactly one element")
                    return scalar.reshape(())
                if key == INTDICT:
                    inner = value[key]
                    if not isinstance(inner, dict) or not all(
                        isinstance(k, str) and k.lstrip("-").isdigit() for k in inner
                    ):
                        raise ValueError("invalid integer-key marker")
                    return {int(k): walk(v, depth + 1) for k, v in inner.items()}
                if key == FLOAT:
                    if value[key] not in _FLOATS:
                        raise ValueError("invalid float marker")
                    return _FLOATS[value[key]]
                if key == BYTES:
                    try:
                        return base64.b64decode(str(value[key]), validate=True)
                    except (binascii.Error, ValueError):
                        raise ValueError("invalid bytes marker") from None
            return {str(k): walk(v, depth + 1) for k, v in value.items()}
        if isinstance(value, list):
            return [walk(v, depth + 1) for v in value]
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        raise ValueError("unsupported value in the state skeleton")

    merged = walk(dict(skeleton), 0)
    if remaining:
        raise ValueError("the file holds tensors the state does not refer to")
    if not isinstance(merged, dict):  # pragma: no cover  # walk(dict) returns a dict
        raise ValueError("state must be a mapping")
    return merged


def validate_plain(value: Any) -> None:
    """Check that a restored state is plain data and bounded.

    Allowed: dicts (string keys, or integer keys from ``$intdict``), lists, strings, booleans,
    ``None``, integers below 2**64 in magnitude, floats (inf and nan only via ``$float``),
    bytes up to the state limit and NumPy arrays that are not arrays of objects; at most
    ``MAX_NODES`` nodes and ``MAX_DEPTH + 2`` levels. Nothing here looks anything up by a name
    from the data.

    Raises:
        ValueError: anything else, or a bound is exceeded.
    """
    nodes = 0

    def walk(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > MAX_NODES or depth > MAX_DEPTH + 2:
            raise ValueError("the restored state is too large or too deep")
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, int):
            if abs(item) >= MAX_INT:
                raise ValueError("the restored state holds an oversized integer")
        elif isinstance(item, float):
            return  # inf and nan arrive only through the $float marker; they are plain data
        elif isinstance(item, str):
            if len(item) > MAX_STATE_JSON_BYTES:
                raise ValueError("the restored state holds an oversized string")
        elif isinstance(item, bytes):
            if len(item) > MAX_STATE_JSON_BYTES:
                raise ValueError("the restored state holds oversized bytes")
        elif isinstance(item, np.ndarray):
            if item.dtype.hasobject:
                raise ValueError("the restored state holds an array of objects")
        elif isinstance(item, dict):
            for key, child in item.items():
                if isinstance(key, bool) or not isinstance(key, (str, int)):
                    raise ValueError("the restored state has an unsupported key")
                if isinstance(key, str) and len(key) > MAX_KEY_CHARS:
                    raise ValueError("the restored state has an oversized key")
                if isinstance(key, int) and abs(key) >= MAX_INT:
                    raise ValueError("the restored state has an oversized key")
                walk(child, depth + 1)
        elif isinstance(item, list):
            for child in item:
                walk(child, depth + 1)
        else:
            raise ValueError(f"the restored state holds a {type(item).__name__}")

    walk(value, 0)
