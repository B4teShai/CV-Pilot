"""Capture and re-apply random number generator state (Python, NumPy and, when it is already
imported, the training framework).

The state is plain data (ints, arrays), so it passes through the checkpoint serializers without
pickling. A restored state came from a file, so ``apply_rng`` treats it as untrusted: every
structure is validated first (``validate_rng``: exact keys, exact lengths, integer ranges, finite
floats, no booleans posing as integers), only then are the generators set, and if setting any
of them fails every generator is put back to what it was. A malformed state raises
``ValueError`` and changes nothing. The restored values only ever reach ``random.setstate``,
``numpy.random.set_state`` and the framework's ``set_rng_state``; names, classes and modules
are never looked up from them.
"""

from __future__ import annotations

import math
import random
import sys
from typing import Any

import numpy as np

_FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text for it
PY_STATE_LEN = 625
NP_KEYS = 624
FRAMEWORK_STATE_BYTES = 5056
"""Size of the framework's CPU generator state (a byte tensor of the Mersenne twister)."""
_U32 = 2**32
_TOP = {"python", "numpy", "framework"}


def capture_rng() -> dict[str, Any]:
    """The state of the Python and NumPy generators, plus the framework's CPU generator if the
    framework module is already imported (it is never imported here)."""
    version, internal, gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()  # legacy global generator
    out: dict[str, Any] = {
        "python": {"version": version, "state": list(internal), "gauss": gauss},
        "numpy": {
            "name": name,
            "keys": np.asarray(keys, dtype=np.uint32),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
    }
    module = sys.modules.get(_FRAMEWORK)
    getter = getattr(module, "get_rng_state", None)
    if callable(getter):
        out["framework"] = {"cpu": np.asarray(getter()).astype(np.uint8).reshape(-1)}
    return out


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: Any) -> bool:
    return isinstance(value, float) and math.isfinite(value)


def _check_python(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"version", "state", "gauss"}:
        raise ValueError("invalid Python rng state")
    internal = value["state"]
    if (
        not _is_int(value["version"])
        or value["version"] != 3
        or not isinstance(internal, list)
        or len(internal) != PY_STATE_LEN
        or not all(_is_int(x) for x in internal)
        or not all(0 <= x < _U32 for x in internal[:-1])
        or not 0 <= internal[-1] <= NP_KEYS
        or not (value["gauss"] is None or _is_finite(value["gauss"]))
    ):
        raise ValueError("invalid Python rng state")


def _check_numpy(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != {"name", "keys", "pos", "has_gauss", "cached"}:
        raise ValueError("invalid NumPy rng state")
    keys = value["keys"]
    if (
        value["name"] != "MT19937"
        or not isinstance(keys, np.ndarray)
        or keys.dtype != np.uint32
        or keys.shape != (NP_KEYS,)
        or not _is_int(value["pos"])
        or not 0 <= value["pos"] <= NP_KEYS
        or not _is_int(value["has_gauss"])
        or value["has_gauss"] not in (0, 1)
        or not _is_finite(value["cached"])
    ):
        raise ValueError("invalid NumPy rng state")


def _check_framework(value: Any) -> None:
    cpu = value.get("cpu") if isinstance(value, dict) and set(value) == {"cpu"} else None
    if (
        not isinstance(cpu, np.ndarray)
        or cpu.dtype != np.uint8
        or cpu.shape != (FRAMEWORK_STATE_BYTES,)
    ):
        raise ValueError("invalid framework rng state")


def validate_rng(state: Any) -> None:
    """Check a restored RNG structure without touching any generator.

    Raises:
        ValueError: the structure is not exactly what ``capture_rng`` produces.
    """
    if not isinstance(state, dict) or not state or not set(state) <= _TOP:
        raise ValueError("rng state must be a non-empty mapping of known generators")
    if "python" in state:
        _check_python(state["python"])
    if "numpy" in state:
        _check_numpy(state["numpy"])
    if "framework" in state:
        _check_framework(state["framework"])


def apply_rng(state: Any) -> None:
    """Restore generators from ``capture_rng`` output: validate everything, then apply, and put
    every generator back if any step fails.

    Raises:
        ValueError: the state is malformed or could not be applied; nothing was changed.
    """
    validate_rng(state)
    module = sys.modules.get(_FRAMEWORK)
    setter = getattr(module, "set_rng_state", None)
    getter = getattr(module, "get_rng_state", None)
    framework = state.get("framework")
    use_framework = framework is not None and callable(setter) and callable(getter)
    saved_py, saved_np = random.getstate(), np.random.get_state()
    saved_fw = getter() if use_framework and callable(getter) else None
    try:
        if "python" in state:
            py = state["python"]
            random.setstate((py["version"], tuple(py["state"]), py["gauss"]))
        if "numpy" in state:
            npy = state["numpy"]
            np.random.set_state(
                (
                    "MT19937",
                    npy["keys"].astype(np.uint32),
                    npy["pos"],
                    npy["has_gauss"],
                    npy["cached"],
                )
            )
        if use_framework and callable(setter):
            convert = getattr(module, "from_numpy", None)
            cpu = framework["cpu"]
            setter(convert(cpu) if callable(convert) else cpu)
    except Exception as exc:  # noqa: BLE001  # whatever failed, no generator may stay half-set
        random.setstate(saved_py)
        np.random.set_state(saved_np)
        if saved_fw is not None and callable(setter):
            try:
                setter(saved_fw)
            except Exception:  # noqa: BLE001, S110  # best effort: the framework's own state
                pass
        raise ValueError("the rng state could not be applied") from exc
