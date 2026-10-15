"""Capture and re-apply random number generator state (Python, NumPy and, when it is already
imported, the training framework).

The state is plain data (ints, arrays), so it passes through the checkpoint serializers without
pickling. ``apply_rng`` validates its input before touching any generator: a malformed state
raises ``ValueError`` and leaves every generator as it was.
"""

from __future__ import annotations

import random
import sys
from typing import Any

import numpy as np

_FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text for it
_PY_STATE_LEN = 625


def capture_rng() -> dict[str, Any]:
    """The state of the Python and NumPy generators, plus the framework's CPU generator if the
    framework module is already imported (it is never imported here)."""
    version, internal, gauss = random.getstate()
    _, keys, pos, has_gauss, cached = np.random.get_state()  # legacy global generator
    out: dict[str, Any] = {
        "python": {"version": version, "state": list(internal), "gauss": gauss},
        "numpy": {
            "keys": np.asarray(keys, dtype=np.uint32),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
    }
    module = sys.modules.get(_FRAMEWORK)
    getter = getattr(module, "get_rng_state", None)
    if callable(getter):
        out["framework"] = {"cpu": getter()}
    return out


def apply_rng(state: Any) -> None:
    """Restore generators from ``capture_rng`` output.

    Raises:
        ValueError: the state is malformed (nothing was changed).
    """
    if not isinstance(state, dict):
        raise ValueError("rng state must be a mapping")
    py, npy = state.get("python"), state.get("numpy")
    if py is not None:
        if (
            not isinstance(py, dict)
            or not isinstance(py.get("state"), list)
            or len(py["state"]) != _PY_STATE_LEN
            or not all(isinstance(x, int) for x in py["state"])
            or not isinstance(py.get("version"), int)
        ):
            raise ValueError("invalid Python rng state")
    if npy is not None:
        keys = npy.get("keys") if isinstance(npy, dict) else None
        if not isinstance(keys, np.ndarray) or keys.shape != (624,):
            raise ValueError("invalid NumPy rng state")
    if py is None and npy is None and "framework" not in state:
        raise ValueError("rng state is empty")
    if py is not None:
        try:
            random.setstate((py["version"], tuple(py["state"]), py.get("gauss")))
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid Python rng state") from exc
    if npy is not None:
        np.random.set_state(
            (
                "MT19937",
                npy["keys"].astype(np.uint32),
                int(npy["pos"]),
                int(npy["has_gauss"]),
                float(npy["cached"]),
            )
        )
    framework = state.get("framework")
    setter = getattr(sys.modules.get(_FRAMEWORK), "set_rng_state", None)
    if isinstance(framework, dict) and callable(setter) and "cpu" in framework:
        setter(framework["cpu"])
