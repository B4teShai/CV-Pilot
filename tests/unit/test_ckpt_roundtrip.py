"""Whatever a save accepts, a restore accepts: the two sides share one set of bounds."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import state as state_mod
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.errors import CheckpointError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
LIMIT = state_mod.MAX_INT


def roundtrip(state: dict[str, Any]) -> dict[str, Any]:
    serializer = get_serializer("safetensors")
    return serializer.loads(serializer.dumps(state))


@pytest.mark.parametrize(
    "value",
    [0, -1, 10**18, 10**19, 2**63, 2**64 - 1, -(2**64 - 1), -(10**19), 12345678901234567890],
)
def test_every_integer_below_the_bound_survives_a_roundtrip(value: int) -> None:
    got = roundtrip({"n": value, "nested": [value], "d": {1: value}})
    assert got["n"] == value and got["nested"] == [value] and got["d"] == {1: value}


@pytest.mark.parametrize("value", [2**64, -(2**64), 10**30, 2**200])
def test_an_integer_the_restore_would_refuse_is_refused_at_save(value: int) -> None:
    with pytest.raises(CheckpointError, match="integer|key"):
        state_mod.split_state({"n": value})
    with pytest.raises(CheckpointError, match="integer|key"):
        state_mod.split_state({"d": {value: 1}})


def test_a_state_with_too_many_nodes_is_refused_at_save() -> None:
    with pytest.raises(CheckpointError, match="too large"):
        state_mod.split_state({"xs": [0] * (state_mod.MAX_NODES + 1)})


def test_an_oversized_key_is_refused_at_save() -> None:
    with pytest.raises(CheckpointError, match="key"):
        state_mod.split_state({"k" * (state_mod.MAX_KEY_CHARS + 1): 1})


def test_depth_at_the_limit_roundtrips_and_beyond_it_is_refused() -> None:
    deep: Any = 1
    for _ in range(state_mod.MAX_DEPTH - 2):
        deep = {"d": deep}
    assert roundtrip({"x": deep})["x"] is not None
    over: Any = 1
    for _ in range(state_mod.MAX_DEPTH + 1):
        over = {"d": over}
    with pytest.raises(CheckpointError):
        state_mod.split_state({"x": over})


def test_floats_bytes_and_arrays_roundtrip_exactly() -> None:
    state = {
        "f": [0.1, 1e308, -1e-308, math.inf, -math.inf],
        "b": b"\x00\xff" * 10,
        "a": np.arange(6, dtype=np.float32).reshape(2, 3),
        "s": np.float64(2.5),
        "scalar": np.array(3.0),
    }
    got = roundtrip(state)
    assert got["f"][:3] == state["f"][:3] and got["f"][3:] == [math.inf, -math.inf]
    assert got["b"] == state["b"] and np.array_equal(got["a"], state["a"])
    assert got["s"] == 2.5 and float(got["scalar"]) == 3.0


def test_a_manager_save_of_a_large_integer_restores(tmp_path: Path) -> None:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "s")
    mgr = CheckpointManager(run, dir=tmp_path / "c", every_n_steps=1, install_signal_handlers=False)
    mgr.save_now(1, model={"w": np.ones(2)}, dataloader_state={"seed": 2**64 - 1})
    restored = mgr.restore()
    assert restored is not None and restored.state["dataloader_state"]["seed"] == 2**64 - 1
    with pytest.raises(CheckpointError):
        mgr.save_now(2, model={"w": np.ones(2)}, dataloader_state={"seed": 2**64})
    run.finish()


def _random_value(rng: Any, depth: int) -> Any:
    kinds = ["int", "bigint", "float", "str", "bool", "none", "bytes"]
    if depth < 5:
        kinds += ["list", "dict", "intdict"]
    kind = rng.choice(kinds)
    if kind == "int":
        return rng.randint(-(10**6), 10**6)
    if kind == "bigint":
        return rng.choice([-1, 1]) * rng.randint(0, 2**65)
    if kind == "float":
        return rng.choice([0.0, 1.5, -2.25, 1e300, math.inf, -math.inf])
    if kind == "str":
        return rng.choice(["", "a", "é" * 5, "x" * 50])
    if kind == "bool":
        return rng.random() < 0.5
    if kind == "none":
        return None
    if kind == "bytes":
        return bytes(rng.randrange(256) for _ in range(rng.randrange(8)))
    if kind == "list":
        return [_random_value(rng, depth + 1) for _ in range(rng.randrange(4))]
    if kind == "intdict":
        return {
            rng.randint(-5, 5): _random_value(rng, depth + 1) for _ in range(rng.randrange(1, 4))
        }
    return {f"k{i}": _random_value(rng, depth + 1) for i in range(rng.randrange(4))}


def test_random_states_either_roundtrip_exactly_or_are_refused_at_save() -> None:
    import random

    rng = random.Random(0)  # noqa: S311  # a reproducible fuzz seed, not a secret
    serializer = get_serializer("safetensors")
    saved = refused = 0
    for _ in range(400):
        state = {"root": _random_value(rng, 0)}
        try:
            data = serializer.dumps(state)
        except CheckpointError:
            refused += 1
            continue
        got = serializer.loads(data)  # whatever was saved must load: no exception allowed here
        saved += 1
        assert got == state or repr(got) == repr(state)
    assert saved > 100 and refused > 0  # both outcomes were exercised
