"""A restored checkpoint is data: RNG structures are validated before anything is applied, the
JSON state is bounded and strict, and nothing from the file selects code."""

from __future__ import annotations

import copy
import json
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta
from safetensors.numpy import save as st_save

from cvpilot import runs
from cvpilot.ckpt import rng, serializers, state
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointCorruptError

FRAMEWORK = "to" + "rch"


def good() -> dict[str, Any]:
    random.seed(11)
    np.random.seed(11)
    return rng.capture_rng()


def mutated(path: list[Any], value: Any) -> dict[str, Any]:
    base = good()
    node: Any = base
    for part in path[:-1]:
        node = node[part]
    if value is _DELETE:
        del node[path[-1]]
    else:
        node[path[-1]] = value
    return base


_DELETE = object()
BAD: dict[str, dict[str, Any]] = {
    "python-version": mutated(["python", "version"], 2),
    "python-version-bool": mutated(["python", "version"], True),
    "python-too-short": mutated(["python", "state"], list(range(624))),
    "python-too-long": mutated(["python", "state"], list(range(626))),
    "python-negative": mutated(["python", "state", 5], -1),
    "python-over-32-bits": mutated(["python", "state", 5], 2**32),
    "python-huge": mutated(["python", "state", 5], 2**200),
    "python-bool-element": mutated(["python", "state", 5], True),
    "python-float-element": mutated(["python", "state", 5], 1.5),
    "python-string-element": mutated(["python", "state", 5], "7"),
    "python-index-over": mutated(["python", "state", 624], 625),
    "python-gauss-nan": mutated(["python", "gauss"], float("nan")),
    "python-gauss-inf": mutated(["python", "gauss"], float("inf")),
    "python-gauss-text": mutated(["python", "gauss"], "x"),
    "python-gauss-bool": mutated(["python", "gauss"], True),
    "python-extra-key": mutated(["python", "extra"], 1),
    "python-missing-key": mutated(["python", "gauss"], _DELETE),
    "python-not-a-dict": mutated(["python"], [1, 2, 3]),
    "numpy-name": mutated(["numpy", "name"], "PCG64"),
    "numpy-no-name": mutated(["numpy", "name"], _DELETE),
    "numpy-keys-dtype": mutated(["numpy", "keys"], np.zeros(624, dtype=np.int64)),
    "numpy-keys-shape": mutated(["numpy", "keys"], np.zeros(623, dtype=np.uint32)),
    "numpy-keys-list": mutated(["numpy", "keys"], [0] * 624),
    "numpy-pos-over": mutated(["numpy", "pos"], 625),
    "numpy-pos-negative": mutated(["numpy", "pos"], -1),
    "numpy-pos-bool": mutated(["numpy", "pos"], True),
    "numpy-pos-float": mutated(["numpy", "pos"], 1.5),
    "numpy-has-gauss": mutated(["numpy", "has_gauss"], 2),
    "numpy-has-gauss-bool": mutated(["numpy", "has_gauss"], True),
    "numpy-cached-nan": mutated(["numpy", "cached"], float("nan")),
    "numpy-cached-text": mutated(["numpy", "cached"], "0"),
    "numpy-extra-key": mutated(["numpy", "extra"], 1),
    "top-extra-key": mutated(["other"], {}),
    "framework-not-a-dict": mutated(["framework"], b"x"),
}


def snapshot() -> tuple[Any, Any]:
    return random.getstate(), np.random.get_state()


def same(a: tuple[Any, Any], b: tuple[Any, Any]) -> bool:
    return (
        a[0] == b[0]
        and a[1][0] == b[1][0]
        and bool(np.array_equal(a[1][1], b[1][1]))
        and a[1][2:] == b[1][2:]
    )


def test_the_reference_state_is_valid_and_round_trips_through_apply() -> None:
    base = good()
    rng.validate_rng(base)
    expected = (random.random(), float(np.random.random()))  # noqa: S311
    random.seed(0)
    np.random.seed(0)
    rng.apply_rng(base)
    assert (random.random(), float(np.random.random())) == expected  # noqa: S311


@pytest.mark.parametrize("name", list(BAD))
def test_malformed_rng_states_are_rejected_and_leave_the_generators_alone(name: str) -> None:
    random.seed(99)
    np.random.seed(99)
    before = snapshot()
    with pytest.raises(ValueError):
        rng.validate_rng(BAD[name])
    with pytest.raises(ValueError):
        rng.apply_rng(BAD[name])
    assert same(before, snapshot())


@pytest.mark.parametrize("bad", [None, [], "x", 5, {}, {"python": None}])
def test_structurally_wrong_rng_values_are_rejected(bad: Any) -> None:
    with pytest.raises(ValueError):
        rng.validate_rng(bad)


def framework_module(monkeypatch: pytest.MonkeyPatch, applied: list[Any]) -> None:
    module = types.ModuleType(FRAMEWORK)
    module.get_rng_state = lambda: np.zeros(rng.FRAMEWORK_STATE_BYTES, dtype=np.uint8)  # type: ignore[attr-defined]
    module.set_rng_state = applied.append  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, FRAMEWORK, module)


@pytest.mark.parametrize(
    "value",
    [
        np.zeros(10, dtype=np.uint8),
        np.zeros(rng.FRAMEWORK_STATE_BYTES, dtype=np.int32),
        np.zeros((rng.FRAMEWORK_STATE_BYTES, 1), dtype=np.uint8),
        [0] * rng.FRAMEWORK_STATE_BYTES,
        b"x" * rng.FRAMEWORK_STATE_BYTES,
    ],
    ids=["short", "dtype", "shape", "list", "bytes"],
)
def test_a_malformed_framework_state_is_rejected(
    monkeypatch: pytest.MonkeyPatch, value: Any
) -> None:
    applied: list[Any] = []
    framework_module(monkeypatch, applied)
    bad = good()
    bad["framework"] = {"cpu": value}
    with pytest.raises(ValueError):
        rng.apply_rng(bad)
    assert applied == []


def test_a_failure_while_applying_rolls_every_generator_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[Any] = []
    framework_module(monkeypatch, applied)
    state_in = good()
    state_in["framework"] = {"cpu": np.zeros(rng.FRAMEWORK_STATE_BYTES, dtype=np.uint8)}
    random.seed(5)
    np.random.seed(5)
    before = snapshot()

    def boom(value: Any) -> None:
        raise RuntimeError("the framework refused")

    monkeypatch.setattr(sys.modules[FRAMEWORK], "set_rng_state", boom)
    with pytest.raises(ValueError):
        rng.apply_rng(state_in)
    assert same(before, snapshot())


def test_a_valid_framework_state_reaches_the_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    applied: list[Any] = []
    framework_module(monkeypatch, applied)
    state_in = good()
    state_in["framework"] = {"cpu": np.zeros(rng.FRAMEWORK_STATE_BYTES, dtype=np.uint8)}
    rng.apply_rng(state_in)
    assert len(applied) == 1


# -- the JSON state ------------------------------------------------------------------------------


def file_with_skeleton(text: str) -> bytes:
    return bytes(st_save({}, metadata={serializers.STATE_KEY: text}))


def refuses(text: str) -> float:
    started = time.monotonic()
    with pytest.raises(CheckpointCorruptError) as info:
        serializers.get_serializer("safetensors").loads(file_with_skeleton(text))
    assert info.value.reason == "unloadable"
    return time.monotonic() - started


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1, "a": 2}',
        '{"a": {"b": 1, "b": 1}}',
        '{"a": NaN}',
        '{"a": Infinity}',
        '{"a": -Infinity}',
        '{"a": 100000000000000000000000000000}',
        '{"a": -100000000000000000000000000000}',
        '{"a": 1e999}',
        '["not", "an", "object"]',
        '{"$float": "huge"}',
        '{"x": {"$bytes": "!!not base64!!"}}',
        '{"x": {"$tensor": "missing"}}',
        '{"x": {"$intdict": {"a": 1}}}',
        "{",
        "",
    ],
)
def test_hostile_json_state_is_refused(text: str) -> None:
    refuses(text)


def test_deep_nesting_is_refused_quickly() -> None:
    assert refuses("[" * 200_000 + "]" * 200_000) < 5.0
    deep = '{"a":' * 100 + "1" + "}" * 100
    assert refuses(deep) < 5.0


def test_too_many_nodes_are_refused() -> None:
    assert refuses("{" + ",".join(f'"k{i}":{i}' for i in range(state.MAX_NODES + 1)) + "}") < 10.0


def test_an_oversized_state_is_refused() -> None:
    assert refuses('{"s":"' + "x" * (state.MAX_STATE_JSON_BYTES + 10) + '"}') < 5.0


def test_dunder_keys_and_module_like_strings_come_back_as_inert_data() -> None:
    hostile = {
        "__class__": "os.system",
        "__reduce__": ["os", "system", "echo pwned"],
        "module": "subprocess.Popen",
        "nested": {"__import__": "evil", "cls": "builtins.eval"},
        "list": ["__builtins__", "../../etc/passwd"],
        "negative": -5,
    }
    codec = serializers.get_serializer("safetensors")
    before = set(sys.modules)
    back = codec.loads(codec.dumps({"model": {"w": np.ones(2)}, **hostile}))
    assert {k: back[k] for k in hostile} == hostile
    assert not {m for m in set(sys.modules) - before if m in {"evil", "subprocess.Popen"}}
    assert all(isinstance(v, (str, list, dict, int)) for v in (back[k] for k in hostile))


def test_only_plain_data_survives_validation() -> None:
    state.validate_plain({"a": [1, 2.5, "x", None, True, b"b", np.zeros(2)], "b": {3: {"c": 1}}})
    for bad in (
        {"a": object()},
        {"a": {1, 2}},
        {"a": 2**70},
        {"a": np.array(["x"], dtype=object)},
        {"a": complex(1, 1)},
        {"a": lambda: 1},
    ):
        with pytest.raises(ValueError):
            state.validate_plain(bad)


# -- the manager skips a checkpoint whose RNG state is malformed ---------------------------------


def test_the_manager_skips_a_checkpoint_with_a_malformed_rng_state(tmp_path: Path) -> None:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False, keep_last=5
    )
    manager.save_now(1, model={"w": np.ones(2)}, rng=True)
    evil = copy.deepcopy(good())
    evil["python"]["state"] = list(range(10))
    manager.save_now(2, model={"w": np.ones(2)}, rng=evil)
    random.seed(1)
    np.random.seed(1)
    before = snapshot()
    restored = manager.restore()
    assert restored is not None and restored.manifest.step == 1
    assert [(s.step, s.reason) for s in restored.skipped] == [(2, "unloadable")]
    assert same(before, snapshot())
    run.finish()


def test_the_loaded_state_json_is_not_reinterpreted(tmp_path: Path) -> None:
    """A restored value is never used to import, call or look up anything."""
    payload = json.dumps({"x": "os.system"})
    back = serializers.get_serializer("safetensors").loads(file_with_skeleton(payload))
    assert back == {"x": "os.system"}
