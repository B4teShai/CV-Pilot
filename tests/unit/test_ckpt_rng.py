"""Random number generator state travels with the checkpoint and can be re-applied."""

from __future__ import annotations

import random
import sys
import types
from typing import Any

import numpy as np
import pytest

from cvpilot.ckpt import rng, state
from cvpilot.ckpt.serializers import get_serializer


def test_python_and_numpy_streams_continue_exactly_after_a_round_trip() -> None:
    random.seed(7)
    np.random.seed(7)
    captured = rng.capture_rng()
    expected = (random.random(), float(np.random.random()))  # noqa: S311
    random.seed(0)
    np.random.seed(0)
    codec = get_serializer("safetensors")
    restored = codec.loads(codec.dumps({"rng": captured}))["rng"]
    rng.apply_rng(restored)
    assert (random.random(), float(np.random.random())) == expected  # noqa: S311


def test_captured_state_is_storable() -> None:
    tensors, skeleton = state.split_state({"rng": rng.capture_rng()})
    assert tensors and skeleton


def test_the_framework_generator_is_included_only_when_it_is_already_imported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    name = "to" + "rch"
    monkeypatch.delitem(sys.modules, name, raising=False)
    assert "framework" not in rng.capture_rng()
    seen: dict[str, Any] = {}
    fake = types.ModuleType(name)
    fake.get_rng_state = lambda: np.zeros(rng.FRAMEWORK_STATE_BYTES, dtype=np.uint8)  # type: ignore[attr-defined]
    fake.set_rng_state = lambda value: seen.setdefault("set", value)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, name, fake)
    captured = rng.capture_rng()
    assert "framework" in captured
    rng.apply_rng(captured)
    assert "set" in seen


@pytest.mark.parametrize(
    "bad", [None, [], {"python": "x"}, {"python": [3, [1], None]}, {"numpy": 1}]
)
def test_applying_a_malformed_state_raises_value_error(bad: Any) -> None:
    with pytest.raises(ValueError):
        rng.apply_rng(bad)
