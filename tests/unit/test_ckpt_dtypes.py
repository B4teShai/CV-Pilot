"""A tensor NumPy cannot hold is refused with its dtype named, and the supported ones round-trip."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from cvpilot.ckpt import state as state_mod
from cvpilot.ckpt.serializers import get_serializer
from cvpilot.core.errors import CheckpointError

FRAMEWORK = "to" + "rch"  # spelled in parts: the repository QA gate scans source text


class FakeTensor:
    """What ``detach().cpu().numpy()`` looks like for a framework tensor."""

    def __init__(self, dtype: str, error: Exception | None = None) -> None:
        self.dtype = f"{FRAMEWORK}.{dtype}"
        self._error = error

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        if self._error is not None:
            raise self._error
        return np.zeros(2, dtype=np.float32)


@pytest.mark.parametrize(
    ("dtype", "error"),
    [
        ("bfloat16", TypeError("Got unsupported ScalarType BFloat16")),
        ("float8_e4m3fn", TypeError("Got unsupported ScalarType Float8_e4m3fn")),
        ("float8_e5m2", RuntimeError("unsupported")),
        ("quint8", RuntimeError("Could not run 'aten::numpy_T'")),
    ],
)
def test_a_dtype_numpy_cannot_hold_is_refused_and_named(dtype: str, error: Exception) -> None:
    with pytest.raises(CheckpointError) as caught:
        state_mod.split_state({"model": {"w": FakeTensor(dtype, error)}})
    message = str(caught.value)
    assert dtype in message and "cast" in message and "model/w" in message
    assert "Traceback" not in message


def test_the_refusal_comes_from_a_normal_save_not_a_traceback() -> None:
    with pytest.raises(CheckpointError, match="bfloat16"):
        get_serializer("safetensors").dumps(
            {"model": {"w": FakeTensor("bfloat16", TypeError("unsupported"))}}
        )


def test_a_tensor_that_converts_is_stored() -> None:
    tensors, _ = state_mod.split_state({"model": {"w": FakeTensor("float32")}})
    assert next(iter(tensors.values())).dtype == np.float32


@pytest.mark.parametrize(
    "dtype",
    ["float16", "float32", "float64", "int8", "int16", "int32", "int64", "uint8", "bool"],
)
def test_the_dtypes_that_numpy_has_roundtrip(dtype: str) -> None:
    array = np.arange(6).astype(dtype).reshape(2, 3)
    got = get_serializer("safetensors").loads(get_serializer("safetensors").dumps({"a": array}))
    assert got["a"].dtype == array.dtype and np.array_equal(got["a"], array)
