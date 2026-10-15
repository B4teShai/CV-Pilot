"""Golden: 100 000 logged points read back identical, through the public API."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

import cvpilot
from cvpilot.store.local import open_store

STEPS, KEYS = 2000, 50


def value(step: int, k: int) -> float:
    if (step + k) % 997 == 0:
        return math.nan
    if (step * 31 + k) % 1009 == 0:
        return -math.inf
    return math.sin(step * 0.01 + k) * 10 ** (k % 7 - 3)


@pytest.mark.filterwarnings("ignore")
def test_one_hundred_thousand_points_read_back_identically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    run = cvpilot.init("golden", task="detection", store_dir=tmp_path / "s")
    keys = [f"val/m{k:02d}" for k in range(KEYS)]
    for step in range(STEPS):
        run.log({key: value(step, k) for k, key in enumerate(keys)}, step=step, epoch=step // 100)
    run.finish()

    store = open_store("golden", tmp_path / "s", create=False)
    points = store.read_points(run.run_id)
    store.close()
    assert len(points) == STEPS * KEYS
    seen = set()
    for p in points:
        k = keys.index(p.key)
        want = value(p.step, k)
        assert (math.isnan(want) and math.isnan(p.value)) or p.value == want
        assert p.epoch == p.step // 100
        seen.add((p.key, p.step))
    assert len(seen) == STEPS * KEYS
