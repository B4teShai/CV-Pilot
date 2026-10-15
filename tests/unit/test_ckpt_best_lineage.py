"""A resumed run must not overwrite the lineage's best with something worse."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.hashing import new_run_id

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def first_run(tmp: Path, values: list[float], mode: str, monitor: str = "m") -> str:
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_best=3, monitor=monitor,
        mode=mode, install_signal_handlers=False,  # type: ignore[arg-type]
    )  # fmt: skip
    for step, value in enumerate(values, start=1):
        mgr.maybe_save(step, model=weights(value), rng=False, metrics={monitor: value})
    mgr.close()
    run.finish("preempted")
    return run.run_id


def bests(mgr: Any) -> list[float | None]:
    return [e.manifest.metric_value for e in mgr.list() if e.kind == "best" and e.manifest]


def child(tmp: Path, parent: str, mode: str, monitor: str = "m") -> tuple[runs.RunHandle, Any]:
    _, repro = run_meta("p")
    meta = runs.resume_from(
        runs.get_run_view("p", parent, store_dir=tmp / "store").meta,
        run_id=new_run_id(), host="h", name="job",
    )  # fmt: skip
    handle = runs.start_run(
        meta, repro.model_copy(update={"run_id": meta.run_id}), store_dir=tmp / "store"
    )
    mgr = CheckpointManager(
        handle, dir=tmp / "ckpt", every_n_steps=1, keep_best=3, monitor=monitor,
        mode=mode, install_signal_handlers=False,  # type: ignore[arg-type]
    )  # fmt: skip
    return handle, mgr


@pytest.mark.parametrize(
    ("mode", "values", "worse", "better"),
    [("max", [0.5, 0.9, 0.7], 0.8, 0.95), ("min", [3.0, 1.0, 2.0], 1.5, 0.5)],
)
def test_a_resumed_run_saves_a_best_only_when_it_beats_its_ancestors(
    tmp_path: Path, mode: str, values: list[float], worse: float, better: float
) -> None:
    parent = first_run(tmp_path, values, mode)
    handle, mgr = child(tmp_path, parent, mode)
    mgr.maybe_save(10, model=weights(worse), rng=False, metrics={"m": worse})
    assert bests(mgr) == []  # a last was saved (the interval elapsed), but no best
    mgr.maybe_save(11, model=weights(better), rng=False, metrics={"m": better})
    assert bests(mgr) == [better]
    handle.finish()


def test_a_different_monitored_metric_does_not_seed_the_comparison(tmp_path: Path) -> None:
    parent = first_run(tmp_path, [0.9], "max", monitor="old/metric")
    handle, mgr = child(tmp_path, parent, "max", monitor="new/metric")
    mgr.maybe_save(10, model=weights(0.1), rng=False, metrics={"new/metric": 0.1})
    assert bests(mgr) == [0.1]
    handle.finish()


def test_a_different_mode_does_not_seed_the_comparison(tmp_path: Path) -> None:
    parent = first_run(tmp_path, [0.9], "max")
    handle, mgr = child(tmp_path, parent, "min")
    mgr.maybe_save(10, model=weights(0.95), rng=False, metrics={"m": 0.95})
    assert bests(mgr) == [0.95]
    handle.finish()


def test_the_best_across_the_lineage_is_what_a_restore_returns(tmp_path: Path) -> None:
    parent = first_run(tmp_path, [0.5, 0.9], "max")
    handle, mgr = child(tmp_path, parent, "max")
    mgr.maybe_save(10, model=weights(0.2), rng=False, metrics={"m": 0.2})  # worse: not saved
    restored = mgr.restore("best")
    assert restored is not None and restored.manifest.metric_value == 0.9
    assert restored.run_id == parent
    handle.finish()
