"""``ckpt prune`` ranks best checkpoints by the mode they were saved with, not by a default."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from ckpt_support import run_meta
from conftest import CliResult

from cvpilot import runs
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Invoke = Callable[..., CliResult]
LOSSES = {1: 3.0, 2: 1.0, 3: 2.0, 4: 2.5}


def seed(store_root: Path, tmp_path: Path, mode: str) -> str:
    meta, repro = run_meta("proj")
    run = runs.start_run(meta, repro, store_dir=store_root)
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, keep_last=10, keep_best=10,
        monitor="val/loss", mode=mode, install_signal_handlers=False,  # type: ignore[arg-type]
    )  # fmt: skip
    for step, loss in LOSSES.items():
        mgr.save_now(step, model={"w": np.full(2, float(step))}, metrics={"val/loss": loss})
        mgr.save_now(
            step, kind="best", model={"w": np.full(2, float(step))}, metrics={"val/loss": loss}
        )
    run_id = run.run_id
    mgr.close()
    run.finish("preempted")
    return run_id


def best_steps(invoke: Invoke, tmp_path: Path) -> list[int]:
    rows = json.loads(
        invoke("ckpt", "list", str(tmp_path / "ckpt"), "--kind", "best", "--json").out
    )
    return sorted(r["step"] for r in rows)


def test_the_default_prune_keeps_the_best_loss_of_a_min_run(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run_id = seed(store_root, tmp_path, "min")
    res = invoke("ckpt", "prune", str(tmp_path / "ckpt"), "--run", run_id, "--keep-best", "1")
    assert res.code == 0, res.err
    assert best_steps(invoke, tmp_path) == [2, 4]  # the lowest loss, and the newest


def test_a_max_run_still_keeps_the_highest_value(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run_id = seed(store_root, tmp_path, "max")
    invoke("ckpt", "prune", str(tmp_path / "ckpt"), "--run", run_id, "--keep-best", "1")
    assert best_steps(invoke, tmp_path) == [1, 4]  # loss 3.0 is the highest, plus the newest


def test_a_conflicting_explicit_mode_is_an_error_that_deletes_nothing(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run_id = seed(store_root, tmp_path, "min")
    res = invoke(
        "ckpt", "prune", str(tmp_path / "ckpt"), "--run", run_id, "--keep-best", "1",
        "--mode", "max",
    )  # fmt: skip
    assert res.code == 2 and "mode" in res.err
    assert best_steps(invoke, tmp_path) == [1, 2, 3, 4]


def test_a_matching_explicit_mode_is_accepted(
    invoke: Invoke, store_root: Path, tmp_path: Path
) -> None:
    run_id = seed(store_root, tmp_path, "min")
    res = invoke(
        "ckpt", "prune", str(tmp_path / "ckpt"), "--run", run_id, "--keep-best", "1",
        "--mode", "min",
    )  # fmt: skip
    assert res.code == 0 and best_steps(invoke, tmp_path) == [2, 4]


def test_the_directory_api_raises_on_a_conflicting_mode(store_root: Path, tmp_path: Path) -> None:
    run_id = seed(store_root, tmp_path, "min")
    rc = RunCheckpoints(tmp_path / "ckpt", run_id)
    with pytest.raises(CheckpointError, match="mode"):
        rc.prune("best", keep=1, mode="max")
    assert [e.step for e in rc.inspect("best")] == [4, 3, 2, 1]
    assert rc.prune("best", keep=1).deleted  # no mode given: taken from the manifests
    assert sorted(e.step for e in rc.inspect("best")) == [2, 4]


def test_prune_leaves_a_restorable_chain(invoke: Invoke, store_root: Path, tmp_path: Path) -> None:
    run_id = seed(store_root, tmp_path, "min")
    invoke("ckpt", "prune", str(tmp_path / "ckpt"), "--run", run_id, "--keep-best", "1",
           "--keep-last", "1")  # fmt: skip
    verdict = invoke("ckpt", "verify", str(tmp_path / "ckpt"))
    assert verdict.code == 0, verdict.out
    meta, repro = run_meta("proj")
    run = runs.start_run(meta, repro, store_dir=store_root)
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    best = mgr.restore("best", run_ids=[run_id])
    assert best is not None and best.manifest.step == 4  # newest wins among the kept
    kept = [e.step for e in RunCheckpoints(tmp_path / "ckpt", run_id).inspect("best")]
    assert 2 in kept  # the lowest loss survived
    run.finish()
