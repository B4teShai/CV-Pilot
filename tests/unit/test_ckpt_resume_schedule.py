"""A restored checkpoint seeds the save schedule: the first save after a resume is not due
at once."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")


def weights(value: float) -> dict[str, np.ndarray]:
    return {"w": np.full(2, value, dtype=np.float32)}


def make(tmp_path: Path, **kw: object) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(run, dir=tmp_path / "ckpt", install_signal_handlers=False, **kw)  # type: ignore[arg-type]
    return run, mgr


def test_the_first_save_after_a_restore_waits_for_the_interval(tmp_path: Path) -> None:
    run, first = make(tmp_path, every_n_steps=3, keep_last=10)
    for step in (1, 2, 3, 4):
        first.save_now(step, model=weights(step))
    first.close()
    second = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=3, keep_last=10, install_signal_handlers=False
    )
    restored = second.restore()
    assert restored is not None and restored.manifest.step == 4
    assert second.maybe_save(5, model=weights(5)) is None  # used to save at once (5 - 0 >= 3)
    assert second.maybe_save(6, model=weights(6)) is None
    saved = second.maybe_save(7, model=weights(7))
    assert saved is not None and saved.step == 7
    second.close()
    run.finish()
