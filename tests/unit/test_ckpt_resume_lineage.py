"""Choosing what to resume: lineages, crashed resumed runs, liveness and job identity."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import resume
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.hashing import new_run_id

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(4, value, dtype=np.float32)}


class World:
    """A project store and a checkpoint directory under ``tmp_path``."""

    def __init__(self, tmp_path: Path) -> None:
        self.store = tmp_path / "store"
        self.ckpt = tmp_path / "ckpt"

    def start(self, *, parent: str | None = None, name: str = "job") -> runs.RunHandle:
        meta, repro = run_meta("p")
        if parent is not None:
            parent_meta = runs.get_run_view("p", parent, store_dir=self.store).meta
            meta = runs.resume_from(parent_meta, run_id=new_run_id(), host="h", name=name)
            repro = repro.model_copy(update={"run_id": meta.run_id})
        else:
            meta = meta.model_copy(update={"name": name})
        return runs.start_run(meta, repro, store_dir=self.store)

    def manager(self, run: runs.RunHandle, **kw: Any) -> CheckpointManager:
        return CheckpointManager(
            run, dir=self.ckpt, every_n_steps=1, keep_last=20, install_signal_handlers=False, **kw
        )

    def find(self, **kw: Any) -> resume.ResumePoint | None:
        return resume.find_resume_point("p", checkpoint_dir=self.ckpt, store_dir=self.store, **kw)


def test_preempt_resume_crash_resume_continues_from_the_crashed_runs_progress(
    tmp_path: Path, events: Events
) -> None:
    world = World(tmp_path)
    a = world.start()
    ma = world.manager(a)
    for step in range(1, 15):
        ma.save_now(step, model=weights(float(step)))
    a_id = a.run_id
    ma.close()
    a.finish("preempted")

    point = world.find()
    assert point is not None and point.run_id == a_id and point.manifest.step == 14
    b = world.start(parent=a_id)
    mb = world.manager(b)
    restored = mb.restore()
    assert restored is not None and restored.manifest.step == 14
    for step in range(15, 26):
        mb.save_now(step, model=weights(float(step)))
    b_id = b.run_id
    mb.close()
    b.abandon()  # killed: the status stays `resumed`

    point = world.find()
    assert point is not None and point.manifest.step == 25  # not A's step 14
    assert point.run_id == b_id and point.status == "resumed" and point.needs_recover is True
    selected = [e for e in events() if e["message"] == "resume_point_selected"]
    assert selected and selected[-1]["step"] == 25 and selected[-1]["run_id"] == b_id
    assert selected[-1]["reason"]

    runs.recover_run("p", b_id, store_dir=world.store)
    c = world.start(parent=b_id)
    restored_c = world.manager(c).restore()
    assert restored_c is not None and restored_c.manifest.step == 25
    assert restored_c.run_id == b_id and c.meta.parent_run_id == b_id
    c.finish()


def test_a_run_that_already_has_a_child_is_not_a_starting_point(tmp_path: Path) -> None:
    world = World(tmp_path)
    a = world.start()
    world.manager(a).save_now(3, model=weights(3.0))
    a_id = a.run_id
    a.finish("preempted")
    b = world.start(parent=a_id)
    b.finish()  # the resumed job completed: the lineage is done
    assert world.find() is None


def test_a_child_that_crashed_before_saving_hands_back_its_parents_checkpoint(
    tmp_path: Path,
) -> None:
    world = World(tmp_path)
    a = world.start()
    world.manager(a).save_now(7, model=weights(7.0))
    a_id = a.run_id
    a.finish("preempted")
    b = world.start(parent=a_id)
    mb = world.manager(b)
    b_id = b.run_id
    mb.close()
    b.abandon()
    point = world.find()
    assert point is not None
    assert point.run_id == b_id  # resume_from this one: its lineage holds the checkpoint
    assert point.checkpoint_run_id == a_id and point.manifest.step == 7
    assert point.needs_recover is True


def test_a_live_run_is_never_picked_or_recovered(tmp_path: Path, events: Events) -> None:
    world = World(tmp_path)
    live = world.start(name="job-a")
    mlive = world.manager(live)  # holds the liveness lease while it runs
    mlive.save_now(4, model=weights(4.0))
    assert world.find() is None
    assert any(e["message"] == "resume_point_skipped_alive" for e in events())
    forced = world.find(force=True)
    assert forced is not None and forced.run_id == live.run_id
    mlive.close()  # the process dies: the lease is released
    after = world.find()
    assert after is not None and after.run_id == live.run_id and after.needs_recover
    live.finish()


def test_two_concurrent_jobs_do_not_resume_each_other(tmp_path: Path) -> None:
    world = World(tmp_path)
    old = world.start(name="job-b")
    mo = world.manager(old)
    mo.save_now(9, model=weights(9.0))
    old_id = old.run_id
    mo.close()
    old.finish("preempted")
    live = world.start(name="job-a")
    ml = world.manager(live)
    ml.save_now(2, model=weights(2.0))
    # job B restarts while job A is alive: it must find its own run, not A's
    point = world.find(name="job-b")
    assert point is not None and point.run_id == old_id and point.manifest.step == 9
    assert world.find(name="job-c") is None
    ml.close()
    live.finish()


def test_without_a_lease_recent_activity_counts_as_alive(tmp_path: Path) -> None:
    world = World(tmp_path)
    run = world.start()
    mgr = world.manager(run)
    mgr.save_now(1, model=weights(1.0))
    mgr.close()
    lease = world.ckpt / run.run_id / ".alive"
    lease.unlink()  # an older directory layout: nothing but timestamps to go by
    run.abandon()
    assert world.find() is None  # modified a moment ago
    old = time.time() - 3600
    for path in (world.ckpt / run.run_id).rglob("*"):
        os.utime(path, (old, old))
    os.utime(world.ckpt / run.run_id, (old, old))
    metrics = world.store / "p" / "metrics" / run.run_id
    for path in metrics.rglob("*"):
        os.utime(path, (old, old))
    point = world.find(alive_within_s=600)
    assert point is not None and point.needs_recover


def test_the_lease_is_held_across_processes_and_dies_with_the_process(tmp_path: Path) -> None:
    world = World(tmp_path)
    code = (
        "import sys, time\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from ckpt_support import run_meta\n"
        "from cvpilot import runs\n"
        "from cvpilot.ckpt.manager import CheckpointManager\n"
        "meta, repro = run_meta('p')\n"
        "run = runs.start_run(meta, repro, store_dir=Path(sys.argv[1]))\n"
        "m = CheckpointManager(run, dir=Path(sys.argv[2]), every_n_steps=1,\n"
        "                      install_signal_handlers=False)\n"
        "m.save_now(1, model={'w': np.ones(2)})\n"
        "print('READY', flush=True)\n"
        "time.sleep(60)\n"
    )
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-c", code, str(world.store), str(world.ckpt)],
        stdout=subprocess.PIPE, text=True, env=env,
    )  # fmt: skip
    assert proc.stdout is not None
    try:
        for line in proc.stdout:  # the library logs to stdout in dev mode: wait for the marker
            if line.strip() == "READY":
                break
        else:
            raise AssertionError("the child never became ready")
        assert world.find() is None  # alive in another process
    finally:
        proc.kill()
        proc.wait()
    point = world.find()  # killed: the kernel released the lease
    assert point is not None and point.needs_recover and point.manifest.step == 1


def test_the_crashed_run_of_the_tutorial_flow_status_is_still_resumable(tmp_path: Path) -> None:
    assert "resumed" in resume.RESUMABLE
