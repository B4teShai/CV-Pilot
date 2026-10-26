"""Ways to get around the rollback floor, each of which must fail."""

from __future__ import annotations

import json
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import blob, run_meta

from cvpilot import runs
from cvpilot.ckpt import auth
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError, StoreError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
KEY = "12" * 32


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(2, value, dtype=np.float32)}


def make(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, *, policy: str = "authenticated", steps: int = 3,
    first: int = 1, **kw: Any,
) -> tuple[runs.RunHandle, CheckpointManager]:  # fmt: skip
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False,
        require_trust=policy, **kw,  # type: ignore[arg-type]
    )  # fmt: skip
    for step in range(first, first + steps):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def remove(tmp: Path, run: runs.RunHandle, *steps: int) -> None:
    for step in steps:
        for suffix in (".manifest.json", ".safetensors"):
            os.unlink(tmp / "ckpt" / run.run_id / "last" / f"step_{step:09d}{suffix}")


def test_an_untagged_pointer_does_not_lower_or_raise_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    remove(tmp_path, run, 3, 2)
    pointer = tmp_path / "ckpt" / run.run_id / "LATEST"
    forged = json.loads(pointer.read_text())
    forged.update(step=1, auth_tag=None)  # a stale, untagged pointer
    pointer.unlink()
    pointer.write_text(json.dumps(forged))
    with pytest.raises(CheckpointUntrustedError) as info:  # the store still knows step 3
        mgr.restore()
    assert info.value.reason == "rolled_back"
    run.finish()


def test_without_a_store_only_a_tag_checked_pointer_sets_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    run.finish("preempted")
    direct = RunCheckpoints(
        tmp_path / "ckpt", run.run_id, key=auth.load_key(), require_trust="authenticated"
    )
    assert direct.freshness_floor("last") == 3  # the genuine, tagged pointer
    pointer = tmp_path / "ckpt" / run.run_id / "LATEST"
    forged = json.loads(pointer.read_text())
    forged.update(step=99, auth_tag="0" * 64)
    pointer.unlink()
    pointer.write_text(json.dumps(forged))
    assert direct.freshness_floor("last") == 0  # a pointer that fails its tag proves nothing
    assert [v.manifest.step for v in direct.iter_valid("last")] == [3, 2, 1]


def test_a_store_row_of_another_run_does_not_move_this_runs_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    other_meta, other_repro = run_meta()
    other = runs.start_run(other_meta, other_repro, store_dir=tmp_path / "store")
    other_mgr = CheckpointManager(
        other, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False,
        require_trust="authenticated",
    )  # fmt: skip
    other_mgr.save_now(50, model=weights(1.0))
    assert mgr._own.freshness_floor("last") == 3
    assert other_mgr._own.freshness_floor("last") == 50
    run.finish()
    other.finish()


def test_step_zero_and_the_first_save_are_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch, steps=1, first=0)
    assert mgr._own.freshness_floor("last") == 0
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 0
    remove(tmp_path, run, 0)  # the only checkpoint ever saved is removed
    with pytest.raises(CheckpointUntrustedError) as info:
        mgr.restore()
    assert info.value.reason == "rolled_back"
    run.finish()


def test_a_rejected_forgery_does_not_raise_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    scratch = tmp_path / "forger"
    forger = RunCheckpoints(scratch, run.run_id, create=True)
    forged = forger.save(
        kind="last", step=999, data=blob(9), format="safetensors", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    folder = tmp_path / "ckpt" / run.run_id / "last"
    for name in (f"step_{999:09d}.manifest.json", forged.path.split("/", 1)[1]):
        shutil.copy(scratch / run.run_id / "last" / name, folder / name)
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 3  # not "rolled back from 999"
    assert mgr._own.freshness_floor("last") == 3
    run.finish()


@pytest.mark.parametrize("policy", ["anchored", "authenticated"])
def test_a_store_that_cannot_answer_fails_closed_with_the_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    run, mgr = make(tmp_path, monkeypatch, policy=policy)

    def broken(*args: object) -> None:
        raise StoreError("the run store cannot be read")

    # every lookup raises (a finished run still answers through a read-only connection, so the
    # failure is injected rather than produced by finishing the run)
    for name in ("checkpoint_digest", "checkpoint_max_step", "checkpoint_auth_evidence"):
        monkeypatch.setattr(run, name, broken)
    with pytest.raises(CheckpointUntrustedError) as info:
        mgr.restore()
    assert info.value.reason == "anchor_error"
    monkeypatch.undo()
    run.finish()


def test_a_lower_external_mark_never_lowers_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    remove(tmp_path, run, 3, 2)
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "0")
    low = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False,
        require_trust="authenticated", min_step=1,
    )  # fmt: skip
    with pytest.raises(CheckpointUntrustedError):
        low.restore()  # step 1 >= the mark 1, but the store still says 3
    run.finish()


def test_there_is_no_environment_variable_that_accepts_a_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    remove(tmp_path, run, 3, 2)
    for name in (
        "CVPILOT_CKPT_ACCEPT_ROLLBACK", "CVPILOT_CKPT_ACCEPT_ROLLBACK_TO",
        "CVPILOT_CKPT_REQUIRE_TRUST_ANY", "CVPILOT_CKPT_MIN_STEP",
    ):  # fmt: skip
        monkeypatch.setenv(name, "0")
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore()
    assert not [e for e in events() if e["message"] == "checkpoint_rollback_accepted"]
    explicit = mgr.restore(accept_rollback_to=1)  # only an argument does it, and it is logged
    assert explicit is not None
    assert [e for e in events() if e["message"] == "checkpoint_rollback_accepted"]
    run.finish()


def test_the_best_kind_has_its_own_floor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=100, monitor="m", keep_best=20,
        install_signal_handlers=False, require_trust="authenticated",
    )  # fmt: skip
    for step, value in ((1, 0.1), (2, 0.2), (3, 0.3)):
        mgr.maybe_save(step, model=weights(1.0), metrics={"m": value})
    assert mgr._own.freshness_floor("best") == 3
    assert mgr._own.freshness_floor("last") == 0
    folder = tmp_path / "ckpt" / run.run_id / "best"
    for step in (3, 2):
        for suffix in (".manifest.json", ".safetensors"):
            os.unlink(folder / f"step_{step:09d}{suffix}")
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore("best")
    assert mgr.restore("last") is None  # nothing was ever saved as last: not a rollback
    run.finish()


def test_the_floor_of_an_ancestor_is_computed_on_the_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = make(tmp_path, monkeypatch)
    parent_id = run.run_id
    run.finish("preempted")
    parent = runs.get_run_view("p", parent_id, store_dir=tmp_path / "store").meta
    child_meta = runs.resume_from(
        parent,
        run_id=__import__("cvpilot.core.hashing", fromlist=["x"]).new_run_id(),
        host="h",
        name=None,
    )
    _, repro = run_meta()
    child = runs.start_run(
        child_meta,
        repro.model_copy(update={"run_id": child_meta.run_id}),
        store_dir=tmp_path / "store",
    )
    remove(tmp_path, run, 3, 2)  # the ancestor's directory is rolled back
    cm = CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False,
        require_trust="authenticated",
    )  # fmt: skip
    with pytest.raises(CheckpointUntrustedError) as info:
        cm.restore()
    assert info.value.reason == "rolled_back"
    child.finish()
