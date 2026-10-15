"""Nothing left is not the same as nothing was ever saved: when a floor above zero is known (run
store, tag-checked pointer, ``min_step``, a damaged newer manifest), finding no valid checkpoint
must raise or say so, never look like a fresh start. Every entry point answers the same way."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

import cvpilot
from cvpilot import runs
from cvpilot.ckpt import resume
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
KEY = "34" * 32
URI = "s3://bucket-name/floor"


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(2, value, dtype=np.float32)}


class World:
    def __init__(
        self, tmp: Path, monkeypatch: pytest.MonkeyPatch, *, keyed: bool, policy: str,
        remote: FakeRemote | None = None,
    ) -> None:  # fmt: skip
        if keyed:
            monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
        else:
            monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
        monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp / "store"))
        monkeypatch.setenv("CVPILOT_CHECKPOINT_DIR", str(tmp / "ckpt"))
        monkeypatch.setenv("CVPILOT_PROJECT", "p")
        monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", policy)
        self.tmp, self.policy, self.fake = tmp, policy, remote
        meta, repro = run_meta("p")
        self.run = runs.start_run(meta, repro, store_dir=tmp / "store")
        self.mgr = self.manager(self.run, tmp / "ckpt")
        for step in (1, 2, 3):
            self.mgr.save_now(step, model=weights(float(step)))
        self.mgr.close()
        self.run.finish("preempted")
        parent = runs.get_run_view("p", self.run.run_id, store_dir=tmp / "store").meta
        child_meta = runs.resume_from(parent, run_id=_new_id(), host="h", name="job")
        child_repro = repro.model_copy(update={"run_id": child_meta.run_id})
        self.child = runs.start_run(child_meta, child_repro, store_dir=tmp / "store")
        self.reader = self.manager(self.child, tmp / "ckpt")

    def manager(self, run: runs.RunHandle, directory: Path) -> CheckpointManager:
        return CheckpointManager(
            run, dir=directory, every_n_steps=1, keep_last=20, install_signal_handlers=False,
            require_trust=self.policy,  # type: ignore[arg-type]
            remote=URI if self.fake else None, remote_store=self.fake,
            remote_sleep=lambda s: None,
        )  # fmt: skip

    @property
    def folder(self) -> Path:
        return self.tmp / "ckpt" / self.run.run_id / "last"

    def wipe(self, *steps: int) -> None:
        for step in steps:
            for suffix in (".manifest.json", ".safetensors"):
                os.unlink(self.folder / f"step_{step:09d}{suffix}")


def _new_id() -> str:
    from cvpilot.core.hashing import new_run_id

    return new_run_id()


Damage = Callable[[World], None]


def d_all_deleted_pointer_left(w: World) -> None:
    w.wipe(1, 2, 3)


def d_all_deleted_pointer_gone(w: World) -> None:
    w.wipe(1, 2, 3)
    os.unlink(w.folder.parent / "LATEST")


def d_directory_replaced_by_empty(w: World) -> None:
    shutil.rmtree(w.folder.parent)
    w.folder.parent.mkdir()


def d_directory_removed(w: World) -> None:
    shutil.rmtree(w.folder.parent)


def d_every_data_file_truncated(w: World) -> None:
    for step in (1, 2, 3):
        path = w.folder / f"step_{step:09d}.safetensors"
        path.chmod(0o600)
        path.write_bytes(b"x")


def d_every_data_file_gone(w: World) -> None:
    for step in (1, 2, 3):
        os.unlink(w.folder / f"step_{step:09d}.safetensors")


DAMAGES: dict[str, Damage] = {
    "all deleted, pointer left": d_all_deleted_pointer_left,
    "all deleted, pointer gone": d_all_deleted_pointer_gone,
    "directory replaced by an empty one": d_directory_replaced_by_empty,
    "directory removed": d_directory_removed,
    "every data file truncated": d_every_data_file_truncated,
    "every data file gone": d_every_data_file_gone,
}

Entry = Callable[[World], Any]


def e_restore(w: World) -> Any:
    return w.reader.restore()


def e_restore_latest_valid(w: World) -> Any:
    return w.reader.restore_latest_valid()


def e_api_resume_path(w: World) -> Any:
    w.child.finish("preempted")
    return cvpilot.resume_path("p")


def e_ckpt_resume_path(w: World) -> Any:
    w.child.finish("preempted")
    return resume.resume_path("p", checkpoint_dir=w.tmp / "ckpt", store_dir=w.tmp / "store")


def e_find_resume_point(w: World) -> Any:
    w.child.finish("preempted")
    return resume.find_resume_point(
        "p", checkpoint_dir=w.tmp / "ckpt", store_dir=w.tmp / "store", force=True
    )


ENTRIES: dict[str, Entry] = {
    "restore": e_restore,
    "restore_latest_valid": e_restore_latest_valid,
    "cvpilot.resume_path": e_api_resume_path,
    "ckpt.resume_path": e_ckpt_resume_path,
    "find_resume_point": e_find_resume_point,
}
SETUPS = [("authenticated", True), ("anchored", False)]


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("damage", DAMAGES)
@pytest.mark.parametrize(("policy", "keyed"), SETUPS)
def test_no_valid_checkpoint_with_a_known_floor_never_looks_like_a_fresh_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str, keyed: bool, damage: str,
    entry: str,
) -> None:  # fmt: skip
    world = World(tmp_path, monkeypatch, keyed=keyed, policy=policy)
    DAMAGES[damage](world)
    with pytest.raises(CheckpointUntrustedError) as info:
        ENTRIES[entry](world)
    assert info.value.reason in {"rolled_back", "checkpoint_missing"}
    assert "3" in str(info.value)  # the floor is in the message
    world.child.finish() if world.child.status == "running" else None


@pytest.mark.parametrize("entry", ENTRIES)
@pytest.mark.parametrize("damage", DAMAGES)
def test_under_any_the_empty_result_is_loud_with_the_floor_and_a_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events, damage: str, entry: str
) -> None:
    world = World(tmp_path, monkeypatch, keyed=False, policy="any")
    DAMAGES[damage](world)
    try:
        result = ENTRIES[entry](world)
    except CheckpointUntrustedError:
        pass  # raising is at least as loud
    else:
        assert result is None
        lost = [e for e in events() if e["message"] == "checkpoint_progress_lost"]
        assert lost, "a bare None looks like a fresh start"
        assert lost[-1]["floor"] == 3 and lost[-1]["reason"] == "rolled_back"
        assert lost[-1]["level"] == "WARNING"


def test_a_genuine_fresh_start_is_still_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    assert mgr.restore() is None and mgr.restore_latest_valid() is None
    run.finish()


def test_accepting_zero_is_the_explicit_way_to_start_over_and_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    world = World(tmp_path, monkeypatch, keyed=True, policy="authenticated")
    d_all_deleted_pointer_left(world)
    assert world.reader.restore(accept_rollback_to=0) is None
    accepted = [e for e in events() if e["message"] == "checkpoint_rollback_accepted"]
    assert accepted and accepted[0]["floor"] == 3 and accepted[0]["accepted_to"] == 0
    world.child.finish()


def test_the_floor_moves_forward_only_with_verified_saves_after_an_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False
    )
    for step in (1, 2, 3):
        mgr.save_now(step, model=weights(float(step)))
    for step in (1, 2, 3):
        for suffix in (".manifest.json", ".safetensors"):
            os.unlink(tmp_path / "ckpt" / run.run_id / "last" / f"step_{step:09d}{suffix}")
    assert mgr.restore(accept_rollback_to=0) is None  # starting over, on purpose
    from cvpilot.core.errors import CheckpointChainError, CheckpointStepError

    with pytest.raises(CheckpointStepError):  # starting over is allowed, going under step 3 is not
        mgr.save_now(1, model=weights(1.0))
    with pytest.raises(CheckpointChainError):  # and nothing trusted is left to chain a save to
        mgr.save_now(4, model=weights(4.0))
    mgr.restart_chain = True  # the explicit, recorded way to begin a new chain
    mgr.save_now(4, model=weights(4.0))  # a verified save past the floor moves it forward
    caught_up = mgr.restore()
    assert caught_up is not None and caught_up.manifest.step == 4
    run.finish()


def test_a_pointer_whose_newest_data_is_gone_walks_back_to_the_older_valid_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    world = World(tmp_path, monkeypatch, keyed=True, policy="authenticated")
    os.unlink(world.folder / "step_000000003.safetensors")
    restored = world.reader.restore()
    assert restored is not None and restored.manifest.step == 2  # damage, not rollback
    assert any(e["message"] == "checkpoint_skipped_corrupt" for e in events())
    world.child.finish()


def test_a_store_that_cannot_answer_fails_closed_even_when_nothing_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = World(tmp_path, monkeypatch, keyed=False, policy="anchored")
    d_directory_removed(world)
    (tmp_path / "store" / "p" / "runs.db").write_bytes(b"not a database") if (
        tmp_path / "store" / "p" / "runs.db"
    ).exists() else None
    with pytest.raises(CheckpointUntrustedError):
        world.reader.restore()
    world.child.finish()


def test_the_remote_still_holding_newer_steps_restores_from_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    world = World(tmp_path, monkeypatch, keyed=True, policy="authenticated", remote=fake)
    d_directory_replaced_by_empty(world)
    restored = world.reader.restore()
    assert restored is not None and restored.manifest.step == 3
    world.child.finish()


def test_a_remote_that_cannot_be_reached_does_not_make_an_emptied_directory_a_fresh_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    world = World(tmp_path, monkeypatch, keyed=True, policy="authenticated", remote=fake)
    d_directory_replaced_by_empty(world)
    fake.objects.clear()  # the remote lost everything too: only the store remembers
    with pytest.raises(CheckpointUntrustedError):
        world.reader.restore()
    world.child.finish()
