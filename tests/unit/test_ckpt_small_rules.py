"""Small rules found by review: one number validator, strict integer keys, a marker is not proof
of an upload, and a lease stops a recovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import state as state_mod
from cvpilot.ckpt.directory import AdmitContext, RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError, InvalidTransitionError
from cvpilot.settings import load_settings

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
URI = "s3://bucket-name/small"


@pytest.mark.parametrize("bad", [-1, True, 1.5, float("nan"), "3", 10**30])
@pytest.mark.parametrize("field", ["min_step", "accept_rollback_to"])
def test_the_gate_context_refuses_numbers_that_are_not_steps(field: str, bad: Any) -> None:
    with pytest.raises(CheckpointError, match=field):
        AdmitContext(**{field: bad})


def test_the_gate_context_accepts_steps_and_a_positive_byte_cap() -> None:
    AdmitContext(min_step=0, accept_rollback_to=7, max_candidate_bytes=1)
    for bad in (0, -5, True, 1.5):
        with pytest.raises(CheckpointError):
            AdmitContext(max_candidate_bytes=bad)  # type: ignore[arg-type]


def test_the_environment_floor_has_an_upper_bound() -> None:
    assert load_settings({"CVPILOT_CKPT_MIN_STEP": "5"}).ckpt_min_step == 5
    with pytest.raises(ValueError):
        load_settings({"CVPILOT_CKPT_MIN_STEP": str(10**20)})
    with pytest.raises(ValueError):
        load_settings({"CVPILOT_CKPT_MIN_STEP": "-1"})


def test_integer_dict_keys_are_strict_and_unique() -> None:
    tensors: dict[str, Any] = {}
    ok = state_mod.merge_state(tensors, {"d": {"$intdict": {"1": 1, "-2": 2}}})
    assert ok["d"] == {1: 1, -2: 2}
    for keys in ({"1": 1, "01": 2}, {"٣": 1}, {"1_0": 1}, {"+1": 1}, {" 1": 1}, {"": 1}):
        with pytest.raises(ValueError):
            state_mod.merge_state(tensors, {"d": {"$intdict": keys}})


def world(tmp: Path, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager, FakeRemote]:
    fake = FakeRemote()
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=2, remote=URI, remote_store=fake,
        install_signal_handlers=False, remote_sleep=lambda s: None, **kw,
    )  # fmt: skip
    return run, mgr, fake


def test_retention_does_not_trust_a_marker_the_remote_no_longer_backs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    run, mgr, fake = world(tmp_path)
    for step in (1, 2):
        mgr.save_now(step, model={"w": np.full(2, float(step))})
    fake.objects.clear()  # the bucket lost everything; the local markers still say "uploaded"
    fake.fail_writes = 10**6  # and nothing can be uploaded again
    for step in (3, 4, 5):
        mgr.save_now(step, model={"w": np.full(2, float(step))})
    folder = tmp_path / "ckpt" / run.run_id / "last"
    assert (folder / "step_000000002.safetensors").exists()  # nothing was deleted on a stale marker
    assert (folder / "step_000000001.safetensors").exists()
    run.finish()


def test_a_marker_is_still_honoured_when_the_remote_holds_the_exact_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    run, mgr, fake = world(tmp_path)
    for step in (1, 2, 3, 4):
        mgr.save_now(step, model={"w": np.full(2, float(step))})
    left = sorted(p.name for p in (tmp_path / "ckpt" / run.run_id / "last").glob("*.safetensors"))
    assert left == ["step_000000003.safetensors", "step_000000004.safetensors"]
    run.finish()


def test_existing_returns_only_what_the_gate_would_admit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "9d" * 32)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    mgr.save_now(1, model={"w": np.ones(2)})
    path = tmp_path / "ckpt" / run.run_id / "last" / "step_000000001.manifest.json"
    doc = json.loads(path.read_text())
    doc["auth_tag"] = None
    path.chmod(0o600)
    path.write_text(json.dumps(doc) + "\n")
    rc = RunCheckpoints(
        tmp_path / "ckpt", run.run_id, key=bytes.fromhex("9d" * 32), require_trust="authenticated"
    )
    assert rc.existing("last", 1) is None  # untagged under a key: not a manifest to adopt against
    run.finish()


def test_recover_run_refuses_while_the_lease_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    mgr.save_now(1, model={"w": np.ones(2)})
    with pytest.raises(InvalidTransitionError, match="live process"):
        runs.recover_run(
            "p", run.run_id, store_dir=tmp_path / "store", checkpoint_dir=tmp_path / "ckpt"
        )
    mgr.close()  # the lease is released: the run is dead and may be recovered
    runs.recover_run(
        "p", run.run_id, store_dir=tmp_path / "store", checkpoint_dir=tmp_path / "ckpt"
    )
    assert runs.get_run_view("p", run.run_id, store_dir=tmp_path / "store").meta.status == "failed"


def test_a_run_that_cannot_take_a_lease_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Any
) -> None:
    monkeypatch.setattr(RunCheckpoints, "hold_lease", lambda self: None)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    CheckpointManager(run, dir=tmp_path / "ckpt", install_signal_handlers=False)
    assert [e for e in events() if e["message"] == "checkpoint_lease_unavailable"]
    run.finish()


def test_list_and_verify_use_the_managers_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, keep_last=5, install_signal_handlers=False,
    )  # fmt: skip
    manifests = [mgr.save_now(step, model={"w": np.ones(2)}) for step in (1, 2, 3)]
    folder = tmp_path / "ckpt" / run.run_id / "last"
    for step in (3, 2):  # the newest are gone
        for suffix in (".manifest.json", ".safetensors"):
            (folder / f"step_{step:09d}{suffix}").unlink()
    assert mgr.verify(manifests[0]) is True  # no floor: step 1 is just the newest left
    mgr.min_step = 3  # the operator raises the floor after the fact
    status = {e.step: e.status for e in mgr.list() if e.kind == "last"}
    assert status[1] != "valid"  # the manager's floor reaches list ...
    assert mgr.verify(manifests[0]) is False  # ... and verify
    run.finish()
