"""What a save counts as 'the newest checkpoint' and 'the previous one' comes only from manifests
the gate admits: a planted manifest cannot block every later save or become its predecessor."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import blob, run_meta

from cvpilot import runs
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
KEY = "9a" * 32


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def start(tmp: Path, monkeypatch: pytest.MonkeyPatch, *, keyed: bool, policy: str) -> Any:
    if keyed:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False,
        require_trust=policy,  # type: ignore[arg-type]
    )  # fmt: skip
    for step in (1, 2):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def plant(tmp: Path, run_id: str, step: int) -> None:
    """A self-consistent, untagged checkpoint at ``step`` written by someone without the key."""
    scratch = tmp / "forger"
    forger = RunCheckpoints(scratch, run_id, create=True)
    manifest = forger.save(
        kind="last", step=step, data=blob(step), format="safetensors", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    target = tmp / "ckpt" / run_id / "last"
    for name in (manifest.path.split("/", 1)[1], f"step_{step:09d}.manifest.json"):
        dest = target / name
        if dest.exists():
            dest.chmod(0o600)
        shutil.copy(scratch / run_id / "last" / name, dest)


@pytest.mark.parametrize(
    ("policy", "keyed"), [("authenticated", True), ("anchored", False), ("any", False)]
)
def test_a_planted_high_step_is_quarantined_and_cleared_by_the_operator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str, keyed: bool
) -> None:
    from cvpilot.core.errors import CheckpointStepError

    run, mgr = start(tmp_path, monkeypatch, keyed=keyed, policy=policy)
    plant(tmp_path, run.run_id, 999_999)
    with pytest.raises(CheckpointStepError, match="quarantine"):
        mgr.save_now(3, model=weights(3.0))
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id)
    assert rc.clear_quarantine() == 1
    assert mgr.save_now(3, model=weights(3.0)).step == 3
    run.finish()


@pytest.mark.parametrize(
    ("policy", "keyed"), [("authenticated", True), ("anchored", False), ("any", False)]
)
def test_a_planted_manifest_never_becomes_the_predecessor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str, keyed: bool
) -> None:
    run, mgr = start(tmp_path, monkeypatch, keyed=keyed, policy=policy)
    genuine = json.loads(
        (tmp_path / "ckpt" / run.run_id / "last" / "step_000000002.manifest.json").read_text()
    )
    plant(tmp_path, run.run_id, 50)
    saved = mgr.save_now(60, model=weights(3.0))  # above the planted step, so it is allowed
    assert saved.prev_chain_sha256 == genuine["chain_sha256"]  # chained to step 2, not to step 50
    assert saved.prev_step == 2
    run.finish()


def test_a_genuine_save_still_chains_to_the_newest_genuine_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = start(tmp_path, monkeypatch, keyed=True, policy="authenticated")
    third = mgr.save_now(3, model=weights(3.0))
    second = json.loads(
        (tmp_path / "ckpt" / run.run_id / "last" / "step_000000002.manifest.json").read_text()
    )
    assert third.prev_chain_sha256 == second["chain_sha256"] and third.prev_step == 2
    run.finish()


def test_the_same_step_and_content_is_still_idempotent_and_other_content_still_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from cvpilot.core.errors import CheckpointExistsError

    run, mgr = start(tmp_path, monkeypatch, keyed=True, policy="authenticated")
    again = mgr.save_now(2, model=weights(2.0))
    assert again.step == 2
    with pytest.raises(CheckpointExistsError):
        mgr.save_now(2, model=weights(9.0))
    run.finish()
