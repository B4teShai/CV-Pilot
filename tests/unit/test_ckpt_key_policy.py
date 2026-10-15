"""A key alone enforces something: authenticated is the default policy once a key is configured."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

import cvpilot
from cvpilot import runs
from cvpilot.ckpt import resume
from cvpilot.ckpt.manager import CheckpointManager, Restored
from cvpilot.core.errors import CheckpointUntrustedError
from cvpilot.settings import load_settings

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
KEY = "cd" * 32


def untagged_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[runs.RunHandle, str]:
    """A preempted run whose checkpoints were written without a key (what a forger produces)."""
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    writer = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    writer.save_now(1, model={"w": np.ones(2)})
    run_id = run.run_id
    run.finish("preempted")
    return run, run_id


def reader(tmp_path: Path, run_id: str, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta("p")
    child = runs.start_run(meta, repro, store_dir=tmp_path / "store2")
    return child, CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


def test_the_settings_default_is_unset_not_any() -> None:
    assert load_settings({}).ckpt_require_trust is None


def test_a_key_makes_authenticated_the_default_and_untagged_files_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run_id = untagged_run(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path, run_id)
    assert mgr.require_trust == "authenticated"
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(run_ids=[run_id])
    child.finish()


def test_an_explicit_any_still_works_and_is_logged_as_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    _, run_id = untagged_run(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path, run_id, require_trust="any")
    restored = mgr.restore(run_ids=[run_id])
    assert restored is not None and restored.trust == "local_chain_only"
    accepted = [e for e in events() if e["message"] == "checkpoint_untrusted_accepted"]
    assert accepted and accepted[0]["level"] == "WARNING"
    assert accepted[0]["trust_level"] == "local_chain_only"
    child.finish()


def test_the_environment_can_say_any_explicitly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run_id = untagged_run(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", "any")
    child, mgr = reader(tmp_path, run_id)
    assert mgr.require_trust == "any"
    child.finish()


def test_without_a_key_the_default_is_any(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, run_id = untagged_run(tmp_path, monkeypatch)
    child, mgr = reader(tmp_path, run_id)
    assert mgr.require_trust == "any"
    child.finish()


def test_restore_latest_valid_takes_a_policy_and_can_return_the_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, run_id = untagged_run(tmp_path, monkeypatch)
    child, mgr = reader(tmp_path, run_id)
    # the child is not a resumed run, so it only sees the directory through run_ids; use restore
    state_only = mgr.restore_latest_valid()
    assert state_only is None  # nothing of the child's own: the type is unchanged
    own = CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    own.save_now(1, model={"w": np.ones(2)})
    plain = own.restore_latest_valid()
    assert isinstance(plain, dict) and "model" in plain
    with_level = own.restore_latest_valid(with_trust=True)
    assert isinstance(with_level, Restored) and with_level.trust in {"anchored", "local_chain_only"}
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    strict = CheckpointManager(
        child, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False,
        require_trust="any",
    )  # fmt: skip
    # a per-call policy overrides the manager's: untagged files fail authenticated
    with pytest.raises(CheckpointUntrustedError):
        strict.restore_latest_valid(require_trust="authenticated")
    assert strict.restore_latest_valid(require_trust="any") is not None
    child.finish()
    del run


def test_resume_path_takes_a_policy_and_can_return_the_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run_id = untagged_run(tmp_path, monkeypatch)
    kw = {"checkpoint_dir": tmp_path / "ckpt", "store_dir": tmp_path / "store"}
    path = resume.resume_path("p", **kw)
    assert isinstance(path, Path)
    point = resume.resume_path("p", with_trust=True, **kw)
    assert isinstance(point, resume.ResumePoint) and point.run_id == run_id
    assert point.trust == "anchored"
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    with pytest.raises(CheckpointUntrustedError):
        resume.resume_path("p", **kw)  # a key is set: authenticated by default
    assert resume.resume_path("p", require_trust="anchored", **kw) == path
    assert resume.resume_path("p", require_trust="any", **kw) == path


def test_the_public_function_exposes_the_same_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, run_id = untagged_run(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    monkeypatch.setenv("CVPILOT_CHECKPOINT_DIR", str(tmp_path / "ckpt"))
    point = cvpilot.resume_path("p", with_trust=True)
    assert point is not None and point.run_id == run_id
    assert isinstance(cvpilot.resume_path("p"), Path)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    with pytest.raises(CheckpointUntrustedError):
        cvpilot.resume_path("p")
    assert cvpilot.resume_path("p", require_trust="any") is not None
