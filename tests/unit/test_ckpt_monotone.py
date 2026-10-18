"""Trust and rollback settings only ever tighten: a call, a flag or a file in the checkpoint
directory cannot weaken what the constructor, the environment or the key already demand."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta
from conftest import CliResult

from cvpilot import runs
from cvpilot.ckpt import auth, resume
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
Invoke = Callable[..., CliResult]
KEY = "ab" * 32


def untagged(tmp: Path, monkeypatch: pytest.MonkeyPatch, steps: int = 3) -> str:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    writer = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=10, install_signal_handlers=False
    )
    for step in range(1, steps + 1):
        writer.save_now(step, model={"w": np.full(2, float(step))})
    writer.close()
    run.finish("preempted")
    return run.run_id


def reader(tmp: Path, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta("p")
    child = runs.start_run(meta, repro, store_dir=tmp / "store2")
    return child, CheckpointManager(
        child, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


# -- the resolution rule ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("explicit", "configured", "keyed", "expected"),
    [
        (None, None, False, "any"),
        (None, None, True, "authenticated"),
        ("any", None, True, "authenticated"),
        (None, "any", True, "authenticated"),
        ("any", "any", True, "authenticated"),
        ("anchored", None, False, "anchored"),
        ("any", "anchored", False, "anchored"),
        ("authenticated", "any", False, "authenticated"),
        ("authenticated", None, True, "authenticated"),
    ],
)
def test_the_effective_policy_is_the_strictest_of_every_source(
    explicit: str | None, configured: str | None, keyed: bool, expected: str
) -> None:
    key = b"k" * 32 if keyed else None
    assert auth.resolve_policy(explicit, configured, key) == expected


def test_the_opt_out_restores_plain_precedence() -> None:
    key = b"k" * 32
    assert auth.resolve_policy("any", None, key, allow_weaker=True) == "any"
    assert auth.resolve_policy(None, "anchored", key, allow_weaker=True) == "anchored"
    assert auth.resolve_policy(None, None, key, allow_weaker=True) == "authenticated"


def test_an_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError, match="require_trust"):
        auth.resolve_policy("bogus", None, None)


# -- the manager -----------------------------------------------------------------------------


def test_a_call_cannot_weaken_the_manager_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path)
    for call in (
        lambda: mgr.restore(run_ids=[run_id], require_trust="any"),
        lambda: mgr.restore(run_ids=[run_id], require_trust="anchored"),
        lambda: mgr.restore(run_ids=[run_id], require_trust="any", accept_rollback_to=0),
    ):
        with pytest.raises(CheckpointUntrustedError):
            call()
    ignored = [e for e in events() if e["message"] == "checkpoint_trust_downgrade_ignored"]
    assert len(ignored) == 3 and ignored[0]["requested"] == "any"
    assert ignored[0]["effective"] == "authenticated" and ignored[0]["level"] == "WARN"
    child.finish()


def test_a_call_can_strengthen_the_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    child, mgr = reader(tmp_path)
    assert mgr.require_trust == "any"
    assert mgr.restore(run_ids=[run_id]) is not None
    with pytest.raises(CheckpointUntrustedError):  # the child store knows nothing of the run
        mgr.restore(run_ids=[run_id], require_trust="anchored")
    child.finish()
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, keyed = reader(tmp_path, require_trust="any", allow_weaker_trust=True)
    assert keyed.restore(run_ids=[run_id]) is not None
    with pytest.raises(CheckpointUntrustedError):
        keyed.restore(run_ids=[run_id], require_trust="authenticated")
    child.finish()


def test_the_constructor_argument_cannot_undercut_the_key_without_the_opt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path, require_trust="any")
    assert mgr.require_trust == "authenticated"
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(run_ids=[run_id])
    child.finish()


def test_the_opt_out_is_explicit_logged_once_and_shown_in_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path, require_trust="any", allow_weaker_trust=True)
    restored = mgr.restore(run_ids=[run_id])
    assert restored is not None and restored.policy == "any" and restored.weaker_allowed
    mgr.restore(run_ids=[run_id])
    allowed = [e for e in events() if e["message"] == "checkpoint_weaker_trust_allowed"]
    assert len(allowed) == 1 and allowed[0]["level"] == "WARN"
    child.finish()


def test_without_the_opt_out_the_result_shows_the_strict_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    child, mgr = reader(tmp_path)
    restored = mgr.restore(run_ids=[run_id])
    assert restored is not None and restored.policy == "any" and not restored.weaker_allowed
    child.finish()


def test_the_opt_out_cannot_come_from_the_environment_or_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    untagged(tmp_path, monkeypatch)
    for name in ("CVPILOT_CKPT_ALLOW_WEAKER_TRUST", "CVPILOT_ALLOW_WEAKER_TRUST"):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path)
    assert not mgr.allow_weaker_trust and mgr.require_trust == "authenticated"
    with pytest.raises(TypeError):
        mgr.restore(allow_weaker_trust=True)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="allow_weaker_trust"):
        reader(tmp_path, allow_weaker_trust="yes")
    child.finish()


def test_dir_for_clamps_a_weaker_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    child, mgr = reader(tmp_path)
    assert mgr._dir_for(run_id, "any").require_trust == "authenticated"
    assert mgr._dir_for(run_id, None).require_trust == "authenticated"
    child.finish()


def test_policy_is_not_read_from_the_checkpoint_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    folder = tmp_path / "ckpt" / run_id
    for name in ("policy.json", "config.json", "cvpilot.json", ".policy", "require_trust"):
        (folder / name).write_text(json.dumps({"require_trust": "any", "min_step": 0}))
    manifest = next((folder / "last").glob("*.manifest.json"))
    planted = json.loads(manifest.read_text())
    planted["require_trust"] = "any"
    manifest.chmod(0o600)
    manifest.write_text(json.dumps(planted))
    child, mgr = reader(tmp_path)
    assert mgr.require_trust == "authenticated"
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(run_ids=[run_id])
    child.finish()


# -- the rollback floor ----------------------------------------------------------------------


def test_the_operator_floor_cannot_be_lowered_by_a_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "9")
    child, mgr = reader(tmp_path)
    assert mgr.min_step == 9
    for accept in (None, 0, 1, 3, 8):
        with pytest.raises(CheckpointUntrustedError) as caught:
            mgr.restore(run_ids=[run_id], accept_rollback_to=accept)
        assert caught.value.reason == "rolled_back"
    child.finish()


def test_the_constructor_floor_cannot_lower_the_operator_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "9")
    child, mgr = reader(tmp_path, min_step=1)
    assert mgr.min_step == 9
    child.finish()
    child, mgr = reader(tmp_path, min_step=12)
    assert mgr.min_step == 12  # tightening is fine
    child.finish()


def test_only_the_opt_out_lets_the_constructor_replace_the_operator_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "9")
    child, mgr = reader(tmp_path, min_step=1, allow_weaker_trust=True)
    assert mgr.min_step == 1
    restored = mgr.restore(run_ids=[run_id])
    assert restored is not None and restored.manifest.step == 3
    assert [e for e in events() if e["message"] == "checkpoint_weaker_trust_allowed"]
    child.finish()


def test_accepting_nothing_does_not_get_around_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "5")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(accept_rollback_to=0)  # the odd special case is gone
    run.finish()


def test_the_floor_in_the_function_api_binds_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "9")
    kw: dict[str, Any] = {"checkpoint_dir": tmp_path / "ckpt", "store_dir": tmp_path / "store"}
    for accept in (None, 0, 1):
        with pytest.raises(CheckpointUntrustedError):
            resume.find_resume_point("p", accept_rollback_to=accept, min_step=0, force=True, **kw)


# -- the function API and the command line ------------------------------------------------------


def test_resume_path_ignores_a_weaker_request_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    kw: dict[str, Any] = {"checkpoint_dir": tmp_path / "ckpt", "store_dir": tmp_path / "store"}
    with pytest.raises(CheckpointUntrustedError):
        resume.resume_path("p", require_trust="any", **kw)
    with pytest.raises(CheckpointUntrustedError):
        resume.find_resume_point("p", require_trust="anchored", force=True, **kw)
    assert [e for e in events() if e["message"] == "checkpoint_trust_downgrade_ignored"]


def test_the_cli_flag_cannot_weaken_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoke: Invoke
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", "authenticated")
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    args = ["ckpt", "verify", str(tmp_path / "ckpt"), "--run", run_id, "--project", "p", "--json"]
    weak = invoke(*args, "--require-trust", "any")
    assert weak.code == 1
    rows = json.loads(weak.out)
    assert rows and {r["policy"] for r in rows} == {"authenticated"}
    assert all(r["status"] != "valid" for r in rows)
    assert any(r["message"] == "checkpoint_trust_downgrade_ignored" for r in weak.events)
    strong = invoke(*args)
    assert strong.code == 1 and {r["policy"] for r in json.loads(strong.out)} == {"authenticated"}


def test_the_cli_flag_can_strengthen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoke: Invoke
) -> None:
    run_id = untagged(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", "anchored")
    monkeypatch.setenv("CVPILOT_STORE_DIR", str(tmp_path / "store"))
    args = ["ckpt", "verify", str(tmp_path / "ckpt"), "--run", run_id, "--project", "p", "--json"]
    res = invoke(*args, "--require-trust", "authenticated")
    assert res.code == 1 and {r["policy"] for r in json.loads(res.out)} == {"authenticated"}
