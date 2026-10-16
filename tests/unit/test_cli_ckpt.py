"""``cvpilot ckpt list|verify|prune``: exit codes, output, trust options."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from ckpt_support import run_meta
from conftest import CliResult

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager

Invoke = Callable[..., CliResult]


def seed(store_root: Path, tmp_path: Path, steps: tuple[int, ...] = (1, 2, 3)) -> str:
    meta, repro = run_meta("proj")
    run = runs.start_run(meta, repro, store_dir=store_root)
    manager = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, keep_last=10, install_signal_handlers=False
    )
    for step in steps:
        manager.save_now(step, model={"w": np.full(4, float(step), dtype=np.float32)})
    run_id = run.run_id
    run.finish("preempted")
    return run_id


@pytest.fixture
def seeded(store_root: Path, tmp_path: Path) -> str:
    return seed(store_root, tmp_path)


def damage(tmp_path: Path, run_id: str, step: int) -> None:
    path = tmp_path / "ckpt" / run_id / "last" / f"step_{step:09d}.safetensors"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes()[:20])


def test_list_shows_every_checkpoint_without_hashing(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    res = invoke("ckpt", "list", str(tmp_path / "ckpt"))
    assert res.code == 0, res.err
    lines = [x for x in res.out.splitlines() if seeded in x]
    assert len(lines) == 3 and "last" in lines[0] and "000000003" in lines[0]
    as_json = json.loads(invoke("ckpt", "list", str(tmp_path / "ckpt"), "--json").out)
    assert [r["step"] for r in as_json] == [3, 2, 1]
    assert {r["run_id"] for r in as_json} == {seeded} and as_json[0]["status"] == "valid"
    assert as_json[0]["size_bytes"] > 0 and len(as_json[0]["sha256"]) == 64


def test_list_can_be_limited_to_a_run_and_a_kind(
    invoke: Invoke, seeded: str, tmp_path: Path, store_root: Path
) -> None:
    other = seed(store_root, tmp_path, (1,))
    rows = json.loads(invoke("ckpt", "list", str(tmp_path / "ckpt"), "--run", other, "--json").out)
    assert {r["run_id"] for r in rows} == {other}
    none = json.loads(
        invoke("ckpt", "list", str(tmp_path / "ckpt"), "--kind", "best", "--json").out
    )
    assert none == []


def test_a_path_that_is_not_a_directory_is_a_usage_error(invoke: Invoke, tmp_path: Path) -> None:
    (tmp_path / "file").write_text("x")
    for target in (tmp_path / "missing", tmp_path / "file"):
        res = invoke("ckpt", "list", str(target))
        assert res.code == 2 and "not a checkpoint directory" in res.human
    assert invoke("ckpt", "verify", str(tmp_path / "missing")).code == 2


def test_run_ids_are_validated(invoke: Invoke, tmp_path: Path) -> None:
    (tmp_path / "ckpt").mkdir()
    res = invoke("ckpt", "list", str(tmp_path / "ckpt"), "--run", "../../etc")
    assert res.code == 2


def test_verify_passes_a_healthy_directory_and_logs_each_checkpoint(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj")
    assert res.code == 0, res.err
    assert len(res.event("checkpoint_verified")) == 3
    assert res.out.count("valid") >= 3 and "anchored" in res.out


def test_verify_exits_1_and_lists_the_corrupt_file_on_stdout(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    damage(tmp_path, seeded, 3)
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"))
    assert res.code == 1
    assert "corrupt" in res.out and "size_mismatch" in res.out
    failed = res.event("checkpoint_verify_failed")
    assert len(failed) == 1 and failed[0]["step"] == 3 and failed[0]["reason"] == "size_mismatch"


def test_verify_of_an_empty_directory_has_nothing_valid_and_exits_1(
    invoke: Invoke, tmp_path: Path
) -> None:
    (tmp_path / "ckpt").mkdir()
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"))
    assert res.code == 1 and "no checkpoints" in res.human


def test_the_store_anchor_catches_a_manifest_swapped_in_the_directory(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    manifest = tmp_path / "ckpt" / seeded / "last" / "step_000000003.manifest.json"
    manifest.chmod(0o644)
    manifest.write_bytes(manifest.read_bytes() + b" ")  # same record, different bytes
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj")
    assert res.code == 1 and "pointer_mismatch" in res.out or "store_mismatch" in res.out


def test_require_trust_authenticated_needs_a_key(
    invoke: Invoke, seeded: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--require-trust", "authenticated")
    assert res.code == 2 and "CVPILOT_CKPT_KEY" in res.human


def test_require_trust_authenticated_rejects_untagged_checkpoints(
    invoke: Invoke, seeded: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "k" * 40)
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--require-trust", "authenticated")
    assert res.code == 1 and "auth_missing" in res.out
    assert len(res.event("checkpoint_untrusted")) == 3


def test_the_trust_policy_default_comes_from_the_environment(
    invoke: Invoke, seeded: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "k" * 40)
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", "authenticated")
    assert invoke("ckpt", "verify", str(tmp_path / "ckpt")).code == 1


def test_prune_keeps_the_newest_and_never_the_last_valid_one(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    res = invoke("ckpt", "prune", str(tmp_path / "ckpt"), "--run", seeded, "--keep-last", "1")
    assert res.code == 0, res.err
    assert res.out.count("deleted") == 2
    rows = json.loads(invoke("ckpt", "list", str(tmp_path / "ckpt"), "--json").out)
    assert [r["step"] for r in rows] == [3]


def test_prune_dry_run_changes_nothing(invoke: Invoke, seeded: str, tmp_path: Path) -> None:
    res = invoke(
        "ckpt", "prune", str(tmp_path / "ckpt"), "--run", seeded, "--keep-last", "1", "--dry-run"
    )
    assert res.code == 0 and "would delete" in res.out
    rows = json.loads(invoke("ckpt", "list", str(tmp_path / "ckpt"), "--json").out)
    assert len(rows) == 3


def test_prune_refuses_to_keep_nothing_and_needs_a_run(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    assert (
        invoke("ckpt", "prune", str(tmp_path / "ckpt"), "--run", seeded, "--keep-last", "0").code
        == 2
    )
    assert invoke("ckpt", "prune", str(tmp_path / "ckpt")).code == 2


def test_prune_with_a_corrupt_newest_keeps_the_last_good_one(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    damage(tmp_path, seeded, 3)
    invoke("ckpt", "prune", str(tmp_path / "ckpt"), "--run", seeded, "--keep-last", "1")
    rows = {
        r["step"]: r["status"]
        for r in json.loads(invoke("ckpt", "list", str(tmp_path / "ckpt"), "--json").out)
    }
    assert rows[2] == "valid"


def test_help_documents_the_exit_codes(invoke: Invoke) -> None:
    for sub in ("list", "verify", "prune"):
        res = invoke("ckpt", sub, "--help")
        assert res.code == 0 and "Exit" in res.out


def test_a_symlink_named_like_a_run_is_a_corrupt_row_and_the_rest_is_still_listed(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    import os

    from cvpilot.core.hashing import new_run_id

    hostile = new_run_id()
    os.symlink("/etc", tmp_path / "ckpt" / hostile)
    for command in ("list", "verify"):
        res = invoke("ckpt", command, str(tmp_path / "ckpt"), "--json")
        rows = json.loads(res.out)
        bad = [r for r in rows if r["run_id"] == hostile]
        assert bad and bad[0]["status"] == "corrupt" and bad[0]["reason"] == "not_a_directory"
        good = [r for r in rows if r["run_id"] == seeded]
        assert len(good) == 3 and all(r["status"] == "valid" for r in good)
        assert res.code == (1 if command == "verify" else 0)


def test_a_file_named_like_a_run_is_reported_the_same_way(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    from cvpilot.core.hashing import new_run_id

    odd = new_run_id()
    (tmp_path / "ckpt" / odd).write_text("not a directory")
    rows = json.loads(invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--json").out)
    assert {r["reason"] for r in rows if r["run_id"] == odd} == {"not_a_directory"}  # last and best


def test_the_root_of_the_file_system_is_a_usage_error(invoke: Invoke) -> None:
    for command in ("list", "verify"):
        res = invoke("ckpt", command, "/")
        assert res.code == 2 and "error:" in res.err
    assert invoke("ckpt", "prune", "/", "--run", "01ARZ3NDEKTSV4RRFFQ69G5FAV").code == 2


def test_an_unreadable_run_directory_is_a_row_not_a_traceback(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    import os

    path = tmp_path / "ckpt" / seeded
    path.chmod(0)
    try:
        if os.access(path, os.R_OK):  # running as root: permissions do not apply
            pytest.skip("permissions are not enforced for this user")
        res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--json")
        assert res.code == 1 and "Traceback" not in res.err
    finally:
        path.chmod(0o755)


def _ids(res: CliResult) -> list[tuple[int, str, str | None]]:
    return [(r["step"], r["status"], r["reason"]) for r in json.loads(res.out)]


def test_verify_can_demand_a_minimum_step(invoke: Invoke, seeded: str, tmp_path: Path) -> None:
    base = ("ckpt", "verify", str(tmp_path / "ckpt"), "--json")
    assert invoke(*base, "--expect-min-step", "3").code == 0
    res = invoke(*base, "--expect-min-step", "5")
    assert res.code == 1
    assert {s for _, st, s in _ids(res) if st == "corrupt"} == {"rolled_back"}


def test_the_operator_floor_cannot_be_accepted_away_on_the_command_line(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    res = invoke(
        "ckpt", "verify", str(tmp_path / "ckpt"), "--json", "--expect-min-step", "5",
        "--accept-rollback-to", "1",
    )  # fmt: skip
    assert res.code == 1


def test_the_environment_floor_applies_to_verify(
    invoke: Invoke, seeded: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "5")
    assert invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--json").code == 1
    monkeypatch.setenv("CVPILOT_CKPT_MIN_STEP", "3")
    assert invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--json").code == 0


def test_a_store_derived_rollback_can_be_accepted_explicitly_and_is_logged(
    invoke: Invoke, seeded: str, tmp_path: Path, store_root: Path
) -> None:
    import os

    folder = tmp_path / "ckpt" / seeded / "last"
    for step in (3, 2):
        for suffix in (".manifest.json", ".safetensors"):
            os.unlink(folder / f"step_{step:09d}{suffix}")
    base = (
        "ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj", "--json",
        "--require-trust", "anchored",
    )  # fmt: skip
    refused = invoke(*base, "--kind", "last")
    assert refused.code == 1 and {s for _, _, s in _ids(refused)} == {"rolled_back"}
    accepted = invoke(*base, "--kind", "last", "--accept-rollback-to", "1")
    assert accepted.code == 0, accepted.err
    assert accepted.event("checkpoint_rollback_accepted")


def test_verify_names_a_rotated_key_and_judges_each_file_only_with_the_acknowledgement(
    invoke: Invoke, store_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "ab12" * 16)
    run_id = seed(store_root, tmp_path, (1, 2))
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "cd34" * 16)
    plain = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj", "--json")
    assert plain.code == 1 and {r["reason"] for r in json.loads(plain.out)} == {"key_changed"}
    accepted = invoke(
        "ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj", "--json",
        "--accept-key-change",
    )  # fmt: skip
    assert accepted.code == 1  # the old tags still do not verify under the new key
    assert {r["reason"] for r in json.loads(accepted.out)} == {"key_changed"}
    assert run_id


def _wipe(tmp_path: Path, run_id: str, *steps: int) -> None:
    for step in steps:
        for suffix in (".manifest.json", ".safetensors"):
            (tmp_path / "ckpt" / run_id / "last" / f"step_{step:09d}{suffix}").unlink()


def test_plain_verify_says_when_the_store_knows_a_newer_step_but_keeps_its_exit_code(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    _wipe(tmp_path, seeded, 3, 2)
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj", "--json")
    assert res.code == 0, res.err  # unchanged: under the default policy this is not an error
    assert [(step, st) for step, st, _ in _ids(res)] == [(1, "valid")]
    assert "step 3" in res.human and "step 1" in res.human and "anchored" in res.human
    strict = invoke(
        "ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj", "--require-trust", "anchored"
    )
    assert strict.code == 1 and "note:" not in strict.human  # the policy reports it itself


def test_plain_verify_prints_no_note_when_nothing_is_missing(
    invoke: Invoke, seeded: str, tmp_path: Path
) -> None:
    res = invoke("ckpt", "verify", str(tmp_path / "ckpt"), "--project", "proj")
    assert res.code == 0 and res.human == ""
