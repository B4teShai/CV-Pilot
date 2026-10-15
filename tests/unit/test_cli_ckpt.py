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
