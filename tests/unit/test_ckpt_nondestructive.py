"""A rejection never moves or deletes a checkpoint file: history stays byte for byte, whatever
the reason a save raised."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import auth
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError, CheckpointUntrustedError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
K1 = "11" * 16 + "aa" * 16
K2 = "22" * 16 + "bb" * 16


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def history(tmp: Path, run: runs.RunHandle) -> dict[str, bytes]:
    return {
        str(p.relative_to(tmp / "ckpt")): p.read_bytes()
        for p in sorted((tmp / "ckpt" / run.run_id).rglob("*"))
        if p.is_file() and p.name not in {".lock", ".alive"}
    }


def world(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, *, key: str | None, **kw: Any
) -> tuple[runs.RunHandle, CheckpointManager]:
    if key is None:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    else:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", key)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False, **kw
    )
    for step in (1, 2):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def other(tmp: Path, run: runs.RunHandle, **kw: Any) -> CheckpointManager:
    return CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


def test_a_save_with_the_wrong_key_changes_nothing_and_the_right_key_still_restores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch, key=K1)
    before = history(tmp_path, run)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", K2)
    with pytest.raises(CheckpointUntrustedError) as caught:
        other(tmp_path, run).save_now(3, model=weights(3.0))
    assert caught.value.reason == "key_changed"
    assert history(tmp_path, run) == before
    monkeypatch.setenv("CVPILOT_CKPT_KEY", K1)
    restored = other(tmp_path, run).restore()
    assert restored is not None and restored.manifest.step == 2
    run.finish()


def test_a_save_without_the_key_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch, key=K1)
    before = history(tmp_path, run)
    monkeypatch.delenv("CVPILOT_CKPT_KEY")
    with pytest.raises(CheckpointUntrustedError) as caught:
        other(tmp_path, run, require_trust="any").save_now(3, model=weights(3.0))
    assert caught.value.reason == "key_required"
    assert history(tmp_path, run) == before
    run.finish()


def test_a_locked_run_store_under_anchored_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = world(tmp_path, monkeypatch, key=None, require_trust="anchored")
    before = history(tmp_path, run)

    class Down:
        def __call__(self, *a: Any) -> str | None:
            raise RuntimeError("store locked")

        def max_step(self, *a: Any) -> int | None:
            raise RuntimeError("store locked")

        def auth_evidence(self, *a: Any) -> Any:
            raise RuntimeError("store locked")

    rc = RunCheckpoints(
        tmp_path / "ckpt", run.run_id, anchor=Down(), require_trust="anchored", create=True
    )
    with pytest.raises(CheckpointError):
        rc.save(
            kind="last", step=3, data=b"x" * 40, format="safetensors", versions={},
            dataset_hash="merkle:" + "c" * 64,
        )  # fmt: skip
    assert history(tmp_path, run) == before
    run.finish()


def test_a_legacy_manifest_present_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from cvpilot.core.hashing import checkpoint_chain

    run, mgr = world(tmp_path, monkeypatch, key=None, require_trust="anchored")
    path = tmp_path / "ckpt" / run.run_id / "last" / "step_000000002.manifest.json"
    doc = json.loads(path.read_text())
    for field in ("schema_version", "chain_version", "tag_alg", "key_id", "auth_tag"):
        doc.pop(field, None)
    doc["chain_sha256"] = checkpoint_chain(
        doc["prev_chain_sha256"], doc["run_id"], "last", 2, doc["sha256"], doc["size_bytes"]
    )
    path.chmod(0o600)
    path.write_text(json.dumps(doc) + "\n")
    before = history(tmp_path, run)
    mgr.save_now(
        3, model=weights(3.0)
    )  # chains to the trusted step 1; the legacy one is left alone
    after = history(tmp_path, run)
    assert all(
        after[name] == data for name, data in before.items() if "LATEST" not in name
    )  # nothing changed or went
    run.finish()


def test_a_planted_manifest_is_listed_not_moved_and_release_deletes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    from ckpt_support import blob

    from cvpilot.core.hashing import checkpoint_chain
    from cvpilot.core.schema import CheckpointManifest

    run, mgr = world(tmp_path, monkeypatch, key=K1)
    folder = tmp_path / "ckpt" / run.run_id / "last"
    parent_raw = (folder / "step_000000002.manifest.json").read_bytes()
    import json

    parent = json.loads(parent_raw)
    data = blob(9)
    sha = hashlib.sha256(data).hexdigest()
    forged = CheckpointManifest(
        run_id=run.run_id, step=999, kind="last", sha256=sha, size_bytes=len(data),
        path="last/step_000000999.safetensors", format="safetensors", versions={},
        dataset_hash="merkle:" + "c" * 64, created_at=parent["created_at"],
        prev_manifest_sha256=hashlib.sha256(parent_raw).hexdigest(),
        prev_chain_sha256=parent["chain_sha256"], prev_step=2, schema_version=2, chain_version=1,
        chain_sha256=checkpoint_chain(
            parent["chain_sha256"], run.run_id, "last", 999, sha, len(data), "safetensors"
        ),
    )  # fmt: skip
    (folder / "step_000000999.safetensors").write_bytes(data)
    (folder / "step_000000999.manifest.json").write_text(forged.model_dump_json() + "\n")
    before = history(tmp_path, run)
    with pytest.raises(CheckpointError, match="quarantine"):
        mgr.save_now(3, model=weights(3.0))
    after = history(tmp_path, run)
    assert {k: v for k, v in after.items() if "QUARANTINE" not in k} == before
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key())
    assert [e.step for e in rc.quarantined()] == [999]
    assert rc.clear_quarantine() == 1
    assert (folder / "step_000000999.manifest.json").exists()  # released, not deleted
    assert mgr.save_now(3, model=weights(3.0)).step == 3
    moved = rc.move_rejected_to_trash()  # the one explicit, separate way to move files
    assert sorted(moved) == ["last/step_000000999.manifest.json", "last/step_000000999.safetensors"]
    assert not (folder / "step_000000999.manifest.json").exists()
    run.finish()


def test_the_cli_trash_does_nothing_without_yes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, invoke: Any
) -> None:
    run, mgr = world(tmp_path, monkeypatch, key=K1)
    before = history(tmp_path, run)
    res = invoke("ckpt", "trash", str(tmp_path / "ckpt"), "--run", run.run_id)
    assert res.code == 2 and history(tmp_path, run) == before
    run.finish()


def test_a_partial_anchor_is_refused_under_a_strict_policy(tmp_path: Path) -> None:
    from cvpilot.core.hashing import new_run_id

    class Partial:
        def __call__(self, *a: Any) -> str | None:
            return None

    with pytest.raises(CheckpointError, match="anchor"):
        RunCheckpoints(tmp_path / "ckpt", new_run_id(), anchor=Partial(), require_trust="anchored")
    RunCheckpoints(tmp_path / "ckpt", new_run_id(), anchor=Partial(), require_trust="any")


def test_a_failed_record_under_anchored_fails_the_save_and_holds_the_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Any
) -> None:
    run, mgr = world(tmp_path, monkeypatch, key=None, require_trust="anchored")
    pointer = (tmp_path / "ckpt" / run.run_id / "LATEST").read_bytes()

    def broken(*a: Any, **k: Any) -> None:
        raise RuntimeError("store locked")

    monkeypatch.setattr(run, "record_checkpoint", broken)
    with pytest.raises(CheckpointError, match="pointer was not moved"):
        mgr.save_now(3, model=weights(3.0))
    assert (tmp_path / "ckpt" / run.run_id / "LATEST").read_bytes() == pointer
    assert [e for e in events() if e["message"] == "checkpoint_unanchored"]
    run.finish()
