"""A run that was ever authenticated cannot be walked back to less: untagged newer manifests, a
missing or changed key, a legacy-format manifest, a bucket that only adds names, and a lower trust
level than the run reached are all refused or reported, whatever the policy."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import blob, run_meta

from cvpilot import runs
from cvpilot.ckpt import auth
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
K1 = "11" * 16 + "aa" * 16
K2 = "22" * 16 + "bb" * 16
URI = "s3://bucket-name/downgrade"


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def keyed_run(
    tmp: Path, monkeypatch: pytest.MonkeyPatch, *, key: str | None = K1, **kw: Any
) -> tuple[runs.RunHandle, CheckpointManager]:
    if key is None:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    else:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", key)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, install_signal_handlers=False,
        require_trust="any" if key is None else None, **kw,  # type: ignore[arg-type]
    )  # fmt: skip
    for step in (1, 2):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def reader(tmp: Path, run: runs.RunHandle, **kw: Any) -> CheckpointManager:
    """A manager on the same run (the store knows it) with its own policy and key."""
    return CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False, **kw
    )


def plant_untagged(tmp: Path, run_id: str, step: int, parent_digest_from: int = 2) -> None:
    """Files an adversary writes by hand: a correctly chained, untagged manifest at ``step`` that
    follows the genuine step ``parent_digest_from``, and data it names."""
    import hashlib

    from cvpilot.core.hashing import checkpoint_chain
    from cvpilot.core.schema import CheckpointManifest

    folder = tmp / "ckpt" / run_id / "last"
    parent_path = folder / f"step_{parent_digest_from:09d}.manifest.json"
    parent_raw = parent_path.read_bytes()
    parent = json.loads(parent_raw)
    data = blob(step)
    chain = checkpoint_chain(
        parent["chain_sha256"], run_id, "last", step, hashlib.sha256(data).hexdigest(), len(data),
        "safetensors",
    )  # fmt: skip
    forged = CheckpointManifest(
        run_id=run_id, step=step, kind="last", path=f"last/step_{step:09d}.safetensors",
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data), format="safetensors",
        versions={}, dataset_hash="merkle:" + "c" * 64, created_at=parent["created_at"],
        prev_manifest_sha256=hashlib.sha256(parent_raw).hexdigest(),
        prev_chain_sha256=parent["chain_sha256"], prev_step=parent_digest_from,
        chain_sha256=chain, schema_version=2, chain_version=1,
    )  # fmt: skip
    (folder / f"step_{step:09d}.safetensors").write_bytes(data)
    (folder / f"step_{step:09d}.manifest.json").write_text(forged.model_dump_json() + "\n")


def drop_pointer(tmp: Path, run_id: str) -> None:
    (tmp / "ckpt" / run_id / "LATEST").unlink()


# -- 1: authentication downgrade -------------------------------------------------------------


@pytest.mark.parametrize("policy", ["any", "anchored"])
def test_an_untagged_newer_manifest_is_refused_once_the_run_was_authenticated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events, policy: str
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    plant_untagged(tmp_path, run.run_id, 3)
    drop_pointer(tmp_path, run.run_id)  # nothing newer than step 2 was ever recorded or tagged
    weak = reader(tmp_path, run, require_trust=policy)  # type: ignore[arg-type]
    restored = weak.restore()
    assert restored is not None and restored.manifest.step == 2  # never the planted step 3
    reasons = {e.get("reason") for e in events() if e["message"] == "checkpoint_untrusted"}
    assert "auth_downgrade" in reasons
    run.finish()


def test_the_downgrade_is_refused_even_when_nothing_else_is_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    plant_untagged(tmp_path, run.run_id, 3)
    for step in (1, 2):
        for suffix in (".manifest.json", ".safetensors"):
            (tmp_path / "ckpt" / run.run_id / "last" / f"step_{step:09d}{suffix}").unlink()
    drop_pointer(tmp_path, run.run_id)
    with pytest.raises(CheckpointUntrustedError):
        reader(tmp_path, run, require_trust="any").restore()
    run.finish()


def test_an_explicit_constructor_acknowledgement_lets_an_any_restore_through_and_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    plant_untagged(tmp_path, run.run_id, 3)
    drop_pointer(tmp_path, run.run_id)
    acknowledged = reader(tmp_path, run, require_trust="any", allow_weaker_trust=True)
    restored = acknowledged.restore()
    assert restored is not None and restored.manifest.step == 3
    assert [e for e in events() if e["message"] == "checkpoint_weaker_trust_allowed"]
    run.finish()


def test_the_authenticated_policy_still_refuses_the_untagged_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    plant_untagged(tmp_path, run.run_id, 3)
    drop_pointer(tmp_path, run.run_id)
    restored = reader(tmp_path, run).restore()
    assert restored is not None and restored.manifest.step == 2
    run.finish()


def test_a_run_that_never_had_a_key_is_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    restored = reader(tmp_path, run, require_trust="any").restore()
    assert restored is not None and restored.manifest.step == 2
    run.finish()


def test_the_pointer_remembers_when_authentication_began_and_is_tag_covered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    pointer = json.loads((tmp_path / "ckpt" / run.run_id / "LATEST").read_text())
    assert pointer["auth_since"] == 1
    pointer["auth_since"] = 99  # a forged pointer cannot move the start of authentication
    path = tmp_path / "ckpt" / run.run_id / "LATEST"
    path.chmod(0o600)
    path.write_text(json.dumps(pointer))
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key())
    assert rc.read_pointer("last") is None
    run.finish()


# -- 2: format and version downgrade ---------------------------------------------------------


def legacy(tmp: Path, run: runs.RunHandle, step: int = 2) -> None:
    """Rewrite a manifest the way releases without versions wrote it: no version fields, and the
    chain computed without the format."""
    from cvpilot.core.hashing import checkpoint_chain

    path = tmp / "ckpt" / run.run_id / "last" / f"step_{step:09d}.manifest.json"
    doc = json.loads(path.read_text())
    for field in ("schema_version", "chain_version", "tag_alg", "key_id", "auth_tag"):
        doc.pop(field, None)
    doc["chain_sha256"] = checkpoint_chain(
        doc["prev_chain_sha256"], doc["run_id"], doc["kind"], doc["step"], doc["sha256"],
        doc["size_bytes"],
    )  # fmt: skip
    path.chmod(0o600)
    path.write_text(json.dumps(doc) + "\n")


@pytest.mark.parametrize("policy", ["anchored"])
def test_a_legacy_format_manifest_is_refused_under_the_strict_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    legacy(tmp_path, run)
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, require_trust=policy)  # type: ignore[arg-type]
    reasons = {e.step: e.reason for e in rc.inspect("last", check_data=False)}
    assert reasons[2] == "legacy_manifest"
    run.finish()


def test_under_any_a_legacy_manifest_is_reported_and_a_current_one_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    legacy(tmp_path, run)
    drop_pointer(tmp_path, run.run_id)  # it names the manifest as it was before the rewrite
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, require_trust="any")
    assert [e.reason for e in rc.inspect("last", check_data=False) if e.step == 2] == [None]
    assert [e for e in events() if e["message"] == "checkpoint_legacy_manifest"]
    run.finish()


def test_a_legacy_manifest_cannot_stand_in_for_a_tagged_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    legacy(tmp_path, run)  # strips the version and the key id; the tag no longer matches either
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key())
    reasons = {e.step: e.reason for e in rc.inspect("last", check_data=False)}
    assert reasons[2] is not None and reasons[1] is None
    run.finish()


def test_an_unknown_version_is_refused_under_every_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    path = tmp_path / "ckpt" / run.run_id / "last" / "step_000000002.manifest.json"
    doc = json.loads(path.read_text())
    doc["schema_version"] = 99
    path.chmod(0o600)
    path.write_text(json.dumps(doc) + "\n")
    for policy in ("any", "anchored"):
        rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, require_trust=policy)  # type: ignore[arg-type]
        assert {e.step: e.reason for e in rc.inspect("last", check_data=False)}[2] in {
            "unknown_version",
            "manifest_invalid",
        }
    run.finish()


def test_the_format_is_bound_into_the_chain_so_a_swapped_suffix_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    folder = tmp_path / "ckpt" / run.run_id / "last"
    (folder / "step_000000002.safetensors").rename(folder / "step_000000002.pt")
    path = folder / "step_000000002.manifest.json"
    doc = json.loads(path.read_text())
    doc["path"], doc["format"] = "last/step_000000002.pt", "pt"
    path.chmod(0o600)
    path.write_text(json.dumps(doc) + "\n")
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, require_trust="any", allow_pt=True)
    reason = {e.step: e.reason for e in rc.inspect("last", check_data=False)}[2]
    assert reason in {"chain_mismatch", "store_mismatch"}
    run.finish()


def test_the_pt_opt_in_applies_even_to_a_manifest_that_is_otherwise_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, require_trust="any", allow_pt=False)
    saved = rc.save(
        kind="last", step=50, data=b"PK\x03\x04" + bytes(64), format="pt", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    assert saved.format == "pt"
    assert [v.manifest.step for v in rc.iter_valid("last")] == [2, 1]  # pt refused: not opted in
    run.finish()


# -- 3: trust-level regression ---------------------------------------------------------------


def test_a_manifest_the_store_never_recorded_is_flagged_as_a_regression_under_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch, key=None)
    plant_untagged(tmp_path, run.run_id, 3)
    drop_pointer(tmp_path, run.run_id)
    restored = reader(tmp_path, run, require_trust="any").restore()
    assert restored is not None  # `any` still serves it ...
    assert [e for e in events() if e["message"] == "trust_regressed"]  # ... but says so
    run.finish()


# -- 4: key handling -------------------------------------------------------------------------


def test_restoring_without_the_key_after_an_authenticated_run_fails_closed_and_says_why(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    monkeypatch.delenv("CVPILOT_CKPT_KEY")
    for policy in ("any", "anchored"):
        with pytest.raises(CheckpointUntrustedError) as caught:
            reader(tmp_path, run, require_trust=policy).restore()  # type: ignore[arg-type]
        assert caught.value.reason == "key_required" and "CVPILOT_CKPT_KEY" in str(caught.value)
    run.finish()


def test_the_explicit_acknowledgement_allows_a_keyless_restore_and_is_logged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    monkeypatch.delenv("CVPILOT_CKPT_KEY")
    ack = reader(tmp_path, run, require_trust="any", allow_weaker_trust=True)
    assert ack.restore() is not None
    assert [e for e in events() if e["message"] == "checkpoint_weaker_trust_allowed"]
    run.finish()


def test_a_rotated_key_is_refused_for_save_and_restore_without_the_acknowledgement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", K2)
    other = reader(tmp_path, run)
    with pytest.raises(CheckpointUntrustedError) as caught:
        other.restore()
    assert caught.value.reason == "key_changed" and "accept_key_change" in str(caught.value)
    from cvpilot.core.errors import CheckpointError

    with pytest.raises(CheckpointError, match="key"):
        other.save_now(3, model=weights(3.0))
    run.finish()


def test_the_documented_rotation_path_old_key_to_restore_then_acknowledge_and_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    run, mgr = keyed_run(tmp_path, monkeypatch)
    restored = reader(tmp_path, run).restore()  # 1. with the OLD key the old checkpoints load
    assert restored is not None and restored.manifest.step == 2
    monkeypatch.setenv("CVPILOT_CKPT_KEY", K2)  # 2. switch keys
    rotated = reader(tmp_path, run, accept_key_change=True, restart_chain=True)
    saved = rotated.save_now(3, model=weights(3.0))  # 3. the new key starts a visibly reset chain
    assert saved.chain_reset and saved.key_id != restored.manifest.key_id
    assert [e for e in events() if e["message"] == "checkpoint_key_change_accepted"]
    again = reader(tmp_path, run).restore()  # 4. from now on only the new key is expected
    assert again is not None and again.manifest.step == 3
    run.finish()


# -- 5: remote downgrade ---------------------------------------------------------------------


def test_a_remote_that_only_lists_a_name_does_not_launder_a_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", K1)
    fake = FakeRemote()
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, keep_last=20, remote=URI, remote_store=fake,
        install_signal_handlers=False, remote_sleep=lambda s: None,
    )  # fmt: skip
    for step in (1, 2, 3):
        mgr.save_now(step, model=weights(float(step)))
    for suffix in (".manifest.json", ".safetensors"):  # the newest is rolled back locally ...
        (tmp_path / "ckpt" / run.run_id / "last" / f"step_000000003{suffix}").unlink()
    for suffix in (".manifest.json", ".safetensors"):  # ... and on the remote
        del fake.objects[f"{run.run_id}/last/step_000000003{suffix}"]
    fake.objects[f"{run.run_id}/last/step_000000009.manifest.json"] = b"{}"  # a name, no evidence
    with pytest.raises(CheckpointUntrustedError) as caught:
        mgr.restore()
    assert caught.value.reason == "rolled_back"
    run.finish()


def test_an_untagged_remote_manifest_does_not_replace_a_stricter_local_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", K1)
    fake = FakeRemote()
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, remote=URI, remote_store=fake,
        install_signal_handlers=False, remote_sleep=lambda s: None,
    )  # fmt: skip
    mgr.save_now(1, model=weights(1.0))
    local = (tmp_path / "ckpt" / run.run_id / "last" / "step_000000001.manifest.json").read_bytes()
    stripped = json.loads(local)
    stripped["auth_tag"] = None
    fake.objects[f"{run.run_id}/last/step_000000001.manifest.json"] = json.dumps(stripped).encode()
    rc = RunCheckpoints(tmp_path / "ckpt", run.run_id, key=auth.load_key())
    from cvpilot.ckpt.remote import Mirror
    from cvpilot.core.uri import parse_remote_uri

    mirror = Mirror(parse_remote_uri(URI), fake, sleep=lambda s: None)
    assert mirror.fetch(rc, "last", 1) is False  # the local copy is kept as it is
    assert (
        tmp_path / "ckpt" / run.run_id / "last" / "step_000000001.manifest.json"
    ).read_bytes() == local
    run.finish()
