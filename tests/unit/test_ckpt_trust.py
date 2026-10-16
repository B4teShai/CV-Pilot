"""Who may vouch for a checkpoint: HMAC tags, the run-store anchor, and the trust policy."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pytest
from ckpt_support import (
    FunctionAnchor,
    blob,
    data_file,
    forge_manifest,
    forge_pointer,
    kind_dir,
    make,
    make_writable,
    manifest_file,
    save,
)
from test_runs import make as make_run_meta

from cvpilot import runs
from cvpilot.ckpt import auth
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError, CheckpointUntrustedError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
KEY = b"k" * 40
OTHER = b"o" * 40
SECRET_TEXT = "TopSecretKeyMaterial-0123456789-abcdefghijklmnop"  # noqa: S105


def steps(rc: RunCheckpoints) -> list[int]:
    return [v.manifest.step for v in rc.iter_valid("last")]


def reasons(rc: RunCheckpoints) -> dict[int | None, str]:
    return {s.step: s.reason for s in rc.skipped}


def reopen(rc: RunCheckpoints, **kw: Any) -> RunCheckpoints:
    return RunCheckpoints(rc.path.parent, rc.run_id, **kw)


def test_the_key_comes_from_the_environment_and_is_length_checked() -> None:
    assert auth.load_key({}) is None
    assert auth.load_key({"CVPILOT_CKPT_KEY": "  "}) is None
    assert auth.load_key({"CVPILOT_CKPT_KEY": "x" * 32}) == b"x" * 32
    for bad in ("short-key", "x" * 5000):
        with pytest.raises(CheckpointError) as info:
            auth.load_key({"CVPILOT_CKPT_KEY": bad})
        assert bad not in str(info.value)


def test_tags_bind_the_label_the_run_and_the_bytes() -> None:
    base = auth.tag(KEY, auth.MANIFEST_LABEL, "A" * 26, b"data")
    assert base != auth.tag(OTHER, auth.MANIFEST_LABEL, "A" * 26, b"data")
    assert base != auth.tag(KEY, auth.POINTER_LABEL, "A" * 26, b"data")
    assert base != auth.tag(KEY, auth.MANIFEST_LABEL, "B" * 26, b"data")
    assert base != auth.tag(KEY, auth.MANIFEST_LABEL, "A" * 26, b"datb")
    assert auth.verify(KEY, auth.MANIFEST_LABEL, "A" * 26, b"data", base)
    assert not auth.verify(KEY, auth.MANIFEST_LABEL, "A" * 26, b"data", "0" * 64)
    assert not auth.verify(KEY, auth.MANIFEST_LABEL, "A" * 26, b"data", "not hex")


def test_tagged_saves_restore_as_authenticated(tmp_path: Path) -> None:
    rc = make(tmp_path, key=KEY)
    manifest = save(rc, 1)
    assert manifest.auth_tag is not None
    pointer = rc.read_pointer("last")
    assert pointer is not None and pointer.auth_tag is not None
    assert [v.trust for v in rc.iter_valid("last")] == ["authenticated"]


def test_an_untagged_directory_is_local_chain_only_and_a_store_match_is_anchored(
    tmp_path: Path,
) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    assert [v.trust for v in rc.iter_valid("last")] == ["local_chain_only"]
    digest = hashlib.sha256(manifest_file(rc, 1).read_bytes()).hexdigest()
    anchored = reopen(rc, anchor=lambda run, kind, step: digest)
    assert [v.trust for v in anchored.iter_valid("last")] == ["anchored"]


def test_a_forged_coherent_directory_without_the_key_is_rejected_when_authenticated(
    tmp_path: Path,
) -> None:
    rc = make(tmp_path, key=KEY)
    save(rc, 1)
    save(rc, 2)
    evil = blob(666)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(evil)
    forged = forge_manifest(rc, 2, sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    forge_pointer(rc, forged, 2)  # the attacker also rewrites LATEST, chain and digests
    strict = reopen(rc, key=KEY, require_trust="authenticated")
    assert steps(strict) == [1]
    assert reasons(strict) == {2: "auth_invalid"}
    permissive = reopen(rc, key=KEY, require_trust="any")
    assert steps(permissive) == [1] and reasons(permissive) == {2: "auth_invalid"}


def test_the_same_directory_without_any_tags_needs_policy_any(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    assert [v.trust for v in reopen(rc).iter_valid("last")] == ["local_chain_only"] * 2
    strict = reopen(rc, key=KEY, require_trust="authenticated")
    assert steps(strict) == []
    assert set(reasons(strict).values()) == {"auth_missing"}
    assert steps(reopen(rc, require_trust="anchored")) == []


def test_a_wrong_key_rejects_everything_that_was_tagged(tmp_path: Path) -> None:
    rc = make(tmp_path, key=KEY)
    save(rc, 1)
    for policy in ("any", "authenticated"):
        wrong = reopen(rc, key=OTHER, require_trust=policy)
        assert steps(wrong) == [] and set(reasons(wrong).values()) == {"key_changed"}


def test_key_rotation_rejects_old_tags_and_accepts_unsigned_files_only_under_any(
    tmp_path: Path,
) -> None:
    rc = make(tmp_path)
    save(rc, 1)  # written before a key existed
    tagged = reopen(rc, key=KEY)
    save(tagged, 2)  # written under the first key
    # without a run store nothing verified says the run was authenticated: a planted or stale tag
    # is not evidence, so only the file with the foreign tag is refused (as key_changed)
    rotated = reopen(rc, key=OTHER)
    assert steps(rotated) == [1]  # 2 fails its tag; 1 has none and policy is any
    assert reasons(rotated) == {2: "key_changed"}  # a different key, said so
    assert [v.trust for v in rotated.iter_valid("last")] == ["local_chain_only"]
    assert steps(reopen(rc, key=OTHER, require_trust="authenticated")) == []


def test_a_broken_anchor_is_not_the_same_as_no_anchor(tmp_path: Path, events: Any) -> None:
    rc = make(tmp_path)
    save(rc, 1)

    def broken(run: str, kind: str, step: int) -> str | None:
        raise RuntimeError("store unreadable")

    strict = reopen(rc, anchor=FunctionAnchor(broken), require_trust="anchored")
    assert steps(strict) == [] and reasons(strict) == {1: "anchor_error"}
    lenient = reopen(rc, anchor=broken, require_trust="any")
    assert [v.trust for v in lenient.iter_valid("last")] == ["local_chain_only"]
    assert any(e["message"] == "checkpoint_anchor_failed" for e in events())


def test_anchored_policy_needs_the_stores_own_digest_for_every_candidate(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in (1, 2, 3):
        save(rc, step)
    digest3 = hashlib.sha256(manifest_file(rc, 3).read_bytes()).hexdigest()
    strict = reopen(
        rc,
        anchor=FunctionAnchor(lambda run, kind, step: digest3 if step == 3 else None),
        require_trust="anchored",
    )
    assert steps(strict) == [3]  # 2 and 1 are vouched for by 3's chain, which does not count
    assert reasons(strict) == {2: "not_anchored", 1: "not_anchored"}


def test_a_forged_newer_manifest_cannot_launder_an_older_one_when_authenticated(
    tmp_path: Path,
) -> None:
    rc = make(tmp_path)
    save(rc, 1)  # untagged history
    tagged = reopen(rc, key=KEY)
    save(tagged, 2)
    # an attacker without the key appends a coherent step 3 whose chain vouches for step 2
    save(reopen(rc, accept_auth_downgrade=True), 3)  # untagged, chained to the tagged 2
    strict = reopen(rc, key=KEY, require_trust="authenticated")
    assert steps(strict) == [2]
    assert reasons(strict) == {3: "auth_downgrade", 1: "auth_missing"}


def test_the_tag_covers_the_run_id(tmp_path: Path) -> None:
    a = make(tmp_path / "a", key=KEY)
    save(a, 1)
    b = make(tmp_path / "b", key=KEY)
    kind_dir(b).mkdir(mode=0o700, exist_ok=True)
    for name in ("step_000000001.safetensors", "step_000000001.manifest.json"):
        shutil.copy(kind_dir(a) / name, kind_dir(b) / name)
    # copied as is: the run id inside disagrees with the directory
    copied = reopen(b, key=KEY)
    assert steps(copied) == [] and reasons(copied) == {1: "name_mismatch"}
    # rewritten for run b with a recomputed chain: only the tag can tell
    forge_manifest(b, 1, run_id=b.run_id)
    got = reopen(b, key=KEY, require_trust="authenticated")
    assert steps(got) == [] and reasons(got) == {1: "auth_invalid"}


def test_a_pointer_without_a_valid_tag_is_ignored_when_a_key_is_set(tmp_path: Path) -> None:
    rc = make(tmp_path, key=KEY)
    save(rc, 1)
    assert reopen(rc, key=KEY).read_pointer("last") is not None
    pointer = rc.path / "LATEST"
    make_writable(pointer)
    fields = json.loads(pointer.read_text())
    fields["auth_tag"] = "0" * 64
    pointer.write_text(json.dumps(fields))
    assert reopen(rc, key=KEY).read_pointer("last") is None


def test_invalid_policies_are_refused(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError):
        make(tmp_path, require_trust="whatever")


def test_the_key_never_reaches_logs_files_the_store_or_error_text(
    tmp_path: Path, events: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", SECRET_TEXT)
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    manifest = manager.save_now(1, model={"w": [1.0]})
    path = tmp_path / "ckpt" / run.run_id / manifest.path
    make_writable(path)
    path.write_bytes(b"damaged")
    with pytest.raises(CheckpointUntrustedError) as gone:  # step 1 is known, nothing verifies
        manager.restore()
    texts = [json.dumps(e, default=str) for e in events()] + [str(gone.value)]
    with pytest.raises(CheckpointError) as info:
        manager.save_now(1, model={"w": [2.0]})
    texts.append(str(info.value))
    run.finish()
    blobs = [t.encode() for t in texts]
    for root, _, files in os.walk(tmp_path):
        for name in files:
            if name.endswith(("-wal", "-shm")) or ".sqlite" in name or name.endswith(".json"):
                blobs.append((Path(root) / name).read_bytes())
            if name in ("LATEST", "LATEST_BEST"):
                blobs.append((Path(root) / name).read_bytes())
    assert all(SECRET_TEXT.encode() not in b for b in blobs)


def test_manager_restore_reports_the_trust_level_and_logs_it(
    tmp_path: Path, events: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "z" * 40)
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    manager.save_now(1, model={"w": [1.0]})
    restored = manager.restore()
    assert restored is not None and restored.trust == "authenticated"
    logged = [e for e in events() if e["message"] == "checkpoint_restored"]
    assert logged[0]["trust_level"] == "authenticated"
    run.finish()


def untrusted_manager(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str) -> Any:
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    writer = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, install_signal_handlers=False
    )
    writer.save_now(1, model={"w": [1.0]})  # no key: local_chain_only
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "q" * 40)
    reader = CheckpointManager(
        run,
        dir=tmp_path / "ckpt",
        every_n_steps=1,
        install_signal_handlers=False,
        require_trust=policy,
        allow_weaker_trust=policy == "any",  # a key is set: weaker than authenticated needs this
    )
    return run, reader


def test_restore_fails_closed_when_the_required_level_is_not_met(
    tmp_path: Path, events: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, reader = untrusted_manager(tmp_path, monkeypatch, "authenticated")
    with pytest.raises(CheckpointUntrustedError):
        reader.restore()
    assert any(e["message"] == "checkpoint_untrusted" for e in events())
    run.finish()


def test_the_policy_default_is_any_and_comes_from_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, reader = untrusted_manager(tmp_path, monkeypatch, "any")
    restored = reader.restore()
    # same run store: its digest matches, so the level is anchored, not merely chained
    assert restored is not None and restored.trust == "anchored"
    run.finish()
    monkeypatch.setenv("CVPILOT_CKPT_REQUIRE_TRUST", "authenticated")
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "q" * 40)
    meta, repro = make_run_meta()
    second = runs.start_run(meta, repro, store_dir=tmp_path / "store2")
    strict = CheckpointManager(
        second, dir=tmp_path / "ckpt2", every_n_steps=1, install_signal_handlers=False
    )
    assert strict.require_trust == "authenticated"
    second.finish()


def test_an_untrusted_nothing_is_not_an_error_when_nothing_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "q" * 40)
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    reader = CheckpointManager(
        run, dir=tmp_path / "ckpt", install_signal_handlers=False, require_trust="authenticated"
    )
    assert reader.restore() is None
    run.finish()
