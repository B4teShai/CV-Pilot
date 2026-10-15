"""A manifest names the project and parent of its run (covered by the tag), an upload marker is
tagged under a key, and a weak key is flagged."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import auth
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError, CheckpointUntrustedError
from cvpilot.core.hashing import new_run_id

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
KEY = "bc" * 32
URI = "s3://bucket-name/bind"


def make(
    tmp: Path, project: str, store: str, *, parent: str | None = None, **kw: Any
) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta(project)
    if parent is not None:
        meta = runs.resume_from(
            runs.get_run_view(project, parent, store_dir=tmp / store).meta,
            run_id=new_run_id(), host="h", name="job",
        )  # fmt: skip
        repro = repro.model_copy(update={"run_id": meta.run_id})
    run = runs.start_run(meta, repro, store_dir=tmp / store)
    options: dict[str, Any] = {"keep_last": 10, **kw}
    return run, CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, install_signal_handlers=False, **options
    )


def test_a_manifest_records_its_project_and_parent_and_the_tag_covers_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    run, mgr = make(tmp_path, "alpha", "s")
    mgr.save_now(1, model={"w": np.ones(2)})
    run.finish("preempted")
    child, cmgr = make(tmp_path, "alpha", "s", parent=run.run_id)
    manifest = cmgr.save_now(2, model={"w": np.ones(2)})
    assert (manifest.project, manifest.parent_run_id) == ("alpha", run.run_id)
    path = tmp_path / "ckpt" / child.run_id / "last" / "step_000000002.manifest.json"
    for field, value in (("project", "beta"), ("parent_run_id", new_run_id())):
        forged = json.loads(path.read_text())
        forged[field] = value
        path.chmod(0o600)
        path.write_text(json.dumps(forged) + "\n")
        rc = RunCheckpoints(tmp_path / "ckpt", child.run_id, key=auth.load_key(),
                            require_trust="authenticated")  # fmt: skip
        assert [e.reason for e in rc.inspect("last", check_data=False) if e.step == 2] == [
            "auth_invalid"
        ]
        forged[field] = getattr(manifest, field)
        path.write_text(json.dumps(forged) + "\n")
    child.finish()


def test_another_projects_checkpoint_is_refused_by_a_restart_of_this_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    other, omgr = make(tmp_path, "other", "so")
    omgr.save_now(1, model={"w": np.ones(2)})
    other.finish("preempted")
    mine, mmgr = make(tmp_path, "mine", "sm")
    assert mmgr.restore(run_ids=[other.run_id]) is None
    mine.finish()


def test_discovery_does_not_pick_another_projects_run_from_a_shared_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    fake = FakeRemote()
    other, omgr = make(tmp_path / "o", "other", "so", remote=URI, remote_store=fake,
                       remote_sleep=lambda s: None)  # fmt: skip
    omgr.save_now(1, model={"w": np.full(2, 7.0)})
    omgr.close()
    other.finish("preempted")
    mine, mmgr = make(tmp_path / "m", "mine", "sm", remote=URI, remote_store=fake,
                      remote_sleep=lambda s: None)  # fmt: skip
    with pytest.raises(CheckpointUntrustedError):
        mmgr.restore(discover=True)
    mine.finish()


def test_a_manifest_that_misstates_its_parent_is_refused_in_a_lineage_restore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    parent, pmgr = make(tmp_path, "alpha", "s")
    pmgr.save_now(1, model={"w": np.ones(2)})
    pmgr.close()
    parent.finish("preempted")
    path = tmp_path / "ckpt" / parent.run_id / "last" / "step_000000001.manifest.json"
    forged = json.loads(path.read_text())
    forged["parent_run_id"] = new_run_id()  # a root run that claims a parent it does not have
    path.chmod(0o600)
    path.write_text(json.dumps(forged) + "\n")
    child, cmgr = make(tmp_path, "alpha", "s", parent=parent.run_id)
    assert cmgr.restore() is None
    child.finish()


def test_an_honest_lineage_restores_with_the_binding_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    parent, pmgr = make(tmp_path, "alpha", "s")
    pmgr.save_now(1, model={"w": np.ones(2)})
    pmgr.close()
    parent.finish("preempted")
    child, cmgr = make(tmp_path, "alpha", "s", parent=parent.run_id)
    restored = cmgr.restore()
    assert restored is not None and restored.run_id == parent.run_id
    child.finish()


# -- the upload marker -----------------------------------------------------------------------


def marker_world(tmp: Path, monkeypatch: pytest.MonkeyPatch, keyed: bool) -> tuple[Any, ...]:
    if keyed:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    fake = FakeRemote()
    run, mgr = make(tmp, "alpha", "s", remote=URI, remote_store=fake, remote_sleep=lambda s: None)
    manifest = mgr.save_now(1, model={"w": np.ones(2)})
    return run, mgr, manifest, tmp / "ckpt" / run.run_id / "last" / "step_000000001.remote.json"


def test_under_a_key_the_marker_carries_a_tag_and_a_forged_one_confirms_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr, manifest, marker = marker_world(tmp_path, monkeypatch, keyed=True)
    body = json.loads(marker.read_text())
    assert len(body["auth_tag"]) == 64
    assert mgr._own.is_uploaded(manifest)
    forged = {k: v for k, v in body.items() if k != "auth_tag"}  # a writer without the key
    marker.chmod(0o600)
    marker.write_text(json.dumps(forged))
    assert not mgr._own.is_uploaded(manifest)
    forged["auth_tag"] = "0" * 64
    marker.write_text(json.dumps(forged))
    assert not mgr._own.is_uploaded(manifest)
    run.finish()


def test_a_marker_moved_to_another_run_confirms_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr, manifest, marker = marker_world(tmp_path, monkeypatch, keyed=True)
    other = RunCheckpoints(tmp_path / "ckpt", new_run_id(), key=auth.load_key(), create=True)
    assert other.run_id != run.run_id
    assert not other.is_uploaded(manifest)
    run.finish()


def test_without_a_key_markers_work_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run, mgr, manifest, marker = marker_world(tmp_path, monkeypatch, keyed=False)
    assert "auth_tag" not in json.loads(marker.read_text())
    assert mgr._own.is_uploaded(manifest)
    run.finish()


def test_retention_does_not_delete_on_a_forged_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    fake = FakeRemote()
    run, mgr = make(tmp_path, "alpha", "s", remote=URI, remote_store=fake,
                    remote_sleep=lambda s: None, keep_last=1)  # fmt: skip
    mgr.save_now(1, model={"w": np.ones(2)})
    fake.down = True  # uploads fail from here on: nothing newer is confirmed
    for step in (2, 3):
        mgr.save_now(step, model={"w": np.ones(2)})
    folder = tmp_path / "ckpt" / run.run_id / "last"
    marker = folder / "step_000000003.remote.json"
    forged = {"uri": URI, "manifest_sha256": "0" * 64, "sha256": "0" * 64, "size_bytes": 1}
    marker.write_text(json.dumps(forged))  # a forged confirmation for the newest
    mgr.save_now(4, model={"w": np.ones(2)})
    assert (folder / "step_000000003.manifest.json").exists()  # nothing was deleted
    run.finish()


# -- the key ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    ["ab" * 31, "0" * 62, "f" * 40, "AB" * 20],
)
def test_a_hex_key_shorter_than_32_bytes_is_refused(raw: str) -> None:
    with pytest.raises(CheckpointError, match="hex"):
        auth.load_key({auth.KEY_ENV: raw})


def test_a_32_byte_hex_or_base64_key_and_a_long_passphrase_are_accepted() -> None:
    import base64

    assert auth.load_key({auth.KEY_ENV: "ab" * 32}) is not None
    assert auth.load_key({auth.KEY_ENV: base64.b64encode(bytes(range(32))).decode()}) is not None
    assert auth.load_key({auth.KEY_ENV: "correct horse battery staple, with extra words"})


def test_a_base64_key_that_decodes_to_fewer_than_32_bytes_is_refused() -> None:
    import base64

    short = base64.b64encode(bytes(range(26))).decode()  # 36 characters, 26 bytes, padded
    with pytest.raises(CheckpointError, match="base64|bytes"):
        auth.load_key({auth.KEY_ENV: short})


@pytest.mark.parametrize("raw", ["k" * 40, "ab" * 32, "0123456789" * 4, "a" * 64])
def test_a_low_entropy_key_is_reported_as_weak(raw: str) -> None:
    assert auth.weakness(raw.encode()) is not None


def test_a_random_key_is_not_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    import secrets

    for _ in range(5):
        assert auth.weakness(secrets.token_hex(32).encode()) is None
        assert auth.weakness(secrets.token_urlsafe(40).encode()) is None


def test_the_manager_warns_once_about_a_weak_key_and_never_prints_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    monkeypatch.setenv("CVPILOT_CKPT_KEY", "k" * 40)
    run, mgr = make(tmp_path, "alpha", "s")
    weak = [e for e in events() if e["message"] == "checkpoint_key_weak"]
    assert len(weak) == 1 and weak[0]["level"] == "WARNING"
    assert "kkkk" not in json.dumps(events())
    run.finish()
