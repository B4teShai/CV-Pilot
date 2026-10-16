"""A bucket cannot hide checkpoints behind junk names, show a poisoned floor, or turn a failed
listing into a permanent fresh start."""

from __future__ import annotations

import hashlib
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
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError
from cvpilot.core.hashing import checkpoint_chain
from cvpilot.core.schema import CheckpointManifest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
Events = Callable[[], list[dict[str, Any]]]
URI = "s3://bucket-name/window"
KEY = "6b" * 32


def weights(v: float) -> dict[str, Any]:
    return {"w": np.full(2, v, dtype=np.float32)}


def writer(
    tmp: Path, fake: FakeRemote, monkeypatch: pytest.MonkeyPatch, *, key: str | None
) -> tuple[runs.RunHandle, CheckpointManager]:
    if key:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", key)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt", every_n_steps=1, keep_last=20, remote=URI, remote_store=fake,
        install_signal_handlers=False, remote_sleep=lambda s: None,
    )  # fmt: skip
    for step in (1, 2, 3):
        mgr.save_now(step, model=weights(float(step)))
    return run, mgr


def forged_remote_manifest(fake: FakeRemote, run: runs.RunHandle, step: int) -> None:
    """A self-consistent, untagged manifest and data at ``step``, as a bucket writer can make."""
    prefix = f"{run.run_id}/last/"
    parent_raw = fake.objects[prefix + "step_000000003.manifest.json"]
    parent = json.loads(parent_raw)
    data = blob(step % 97)
    sha = hashlib.sha256(data).hexdigest()
    manifest = CheckpointManifest(
        run_id=run.run_id, step=step, kind="last", sha256=sha, size_bytes=len(data),
        path=f"last/step_{step:09d}.safetensors", format="safetensors", versions={},
        dataset_hash="merkle:" + "c" * 64, created_at=parent["created_at"],
        prev_manifest_sha256=hashlib.sha256(parent_raw).hexdigest(),
        prev_chain_sha256=parent["chain_sha256"], prev_step=3, schema_version=2, chain_version=1,
        chain_sha256=checkpoint_chain(
            parent["chain_sha256"], run.run_id, "last", step, sha, len(data), "safetensors"
        ),
    )  # fmt: skip
    fake.objects[prefix + f"step_{step:09d}.safetensors"] = data
    fake.objects[prefix + f"step_{step:09d}.manifest.json"] = (
        manifest.model_dump_json() + "\n"
    ).encode()


def test_junk_names_above_the_real_steps_do_not_hide_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    run, mgr = writer(tmp_path, fake, monkeypatch, key=KEY)
    for step in range(100, 130):  # 30 canonical-looking names with nothing behind them
        fake.objects[f"{run.run_id}/last/step_{step:09d}.manifest.json"] = b"{}"
    import shutil

    shutil.rmtree(tmp_path / "ckpt" / run.run_id)  # a fresh node: the bucket, the key, the store
    restored = mgr.restore()
    assert restored is not None and restored.manifest.step == 3
    run.finish()


def test_alias_names_and_duplicates_are_not_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    run, mgr = writer(tmp_path, fake, monkeypatch, key=KEY)
    fake.objects[f"{run.run_id}/last/step_0000000000009.manifest.json"] = b"{}"  # an alias
    from cvpilot.ckpt.remote import Mirror
    from cvpilot.core.uri import parse_remote_uri

    mirror = Mirror(parse_remote_uri(URI), fake, sleep=lambda s: None)
    assert mirror.list_steps(run.run_id, "last") == [3, 2, 1]
    run.finish()


def test_a_remote_that_lists_only_junk_is_an_error_not_a_fresh_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    run, mgr = writer(tmp_path, fake, monkeypatch, key=KEY)
    for key in [k for k in fake.objects if k.startswith(run.run_id)]:
        if key.endswith(".manifest.json"):
            fake.objects[key] = b"{}"
    import shutil

    shutil.rmtree(tmp_path / "ckpt" / run.run_id)
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore()
    run.finish()


def test_a_bucket_writer_cannot_raise_the_save_floor_under_any(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, events: Events
) -> None:
    fake = FakeRemote()
    run, mgr = writer(tmp_path, fake, monkeypatch, key=None)
    forged_remote_manifest(fake, run, 10**12)
    mgr._remote_max.clear()
    assert mgr.save_now(4, model=weights(4.0)).step == 4  # was CheckpointStepError
    assert [e for e in events() if e["message"] == "checkpoint_remote_floor_ignored"]
    run.finish()


def test_a_failed_listing_is_not_remembered_and_the_floor_comes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    run, mgr = writer(tmp_path, fake, monkeypatch, key=KEY)
    mgr._remote_max.clear()
    fake.down = True
    mgr._floor_hint("last")
    assert "last" not in mgr._remote_max  # nothing cached from the failure
    fake.down = False
    assert mgr._floor_hint("last") == 3
    run.finish()
