"""Remote mirror: upload after local verification, verify size and digest, retries, resume,
restore on a node that has nothing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt import remote as remote_mod
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointError, CheckpointPathError

Events = Callable[[], list[dict[str, Any]]]
URI = "s3://bucket-name/runs/exp"


def build(
    tmp_path: Path, fake: FakeRemote, *, sub: str = "a", **kw: Any
) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / f"store-{sub}")
    kw.setdefault("remote_sleep", lambda s: None)
    return run, CheckpointManager(
        run,
        dir=tmp_path / f"ckpt-{sub}",
        remote=URI,
        remote_store=fake,
        every_n_steps=1,
        install_signal_handlers=False,
        **kw,
    )


def weights(value: float) -> dict[str, Any]:
    return {"w": np.full(1000, value, dtype=np.float32)}


def keys(fake: FakeRemote) -> list[str]:
    return sorted(fake.objects)


def test_a_saved_checkpoint_is_mirrored_data_first_and_manifest_last(
    tmp_path: Path, events: Events
) -> None:
    fake = FakeRemote()
    run, mgr = build(tmp_path, fake)
    manifest = mgr.save_now(3, model=weights(3.0))
    base = f"{run.run_id}/last/step_000000003"
    assert keys(fake) == [f"{base}.manifest.json", f"{base}.safetensors"]
    assert hashlib.sha256(fake.objects[f"{base}.safetensors"]).hexdigest() == manifest.sha256
    assert (
        fake.objects[f"{base}.manifest.json"]
        == (tmp_path / "ckpt-a" / run.run_id / "last" / "step_000000003.manifest.json").read_bytes()
    )
    assert [k for k, _ in fake.moves][-1].endswith(".manifest.json.part")
    assert manifest.uri == f"{URI}/{run.run_id}/last/step_000000003.safetensors"
    uploaded = [e for e in events() if e["message"] == "checkpoint_uploaded"]
    assert (
        uploaded and uploaded[0]["step"] == 3 and uploaded[0]["size_bytes"] == manifest.size_bytes
    )
    assert mgr._own.is_uploaded(manifest)
    run.finish()


def test_the_manifest_is_never_uploaded_when_the_data_fails(tmp_path: Path, events: Events) -> None:
    fake = FakeRemote()
    fake.down = True
    run, mgr = build(tmp_path, fake, remote_retries=2)
    manifest = mgr.save_now(1, model=weights(1.0))  # the local save succeeds regardless
    assert fake.objects == {} and not mgr._own.is_uploaded(manifest)
    failed = [e for e in events() if e["message"] == "checkpoint_upload_failed"]
    assert failed and failed[0]["error_type"] == "OSError" and failed[0]["level"] == "WARNING"
    run.finish()


def test_transient_failures_are_retried_with_backoff(tmp_path: Path) -> None:
    fake = FakeRemote()
    fake.fail_writes = 2
    sleeps: list[float] = []
    run, mgr = build(tmp_path, fake, remote_sleep=sleeps.append, remote_retries=4)
    manifest = mgr.save_now(1, model=weights(1.0))
    assert mgr._own.is_uploaded(manifest)
    assert len(sleeps) == 2 and sleeps[1] > sleeps[0] > 0
    run.finish()


def test_the_total_time_is_bounded_by_a_deadline(tmp_path: Path) -> None:
    fake = FakeRemote()
    fake.down = True
    now = {"t": 0.0}

    def sleep(seconds: float) -> None:
        now["t"] += seconds

    run, mgr = build(
        tmp_path, fake, remote_sleep=sleep, remote_retries=50, remote_deadline=10.0,
        remote_clock=lambda: now["t"],
    )  # fmt: skip
    mgr.save_now(1, model=weights(1.0))
    assert now["t"] <= 10.0 + 8.0  # the deadline stops the retries long before 50 attempts
    run.finish()


def test_a_remote_that_silently_corrupts_is_detected_by_the_digest_check(
    tmp_path: Path, events: Events
) -> None:
    fake = FakeRemote()
    fake.corrupt = True
    run, mgr = build(tmp_path, fake, remote_retries=2)
    manifest = mgr.save_now(1, model=weights(1.0))
    assert not mgr._own.is_uploaded(manifest)
    assert not [k for k in fake.objects if k.endswith(".manifest.json")]
    assert not [k for k in fake.objects if k.endswith(".part")]  # partial objects are removed
    run.finish()


def test_an_interrupted_upload_resumes_from_the_verified_prefix(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, mgr = build(tmp_path, fake)
    fake.down = True
    manifest = mgr.save_now(1, model=weights(1.0))
    fake.down = False
    path = tmp_path / "ckpt-a" / run.run_id / manifest.path
    part = f"{run.run_id}/last/step_000000001.safetensors.part"
    half = manifest.size_bytes // 2
    fake.objects[part] = path.read_bytes()[:half]
    mgr._mirror_pending()
    assert mgr._own.is_uploaded(manifest)
    data_writes = [w for w in fake.writes if w[0] == part]
    assert data_writes == [(part, manifest.size_bytes - half, True)]  # only the rest was sent
    run.finish()


def test_a_wrong_partial_object_is_discarded_and_rewritten(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, mgr = build(tmp_path, fake)
    fake.down = True
    manifest = mgr.save_now(1, model=weights(1.0))
    fake.down = False
    part = f"{run.run_id}/last/step_000000001.safetensors.part"
    fake.objects[part] = b"\x00" * (manifest.size_bytes // 2)
    mgr._mirror_pending()
    assert mgr._own.is_uploaded(manifest)
    assert [w for w in fake.writes if w[0] == part] == [(part, manifest.size_bytes, False)]
    run.finish()


def test_an_already_verified_remote_copy_is_not_uploaded_again(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, mgr = build(tmp_path, fake)
    manifest = mgr.save_now(1, model=weights(1.0))
    writes = len(fake.writes)
    marker = tmp_path / "ckpt-a" / run.run_id / "last" / "step_000000001.remote.json"
    marker.chmod(0o644)
    marker.unlink()  # forget that it was uploaded: the remote itself must prove it
    mgr._mirror_pending()
    assert len(fake.writes) == writes and mgr._own.is_uploaded(manifest)
    run.finish()


def test_earlier_checkpoints_are_caught_up_by_the_next_successful_save(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, mgr = build(tmp_path, fake, keep_last=5)
    fake.down = True
    first = mgr.save_now(1, model=weights(1.0))
    second = mgr.save_now(2, model=weights(2.0))
    fake.down = False
    third = mgr.save_now(3, model=weights(3.0))
    assert all(mgr._own.is_uploaded(m) for m in (first, second, third))
    run.finish()


def test_nothing_is_deleted_locally_until_the_newest_is_confirmed_remotely(
    tmp_path: Path,
) -> None:
    fake = FakeRemote()
    run, mgr = build(tmp_path, fake, keep_last=1)
    mgr.save_now(1, model=weights(1.0))
    fake.down = True
    mgr.save_now(2, model=weights(2.0))
    mgr.save_now(3, model=weights(3.0))
    steps = sorted(e.step for e in mgr.list() if e.status == "valid" and e.step is not None)
    assert steps == [1, 2, 3]  # held back: the newest was never confirmed
    fake.down = False
    mgr.save_now(4, model=weights(4.0))
    steps = sorted(e.step for e in mgr.list() if e.status == "valid" and e.step is not None)
    assert steps == [4]
    run.finish()


def test_a_node_with_nothing_restores_from_the_remote(tmp_path: Path, events: Events) -> None:
    fake = FakeRemote()
    run, writer = build(tmp_path, fake, sub="a", keep_last=5)
    for step in (1, 2, 3):
        writer.save_now(step, model=weights(float(step)))
    old_id = run.run_id
    run.finish("preempted")
    run2, reader = build(tmp_path, fake, sub="b")
    restored = reader.restore(run_ids=[old_id])
    assert restored is not None and restored.manifest.step == 3 and restored.run_id == old_id
    assert np.all(restored.state["model"]["w"] == 3.0)
    assert any(e["message"] == "checkpoint_restored" for e in events())
    run2.finish()


def test_a_damaged_remote_checkpoint_is_skipped_for_an_older_one(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, writer = build(tmp_path, fake, sub="a", keep_last=5)
    for step in (1, 2, 3):
        writer.save_now(step, model=weights(float(step)))
    old_id = run.run_id
    run.finish("preempted")
    newest = f"{old_id}/last/step_000000003.safetensors"
    fake.objects[newest] = fake.objects[newest][:-10]  # truncated in the bucket
    run2, reader = build(tmp_path, fake, sub="b")
    restored = reader.restore(run_ids=[old_id])
    assert restored is not None and restored.manifest.step == 2
    run2.finish()


def test_a_forged_remote_manifest_is_never_installed(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, writer = build(tmp_path, fake, sub="a", keep_last=5)
    writer.save_now(1, model=weights(1.0))
    old_id = run.run_id
    run.finish("preempted")
    name = f"{old_id}/last/step_000000001.manifest.json"
    fields = json.loads(fake.objects[name])
    fields["size_bytes"] += 1  # the chain value no longer matches
    fake.objects[name] = json.dumps(fields).encode()
    run2, reader = build(tmp_path, fake, sub="b")
    assert reader.restore(run_ids=[old_id]) is None
    run2.finish()


def test_a_remote_manifest_of_another_run_is_refused(tmp_path: Path) -> None:
    fake = FakeRemote()
    run, writer = build(tmp_path, fake, sub="a")
    writer.save_now(1, model=weights(1.0))
    victim = "01JBXQ5S7H8E9K2M3N4P5Q6R7S"
    for key in list(fake.objects):
        fake.objects[key.replace(run.run_id, victim)] = fake.objects[key]
    run.finish("preempted")
    run2, reader = build(tmp_path, fake, sub="b")
    assert reader.restore(run_ids=[victim]) is None
    run2.finish()


@pytest.mark.parametrize(
    "uri",
    [
        "s3://user:secret@bucket-name/x",
        "s3://bucket-name/x?X-Amz-Signature=abc",
        "https://bucket-name/x",
        "s3://bucket-name/../x",
        "file:///etc",
    ],
)
def test_remote_locations_with_credentials_or_odd_schemes_are_refused(
    tmp_path: Path, uri: str
) -> None:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    with pytest.raises((CheckpointPathError, ValueError)) as info:
        CheckpointManager(
            run, dir=tmp_path / "c", remote=uri, remote_store=FakeRemote(),
            install_signal_handlers=False,
        )  # fmt: skip
    assert "secret" not in str(info.value) and "Signature" not in str(info.value)
    run.finish()


def test_the_remote_location_never_reaches_the_logs_with_more_than_scheme_and_bucket(
    tmp_path: Path, events: Events
) -> None:
    fake = FakeRemote()
    fake.down = True
    run, mgr = build(tmp_path, fake, remote_retries=1)
    mgr.save_now(1, model=weights(1.0))
    text = json.dumps(events(), default=str)
    assert "runs/exp" not in text and "OSError" in text
    run.finish()


def test_the_default_remote_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CVPILOT_CHECKPOINT_REMOTE", URI)
    monkeypatch.setenv("CVPILOT_CHECKPOINT_DIR", str(tmp_path / "envdir"))
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(run, remote_store=FakeRemote(), install_signal_handlers=False)
    assert mgr.remote_url == URI
    mgr.save_now(1, model=weights(1.0))
    assert (tmp_path / "envdir" / run.run_id / "last").is_dir()
    run.finish()


def test_a_missing_directory_is_a_usage_error(tmp_path: Path) -> None:
    meta, repro = run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    with pytest.raises(ValueError, match="dir"):
        CheckpointManager(run, install_signal_handlers=False)
    run.finish()


def test_retry_helper_reports_attempts_and_never_retries_non_io_errors() -> None:
    calls = {"n": 0}

    def boom() -> None:
        calls["n"] += 1
        raise ValueError("a bug, not a network problem")

    with pytest.raises(ValueError):
        remote_mod.retrying(boom, retries=5, backoff=0.1, deadline=10.0, sleep=lambda s: None)
    assert calls["n"] == 1
    assert CheckpointError
