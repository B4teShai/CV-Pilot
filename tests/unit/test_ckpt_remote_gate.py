"""A remote candidate is judged from its manifest alone, before any data is read, and a refused
one leaves nothing in the local directory."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_remote_support import FakeRemote
from ckpt_support import run_meta

from cvpilot import runs
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointUntrustedError

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")
URI = "s3://bucket-name/gate"
KEY = "78" * 32


def writer(
    tmp: Path, fake: FakeRemote, monkeypatch: pytest.MonkeyPatch, *, key: bool, mb: float = 0.0
) -> str:
    if key:
        monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    else:
        monkeypatch.delenv("CVPILOT_CKPT_KEY", raising=False)
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store-w")
    mgr = CheckpointManager(
        run, dir=tmp / "ckpt-w", remote=URI, remote_store=fake, every_n_steps=1, keep_last=5,
        install_signal_handlers=False, remote_sleep=lambda s: None,
    )  # fmt: skip
    extra = np.zeros(int(mb * 262144), dtype=np.float32)
    for step in (1, 2):
        mgr.save_now(step, model={"w": np.full(2, float(step)), "pad": extra})
    mgr.close()
    run.finish("preempted")
    return run.run_id


def reader(tmp: Path, fake: FakeRemote, **kw: Any) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp / "store-r")
    return run, CheckpointManager(
        run, dir=tmp / "ckpt-r", remote=URI, remote_store=fake, every_n_steps=1,
        install_signal_handlers=False, remote_sleep=lambda s: None, **kw,
    )  # fmt: skip


def installed(tmp: Path) -> list[str]:
    root = tmp / "ckpt-r"
    return (
        sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
        if root.exists()
        else []
    )


def data_reads(fake: FakeRemote) -> list[str]:
    return [k for k in fake.reads if k.endswith(".safetensors")]


def test_authenticated_refuses_an_untagged_candidate_before_reading_its_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake, monkeypatch, key=False)
    monkeypatch.setenv("CVPILOT_CKPT_KEY", KEY)
    run, mgr = reader(tmp_path, fake, require_trust="authenticated")
    fake.reads.clear()
    with pytest.raises(CheckpointUntrustedError):
        mgr.restore(run_ids=[old])
    assert data_reads(fake) == []  # judged from the manifest
    assert [p for p in installed(tmp_path) if p.endswith((".safetensors", ".part"))] == []
    run.finish()


def test_a_remote_manifest_that_disagrees_with_the_run_store_is_refused_before_the_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake, monkeypatch, key=False)
    # the reader shares the writer's run store, so it holds the genuine digests; the remote is
    # then altered (the manifest changes, the data stays)
    key = f"{old}/last/step_000000002.manifest.json"
    forged = json.loads(fake.objects[key])
    forged["git_commit"] = "a" * 40
    fake.objects[key] = (json.dumps(forged) + "\n").encode()
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store-w")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt-r", remote=URI, remote_store=fake, every_n_steps=1,
        install_signal_handlers=False, remote_sleep=lambda s: None, require_trust="anchored",
    )  # fmt: skip
    fake.reads.clear()
    # step 2 is refused, and step 1 alone is a rollback: the store knows step 2 existed, and the
    # altered remote manifest does not count as "still there"
    with pytest.raises(CheckpointUntrustedError) as caught:
        mgr.restore(run_ids=[old])
    assert caught.value.reason == "rolled_back"
    assert not any(k.endswith("step_000000002.safetensors") for k in fake.reads)
    accepted = mgr.restore(run_ids=[old], accept_rollback_to=1)
    assert accepted is not None and accepted.manifest.step == 1
    run.finish()


def test_a_candidate_over_the_per_candidate_cap_is_refused_without_reading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake, monkeypatch, key=False, mb=1.0)
    run, mgr = reader(tmp_path, fake, remote_max_candidate_bytes=200_000)
    fake.reads.clear()
    assert mgr.restore(run_ids=[old]) is None
    assert data_reads(fake) == [] and [p for p in installed(tmp_path) if "step_" in p] == []
    run.finish()


def test_the_total_budget_of_one_restore_is_enforced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake, monkeypatch, key=False, mb=1.0)
    size = len(fake.objects[f"{old}/last/step_000000002.safetensors"])
    run, mgr = reader(
        tmp_path, fake, remote_max_candidate_bytes=size + 10, remote_max_total_bytes=size + 10
    )
    first = mgr.restore(run_ids=[old])
    assert first is not None and first.manifest.step == 2  # the newest fits the budget
    run.finish()
    corrupt = FakeRemote()
    old2 = writer(tmp_path / "again", corrupt, monkeypatch, key=False, mb=1.0)
    corrupt.objects[f"{old2}/last/step_000000002.safetensors"] = b"x" * size  # damaged newest
    run2, mgr2 = reader(
        tmp_path / "again", corrupt, remote_max_candidate_bytes=size + 10,
        remote_max_total_bytes=size + 10,
    )  # fmt: skip
    assert mgr2.restore(run_ids=[old2]) is None  # the damaged one spent the budget: no 2nd fetch
    run2.finish()


def test_an_oversized_remote_manifest_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRemote()
    old = writer(tmp_path, fake, monkeypatch, key=False)
    fake.objects[f"{old}/last/step_000000002.manifest.json"] = b"{" + b" " * 200_000 + b"}"
    run, mgr = reader(tmp_path, fake)
    restored = mgr.restore(run_ids=[old])
    assert restored is not None and restored.manifest.step == 1
    run.finish()
