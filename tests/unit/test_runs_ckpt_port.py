"""What the checkpoint manager needs from a run: lineage, repro facts and the digest anchor."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from ckpt_support import DATASET
from test_runs import make

from cvpilot import runs
from cvpilot.core.errors import RunClosedError
from cvpilot.core.hashing import checkpoint_chain, new_run_id
from cvpilot.core.schema import GENESIS_DIGEST, CheckpointManifest


def manifest_for(run_id: str, step: int = 1) -> CheckpointManifest:
    sha = "a" * 64
    return CheckpointManifest(
        run_id=run_id,
        step=step,
        kind="last",
        path=f"last/step_{step:09d}.safetensors",
        sha256=sha,
        size_bytes=10,
        format="safetensors",
        versions={},
        dataset_hash=DATASET,
        created_at=datetime(2026, 10, 15, tzinfo=timezone.utc),
        prev_manifest_sha256=GENESIS_DIGEST,
        prev_chain_sha256=GENESIS_DIGEST,
        chain_sha256=checkpoint_chain(GENESIS_DIGEST, run_id, "last", step, sha, 10),
    )


def start_chain(tmp_path: Path, depth: int) -> list[runs.RunHandle]:
    """``depth`` runs, each resuming the previous (which finished ``preempted``)."""
    handles: list[runs.RunHandle] = []
    parent_meta = None
    for _ in range(depth):
        if parent_meta is None:
            meta, repro = make()
        else:
            child_id = new_run_id()
            meta = runs.resume_from(parent_meta, run_id=child_id, host="h", name=None)
            _, repro = make(run_id=child_id)
        handle = runs.start_run(meta, repro, store_dir=tmp_path)
        handles.append(handle)
        if len(handles) < depth:
            handle.finish("preempted")
            parent_meta = runs.get_run_view("p", handle.run_id, store_dir=tmp_path).meta
    return handles


def test_ancestors_lists_the_resume_chain_nearest_first(tmp_path: Path) -> None:
    chain = start_chain(tmp_path, 3)
    try:
        assert chain[2].ancestors() == [chain[1].run_id, chain[0].run_id]
        assert chain[0].ancestors() == []
    finally:
        chain[2].finish()


def test_ancestors_is_bounded(tmp_path: Path) -> None:
    chain = start_chain(tmp_path, 5)
    try:
        assert len(chain[4].ancestors(limit=2)) == 2
    finally:
        chain[4].finish()


def test_repro_facts_are_the_dataset_hash_and_commit(tmp_path: Path) -> None:
    meta, repro = make()
    handle = runs.start_run(meta, repro, store_dir=tmp_path)
    try:
        assert handle.repro_facts() == ("merkle:" + "c" * 64, None)
    finally:
        handle.finish()


def test_recording_a_checkpoint_stores_its_digest_for_the_anchor(tmp_path: Path) -> None:
    meta, repro = make()
    handle = runs.start_run(meta, repro, store_dir=tmp_path)
    handle.record_checkpoint(manifest_for(meta.run_id), "d" * 64)
    assert handle.checkpoint_digest(meta.run_id, "last", 1) == "d" * 64
    assert handle.checkpoint_digest(meta.run_id, "last", 2) is None
    handle.finish()
    with pytest.raises(RunClosedError):
        handle.record_checkpoint(manifest_for(meta.run_id, 2), "d" * 64)


def test_a_read_only_anchor_can_be_opened_without_a_run(tmp_path: Path) -> None:
    meta, repro = make()
    handle = runs.start_run(meta, repro, store_dir=tmp_path)
    handle.record_checkpoint(manifest_for(meta.run_id), "d" * 64)
    handle.finish()
    opened = runs.open_anchor("p", tmp_path)
    assert opened is not None
    anchor, close = opened
    try:
        assert anchor(meta.run_id, "last", 1) == "d" * 64
        assert anchor(meta.run_id, "last", 9) is None
    finally:
        close()
    assert runs.open_anchor("other", tmp_path) is None
