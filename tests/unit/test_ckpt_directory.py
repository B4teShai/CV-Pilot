"""One run's checkpoint directory: atomic save, trusted newest-first walk, retention."""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from ckpt_support import (
    DATASET,
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

from cvpilot.ckpt import fsops
from cvpilot.ckpt.directory import RunCheckpoints
from cvpilot.core.errors import (
    CheckpointBusyError,
    CheckpointError,
    CheckpointExistsError,
    CheckpointPathError,
    CheckpointStepError,
)
from cvpilot.core.schema import GENESIS_DIGEST

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="needs POSIX directory fds")


def valid_steps(rc: RunCheckpoints, kind: str = "last") -> list[int]:
    return [v.manifest.step for v in rc.iter_valid(kind)]  # type: ignore[arg-type]


def reasons(rc: RunCheckpoints) -> dict[int | None, str]:
    return {s.step: s.reason for s in rc.skipped}


# -- saving --------------------------------------------------------------------------------------


def test_save_writes_data_manifest_and_pointer_read_only_with_no_temp_left(tmp_path: Path) -> None:
    rc = make(tmp_path)
    manifest = save(rc, 5)
    assert manifest.path == "last/step_000000005.safetensors"
    assert manifest.sha256 == hashlib.sha256(data_file(rc, 5).read_bytes()).hexdigest()
    assert manifest.size_bytes == data_file(rc, 5).stat().st_size
    for path in (data_file(rc, 5), manifest_file(rc, 5), rc.path / "LATEST"):
        assert stat.S_IMODE(path.stat().st_mode) == 0o444, path
    assert not [p for p in kind_dir(rc).iterdir() if p.name.endswith(".tmp")]
    pointer = rc.read_pointer("last")
    assert pointer is not None and pointer.step == 5
    assert pointer.manifest_sha256 == hashlib.sha256(manifest_file(rc, 5).read_bytes()).hexdigest()
    assert stat.S_IMODE(rc.path.stat().st_mode) == 0o700


def test_manifests_chain_to_their_predecessor(tmp_path: Path) -> None:
    rc = make(tmp_path)
    first, second = save(rc, 1), save(rc, 2)
    assert first.prev_manifest_sha256 == GENESIS_DIGEST == first.prev_chain_sha256
    assert second.prev_chain_sha256 == first.chain_sha256
    assert (
        second.prev_manifest_sha256 == hashlib.sha256(manifest_file(rc, 1).read_bytes()).hexdigest()
    )


def test_resaving_the_same_step_with_the_same_bytes_is_a_no_op(tmp_path: Path) -> None:
    rc = make(tmp_path)
    first = save(rc, 3)
    again = save(rc, 3)
    assert again == first
    with pytest.raises(CheckpointExistsError):
        save(rc, 3, seed=99)
    assert valid_steps(rc) == [3]


def test_steps_only_increase_per_kind(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 10)
    with pytest.raises(CheckpointStepError):
        save(rc, 9)
    save(rc, 9, kind="best")  # another kind has its own timeline


@pytest.mark.parametrize("step", [-1, 2**63, True, 1.5, "7"])
def test_invalid_steps_are_refused(tmp_path: Path, step: Any) -> None:
    rc = make(tmp_path)
    with pytest.raises((CheckpointPathError, CheckpointError)):
        rc.save(
            kind="last",
            step=step,
            data=blob(),
            format="safetensors",
            versions={},
            dataset_hash=DATASET,
        )


@pytest.mark.parametrize("run_id", ["../x", "x", "", "01JBXQ5S7H8E9K2M3N4P5Q6R7S/..", "a" * 26])
def test_run_ids_are_validated_before_any_path_use(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(CheckpointPathError):
        RunCheckpoints(tmp_path / "ckpt", run_id, create=True)
    assert not (tmp_path / "ckpt").exists() or not any((tmp_path / "ckpt").iterdir())


def test_oversize_data_is_refused_and_leaves_nothing(tmp_path: Path) -> None:
    rc = make(tmp_path, max_bytes=100)
    with pytest.raises(CheckpointError):
        save(rc, 1)
    assert sorted(p.name for p in kind_dir(rc).iterdir()) == []


def test_disk_full_leaves_the_previous_checkpoint_restorable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    real_write = os.write

    def full(fd: int, data: Any) -> int:
        raise OSError(errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(os, "write", full)
    with pytest.raises(CheckpointError, match="space"):
        save(rc, 2)
    monkeypatch.setattr(os, "write", real_write)
    assert valid_steps(rc) == [1]
    assert sorted(p.name for p in kind_dir(rc).iterdir()) == [
        "step_000000001.manifest.json",
        "step_000000001.safetensors",
    ]


def test_a_failed_verification_after_save_removes_the_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    real = fsops.hash_regular

    def lying(dir_fd: int, name: str, *, max_bytes: int) -> tuple[str, int]:
        if "000000002" in name:
            return "0" * 64, 1
        return real(dir_fd, name, max_bytes=max_bytes)

    monkeypatch.setattr("cvpilot.ckpt.directory.fsops.hash_regular", lying)
    with pytest.raises(CheckpointError):
        save(rc, 2)
    monkeypatch.undo()
    assert valid_steps(rc) == [1]
    assert not data_file(rc, 2).exists() and not manifest_file(rc, 2).exists()
    assert rc.read_pointer("last").step == 1  # type: ignore[union-attr]


def test_concurrent_savers_are_serialised_by_the_directory_lock(tmp_path: Path) -> None:
    rc = make(tmp_path, lock_timeout=0.2)
    other = RunCheckpoints(tmp_path / "ckpt", rc.run_id, create=False, lock_timeout=0.2)
    with rc._locked():
        with pytest.raises(CheckpointBusyError):
            save(other, 1)
    save(other, 1)
    assert valid_steps(rc) == [1]


def test_clock_skew_never_changes_the_order(tmp_path: Path) -> None:
    rc = make(tmp_path)
    now = datetime.now(timezone.utc)
    save(rc, 1, created_at=now + timedelta(days=3650))
    save(rc, 2, created_at=now - timedelta(days=3650))
    save(rc, 3, created_at=now)
    assert valid_steps(rc) == [3, 2, 1]


# -- walking back --------------------------------------------------------------------------------


def test_walk_yields_newest_first_and_loads_bytes_once(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in (1, 2, 3):
        save(rc, step)
    got = list(rc.iter_valid("last", load=True))
    assert [v.manifest.step for v in got] == [3, 2, 1]
    assert got[0].data == data_file(rc, 3).read_bytes()
    assert not rc.skipped


def test_truncated_newest_file_is_skipped_and_counted(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    make_writable(data_file(rc, 2))
    with data_file(rc, 2).open("r+b") as handle:
        handle.truncate(10)
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {2: "size_mismatch"}


def test_bit_flip_with_the_same_size_is_a_digest_mismatch(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    path = data_file(rc, 2)
    make_writable(path)
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 0xFF
    path.write_bytes(bytes(raw))
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {2: "digest_mismatch"}


def test_header_damage_is_found_without_hashing_the_whole_file(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    path = data_file(rc, 1)
    make_writable(path)
    raw = bytearray(path.read_bytes())
    raw[:8] = b"\xff" * 8
    path.write_bytes(bytes(raw))
    assert valid_steps(rc) == []
    assert reasons(rc) == {1: "unloadable"}


def test_a_file_replaced_by_a_symlink_is_never_followed(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    real = tmp_path / "elsewhere.safetensors"
    real.write_bytes(data_file(rc, 2).read_bytes())  # even identical bytes must not count
    os.unlink(data_file(rc, 2))
    os.symlink(real, data_file(rc, 2))
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {2: "not_regular_file"}


def test_a_manifest_replaced_by_a_symlink_is_skipped(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    copy = tmp_path / "m.json"
    copy.write_bytes(manifest_file(rc, 2).read_bytes())
    os.unlink(manifest_file(rc, 2))
    os.symlink(copy, manifest_file(rc, 2))
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {2: "manifest_unreadable"}


def test_the_kind_directory_swapped_for_a_symlink_is_refused(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    elsewhere = tmp_path / "elsewhere"
    kind_dir(rc).rename(elsewhere)
    os.symlink(elsewhere, kind_dir(rc))
    assert valid_steps(rc) == []
    with pytest.raises(CheckpointError):
        save(rc, 2)
    assert sorted(p.name for p in elsewhere.iterdir()) == [
        "step_000000001.manifest.json",
        "step_000000001.safetensors",
    ]


def test_a_data_file_without_a_manifest_is_never_loaded(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    data_file(rc, 9).write_bytes(blob(9))
    assert valid_steps(rc) == [1]
    entries = {e.name: e for e in rc.inspect("last")}
    assert entries["step_000000009.safetensors"].reason == "no_manifest"


def test_forged_newest_manifest_fails_the_pointer_digest(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    evil = blob(666)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(evil)
    forge_manifest(rc, 2, sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {2: "pointer_mismatch"}


def test_forged_manifest_and_pointer_are_caught_by_the_store_anchor(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    genuine = save(rc, 2)
    digest = hashlib.sha256(manifest_file(rc, 2).read_bytes()).hexdigest()
    evil = blob(666)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(evil)
    forged = forge_manifest(rc, 2, sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    forge_pointer(rc, forged, 2)
    anchored = RunCheckpoints(
        tmp_path / "ckpt",
        rc.run_id,
        anchor=lambda run, kind, step: digest if (kind, step) == ("last", genuine.step) else None,
    )
    assert valid_steps(anchored) == [1]
    assert reasons(anchored) == {2: "store_mismatch"}


def test_a_forged_older_manifest_breaks_the_link_its_successor_vouches_for(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in (1, 2, 3):
        save(rc, step)
    evil = blob(666)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(evil)
    forge_manifest(rc, 2, sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    assert valid_steps(rc) == [3, 1]
    assert reasons(rc) == {2: "link_mismatch"}


def test_a_gap_left_by_retention_is_not_corruption(tmp_path: Path) -> None:
    """Retention of the best checkpoints keeps non-adjacent steps: the older one must stay
    usable although the manifest above it vouches for a predecessor that was deleted."""
    rc = make(tmp_path)
    for step in (1, 2, 3, 4):
        save(rc, step)
    for step in (2, 3):
        for name in (f"step_{step:09d}.safetensors", f"step_{step:09d}.manifest.json"):
            (kind_dir(rc) / name).unlink()
    assert valid_steps(rc) == [4, 1]
    assert reasons(rc) == {}


def test_a_replacement_for_the_vouched_predecessor_is_still_refused(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in (1, 2, 3):
        save(rc, step)
    evil = blob(666)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(evil)
    forge_manifest(rc, 2, sha256=hashlib.sha256(evil).hexdigest(), size_bytes=len(evil))
    for name in ("step_000000001.safetensors", "step_000000001.manifest.json"):
        (kind_dir(rc) / name).unlink()  # the forgery is now the only candidate below step 3
    assert valid_steps(rc) == [3]
    assert reasons(rc) == {2: "link_mismatch"}


def test_a_planted_older_manifest_below_the_start_of_the_chain_is_refused(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in (5, 6):
        save(rc, step)
    other = make(tmp_path / "other", run_id=rc.run_id)
    save(other, 3)
    for name in ("step_000000003.safetensors", "step_000000003.manifest.json"):
        (kind_dir(rc) / name).write_bytes((kind_dir(other) / name).read_bytes())
    assert valid_steps(rc) == [6, 5]
    assert reasons(rc) == {3: "link_mismatch"}


def test_a_manifest_that_disagrees_with_its_name_or_run_is_skipped(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    make_writable(manifest_file(rc, 2))
    fields = json.loads(manifest_file(rc, 2).read_text())
    fields["step"] = 7
    fields["path"] = "last/step_000000007.safetensors"
    manifest_file(rc, 2).write_text(json.dumps(fields))
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {2: "name_mismatch"}


def test_garbage_and_oversize_manifests_are_skipped(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    make_writable(manifest_file(rc, 2))
    manifest_file(rc, 2).write_bytes(b"{not json")
    save(rc, 3)
    make_writable(manifest_file(rc, 3))
    manifest_file(rc, 3).write_bytes(b" " * (70 * 1024))
    assert valid_steps(rc) == [1]
    assert set(reasons(rc).values()) <= {"manifest_invalid", "manifest_unreadable"}


def test_a_lost_or_garbled_pointer_does_not_hide_valid_checkpoints(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    make_writable(rc.path / "LATEST")
    (rc.path / "LATEST").write_text("garbage")
    assert valid_steps(rc) == [2, 1]
    os.unlink(rc.path / "LATEST")
    assert valid_steps(rc) == [2, 1]
    assert rc.read_pointer("last") is None


def test_killed_between_rename_and_pointer_update_still_finds_the_new_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rc = make(tmp_path)
    save(rc, 1)

    def die(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(RunCheckpoints, "_write_pointer", die)
    with pytest.raises(KeyboardInterrupt):
        save(rc, 2)
    monkeypatch.undo()
    assert rc.read_pointer("last").step == 1  # type: ignore[union-attr]
    assert valid_steps(rc) == [2, 1]  # chained to the pointer target, so it is trusted


def test_an_unchained_manifest_newer_than_the_pointer_is_not_trusted(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    other = make(tmp_path / "other", run_id=rc.run_id)
    evil = save(other, 9)
    assert evil.prev_manifest_sha256 == GENESIS_DIGEST
    for name in (f"step_{9:09d}.safetensors", f"step_{9:09d}.manifest.json"):
        make_writable(kind_dir(other) / name)
        (kind_dir(rc) / name).write_bytes((kind_dir(other) / name).read_bytes())
    assert valid_steps(rc) == [1]
    assert reasons(rc) == {9: "unanchored"}


def test_orphans_from_a_killed_save_are_ignored_then_cleaned_by_the_next_save(
    tmp_path: Path,
) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    (kind_dir(rc) / ".step_000000002.safetensors.123.abcdef.tmp").write_bytes(b"half")
    # a manifest whose data file never got renamed into place
    fields = json.loads(manifest_file(rc, 1).read_text())
    fields.update(step=2, path="last/step_000000002.safetensors")
    (kind_dir(rc) / "step_000000002.manifest.json").write_text(json.dumps(fields))
    assert valid_steps(rc) == [1]
    save(rc, 3)
    assert sorted(p.name for p in kind_dir(rc).iterdir()) == [
        "step_000000001.manifest.json",
        "step_000000001.safetensors",
        "step_000000003.manifest.json",
        "step_000000003.safetensors",
    ]
    assert valid_steps(rc) == [3, 1]


def test_walk_is_bounded_by_the_number_of_entries(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    for i in range(10, 25):
        (kind_dir(rc) / f"step_{i:09d}.manifest.json").write_text("{}")
    small = RunCheckpoints(tmp_path / "ckpt", rc.run_id, max_entries=10)
    with pytest.raises(CheckpointError):
        list(small.iter_valid("last"))


def test_inspect_reports_every_entry(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    save(rc, 2)
    make_writable(data_file(rc, 2))
    data_file(rc, 2).write_bytes(b"x")
    status = {e.step: (e.status, e.reason) for e in rc.inspect("last")}
    assert status == {2: ("corrupt", "size_mismatch"), 1: ("valid", None)}
    assert [e.step for e in rc.inspect("last", check_data=False)] == [2, 1]


# -- retention -----------------------------------------------------------------------------------


def names(rc: RunCheckpoints, kind: str = "last") -> list[str]:
    return sorted(p.name for p in kind_dir(rc, kind).iterdir() if p.name.endswith("safetensors"))


def test_keep_last_removes_only_the_oldest_valid_ones(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in range(1, 6):
        save(rc, step)
    result = rc.prune("last", keep=2)
    assert [m.step for m in result.deleted] == [1, 2, 3]
    assert names(rc) == ["step_000000004.safetensors", "step_000000005.safetensors"]
    assert sorted(p.name for p in kind_dir(rc).iterdir() if p.name.endswith(".manifest.json")) == [
        "step_000000004.manifest.json",
        "step_000000005.manifest.json",
    ]
    assert valid_steps(rc) == [5, 4]


def test_retention_counts_only_valid_checkpoints_so_a_corrupt_newest_never_costs_the_last_good_one(
    tmp_path: Path,
) -> None:
    rc = make(tmp_path)
    for step in range(1, 5):
        save(rc, step)
    make_writable(data_file(rc, 4))
    data_file(rc, 4).write_bytes(b"x")
    rc.prune("last", keep=1)
    assert 3 in valid_steps(rc)
    assert names(rc)[-2:] == ["step_000000003.safetensors", "step_000000004.safetensors"]


def test_the_last_valid_checkpoint_is_never_deleted(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in range(1, 4):
        save(rc, step)
    for step in (2, 3):
        make_writable(data_file(rc, step))
        data_file(rc, step).write_bytes(b"x")
    result = rc.prune("last", keep=1)
    assert result.deleted == []
    assert valid_steps(rc) == [1]


def test_keep_zero_is_refused(tmp_path: Path) -> None:
    rc = make(tmp_path)
    save(rc, 1)
    with pytest.raises(CheckpointError):
        rc.prune("last", keep=0)


def test_nothing_is_deleted_until_the_newest_is_confirmed_on_the_remote(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in range(1, 5):
        save(rc, step)
    held = rc.prune("last", keep=1, is_confirmed=lambda m: False)
    assert held.deleted == [] and held.held_back is True
    confirmed = rc.prune("last", keep=1, is_confirmed=lambda m: m.step == 4)
    assert [m.step for m in confirmed.deleted] == [1, 2, 3]


def test_dry_run_deletes_nothing(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in range(1, 4):
        save(rc, step)
    result = rc.prune("last", keep=1, dry_run=True)
    assert [m.step for m in result.deleted] == [1, 2]
    assert valid_steps(rc) == [3, 2, 1]


@pytest.mark.parametrize(("mode", "deleted"), [("max", {1, 3}), ("min", {2})])
def test_keep_best_keeps_the_top_by_metric_and_the_newest(
    tmp_path: Path, mode: str, deleted: set[int]
) -> None:
    rc = make(tmp_path)
    metrics = {1: 0.1, 2: 0.9, 3: 0.2, 4: 0.8}
    for step, value in metrics.items():
        save(rc, step, kind="best", metric_value=value, metric_mode=mode)
    result = rc.prune("best", keep=2, mode=mode)  # type: ignore[arg-type]
    assert {m.step for m in result.deleted} == deleted
    assert 4 in {m.step for m in result.kept}  # the newest valid is always kept


def test_a_run_without_checkpoints_walks_to_nothing(tmp_path: Path) -> None:
    rc = RunCheckpoints(tmp_path / "ckpt", "01JBXQ5S7H8E9K2M3N4P5Q6R7S")
    assert valid_steps(rc) == [] and rc.inspect("last") == []
    assert rc.read_pointer("last") is None
    assert rc.prune("last", keep=1).deleted == []


def test_uploaded_markers_round_trip_and_are_bound_to_the_manifest(tmp_path: Path) -> None:
    rc = make(tmp_path)
    manifest = save(rc, 1)
    assert not rc.is_uploaded(manifest)
    rc.mark_uploaded(manifest, "memory://x/y")
    assert rc.is_uploaded(manifest)
    other = save(rc, 2)
    assert not rc.is_uploaded(other)
    marker = kind_dir(rc) / "step_000000001.remote.json"
    make_writable(marker)
    marker.write_text(json.dumps({"uri": "memory://x/y", "manifest_sha256": "0" * 64}))
    assert not rc.is_uploaded(manifest)


def test_prune_removes_the_markers_of_what_it_deletes(tmp_path: Path) -> None:
    rc = make(tmp_path)
    for step in (1, 2):
        mark = save(rc, step)
        rc.mark_uploaded(mark, "memory://x/y")
    rc.prune("last", keep=1)
    assert sorted(p.name for p in kind_dir(rc).iterdir() if "000000001" in p.name) == []
