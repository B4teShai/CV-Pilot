"""Adopting a framework file scans it first: a file that resume would refuse is never stored.

A real YOLO ``last.pt`` pickles the model object. Before this check ``adopt`` stored it (and
logged it as verified) and only ``resume_path`` refused it, at restart, as a quiet ``None``."""

from __future__ import annotations

import pickle
from collections import OrderedDict
from pathlib import Path
from typing import Any

import pytest
from ckpt_support import run_meta
from zip_support import framework_archive

from cvpilot import runs
from cvpilot.ckpt import resume, serializers
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import CheckpointCorruptError, CheckpointError

pytestmark = pytest.mark.usefixtures("new_enough_framework")


class ModelObject:
    """Stands in for a framework model class that a checkpoint pickles by reference."""

    def __init__(self) -> None:
        self.w = [1.0]


@pytest.fixture
def new_enough_framework(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")


def start(tmp_path: Path) -> tuple[runs.RunHandle, CheckpointManager]:
    meta, repro = run_meta("p")
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    mgr = CheckpointManager(
        run, dir=tmp_path / "ckpt", every_n_steps=1, allow_pt_format=True,
        install_signal_handlers=False,
    )  # fmt: skip
    return run, mgr


def archive(obj: Any) -> bytes:
    return framework_archive(
        {"archive/data.pkl": pickle.dumps(obj, protocol=2), "archive/version": b"3\n"}
    )


def test_a_pickled_model_object_is_refused_by_adopt_and_nothing_is_stored(tmp_path: Path) -> None:
    source = tmp_path / "last.pt"
    source.write_bytes(archive({"model": ModelObject(), "epoch": 1}))
    run, mgr = start(tmp_path)
    with pytest.raises(CheckpointCorruptError, match="ModelObject") as info:
        mgr.adopt(source, step=1)
    assert info.value.reason == "unsafe_archive"
    assert mgr.list() == []
    assert not any(p.suffix == ".pt" for p in (tmp_path / "ckpt").rglob("*"))
    run.finish()


def test_a_plain_state_in_the_framework_layout_is_adopted_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "last.ckpt"
    source.write_bytes(archive(OrderedDict(epoch=1, lr=[0.1])))
    run, mgr = start(tmp_path)
    manifest = mgr.adopt(source, step=1)
    assert manifest.format == "pt"
    mgr.close()
    run.finish("preempted")
    point = resume.find_resume_point(
        "p", checkpoint_dir=tmp_path / "ckpt", store_dir=tmp_path / "store"
    )
    assert point is not None and point.manifest.step == 1


def test_adopt_refuses_when_the_framework_is_too_old_to_load_it_safely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 5))
    source = tmp_path / "last.ckpt"
    source.write_bytes(archive(OrderedDict(epoch=1)))
    run, mgr = start(tmp_path)
    with pytest.raises(CheckpointCorruptError) as info:
        mgr.adopt(source, step=1)
    assert info.value.reason == "unsafe_loader" and mgr.list() == []
    run.finish()


def test_a_set_pickled_by_the_older_protocol_passes_the_scan() -> None:
    """Protocol 2 names the builtin module ``__builtin__``; the framework's loader accepts it."""
    data = pickle.dumps(OrderedDict(seen={1, 2}), protocol=2)
    assert b"__builtin__" in data
    serializers.scan_pt_archive(framework_archive({"archive/data.pkl": data}))


def planted(tmp_path: Path, data: bytes) -> Any:
    """A directory whose ``pt`` checkpoint was written below the manager (no adopt-time scan)."""
    from ckpt_support import make

    rc = make(tmp_path, allow_pt=True)
    rc.save(
        kind="last", step=1, data=data, format="pt", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    return rc


def test_verify_scans_a_pt_file_so_it_agrees_with_resume(tmp_path: Path) -> None:
    """``inspect`` (``cvpilot ckpt verify``) used to check only size and digest: a planted file
    that resume refuses showed as valid."""
    bad = planted(tmp_path / "bad", archive({"model": ModelObject()}))
    assert [(e.status, e.reason) for e in bad.inspect("last")] == [("corrupt", "unsafe_archive")]
    good = planted(tmp_path / "good", archive(OrderedDict(epoch=1)))
    assert [(e.status, e.reason) for e in good.inspect("last")] == [("valid", None)]


def test_verify_does_not_need_the_framework_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The loader version matters to the machine that resumes, not to one that only verifies."""
    rc = planted(tmp_path, archive(OrderedDict(epoch=1)))
    monkeypatch.setattr(serializers, "framework_version", lambda: None)
    assert [(e.status, e.reason) for e in rc.inspect("last")] == [("valid", None)]


def test_a_file_that_is_not_a_zip_archive_is_refused_with_a_message_that_says_what_is_adopted(
    tmp_path: Path,
) -> None:
    for name, data in (
        ("model.safetensors", b"\x10\x00\x00\x00\x00\x00\x00\x00{}" + b" " * 14),
        ("raw.pt", pickle.dumps({"a": 1})),
    ):
        source = tmp_path / name
        source.write_bytes(data)
        run, mgr = start(tmp_path / name.split(".")[0])
        with pytest.raises(CheckpointError, match="not adopted"):
            mgr.adopt(source, step=1)
        run.finish()
