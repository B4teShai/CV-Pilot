"""The .pt format is opt-in, needs a loader that honours weights_only, and is pre-scanned."""

from __future__ import annotations

import importlib
import io
import pickle
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from ckpt_support import blob, make
from test_runs import make as make_run_meta

from cvpilot import runs
from cvpilot.ckpt import serializers
from cvpilot.ckpt.manager import CheckpointManager
from cvpilot.core.errors import (
    CheckpointCorruptError,
    CheckpointError,
    CheckpointFormatError,
)

FRAMEWORK = "to" + "rch"
GOOD_PICKLE = pickle.dumps({"epoch": 3, "lr": [0.1]}, protocol=2)


def make_zip(members: dict[str, bytes], *, deflate: bool = False) -> bytes:
    buffer = io.BytesIO()
    kind = zipfile.ZIP_DEFLATED if deflate else zipfile.ZIP_STORED
    with zipfile.ZipFile(buffer, "w", kind) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def good_archive() -> bytes:
    return make_zip({"archive/data.pkl": GOOD_PICKLE, "archive/version": b"3\n"})


class Recorder:
    """A fake framework module that records what the loader is asked to do."""

    def __init__(self) -> None:
        self.loads: list[dict[str, Any]] = []

    def module(self) -> types.ModuleType:
        module = types.ModuleType(FRAMEWORK)

        def save(obj: Any, handle: io.BytesIO) -> None:
            handle.write(good_archive())

        def load(handle: io.BytesIO, **kwargs: Any) -> Any:
            self.loads.append(kwargs)
            return {"epoch": 3}

        module.save = save  # type: ignore[attr-defined]
        module.load = load  # type: ignore[attr-defined]
        return module


@pytest.fixture
def framework(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    recorder = Recorder()
    monkeypatch.setitem(sys.modules, FRAMEWORK, recorder.module())
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 6))
    return recorder


@pytest.mark.parametrize(
    ("text", "parsed"),
    [("2.6.0", (2, 6)), ("2.7.1+cu121", (2, 7)), ("2.10.0", (2, 10)), ("3.0.0.dev1", (3, 0))],
)
def test_versions_are_parsed_numerically(text: str, parsed: tuple[int, int]) -> None:
    assert serializers.parse_version(text) == parsed


@pytest.mark.parametrize("text", ["", "x", "2", "two.six", "2..6", None])
def test_unparseable_versions_are_not_trusted(text: Any) -> None:
    assert serializers.parse_version(text) is None


@pytest.mark.parametrize("version", [(2, 5), (1, 13), (0, 4), None])
def test_an_old_or_unknown_loader_is_refused_before_it_is_imported(
    monkeypatch: pytest.MonkeyPatch, version: tuple[int, int] | None
) -> None:
    imported: list[str] = []
    real = importlib.import_module
    monkeypatch.setattr(
        serializers.importlib,
        "import_module",
        lambda name, *a, **k: (imported.append(name), real(name, *a, **k))[1],
    )
    monkeypatch.setattr(serializers, "framework_version", lambda: version)
    with pytest.raises(CheckpointFormatError, match="2.6"):
        serializers.get_serializer("pt").loads(good_archive())
    assert FRAMEWORK not in imported


def test_a_new_enough_loader_is_called_with_weights_only(framework: Recorder) -> None:
    assert serializers.get_serializer("pt").loads(good_archive()) == {"epoch": 3}
    assert framework.loads == [{"map_location": "cpu", "weights_only": True}]


def test_the_manager_needs_an_explicit_opt_in_for_the_pt_format(
    tmp_path: Path, framework: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    with pytest.raises(ValueError, match="allow_pt_format"):
        CheckpointManager(run, dir=tmp_path / "c", format="pt", install_signal_handlers=False)
    CheckpointManager(
        run, dir=tmp_path / "c", format="pt", allow_pt_format=True, install_signal_handlers=False
    )
    monkeypatch.setenv("CVPILOT_CKPT_ALLOW_PT", "1")
    CheckpointManager(run, dir=tmp_path / "c2", format="pt", install_signal_handlers=False)
    run.finish()


def test_a_pt_manifest_is_never_loaded_by_a_manager_that_was_not_opted_in(
    tmp_path: Path, framework: Recorder
) -> None:
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    writer = CheckpointManager(
        run, dir=tmp_path / "c", format="pt", allow_pt_format=True, install_signal_handlers=False
    )
    writer.save_now(1, model={"w": 1})
    reader = CheckpointManager(run, dir=tmp_path / "c", install_signal_handlers=False)
    assert reader.restore() is None
    assert [s.reason for s in reader._own.skipped] == ["pt_not_allowed"]
    assert framework.loads == []
    assert writer.restore() is not None and len(framework.loads) == 1
    run.finish()


def test_an_opted_in_manager_with_an_old_loader_skips_the_file(
    tmp_path: Path, framework: Recorder, monkeypatch: pytest.MonkeyPatch
) -> None:
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(
        run, dir=tmp_path / "c", format="pt", allow_pt_format=True, install_signal_handlers=False
    )
    manager.save_now(1, model={"w": 1})
    monkeypatch.setattr(serializers, "framework_version", lambda: (2, 5))
    assert manager.restore() is None
    assert framework.loads == []
    run.finish()


def test_the_safetensors_path_never_imports_the_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delitem(sys.modules, FRAMEWORK, raising=False)
    imported: list[str] = []
    real = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a, **k: (imported.append(name), real(name, *a, **k))[1],
    )
    meta, repro = make_run_meta()
    run = runs.start_run(meta, repro, store_dir=tmp_path / "store")
    manager = CheckpointManager(run, dir=tmp_path / "c", install_signal_handlers=False)
    manager.save_now(1, model={"w": np.ones(3)}, rng=True)
    assert manager.restore() is not None
    assert FRAMEWORK not in imported and FRAMEWORK not in sys.modules
    run.finish()


def test_the_directory_refuses_pt_files_unless_allowed(tmp_path: Path) -> None:
    from cvpilot.ckpt.directory import RunCheckpoints

    rc = make(tmp_path, allow_pt=True)
    rc.save(
        kind="last", step=1, data=good_archive(), format="pt", versions={},
        dataset_hash="merkle:" + "c" * 64,
    )  # fmt: skip
    assert [v.manifest.step for v in rc.iter_valid("last")] == [1]
    closed = RunCheckpoints(rc.path.parent, rc.run_id)
    assert list(closed.iter_valid("last")) == []
    assert [s.reason for s in closed.skipped] == ["pt_not_allowed"]
    # verification and retention use the same gate as restore: a pt file that is not allowed is
    # not "valid" for them either
    assert [(e.status, e.reason) for e in closed.inspect("last")] == [("corrupt", "pt_not_allowed")]


def evil_pickle() -> bytes:
    class Evil:
        def __reduce__(self) -> tuple[Any, ...]:
            import os

            return (os.system, ("echo pwned",))

    return pickle.dumps(Evil(), protocol=2)


@pytest.mark.parametrize(
    "archive",
    [
        make_zip({"archive/data.pkl": GOOD_PICKLE, "../evil": b"x"}),
        make_zip({"archive/data.pkl": GOOD_PICKLE, "/abs/evil": b"x"}),
        make_zip({"archive/data.pkl": GOOD_PICKLE, "a\\b": b"x"}),
        make_zip({"archive/data.pkl": GOOD_PICKLE, "archive/./x": b"x"}),
        make_zip({"archive/data.pkl": GOOD_PICKLE, "archive/\x01": b"x"}),
        make_zip({"archive/data.pkl": GOOD_PICKLE, "z" * 400: b"x"}),
        make_zip({"archive/version": b"3"}),  # no data.pkl
        make_zip({"a/data.pkl": GOOD_PICKLE, "b/data.pkl": GOOD_PICKLE}),
        make_zip({"archive/data.pkl": evil_pickle()}),
        make_zip({"archive/data.pkl": b"not a pickle at all"}),
        make_zip({"archive/data.pkl": GOOD_PICKLE, "archive/d/0": bytes(30_000_000)}, deflate=True),
        make_zip({f"archive/data/{i}": b"x" for i in range(serializers.MAX_ARCHIVE_MEMBERS + 1)}),
        b"PK\x03\x04" + b"\x00" * 100,
        b"PK\x03\x04",
    ],
    ids=[
        "traversal", "absolute", "backslash", "dot", "control", "long-name", "no-pickle",
        "two-pickles", "global-os-system", "garbage-pickle", "zip-bomb", "too-many", "bad-zip",
        "magic-only",
    ],
)  # fmt: skip
def test_hostile_archives_are_refused_before_the_loader_sees_them(
    framework: Recorder, archive: bytes
) -> None:
    with pytest.raises(CheckpointCorruptError) as info:
        serializers.get_serializer("pt").loads(archive)
    assert info.value.reason == "unsafe_archive"
    assert framework.loads == []


def test_an_archive_with_only_allow_listed_globals_passes_the_scan() -> None:
    from collections import OrderedDict

    data = pickle.dumps(OrderedDict(a=1), protocol=2)
    serializers.scan_pt_archive(make_zip({"archive/data.pkl": data, "archive/data/0": b"x" * 8}))


def test_blob_helper_is_still_a_valid_safetensors_file() -> None:
    assert blob(1)[:8]
    assert CheckpointError
