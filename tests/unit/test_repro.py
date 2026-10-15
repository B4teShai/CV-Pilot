from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from cvpilot.core.hashing import new_run_id
from cvpilot.core.redact import REDACTED
from cvpilot.repro import capture_repro


@pytest.fixture(autouse=True)
def _cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.chdir(tmp_path)


def make_dataset(root: Path) -> Path:
    (root / "img").mkdir(parents=True)
    (root / "img" / "a.jpg").write_bytes(b"1")
    return root


def test_capture_outside_git_and_without_gpu_never_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr("shutil.which", lambda _n: None)
    ds = make_dataset(tmp_path / "ds")
    got = capture_repro(new_run_id(), str(ds), {"lr": 0.1}, 7, argv=["train.py", "--epochs", "2"])
    r = got.repro
    assert (r.git_commit, r.git_dirty, r.git_diff_sha256, r.cuda, r.gpu_names) == (
        None,
        False,
        None,
        None,
        [],
    )
    assert got.git_diff is None and got.host
    assert r.dataset_hash.startswith("merkle:") and r.dataset_uri == str(ds)
    assert r.seed == 7 and r.config == {"lr": 0.1} and r.command_line == "train.py --epochs 2"
    assert len(r.packages_lock_sha256) == 64


def test_dataset_uri_with_datacopilot_manifest_uses_dcp(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path / "ds")
    (ds / "manifest.json").write_text(json.dumps({"version": "ab" * 16}))
    got = capture_repro(new_run_id(), str(ds), {}, None, argv=["t.py"])
    assert got.repro.dataset_hash == "dcp:" + "ab" * 16


def test_no_dataset_is_recorded_as_the_empty_set() -> None:
    got = capture_repro(new_run_id(), None, {}, None, argv=["t.py"])
    assert got.repro.dataset_uri == ""
    assert got.repro.dataset_hash == "merkle:" + hashlib.sha256(b"").hexdigest()


def test_missing_dataset_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        capture_repro(new_run_id(), str(tmp_path / "nope"), {}, None, argv=["t.py"])


def test_dirty_tree_returns_diff_and_its_hash(tmp_path: Path) -> None:
    import subprocess

    def g(*a: str) -> None:
        subprocess.run(  # noqa: S603
            ["git", "-c", "user.name=t", "-c", "user.email=t@e.invalid", *a],  # noqa: S607
            check=True,
            capture_output=True,
        )

    g("init", "-q")
    (tmp_path / "f.txt").write_text("1\n")
    g("add", "f.txt")
    g("commit", "-q", "-m", "i")
    (tmp_path / "f.txt").write_text("2\n")
    got = capture_repro(new_run_id(), None, {}, None, argv=["t.py"])
    assert got.repro.git_dirty and got.git_diff is not None
    assert got.repro.git_diff_sha256 == hashlib.sha256(got.git_diff).hexdigest()


def test_config_secrets_are_redacted_in_the_stored_repro() -> None:
    got = capture_repro(new_run_id(), None, {"api_key": "k", "lr": 1}, None, argv=["t.py"])
    assert got.repro.config == {"api_key": REDACTED, "lr": 1}


def test_config_accepts_paths_and_non_finite_floats() -> None:
    got = capture_repro(
        new_run_id(), None, {"p": Path("/x"), "nan": float("nan"), "t": (1, 2)}, None, argv=["t.py"]
    )
    assert got.repro.config == {"p": "/x", "nan": "nan", "t": [1, 2]}


def test_default_stores_the_redacted_command_line() -> None:
    got = capture_repro(
        new_run_id(), None, {}, None, argv=["train.py", "--token=abc123", "--lr", "0.1"]
    )
    assert got.repro.command_line == f"train.py '--token={REDACTED}' --lr 0.1"


def test_setting_false_stores_no_argument_text_through_the_real_capture_path() -> None:
    argv = ["/opt/proj/train.py", "--token=abc123", "--data", "/very/private", "--lr", "0.1"]
    got = capture_repro(new_run_id(), None, {}, None, argv=argv, store_command_line=False)
    stored = got.repro.model_dump_json()
    assert got.repro.command_line == "train.py [5 arguments omitted]"
    for text in ("abc123", "/very/private", "--lr", "--data", "/opt/proj"):
        assert text not in stored


def test_store_command_line_comes_from_settings_not_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cvpilot.settings import load_settings

    s = load_settings({"CVPILOT_STORE_COMMAND_LINE": "false"})
    got = capture_repro(
        new_run_id(),
        None,
        {},
        None,
        argv=["t.py", "--lr", "1"],
        store_command_line=s.store_command_line,
    )
    assert "--lr" not in got.repro.command_line
