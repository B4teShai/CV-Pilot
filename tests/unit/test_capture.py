from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from cvpilot.capture import env as capture
from cvpilot.core.redact import REDACTED
from cvpilot.registry import capturers


def git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", *args],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
    )


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    r = tmp_path / "repo"
    r.mkdir()
    git(r, "init", "-q")
    (r / "a.txt").write_text("one\n")
    git(r, "add", "a.txt")
    git(r, "commit", "-q", "-m", "init")
    return r


# --- git ------------------------------------------------------------------


def test_clean_repo(repo: Path) -> None:
    got = capture.capture_git(cwd=repo)
    assert re.fullmatch(r"[0-9a-f]{40}", got["git_commit"])
    assert got["git_dirty"] is False and got["git_diff"] is None


def test_dirty_repo_captures_the_diff(repo: Path) -> None:
    (repo / "a.txt").write_text("two\n")
    (repo / "untracked.txt").write_text("x")
    got = capture.capture_git(cwd=repo)
    assert got["git_dirty"] is True
    assert b"-one" in got["git_diff"] and b"+two" in got["git_diff"]
    assert b"untracked" not in got["git_diff"]


def test_untracked_files_alone_do_not_make_the_tree_dirty(repo: Path) -> None:
    (repo / "untracked.txt").write_text("x")
    assert capture.capture_git(cwd=repo)["git_dirty"] is False


def test_outside_a_repository_everything_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    plain = tmp_path / "plain"
    plain.mkdir()
    assert capture.capture_git(cwd=plain) == {
        "git_commit": None,
        "git_dirty": False,
        "git_diff": None,
    }


def test_repository_without_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    r = tmp_path / "empty"
    r.mkdir()
    git(r, "init", "-q")
    got = capture.capture_git(cwd=r)
    assert got["git_commit"] is None and got["git_dirty"] is False


def test_missing_git_binary_and_missing_cwd(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert capture.capture_git(cwd=repo / "nope")["git_commit"] is None
    monkeypatch.setattr(capture.shutil, "which", lambda _name: None)
    assert capture.capture_git(cwd=repo) == {
        "git_commit": None,
        "git_dirty": False,
        "git_diff": None,
    }


def test_oversized_diff_is_flagged_dirty_but_not_stored(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(capture, "MAX_DIFF_BYTES", 10)
    (repo / "a.txt").write_text("two\n" * 50)
    got = capture.capture_git(cwd=repo)
    assert got["git_dirty"] is True and got["git_diff"] is None


def test_git_failures_are_none_not_errors(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired("git", 1)

    monkeypatch.setattr(capture.subprocess, "run", boom)
    assert capture.capture_git(cwd=repo)["git_commit"] is None


def test_git_runs_without_repo_config_hooks(repo: Path) -> None:
    marker = repo / "ran"
    hook = repo / "fsmon.sh"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n")
    hook.chmod(0o755)
    git(repo, "config", "core.fsmonitor", str(hook))
    capture.capture_git(cwd=repo)
    assert not marker.exists()


# --- lockfile hash --------------------------------------------------------


def test_lockfile_hash_prefers_uv_then_poetry(tmp_path: Path) -> None:
    (tmp_path / "poetry.lock").write_bytes(b"poetry")
    assert (
        capture.capture_packages(cwd=tmp_path)["packages_lock_sha256"]
        == hashlib.sha256(b"poetry").hexdigest()
    )
    (tmp_path / "uv.lock").write_bytes(b"uv")
    assert (
        capture.capture_packages(cwd=tmp_path)["packages_lock_sha256"]
        == hashlib.sha256(b"uv").hexdigest()
    )


def test_lockfile_is_found_up_to_the_repository_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "proj"
    (root / ".git").mkdir(parents=True)
    (root / "uv.lock").write_bytes(b"root")
    sub = root / "src" / "pkg"
    sub.mkdir(parents=True)
    assert (
        capture.capture_packages(cwd=sub)["packages_lock_sha256"]
        == hashlib.sha256(b"root").hexdigest()
    )


def test_lockfile_above_the_repository_root_is_not_used(tmp_path: Path) -> None:
    (tmp_path / "uv.lock").write_bytes(b"outer")
    inner = tmp_path / "inner"
    (inner / ".git").mkdir(parents=True)
    got = capture.capture_packages(cwd=inner)["packages_lock_sha256"]
    assert got != hashlib.sha256(b"outer").hexdigest()


def test_without_a_lockfile_the_installed_set_is_hashed(tmp_path: Path) -> None:
    a = capture.capture_packages(cwd=tmp_path)["packages_lock_sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", a)
    assert capture.capture_packages(cwd=tmp_path)["packages_lock_sha256"] == a


# --- python, host, gpu ------------------------------------------------------


def test_python_and_host() -> None:
    assert capture.capture_python() == {"python_version": platform.python_version()}
    host = capture.capture_host()["host"]
    assert host and host == socket.gethostname()
    assert "@" not in host


def test_host_fallback_when_hostname_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capture.socket, "gethostname", lambda: "")
    assert capture.capture_host() == {"host": "unknown"}


def block_pynvml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pynvml", None)  # import raises ImportError


def test_no_gpu_tooling_gives_none_and_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    block_pynvml(monkeypatch)
    monkeypatch.setattr(capture.shutil, "which", lambda _name: None)
    assert capture.capture_gpu() == {"cuda": None, "gpu_names": []}


def test_nvidia_smi_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    block_pynvml(monkeypatch)
    monkeypatch.setattr(capture.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(cmd: list[str], **_k: Any) -> bytes | None:
        if "--query-gpu=name,driver_version" in " ".join(cmd):
            return b"NVIDIA A10G, 550.54.15\nNVIDIA A10G, 550.54.15\n"
        return b"| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   CUDA Version: 12.4 |\n"

    monkeypatch.setattr(capture, "_run", fake_run)
    assert capture.capture_gpu() == {
        "cuda": "12.4 / 550.54.15",
        "gpu_names": ["NVIDIA A10G", "NVIDIA A10G"],
    }


def test_nvidia_smi_failure_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    block_pynvml(monkeypatch)
    monkeypatch.setattr(capture.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(capture, "_run", lambda *_a, **_k: None)
    assert capture.capture_gpu() == {"cuda": None, "gpu_names": []}


def test_pynvml_is_used_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("pynvml")
    fake.nvmlInit = lambda: None  # type: ignore[attr-defined]
    fake.nvmlShutdown = lambda: None  # type: ignore[attr-defined]
    fake.nvmlDeviceGetCount = lambda: 2  # type: ignore[attr-defined]
    fake.nvmlDeviceGetHandleByIndex = lambda i: i  # type: ignore[attr-defined]
    fake.nvmlDeviceGetName = lambda h: b"RTX 4090"  # type: ignore[attr-defined]
    fake.nvmlSystemGetDriverVersion = lambda: b"550.54"  # type: ignore[attr-defined]
    fake.nvmlSystemGetCudaDriverVersion = lambda: 12040  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    assert capture.capture_gpu() == {"cuda": "12.4 / 550.54", "gpu_names": ["RTX 4090"] * 2}


def test_pynvml_errors_fall_back_quietly(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = ModuleType("pynvml")

    def fail() -> None:
        raise RuntimeError("NVML Shared Library Not Found")

    fake.nvmlInit = fail  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    monkeypatch.setattr(capture.shutil, "which", lambda _name: None)
    assert capture.capture_gpu() == {"cuda": None, "gpu_names": []}


def test_capture_never_imports_the_framework_or_pynvml() -> None:
    code = (
        "import sys; from cvpilot.capture import env as c;"
        "c.capture_gpu(); c.capture_git(); c.capture_packages();"
        "assert 'to' + 'rch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)  # noqa: S603


# --- command line -----------------------------------------------------------


def test_command_line_is_redacted_and_shell_quoted() -> None:
    got = capture.capture_command(argv=["train.py", "--token=abc123", "--name", "my run"])
    line = got["command_line"]
    assert "abc123" not in line and REDACTED in line
    assert "'my run'" in line and line.startswith("train.py")


def test_command_line_redacts_by_argument_boundary_before_quoting() -> None:
    got = capture.capture_command(
        argv=["t.py", "--token=a b", "--password", "x y", "PGPASSWORD=zz", "--note", "keep me"]
    )["command_line"]
    for secret in ("a b", "x y", "zz"):
        assert secret not in got
    assert "keep me" in got
    import shlex

    parts = shlex.split(got)
    assert parts == ["t.py", f"--token={REDACTED}", "--password", REDACTED,
                     f"PGPASSWORD={REDACTED}", "--note", "keep me"]  # fmt: skip


def test_command_line_executable_path_is_credential_checked() -> None:
    assert "tok" not in capture.capture_command(argv=["https://tok@h/x.sh"])["command_line"]


def test_command_line_default_reads_sys_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["x.py", "--epochs", "3"])
    assert capture.capture_command()["command_line"] == "x.py --epochs 3"


def test_command_line_without_storing_arguments_keeps_only_name_and_count() -> None:
    argv = ["/home/alice/proj/train.py", "--token=abc123", "--data", "/secret/path", "--lr", "0.1"]
    line = capture.capture_command(argv=argv, store_command_line=False)["command_line"]
    assert line == "train.py [5 arguments omitted]"
    for text in ("abc123", "/secret/path", "alice", "--lr", "0.1", "data"):
        assert text not in line


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], "<unknown> [0 arguments omitted]"), ([""], "<unknown> [0 arguments omitted]"),
     (["-c", "x=1"], "-c [1 arguments omitted]")],
)  # fmt: skip
def test_command_line_edge_cases_without_arguments(argv: list[str], expected: str) -> None:
    assert capture.capture_command(argv=argv, store_command_line=False)["command_line"] == expected


def test_empty_argv_still_gives_a_command_line() -> None:
    assert capture.capture_command(argv=[])["command_line"] == "<unknown>"


def test_executable_name_that_looks_like_a_secret_is_redacted() -> None:
    line = capture.capture_command(argv=["--token=abc123"], store_command_line=False)[
        "command_line"
    ]
    assert "abc123" not in line


# --- registry ---------------------------------------------------------------


def test_first_party_capturers_are_registered() -> None:
    assert {"git", "python", "packages", "gpu", "host", "command"} <= set(capturers.names())
    assert capturers.get("python")() == {"python_version": platform.python_version()}


def test_environment_variables_are_not_consulted_for_the_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CVPILOT_STORE_COMMAND_LINE", "false")
    assert "--lr" in capture.capture_command(argv=["t.py", "--lr", "1"])["command_line"]
    assert os.environ["CVPILOT_STORE_COMMAND_LINE"] == "false"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="needs POSIX FIFOs")
def test_a_lockfile_swapped_for_a_fifo_does_not_make_init_wait(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    os.mkfifo(tmp_path / "uv.lock")  # reading it would wait for a writer for ever
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path, "is_file", lambda self: True if self.name == "uv.lock" else real_is_file(self)
    )  # the check passed, then the file was swapped
    box: list[dict[str, object]] = []
    thread = threading.Thread(
        target=lambda: box.append(capture.capture_packages(cwd=tmp_path)), daemon=True
    )
    thread.start()
    thread.join(10)
    assert not thread.is_alive(), "capture_packages blocked on a FIFO"
    assert len(str(box[0]["packages_lock_sha256"])) == 64  # fell back to the installed set
