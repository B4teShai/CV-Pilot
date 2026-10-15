"""Environment capturers, registered under the ``cvpilot.capturers`` entry-point group.

Every capturer returns plain values and never raises for an absent facility: no git
repository, no git binary, no GPU and no lockfile all give ``None`` or empty values. No
deep-learning framework is imported; ``pynvml`` is imported lazily and ``nvidia-smi`` is used
when it is missing.
Settings reach this module only as function parameters.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

MAX_DIFF_BYTES = 10 * 1024 * 1024
_TOOL_TIMEOUT_S = 10.0
_COMMIT = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
_LOCKFILES = ("uv.lock", "poetry.lock")
_CUDA_VERSION = re.compile(r"CUDA Version:\s*(\d+\.\d+)")


def _run(cmd: Sequence[str], *, cwd: Path | None = None) -> bytes | None:
    """Run a trusted local tool without a shell; stdout on exit 0, else None."""
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0"}
    try:
        done = subprocess.run(  # noqa: S603  # argv list, absolute executable, no shell
            list(cmd),
            cwd=cwd,
            env=env,
            capture_output=True,
            timeout=_TOOL_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


# -- git ----------------------------------------------------------------------


def capture_git(*, cwd: Path | None = None) -> dict[str, Any]:
    """Commit, dirty flag and the diff of tracked changes against HEAD.

    ``git_dirty`` ignores untracked files. A diff above ``MAX_DIFF_BYTES`` is not returned
    (``git_diff`` None) but the tree is still reported dirty. Repository config that could run
    programs (fsmonitor, external diff, textconv) is disabled.
    """
    empty: dict[str, Any] = {"git_commit": None, "git_dirty": False, "git_diff": None}
    exe = shutil.which("git")
    where = Path.cwd() if cwd is None else Path(cwd)
    if exe is None or not where.is_dir():
        return empty
    base = [exe, "-c", "core.fsmonitor=false", "-c", "core.pager=cat"]
    head = _run([*base, "rev-parse", "--verify", "-q", "HEAD"], cwd=where)
    commit = head.decode("ascii", errors="replace").strip() if head else ""
    if not _COMMIT.fullmatch(commit):
        return empty
    status = _run([*base, "status", "--porcelain", "--untracked-files=no"], cwd=where)
    dirty = bool(status and status.strip())
    diff: bytes | None = None
    if dirty:
        raw = _run(
            [*base, "diff", "HEAD", "--binary", "--no-ext-diff", "--no-textconv", "--no-color"],
            cwd=where,
        )
        diff = raw if raw is not None and len(raw) <= MAX_DIFF_BYTES else None
    return {"git_commit": commit, "git_dirty": dirty, "git_diff": diff}


# -- packages ------------------------------------------------------------------


def _lockfile(cwd: Path) -> Path | None:
    chain = [cwd, *cwd.parents]
    top = next((i for i, d in enumerate(chain) if (d / ".git").exists()), 0)
    for directory in chain[: top + 1]:
        for name in _LOCKFILES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def capture_packages(*, cwd: Path | None = None) -> dict[str, Any]:
    """SHA-256 of ``uv.lock``/``poetry.lock`` (searched up to the repo root), else of the
    sorted ``name==version`` lines of the installed distributions."""
    where = (Path.cwd() if cwd is None else Path(cwd)).resolve()
    lock = _lockfile(where)
    if lock is not None:
        return {"packages_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
    lines = sorted(
        {
            f"{(dist.metadata['Name'] or '').lower()}=={dist.version}"
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        }
    )
    return {"packages_lock_sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest()}


# -- python, host --------------------------------------------------------------


def capture_python() -> dict[str, Any]:
    return {"python_version": platform.python_version()}


def capture_host() -> dict[str, Any]:
    """Hostname only; never user name or address."""
    return {"host": socket.gethostname() or "unknown"}


# -- gpu -----------------------------------------------------------------------


def _text(value: object) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _gpu_from_nvml() -> dict[str, Any] | None:
    try:
        import pynvml  # lazy and optional

        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001  # missing module or library: no GPU information
        return None
    try:
        names = [
            _text(pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i)))
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        driver = _text(pynvml.nvmlSystemGetDriverVersion())
        raw = int(pynvml.nvmlSystemGetCudaDriverVersion())
        return {"names": names, "driver": driver, "cuda": f"{raw // 1000}.{raw % 1000 // 10}"}
    except Exception:  # noqa: BLE001  # driver errors must not break a run
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001, S110  # best-effort cleanup
            pass


def _gpu_from_smi() -> dict[str, Any] | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    query = _run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    if not query:
        return None
    names, driver = [], None
    for line in query.decode("utf-8", errors="replace").splitlines():
        name, _, drv = line.partition(",")
        if name.strip():
            names.append(name.strip())
            driver = drv.strip() or driver
    banner = _run([exe]) or b""
    match = _CUDA_VERSION.search(banner.decode("utf-8", errors="replace"))
    return {"names": names, "driver": driver, "cuda": match.group(1) if match else None}


def capture_gpu() -> dict[str, Any]:
    """``cuda`` as ``"<cuda> / <driver>"`` and GPU names; ``None`` and ``[]`` without a GPU."""
    info = _gpu_from_nvml() or _gpu_from_smi()
    if not info or not info["names"]:
        return {"cuda": None, "gpu_names": []}
    parts = [p for p in (info["cuda"], info["driver"]) if p]
    return {"cuda": " / ".join(parts) or None, "gpu_names": list(info["names"])}


# -- command line ----------------------------------------------------------------


def capture_command(
    *, argv: Sequence[str] | None = None, store_command_line: bool = True
) -> dict[str, Any]:
    """The shell-quoted command line with secrets redacted.

    With ``store_command_line=False`` only the executable's base name and the number of
    arguments are kept; no argument text is ever read into the result.
    """
    from cvpilot.core.redact import redact_argv

    args = list(sys.argv if argv is None else argv)
    executable = args[0] if args and args[0] else ""
    rest = args[1:]
    # Redact the argument list first (structural), then quote each token for the shell.
    redacted = redact_argv([executable or "<unknown>", *rest])
    if store_command_line:
        head = shlex.quote(redacted[0]) if executable else "<unknown>"
        return {"command_line": " ".join([head, *map(shlex.quote, redacted[1:])])}
    name = redact_argv([re.split(r"[\\/]", executable)[-1]])[0] if executable else "<unknown>"
    return {"command_line": f"{name} [{len(rest)} arguments omitted]"}
