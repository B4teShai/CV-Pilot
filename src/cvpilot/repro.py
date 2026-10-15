"""Assemble the ``Repro`` record at ``init`` (application layer).

Environment facts come from the ``capturers`` registry, never from a direct import of
``capture/``. Settings are received as parameters.
"""

from __future__ import annotations

import hashlib
import re
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from cvpilot.core.hashing import dataset_hash
from cvpilot.core.jsonable import to_jsonable
from cvpilot.core.schema import Repro
from cvpilot.registry import capturers
from cvpilot.runs.queries import RunView

_EMPTY_DATASET_HASH = "merkle:" + hashlib.sha256(b"").hexdigest()


class Captured(NamedTuple):
    """A ``Repro`` plus what must be stored next to it: the diff bytes and the host name."""

    repro: Repro
    git_diff: bytes | None
    host: str


def capture_repro(
    run_id: str,
    dataset_uri: str | None,
    config: Mapping[str, Any] | None,
    seed: int | None,
    *,
    store_command_line: bool = True,
    argv: Sequence[str] | None = None,
    cwd: Path | None = None,
) -> Captured:
    """Capture git, environment, dataset hash and the (redacted) command line.

    Args:
        dataset_uri: local dataset directory, or None when the run has no dataset (recorded as
            the empty set).
        store_command_line: pass ``Settings.store_command_line``; when False only the
            executable name and argument count are stored.

    Raises:
        FileNotFoundError, NotADirectoryError, ValueError: the dataset path is unusable.
    """
    git = capturers.get("git")(cwd=cwd)
    packages = capturers.get("packages")(cwd=cwd)
    python = capturers.get("python")()
    gpu = capturers.get("gpu")()
    host = capturers.get("host")()
    command = capturers.get("command")(argv=argv, store_command_line=store_command_line)
    diff: bytes | None = git["git_diff"]
    repro = Repro(
        run_id=run_id,
        git_commit=git["git_commit"],
        git_dirty=git["git_dirty"],
        git_diff_sha256=None if diff is None else hashlib.sha256(diff).hexdigest(),
        packages_lock_sha256=packages["packages_lock_sha256"],
        python_version=python["python_version"],
        cuda=gpu["cuda"],
        gpu_names=gpu["gpu_names"],
        dataset_uri="" if dataset_uri is None else dataset_uri,
        dataset_hash=_EMPTY_DATASET_HASH if dataset_uri is None else dataset_hash(dataset_uri),
        config=to_jsonable(dict(config or {})),
        seed=seed,
        command_line=command["command_line"],
    )
    return Captured(repro=repro, git_diff=diff, host=host["host"])


_OMITTED = re.compile(r"\[\d+ arguments omitted\]$")


def _comment(text: object) -> str:
    """One line of ``text`` safe inside a ``#`` comment: control characters become spaces."""
    return "".join(" " if ord(c) < 32 or ord(c) == 127 else c for c in str(text))


def script(view: RunView) -> str:
    """A bash script that recreates the recorded code state and runs the recorded command.

    Environment facts (Python, packages, CUDA, dataset, seed) are printed as comments for the
    reader to check. Values that reach executable lines are shell-quoted; the command line is
    the redacted, already shell-quoted line captured at ``init``, so secrets must be supplied
    through the environment. Read the script before running it.
    """
    meta, repro = view.meta, view.repro
    lines = [
        "#!/usr/bin/env bash",
        f"# Reproduction script for CV-Pilot run {meta.run_id} (project {_comment(meta.project)}).",
        "# Generated from the stored run record. Read it before running it.",
        "set -euo pipefail",
        "",
        "# 1. Code",
    ]
    if repro.git_commit is None:
        lines.append(
            "# no git commit was recorded (the run did not start in a git repository); "
            "run the command from the original source tree"
        )
    else:
        lines.append(
            f"git rev-parse --verify {shlex.quote(repro.git_commit + '^{commit}')} >/dev/null"
        )
        lines.append(f"git checkout --detach {shlex.quote(repro.git_commit)}")
        if repro.git_dirty:
            diff = next(
                (
                    a
                    for a in view.artifacts
                    if a.kind == "diff" and a.sha256 == repro.git_diff_sha256
                ),
                None,
            )
            if diff is None:
                lines.append("# the working tree had uncommitted changes that were not stored")
            else:
                lines.append(f"git apply -- {shlex.quote(str(view.store_path / diff.uri))}")
    lines += [
        "",
        "# 2. Environment (check before running)",
        f"# python: {_comment(repro.python_version)}",
        f"# packages lock sha256: {_comment(repro.packages_lock_sha256)}",
        f"# cuda / driver: {_comment(repro.cuda or 'none')}",
        f"# gpus: {_comment(', '.join(repro.gpu_names) or 'none')}",
        f"# seed: {_comment(repro.seed if repro.seed is not None else 'not set')}",
        "",
        "# 3. Dataset (hash recorded at init)",
        f"# uri: {_comment(repro.dataset_uri or 'none')}",
        f"# hash: {_comment(repro.dataset_hash)}",
        "",
        "# 4. Command (secrets were redacted at capture time; supply them via the environment)",
    ]
    command = repro.command_line
    if _OMITTED.search(command):
        lines += [
            f"# recorded: {_comment(command)}",
            "# the command line was not stored (CVPILOT_STORE_COMMAND_LINE=false)",
            'echo "cvpilot: the command line was not stored for this run" >&2',
            "exit 1",
        ]
    else:
        try:
            first = shlex.split(command)[0]
        except (ValueError, IndexError):
            first = ""
        prefix = '"${PYTHON:-python3}" ' if first.endswith(".py") else ""
        lines.append(prefix + command)
    return "\n".join(lines) + "\n"
