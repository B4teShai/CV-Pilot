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
from urllib.parse import urlsplit

from cvpilot.core.hashing import dataset_hash as compute_dataset_hash
from cvpilot.core.jsonable import to_jsonable
from cvpilot.core.schema import Repro, check_dataset_hash
from cvpilot.registry import capturers
from cvpilot.runs.queries import RunView

_EMPTY_DATASET_HASH = "merkle:" + hashlib.sha256(b"").hexdigest()
MAX_DATASET_URI = 1024
"""Longest ``dataset_uri`` recorded with an externally computed ``dataset_hash``."""
_REMOTE_SPEC = re.compile(r"^[^/@:\s]+@[^/:\s]+:")
"""``user@host:path`` (scp style): a login, not a dataset name."""


def _check_dataset_name(uri: str) -> None:
    """Refuse a dataset name that could carry a credential.

    A URL (``scheme://...`` or scheme-relative ``//...``) must have no user information in its
    authority and no query or fragment: presigned and SAS URLs carry their signature there, and a
    dataset identity needs neither. ``name@version`` without a scheme is a name; a URL spelling of
    it must put the version in the path (``duvas://fridge-sku/1.4.0``), because
    ``duvas://fridge-sku@1.4.0`` cannot be told apart from ``scheme://user@host``.
    """
    if "://" in uri or uri.startswith("//"):
        try:
            parts = urlsplit(uri)
            has_user = parts.username is not None or parts.password is not None
        except ValueError:
            raise ValueError("dataset_uri is not a valid URL") from None
        if has_user or "@" in parts.netloc:
            raise ValueError(
                "dataset_uri must not carry user information (user@ or user:password@); "
                "write a versioned name as name@version or scheme://name/version"
            )
        if parts.query or parts.fragment or "?" in uri or "#" in uri:
            raise ValueError(
                "dataset_uri must not carry a query or fragment (signed URLs keep their "
                "credentials there); name the dataset without them"
            )
    elif _REMOTE_SPEC.match(uri):
        raise ValueError("dataset_uri must not carry a login (user@host:path)")


def external_dataset(dataset_uri: str | None, dataset_hash: str) -> tuple[str, str]:
    """Check a dataset identity computed outside CV-Pilot and return ``(uri, hash)``.

    ``dataset_hash`` must be ``sha256:<64 lower-case hex>`` (for example the SHA-256 of a dataset
    registry's manifest file); ``dataset_uri`` names the dataset (``fridge-sku@1.4.0``,
    ``duvas://fridge-sku/1.4.0``, a URL without credentials or query, a path) and is recorded as
    given, not read. Nothing in it is redacted: a value that could hold a secret is refused.

    Raises:
        ValueError: the hash is not ``sha256:<hex>``, or the name is missing, too long, holds
            control characters, user information, a login, or a URL query or fragment.
    """
    if not dataset_hash.startswith("sha256:"):
        raise ValueError('an external dataset_hash must be "sha256:<64 lower-case hex>"')
    check_dataset_hash(dataset_hash)
    if not dataset_uri:
        raise ValueError("dataset_uri must name the dataset when dataset_hash is given")
    if len(dataset_uri) > MAX_DATASET_URI:
        raise ValueError(f"dataset_uri is longer than {MAX_DATASET_URI} characters")
    if any(ord(c) < 32 or ord(c) == 127 for c in dataset_uri):
        raise ValueError("dataset_uri contains control characters")
    _check_dataset_name(dataset_uri)
    return dataset_uri, dataset_hash


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
    dataset_hash: str | None = None,
) -> Captured:
    """Capture git, environment, dataset hash and the (redacted) command line.

    Args:
        dataset_uri: local dataset directory, or None when the run has no dataset (recorded as
            the empty set). With ``dataset_hash`` it is only a name (see
            :func:`external_dataset`).
        dataset_hash: ``sha256:<hex>`` computed by another tool; the dataset is then not hashed
            here.
        store_command_line: pass ``Settings.store_command_line``; when False only the
            executable name and argument count are stored.

    Raises:
        FileNotFoundError, NotADirectoryError, ValueError: the dataset path is unusable, or the
            external dataset identity is invalid.
    """
    if dataset_hash is not None:
        uri, digest = external_dataset(dataset_uri, dataset_hash)
    elif dataset_uri is None:
        uri, digest = "", _EMPTY_DATASET_HASH
    else:
        uri, digest = dataset_uri, compute_dataset_hash(dataset_uri)
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
        dataset_uri=uri,
        dataset_hash=digest,
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
