"""Run the commands of a Markdown page in one persistent shell and check what it says.

Fenced blocks are read in order. ``bash`` blocks run in a single ``bash -e`` session (so
``export`` and ``cd`` carry over) and must succeed; a block that is expected to fail says so
with ``|| echo "...$?"`` in its own text. A block of another language with ``file=NAME`` in its
info string is written to NAME in the working directory (not executed). A ``text expect`` block
right after a ``bash`` block lists whole lines of that block's **stdout** (``text expect stderr``:
of its stderr) that must appear in that order, each line matched once and in full. ``{*}`` inside
a line matches any text there and nothing else is a wildcard; ``json: key=value ...`` matches a
JSON log line whose fields have exactly those values. A block marked ``notest`` is never run: the
docs-lint test requires that every shell or Python block of a guide page is either executed here
or carries ``notest``.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

FENCE = re.compile(r"^```([A-Za-z0-9_+-]*)[ \t]*([^\n]*)\n(.*?)^```[ \t]*$", re.S | re.M)
RUNNABLE_LANGS = {"bash", "sh", "shell", "console", "python", "py"}


@dataclass
class Block:
    lang: str
    info: str
    body: str
    line: int

    @property
    def notest(self) -> bool:
        return "notest" in self.info.split()

    @property
    def file(self) -> str | None:
        match = re.search(r"\bfile=([A-Za-z0-9_./-]+)", self.info)
        return match.group(1) if match else None

    @property
    def expect(self) -> bool:
        return self.lang == "text" and "expect" in self.info.split()

    @property
    def stream(self) -> str:
        return "stderr" if "stderr" in self.info.split() else "stdout"


def blocks(text: str) -> list[Block]:
    return [
        Block(m.group(1), m.group(2).strip(), m.group(3), text[: m.start()].count("\n") + 1)
        for m in FENCE.finditer(text)
    ]


@dataclass
class Session:
    """A persistent bash process in a working directory with a controlled environment."""

    cwd: Path
    env: dict[str, str]
    proc: subprocess.Popen[str] = field(init=False)
    capture: Path = field(init=False)

    def __post_init__(self) -> None:
        self.capture = Path(tempfile.mkdtemp(prefix="docs-harness-"))
        self.proc = subprocess.Popen(  # noqa: S603
            ["bash", "--noprofile", "--norc", "-e"],  # noqa: S607
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def run(self, script: str) -> tuple[int, str, str]:
        """``(status, stdout, stderr)`` of the script, which runs in the persistent shell (so
        ``export`` and ``cd`` carry over) with its two streams captured separately."""
        assert self.proc.stdin is not None and self.proc.stdout is not None
        marker = f"__DONE_{uuid.uuid4().hex}__"
        out, err = self.capture / "out", self.capture / "err"
        self.proc.stdin.write(f"{{\n{script}\n}} > '{out}' 2> '{err}'\necho {marker}$?\n")
        self.proc.stdin.flush()
        status = 255  # the shell died: set -e stopped it on a failing command
        for line in self.proc.stdout:
            if line.startswith(marker):
                status = int(line[len(marker) :].strip())
                break
        read = lambda p: p.read_text(errors="replace") if p.exists() else ""  # noqa: E731
        return status, read(out), read(err)

    def close(self) -> None:
        if self.proc.stdin:
            self.proc.stdin.close()
        self.proc.wait(timeout=30)


def clean_env(home: Path) -> dict[str, str]:
    """A minimal environment whose ``python`` and ``cvpilot`` are the interpreter under test."""
    bin_dir = str(Path(sys.executable).parent)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin{os.pathsep}/bin",
        "HOME": str(home),
        "GIT_CEILING_DIRECTORIES": str(home.parent),
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "NO_COLOR": "1",
    }
    return env


@dataclass
class Result:
    block: Block
    status: int
    stdout: str
    stderr: str

    @property
    def output(self) -> str:
        return self.stdout + self.stderr


def _line_matches(expected: str, actual: str) -> bool:
    """Whole-line equality; ``{*}`` is the only wildcard."""
    pieces = [re.escape(part) for part in expected.split("{*}")]
    return re.fullmatch(".*".join(pieces), actual) is not None


def _json_matches(spec: str, actual: str) -> bool:
    try:
        document = json.loads(actual)
    except ValueError:
        return False
    if not isinstance(document, dict):
        return False
    for pair in spec.split():
        key, _, value = pair.partition("=")
        if str(document.get(key)) != value:
            return False
    return True


def check_expected(expected: str, actual: str, where: str) -> None:
    """Every expected line, in order, matched against a whole line of ``actual``, once."""
    lines = actual.splitlines()
    cursor = 0
    for wanted in (line.rstrip() for line in expected.splitlines() if line.strip()):
        found = None
        for index in range(cursor, len(lines)):
            line = lines[index].rstrip()
            hit = (
                _json_matches(wanted[len("json:") :].strip(), line)
                if wanted.startswith("json:")
                else _line_matches(wanted, line)
            )
            if hit:
                found = index
                break
        if found is None:
            elsewhere = any(
                (_line_matches(wanted, ln.rstrip()) if not wanted.startswith("json:") else False)
                for ln in lines[:cursor]
            )
            reason = "in the wrong order or already matched" if elsewhere else "not found"
            raise AssertionError(
                f"{where}: expected the whole line {wanted!r} ({reason}) in:\n{actual}"
            )
        cursor = found + 1


def run_page(text: str, workdir: Path, env: dict[str, str]) -> list[Result]:
    """Execute a page; raises ``AssertionError`` naming the line of the first failing block."""
    workdir.mkdir(parents=True, exist_ok=True)
    session = Session(workdir, env)
    results: list[Result] = []
    last_bash: Result | None = None
    try:
        for block in blocks(text):
            if block.notest:
                continue
            if block.expect:
                assert last_bash is not None, f"line {block.line}: expect block without a command"
                actual = last_bash.stderr if block.stream == "stderr" else last_bash.stdout
                check_expected(
                    block.body,
                    actual,
                    f"line {block.line} ({block.stream} of the command at line "
                    f"{last_bash.block.line})",
                )
                continue
            if block.file is not None:
                target = workdir / block.file
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(block.body)
                continue
            if block.lang in {"bash", "sh", "shell"}:
                status, stdout, stderr = session.run(block.body)
                last_bash = Result(block, status, stdout, stderr)
                results.append(last_bash)
                assert status == 0, (
                    f"line {block.line}: exit {status}\n{block.body}\n{stdout}{stderr}"
                )
    finally:
        session.close()
    return results
