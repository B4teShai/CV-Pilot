"""Run the commands of a Markdown page in one persistent shell and check what it says.

Fenced blocks are read in order. ``bash`` blocks run in a single ``bash -e`` session (so
``export`` and ``cd`` carry over) and must succeed; a block that is expected to fail says so
with ``|| echo "...$?"`` in its own text. A block of another language with ``file=NAME`` in its
info string is written to NAME in the working directory (not executed). A ``text expect`` block
right after a ``bash`` block lists lines that must each appear in that block's output (stdout and
stderr together). A block marked ``notest`` is never run: the docs-lint test requires that every
shell or Python block of a guide page is either executed here or carries ``notest``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
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

    def __post_init__(self) -> None:
        self.proc = subprocess.Popen(  # noqa: S603
            ["bash", "--noprofile", "--norc", "-e"],  # noqa: S607
            cwd=self.cwd,
            env=self.env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def run(self, script: str) -> tuple[int, str]:
        assert self.proc.stdin is not None and self.proc.stdout is not None
        marker = f"__DONE_{uuid.uuid4().hex}__"
        self.proc.stdin.write(f"{script}\necho {marker}$?\n")
        self.proc.stdin.flush()
        out: list[str] = []
        for line in self.proc.stdout:
            if line.startswith(marker):
                return int(line[len(marker) :].strip()), "".join(out)
            out.append(line)
        return 255, "".join(out)  # the shell died: set -e stopped it on a failing command

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
    output: str


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
                for wanted in (line for line in block.body.splitlines() if line.strip()):
                    assert wanted in last_bash.output, (
                        f"line {block.line}: expected {wanted!r} in the output of the command "
                        f"at line {last_bash.block.line}:\n{last_bash.output}"
                    )
                continue
            if block.file is not None:
                target = workdir / block.file
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(block.body)
                continue
            if block.lang in {"bash", "sh", "shell"}:
                status, output = session.run(block.body)
                last_bash = Result(block, status, output)
                results.append(last_bash)
                assert status == 0, f"line {block.line}: exit {status}\n{block.body}\n{output}"
    finally:
        session.close()
    return results
