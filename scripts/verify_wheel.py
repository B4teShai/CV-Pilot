"""Install a built wheel into clean virtual environments and run the commands users run first.

``python scripts/verify_wheel.py dist/cvpilot-0.1.0-py3-none-any.whl --python 3.10 --python 3.13``
creates one throw-away environment per interpreter with ``uv venv``, installs only the wheel
(and what it depends on) into it, then runs ``cvpilot --version`` and ``cvpilot doctor`` with a
scrubbed environment, and checks that the package that was imported is the installed one and not
a checkout. Nothing is uploaded and nothing outside the temporary directory is touched. Exit 0
when every interpreter passes.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Report:
    python: str
    version_output: str
    doctor_output: str
    module_path: str


def _run(argv: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        argv, env=env, cwd=cwd, capture_output=True, text=True, check=False, timeout=300
    )


def verify_one(wheel: Path, python: str, work: Path, *, repo: Path | None = None) -> Report:
    """Install ``wheel`` into a fresh environment for ``python`` and exercise it.

    Raises:
        RuntimeError: a step failed; the message holds the command and its output.
    """
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("uv is required (https://docs.astral.sh/uv/)")
    venv = work / f"venv-{python}"
    env = {"PATH": "/usr/bin:/bin", "HOME": str(work), "NO_COLOR": "1", "LC_ALL": "C.UTF-8"}
    for argv in (
        [uv, "venv", "--python", python, str(venv)],
        [uv, "pip", "install", "--python", str(venv / "bin" / "python"), str(wheel)],
    ):
        done = _run(argv, env={**env, "PATH": f"/usr/bin:/bin:{Path(uv).parent}"}, cwd=work)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} failed:\n{done.stdout}\n{done.stderr}")
    exe = venv / "bin" / "cvpilot"
    steps = {
        "version": [str(exe), "--version"],
        "doctor": [str(exe), "doctor"],
        "path": [
            str(venv / "bin" / "python"),
            "-I",
            "-c",
            "import cvpilot; print(cvpilot.__file__)",
        ],
    }
    out: dict[str, str] = {}
    for name, argv in steps.items():
        done = _run(argv, env={**env, "PATH": str(venv / "bin")}, cwd=work)
        if done.returncode != 0:
            raise RuntimeError(f"{' '.join(argv)} failed ({done.returncode}):\n{done.stderr}")
        out[name] = done.stdout.strip()
    if repo is not None and Path(out["path"]).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError(f"imported the checkout, not the wheel: {out['path']}")
    if "site-packages" not in out["path"]:
        raise RuntimeError(f"cvpilot was not imported from site-packages: {out['path']}")
    if not re.fullmatch(r"cvpilot [0-9]+\.[0-9]+\.[0-9]+\S*", out["version"]):
        raise RuntimeError(f"unexpected --version output: {out['version']!r}")
    if out["doctor"] != "ok":
        raise RuntimeError(f"unexpected doctor output: {out['doctor']!r}")
    return Report(python, out["version"], out["doctor"], out["path"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--python", action="append", required=True, help="interpreter, repeatable")
    args = parser.parse_args(argv)
    if not args.wheel.is_file() or args.wheel.suffix != ".whl":
        parser.error("pass the path of a .whl file")
    repo = Path(__file__).resolve().parents[1]
    failed = False
    with tempfile.TemporaryDirectory(prefix="cvpilot-verify-") as tmp:
        for python in args.python:
            try:
                report = verify_one(args.wheel.resolve(), python, Path(tmp), repo=repo)
            except RuntimeError as exc:
                failed = True
                sys.stdout.write(f"FAIL python {python}: {exc}\n")
                continue
            sys.stdout.write(
                f"ok python {python}: {report.version_output}; doctor {report.doctor_output}; "
                f"{report.module_path}\n"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
