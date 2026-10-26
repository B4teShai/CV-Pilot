"""A person without a package index can install CV-Pilot from a clone and run the quick start.

Marked ``clone`` and off by default: ``pip install .`` fetches the build backend and the
dependencies from the network. Run with ``uv run pytest -m clone``. The test runs
``scripts/check_clone.sh``, the same script a person can run by hand.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.clone
ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check_clone.sh"


def interpreter(version: str) -> str | None:
    """A real interpreter of ``version`` found by uv, or None."""
    if shutil.which("uv") is None:
        return None
    found = subprocess.run(  # noqa: S603
        ["uv", "python", "find", version],  # noqa: S607
        capture_output=True, text=True, check=False, timeout=60,
    )  # fmt: skip
    path = found.stdout.strip()
    return path if found.returncode == 0 and path else None


@pytest.mark.parametrize("version", ["3.10", "3.14"])
def test_a_fresh_clone_installs_with_pip_and_runs_the_readme_quick_start(
    version: str, tmp_path: Path
) -> None:
    (tmp_path / "home").mkdir()
    python = interpreter(version)
    if python is None:
        pytest.skip(f"python {version} is not available")
    done = subprocess.run(  # noqa: S603
        ["bash", str(SCRIPT), python],  # noqa: S607
        capture_output=True, text=True, check=False, timeout=900,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "TMPDIR": str(tmp_path),
             "HOME": str(tmp_path / "home")},  # private: nothing of the real home is read
    )  # fmt: skip
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    lines = done.stdout.splitlines()
    assert f"python: Python {version}." in "\n".join(lines)
    assert any(line.startswith("cvpilot ") for line in lines)
    assert "ok" in lines and "status: finished" in lines and "clone check passed" in lines
