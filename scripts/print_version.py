"""Print the package version from ``src/cvpilot/__init__.py`` without importing the package.

The release workflow compares it with the tag. The file is read as text and matched with one
anchored pattern, so nothing from the repository (and nothing a build might have changed) is
executed in the job that holds the repository.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PATTERN = re.compile(r'^__version__ = "([0-9]+\.[0-9]+\.[0-9]+(?:[.\-+a-z0-9]*))"$', re.M)


def version(root: Path) -> str:
    text = (root / "src" / "cvpilot" / "__init__.py").read_text(encoding="utf-8")
    found = PATTERN.findall(text)
    if len(found) != 1:
        raise SystemExit("could not find exactly one __version__ in src/cvpilot/__init__.py")
    return found[0]


if __name__ == "__main__":
    sys.stdout.write(version(Path(__file__).resolve().parents[1]) + "\n")
