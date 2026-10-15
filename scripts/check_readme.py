"""Refuse a README that cannot be a package index page: a relative link or image (it would break
on the index) or a frozen release-status line (it would be wrong the day after release).

Run by the release workflow before the build and by ``tests/release/test_readme_longdesc.py``.
Reads the file as text only; exits 1 and prints one line per problem.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LINK = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)[^)]*\)")
STATUS = re.compile(
    r"release candidate|nothing has been (?:published|uploaded)|not (?:yet )?published|"
    r"install \(from source\)|unreleased|pre-release",
    re.I,
)
ALLOWED_HOSTS = (
    "https://github.com/B4teShai/CV-Pilot",
    "https://raw.githubusercontent.com/B4teShai/CV-Pilot/",
    "https://docs.astral.sh/",
    "https://github.com/huggingface/safetensors",
)


def fenced(text: str) -> str:
    """The text with fenced code blocks blanked: only prose and links are judged."""
    return re.sub(r"^```.*?^```[ \t]*$", "", text, flags=re.S | re.M)


def problems(text: str) -> list[str]:
    prose = fenced(text)
    found: list[str] = []
    for match in LINK.finditer(prose):
        target = match.group(1)
        if target.startswith("#"):
            continue
        if not target.startswith("https://"):
            found.append(f"relative or non-https link: {target}")
        elif not any(
            target == host
            or target.startswith(host.rstrip("/") + "/")
            or target.startswith(host + "#")
            for host in ALLOWED_HOSTS
        ):
            found.append(f"link outside the project and the tools it names: {target}")
    for line in prose.splitlines():
        if STATUS.search(line):
            found.append(f"release-status wording on a line of the README: {line.strip()[:70]}")
    return found


def main(argv: list[str]) -> int:
    path = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parents[1] / "README.md"
    issues = problems(path.read_text(encoding="utf-8"))
    for issue in issues:
        sys.stderr.write(f"{path.name}: {issue}\n")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
