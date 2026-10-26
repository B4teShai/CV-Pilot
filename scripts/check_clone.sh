#!/usr/bin/env bash
# Install CV-Pilot the way a person without a package index does, and run what the README shows.
#
# Usage: scripts/check_clone.sh [PYTHON]    (default: python3)
#
# Clones the current commit (committed files only) into a temporary directory, creates a fresh
# virtual environment with PYTHON, runs `pip install .` from the clone (the network is used for
# the build backend and the dependencies only), then `cvpilot --version`, `cvpilot doctor` and the
# README quick start: the `file=train.py` block is written to disk and the first command block
# after it runs. The run store lives in the temporary directory, never in your home directory.
# Exits non-zero on the first failure; the temporary directory is removed either way.
set -euo pipefail

PYTHON="${1:-python3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/cvpilot-clone.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

git clone --quiet --no-hardlinks "$REPO" "$WORK/src"
"$PYTHON" -m venv "$WORK/venv"
"$WORK/venv/bin/python" -m pip install --quiet --disable-pip-version-check "$WORK/src"

mkdir -p "$WORK/home" "$WORK/quickstart"
export HOME="$WORK/home"
export GIT_CEILING_DIRECTORIES="$WORK"
export PATH="$WORK/venv/bin:/usr/bin:/bin"
unset VIRTUAL_ENV PYTHONPATH
for name in $(env | sed -n 's/^\(CVPILOT_[A-Z_]*\)=.*/\1/p'); do unset "$name"; done

echo "python: $(python --version 2>&1)"
cvpilot --version
cvpilot doctor 2>/dev/null

# The quick start from the cloned README: write train.py, then run the command block after it.
python - "$WORK/src/README.md" "$WORK/quickstart" <<'EOF'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])
fences = re.findall(r"^```([A-Za-z]*)[ \t]*([^\n]*)\n(.*?)^```[ \t]*$", text, flags=re.S | re.M)
start = next(i for i, (_, info, _) in enumerate(fences) if "file=train.py" in info)
(out / "train.py").write_text(fences[start][2], encoding="utf-8")
command = next(
    body for lang, info, body in fences[start + 1 :] if lang == "bash" and "notest" not in info
)
(out / "quickstart.sh").write_text(command, encoding="utf-8")
EOF
cd "$WORK/quickstart"
bash -e quickstart.sh 2>/dev/null | tee output.txt
grep -q '^status: finished$' output.txt
grep -q '^n_points: 3$' output.txt
echo "clone check passed"
