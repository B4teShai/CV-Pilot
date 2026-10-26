#!/usr/bin/env bash
# Install CV-Pilot the way a person without a package index does, and run what the README shows.
#
# Usage: scripts/check_clone.sh [PYTHON]    (default: python3; needs git and uv on PATH)
#
# Clones the current commit (committed files only) into a temporary directory, creates a fresh
# virtual environment with PYTHON, installs the dependencies exactly as the committed uv.lock
# pins them (runtime dependencies only: uv export, then pip install --require-hashes; the
# network is used for those files only), installs the clone itself with pip, then runs `cvpilot --version`, `cvpilot doctor` and
# the README quick start: the `file=train.py` block is written to disk and the first command block
# after it runs. Everything after the clone runs in an environment built from an allow-list
# (env -i): a private HOME and TMPDIR inside the temporary directory, no inherited CVPILOT_,
# PIP_, PYTHON or proxy variables, so nothing of your home directory is read or written.
# Exits non-zero on the first failure; the temporary directory is removed either way.
set -euo pipefail

PYTHON="${1:-python3}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/cvpilot-clone.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home" "$WORK/tmp" "$WORK/quickstart"

# the only variables a step may see (the ones pip needs to verify certificates are the system's)
clean() {
  env -i PATH="$WORK/venv/bin:/usr/bin:/bin" HOME="$WORK/home" TMPDIR="$WORK/tmp" \
    LC_ALL=C.UTF-8 GIT_CEILING_DIRECTORIES="$WORK" PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_INPUT=1 NO_COLOR=1 "$@"
}

git clone --quiet --no-hardlinks "$REPO" "$WORK/src"
"$PYTHON" -m venv "$WORK/venv"
uv export --quiet --locked --no-default-groups --no-emit-project --project "$WORK/src" \
  -o "$WORK/requirements.txt"
clean "$WORK/venv/bin/python" -m pip install --quiet --require-hashes -r "$WORK/requirements.txt"
clean "$WORK/venv/bin/python" -m pip install --quiet --no-deps "$WORK/src"

echo "python: $(clean python --version 2>&1)"
clean cvpilot --version
clean cvpilot doctor 2>/dev/null

# The quick start from the cloned README: write train.py, then run the command block after it.
clean python - "$WORK/src/README.md" "$WORK/quickstart" <<'EOF'
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
clean bash -e quickstart.sh 2>/dev/null | tee output.txt
grep -q '^status: finished$' output.txt
grep -q '^n_points: 3$' output.txt
echo "clone check passed"
