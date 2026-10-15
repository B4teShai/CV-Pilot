#!/usr/bin/env bash
# Install the core wheel into a clean venv and report its size; fail if torch or pynvml is present.
# Usage: benchmarks/install_size.sh [python-version]   (default 3.12)
set -euo pipefail
PY="${1:-3.12}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
uv build --wheel --out-dir "$WORK/dist" "$ROOT" >/dev/null
uv venv --python "$PY" "$WORK/venv" >/dev/null
start=$(date +%s)
uv pip install --python "$WORK/venv/bin/python" "$WORK"/dist/cvpilot-*.whl >/dev/null
end=$(date +%s)
SITE="$("$WORK/venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "python            : $("$WORK/venv/bin/python" --version)"
echo "install time      : $((end - start)) s"
echo "site-packages size: $(du -sk "$SITE" | cut -f1) KB"
echo "largest packages  :"
du -sk "$SITE"/* | sort -rn | head -5 | sed "s#$SITE/#  #"
echo "installed distributions:"
"$WORK/venv/bin/python" -m pip --version >/dev/null 2>&1 || true
uv pip list --python "$WORK/venv/bin/python" 2>/dev/null | sed 's/^/  /'
if ls "$SITE" | grep -Eiq '^(torch|pynvml|nvidia)'; then
  echo "FAIL: torch/pynvml/nvidia packages found in the core install" >&2
  exit 1
fi
"$WORK/venv/bin/python" -c "import cvpilot.api, cvpilot.cli.app, sys; assert 'torch' not in sys.modules; print('import check      : ok, torch not imported')"
"$WORK/venv/bin/cvpilot" --version
# smoke test from the wheel: entry points, migrations and pyarrow work outside the source tree
export CVPILOT_STORE_DIR="$WORK/store" CVPILOT_PROJECT=smoke
(cd "$WORK" && "$WORK/venv/bin/python" -c "
import cvpilot
r = cvpilot.init(task='detection'); cvpilot.log({'loss': 1.0}, step=0); cvpilot.finish()
print('smoke run         :', r.run_id)" 2>/dev/null)
"$WORK/venv/bin/cvpilot" runs list | head -3
echo "RESULT            : no torch in the core install"
