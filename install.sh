#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON:-python3}"

"$python_bin" - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit("AlphaMotion requires Python 3.10 or newer")
PY

if [[ ! -x "$project_root/.venv/bin/python" ]]; then
  "$python_bin" -m venv "$project_root/.venv"
fi

"$project_root/.venv/bin/python" -m pip install --upgrade pip
"$project_root/.venv/bin/python" -m pip install -e "$project_root"
"$project_root/.venv/bin/alphamotion" setup "$@"

echo
echo "AlphaMotion is ready. Start it with: ./run.sh"
