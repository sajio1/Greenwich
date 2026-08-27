#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$project_root/.venv/bin/alphamotion" ]]; then
  echo "AlphaMotion is not installed. Run ./install.sh first." >&2
  exit 1
fi
exec "$project_root/.venv/bin/alphamotion" serve "$@"
