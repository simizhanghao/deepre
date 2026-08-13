#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
# shellcheck source=../config/project.env
source "$PROJECT_ROOT/config/project.env"
PYTHON_BIN=${PYTHON_BIN:-$VEXACT_ROOT/.venv/bin/python}

mkdir -p "$PROJECT_ROOT/artifacts" "$PROJECT_ROOT/logs" "$PROJECT_ROOT/results"

require_file() {
  [[ -f "$1" ]] || { echo "ERROR missing file: $1" >&2; exit 1; }
}

require_dir() {
  [[ -d "$1" ]] || { echo "ERROR missing directory: $1" >&2; exit 1; }
}

run_llamafactory() {
  if command -v llamafactory-cli >/dev/null 2>&1; then
    llamafactory-cli "$@"
  else
    PYTHONPATH="$LLAMAFACTORY_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
      python3 -m llamafactory.cli "$@"
  fi
}
