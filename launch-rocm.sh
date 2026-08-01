#!/usr/bin/env sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

export TORCH_BACKEND=rocm
export VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv-rocm}"

exec sh "$SCRIPT_DIR/launch.sh" "$@"
