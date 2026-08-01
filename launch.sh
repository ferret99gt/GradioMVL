#!/usr/bin/env sh

set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

PYTHON="${PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$SCRIPT_DIR/venv}"
TORCH_BACKEND="${TORCH_BACKEND:-cuda}"

if ! "$PYTHON" -c "" >/dev/null 2>&1; then
    echo "Couldn't launch Python using: $PYTHON" >&2
    exit 1
fi

if [ "$VENV_DIR" != "-" ] && [ "${SKIP_VENV:-0}" != "1" ]; then
    if [ ! -x "$VENV_DIR/bin/python" ]; then
        echo "Creating venv in $VENV_DIR using $PYTHON"
        if ! "$PYTHON" -m venv "$VENV_DIR"; then
            echo "Unable to create the virtual environment. Install Python's venv package and try again." >&2
            exit 1
        fi
    fi
    PYTHON="$VENV_DIR/bin/python"
fi

if ! "$PYTHON" -m pip --help >/dev/null 2>&1; then
    echo "pip is unavailable in the selected Python environment." >&2
    exit 1
fi

export PYTHON VENV_DIR TORCH_BACKEND
exec "$PYTHON" launch.py "$@"
