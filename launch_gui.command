#!/bin/sh
# Prefer the checkout virtual environment; otherwise use python3 on PATH.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$project_dir/.venv/bin/python" ]; then
    exec "$project_dir/.venv/bin/python" "$project_dir/gui.py"
fi
exec python3 "$project_dir/gui.py"
