#!/bin/sh
# Open PyRADAGE using the checkout's virtual environment when present, otherwise active Python.
set -eu
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
if [ -x "$project_dir/.venv/bin/python" ]; then
    exec "$project_dir/.venv/bin/python" "$project_dir/gui.py"
fi
exec python3 "$project_dir/gui.py"
