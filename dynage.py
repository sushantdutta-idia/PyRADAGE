#!/usr/bin/env python3
"""Keep the legacy dynage.py command available for existing source checkouts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pyradage.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

