#!/usr/bin/env python3
"""Open the guided interface from a source checkout."""
from pathlib import Path
import sys

source=Path(__file__).resolve().parent/"src"
if str(source) not in sys.path:
    sys.path.insert(0,str(source))
from pyradage.gui import main

if __name__=="__main__":
    raise SystemExit(main())
