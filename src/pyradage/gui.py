"""Canonical PyRADAGE alias for :mod:`fr2dynage.gui`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.gui")

if __name__ == "__main__":
    raise SystemExit(_implementation.main())

sys.modules[__name__] = _implementation
