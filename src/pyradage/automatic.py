"""Canonical PyRADAGE alias for :mod:`fr2dynage.automatic`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.automatic")

sys.modules[__name__] = _implementation
