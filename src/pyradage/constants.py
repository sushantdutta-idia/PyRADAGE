"""Canonical PyRADAGE alias for :mod:`fr2dynage.constants`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.constants")

sys.modules[__name__] = _implementation
