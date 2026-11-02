"""Canonical PyRADAGE alias for :mod:`fr2dynage.plotting`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.plotting")

sys.modules[__name__] = _implementation
