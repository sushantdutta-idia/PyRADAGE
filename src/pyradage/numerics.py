"""Canonical PyRADAGE alias for :mod:`fr2dynage.numerics`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.numerics")

sys.modules[__name__] = _implementation
