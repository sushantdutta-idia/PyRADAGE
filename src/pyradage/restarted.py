"""Canonical PyRADAGE alias for :mod:`fr2dynage.restarted`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.restarted")

sys.modules[__name__] = _implementation
