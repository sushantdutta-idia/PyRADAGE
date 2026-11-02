"""Canonical PyRADAGE alias for :mod:`fr2dynage.fitting`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.fitting")

sys.modules[__name__] = _implementation
