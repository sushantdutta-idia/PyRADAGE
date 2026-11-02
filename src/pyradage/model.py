"""Canonical PyRADAGE alias for :mod:`fr2dynage.model`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.model")

sys.modules[__name__] = _implementation
