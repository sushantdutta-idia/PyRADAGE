"""Canonical PyRADAGE alias for :mod:`fr2dynage.source_analysis`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.source_analysis")

sys.modules[__name__] = _implementation
