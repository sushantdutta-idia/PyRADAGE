"""Canonical PyRADAGE alias for :mod:`fr2dynage.dynage_diagnostics`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.dynage_diagnostics")

sys.modules[__name__] = _implementation
