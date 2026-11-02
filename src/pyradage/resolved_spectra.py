"""Canonical PyRADAGE alias for :mod:`fr2dynage.resolved_spectra`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.resolved_spectra")

sys.modules[__name__] = _implementation
