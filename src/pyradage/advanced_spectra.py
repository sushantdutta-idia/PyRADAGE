"""Canonical PyRADAGE alias for :mod:`fr2dynage.advanced_spectra`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.advanced_spectra")

sys.modules[__name__] = _implementation
