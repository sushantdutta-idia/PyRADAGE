"""Canonical PyRADAGE alias for :mod:`fr2dynage.spectral_maps`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.spectral_maps")

sys.modules[__name__] = _implementation
