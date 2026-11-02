"""Canonical PyRADAGE alias for :mod:`fr2dynage.maps_io`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.maps_io")

sys.modules[__name__] = _implementation
