"""Canonical PyRADAGE alias for :mod:`fr2dynage.maps`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.maps")

sys.modules[__name__] = _implementation
