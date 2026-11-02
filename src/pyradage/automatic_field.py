"""Canonical PyRADAGE alias for :mod:`fr2dynage.automatic_field`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.automatic_field")

sys.modules[__name__] = _implementation
