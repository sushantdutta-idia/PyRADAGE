"""Canonical PyRADAGE alias for :mod:`fr2dynage.reliability`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.reliability")

sys.modules[__name__] = _implementation
