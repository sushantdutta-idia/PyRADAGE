"""Canonical PyRADAGE alias for :mod:`fr2dynage.pyrats_bridge`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.pyrats_bridge")

sys.modules[__name__] = _implementation
