"""Canonical PyRADAGE alias for :mod:`fr2dynage.radio_inference`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.radio_inference")

sys.modules[__name__] = _implementation
