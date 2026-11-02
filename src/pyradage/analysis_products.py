"""Canonical PyRADAGE alias for :mod:`fr2dynage.analysis_products`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.analysis_products")

sys.modules[__name__] = _implementation
