"""Canonical PyRADAGE alias for :mod:`fr2dynage.reference_priors`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.reference_priors")

sys.modules[__name__] = _implementation
