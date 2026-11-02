"""Canonical PyRADAGE alias for :mod:`fr2dynage.kinematics`."""
from importlib import import_module
import sys

_implementation = import_module("fr2dynage.kinematics")

sys.modules[__name__] = _implementation
