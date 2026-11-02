"""PyRADAGE: radio-source radiative ageing and constrained dynamics.

The established numerical implementation remains in ``fr2dynage`` so old
imports and saved scientific schemas remain compatible. Submodule aliases
share the same objects and state; optional dependencies are imported lazily.
"""
from fr2dynage import *  # noqa: F401,F403
from fr2dynage import __all__, __version__
