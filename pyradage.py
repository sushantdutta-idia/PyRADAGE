#!/usr/bin/env python3
"""Run PyRADAGE from a checkout, or forward imports to its source package."""
from pathlib import Path
import sys

source = Path(__file__).resolve().parent / "src"
if str(source) not in sys.path:
    sys.path.insert(0, str(source))

if __name__ == "__main__":
    from pyradage.cli import main
    raise SystemExit(main())
else:
    # A same-named checkout launcher must not shadow the importable package.
    from importlib.util import module_from_spec, spec_from_file_location
    package = source / "pyradage"
    spec = spec_from_file_location(__name__, package / "__init__.py",
                                  submodule_search_locations=[str(package)])
    module = module_from_spec(spec)
    sys.modules[__name__] = module
    spec.loader.exec_module(module)
