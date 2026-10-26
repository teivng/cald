"""Locate and import the pinned upstream nanoTabICL source.

The upstream model (soda-inria/nanotabicl, BSD-3, pinned by git submodule at
third_party/nanotabicl) is used as-is — we never edit its source. It is found,
in order, at:

1. ``$CALD_NANOTABICL_PATH`` (a directory containing model.py), for overrides;
2. ``cald/_upstream/nanotabicl/`` — present in built wheels/sdists via the
   build-time force-include;
3. ``<repo root>/third_party/nanotabicl/`` — the git submodule, for editable
   installs from a checkout (run ``git submodule update --init`` first).
"""

import importlib.util
import os
import sys
from pathlib import Path

_MODULE_NAME = "cald_nanotabicl_upstream_model"


def _candidate_dirs():
    env = os.environ.get("CALD_NANOTABICL_PATH")
    if env:
        yield Path(env)
    pkg_root = Path(__file__).resolve().parents[2]          # .../cald
    yield pkg_root / "_upstream" / "nanotabicl"
    repo_root = Path(__file__).resolve().parents[4]          # editable checkout
    yield repo_root / "third_party" / "nanotabicl"


def load_upstream():
    """Import upstream model.py once; return the module."""
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    for d in _candidate_dirs():
        path = d / "model.py"
        if path.is_file():
            spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[_MODULE_NAME] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        "nanoTabICL upstream source not found. From a repo checkout, run "
        "`git submodule update --init`; or set CALD_NANOTABICL_PATH to a "
        "directory containing nanotabicl's model.py.")
