"""Guard: ``import iai_mcp.core`` must resolve to the package, never the
shadowed flat module of the same name.

``src/iai_mcp/`` contains both a ``core/`` package (the live recall
dispatch) and a flat ``core.py`` file (dead, unreferenced by any import).
Python's package-over-module precedence means the package always wins, but
the flat file is a standing hazard: a future edit that targets `core.py`
by name (e.g. a grep-driven patch) silently does nothing to the live
dispatch path. This pins the resolution so any future change to that
precedence — or accidental removal of the package's ``__init__.py`` — fails
loudly here instead of surfacing as a "the fix didn't take" bug report.
"""
from __future__ import annotations

import os

import iai_mcp.core


def test_iai_mcp_core_resolves_to_the_package_not_the_flat_module():
    resolved = os.path.normpath(iai_mcp.core.__file__)
    assert resolved.endswith(os.path.join("core", "__init__.py")), (
        f"iai_mcp.core resolved to {resolved!r}, expected the core/ package's "
        f"__init__.py — a future edit aimed at the flat core.py module would "
        f"silently miss the live recall dispatch path"
    )
