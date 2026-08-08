"""A numba that cannot import degrades community detection, never breaks it.

numba raises ImportError from its own package __init__ on a numpy it does not
support. The numba-tainted imports (mosaic, mosaic_policy) therefore live
inside detect_communities' degrade guard: when they raise, the caller gets a
flat assignment instead of an exception on every cascade iteration.
"""
from __future__ import annotations

import builtins
import random
import sys
from uuid import uuid4

from iai_mcp.community import SMALL_N_FLAT, detect_communities
from iai_mcp.graph import MemoryGraph


def _graph(n: int) -> MemoryGraph:
    g = MemoryGraph()
    for i in range(n):
        rng = random.Random(i)
        g.add_node(
            uuid4(), community_id=None,
            embedding=[rng.random() for _ in range(384)],
        )
    return g


def test_mosaic_import_error_degrades_to_flat(monkeypatch) -> None:
    # The mosaic module must be re-imported inside the call for the failure
    # to fire; drop any cached copy first.
    for mod in ("iai_mcp.mosaic", "iai_mcp.mosaic_policy"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name in ("iai_mcp.mosaic", "iai_mcp.mosaic_policy"):
            raise ImportError("Numba needs NumPy 2.4 or less. Got NumPy 2.5.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    a = detect_communities(_graph(SMALL_N_FLAT + 50), prior=None)
    assert a.backend == "flat"


def test_lineage_report_importable_without_mosaic(monkeypatch) -> None:
    # The n==0 and small-n returns need LineageReport before the guard runs —
    # its import must never route through the numba-tainted modules.
    for mod in ("iai_mcp.mosaic", "iai_mcp.mosaic_policy"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name in ("iai_mcp.mosaic", "iai_mcp.mosaic_policy"):
            raise ImportError("numba unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    empty = detect_communities(MemoryGraph(), prior=None)
    assert empty.backend == "flat"
    small = detect_communities(_graph(5), prior=None)
    assert small.backend == "flat"
