from __future__ import annotations

import builtins
import random
import sys
from uuid import uuid4

import pytest

from iai_mcp.community import detect_communities
from iai_mcp.graph import MemoryGraph


def _random_emb(seed: int) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(384)]


def _two_clique_graph() -> MemoryGraph:
    """A graph large enough to take the mosaic path rather than a flat return."""
    g = MemoryGraph()
    clique_a = [uuid4() for _ in range(150)]
    clique_b = [uuid4() for _ in range(150)]
    for i, n in enumerate(clique_a):
        g.add_node(n, community_id=None, embedding=_random_emb(i))
    for i, n in enumerate(clique_b):
        g.add_node(n, community_id=None, embedding=_random_emb(10_000 + i))
    for i in range(150):
        for j in range(i + 1, 150):
            g.add_edge(clique_a[i], clique_a[j])
            g.add_edge(clique_b[i], clique_b[j])
    return g


@pytest.fixture
def broken_numba(monkeypatch):
    """Make `import numba` fail the way a numpy ABI mismatch does.

    numba raises ImportError from its own package __init__ when the installed
    numpy is newer than it supports, so the failure lands at import time
    rather than on first kernel call.
    """
    for name in [m for m in sys.modules if m == "numba" or m.startswith("numba.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in [m for m in sys.modules if m.startswith("iai_mcp.mosaic")]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "numba" or name.startswith("numba."):
            raise ImportError("Numba needs NumPy 2.4 or less. Got NumPy 2.5.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_detect_communities_falls_back_when_numba_import_fails(broken_numba):
    """A broken numba must degrade to the flat backend instead of raising.

    The ImportError arm of the try/except was already there but unreachable:
    `from iai_mcp.mosaic import run_mosaic` sat above the guard, so the error
    escaped detect_communities and surfaced on every MCP tool call.
    """
    g = _two_clique_graph()

    assignment = detect_communities(g, prior=None, prior_mode="seeded")

    assert assignment.backend == "flat"
    assert assignment.node_to_community


def test_detect_communities_uses_mosaic_when_numba_is_healthy():
    """Guard against the fallback quietly masking a real mosaic regression."""
    pytest.importorskip("numba")
    g = _two_clique_graph()

    assignment = detect_communities(g, prior=None, prior_mode="seeded")

    assert assignment.backend.startswith("leiden")
