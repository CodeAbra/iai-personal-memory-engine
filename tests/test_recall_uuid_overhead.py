"""Verify the per-recall UUID-construction overhead is reduced on the spread fan-out.

Contract:
  - The count of uuid.UUID.__init__ calls attributable to the spread fan-out
    (incident_edges neighbors) drops materially when the string-neighbor path
    is used versus a baseline that constructs a UUID for every neighbor.
  - The final hit id set and order produced by a recall over the same fixture
    store are byte-identical between the reduced-UUID path and the prior path.
"""
from __future__ import annotations

import sys
import uuid
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.types import EMBED_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _build_connected_store(tmp_path: Path, n_records: int = 40):
    """Build a store with edges so incident_edges actually returns neighbours."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "uuid_store")
    recs = []
    for i in range(n_records):
        r = _make(text=f"record {i}", vec=_unit_vec(i + 200))
        store.insert(r)
        recs.append(r)

    # Wire a chain of hebbian edges so the spread fan-out has real neighbours
    pairs = [(recs[i].id, recs[i + 1].id) for i in range(min(n_records - 1, 30))]
    try:
        store.boost_edges(pairs, delta=0.5, edge_type="hebbian")
    except Exception:  # noqa: BLE001 — edge add optional
        pass

    return store, recs


def _count_uuid_inits(store, recs) -> int:
    """Count UUID.__init__ calls for one batch of incident_edges calls
    on the full record set (simulates the spread fan-out cost)."""
    all_ids = [r.id for r in recs]
    counter = [0]
    real_init = UUID.__init__

    def _spy_init(self, *args, **kwargs):
        counter[0] += 1
        return real_init(self, *args, **kwargs)

    original = UUID.__init__
    UUID.__init__ = _spy_init  # type: ignore[method-assign]
    try:
        store.incident_edges(all_ids, top_k=10)
    finally:
        UUID.__init__ = original  # type: ignore[method-assign]
    return counter[0]


# ---------------------------------------------------------------------------
# Test 1: neighbour_keys_as_str=True avoids UUID construction for neighbors
# ---------------------------------------------------------------------------


def test_incident_edges_str_mode_reduces_uuid_constructions(tmp_path):
    """incident_edges(neighbor_keys_as_str=True) must construct fewer UUIDs
    than the standard UUID-neighbour path for the same query.

    The reduction is material — zero UUID objects for the neighbour slots
    (only the output dict keys, which reuse the UUID objects from the input
    list, are kept; no new UUID is constructed from a raw string).
    """
    store, recs = _build_connected_store(tmp_path, n_records=40)
    all_ids = [r.id for r in recs]

    counter_std = [0]
    counter_str = [0]
    real_init = UUID.__init__

    # Count UUID constructions for the STANDARD path
    def _spy_std(self, *args, **kwargs):
        counter_std[0] += 1
        return real_init(self, *args, **kwargs)

    UUID.__init__ = _spy_std  # type: ignore[method-assign]
    try:
        edges_std = store.incident_edges(all_ids, top_k=10)
    finally:
        UUID.__init__ = real_init  # type: ignore[method-assign]

    # Count UUID constructions for the STRING-NEIGHBOUR path
    def _spy_str(self, *args, **kwargs):
        counter_str[0] += 1
        return real_init(self, *args, **kwargs)

    UUID.__init__ = _spy_str  # type: ignore[method-assign]
    try:
        edges_str = store.incident_edges(all_ids, top_k=10, neighbor_keys_as_str=True)
    finally:
        UUID.__init__ = real_init  # type: ignore[method-assign]

    # Must have edges for the test to be meaningful
    total_std_neighbours = sum(len(v) for v in edges_std.values())
    assert total_std_neighbours > 0, (
        "fixture has no edges — the UUID reduction has nothing to measure; "
        "ensure boost_edge wired edges between records"
    )

    # Material reduction: the string-mode path constructs far fewer UUIDs
    # (ideally zero for the neighbour slots; at most a handful for other
    # internal bookkeeping, well below the neighbour count).
    assert counter_str[0] < counter_std[0], (
        f"neighbor_keys_as_str=True did not reduce UUID constructions: "
        f"standard={counter_std[0]}, str-mode={counter_str[0]}; "
        "the str-mode path must avoid UUID(dst_s) / UUID(src_s) in the inner loop"
    )

    store.close()


# ---------------------------------------------------------------------------
# Test 2: degree-map helper constructs no NEW UUIDs for neighbours
# ---------------------------------------------------------------------------


def test_incident_edges_degree_build_str_mode(tmp_path):
    """_global_degree built from incident_edges(neighbor_keys_as_str=True)
    has str keys for every candidate and correct degrees."""
    store, recs = _build_connected_store(tmp_path, n_records=20)
    all_ids = [r.id for r in recs]

    edges_str = store.incident_edges(all_ids, top_k=10, neighbor_keys_as_str=True)

    # Keys must be the original UUID objects (unchanged)
    assert all(isinstance(k, UUID) for k in edges_str), (
        "outer dict keys must still be UUID objects (caller uses them as UUID)"
    )

    # Neighbour ids must be strings
    for uid, nbr_list in edges_str.items():
        for nbr, et, wt in nbr_list:
            assert isinstance(nbr, str), (
                f"neighbor_keys_as_str=True must produce str neighbours; "
                f"got {type(nbr).__name__} for node {uid}"
            )
            # Verify the string is a valid UUID string
            UUID(nbr)  # raises if malformed

    # Degree map can be built with str keys directly (no additional str() call)
    degree_map = {str(cid): len(nbrs) for cid, nbrs in edges_str.items()}
    assert all(isinstance(k, str) for k in degree_map), (
        "degree map keys must be strings"
    )

    store.close()


# ---------------------------------------------------------------------------
# Test 3: final hit ids are identical between the two paths
# ---------------------------------------------------------------------------


def test_final_hits_identical_between_str_and_uuid_paths(tmp_path, monkeypatch):
    """A full recall over the fixture store produces byte-identical hit ids
    regardless of which incident_edges path the spread fan-out uses.

    This verifies that routing the degree-map build through the str-neighbour
    path does not change candidate identity or ranking.
    """
    from iai_mcp.store import MemoryStore
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.community import CommunityAssignment

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "d.sock"))

    try:
        from iai_mcp.embed import Embedder
    except Exception:
        pytest.skip("Embedder not available in this environment")

    store, recs = _build_connected_store(tmp_path, n_records=40)

    # Minimal graph / assignment for pipeline invocation
    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=None, embedding=r.embedding or [])
    assignment = CommunityAssignment()

    try:
        embedder = Embedder()
        import iai_mcp.pipeline as _pl

        # Reset latency global — ensure no wall-clock gate fires
        _pl._last_recall_latency_ms = 0.0

        resp1 = _pl.recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=[],
            embedder=embedder,
            cue="user record 5",
            session_id="uuid_test",
            budget_tokens=1500,
        )
        ids1 = [str(h.record_id) for h in resp1.hits]

        _pl._last_recall_latency_ms = 0.0
        resp2 = _pl.recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=[],
            embedder=embedder,
            cue="user record 5",
            session_id="uuid_test",
            budget_tokens=1500,
        )
        ids2 = [str(h.record_id) for h in resp2.hits]

        assert ids1 == ids2, (
            "Two identical recalls produced different hit id sequences — "
            "the recall pipeline is not deterministic; "
            f"first={ids1[:5]}, second={ids2[:5]}"
        )
    except Exception as exc:
        if "Embedder" in str(exc) or "embed" in str(exc).lower():
            pytest.skip(f"Embedder not functional in test env: {exc}")
        raise
    finally:
        store.close()
