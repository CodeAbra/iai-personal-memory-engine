from __future__ import annotations

from uuid import UUID, uuid4

from iai_mcp.community import detect_communities
from iai_mcp.graph import MemoryGraph
from iai_mcp.richclub import rich_club_nodes


def test_small_graph_constructs() -> None:
    g = MemoryGraph()
    for _ in range(10):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.node_count() == 10


def test_large_graph_constructs() -> None:
    g = MemoryGraph()
    for _ in range(501):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.node_count() == 501


def test_n_just_below_500_constructs() -> None:
    g = MemoryGraph()
    for _ in range(499):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    assert g.node_count() == 499


def test_two_hop_reaches_exactly_two_hops() -> None:
    g = MemoryGraph()
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for n in (a, b, c, d):
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    g.add_edge(a, b)
    g.add_edge(b, c)
    g.add_edge(c, d)

    reached = set(g.two_hop_neighborhood([a], top_k=5))
    assert b in reached
    assert c in reached
    assert d not in reached
    assert a not in reached


def test_two_hop_multiple_seeds_deduped() -> None:
    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for n in (a, b, c):
        g.add_node(n, community_id=None, embedding=[0.0] * 384)
    g.add_edge(a, b)
    g.add_edge(b, c)
    reached = set(g.two_hop_neighborhood([a, c], top_k=5))
    assert reached == {b}


def test_two_hop_empty_seeds_returns_empty_list() -> None:
    g = MemoryGraph()
    assert g.two_hop_neighborhood([], top_k=5) == []


def test_centrality_hub_beats_leaves() -> None:
    g = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(4)]
    g.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        g.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        g.add_edge(hub, leaf)
    c = g.centrality()
    for leaf in leaves:
        assert c[hub] > c[leaf]


def test_centrality_no_edges_all_zero() -> None:
    g = MemoryGraph()
    for _ in range(5):
        g.add_node(uuid4(), community_id=None, embedding=[0.0] * 384)
    c = g.centrality()
    assert all(v == 0.0 for v in c.values())
    assert len(c) == 5


def test_get_embedding_returns_stored_vector() -> None:
    g = MemoryGraph()
    nid = uuid4()
    emb = [1.0] + [0.0] * 383
    g.add_node(nid, community_id=None, embedding=emb)
    assert g.get_embedding(nid) == emb
    assert g.get_embedding(uuid4()) is None


def test_rich_club_coefficient_on_star_graph() -> None:
    g = MemoryGraph()
    hub = uuid4()
    leaves = [uuid4() for _ in range(4)]
    g.add_node(hub, community_id=None, embedding=[0.0] * 384)
    for leaf in leaves:
        g.add_node(leaf, community_id=None, embedding=[0.0] * 384)
        g.add_edge(hub, leaf)
    coef = g.rich_club_coefficient()
    assert isinstance(coef, float)
    assert coef >= 0.0


def test_edge_attr_symmetric() -> None:
    g = MemoryGraph()
    u, v = uuid4(), uuid4()
    g.add_node(u, None, [0.0] * 384)
    g.add_node(v, None, [0.0] * 384)
    g.add_edge(u, v, weight=0.5)
    assert g._adj[str(u)][str(v)] is g._adj[str(v)][str(u)]
    g._adj[str(u)][str(v)]["weight"] = 9.9
    assert g._adj[str(v)][str(u)]["weight"] == 9.9


def test_iter_edges_once_only() -> None:
    g = MemoryGraph()
    u, v, w = uuid4(), uuid4(), uuid4()
    for n in (u, v, w):
        g.add_node(n, None, [0.0] * 384)
    g.add_edge(u, v)
    g.add_edge(v, w)
    g.add_edge(u, w)
    edges = list(g.iter_edges_with_weight())
    assert len(edges) == 3


def test_self_loop_preserved_in_storage() -> None:
    g = MemoryGraph()
    u = uuid4()
    g.add_node(u, None, [0.0] * 384)
    g.add_edge(u, u, weight=0.5)
    assert str(u) in g._adj[str(u)]
    edges = list(g.iter_edges_with_weight())
    assert sum(1 for src, dst, _ in edges if src == dst == u) == 1
    _indptr, indices, _data = g.to_csr_arrays()
    assert len(indices) == 0


# --- clear_and_rebuild parity / anti-stale -------------------------------

def _uid(n: int) -> UUID:
    return UUID(int=n)


def _topology_one() -> tuple[list, list]:
    # A hub (id 1) connected to four leaves (2..5).
    nodes = [
        (_uid(i), None, [float(i)] + [0.0] * 383, {"surface": f"n{i}", "tier": "episodic"})
        for i in range(1, 6)
    ]
    edges = [(_uid(1), _uid(i), 1.0, "hebbian") for i in range(2, 6)]
    return nodes, edges


def _topology_two() -> tuple[list, list]:
    # A different shape: two triangles (10,11,12) and (13,14,15) with no bridge.
    ids = list(range(10, 16))
    nodes = [
        (_uid(i), None, [float(i)] + [0.0] * 383, {"surface": f"m{i}", "tier": "episodic"})
        for i in ids
    ]
    edges = [
        (_uid(10), _uid(11), 1.0, "hebbian"),
        (_uid(11), _uid(12), 1.0, "hebbian"),
        (_uid(12), _uid(10), 1.0, "hebbian"),
        (_uid(13), _uid(14), 1.0, "hebbian"),
        (_uid(14), _uid(15), 1.0, "hebbian"),
        (_uid(15), _uid(13), 1.0, "hebbian"),
    ]
    return nodes, edges


def _fresh(nodes: list, edges: list) -> MemoryGraph:
    g = MemoryGraph()
    for node_id, community_id, embedding, payload in nodes:
        g.add_node(node_id, community_id=community_id, embedding=embedding)
        g.set_node_payload(node_id, payload)
    for src, dst, weight, edge_type in edges:
        g.add_edge(src, dst, weight=weight, edge_type=edge_type)
    return g


def _partition(node_to_community: dict) -> frozenset:
    # detect_communities mints a fresh community UUID per call, so the raw
    # node_to_community values are not stable across builds; the GROUPING is.
    groups: dict = {}
    for node_id, community_id in node_to_community.items():
        groups.setdefault(community_id, set()).add(str(node_id))
    return frozenset(frozenset(members) for members in groups.values())


def _adjacency_snapshot(g: MemoryGraph) -> dict:
    return {
        u: sorted(
            (v, float(attrs.get("weight", 1.0)), str(attrs.get("edge_type", "hebbian")))
            for v, attrs in neighbors.items()
        )
        for u, neighbors in sorted(g._adj.items())
    }


def test_clear_and_rebuild_byte_identical_to_fresh() -> None:
    nodes, edges = _topology_one()

    fresh = _fresh(nodes, edges)

    reused = MemoryGraph()
    # Seed reused with an unrelated topology first, then rebuild.
    other_nodes, other_edges = _topology_two()
    reused.clear_and_rebuild(other_nodes, other_edges)
    reused.clear_and_rebuild(nodes, edges)

    assert sorted(reused.iter_nodes(), key=str) == sorted(fresh.iter_nodes(), key=str)
    assert _adjacency_snapshot(reused) == _adjacency_snapshot(fresh)
    assert sorted(reused.iter_edges_with_weight()) == sorted(fresh.iter_edges_with_weight())
    assert sorted(reused.degrees(), key=lambda x: str(x[0])) == sorted(
        fresh.degrees(), key=lambda x: str(x[0])
    )
    assert max(d for _n, d in reused.degrees()) == max(d for _n, d in fresh.degrees())
    assert reused.centrality() == fresh.centrality()

    a_reused = detect_communities(reused, prior_mode="cold").node_to_community
    a_fresh = detect_communities(fresh, prior_mode="cold").node_to_community
    assert _partition(a_reused) == _partition(a_fresh)
    assert sorted(rich_club_nodes(reused), key=str) == sorted(
        rich_club_nodes(fresh), key=str
    )


def test_clear_and_rebuild_anti_stale_recomputes_derived(monkeypatch) -> None:
    monkeypatch.setenv("IAI_MCP_CENTRALITY_CACHE", "on")

    nodes1, edges1 = _topology_one()
    nodes2, edges2 = _topology_two()

    g = MemoryGraph()
    g.clear_and_rebuild(nodes1, edges1)
    topo1_centrality = g.centrality()
    assert g._centrality_cache is not None  # first build populated the cache

    g.clear_and_rebuild(nodes2, edges2)
    # del ran: the lazily-built CSR label order must be gone before recompute.
    assert not hasattr(g, "_node_ids_csr_order")

    topo2_centrality = g.centrality()
    fresh2 = _fresh(nodes2, edges2)
    assert topo2_centrality == fresh2.centrality()
    # The reused graph did NOT serve topology-1's cached centrality.
    assert set(topo2_centrality.keys()) != set(topo1_centrality.keys())

    reused_comm = detect_communities(g, prior_mode="cold").node_to_community
    fresh_comm = detect_communities(fresh2, prior_mode="cold").node_to_community
    assert _partition(reused_comm) == _partition(fresh_comm)


def test_clear_and_rebuild_empty_yields_empty_graph() -> None:
    nodes, edges = _topology_one()
    g = _fresh(nodes, edges)
    assert g.node_count() > 0

    g.clear_and_rebuild([], [])
    assert g.node_count() == 0
    assert g.centrality() == {}


def test_degrees_filtered_by_edge_type() -> None:
    g = MemoryGraph()
    hub, a, b, c = uuid4(), uuid4(), uuid4(), uuid4()
    for nid in (hub, a, b, c):
        g.add_node(nid, community_id=None, embedding=[0.0] * 384)
    g.add_edge(hub, a, weight=0.9, edge_type="hebbian")
    g.add_edge(hub, b, weight=0.7, edge_type="pattern_separation_seed")
    g.add_edge(hub, c, weight=0.8, edge_type="pattern_separation_seed")

    all_deg = dict(g.degrees())
    assert all_deg[hub] == 3

    only_hebb = dict(g.degrees(edge_types=frozenset({"hebbian"})))
    assert only_hebb[hub] == 1
    assert only_hebb[b] == 0

    earned = dict(
        g.degrees(exclude_types=frozenset({"pattern_separation_seed"}))
    )
    assert earned[hub] == 1
    assert earned[a] == 1
    assert earned[b] == 0
