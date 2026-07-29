"""Activation transfer at rank: a graph-reached candidate inherits a decayed
fraction of its originating seed's cue-cosine. Spread ADMITS structurally
linked candidates; without the transfer term they are ranked by their own
cosine alone and drown even when the link is exactly why they are relevant.
The term is weight-gated: absent weight, ranking must stay byte-identical.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import recall_for_benchmark
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _graph_from_edges(
    edges: list[tuple[str, str, float]], edge_type: str = "hebbian"
) -> MemoryGraph:
    g = MemoryGraph()
    for src, dst, w in edges:
        g.add_edge(UUID(src), UUID(dst), weight=w, edge_type=edge_type)
    return g


def _uuids(n: int) -> list[str]:
    return sorted(str(uuid4()) for _ in range(n))


class TestProvenanceTraversal:
    def test_reached_set_matches_plain_variant(self):
        a, b, c, d, e = _uuids(5)
        g = _graph_from_edges(
            [(a, b, 1.0), (b, c, 0.9), (c, d, 0.8), (a, e, 0.5)]
        )
        seeds = [UUID(a)]
        plain = g.two_hop_neighborhood(seeds, top_k=5)
        prov = g.two_hop_neighborhood_with_provenance(seeds, top_k=5)
        assert plain == sorted(prov, key=str)

    def test_hop_depths_and_seed_attribution(self):
        a, b, c, d = _uuids(4)
        g = _graph_from_edges(
            [(a, b, 1.0), (b, c, 1.0), (c, d, 1.0)], edge_type="entity_shared"
        )
        prov = g.two_hop_neighborhood_with_provenance([UUID(a)], top_k=5)
        assert prov[UUID(b)] == (UUID(a), 1, True)
        assert prov[UUID(c)] == (UUID(a), 2, True)
        assert UUID(d) not in prov, "three hops away must stay unreached"

    def test_first_discoverer_attribution_is_deterministic(self):
        s1, s2, shared = _uuids(3)
        g = _graph_from_edges([(s1, shared, 1.0), (s2, shared, 1.0)])
        for _ in range(3):
            prov = g.two_hop_neighborhood_with_provenance(
                [UUID(s2), UUID(s1)], top_k=5
            )
            # Sorted frontier sweep: the lexicographically first seed wins.
            assert prov[UUID(shared)][:2] == (UUID(min(s1, s2)), 1)

    def test_hebbian_path_does_not_carry_transfer(self):
        a, b = _uuids(2)
        g = _graph_from_edges([(a, b, 1.0)], edge_type="hebbian")
        prov = g.two_hop_neighborhood_with_provenance([UUID(a)], top_k=5)
        assert prov[UUID(b)] == (UUID(a), 1, False)

    def test_mixed_path_loses_transfer_at_the_non_entity_hop(self):
        a, b, c = _uuids(3)
        g = MemoryGraph()
        g.add_edge(UUID(a), UUID(b), weight=1.0, edge_type="entity_shared")
        g.add_edge(UUID(b), UUID(c), weight=1.0, edge_type="hebbian")
        prov = g.two_hop_neighborhood_with_provenance([UUID(a)], top_k=5)
        assert prov[UUID(b)] == (UUID(a), 1, True)
        assert prov[UUID(c)] == (UUID(a), 2, False)

    def test_seeds_never_reported_as_reached(self):
        a, b = _uuids(2)
        g = _graph_from_edges([(a, b, 1.0)])
        prov = g.two_hop_neighborhood_with_provenance([UUID(a), UUID(b)], top_k=5)
        assert prov == {}


class _FakeEmbedder:
    DIM = EMBED_DIM

    def __init__(self, vec: "list[float]") -> None:
        self._vec = vec

    def embed(self, text: str) -> "list[float]":
        return list(self._vec)

    def embed_batch(self, texts: "list[str]") -> "list[list[float]]":
        return [list(self._vec) for _ in texts]


def _one_hot(i: int) -> "list[float]":
    vec = [0.0] * EMBED_DIM
    vec[i] = 1.0
    return vec


def _make(vec: "list[float]", text: str) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=vec, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=now, updated_at=now, tags=[], language="en",
    )


def _blend(cos_to_e0: float, other_dim: int) -> "list[float]":
    vec = [0.0] * EMBED_DIM
    vec[0] = cos_to_e0
    vec[other_dim] = (1.0 - cos_to_e0 * cos_to_e0) ** 0.5
    return vec


@pytest.fixture()
def linked_setup(tmp_path):
    store = MemoryStore(path=tmp_path / "lancedb")
    seed = _make(_one_hot(0), "seed record about warehouse kappa")
    linked = _make(_one_hot(1), "overflow inventory racks in building nine")
    off = _make(_one_hot(2), "cafeteria lasagna rotates on thursdays")
    # Decoys outrank linked/off on cosine so the top-3 seed cut never
    # swallows them — a record picked as seed is excluded from spread.
    # Distinct cosines: a tie would make their relative order flip on the
    # age-penalty drift between two recall calls.
    decoy_a = _make(_blend(0.55, 3), "decoy record alpha")
    decoy_b = _make(_blend(0.5, 4), "decoy record beta")
    recs = [seed, linked, off, decoy_a, decoy_b]
    for r in recs:
        store.insert(r)
    flush_record_buffer(store)

    graph = MemoryGraph()
    for r in recs:
        graph.add_node(r.id, community_id=None, embedding=list(r.embedding))
        graph.set_node_payload(r.id, {
            "embedding": list(r.embedding), "surface": r.literal_surface,
            "centrality": 0.0, "tier": r.tier, "tags": [], "language": "en",
        })
    graph.add_edge(seed.id, linked.id, weight=1.0, edge_type="entity_shared")

    cid = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: _one_hot(0)},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )
    return store, graph, assignment, seed, linked, off


def _recall(store, graph, assignment, k_hits: int = 10):
    return recall_for_benchmark(
        store=store, graph=graph, assignment=assignment, rich_club=[],
        embedder=_FakeEmbedder(_one_hot(0)), cue="warehouse kappa shipments",
        session_id="bench", k_hits=k_hits,
    )


def _score_of(resp, rid):
    for h in resp.hits:
        if h.record_id == rid:
            return h.score
    return None


class TestTransferTerm:
    def test_weight_off_is_default_and_changes_nothing(
        self, linked_setup, monkeypatch
    ):
        store, graph, assignment, *_ = linked_setup
        base = _recall(store, graph, assignment)
        monkeypatch.setenv("IAI_MCP_W_SPREAD_ACT", "0")
        off = _recall(store, graph, assignment)
        assert [h.record_id for h in base.hits] == [h.record_id for h in off.hits]
        # The age penalty ticks with wall-clock between the two calls, so
        # scores match only approximately.
        for hb, ho in zip(base.hits, off.hits):
            assert ho.score == pytest.approx(hb.score, abs=1e-4)

    def test_transfer_lifts_graph_reached_candidate_only(
        self, linked_setup, monkeypatch
    ):
        store, graph, assignment, seed, linked, off = linked_setup
        base = _recall(store, graph, assignment)
        monkeypatch.setenv("IAI_MCP_W_SPREAD_ACT", "0.3")
        boosted = _recall(store, graph, assignment)

        base_linked = _score_of(base, linked.id)
        boosted_linked = _score_of(boosted, linked.id)
        assert base_linked is not None and boosted_linked is not None
        # Seed cosine 1.0, one hop: transfer = 0.3 * 1.0 * 0.6. The age
        # penalty drifts a hair between calls, hence the loose tolerance.
        assert boosted_linked - base_linked == pytest.approx(0.18, abs=1e-4)

        for rid in (seed.id, off.id):
            b, a = _score_of(base, rid), _score_of(boosted, rid)
            if b is not None and a is not None:
                assert a == pytest.approx(b, abs=1e-4), (
                    "only the graph-reached candidate may gain from transfer"
                )
