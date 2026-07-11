"""Regression: the full-pool sanitize + L2 renorm must run once per build, and
the once-resolved centrality must preserve seed ranking on an edge-dense corpus.

Part A (renorm-once): the recall path sanitizes (nan_to_num) and L2-normalizes the
whole N x D pool matrix. On the per-recall-renorm path this full pass runs on every
recall, an O(N*D) cost that dominates p95 at scale. The fix normalizes the pool
once at assembly and caches it on the graph; the per-recall guard is only a cheap
finite-check. This part spies ``np.linalg.norm`` and counts the full-pool-sized
passes; RED when they fire M times, GREEN when reuse caps them at <= 1.

Part B (quality trap guard): the perf fix must NOT silently drop the centrality
ranking term. On an EDGE-DENSE corpus build-time betweenness is genuinely nonzero,
and the seed order produced from the build-resolved centrality must match the order
the per-recall recompute would produce. Identical order proves the recompute is pure
waste on a static graph and the fix preserves ranking quality. These invariants are
GREEN today and must stay GREEN after the fix lands.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import _normalize_pool_cached, _pick_seeds, recall_for_response
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


def _make_rec(vec: list[float], text: str, tags: list[str]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
    )


def _seed_store_connected(path, n: int, seed: int = 0):
    """Seed records AND a connected edge backbone + chords so build-time
    betweenness is genuinely nonzero (mirrors the connected-corpus pattern in
    test_runtime_graph_centrality_child.py)."""
    from tests.test_pipeline_perf import _PerfEmbedder
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=path)
    embedder = _PerfEmbedder(base_seed=seed)
    tag_pool = [
        ["topic:auth"], ["topic:db"], ["topic:web"],
        ["topic:net"], ["topic:cli"],
    ]
    ids: list[UUID] = []
    for i in range(n):
        vec = embedder.embed(f"alice-seed-{i}")
        tags = list(tag_pool[i % len(tag_pool)])
        rec = _make_rec(vec, text=f"alice fact {i}", tags=tags)
        store.insert(rec)
        ids.append(rec.id)

    backbone = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(backbone, delta=1.0, edge_type="hebbian")
    rng = np.random.default_rng(seed)
    chords: list[tuple[UUID, UUID]] = []
    for _ in range(max(1, n // 10)):
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a != b:
            chords.append((ids[a], ids[b]))
    if chords:
        store.boost_edges(chords, delta=1.0, edge_type="hebbian")

    graph, assignment, rich_club = build_runtime_graph(store)
    return store, embedder, graph, assignment, rich_club, ids


def test_full_pool_renorm_not_repeated_per_recall(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Part A: the full-pool sanitize + L2 renorm must run at most once across M
    recalls. RED when the per-recall renorm runs M times; GREEN once the
    normalized pool is cached on the graph and reused."""
    from tests.test_pipeline_perf import _seed_store

    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=300, seed=0,
    )

    real_norm = np.linalg.norm
    pool_size = graph.node_count()
    full_pool_norm_calls = {"n": 0}

    def _counting_norm(x, *args, **kwargs):
        arr = np.asarray(x)
        # A full-pool renorm passes the whole N x D matrix with axis=1.
        if (
            arr.ndim == 2
            and arr.shape[0] == pool_size
            and kwargs.get("axis", None) == 1
        ):
            full_pool_norm_calls["n"] += 1
        return real_norm(x, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", _counting_norm)

    cues = [
        "what did alice cover about auth yesterday?",
        "explain the db migration plan",
        "how does the web cache invalidation work",
        "summary of the cli subcommand changes",
        "recent network stack bug report",
        "alice notes on the indexing rebuild",
    ]
    m = len(cues)

    for cue in cues:
        recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=embedder,
            cue=cue,
            session_id="renorm_once",
            budget_tokens=1500,
        )

    assert full_pool_norm_calls["n"] <= 1, (
        f"full-pool L2 renorm ran {full_pool_norm_calls['n']} times over {m} "
        f"recalls; it must run at most once per build and be reused thereafter."
    )


def _build_centrality_arr(graph, pool_ids):
    arr = np.zeros(len(pool_ids), dtype=np.float32)
    for i, rid in enumerate(pool_ids):
        arr[i] = float(graph.get_centrality(rid))
    return arr


def test_edge_dense_centrality_nonzero_and_seed_parity(tmp_path):
    """Part B: on an edge-dense corpus build-time centrality is nonzero, and the
    seed order from the build-resolved centrality matches the order the per-recall
    recompute would produce. GREEN today; must stay GREEN after the fix."""
    from iai_mcp.centrality_approx import centrality_for_runtime

    store, embedder, graph, assignment, rich_club, ids = _seed_store_connected(
        tmp_path, n=300, seed=7,
    )

    # Degenerate-fixture guard: the connected backbone + chords must yield at
    # least one genuinely nonzero build-time betweenness value, else the parity
    # check below is vacuous.
    nodes = list(graph.iter_nodes())
    build_centrality = {nid: graph.get_centrality(nid) for nid in nodes}
    assert any(v != 0.0 for v in build_centrality.values()), (
        "edge-dense corpus produced all-zero build-time centrality — the fixture "
        "is degenerate and the seed-parity guard would be vacuous."
    )

    # Pool order mirrors the pipeline (graph.iter_nodes() with present embeddings).
    pool_ids = [nid for nid in nodes if graph.get_embedding(nid)]
    assert pool_ids, "no pooled embeddings on the connected corpus"

    # Resolved arm: read centrality straight from the build-resolved map.
    resolved_arr = _build_centrality_arr(graph, pool_ids)

    # Baseline arm: force a fresh per-recall recompute and read that map.
    recomputed = centrality_for_runtime(graph)
    recomputed_arr = np.array(
        [float(recomputed.get(rid, 0.0)) for rid in pool_ids], dtype=np.float32
    )

    # A representative candidate slice (cosine-top proxy): rank all pooled ids by
    # a fixed cue so the seed selection has a real candidate set.
    cue_vec = np.asarray(embedder.embed("alice auth db web"), dtype=np.float32)
    cue_vec = cue_vec / (np.linalg.norm(cue_vec) or 1.0)
    pool_embs = np.array(
        [graph.get_embedding(rid) for rid in pool_ids], dtype=np.float32
    )
    pool_embs = pool_embs / (
        np.linalg.norm(pool_embs, axis=1)[:, None].clip(min=1e-12)
    )
    shared_cos = pool_embs @ cue_vec
    candidate_indices = np.argsort(-shared_cos, kind="stable")[:64].astype(np.int64)

    seeds_resolved = _pick_seeds(candidate_indices, shared_cos, resolved_arr, n=3)
    seeds_recomputed = _pick_seeds(
        candidate_indices, shared_cos, recomputed_arr, n=3
    )

    assert list(seeds_resolved) == list(seeds_recomputed), (
        "seed order from the build-resolved centrality differs from the per-recall "
        "recompute order; the fix would silently change ranking quality.\n"
        f"resolved={list(seeds_resolved)} recomputed={list(seeds_recomputed)}"
    )


def test_pool_cache_key_is_content_sensitive_not_id_only():
    """WR-01 fence: the normalized-pool cache key must encode embedding CONTENT,
    not just the id-sequence. A content change that leaves the id-sequence intact
    must invalidate the cache so a stale normalized matrix (= silently wrong
    cosine) can never be served.

    This isolates the cache-KEY contract: a node embedding is mutated in place via
    a graph mutator (same id-sequence), and the cached ``_normalized_pool`` entry
    is re-warmed AFTER the mutation so the only thing standing between a warm hit
    and a stale result is whether the KEY noticed the content change. Under
    identity-only keying the re-warmed key matches and the stale V1 matrix is
    served (RED). Under the content-version key the mutator bumped the version, so
    the re-warmed key carries V2 and the V1 entry can never alias it (GREEN).
    """
    a, b = uuid4(), uuid4()
    graph = MemoryGraph()
    # id-sequence is [a, b] and never changes across the whole test.
    graph.add_node(a, community_id=None, embedding=[1.0, 0.0])
    graph.add_node(b, community_id=None, embedding=[0.0, 1.0])
    pool_ids = [a, b]
    probe = np.array([1.0, 0.0], dtype=np.float32)

    # --- V1: node `a` aligned with the probe (cosine 1.0). Warm the cache. ---
    pool_embs_v1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    normalized_v1 = _normalize_pool_cached(graph, pool_ids, pool_embs_v1)
    assert float(normalized_v1[0] @ probe) == pytest.approx(1.0)
    # Capture the exact key the implementation stored for the warm V1 entry.
    warm_key_v1, warm_mat_v1 = graph._normalized_pool
    assert warm_mat_v1 is normalized_v1

    # --- Mutate `a` in place THROUGH the mutator (same id-sequence). A correct
    # implementation nulls _normalized_pool here; we re-pin the STALE V1 entry
    # under its ORIGINAL key right after, modelling the WR-01 regression class —
    # a content change reached the pool without the warm cache being dropped, so
    # the KEY is the only thing that can reject the stale matrix. ---
    graph.set_node_payload(a, {"embedding": [0.0, 1.0]})
    graph._normalized_pool = (warm_key_v1, normalized_v1)  # re-pin stale entry

    pool_embs_v2 = np.array(
        [graph.get_embedding(a), graph.get_embedding(b)], dtype=np.float32
    )
    normalized_v2 = _normalize_pool_cached(graph, pool_ids, pool_embs_v2)

    cos_v2 = float(normalized_v2[0] @ probe)
    # Node `a` is now orthogonal to the probe → fresh cosine is ~0. A stale V1
    # warm hit would return 1.0. Under an id-only key the re-pinned key still
    # matches (RED). Under the content-version key the mutator advanced the
    # version, so the re-pinned V1 key is stale and is rejected (GREEN).
    assert cos_v2 == pytest.approx(0.0, abs=1e-6), (
        f"stale V1 normalized pool served (cosine {cos_v2}, expected ~0); the "
        "cache key is content-blind — it matched the id-sequence alone and "
        "ignored the in-place embedding change."
    )
    assert normalized_v2 is not normalized_v1, (
        "the returned matrix is the cached V1 object; the content change with an "
        "unchanged id-sequence did not invalidate the cache."
    )
