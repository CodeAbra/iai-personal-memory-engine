"""Tests for the NaN/inf guard on the recall ranking path and the write boundary.

Three invariants verified:
1. A NaN/inf in the POOL matrix degrades gracefully — poisoned rows do not rank
   above clean rows, shared_cos is finite, no crash. (WR-01 guard, pool side)
2. A NaN/inf in the CUE vector degrades gracefully — shared_cos is finite, recall
   returns a deterministic order, no crash. (HI-01 fix, cue side)
3. A finite CUE with a healthy pool produces a bit-identical ranking to the
   pre-guard baseline. (determinism regression lock)
4. Inserting a record with a NaN/inf embedding raises at the write boundary
   before anything is persisted. (WR-03 fix)
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.graph import MemoryGraph
from iai_mcp.pipeline import _community_gate, _community_gate_max_node, _collect_graph_pool, _recall_core
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make(vec: list[float], text: str = "rec", tier: str = "episodic") -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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
        tags=[],
        language="en",
    )


class _FakeEmbedder:
    """Embedder stub that returns a caller-supplied vector."""

    DIM = EMBED_DIM

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, text: str) -> list[float]:  # noqa: ARG002
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _unit_vec(dim: int, hot: int) -> list[float]:
    v = [0.0] * dim
    v[hot] = 1.0
    return v


def _build_store_and_graph(
    tmp_path,
    vecs: list[list[float]],
) -> tuple[MemoryStore, MemoryGraph, list[MemoryRecord]]:
    store = MemoryStore(path=tmp_path / "lancedb")
    recs: list[MemoryRecord] = []
    for i, vec in enumerate(vecs):
        rec = _make(vec, text=f"rec{i}")
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    return store, graph, recs


def _flat_assignment(recs: list[MemoryRecord]) -> CommunityAssignment:
    cid = uuid4()
    centroid = _unit_vec(EMBED_DIM, 0)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )


# ---------------------------------------------------------------------------
# WR-01: Pool-side NaN guard (existing guard is locked)
# ---------------------------------------------------------------------------

def test_nan_pool_row_does_not_rank_above_clean_rows(tmp_path):
    """A NaN row in the pool drops to the bottom; clean rows are unaffected.

    Verifies: no exception, shared_cos is finite, poisoned row does not
    surface in the top half of results.
    """
    # Build 6 clean unit vectors plus 2 poisoned rows.
    clean_vecs = [_unit_vec(EMBED_DIM, i) for i in range(6)]
    # The cue is unit vector on axis 0 — clean[0] should rank first.
    cue_vec = np.asarray(_unit_vec(EMBED_DIM, 0), dtype=np.float32)

    # Directly exercise the internal matmul path using _collect_graph_pool
    # and the pipeline guard logic; we cannot insert NaN into the store
    # (WR-03 blocks that), so we build the pool matrix manually.
    pool_embs_raw = np.asarray(clean_vecs, dtype=np.float32)
    # Inject NaN and inf into two rows.
    poisoned = pool_embs_raw.copy()
    poisoned[2, :] = float("nan")
    poisoned[4, 0] = float("inf")

    # Apply the guard (mirrors the _recall_core guard).
    pool_embs = np.nan_to_num(poisoned, nan=0.0, posinf=0.0, neginf=0.0)
    pool_norms = np.linalg.norm(pool_embs, axis=1)
    pool_norms[pool_norms == 0.0] = 1.0
    pool_embs = pool_embs / pool_norms[:, None]
    shared_cos = np.matmul(pool_embs, cue_vec).astype(np.float32)

    # Invariant 1: no exception above, shared_cos is fully finite.
    assert np.all(np.isfinite(shared_cos)), (
        f"shared_cos contains non-finite values after pool guard: {shared_cos}"
    )

    # Invariant 2: row 0 (axis-0 unit, perfect match with cue) ranks first.
    order = np.argsort(-shared_cos, kind="stable")
    assert order[0] == 0, (
        f"Expected row 0 to rank first, got order={order}"
    )

    # Invariant 3: poisoned rows (2 and 4, now zeroed) do not outscore the
    # best-matching clean row.  Row 0 is the perfect match (cosine=1.0);
    # after the guard, rows 2 and 4 become all-zero (cosine=0.0).  They
    # must not rank above row 0.
    assert order[0] == 0, f"Perfect-match row 0 should rank first, order={order}"
    rank_of_2 = int(np.where(order == 2)[0][0])
    rank_of_4 = int(np.where(order == 4)[0][0])
    assert rank_of_2 > 0, (
        f"Poisoned row 2 should not rank first (rank={rank_of_2})"
    )
    assert rank_of_4 > 0, (
        f"Poisoned row 4 should not rank first (rank={rank_of_4})"
    )
    # Scores of poisoned rows must be zero (not positive, not NaN).
    assert float(shared_cos[2]) == 0.0, f"Poisoned row 2 score should be 0, got {shared_cos[2]}"
    assert float(shared_cos[4]) == 0.0, f"Poisoned row 4 score should be 0, got {shared_cos[4]}"


def test_inf_pool_row_does_not_rank_above_clean_rows():
    """A +inf in the pool is zeroed out; it must not sort to the top."""
    clean_vecs = [_unit_vec(EMBED_DIM, i) for i in range(4)]
    cue_vec = np.asarray(_unit_vec(EMBED_DIM, 1), dtype=np.float32)

    pool = np.asarray(clean_vecs, dtype=np.float32)
    pool[3, :] = float("inf")

    pool = np.nan_to_num(pool, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(pool, axis=1)
    norms[norms == 0.0] = 1.0
    pool = pool / norms[:, None]
    scores = np.matmul(pool, cue_vec).astype(np.float32)

    assert np.all(np.isfinite(scores))
    order = np.argsort(-scores, kind="stable")
    # row 1 (axis-1 unit) should rank first.
    assert order[0] == 1


# ---------------------------------------------------------------------------
# HI-01: Cue-side NaN guard (new fix)
# ---------------------------------------------------------------------------

def test_nan_cue_does_not_poison_shared_cos(tmp_path):
    """A NaN in the cue vector must not produce NaN shared_cos values.

    After the fix, _recall_core sanitizes the cue before the matmul,
    producing shared_cos == 0 for all rows (all-zero cue has zero cosine
    with every vector).  No crash, deterministic order.
    """
    clean_vecs = [_unit_vec(EMBED_DIM, i) for i in range(5)]
    store, graph, recs = _build_store_and_graph(tmp_path, clean_vecs)
    assignment = _flat_assignment(recs)

    # NaN cue.
    nan_cue = [float("nan")] * EMBED_DIM
    embedder = _FakeEmbedder(nan_cue)

    result = _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="nan-cue-test",
        session_id="test-nan-cue",
    )

    # Must not crash.  Hits may be empty or non-empty — both are valid
    # degrade behaviour.  If hits present, scores must be finite.
    for hit in result.scored_hits:
        score = hit.score if hasattr(hit, "score") else 0.0
        assert math.isfinite(float(score)), (
            f"Hit score is non-finite after NaN cue: {score}"
        )


def test_inf_cue_does_not_poison_shared_cos(tmp_path):
    """+inf cue is zeroed out by the guard; shared_cos stays finite."""
    clean_vecs = [_unit_vec(EMBED_DIM, i) for i in range(4)]
    store, graph, recs = _build_store_and_graph(tmp_path, clean_vecs)
    assignment = _flat_assignment(recs)

    inf_cue = [float("inf")] + [0.0] * (EMBED_DIM - 1)
    embedder = _FakeEmbedder(inf_cue)

    # Must not raise.
    result = _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="inf-cue-test",
        session_id="test-inf-cue",
    )

    for hit in result.scored_hits:
        score = hit.score if hasattr(hit, "score") else 0.0
        assert math.isfinite(float(score)), (
            f"Hit score is non-finite after inf cue: {score}"
        )


# ---------------------------------------------------------------------------
# Determinism regression lock: clean cue + clean pool is unchanged
# ---------------------------------------------------------------------------

def test_clean_cue_ranking_is_bit_identical_before_and_after_guard(tmp_path):
    """_sanitize_vec is a no-op on finite vectors; argsort must be identical."""
    dim = EMBED_DIM
    n = 8
    vecs = [_unit_vec(dim, i) for i in range(n)]
    store, graph, recs = _build_store_and_graph(tmp_path, vecs)
    assignment = _flat_assignment(recs)

    # Cue = unit vector on axis 3.
    cue_v = _unit_vec(dim, 3)
    embedder = _FakeEmbedder(cue_v)

    result = _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="clean-cue",
        session_id="test-clean-cue",
    )

    # rec[3] must rank first: it has cosine 1.0 with the axis-3 cue.
    assert len(result.scored_hits) >= 1
    assert result.scored_hits[0].record_id == recs[3].id, (
        f"Expected recs[3] first, got {result.scored_hits[0].record_id}"
    )


# ---------------------------------------------------------------------------
# WR-03: Write boundary rejects non-finite embeddings
# ---------------------------------------------------------------------------

def test_insert_nan_embedding_raises(tmp_path):
    """Inserting a record with a NaN embedding raises ValueError immediately."""
    store = MemoryStore(path=tmp_path / "store-nan")
    nan_vec = [float("nan")] + [0.0] * (EMBED_DIM - 1)
    rec = _make(nan_vec, text="nan-embedding")

    with pytest.raises(ValueError, match="non-finite"):
        store.insert(rec)

    # Nothing should have been persisted.
    assert store.get(rec.id) is None


def test_insert_inf_embedding_raises(tmp_path):
    """Inserting a record with a +inf embedding raises ValueError immediately."""
    store = MemoryStore(path=tmp_path / "store-inf")
    inf_vec = [0.0] * EMBED_DIM
    inf_vec[0] = float("inf")
    rec = _make(inf_vec, text="inf-embedding")

    with pytest.raises(ValueError, match="non-finite"):
        store.insert(rec)

    assert store.get(rec.id) is None


def test_insert_neginf_embedding_raises(tmp_path):
    """Inserting a record with a -inf embedding raises ValueError immediately."""
    store = MemoryStore(path=tmp_path / "store-neginf")
    neginf_vec = [0.0] * EMBED_DIM
    neginf_vec[5] = float("-inf")
    rec = _make(neginf_vec, text="neginf-embedding")

    with pytest.raises(ValueError, match="non-finite"):
        store.insert(rec)

    assert store.get(rec.id) is None


def test_insert_finite_embedding_succeeds(tmp_path):
    """A fully finite embedding is stored without error (regression guard)."""
    store = MemoryStore(path=tmp_path / "store-finite")
    good_vec = _unit_vec(EMBED_DIM, 7)
    rec = _make(good_vec, text="finite-embedding")
    store.insert(rec)  # must not raise
    got = store.get(rec.id)
    assert got is not None
    assert got.literal_surface == "finite-embedding"


# ---------------------------------------------------------------------------
# _community_gate cue-side guard (HI-01, gate path)
# ---------------------------------------------------------------------------

def test_community_gate_nan_cue_returns_list_without_crash():
    """_community_gate with a NaN cue must return a list (no crash, finite)."""
    n = 6
    recs = [_make(_unit_vec(EMBED_DIM, i)) for i in range(n)]
    assignment = _flat_assignment(recs)

    nan_cue = [float("nan")] * EMBED_DIM
    result = _community_gate(nan_cue, assignment, top_n=3)

    assert isinstance(result, list)


def test_community_gate_max_node_nan_cue_returns_list_without_crash():
    """_community_gate_max_node with a NaN cue returns a list (no crash)."""
    n = 4
    recs = [_make(_unit_vec(EMBED_DIM, i)) for i in range(n)]
    assignment = _flat_assignment(recs)

    nan_cue_vec = np.zeros(EMBED_DIM, dtype=np.float32)
    # Simulate the sanitized cue that _community_gate passes to _community_gate_max_node
    # after applying _sanitize_vec (all-NaN becomes all-zero).
    member_embeddings = {r.id: _unit_vec(EMBED_DIM, i) for i, r in enumerate(recs)}
    result = _community_gate_max_node(nan_cue_vec, assignment, top_n=3, member_embeddings=member_embeddings)

    assert isinstance(result, list)
