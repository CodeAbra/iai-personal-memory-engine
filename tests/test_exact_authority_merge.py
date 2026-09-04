"""Contract for merge_authority_hits and its production wiring.

merge_authority_hits is a pure function: exact-cosine authority hits form the
head (given order, never re-ranked), the graph pipeline's associative hits
follow as the tail (existing order preserved, including pending-recency
markers), and the token budget-pack is re-applied over the union so no tail
hit can ever consume budget before a head hit.

The second half of this file wires the merge into iai_mcp.core.dispatch's
recall branch: always-on authority, env kill-switch, fail-safe degrade to
pipeline-only on any internal fault, and a live-id-set liveness filter so a
stale matrix (or a missed invalidate seam) can never surface a tombstoned
record through the authority path.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from iai_mcp.pipeline import merge_authority_hits, _POST_RANK_MAX_HITS
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord
from tests._helpers import stub_embedder_for_store


# ---------------------------------------------------------------------------
# Part 1: merge_authority_hits — pure-function contract
# ---------------------------------------------------------------------------


def _hit(rid: str, score: float, reason: str = "cosine", surface: str = "") -> MemoryHit:
    return MemoryHit(
        record_id=rid,
        score=score,
        reason=reason,
        literal_surface=surface,
        adjacent_suggestions=[],
    )


def test_never_demoted_authority_first_regardless_of_pipeline_score():
    authority = [_hit("a1", 0.10), _hit("a2", 0.05)]
    pipeline = [_hit("p1", 0.99), _hit("p2", 0.95)]

    hits, _ = merge_authority_hits(pipeline, authority, budget_tokens=100_000)

    assert [h.record_id for h in hits] == ["a1", "a2", "p1", "p2"]


def test_dedup_keeps_authority_hit_object_at_head_position():
    authority = [_hit("shared", 0.77, reason="exact-cosine", surface="AUTH SURFACE")]
    pipeline = [
        _hit("other", 0.99),
        _hit("shared", 0.11, reason="graph-rank", surface="PIPELINE SURFACE"),
    ]

    hits, _ = merge_authority_hits(pipeline, authority, budget_tokens=100_000)

    assert [h.record_id for h in hits] == ["shared", "other"]
    shared_hit = hits[0]
    assert shared_hit.reason == "exact-cosine"
    assert shared_hit.literal_surface == "AUTH SURFACE"
    assert shared_hit.score == 0.77


def test_dedup_copies_displaced_tail_hits_adjacent_suggestions_onto_head():
    """When a record appears in both head (authority) and tail (pipeline),
    the surviving head hit must inherit the displaced tail hit's
    ``adjacent_suggestions`` — a fresh authority hit is built with an empty
    list, so this association data would otherwise be silently lost to the
    authority promotion."""
    graph_derived_suggestions = [uuid4(), uuid4()]
    authority = [_hit("shared", 0.77, reason="exact-cosine", surface="AUTH SURFACE")]
    pipeline_shared = _hit("shared", 0.11, reason="graph-rank", surface="PIPELINE SURFACE")
    pipeline_shared.adjacent_suggestions = graph_derived_suggestions
    pipeline = [_hit("other", 0.99), pipeline_shared]

    hits, _ = merge_authority_hits(pipeline, authority, budget_tokens=100_000)

    assert [h.record_id for h in hits] == ["shared", "other"]
    shared_hit = hits[0]
    # Head identity (reason/surface/score) is preserved exactly as before —
    # only adjacent_suggestions is backfilled.
    assert shared_hit.reason == "exact-cosine"
    assert shared_hit.literal_surface == "AUTH SURFACE"
    assert shared_hit.adjacent_suggestions == graph_derived_suggestions


def test_dedup_does_not_overwrite_a_head_hit_that_already_has_suggestions():
    """If an authority hit is ever built with its own adjacent_suggestions,
    the displaced tail hit's suggestions must NOT clobber it."""
    own_suggestions = [uuid4()]
    authority_hit = _hit("shared", 0.77, reason="exact-cosine", surface="AUTH SURFACE")
    authority_hit.adjacent_suggestions = own_suggestions
    authority = [authority_hit]
    pipeline_shared = _hit("shared", 0.11, reason="graph-rank", surface="PIPELINE SURFACE")
    pipeline_shared.adjacent_suggestions = [uuid4(), uuid4()]
    pipeline = [pipeline_shared]

    hits, _ = merge_authority_hits(pipeline, authority, budget_tokens=100_000)

    assert hits[0].adjacent_suggestions == own_suggestions


def test_associative_tail_preserved_including_pending_recency_markers():
    authority = [_hit("a1", 0.9)]
    pipeline = [
        _hit("p1", 0.8),
        _hit("marker1", 0.0, reason="pending-recency"),
        _hit("p2", 0.7),
    ]

    hits, _ = merge_authority_hits(pipeline, authority, budget_tokens=100_000)

    assert [h.record_id for h in hits] == ["a1", "p1", "marker1", "p2"]
    assert hits[2].reason == "pending-recency"


def test_budget_pack_authority_consumes_budget_first():
    # Each surface is 40 chars -> 10 tokens (len // 4).
    authority = [_hit("a1", 0.9, surface="A" * 40)]
    pipeline = [_hit("p1", 0.8, surface="B" * 40), _hit("p2", 0.7, surface="C" * 40)]

    hits, budget_used = merge_authority_hits(pipeline, authority, budget_tokens=25)

    assert [h.record_id for h in hits] == ["a1", "p1"]
    assert budget_used == 20


def test_budget_pack_overflow_breaks_with_at_least_one_kept():
    authority = [_hit("a1", 0.9, surface="A" * 4000)]
    pipeline = [_hit("p1", 0.8, surface="B" * 4000)]

    hits, budget_used = merge_authority_hits(pipeline, authority, budget_tokens=1)

    assert [h.record_id for h in hits] == ["a1"]
    assert budget_used == 1000


def test_at_least_one_authority_hit_survives_tiny_budget():
    authority = [_hit("a1", 0.9, surface="A" * 400)]
    pipeline = []

    hits, budget_used = merge_authority_hits(pipeline, authority, budget_tokens=1)

    assert [h.record_id for h in hits] == ["a1"]
    assert budget_used == 100


def test_max_hits_cap_truncates_union_head_intact():
    authority = [_hit(f"a{i}", 1.0 - i * 0.001) for i in range(5)]
    pipeline = [_hit(f"p{i}", 0.5 - i * 0.001) for i in range(_POST_RANK_MAX_HITS)]

    hits, _ = merge_authority_hits(
        pipeline, authority, budget_tokens=1_000_000, max_hits=_POST_RANK_MAX_HITS,
    )

    assert len(hits) == _POST_RANK_MAX_HITS
    assert [h.record_id for h in hits[:5]] == [f"a{i}" for i in range(5)]


def test_identity_fast_path_empty_authority_returns_pipeline_unchanged():
    pipeline = [_hit("p1", 0.8, surface="hello world")]

    hits, budget_used = merge_authority_hits(pipeline, [], budget_tokens=100_000)

    assert hits is pipeline
    assert budget_used == sum(len(h.literal_surface) // 4 for h in pipeline)


def test_determinism_same_inputs_same_output():
    authority = [_hit("a1", 0.9), _hit("a2", 0.5)]
    pipeline = [_hit("p1", 0.8), _hit("p2", 0.1)]

    hits1, budget1 = merge_authority_hits(list(pipeline), list(authority), budget_tokens=1000)
    hits2, budget2 = merge_authority_hits(list(pipeline), list(authority), budget_tokens=1000)

    assert [h.record_id for h in hits1] == [h.record_id for h in hits2]
    assert budget1 == budget2


# ---------------------------------------------------------------------------
# Part 2: production wiring — core.dispatch recall branch
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture(autouse=True)
def _clear_authority_kill_switch(monkeypatch):
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)
    yield


class _StubEmbedder:
    """Deterministic stand-in embedder — a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, seed: int, surface: str, *, embedding: list[float] | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding if embedding is not None else _seeded_vec(seed),
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
        tags=["capture"],
        language="en",
    )


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def _stub_embedder_for_store(monkeypatch, vec: list[float]) -> None:
    stub_embedder_for_store(monkeypatch, _StubEmbedder(vec))


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], budget: int = 2000) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    # These tests exercise the authority MERGE wiring, which requires a WARM
    # exact-cosine matrix: the recall dispatch never builds it synchronously
    # (a cold matrix kicks a background build and the recall proceeds
    # authority-degraded), so warm it deterministically here.
    store._build_exact_index_sync()

    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "irrelevant text, embedder is stubbed",
        "session_id": "authority-wiring-test",
        "budget_tokens": budget,
        "cue_embedding": cue_vec,
    })


def test_authority_hit_surfaces_at_head_when_ann_tier_misses_it(store, monkeypatch):
    cue_vec = _seeded_vec(1)
    target_id = uuid4()
    target = _make_rec(target_id, seed=1, surface="the exact target record", embedding=cue_vec)
    store.insert(target)

    for i in range(5):
        store.insert(_make_rec(uuid4(), seed=100 + i, surface=f"noise record {i}"))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    # Simulate an ANN-tier miss: query_similar never returns the target, but
    # exact_top_k (the authority) still finds it via the resident matrix.
    _orig_query_similar = store.query_similar

    def _query_similar_missing_target(vec, *args, **kwargs):
        pairs = _orig_query_similar(vec, *args, **kwargs)
        return [(r, s) for r, s in pairs if getattr(r, "id", None) != target_id]

    monkeypatch.setattr(store, "query_similar", _query_similar_missing_target)

    resp = _dispatch_recall(store, cue_vec)

    assert resp["hits"], f"expected non-empty hits, got {resp}"
    assert resp["hits"][0]["record_id"] == str(target_id), (
        f"authority hit must surface at head; got hits={resp['hits']}"
    )
    reason = resp["hits"][0]["reason"]
    assert reason.startswith("exact-cosine")
    # When the pipeline rank outscored the bare cosine, the served score is
    # the pipeline's and the reason must carry that provenance.
    if " | score from pipeline rank: " not in reason:
        assert reason == "exact-cosine"
    assert resp.get("exact_authority_used") is True


def test_associative_pipeline_hits_preserved_behind_authority_head(store, monkeypatch):
    cue_vec = _seeded_vec(2)
    target_id = uuid4()
    target = _make_rec(target_id, seed=2, surface="authority head record", embedding=cue_vec)
    store.insert(target)

    associative_id = uuid4()
    store.insert(_make_rec(associative_id, seed=3, surface="associative pipeline record"))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_recall(store, cue_vec)

    hit_ids = [h["record_id"] for h in resp["hits"]]
    assert str(target_id) in hit_ids
    assert str(associative_id) in hit_ids, (
        f"associative pipeline hit must be preserved behind the authority head; got {hit_ids}"
    )


def test_stale_matrix_tombstone_fence_id_never_surfaces(store, monkeypatch):
    cue_vec = _seeded_vec(3)
    victim_id = uuid4()
    victim = _make_rec(victim_id, seed=3, surface="soon to be tombstoned", embedding=cue_vec)
    store.insert(victim)

    for i in range(3):
        store.insert(_make_rec(uuid4(), seed=200 + i, surface=f"filler {i}"))
    flush_record_buffer(store)

    # Warm the matrix so it holds the victim's row.
    warm_pairs = store.exact_top_k(cue_vec, k=5)
    assert any(rid == victim_id for rid, _ in warm_pairs), (
        "test setup: victim must be present in the warmed matrix before tombstoning"
    )

    # Tombstone via a direct SQL UPDATE, bypassing invalidate_exact_index() —
    # this simulates a missed invalidate seam, leaving the resident matrix
    # stale. exact_top_k on the stale matrix still returns the victim's id;
    # only the live-id-set SQL filter can catch it from here.
    with store.db._conn_lock:
        store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), str(victim_id)),
        )

    stale_pairs = store.exact_top_k(cue_vec, k=5)
    assert any(rid == victim_id for rid, _ in stale_pairs), (
        "test setup: the stale matrix must still return the tombstoned id "
        "from exact_top_k directly (proves the fence lives in dispatch, not "
        "in exact_top_k itself)"
    )

    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_recall(store, cue_vec)

    hit_ids = {h["record_id"] for h in resp["hits"]}
    anti_hit_ids = {h["record_id"] for h in resp.get("anti_hits", [])}
    assert str(victim_id) not in hit_ids, (
        f"tombstoned id must never surface via the authority path; hits={hit_ids}"
    )
    assert str(victim_id) not in anti_hit_ids


def test_kill_switch_restores_pipeline_only_behavior(store, monkeypatch):
    cue_vec = _seeded_vec(4)
    target_id = uuid4()
    target = _make_rec(target_id, seed=4, surface="killswitch target", embedding=cue_vec)
    store.insert(target)
    flush_record_buffer(store)

    _orig_query_similar = store.query_similar

    def _query_similar_missing_target(vec, *args, **kwargs):
        pairs = _orig_query_similar(vec, *args, **kwargs)
        return [(r, s) for r, s in pairs if getattr(r, "id", None) != target_id]

    monkeypatch.setattr(store, "query_similar", _query_similar_missing_target)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")

    resp = _dispatch_recall(store, cue_vec)

    hit_ids = {h["record_id"] for h in resp["hits"]}
    assert str(target_id) not in hit_ids, (
        "kill-switch must restore byte-identical pre-change behavior: the "
        "ANN-missed target must NOT be recovered by the authority"
    )
    assert not resp.get("exact_authority_used", False)


def test_fail_safe_exact_top_k_raises_degrades_to_pipeline_only(store, monkeypatch):
    cue_vec = _seeded_vec(5)
    store.insert(_make_rec(uuid4(), seed=5, surface="ordinary record", embedding=cue_vec))
    flush_record_buffer(store)

    def _boom(*_a, **_kw):
        raise RuntimeError("synthetic exact_top_k failure")

    monkeypatch.setattr(store, "exact_top_k", _boom)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_recall(store, cue_vec)

    assert "hits" in resp
    assert not resp.get("exact_authority_used", False)


def test_exact_authority_used_false_when_merge_resolves_no_hits(store, monkeypatch):
    """exact_authority_used must reflect whether the merge actually folded in
    an authority hit, not merely whether seeding found authority candidates.
    Simulate the id-set liveness re-check racing a tombstone between seed
    and merge time: exact_top_k still returns a pair (seeding succeeds), but
    every authority id is unresolvable by the time the merge's get_batch
    runs, so _auth_hits stays empty and the merge never fires."""
    cue_vec = _seeded_vec(8)
    target_id = uuid4()
    target = _make_rec(target_id, seed=8, surface="authority seed only", embedding=cue_vec)
    store.insert(target)
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    real_get_batch = store.get_batch
    seed_call_done = {"n": 0}

    def _get_batch_empty_on_second_call(ids):
        seed_call_done["n"] += 1
        if seed_call_done["n"] == 1:
            return real_get_batch(ids)
        # Second call is the merge-time re-fetch of live authority ids:
        # simulate it resolving nothing (record vanished/unresolvable
        # between seed and merge), so _auth_hits stays empty.
        return {}

    monkeypatch.setattr(store, "get_batch", _get_batch_empty_on_second_call)

    resp = _dispatch_recall(store, cue_vec)

    assert "hits" in resp
    assert not resp.get("exact_authority_used", False), (
        "exact_authority_used must be False when the merge resolved zero "
        "authority hits, even though seeding found an authority candidate"
    )


def test_reinforce_receives_merged_hit_ids(store, monkeypatch):
    cue_vec = _seeded_vec(6)
    target_id = uuid4()
    target = _make_rec(target_id, seed=6, surface="reinforced target", embedding=cue_vec)
    store.insert(target)
    flush_record_buffer(store)

    _orig_query_similar = store.query_similar

    def _query_similar_missing_target(vec, *args, **kwargs):
        pairs = _orig_query_similar(vec, *args, **kwargs)
        return [(r, s) for r, s in pairs if getattr(r, "id", None) != target_id]

    monkeypatch.setattr(store, "query_similar", _query_similar_missing_target)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    reinforced: list = []
    _orig_queue_reinforce = store.queue_reinforce

    def _spy_queue_reinforce(ids):
        reinforced.extend(ids)
        return _orig_queue_reinforce(ids)

    monkeypatch.setattr(store, "queue_reinforce", _spy_queue_reinforce)

    _dispatch_recall(store, cue_vec)

    assert target_id in reinforced, (
        f"queue_reinforce must receive the merged (authority-included) hit ids; got {reinforced}"
    )


def test_authority_hit_surface_matches_full_decrypted_surface(store, monkeypatch):
    cue_vec = _seeded_vec(7)
    target_id = uuid4()
    real_surface = "the real decrypted literal surface for the authority hit"
    target = _make_rec(target_id, seed=7, surface=real_surface, embedding=cue_vec)
    store.insert(target)
    flush_record_buffer(store)

    _orig_query_similar = store.query_similar

    def _query_similar_missing_target(vec, *args, **kwargs):
        pairs = _orig_query_similar(vec, *args, **kwargs)
        return [(r, s) for r, s in pairs if getattr(r, "id", None) != target_id]

    monkeypatch.setattr(store, "query_similar", _query_similar_missing_target)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_recall(store, cue_vec)

    matching = [h for h in resp["hits"] if h["record_id"] == str(target_id)]
    assert matching, f"authority hit for target not found in {resp['hits']}"
    assert matching[0]["literal_surface"] == real_surface
    assert matching[0]["literal_surface"] == store.get(target_id).literal_surface
