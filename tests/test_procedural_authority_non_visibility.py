"""Contract: a tier=procedural record never surfaces via the secondary
merge points (exact-authority merge, crisis-degraded direct path) that sit
downstream of the concept-mode strip.

The shared predicate _passes_mode_filter gates both merge points; its
"Concept mode applies no tier filter" behavior was correct for the world
where only pattern:-tagged semantic records needed protecting, but leaves
a procedural chunk that wins the always-on exact-cosine authority scan
free to reach the served recall response under the default concept mode.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.core import _passes_mode_filter
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store


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


def _make_proc_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="procedural",
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.7,
        difficulty=0.3,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["chunk", "source:cofire"],
        language="en",
    )


def _make_semantic_pattern_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="semantic",
        literal_surface=surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.0,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["schema", "pattern:capture"],
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

    store._build_exact_index_sync()

    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "irrelevant text, embedder is stubbed",
        "session_id": "procedural-authority-test",
        "budget_tokens": budget,
        "cue_embedding": cue_vec,
    })


def test_procedural_chunk_excluded_from_authority_merge_under_concept_mode(store, monkeypatch):
    cue_vec = _seeded_vec(1)
    proc_id = uuid4()
    proc_rec = _make_proc_rec(proc_id, "procedural chunk cofire pair label", embedding=cue_vec)
    store.insert(proc_rec)

    for i in range(5):
        store.insert(_make_rec(uuid4(), seed=100 + i, surface=f"noise record {i}"))
    flush_record_buffer(store)

    store._build_exact_index_sync()

    assert any(rid == proc_id for rid, _ in store.exact_top_k(cue_vec, k=10)), (
        "test setup: the procedural chunk must provably win the always-on "
        "exact-cosine authority scan before we can assert its exclusion"
    )

    _orig_query_similar = store.query_similar

    def _query_similar_missing_proc(vec, *args, **kwargs):
        pairs = _orig_query_similar(vec, *args, **kwargs)
        return [(r, s) for r, s in pairs if getattr(r, "id", None) != proc_id]

    monkeypatch.setattr(store, "query_similar", _query_similar_missing_proc)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp_with = _dispatch_recall(store, cue_vec)

    assert resp_with["cue_mode"] == "concept", (
        "load-bearing: if the stubbed cue ever classified as verbatim, "
        "exclusion would come from the pre-existing verbatim branch and "
        f"leave this fix unproven; got cue_mode={resp_with['cue_mode']!r}"
    )

    hit_ids = {h["record_id"] for h in resp_with["hits"]}
    anti_hit_ids = {h["record_id"] for h in resp_with.get("anti_hits", [])}
    assert str(proc_id) not in hit_ids, (
        f"procedural chunk must not surface via the authority merge; hits={hit_ids}"
    )
    assert str(proc_id) not in anti_hit_ids

    # Budget non-inflation: the SAME store, with store.exact_top_k also
    # patched to drop proc_id for this comparison call, so the ANN tail
    # (already identical via the query_similar drop above) and the
    # candidate-pool cardinality are identical between the two calls —
    # the only remaining variable is whether the authority head can see
    # the chunk at all.
    _orig_exact_top_k = store.exact_top_k

    def _exact_top_k_missing_proc(vec, *args, **kwargs):
        pairs = _orig_exact_top_k(vec, *args, **kwargs)
        return [(rid, s) for rid, s in pairs if rid != proc_id]

    monkeypatch.setattr(store, "exact_top_k", _exact_top_k_missing_proc)

    resp_without = _dispatch_recall(store, cue_vec)

    assert resp_with["budget_used"] == resp_without["budget_used"], (
        f"excluding the procedural chunk must not change budget_used: "
        f"with={resp_with['budget_used']}, without={resp_without['budget_used']}"
    )


def test_passes_mode_filter_excludes_procedural_unconditionally():
    proc_rec = _make_proc_rec(uuid4(), "chunk label", embedding=_seeded_vec(9))
    assert _passes_mode_filter(proc_rec, "concept") is False
    assert _passes_mode_filter(proc_rec, "verbatim") is False

    episodic_rec = _make_rec(uuid4(), seed=9, surface="plain episodic record")
    assert _passes_mode_filter(episodic_rec, "concept") is True

    semantic_pattern_rec = _make_semantic_pattern_rec(
        uuid4(), "schema pattern hub", embedding=_seeded_vec(10),
    )
    assert _passes_mode_filter(semantic_pattern_rec, "verbatim") is False
