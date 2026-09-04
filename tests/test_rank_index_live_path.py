"""Live-path proof for the store-resident rank index adapter.

Guards, in one process against the exact `core.dispatch("memory_recall", ...)`
construction sequence (ANN hydrate -> authority -> hop1/hop2 spread ->
rich-club -> candidate-graph population, `core/__init__.py`):

- the residency source the live recall path actually has available (no
  warm, sync-hook-maintained graph reaches this dispatch branch -- see the
  structural probe below);
- that a second live recall reuses one persistent Rust index without a
  Python-side rebuild;
- that the whole-recall decrypt-call count strictly drops on a second call
  once the resident index has hydrated a prior call's candidates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.store._store import MemoryStore as _StoreClass
from iai_mcp.store._rank_index import _RankIndexHandle
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16  # small synthetic dim -- keeps this a single-file, native-embedder-free test


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake.pop((s, u), None))
    yield fake


@pytest.fixture(autouse=True)
def _authority_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # The exact-cosine authority backstop (k=10) is an EXACT scan: on a tiny
    # corpus it would independently rediscover every record regardless of
    # ANN rank, hydrating the hop-only candidate before hop1 ever sees it as
    # "new" -- disabling it isolates the hop/rich-club residency path.
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")


class _StubEmbedder:
    """Deterministic stand-in embedder -- a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _orthogonal_vec(base: list[float], seed: int) -> list[float]:
    """A unit vector with near-zero cosine to `base` -- far enough from the
    cue that a small ANN k never surfaces it, so it is reachable ONLY via a
    graph edge, matching the live path's hop-discovery shape."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(_DIM).astype(np.float32)
    b = np.asarray(base, dtype=np.float32)
    noise -= float(np.dot(noise, b)) * b
    norm = float(np.linalg.norm(noise))
    if norm < 1e-6:
        noise = np.roll(b, 1) - b
        norm = float(np.linalg.norm(noise)) or 1.0
    return (noise / norm).tolist()


def _make_rec(
    rid: UUID, surface: str, embedding: list[float],
) -> MemoryRecord:
    ts = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=surface,
        aaak_index="",
        embedding=embedding,
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
        created_at=ts,
        updated_at=ts,
        tags=["t"],
        language="en",
    )


def _build_hop_reachable_corpus(store: MemoryStore, cue_vec: list[float]) -> dict[str, UUID]:
    """A seed near the cue, two near-cue fillers (so a small ANN k=3 fills
    on cue-adjacent records only), and a hop target reachable ONLY through a
    hebbian edge from the seed -- production-shaped in structure (a real
    multi-hop graph, not a single-record store), scoped small so this stays
    a single-file test with no native embedder load."""
    ids = {
        "seed": uuid4(),
        "near_1": uuid4(),
        "near_2": uuid4(),
        "hop_target": uuid4(),
        "hop2_target": uuid4(),
    }
    store.insert(_make_rec(ids["seed"], "seed near the cue", cue_vec))
    store.insert(_make_rec(ids["near_1"], "near-cue filler one", cue_vec))
    store.insert(_make_rec(ids["near_2"], "near-cue filler two", cue_vec))
    store.insert(_make_rec(
        ids["hop_target"], "hop target reachable only by edge",
        _orthogonal_vec(cue_vec, 20),
    ))
    store.insert(_make_rec(
        ids["hop2_target"], "hop2 target reachable only by a second edge",
        _orthogonal_vec(cue_vec, 30),
    ))
    flush_record_buffer(store)
    store.boost_edges([(ids["seed"], ids["hop_target"])], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(ids["hop_target"], ids["hop2_target"])], edge_type="hebbian", delta=[5.0])
    return ids


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], session_id: str) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pipeline_mod

    store._build_exact_index_sync()
    _pipeline_mod._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "hop residency probe",
        "session_id": session_id,
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def _prepare(store: MemoryStore, monkeypatch: pytest.MonkeyPatch):
    import iai_mcp.pipeline as _pipeline_mod

    cue_vec = _seeded_vec(1)
    ids = _build_hop_reachable_corpus(store, cue_vec)
    stub_embedder_for_store(monkeypatch, _StubEmbedder(cue_vec))
    # A small ANN k keeps hop_target/hop2_target OUTSIDE the ANN top-k, so
    # they are discoverable ONLY through the hop1/hop2 edge traversal --
    # exactly the residency path exercised below.
    monkeypatch.setattr(_pipeline_mod, "K_CANDIDATES", 3)
    return cue_vec, ids


# ---------------------------------------------------------------------------
# Residency-source structural probe, pinned so a later regression that
# reroutes the live path onto build_runtime_graph's corpus-wide warm graph
# fails loudly. This assertion is about SOURCE STRUCTURE, not the
# presence/absence of a resident index, so it holds both before and after
# the persistence/wiring changes below land.
# ---------------------------------------------------------------------------

def test_memory_recall_dispatch_never_calls_build_runtime_graph_residency_source():
    """No warm, sync-hook-maintained `MemoryGraph` is reachable from the
    live `memory_recall` dispatch branch -- `retrieve.build_runtime_graph`
    is called only by the `topology` and `session_start_payload` branches,
    never by `memory_recall`. Residency for the live recall path therefore
    comes from a dedicated, store-resident builder graph, never the
    corpus-wide warm bundle."""
    import inspect
    from iai_mcp import core as _core

    src = inspect.getsource(_core)
    start = src.index('if method == "memory_recall":')
    end = src.index('if method == "brain_view":')
    memory_recall_branch = src[start:end]
    assert "build_runtime_graph" not in memory_recall_branch, (
        "the live memory_recall path must never source its candidate graph "
        "from retrieve.build_runtime_graph -- that graph is not on this "
        "path; residency here comes from the dedicated store-resident "
        "builder graph instead"
    )
    # The construction this residency source relies on must still be present.
    assert "_rank_builder_graph_for" in src
    assert "rank_index_for" in memory_recall_branch


# ---------------------------------------------------------------------------
# Persistence: a second recall reuses the same Rust index without a
# Python-side rebuild.
# ---------------------------------------------------------------------------

def test_persistent_index_survives_second_recall(_store: MemoryStore, monkeypatch: pytest.MonkeyPatch):
    cue_vec, ids = _prepare(_store, monkeypatch)

    build_calls = {"n": 0}
    orig_build = _RankIndexHandle._build

    def _counting_build(self, graph):
        build_calls["n"] += 1
        return orig_build(self, graph)

    monkeypatch.setattr(_RankIndexHandle, "_build", _counting_build)

    resp1 = _dispatch_recall(_store, cue_vec, session_id="persist-1")
    assert build_calls["n"] == 1, "the first recall must build the index once"
    assert resp1["hits"], "recall must return real hits, not a stub"

    resp2 = _dispatch_recall(_store, cue_vec, session_id="persist-2")
    assert build_calls["n"] == 1, (
        "a second recall on the same store must reuse the persistent index "
        "-- _RankIndexHandle._build must not run a second time"
    )
    assert resp2["hits"], "the second recall must also return real hits, not a stub"

    handle = getattr(_store, "_rank_index_handle", None)
    assert handle is not None
    assert handle._index is not None

    graph1 = getattr(_store, "_rank_builder_graph", None)
    assert graph1 is not None, "the store must carry one persistent builder graph"
    assert ids["hop_target"] in {UUID(str(_l)) for _l in graph1._adj}, (
        "the hop-only candidate discovered on call 1 must be resident in "
        "the builder graph feeding the index"
    )


# ---------------------------------------------------------------------------
# The whole-recall decrypt-call count strictly drops on a second recall,
# with the resident-index feature read observed at runtime.
# ---------------------------------------------------------------------------

def test_live_path_decrypt_count_drops(_store: MemoryStore, monkeypatch: pytest.MonkeyPatch):
    cue_vec, ids = _prepare(_store, monkeypatch)

    decrypt_calls = {"n": 0}
    orig_decrypt = _StoreClass._decrypt_for_record

    def _counting_decrypt(self, record_id, value):
        decrypt_calls["n"] += 1
        return orig_decrypt(self, record_id, value)

    monkeypatch.setattr(_StoreClass, "_decrypt_for_record", _counting_decrypt)

    decrypt_calls["n"] = 0
    resp1 = _dispatch_recall(_store, cue_vec, session_id="decrypt-1")
    baseline_count = decrypt_calls["n"]
    assert baseline_count > 0, "the first (cold) recall must decrypt at least the hop-only candidate"
    assert resp1["hits"], "recall must return real hits, not a stub"

    decrypt_calls["n"] = 0
    resp2 = _dispatch_recall(_store, cue_vec, session_id="decrypt-2")
    second_count = decrypt_calls["n"]
    assert resp2["hits"], "the second recall must also return real hits, not a stub"

    assert second_count < baseline_count, (
        f"whole-recall decrypt-call count must strictly drop on a second "
        f"identical-cue recall once the resident index has hydrated the "
        f"prior call's candidates: baseline={baseline_count} second={second_count}"
    )

    # Resident-index feature read observed at runtime, not inferred: the
    # live path's winner-feature cross-check against the resident Rust
    # index (core/__init__.py) increments this counter once per recall
    # that returned hits.
    assert getattr(_store, "_rank_resident_feature_reads", 0) >= 2, (
        "the live path must read winner features off the resident Rust "
        "index on both recalls, observable via this runtime counter"
    )
