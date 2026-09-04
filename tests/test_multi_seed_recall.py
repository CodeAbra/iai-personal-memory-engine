"""Exercise for multi-seed widening.

``_pick_seeds`` stays a fixed cosine+centrality blend (n=3); the recall path
unconditionally unions additional seed ids from ``fts_hits`` (an
exact-substring match already computed for ranking) into the seed set,
capped at ``MULTI_SEED_CAP``, unless ``IAI_MCP_MULTI_SEED_OFF=true``.

Dual-driver: every test parametrizes ``LILLI_STORAGE_DRIVER`` since each
dispatch exercises a real ``MemoryStore`` insert + recall round-trip.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.pipeline import MULTI_SEED_CAP
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder


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
def _clear_authority_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_EXACT_AUTHORITY_OFF", raising=False)


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


class _StubEmbedder:
    """Deterministic stand-in embedder — a fixed cue vector regardless of text."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed(self, _text: str) -> list[float]:
        return list(self._vec)


def _seeded_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _orthogonal_vec(seed: int) -> list[float]:
    """A unit vector with near-zero cosine similarity to _seeded_vec(seed)."""
    rng = np.random.default_rng(seed + 9000)
    v = rng.random(_DIM).astype(np.float32) - 0.5
    return (v / np.linalg.norm(v)).tolist()


def _make_rec(rid: UUID, surface: str, embedding: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
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
        created_at=now,
        updated_at=now,
        tags=["capture"],
        language="en",
    )


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    stub_embedder_for_store(monkeypatch, _StubEmbedder(vec))


def _dispatch_recall(store: MemoryStore, cue_vec: list[float], cue_text: str) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": cue_text,
        "session_id": "multi-seed-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _make_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


_FTS_MARKER = "zzz-exact-substring-marker-zzz"


def _mild_cosine_vec(cue_vec: list[float], seed: int, weight: float = 0.3) -> list[float]:
    """A vector with SOME cosine similarity to cue_vec -- dominates the
    cosine+centrality seed blend without being an exact match."""
    rng = np.random.default_rng(seed)
    noise = rng.random(_DIM).astype(np.float32) - 0.5
    v = np.array(cue_vec, dtype=np.float32) * weight + noise * (1.0 - weight)
    return (v / np.linalg.norm(v)).tolist()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_fts_hit_promoted_to_seed(driver, _make_store, monkeypatch):
    """An fts_hits id (exact substring match, near-zero cosine to the cue,
    NOT among the base-3 cosine+centrality seeds) MUST be promoted into the
    seed set — observable via activation_trace (union of seed_ids and their
    2-hop spread)."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _make_store(driver)
    cue_vec = _seeded_vec(41)
    # fts_hits matches when the WHOLE cue text is a substring of a record's
    # literal_surface — so the cue itself must be the marker, contained
    # verbatim inside the target record's surface.
    cue_text = _FTS_MARKER

    # The fts target: near-zero cosine to the cue (orthogonal), so the base
    # cosine+centrality blend does NOT naturally pick it as one of the top 3.
    fts_id = uuid4()
    fts_rec = _make_rec(fts_id, f"contains {_FTS_MARKER} verbatim", _orthogonal_vec(41))
    store.insert(fts_rec)
    for i in range(8):
        store.insert(_make_rec(
            uuid4(), f"unrelated filler {i}", _mild_cosine_vec(cue_vec, 900 + i),
        ))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_recall(store, cue_vec, cue_text)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "multi_seed" in marks, (
        f"expected multi_seed trace mark, got {sorted(marks)}"
    )
    assert str(fts_id) in resp["activation_trace"], (
        "fts_hits id was not promoted into the seed set (or its 2-hop spread)"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_multi_seed_off_toggle_disables_promotion(driver, _make_store, monkeypatch):
    """IAI_MCP_MULTI_SEED_OFF=true disables the fts_hits promotion and its
    trace mark, even on the same fixture that promotes by default."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    monkeypatch.setenv("IAI_MCP_MULTI_SEED_OFF", "true")
    store = _make_store(driver)
    cue_vec = _seeded_vec(51)
    cue_text = _FTS_MARKER

    fts_id = uuid4()
    fts_rec = _make_rec(fts_id, f"contains {_FTS_MARKER} verbatim", _orthogonal_vec(51))
    store.insert(fts_rec)
    for i in range(8):
        store.insert(_make_rec(
            uuid4(), f"unrelated filler {i}", _mild_cosine_vec(cue_vec, 800 + i),
        ))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_recall(store, cue_vec, cue_text)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "multi_seed" not in marks, (
        f"multi_seed mark fired with the toggle set, got {sorted(marks)}"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_seed_count_never_exceeds_multi_seed_cap(driver, _make_store, monkeypatch):
    """Even with many fts_hits candidates, the total seed count post-union
    must never exceed MULTI_SEED_CAP."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _make_store(driver)
    cue_vec = _seeded_vec(61)
    cue_text = _FTS_MARKER  # fts_hits matches when the cue is a substring of the surface

    fts_ids = []
    for i in range(10):
        fid = uuid4()
        fts_ids.append(fid)
        store.insert(_make_rec(fid, f"contains {_FTS_MARKER} copy {i}", _orthogonal_vec(700 + i)))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    import iai_mcp.pipeline as _pm
    spy_calls: list[list] = []
    _orig_two_hop = None

    def _spy_two_hop(self, seed_ids, top_k=5):
        spy_calls.append(list(seed_ids))
        return _orig_two_hop(self, seed_ids, top_k=top_k)

    from iai_mcp.graph import MemoryGraph
    _orig_two_hop = MemoryGraph.two_hop_neighborhood.__get__(None, MemoryGraph)
    monkeypatch.setattr(MemoryGraph, "two_hop_neighborhood", _spy_two_hop, raising=True)

    _dispatch_recall(store, cue_vec, cue_text)

    assert spy_calls, "two_hop_neighborhood was never called"
    for seeds in spy_calls:
        assert len(seeds) <= MULTI_SEED_CAP, (
            f"seed count {len(seeds)} exceeded MULTI_SEED_CAP={MULTI_SEED_CAP}"
        )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_no_new_store_calls_from_multi_seed_widen(driver, _make_store, monkeypatch):
    """The fts_hits union reuses ids already resident in the pool/records
    cache — it must not trigger any new store round trip."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _make_store(driver)
    cue_vec = _seeded_vec(71)
    cue_text = _FTS_MARKER  # fts_hits matches when the cue is a substring of the surface

    fts_id = uuid4()
    store.insert(_make_rec(fts_id, f"contains {_FTS_MARKER} verbatim", _orthogonal_vec(71)))
    for i in range(8):
        store.insert(_make_rec(
            uuid4(), f"unrelated filler {i}", _mild_cosine_vec(cue_vec, 971 + i),
        ))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    _orig_get_batch = store.get_batch
    calls = {"n": 0}

    def _counting_get_batch(*a, **k):
        calls["n"] += 1
        return _orig_get_batch(*a, **k)

    monkeypatch.setattr(store, "get_batch", _counting_get_batch)

    resp = _dispatch_recall(store, cue_vec, cue_text)
    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "multi_seed" in marks
    # get_batch may legitimately be called for anti-hits/pool fallback, but the
    # multi-seed union itself (a pure id_to_idx lookup) must not add an extra
    # per-seed store round trip. This is a smoke bound, not a strict zero.
    # One additional bounded call backfills session_id/captured_at on any
    # graph/rank-view-sourced finalist (anti-hit or hit); it is O(finalists),
    # never O(seeds) or O(candidate pool).
    assert calls["n"] <= 4
