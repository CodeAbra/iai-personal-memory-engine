"""RED guard for SC-3 (confidence-gated escalation trigger).

``iai_mcp.pipeline.escalate_recall_candidates`` (a bounded, single-shot
ANN-widening step, hard-capped at ``ADAPTIVE_ESCALATION_CAP``) and its
confidence signal ``compute_spread_depth`` (top cosine / hit-count-above-
threshold) both exist today with ZERO production callers — the module's own
comment block documents "NOT YET ON THE RECALL PATH" (``pipeline.py:499``).

This phase (plan 03) wires ``escalate_recall_candidates`` to fire when
``compute_spread_depth`` signals low confidence (top cosine <
``_CONF_HIGH_COSINE_THRESHOLD=0.75`` and hit-count-above-threshold <
``_CONF_HIGH_HIT_COUNT_MIN=3``), and to NOT fire on a high-confidence cue.

- Test 1 (low-confidence cue): RED today — the spy is never called because
  the function has zero callers.
- Test 2 (high-confidence cue): GREEN today by default (zero callers means
  it's never called either way) — this test locks the invariant that a
  high-confidence cue must continue to skip the widen once plan 03 wires
  the low-confidence branch, so the gate doesn't regress into an always-on
  merge.

Dual-driver: both tests parametrize ``LILLI_STORAGE_DRIVER`` since each
exercises a real ``MemoryStore`` insert + recall round-trip.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.pipeline import (
    ADAPTIVE_ESCALATION_CAP,
    _CONF_HIGH_COSINE_THRESHOLD,
    _CONF_HIGH_HIT_COUNT_MIN,
    escalate_recall_candidates,
)
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


def _dispatch_recall(store: MemoryStore, cue_vec: list[float]) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "confidence-escalation-gate test cue, embedder is stubbed",
        "session_id": "confidence-escalation-gate-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _make_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


def test_low_confidence_signal_shape_below_threshold():
    """Sanity: compute_spread_depth returns 1 (low confidence) for a top
    cosine below the high-confidence threshold with few hits — pins the
    exact scalars this file's fixtures are designed to reproduce."""
    depth = compute_spread_depth_import()(
        top_cosine=_CONF_HIGH_COSINE_THRESHOLD - 0.2,
        hit_count_above_threshold=_CONF_HIGH_HIT_COUNT_MIN - 2,
    )
    assert depth == 1


def test_high_confidence_signal_shape_above_threshold():
    """Sanity: compute_spread_depth returns 0 (high confidence) for a top
    cosine at/above the threshold with enough hits."""
    depth = compute_spread_depth_import()(
        top_cosine=_CONF_HIGH_COSINE_THRESHOLD + 0.2,
        hit_count_above_threshold=_CONF_HIGH_HIT_COUNT_MIN + 2,
    )
    assert depth == 0


def compute_spread_depth_import():
    from iai_mcp.pipeline import compute_spread_depth
    return compute_spread_depth


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_low_confidence_cue_triggers_escalation(driver, _make_store, monkeypatch):
    """A cue whose top cosine is below threshold and whose hit-count-above-
    threshold is low MUST trigger escalate_recall_candidates at least once.

    RED today: escalate_recall_candidates has zero production callers, so
    the spy is never invoked regardless of confidence.
    """
    store = _make_store(driver)
    cue_vec = _seeded_vec(11)

    target_id = uuid4()
    target = _make_rec(target_id, "the associative target record", cue_vec)
    store.insert(target)
    # Deliberately sparse, low-similarity filler so the top-cosine stays low
    # and hit-count-above-threshold stays under _CONF_HIGH_HIT_COUNT_MIN.
    for i in range(2):
        store.insert(_make_rec(uuid4(), f"unrelated filler {i}", _orthogonal_vec(11 + i)))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    spy = MagicMock(wraps=escalate_recall_candidates)
    monkeypatch.setattr("iai_mcp.pipeline.escalate_recall_candidates", spy)

    _dispatch_recall(store, cue_vec)

    assert spy.call_count >= 1, (
        "escalate_recall_candidates was not called on a low-confidence cue — "
        "confidence-gated escalation (plan 03) not yet wired onto the recall path"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_high_confidence_cue_does_not_trigger_escalation(driver, _make_store, monkeypatch):
    """A cue whose top cosine is at/above threshold with enough corroborating
    hits must NOT trigger escalate_recall_candidates.

    Passes today (zero callers means it's never invoked either way) and must
    remain green once plan 03 wires the low-confidence branch — the gate
    must correctly skip the widen on a high-confidence cue.
    """
    store = _make_store(driver)
    cue_vec = _seeded_vec(21)

    target_id = uuid4()
    target = _make_rec(target_id, "the associative target record", cue_vec)
    store.insert(target)
    # A dense cluster of near-duplicate embeddings around the cue vector so
    # top cosine stays high and hit-count-above-threshold clears the min.
    for i in range(5):
        nudge = (np.array(cue_vec, dtype=np.float32) * 0.97
                 + np.random.default_rng(500 + i).normal(0, 0.01, _DIM).astype(np.float32))
        nudge = nudge / np.linalg.norm(nudge)
        store.insert(_make_rec(uuid4(), f"near-duplicate {i}", nudge.tolist()))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    spy = MagicMock(wraps=escalate_recall_candidates)
    monkeypatch.setattr("iai_mcp.pipeline.escalate_recall_candidates", spy)

    _dispatch_recall(store, cue_vec)

    assert spy.call_count == 0, (
        "escalate_recall_candidates was called on a high-confidence cue — "
        "the gate must skip the widen when confidence is already high"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recall_read_path_no_o_corpus_widen(driver, _make_store, monkeypatch):
    """Structural guard: a low-confidence recall MUST NEVER call
    store.all_records(), store._exact_scan_full_corpus() (a write-time-only
    helper), or query_similar with an unbounded k on the read path. The
    confidence-gated widen is bounded to ADAPTIVE_ESCALATION_CAP — it must
    never reopen a full-corpus scan on the awake recall path.
    """
    store = _make_store(driver)
    cue_vec = _seeded_vec(31)

    target_id = uuid4()
    target = _make_rec(target_id, "the associative target record", cue_vec)
    store.insert(target)
    for i in range(2):
        store.insert(_make_rec(uuid4(), f"unrelated filler {i}", _orthogonal_vec(31 + i)))
    flush_record_buffer(store)

    _stub_embedder_for_store(monkeypatch, cue_vec)

    def _raise_all_records(*_a, **_k):
        raise AssertionError("store.all_records() called on the recall path")
    monkeypatch.setattr(store, "all_records", _raise_all_records)

    if hasattr(store, "_exact_scan_full_corpus"):
        def _raise_exact_scan(*_a, **_k):
            raise AssertionError(
                "store._exact_scan_full_corpus() called on the recall path"
            )
        monkeypatch.setattr(store, "_exact_scan_full_corpus", _raise_exact_scan)

    _orig_query_similar = store.query_similar

    def _bounded_query_similar(vec, k=10, *args, **kwargs):
        if k > ADAPTIVE_ESCALATION_CAP + 100:
            raise AssertionError(
                f"query_similar called with unbounded k={k} on the recall path"
            )
        return _orig_query_similar(vec, k, *args, **kwargs)
    monkeypatch.setattr(store, "query_similar", _bounded_query_similar)

    _dispatch_recall(store, cue_vec)
