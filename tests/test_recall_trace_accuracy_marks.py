"""RED guards for SC-2 (recall-trace observability of the new accuracy marks).

With ``IAI_MCP_RECALL_TRACE=1`` set, ``core.dispatch``'s ``memory_recall``
branch returns a ``_recall_trace_ms`` list of ``(name, cumulative_ms)`` tuples
(see ``core/__init__.py:_trace_mark``). This phase adds four new named marks
to that trace as the accompanying code plans (02/03/04) wire their mechanisms:

- ``soft_gate`` (plan 02 — graded community-gate bonus)
- ``multi_seed`` (plan 04 — widened seed sources)
- ``cleanup_attractor`` (plan 04 — thresholded cleanup() attractor)
- ``conf_escalate`` (plan 03 — confidence-gated escalation)

Every test below asserts one of these mark names is present in the trace.
All four are RED today (zero occurrences of any of these names anywhere in
``src/iai_mcp/core/__init__.py`` or ``src/iai_mcp/pipeline.py`` — confirmed
by grep this session). No ``working_tier_bias`` mark is asserted anywhere in
this file: working-tier consultation was dropped from the recall path for
this phase (see 184-04-PLAN.md, revision 1, decision D-C).

Dual-driver: every test parametrizes ``LILLI_STORAGE_DRIVER`` since each
dispatch exercises a real ``MemoryStore`` insert + recall round-trip.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord
from tests._helpers import stub_embedder_for_store

_DIM = 16  # small synthetic dim; avoids loading the Rust embedder


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_DIM))


def _select_driver(driver: str, monkeypatch: pytest.MonkeyPatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


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
    """A unit vector with near-zero cosine similarity to _seeded_vec(seed).

    In a low-dimensional (16-d) space, two independently-drawn ``_seeded_vec``
    calls are not reliably low-cosine against each other — the shift by -0.5
    before normalizing is what keeps this genuinely orthogonal-ish, matching
    the low-confidence-inducing filler this fixture's docstring requires.
    """
    rng = np.random.default_rng(seed + 9000)
    v = rng.random(_DIM).astype(np.float32) - 0.5
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


def _stub_embedder_for_store(monkeypatch: pytest.MonkeyPatch, vec: list[float]) -> None:
    stub_embedder_for_store(monkeypatch, _StubEmbedder(vec))


def _dispatch_traced_recall(store: MemoryStore, cue_vec: list[float]) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "low-confidence cue, embedder is stubbed",
        "session_id": "trace-marks-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


@pytest.fixture
def _store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryStore:
    def _make(driver: str) -> MemoryStore:
        _select_driver(driver, monkeypatch)
        return MemoryStore(path=tmp_path / f"store-{driver}")
    return _make


def _seed_small_corpus(store: MemoryStore, cue_vec: list[float]) -> None:
    """A small, low-confidence-inducing corpus: no near-duplicate cluster, so
    the top cosine stays below the high-confidence threshold and the
    hit-count-above-threshold stays low — exactly the low-confidence signal
    the confidence-gated marks (conf_escalate, soft_gate) should fire on."""
    target_id = uuid4()
    target = _make_rec(target_id, seed=1, surface="the associative target record", embedding=cue_vec)
    store.insert(target)
    for i in range(4):
        store.insert(_make_rec(
            uuid4(), seed=100 + i, surface=f"unrelated filler {i}",
            embedding=_orthogonal_vec(100 + i),
        ))
    flush_record_buffer(store)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_trace_contains_soft_gate_mark(driver, _store, monkeypatch):
    """IAI_MCP_RECALL_TRACE=1 trace MUST contain a `soft_gate` mark (plan 02)."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(1)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "soft_gate" in marks, (
        f"expected 'soft_gate' trace mark, got marks={sorted(marks)} — "
        "graded community-gate bonus (plan 02) not yet wired"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_trace_contains_multi_seed_and_cleanup_attractor_marks(driver, _store, monkeypatch):
    """Same trace MUST contain `multi_seed` and `cleanup_attractor` marks (plan 04)."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(2)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "multi_seed" in marks, (
        f"expected 'multi_seed' trace mark, got marks={sorted(marks)} — "
        "widened seed sources (plan 04) not yet wired"
    )
    assert "cleanup_attractor" in marks, (
        f"expected 'cleanup_attractor' trace mark, got marks={sorted(marks)} — "
        "thresholded cleanup() attractor (plan 04) not yet wired"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_trace_contains_conf_escalate_mark(driver, _store, monkeypatch):
    """Same trace MUST contain a `conf_escalate` mark (plan 03)."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(3)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "conf_escalate" in marks, (
        f"expected 'conf_escalate' trace mark, got marks={sorted(marks)} — "
        "confidence-gated escalation (plan 03) not yet wired"
    )


