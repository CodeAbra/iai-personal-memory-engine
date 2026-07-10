"""Bench-only kill-switch guards for the conf_escalate/multi_seed gates.

``IAI_MCP_CONF_ESCALATE_OFF`` and ``IAI_MCP_MULTI_SEED_OFF`` are additive
env-var toggles mirroring the existing ``IAI_MCP_EXACT_AUTHORITY_OFF``
precedent (``core/__init__.py:120,388``). Default (env unset) reproduces the
shipped mechanism-on behavior byte-identically; setting a toggle to
``"true"`` skips that mechanism's widen step and its trace_mark.

Dual-driver: every test parametrizes the storage driver since each dispatch
exercises a real ``MemoryStore`` insert + recall round-trip.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import MemoryRecord

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


@pytest.fixture(autouse=True)
def _clear_new_toggles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_CONF_ESCALATE_OFF", raising=False)
    monkeypatch.delenv("IAI_MCP_MULTI_SEED_OFF", raising=False)


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
    import iai_mcp.embed as _embed_mod

    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _store: _StubEmbedder(vec))


def _dispatch_traced_recall(store: MemoryStore, cue_vec: list[float]) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "low-confidence cue, embedder is stubbed",
        "session_id": "toggle-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })


def _dispatch_traced_recall_high_conf(store: MemoryStore, cue_vec: list[float]) -> dict:
    from iai_mcp import core as _core
    import iai_mcp.pipeline as _pm

    store._build_exact_index_sync()
    _pm._last_recall_latency_ms = 0.0
    return _core.dispatch(store, "memory_recall", {
        "cue": "high-confidence cue, exact stored embedding, embedder is stubbed",
        "session_id": "toggle-test-high-conf",
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
    """A small, low-confidence-inducing corpus (mirrors
    test_recall_trace_accuracy_marks.py's fixture exactly): no near-duplicate
    cluster, so top cosine stays below the high-confidence threshold and both
    conf_escalate/multi_seed gates fire."""
    target_id = uuid4()
    target = _make_rec(target_id, seed=1, surface="the associative target record", embedding=cue_vec)
    store.insert(target)
    for i in range(4):
        store.insert(_make_rec(
            uuid4(), seed=100 + i, surface=f"unrelated filler {i}",
            embedding=_orthogonal_vec(100 + i),
        ))
    flush_record_buffer(store)


def _seed_high_confidence_corpus(store: MemoryStore, cue_vec: list[float]) -> None:
    """A HIGH-confidence corpus: many near-duplicate records at the exact cue
    embedding, well above `_CONF_HIGH_COSINE_THRESHOLD` and comfortably above
    the hit-count-above-threshold floor `compute_spread_depth` requires to
    return 0 (gate skipped by CONFIDENCE, not by a toggle) — the negative
    control for Test 5."""
    for i in range(6):
        store.insert(_make_rec(
            uuid4(), seed=1, surface=f"exact duplicate of the cue {i}", embedding=cue_vec,
        ))
    flush_record_buffer(store)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_env_unset_both_mechanisms_fire(driver, _store, monkeypatch):
    """Test 1: env unset (default) + low-confidence cue -> both marks present."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(1)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "conf_escalate" in marks, f"expected 'conf_escalate' with toggles unset, got {sorted(marks)}"
    assert "multi_seed" in marks, f"expected 'multi_seed' with toggles unset, got {sorted(marks)}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_conf_escalate_off_disables_only_conf_escalate(driver, _store, monkeypatch):
    """Test 2: IAI_MCP_CONF_ESCALATE_OFF=true -> conf_escalate absent, multi_seed still fires."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    monkeypatch.setenv("IAI_MCP_CONF_ESCALATE_OFF", "true")
    store = _store(driver)
    cue_vec = _seeded_vec(2)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "conf_escalate" not in marks, f"expected 'conf_escalate' absent when toggled off, got {sorted(marks)}"
    assert "multi_seed" in marks, f"expected 'multi_seed' unaffected by conf_escalate toggle, got {sorted(marks)}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_multi_seed_off_disables_only_multi_seed(driver, _store, monkeypatch):
    """Test 3: IAI_MCP_MULTI_SEED_OFF=true -> multi_seed absent, conf_escalate still fires."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    monkeypatch.setenv("IAI_MCP_MULTI_SEED_OFF", "true")
    store = _store(driver)
    cue_vec = _seeded_vec(3)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "multi_seed" not in marks, f"expected 'multi_seed' absent when toggled off, got {sorted(marks)}"
    assert "conf_escalate" in marks, f"expected 'conf_escalate' unaffected by multi_seed toggle, got {sorted(marks)}"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_both_toggles_off_disables_both_and_returns_well_formed_hits(driver, _store, monkeypatch):
    """Test 4: both toggles set -> neither mark fires; response degrades non-crashing."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    monkeypatch.setenv("IAI_MCP_CONF_ESCALATE_OFF", "true")
    monkeypatch.setenv("IAI_MCP_MULTI_SEED_OFF", "true")
    store = _store(driver)
    cue_vec = _seeded_vec(4)
    _seed_small_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "conf_escalate" not in marks, f"expected neither mark, got {sorted(marks)}"
    assert "multi_seed" not in marks, f"expected neither mark, got {sorted(marks)}"
    assert isinstance(resp.get("hits"), list), "response 'hits' must remain a well-formed list"


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_high_confidence_cue_skips_both_marks_regardless_of_toggles(driver, _store, monkeypatch):
    """Test 5 (negative control): env unset + HIGH-confidence cue -> neither mark
    fires because the confidence gate itself returns 0, not because a toggle
    is set. Locks the 3-way semantics: fired / skipped-by-confidence /
    skipped-by-toggle are distinguishable."""
    monkeypatch.setenv("IAI_MCP_RECALL_TRACE", "1")
    store = _store(driver)
    cue_vec = _seeded_vec(5)
    _seed_high_confidence_corpus(store, cue_vec)
    _stub_embedder_for_store(monkeypatch, cue_vec)

    resp = _dispatch_traced_recall_high_conf(store, cue_vec)

    marks = {name for name, _ms in resp.get("_recall_trace_ms", [])}
    assert "conf_escalate" not in marks, (
        f"expected 'conf_escalate' absent on a high-confidence cue (toggles unset), got {sorted(marks)}"
    )
    assert "multi_seed" not in marks, (
        f"expected 'multi_seed' absent on a high-confidence cue (toggles unset), got {sorted(marks)}"
    )
