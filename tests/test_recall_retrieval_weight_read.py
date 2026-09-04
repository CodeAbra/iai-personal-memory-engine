"""Recall-read wiring: the tuned W_COSINE is read into an explicit
effective_w_cosine local, cached per store instance, externally
invalidatable, and applied in the non-Rust scoring branch.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp import core, retrieval_weight_cache
from iai_mcp.lilli.profile import retrieval_tuning
from iai_mcp.lilli.profile.retrieval_tuning import (
    DEFAULT_W_COSINE,
    PROD_W_COSINE_MAX,
    PROD_W_COSINE_MIN,
    save_retrieval_weights_state,
)
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store

_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]

_CUE_TEXT = "retrieval weight read probe"


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


class _MappedEmbedder:
    """Deterministic embedder: exact vector for the registered surface, an
    orthogonal fallback for anything else -- the store re-embeds on insert,
    so a crafted cosine must come through the embedder, not the record
    field (test_recall_rank_fusion.py precedent)."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = dict(mapping)

    def embed(self, text: str) -> list[float]:
        vec = self._mapping.get(text)
        if vec is not None:
            return list(vec)
        fallback = [0.0] * EMBED_DIM
        fallback[EMBED_DIM - 1] = 1.0
        return fallback


def _rec(text: str, emb: list[float]) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(), tier="episodic", literal_surface=text, aaak_index="",
        embedding=emb, community_id=None, centrality=0.0, detail_level=2,
        pinned=False, stability=0.0, difficulty=0.0, last_reviewed=None,
        never_decay=False, never_merge=False, provenance=[],
        created_at=now, updated_at=now, tags=[], language="en",
    )


def _setup_store(tmp_path, monkeypatch: pytest.MonkeyPatch, driver: str) -> MemoryStore:
    _set_driver(monkeypatch, driver)
    vec = _unit(0)
    stub_embedder_for_store(monkeypatch, _MappedEmbedder({_CUE_TEXT: vec}))
    store = MemoryStore(path=tmp_path)
    store.insert(_rec(_CUE_TEXT, vec))
    flush_record_buffer(store)
    return store


def _dispatch_non_rust(monkeypatch: pytest.MonkeyPatch, store: MemoryStore) -> dict:
    # Forces the reference scoring path even through the forced-Rust
    # production dispatch (core/__init__.py:1244, use_rust_scorer=True) --
    # this exercises the ACTUAL retrieval_weight_cache.load() call site,
    # not a bypass around it. The exact-cosine authority merge is a
    # separate, unconditional-once-warm mechanism that can override a
    # served score independent of which scorer branch ran (it runs AFTER
    # the pipeline returns) -- disabled here so a second dispatch against
    # the same store cannot silently replace the reason string this test
    # asserts on.
    monkeypatch.setenv("IAI_MCP_RECALL_RUST_SCORER_OFF", "1")
    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
    resp = core.dispatch(store, "memory_recall", {
        "cue": _CUE_TEXT, "session_id": "retrieval-weight-read-test",
        "budget_tokens": 2000, "cue_embedding": _unit(0),
    })
    assert "error" not in resp, f"recall dispatch errored: {resp.get('error')}"
    assert resp["hits"], "expected at least one hit for the cosine=1.0 probe record"
    return resp


def _cos_term(weight: float) -> str:
    # The non-Rust reason string is "cos {cos:.3f}*{effective_w_cosine:g} ...";
    # the probe record's cosine to its own cue vector is exactly 1.0.
    return f"cos 1.000*{weight:g}"


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_tuned_weight_is_read_into_non_rust_reason(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)
    tuned = PROD_W_COSINE_MIN
    save_retrieval_weights_state(store, {"W_COSINE": tuned})

    resp = _dispatch_non_rust(monkeypatch, store)
    reason = resp["hits"][0]["reason"]
    assert _cos_term(tuned) in reason, (
        f"tuned W_COSINE={tuned} not reflected in the non-Rust reason: {reason!r}"
    )
    assert _cos_term(DEFAULT_W_COSINE) not in reason


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_weight_loaded_at_most_once_per_store_generation(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)
    save_retrieval_weights_state(store, {"W_COSINE": PROD_W_COSINE_MIN})

    calls = {"n": 0}
    _orig = retrieval_tuning.load_retrieval_weights_state

    def _spy(s):
        calls["n"] += 1
        return _orig(s)

    monkeypatch.setattr(retrieval_tuning, "load_retrieval_weights_state", _spy)

    for _ in range(5):
        _dispatch_non_rust(monkeypatch, store)

    assert calls["n"] <= 1, (
        f"tuned weight decrypted {calls['n']} times across 5 recalls against the "
        f"same store generation; the cache must serve every call after the first"
    )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_invalidate_forces_reread_after_persist(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)
    w1 = PROD_W_COSINE_MIN
    w2 = PROD_W_COSINE_MAX
    save_retrieval_weights_state(store, {"W_COSINE": w1})

    resp1 = _dispatch_non_rust(monkeypatch, store)
    assert _cos_term(w1) in resp1["hits"][0]["reason"]

    save_retrieval_weights_state(store, {"W_COSINE": w2})
    resp_stale = _dispatch_non_rust(monkeypatch, store)
    assert _cos_term(w1) in resp_stale["hits"][0]["reason"], (
        "without invalidate() the cache must keep serving the prior weight"
    )

    retrieval_weight_cache.invalidate(store)
    resp2 = _dispatch_non_rust(monkeypatch, store)
    assert _cos_term(w2) in resp2["hits"][0]["reason"], (
        "invalidate(store) must force a re-read of the newly persisted weight"
    )


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_absent_weight_defaults_without_raise(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)
    # No save_retrieval_weights_state call: the store carries no weight blob.
    resp = _dispatch_non_rust(monkeypatch, store)
    assert _cos_term(DEFAULT_W_COSINE) in resp["hits"][0]["reason"]


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_corrupted_weight_blob_defaults_without_raise(tmp_path, monkeypatch, driver) -> None:
    store = _setup_store(tmp_path, monkeypatch, driver)

    def _raise(_store):
        raise RuntimeError("simulated undecryptable weight blob")

    monkeypatch.setattr(retrieval_tuning, "load_retrieval_weights_state", _raise)

    resp = _dispatch_non_rust(monkeypatch, store)
    assert _cos_term(DEFAULT_W_COSINE) in resp["hits"][0]["reason"]
