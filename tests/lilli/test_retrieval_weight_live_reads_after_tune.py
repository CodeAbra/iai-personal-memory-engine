"""Nightly `tune_retrieval_weight` must self-invalidate the recall-side
cache -- without an explicit `retrieval_weight_cache.invalidate(store)`
call, a live daemon keeps serving the pre-tune weight until restart. Every
existing cache test calls `invalidate()` itself, which would mask a missing
production call; this file never does.
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from iai_mcp import retrieval_weight_cache
from iai_mcp.events import write_event
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune import tune_retrieval_weight
from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import MAX_EVENTS, WINDOW_DAYS
from iai_mcp.lilli.profile.retrieval_tuning import (
    DEFAULT_W_COSINE,
    RETRIEVAL_MIN_SAMPLES,
)
from iai_mcp.store import MemoryStore


def _emit_used(store: MemoryStore, sid: str, hit_ids: list[str]) -> None:
    write_event(
        store, kind="retrieval_used",
        data={"hit_ids": hit_ids, "query": "q", "used": True, "budget_used": 1, "path": "baseline_recall"},
        severity="info", session_id=sid, buffered=False,
    )


def _emit_reinforced(store: MemoryStore, sid: str, reinforced_ids: list[str]) -> None:
    write_event(
        store, kind="retrieval_reinforced",
        data={
            "session_id": sid, "reinforced_ids": reinforced_ids,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info", session_id=sid, buffered=False,
    )


def _seed_paired_retrieval_window(store: MemoryStore, n: int, use_rate: float = 1.0) -> None:
    reinforced_count = round(use_rate * 10)
    for i in range(n):
        sid = f"bob-{i}"
        hit_ids = [str(uuid4()) for _ in range(10)]
        _emit_used(store, sid, hit_ids)
        time.sleep(0.001)
        _emit_reinforced(store, sid, hit_ids[:reinforced_count])
        time.sleep(0.001)


def test_tuned_weight_observed_via_recall_shaped_cache_read_without_manual_invalidate(
    tmp_path,
) -> None:
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)

    # Warm the cache with the pre-tune value first, simulating a live recall
    # that already read+cached W_COSINE before the sleep cycle runs. If
    # tune_retrieval_weight never invalidates, this stale cached dict is
    # exactly what a subsequent recall-shaped read would keep returning.
    pre = retrieval_weight_cache.load(store)
    assert pre["W_COSINE"] == DEFAULT_W_COSINE

    _seed_paired_retrieval_window(store, RETRIEVAL_MIN_SAMPLES, use_rate=1.0)
    cutoff = now - timedelta(days=WINDOW_DAYS)

    result = tune_retrieval_weight(store, cutoff, MAX_EVENTS)

    assert result["skipped"] is False
    assert result["persisted"] is True
    assert result["w_cosine"] != DEFAULT_W_COSINE

    # Production-shaped read: no manual invalidate() call here -- proving
    # tune_retrieval_weight's own persist path already armed the cache.
    post = retrieval_weight_cache.load(store)
    assert post["W_COSINE"] == result["w_cosine"], (
        "recall-shaped cache read served a stale weight after a successful "
        "nightly tune -- tune_retrieval_weight must self-invalidate the "
        "retrieval_weight_cache on every successful persist"
    )


def test_without_self_invalidation_the_cache_would_serve_the_stale_weight(
    tmp_path, monkeypatch,
) -> None:
    """Proves the first test has teeth: intercepting the production
    invalidate() call reproduces the stale read this guard exists to catch,
    confirming the assertion above is not vacuously true regardless of
    whether tune_retrieval_weight invalidates.
    """
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)

    pre = retrieval_weight_cache.load(store)
    assert pre["W_COSINE"] == DEFAULT_W_COSINE

    _seed_paired_retrieval_window(store, RETRIEVAL_MIN_SAMPLES, use_rate=1.0)
    cutoff = now - timedelta(days=WINDOW_DAYS)

    monkeypatch.setattr(retrieval_weight_cache, "invalidate", lambda _store: None)

    result = tune_retrieval_weight(store, cutoff, MAX_EVENTS)
    assert result["skipped"] is False
    assert result["persisted"] is True
    assert result["w_cosine"] != DEFAULT_W_COSINE

    stale = retrieval_weight_cache.load(store)
    assert stale["W_COSINE"] == DEFAULT_W_COSINE, (
        "with invalidate() neutralized, the cache must keep serving the "
        "pre-tune value -- proving tune_retrieval_weight's self-invalidation "
        "is what prevents this in production"
    )
