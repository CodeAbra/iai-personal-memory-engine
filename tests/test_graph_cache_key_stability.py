"""The warm-graph cache key must not flip under ambient write churn: every
flip re-keys the warm bundle and spawns a full-corpus background refresh, so a
key quantized too finely makes an idle daemon burn a core rebuilding a graph
nobody asked to change."""
from __future__ import annotations

import pytest


def test_quantize_keeps_small_corpus_behavior_and_is_monotone():
    from iai_mcp.runtime_graph_cache import _quantize_count

    # Historical fixed 10-wide steps survive at test-scale corpora.
    assert _quantize_count(0) == 0
    assert _quantize_count(19) == 1
    assert _quantize_count(199) == 19
    # Sentinel counts pass through untouched.
    assert _quantize_count(-1) == -1
    # Monotone across the fixed/geometric seam and beyond.
    prev = -1
    for n in [0, 5, 50, 150, 199, 200, 300, 1000, 10_000, 37_000, 200_000]:
        cur = _quantize_count(n)
        assert cur >= prev, f"bucket regressed at {n}"
        prev = cur


def test_key_flip_rate_is_bounded_at_scale():
    from iai_mcp.runtime_graph_cache import _quantize_count

    # The quantization guarantee is a flip RATE, not distance from an
    # arbitrary point: over +2000 records at a 37k corpus the bucket may
    # advance at most twice (~5%-wide buckets). A fixed step of 10 would
    # have flipped the key 200 times over the same drift — that flip storm
    # is what kept the background refresher running around the clock.
    n = 37_000
    flips = sum(
        1
        for i in range(2000)
        if _quantize_count(n + i) != _quantize_count(n + i + 1)
    )
    assert 1 <= flips <= 2
    # ~6% real growth is always past a bucket edge, so stale caches still die.
    assert _quantize_count(n) != _quantize_count(int(n * 1.06))


class _KeyProbeStore:
    """Minimal store surface for _cache_key: fixed counts, and a pending
    counter that FAILS the test if the key derivation ever consults it."""

    embed_dim = 384

    def __init__(self, active: int, edges: int) -> None:
        self._active = active
        self._edges = edges
        self._corpus_count_cache = None

    def active_records_count(self) -> int:
        return self._active

    def edges_count(self) -> int:
        return self._edges

    def pending_records_count(self) -> int:
        raise AssertionError(
            "the cache key must not depend on the pending count — pending rows "
            "are not graph nodes, and the pending->active flip invalidates "
            "explicitly at the data boundary"
        )


def test_key_ignores_pending_rows():
    from iai_mcp.runtime_graph_cache import _cache_key

    key = _cache_key(_KeyProbeStore(1000, 2000))
    assert key == _cache_key(_KeyProbeStore(1000, 2000))


def test_key_still_flips_on_large_drift():
    from iai_mcp.runtime_graph_cache import _cache_key, _quantize_count

    # Walk to a bucket interior so the stability half is deterministic.
    n = 37_000
    while _quantize_count(n + 1) != _quantize_count(n):
        n += 1
    base = _cache_key(_KeyProbeStore(n, 0))
    # Within one bucket: stable.
    assert base == _cache_key(_KeyProbeStore(n + 1, 0))
    # Past the bucket: the key must move so a genuinely stale cache dies.
    assert base != _cache_key(_KeyProbeStore(int(n * 1.08), 0))


def test_dirty_fuse_scales_with_corpus():
    from iai_mcp.runtime_graph_cache import (
        _FUSE_DIRTY_THRESHOLD,
        _fuse_dirty_threshold,
    )

    # Small (test-scale) corpora keep the historical constant exactly.
    assert _fuse_dirty_threshold(_KeyProbeStore(100, 0)) == _FUSE_DIRTY_THRESHOLD
    # At production scale the fuse widens (~2% of the corpus): 50 ambient
    # writes at 37k records must not declare the warm bundle stale.
    assert _fuse_dirty_threshold(_KeyProbeStore(37_000, 0)) == 740
    # A broken store degrades to the constant, never raises.
    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("no store surface")

    assert _fuse_dirty_threshold(_Broken()) == _FUSE_DIRTY_THRESHOLD


def test_refresher_cooldown_blocks_back_to_back_refreshes(tmp_path, monkeypatch):
    from iai_mcp import retrieve
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    store = MemoryStore(path=tmp_path / "store")
    try:
        monkeypatch.delenv("IAI_MCP_GRAPH_REFRESH_COOLDOWN_S", raising=False)
        t1 = retrieve._refresh_graph_bundle_async(store)
        assert t1 is not None, "first refresh must spawn"
        t1.join(timeout=120)
        assert retrieve._refresh_graph_bundle_async(store) is None, (
            "a refresh right after the previous one must be swallowed by the "
            "cooldown — this is the churn the rate limit exists to stop"
        )

        # Zero cooldown (the test escape hatch) restores back-to-back spawns.
        monkeypatch.setenv("IAI_MCP_GRAPH_REFRESH_COOLDOWN_S", "0")
        t2 = retrieve._refresh_graph_bundle_async(store)
        assert t2 is not None
        t2.join(timeout=120)
    finally:
        store.close()


def test_explicit_invalidation_is_never_throttled(tmp_path, monkeypatch):
    """The cooldown must not delay a demanded fresh view: invalidation unlinks
    the disk cache, which routes the next build INLINE (SWR refuses a missing
    file), bypassing the refresher and its rate limit entirely."""
    from iai_mcp import retrieve, runtime_graph_cache
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    store = MemoryStore(path=tmp_path / "store")
    store.root = tmp_path
    try:
        retrieve.build_runtime_graph(store)
        # Exhaust the cooldown budget.
        store._graph_refresh_last_mono = __import__("time").monotonic()
        runtime_graph_cache.invalidate(store)
        graph, assignment, _rc = retrieve.build_runtime_graph(store)
        assert graph is not None and assignment is not None
        assert runtime_graph_cache._cache_path(store).exists(), (
            "the inline rebuild after an explicit invalidation must have "
            "re-saved the cache — if it served a stale bundle instead, the "
            "cooldown throttled a demanded fresh view"
        )
    finally:
        store.close()
