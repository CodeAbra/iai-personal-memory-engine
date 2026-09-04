from __future__ import annotations

import gc
from datetime import datetime, timezone



def test_close_purges_all_six_buffer_dicts(tmp_path):
    from iai_mcp import events, store as store_mod
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    s = MemoryStore(path=tmp_path)
    try:
        store_id = id(s)

        write_event(s, kind="reg_test", data={"k": "v"}, buffered=True)
        assert store_id in events._event_buffer

        events._last_flush_at[store_id] = datetime.now(timezone.utc)
        store_mod._record_buffer[store_id] = []
        store_mod._record_last_flush_at[store_id] = datetime.now(timezone.utc)
        store_mod._edge_buffer[store_id] = []
        store_mod._edge_last_flush_at[store_id] = datetime.now(timezone.utc)

        for dct_name, dct in (
            ("events._event_buffer", events._event_buffer),
            ("events._last_flush_at", events._last_flush_at),
            ("store._record_buffer", store_mod._record_buffer),
            ("store._record_last_flush_at", store_mod._record_last_flush_at),
            ("store._edge_buffer", store_mod._edge_buffer),
            ("store._edge_last_flush_at", store_mod._edge_last_flush_at),
        ):
            assert store_id in dct, f"pre-close: {dct_name} missing store_id"

        s.close()

        assert store_id not in events._event_buffer
        assert store_id not in events._last_flush_at
        assert store_id not in store_mod._record_buffer
        assert store_id not in store_mod._record_last_flush_at
        assert store_id not in store_mod._edge_buffer
        assert store_id not in store_mod._edge_last_flush_at
    finally:
        s.close()


def test_reset_purges_events_across_abandoned_id_reuse(tmp_path):
    """Simulates a store abandoned WITHOUT close() (only close() purges the
    events buffer today), then a later store deterministically reusing its
    freed id() via the exact call MemoryStore.__init__ makes at construction.
    """
    from iai_mcp import events
    from iai_mcp.events import flush_event_buffer, write_event
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import reset_store_buffers

    store_a = MemoryStore(path=tmp_path)
    write_event(store_a, kind="ghost_evt", data={"author": "store_a"}, buffered=True)

    gid = id(store_a)
    assert gid in events._event_buffer

    # Populate _last_flush_at via the real flush path (only flush_event_buffer
    # sets it) so the post-purge assert on it is load-bearing, not vacuous.
    flush_event_buffer(store_a)
    assert gid in events._last_flush_at

    write_event(store_a, kind="ghost_evt_2", data={"author": "store_a"}, buffered=True)
    assert gid in events._event_buffer

    del store_a

    reset_store_buffers(gid)

    assert gid not in events._event_buffer
    assert gid not in events._last_flush_at


def test_reset_purges_curiosity_cache_across_abandoned_id_reuse(tmp_path):
    """Same abandoned-without-close id()-reuse path as
    test_reset_purges_events_across_abandoned_id_reuse, for curiosity's
    read-through cache: reset_store_buffers(gid) must also drop
    curiosity._caches[gid] so a reused id() cannot be served a dead store's
    stale cached answer.
    """
    from iai_mcp import curiosity
    from iai_mcp.store import MemoryStore
    from iai_mcp.store._buffers import reset_store_buffers

    store_a = MemoryStore(path=tmp_path)
    gid = id(store_a)
    # Seed via the real population path (mirrors the liveness-guard test
    # below) rather than hand-building a dict entry, so a future change to
    # _cache_for's keying is caught here too.
    curiosity._cache_for(store_a)
    assert gid in curiosity._caches

    del store_a

    reset_store_buffers(gid)

    assert gid not in curiosity._caches


def test_curiosity_cache_stable_identity_across_live_store_calls(tmp_path):
    """A live store's warm curiosity cache is never regressed: repeated
    lookups for the SAME live store return the SAME cache object, and its
    entry stays present in curiosity._caches for the store's whole life.
    """
    from iai_mcp import curiosity
    from iai_mcp.curiosity import get_pending_questions_cached
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    try:
        get_pending_questions_cached(store, limit=2)
        first_cache = curiosity._cache_for(store)

        get_pending_questions_cached(store, limit=2)
        second_cache = curiosity._cache_for(store)

        assert first_cache is second_cache
        assert id(store) in curiosity._caches
    finally:
        store.close()


def test_no_ghost_ciphertext_across_id_reuse(tmp_path, tmp_path_factory):
    from iai_mcp import events, store as store_mod
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    path_b = tmp_path_factory.mktemp("store_b")

    store_a = MemoryStore(path=tmp_path)
    write_event(store_a, kind="sentinel_A", data={"author": "store_a"}, buffered=True)
    store_a.close()

    del store_a
    gc.collect()

    store_b = MemoryStore(path=path_b)
    try:
        store_b_id = id(store_b)

        ghost_event_rows = [
            r for r in events._event_buffer.get(store_b_id, [])
            if r.get("kind") == "sentinel_A"
        ]
        assert not ghost_event_rows, (
            f"ghost ciphertext leaked into events._event_buffer: {ghost_event_rows}"
        )

        for dct_name, dct in (
            ("store._record_buffer", store_mod._record_buffer),
            ("store._edge_buffer", store_mod._edge_buffer),
        ):
            ghost_rows = dct.get(store_b_id, [])
            assert not ghost_rows, (
                f"ghost rows in {dct_name} at id(store_b)={store_b_id}: {ghost_rows}"
            )

        from iai_mcp.events import query_events
        rows = query_events(store_b, kind="sentinel_A")
        assert not rows, (
            f"sentinel_A row leaked onto store_b's disk; expected store_a's path only: {rows}"
        )
    finally:
        store_b.close()


def test_purge_callback_failure_is_telemetered():
    """A raising purge callback must not be silent: purge_store logs it AND
    bumps a countable failure counter, so a persistently-poisoned family is
    observable without a store.
    """
    from iai_mcp.store._purge_registry import (
        _PURGE_CALLBACKS,
        _REGISTRY_LOCK,
        purge_failure_counts,
        purge_store,
        register_store_purge,
    )

    def _boom(store_id: int) -> None:
        raise RuntimeError("test_purge_callback_failure_is_telemetered")

    register_store_purge(_boom)
    try:
        before = purge_failure_counts().get("_boom", 0)
        purge_store(id(object()))
        after = purge_failure_counts().get("_boom", 0)
        assert after == before + 1
    finally:
        with _REGISTRY_LOCK:
            try:
                _PURGE_CALLBACKS.remove(_boom)
            except ValueError:
                pass
