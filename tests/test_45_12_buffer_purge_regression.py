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
