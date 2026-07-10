from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest


@pytest.fixture(autouse=True)
def _opt_out_of_buffer_autoflush(monkeypatch):
    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")

import pytest


def _clear_edge_buffer(store) -> None:
    from iai_mcp import store as store_mod

    store_mod._edge_buffer.pop(id(store), None)
    store_mod._edge_last_flush_at.pop(id(store), None)


def test_boost_edges_insert_buffers_rows_not_lancedb(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        tbl = store.db.open_table(EDGES_TABLE)
        n_before = len(tbl.to_pandas())

        src_id = uuid4()
        dst_id = uuid4()

        store.boost_edges([(src_id, dst_id)], delta=0.5, edge_type="hebbian")

        buf = store_mod._edge_buffer.get(id(store), [])
        assert len(buf) >= 1, "expected _edge_buffer to accumulate rows after boost_edges insert"

        tbl = store.db.open_table(EDGES_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before, (
            f"boost_edges insert changed LanceDB row count before flush: {n_before} -> {n_after}"
        )


def test_flush_edge_buffer_writes_batch_and_clears(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        tbl = store.db.open_table(EDGES_TABLE)
        n_before = len(tbl.to_pandas())

        for i in range(3):
            row = {
                "src": str(uuid4()),
                "dst": str(uuid4()),
                "edge_type": "hebbian",
                "weight": float(i + 1) * 0.1,
                "updated_at": datetime.now(timezone.utc),
            }
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        assert len(store_mod._edge_buffer.get(id(store), [])) == 3

        flushed = flush_edge_buffer(store)
        assert flushed == 3

        assert not store_mod._edge_buffer.get(id(store))

        tbl = store.db.open_table(EDGES_TABLE)
        n_after = len(tbl.to_pandas())
        assert n_after == n_before + 3


def test_flush_edge_buffer_empty_returns_zero(tmp_path):
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        flushed = flush_edge_buffer(store)
        assert flushed == 0

        flushed2 = flush_edge_buffer(store)
        assert flushed2 == 0


def test_should_flush_edge_buffer_size_threshold(tmp_path, monkeypatch):
    from iai_mcp import store as store_mod
    from iai_mcp.store import MemoryStore, should_flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        monkeypatch.setenv("IAI_MCP_EDGE_BUFFER_MAX", "5")

        assert should_flush_edge_buffer(id(store)) is False

        for i in range(4):
            row = {
                "src": str(uuid4()),
                "dst": str(uuid4()),
                "edge_type": "hebbian",
                "weight": 0.1,
                "updated_at": datetime.now(timezone.utc),
            }
            store_mod._edge_buffer.setdefault(id(store), []).append(row)
        assert should_flush_edge_buffer(id(store)) is False

        row = {
            "src": str(uuid4()),
            "dst": str(uuid4()),
            "edge_type": "hebbian",
            "weight": 0.1,
            "updated_at": datetime.now(timezone.utc),
        }
        store_mod._edge_buffer.setdefault(id(store), []).append(row)
        assert should_flush_edge_buffer(id(store)) is True

        assert should_flush_edge_buffer(id(store), max_size=100) is False


def test_merge_insert_remains_synchronous():
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store" / "_store.py"
    text = store_py.read_text(encoding="utf-8")

    assert "tbl.merge_insert" in text, (
        "tbl.merge_insert not found in store.py — the synchronous update path was removed or renamed"
    )

    assert ".when_matched_update_all()" in text, (
        ".when_matched_update_all() not found in store.py — update path is broken"
    )

    boost_start = text.find("def boost_edges(")
    return_idx = text.find("return new_weights", boost_start)
    assert boost_start > 0, "def boost_edges( not found in store.py"
    assert return_idx > boost_start, "return new_weights not found after def boost_edges"

    body = text[boost_start:return_idx]
    assert "tbl.merge_insert" in body, (
        "tbl.merge_insert not found in boost_edges body — synchronous update path was removed"
    )
    assert ".when_matched_update_all()" in body, (
        ".when_matched_update_all() not found in boost_edges body"
    )


def test_boost_edges_returns_weights_for_buffered_rows(tmp_path):
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        uuid_a = uuid4()
        uuid_b = uuid4()
        uuid_c = uuid4()

        store.boost_edges([(uuid_a, uuid_b)], delta=0.5, edge_type="hebbian")
        flush_edge_buffer(store)

        result = store.boost_edges(
            [(uuid_a, uuid_b), (uuid_a, uuid_c)], delta=0.1, edge_type="hebbian"
        )

        assert len(result) == 2, (
            f"expected 2 keys in new_weights (one update, one insert); got {len(result)}: {result}"
        )

        str_a = str(uuid_a)
        str_b = str(uuid_b)
        str_c = str(uuid_c)
        key_ab = tuple(sorted([str_a, str_b]))
        key_ac = tuple(sorted([str_a, str_c]))
        assert key_ab in result, f"key_ab={key_ab} not in result; keys={list(result.keys())}"
        assert key_ac in result, f"key_ac={key_ac} not in result; keys={list(result.keys())}"


def test_add_contradicts_edge_flushes_immediately(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.store import EDGES_TABLE, MemoryStore

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        tbl = store.db.open_table(EDGES_TABLE)
        n_before = len(tbl.to_pandas())

        uuid_orig = uuid4()
        uuid_new = uuid4()
        store.add_contradicts_edge(uuid_orig, uuid_new)

        buf = store_mod._edge_buffer.get(id(store), [])
        assert len(buf) == 0, (
            f"contradicts edge must flush immediately; found {len(buf)} "
            f"row(s) still buffered"
        )

        tbl = store.db.open_table(EDGES_TABLE)
        assert len(tbl.to_pandas()) == n_before + 1, (
            "contradicts edge row did not land in LanceDB on write"
        )


def test_flush_edge_buffer_emits_telemetry_event(tmp_path):
    from iai_mcp import store as store_mod
    from iai_mcp.events import query_events
    from iai_mcp.store import MemoryStore, flush_edge_buffer

    with MemoryStore(path=tmp_path) as store:
        _clear_edge_buffer(store)

        for _ in range(2):
            row = {
                "src": str(uuid4()),
                "dst": str(uuid4()),
                "edge_type": "hebbian",
                "weight": 0.3,
                "updated_at": datetime.now(timezone.utc),
            }
            store_mod._edge_buffer.setdefault(id(store), []).append(row)

        flushed = flush_edge_buffer(store)
        assert flushed == 2

        events = query_events(store, kind="lance_buffer_flush")
        edges_events = [e for e in events if e["data"].get("table") == "edges"]
        assert len(edges_events) >= 1, (
            f"expected lance_buffer_flush event for edges table; found: {events}"
        )
        latest = edges_events[0]
        assert latest["data"]["count"] == 2, (
            f"expected count=2 in telemetry event; got: {latest['data']}"
        )


def test_edge_buffer_setdefault_used_at_two_call_sites():
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store" / "_store.py"
    text = store_py.read_text(encoding="utf-8")

    count = text.count("_edge_buffer.setdefault")
    assert count == 2, (
        f"expected exactly 2 '_edge_buffer.setdefault' in store.py; got {count}"
    )


def test_store_has_three_edge_flush_helpers():
    store_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "store" / "_buffers.py"
    text = store_py.read_text(encoding="utf-8")

    for fn_name in (
        "def flush_edge_buffer",
        "def should_flush_edge_buffer",
        "def should_flush_edge_buffer_by_time",
    ):
        assert fn_name in text, (
            f"expected '{fn_name}' to be defined in store.py"
        )


def test_daemon_periodic_tick_calls_flush_edge_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon" / "__init__.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_edge_buffer" in text, (
        "flush_edge_buffer not found in daemon.py"
    )
    assert "should_flush_edge_buffer_by_time" in text, (
        "periodic-tick wiring uses should_flush_edge_buffer_by_time helper — missing from daemon.py"
    )

    tick_idx = text.find("should_flush_edge_buffer_by_time")
    assert tick_idx > 0, "should_flush_edge_buffer_by_time must appear in daemon.py"


def test_daemon_wake_drain_calls_flush_edge_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon" / "__init__.py"
    text = daemon_py.read_text(encoding="utf-8")

    assert "flush_edge_buffer" in text, (
        "flush_edge_buffer not found in daemon.py — per-tick flush wiring missing"
    )
    assert "should_flush_edge_buffer_by_time" in text, (
        "should_flush_edge_buffer_by_time gate not found in daemon.py — per-tick time-threshold missing"
    )
    records_gate_idx = text.find("should_flush_record_buffer_by_time")
    edges_gate_idx = text.find("should_flush_edge_buffer_by_time")
    assert edges_gate_idx > records_gate_idx, (
        "should_flush_edge_buffer_by_time must appear after should_flush_record_buffer_by_time "
        "(records before edges ordering); "
        f"records_gate_idx={records_gate_idx}, edges_gate_idx={edges_gate_idx}"
    )
    edges_flush_idx = text.find("flush_edge_buffer", edges_gate_idx)
    assert edges_flush_idx > edges_gate_idx, (
        "flush_edge_buffer must appear after should_flush_edge_buffer_by_time; "
        f"edges_gate_idx={edges_gate_idx}, edges_flush_idx={edges_flush_idx}"
    )


def test_daemon_shutdown_calls_flush_edge_buffer():
    daemon_py = Path(__file__).resolve().parent.parent / "src" / "iai_mcp" / "daemon" / "__init__.py"
    text = daemon_py.read_text(encoding="utf-8")

    shutdown_idx = text.find("edges buffer flushed on shutdown")
    assert shutdown_idx > 0, (
        "'edges buffer flushed on shutdown' marker not found in daemon.py"
    )

    daemon_stopped_idx = text.find("daemon_stopped", shutdown_idx)
    assert daemon_stopped_idx > shutdown_idx, (
        "edges buffer flush must precede 'daemon_stopped' event write in daemon.py shutdown"
    )


def test_contradict_buffered_src_no_unknown_record_error(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.retrieve import contradict
    from iai_mcp.store import MemoryStore, _record_buffer
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "9999")

    with MemoryStore(path=tmp_path) as store:
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface="alice uses monotropic focus",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)

        buf = _record_buffer.get(id(store), [])
        assert any(r["id"] == str(rec.id) for r in buf), (
            "Pre-condition failed: record should be in _record_buffer (autoflush is disabled)"
        )

        receipt = contradict(
            store,
            rec.id,
            "alice uses hyper-focus on single topic",
            [0.2] * EMBED_DIM,
        )

        assert str(receipt.original_id) == str(rec.id), (
            f"expected original_id={rec.id}, got {receipt.original_id}"
        )
        assert receipt.edge_type == "contradicts", (
            f"expected edge_type='contradicts', got {receipt.edge_type!r}"
        )

        assert store.get(rec.id) is not None, (
            "SRC record must be durable in SQLite after contradict()"
        )
        assert store.get(receipt.new_record_id) is not None, (
            "DST (new_rec) must be durable in SQLite after contradict() returns"
        )

        from iai_mcp.store import EDGES_TABLE
        tbl = store.db.open_table(EDGES_TABLE)
        df = tbl.to_pandas()
        c_edges = df[
            (df["src"] == str(rec.id)) & (df["edge_type"] == "contradicts")
        ]
        assert len(c_edges) >= 1, (
            f"expected at least one contradicts edge from {rec.id} in edges table; "
            f"found {len(c_edges)} edge(s)"
        )


def test_contradict_chain_second_contradict_no_unknown_record_error(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.retrieve import contradict
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM, MemoryRecord

    monkeypatch.setenv("IAI_MCP_RECORD_BUFFER_MAX", "9999")

    with MemoryStore(path=tmp_path) as store:
        now = datetime.now(timezone.utc)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface="alice prefers written communication",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)

        receipt1 = contradict(
            store,
            rec.id,
            "alice prefers async written communication",
            [0.2] * EMBED_DIM,
        )
        new_id = receipt1.new_record_id

        receipt2 = contradict(
            store,
            new_id,
            "alice prefers async text with bullet points",
            [0.3] * EMBED_DIM,
        )

        assert str(receipt2.original_id) == str(new_id), (
            "second contradict's original_id must be the first contradict's new_record_id"
        )
        assert store.get(rec.id) is not None, "gen-0 record must be durable"
        assert store.get(new_id) is not None, "gen-1 (first new_rec) must be durable"
        assert store.get(receipt2.new_record_id) is not None, "gen-2 (second new_rec) must be durable"
