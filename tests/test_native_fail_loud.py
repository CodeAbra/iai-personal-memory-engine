from __future__ import annotations

import pytest

from iai_mcp.store import MemoryStore

def _make_store(tmp_path) -> MemoryStore:
    return MemoryStore(path=tmp_path)

def _seed_one_record(store: MemoryStore, text: str = "seed record") -> None:
    from datetime import datetime, timezone
    from uuid import uuid4

    from iai_mcp.types import EMBED_DIM, MemoryRecord

    now = datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
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

def test_topology_native_failure_emits_and_raises(tmp_path, monkeypatch):
    from iai_mcp import core, retrieve
    from iai_mcp.events import flush_event_buffer, query_events

    store = _make_store(tmp_path)
    _seed_one_record(store)

    def _fail(*args, **kwargs):
        raise RuntimeError("simulated native build failure")

    monkeypatch.setattr(retrieve, "build_runtime_graph", _fail)
    # The topology snapshot cache is module-global with a 900s TTL; a warm
    # entry from any earlier test would short-circuit dispatch before the
    # patched build ever runs. Same for a held single-flight lock — it makes
    # dispatch answer with a placeholder instead of raising.
    import threading

    monkeypatch.setattr(core, "_topology_cache", None)
    monkeypatch.setattr(core, "_topology_cache_at", 0.0)
    monkeypatch.setattr(core, "_topology_cache_key", None)
    monkeypatch.setattr(core, "_topology_inflight", threading.Lock())

    with pytest.raises(RuntimeError, match="simulated native build failure"):
        core.dispatch(store, "topology", {})

    flush_event_buffer(store)

    rows = query_events(store, kind="topology_native_failed", limit=10)
    assert rows, (
        "no topology_native_failed event written; expected emit before raise"
    )
    ev = rows[0]
    data = ev.get("data") or {}
    assert "error_type" in data, f"event data missing error_type: {data}"
    assert "error" in data, f"event data missing error field: {data}"

def test_topology_empty_graph_returns_stub_without_raising(tmp_path):
    from iai_mcp import core
    from iai_mcp.events import flush_event_buffer, query_events

    store = _make_store(tmp_path)

    result = core.dispatch(store, "topology", {})

    assert result.get("regime") == "insufficient_data", (
        f"expected regime=insufficient_data on empty store, got: {result}"
    )
    assert result.get("sigma") is None, (
        f"expected sigma=None on empty store, got: {result.get('sigma')}"
    )
    assert result.get("N") == 0, (
        f"expected N=0 on empty store, got: {result.get('N')}"
    )

    flush_event_buffer(store)
    rows = query_events(store, kind="topology_native_failed", limit=5)
    assert not rows, (
        "topology_native_failed event must NOT be emitted for the empty-graph "
        f"legitimate path; got {rows}"
    )

def test_recall_cue_encode_failure_emits_store_event_and_raises(
    tmp_path, monkeypatch
):
    from iai_mcp import core
    from iai_mcp.embed import Embedder
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.exceptions import NativeError

    store = _make_store(tmp_path)
    _seed_one_record(store, "test content for recall")

    def _broken_embed(self, text: str) -> list[float]:
        raise RuntimeError("boom native encode")

    monkeypatch.setattr(Embedder, "_encode_batch", lambda self, texts, **kw: (_ for _ in ()).throw(RuntimeError("boom native encode")))
    monkeypatch.setattr(Embedder, "embed", _broken_embed)

    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"current_state": "WAKE"},
    )
    monkeypatch.setattr(
        "iai_mcp.daemon_state.save_state",
        lambda s: None,
    )

    with pytest.raises((NativeError, RuntimeError)):
        core.dispatch(store, "memory_recall", {
            "cue": "test content",
            "session_id": "test-session",
        })

    flush_event_buffer(store)
    rows = query_events(store, kind="embed_native_failure", limit=10)
    assert rows, (
        "no embed_native_failure store event written on recall cue-encode failure; "
        "expected Layer-2 store-backed emit before raise"
    )
    ev = rows[0]
    data = ev.get("data") or {}
    assert data.get("op_type") == "recall_cue", (
        f"expected op_type='recall_cue', got: {data}"
    )

def test_capture_encode_failure_emits_store_event_and_raises(
    tmp_path, monkeypatch
):
    from iai_mcp.capture import capture_turn
    from iai_mcp.embed import Embedder
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.exceptions import NativeError

    store = _make_store(tmp_path)

    def _broken_embed(self, text: str) -> list[float]:
        raise RuntimeError("boom capture encode")

    monkeypatch.setattr(Embedder, "embed", _broken_embed)

    with pytest.raises((NativeError, RuntimeError)):
        capture_turn(
            store,
            cue="test cue",
            text="this is a long enough capture text to pass the minimum length guard",
            tier="episodic",
            session_id="test-session",
        )

    flush_event_buffer(store)
    rows = query_events(store, kind="embed_native_failure", limit=10)
    assert rows, (
        "no embed_native_failure store event written on capture encode failure; "
        "expected Layer-2 store-backed emit before raise"
    )
    ev = rows[0]
    data = ev.get("data") or {}
    assert data.get("op_type") == "capture", (
        f"expected op_type='capture', got: {data}"
    )

def test_topology_cache_never_crosses_stores(tmp_path, monkeypatch):
    """A snapshot cached for one store must never be served for another —
    the cache is process-global and only the store key keeps it honest."""
    import threading
    import time

    from iai_mcp import core, retrieve

    store = _make_store(tmp_path)
    _seed_one_record(store)

    def _fail(*args, **kwargs):
        raise RuntimeError("cross-store build attempted")

    monkeypatch.setattr(retrieve, "build_runtime_graph", _fail)
    monkeypatch.setattr(core, "_topology_inflight", threading.Lock())
    monkeypatch.setattr(
        core,
        "_topology_cache",
        {"N": 1, "C": 0.0, "L": 0.0, "sigma": None,
         "community_count": 1, "rich_club_ratio": 0.0, "regime": "ok"},
    )
    monkeypatch.setattr(core, "_topology_cache_at", time.monotonic())
    monkeypatch.setattr(core, "_topology_cache_key", "some-other-store")

    with pytest.raises(RuntimeError, match="cross-store build attempted"):
        core.dispatch(store, "topology", {})

def test_recall_raises_on_embedder_config_error(tmp_path, monkeypatch):
    """An embedder-selection refusal must surface from recall — the soft
    availability fallback would otherwise serve a zero-vector cue."""
    import iai_mcp.embed as embed_mod
    from iai_mcp import core
    from iai_mcp.embed import EmbedderConfigError

    store = _make_store(tmp_path)
    _seed_one_record(store)

    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state", lambda: {"current_state": "WAKE"}
    )
    monkeypatch.setattr("iai_mcp.daemon_state.save_state", lambda s: None)

    from iai_mcp.embed import EmbedIdentityMismatch

    for refusal_type in (EmbedderConfigError, EmbedIdentityMismatch):
        def _refuse(store, **kwargs):
            raise refusal_type("vector space mismatch")

        monkeypatch.setattr(embed_mod, "embedder_for_store", _refuse)
        with pytest.raises(refusal_type):
            core.dispatch(store, "memory_recall", {
                "cue": "any cue", "session_id": "test-session",
            })
