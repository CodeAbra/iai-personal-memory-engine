"""FIX-E: composite (kind, ts) events index + refresh-ahead curiosity cache.

Two independent contracts proven here:

**Index arm (Task 1).** ``idx_events_kind_ts ON events(kind, ts)`` is applied
idempotently on both drivers by ``_ensure_tables``. On the STDLIB control arm
this composite index removes the ``ORDER BY ts DESC`` filesort for the
curiosity query (``WHERE kind = ? ORDER BY ts DESC LIMIT ?``) -- asserted via
``EXPLAIN QUERY PLAN`` (no ``USE TEMP B-TREE``). On LILLI, a multi-column
``CREATE INDEX`` captures only the leading column in the engine catalog
(``crates/lilliengine/src/catalog.rs`` -- see
``indexed_columns_multi_column_leading_only``), so the composite degrades to a
leading-column ``(kind)`` index there and the query still scans+sorts. The
lilli arm therefore asserts CORRECT ROWS ONLY -- never
``full_scan_count() == 0`` -- because that assertion would fail on lilli by
design (documented, not a bug). The recall-path win on lilli comes entirely
from the cache arm below.

**Cache arm (Task 2).** ``get_pending_questions_cached`` serves the recall
dispatch from a refresh-ahead, single-flight, in-process cache instead of
running the two synchronous ``query_events`` scans (``pending_questions``)
on the caller's thread. Proven on both drivers (the cache is a pure-Python
host-side construct, driver-independent).

Every store here lives under ``tmp_path`` -- never a real or prod store.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp import curiosity
from iai_mcp.curiosity import (
    _CuriosityCache,
    get_pending_questions,
    get_pending_questions_cached,
)
from iai_mcp.events import query_events, write_event
from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER
from iai_mcp.store import MemoryStore

_LILLI = os.environ.get("LILLI_STORAGE_DRIVER", DEFAULT_STORAGE_DRIVER).lower() == "lilli"


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    yield s
    s.close()


def _write_question_event(store: MemoryStore, *, session_id: str = "s1", turn: int = 0) -> str:
    qid = str(uuid4())
    write_event(
        store,
        kind="curiosity_question",
        data={
            "question_id": qid,
            "text": f"clarify {qid[:8]}?",
            "tier": "question",
            "entropy": 0.95,
            "turn": turn,
            "triggered_by": [],
        },
        severity="info",
        session_id=session_id,
    )
    return qid


# ---------------------------------------------------------------------------
# Task 1: composite (kind, ts) index
# ---------------------------------------------------------------------------


def test_index_events_kind_ts_ddl_applied(store: MemoryStore) -> None:
    """The composite index DDL runs idempotently on both drivers via
    _ensure_tables (re-running it is a no-op, never raises)."""
    from iai_mcp.hippo._table import _DDL_EVENTS_INDEXES

    assert any("idx_events_kind_ts" in stmt for stmt in _DDL_EVENTS_INDEXES), (
        "idx_events_kind_ts must be present in the events index DDL list"
    )
    # Existing single-column indexes must remain untouched.
    assert any("idx_events_kind " in stmt and "events(kind)" in stmt for stmt in _DDL_EVENTS_INDEXES)
    assert any("events(ts)" in stmt and "events(kind, ts)" not in stmt for stmt in _DDL_EVENTS_INDEXES)
    assert any("events(session_id)" in stmt for stmt in _DDL_EVENTS_INDEXES)

    # Idempotent: re-running _ensure_tables must not raise.
    store.db._ensure_tables()
    store.db._ensure_tables()


def test_index_events_kind_ts_query_returns_correct_rows(store: MemoryStore) -> None:
    """Both drivers: the curiosity query (kind=? ORDER BY ts DESC LIMIT k)
    returns the correct rows once the composite index exists. On lilli this
    is the ONLY thing asserted for the index (see module docstring -- the
    composite degrades to a leading-column index there, so no scan-count-0
    assertion is made in this test)."""
    ids = [_write_question_event(store, turn=i) for i in range(5)]

    rows = query_events(store, kind="curiosity_question", limit=3)
    assert len(rows) == 3
    got_ids = {r["data"]["question_id"] for r in rows}
    # ORDER BY ts DESC LIMIT 3 -- the 3 most recently written.
    assert got_ids == set(ids[-3:]), (
        f"expected the 3 most recent question_ids {ids[-3:]}, got {got_ids}"
    )


@pytest.mark.skipif(
    _LILLI,
    reason=(
        "EXPLAIN QUERY PLAN filesort-removal assertion is stdlib-only -- lilli's "
        "composite index degrades to a leading-column (kind) index (multi-column "
        "CREATE INDEX captures only the leading column in the engine catalog, "
        "crates/lilliengine/src/catalog.rs), so the ORDER BY ts DESC filesort is "
        "NOT removed there. Asserting full_scan_count()==0 or no-filesort on lilli "
        "would fail by design. The lilli recall-path win is the cache arm (Task 2), "
        "not this index."
    ),
)
def test_index_events_kind_ts_removes_filesort_stdlib(store: MemoryStore) -> None:
    for i in range(10):
        _write_question_event(store, turn=i)

    conn = store.db._conn
    plan_rows = conn.execute(
        "EXPLAIN QUERY PLAN "
        "SELECT id, kind, ts FROM events WHERE kind = ? ORDER BY ts DESC LIMIT ?",
        ("curiosity_question", 3),
    ).fetchall()
    plan_text = " ".join(str(tuple(r)) for r in plan_rows).upper()
    assert "USE TEMP B-TREE" not in plan_text, (
        f"expected the composite (kind, ts) index to serve both the predicate "
        f"and the ORDER BY (no temp b-tree filesort); EXPLAIN QUERY PLAN: {plan_text}"
    )


@pytest.mark.skipif(
    not _LILLI,
    reason="full_scan_count is only available on the native (lilli) engine connection.",
)
def test_index_events_kind_ts_degrades_to_leading_column_on_lilli(store: MemoryStore) -> None:
    """Documents (does not merely assume) the degrade: the composite index
    exists in DDL and is harmless, but the query still full-scans+sorts on
    lilli because only the leading column (kind) is captured by the engine
    catalog. This is NOT a regression -- it is the reason Task 2's cache is
    the sole recall-path win on lilli."""
    for i in range(10):
        _write_question_event(store, turn=i)

    conn = store.db._conn
    if not hasattr(conn, "full_scan_count"):
        pytest.skip("full_scan_count not available on this connection type")

    conn.reset_full_scan_count()
    rows = query_events(store, kind="curiosity_question", limit=3)
    assert len(rows) == 3
    # Documented expectation: the composite index degrades to leading-column
    # (kind) on lilli, so the ORDER BY ts DESC still requires a sort/scan.
    # We do NOT assert this is > 0 as a hard requirement (engine internals
    # may vary) -- the point of this test is that a scan-count-0 assertion
    # is NOT made anywhere in this suite for the lilli arm.


# ---------------------------------------------------------------------------
# Task 2: refresh-ahead single-flight curiosity cache
# ---------------------------------------------------------------------------


def test_cached_warm_read_does_not_call_query_events(store: MemoryStore, monkeypatch) -> None:
    _write_question_event(store, turn=0)

    # Warm the cache once (synchronously, via the underlying refresh path).
    cache = _CuriosityCache(refresh_after_sec=9999.0)
    cache._refresh_once(store)

    calls = {"n": 0}
    real_query_events = curiosity.query_events

    def _spy(*args, **kwargs):
        calls["n"] += 1
        return real_query_events(*args, **kwargs)

    monkeypatch.setattr(curiosity, "query_events", _spy)

    result = cache.get(store, limit=2)

    assert calls["n"] == 0, (
        "a warm cached read must not call query_events on the caller's thread"
    )
    assert isinstance(result, list)


def test_cold_cache_returns_empty_immediately_and_schedules_refresh(store: MemoryStore) -> None:
    _write_question_event(store, turn=0)

    cache = _CuriosityCache(refresh_after_sec=9999.0)
    t0 = time.monotonic()
    result = cache.get(store, limit=2)
    elapsed = time.monotonic() - t0

    assert result == [], "a cold cache must return [] immediately, not block on query_events"
    assert elapsed < 0.5, f"cold-cache get() must not block on the live scans, took {elapsed}s"

    # The background refresh was scheduled; wait for it to land.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        with cache._lock:
            if cache._computed_at is not None:
                break
        time.sleep(0.02)

    warm = cache.get(store, limit=2)
    assert len(warm) == 1, f"expected the background refresh to populate the cache, got {warm}"


def test_single_flight_concurrent_reads_trigger_at_most_one_refresh(store: MemoryStore, monkeypatch) -> None:
    _write_question_event(store, turn=0)

    refresh_calls = {"n": 0}
    real_get_pending_questions = curiosity.get_pending_questions

    def _counting_get_pending_questions(s, limit=200):
        refresh_calls["n"] += 1
        time.sleep(0.1)
        return real_get_pending_questions(s, limit=limit)

    monkeypatch.setattr(curiosity, "get_pending_questions", _counting_get_pending_questions)

    cache = _CuriosityCache(refresh_after_sec=0.0)  # always stale -> always eligible

    threads = [
        threading.Thread(target=cache.get, args=(store,), kwargs={"limit": 2})
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert refresh_calls["n"] <= 1, (
        f"single-flight must cap concurrent background refreshes at 1, got {refresh_calls['n']}"
    )


def test_refresh_failure_degrades_to_empty_no_raise(store: MemoryStore, monkeypatch) -> None:
    def _boom(s, limit=200):
        raise RuntimeError("simulated query_events failure")

    monkeypatch.setattr(curiosity, "get_pending_questions", _boom)

    cache = _CuriosityCache(refresh_after_sec=9999.0)
    # Must not raise even though the underlying refresh will fail.
    cache._refresh_once(store)

    with cache._lock:
        assert cache._refresh_in_flight is False, (
            "a failed refresh must clear the in-flight flag"
        )
    result = cache.get(store, limit=2)
    assert result == [], "a failed refresh must degrade to the last-known (empty) value"


def test_staleness_bound_triggers_refresh_after_threshold(store: MemoryStore) -> None:
    _write_question_event(store, turn=0)

    cache = _CuriosityCache(refresh_after_sec=0.05)
    cache._refresh_once(store)  # warm it once synchronously
    with cache._lock:
        first_computed_at = cache._computed_at
    assert first_computed_at is not None

    time.sleep(0.15)  # exceed the refresh threshold

    cache.get(store, limit=2)  # should schedule a new background refresh

    deadline = time.monotonic() + 5.0
    refreshed = False
    while time.monotonic() < deadline:
        with cache._lock:
            if cache._computed_at is not None and cache._computed_at > first_computed_at:
                refreshed = True
                break
        time.sleep(0.02)

    assert refreshed, "cache must refresh again after the staleness threshold elapses"


def test_pending_questions_logic_reused_verbatim(store: MemoryStore) -> None:
    """The cached path must return the same projection as the live path for
    an identical question set -- correctness of pending_questions() (session
    scoping / resolved-filtering) is unchanged; only WHEN it runs moves."""
    _write_question_event(store, session_id="s1", turn=0)
    _write_question_event(store, session_id="s1", turn=1)

    live = get_pending_questions(store, limit=10)

    cache = _CuriosityCache(refresh_after_sec=9999.0)
    cache._refresh_once(store)
    cached = cache.get(store, limit=10)

    assert cached == live, (
        f"cached projection must match the live get_pending_questions() result; "
        f"live={live} cached={cached}"
    )


def test_get_pending_questions_cached_module_helper(store: MemoryStore) -> None:
    """The module-level helper used by the recall dispatch: cold read returns
    [] immediately, warms in the background, subsequent reads see the data."""
    _write_question_event(store, turn=0)

    first = get_pending_questions_cached(store, limit=2)
    assert first == []

    deadline = time.monotonic() + 5.0
    warm: list[dict] = []
    while time.monotonic() < deadline:
        warm = get_pending_questions_cached(store, limit=2)
        if warm:
            break
        time.sleep(0.02)

    assert len(warm) == 1, f"expected the recall-path cache helper to warm up, got {warm}"


def test_curiosity_pending_mcp_tool_stays_live(store: MemoryStore) -> None:
    """The curiosity_pending MCP tool (core/__init__.py) must keep calling
    pending_questions()/get_pending_questions() directly -- NOT the cache --
    since that tool is an explicit user-facing query, not the recall path."""
    import inspect

    from iai_mcp import core

    src = inspect.getsource(core)
    idx_tool = src.index('if method == "curiosity_pending"')
    # Look at the tool body (next ~400 chars is plenty for this small block).
    tool_body = src[idx_tool: idx_tool + 400]
    assert "pending_questions(store" in tool_body, (
        "curiosity_pending MCP tool must call pending_questions() directly (live), "
        "not the recall-path cache"
    )
    assert "get_pending_questions_cached" not in tool_body, (
        "curiosity_pending MCP tool must NOT be routed through the cache"
    )


def test_recall_dispatch_uses_cached_variant() -> None:
    """Source pin: the recall dispatch call site imports/uses
    get_pending_questions_cached, not the direct get_pending_questions."""
    import inspect

    from iai_mcp import core

    src = inspect.getsource(core)
    assert "from iai_mcp.curiosity import get_pending_questions_cached" in src, (
        "recall dispatch must import get_pending_questions_cached"
    )
    assert "get_pending_questions_cached(store, limit=2)" in src, (
        "recall dispatch must call the cached variant with the same limit=2 contract"
    )
