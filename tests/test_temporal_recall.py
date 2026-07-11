"""Temporal recall surface: time-travel by as_of (records) and changed_since (events)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _make_record(
    *,
    literal_surface: str,
    embedding: list[float],
    created_at: datetime,
    tier: str = "episodic",
    detail_level: int = 1,
    tags: list[str] | None = None,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=created_at,
        updated_at=created_at,
        tags=tags or [],
        language=language,
    )


def _zero_vec() -> list[float]:
    return [0.0] * EMBED_DIM


def test_memory_temporal_recall_no_args(hermetic_store):
    """No temporal args: dict shape returned with empty hits and empty changed_since_events."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    result = core_dispatch(store, "memory_temporal_recall", {})

    assert isinstance(result, dict)
    assert "hits" in result
    assert "changed_since_events" in result
    assert result["hits"] == []
    assert result["changed_since_events"] == []
    # _scope is absent or None when no temporal args were passed
    scope = result.get("_scope")
    assert scope is None


def test_as_of_committed_by_time(hermetic_store):
    """as_of(yesterday) excludes today's record; as_of(today + 1h) includes it."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    today = datetime.now(timezone.utc)
    yesterday = today - timedelta(days=1)
    record_a = _make_record(
        literal_surface="record A captured today",
        embedding=_zero_vec(),
        created_at=today,
    )
    store.insert(record_a)

    out_before = core_dispatch(
        store,
        "memory_temporal_recall",
        {"as_of": yesterday.isoformat(), "limit": 10},
    )
    ids_before = {hit.get("id") for hit in out_before.get("hits", [])}
    assert str(record_a.id) not in ids_before
    assert out_before.get("_scope") == "committed_by_time_t"

    future = today + timedelta(hours=1)
    out_after = core_dispatch(
        store,
        "memory_temporal_recall",
        {"as_of": future.isoformat(), "limit": 10},
    )
    ids_after = {hit.get("id") for hit in out_after.get("hits", [])}
    assert str(record_a.id) in ids_after
    assert out_after.get("_scope") == "committed_by_time_t"


def test_changed_since_returns_insert_event(hermetic_store):
    """changed_since(yesterday) returns a record_inserted event written today."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    write_event(
        store,
        kind="record_inserted",
        data={"record_id": str(uuid4())},
        severity="info",
    )

    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    out = core_dispatch(
        store,
        "memory_temporal_recall",
        {"changed_since": yesterday, "limit": 100},
    )
    events = out.get("changed_since_events", [])
    assert len(events) >= 1
    kinds = {ev.get("kind") for ev in events}
    assert "record_inserted" in kinds


def test_flush_at_entry_visible(hermetic_store):
    """Buffered (un-flushed) events become visible to the temporal query."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    # The autouse autoflush fixture in conftest hooks insert(); for write_event we
    # use buffered=True directly and rely on the dispatch branch flushing at entry.
    os.environ["IAI_MCP_TEST_NO_AUTOFLUSH"] = "1"
    try:
        store = MemoryStore(path=hermetic_store)
        write_event(
            store,
            kind="record_inserted",
            data={"buffered": True},
            severity="info",
            buffered=True,
        )

        past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        out = core_dispatch(
            store,
            "memory_temporal_recall",
            {"changed_since": past, "limit": 100},
        )
    finally:
        os.environ.pop("IAI_MCP_TEST_NO_AUTOFLUSH", None)

    events = out.get("changed_since_events", [])
    assert len(events) >= 1
    payloads = [ev.get("data", {}) for ev in events]
    assert any(p.get("buffered") is True for p in payloads)


@pytest.mark.live
def test_daemon_down_direct_rail(tmp_path, monkeypatch):
    """Daemon-down: `iai temporal-recall --json --as-of <iso> --limit 5 <cue>` exits 0 with _source direct-store."""
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)
    store_root = tmp_path / "store"

    env = dict(os.environ)
    env["HOME"] = str(fake_home)
    env["IAI_MCP_STORE"] = str(store_root)
    env.pop("IAI_DAEMON_SOCKET_PATH", None)

    import iai_mcp as _iai_mcp_pkg

    pkg_root = Path(_iai_mcp_pkg.__file__).resolve().parent.parent
    env["PYTHONPATH"] = (
        f"{pkg_root}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(pkg_root)
    )

    as_of = datetime.now(timezone.utc).isoformat()
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "iai_mcp.iai_cli",
            "temporal-recall",
            "--json",
            "--as-of",
            as_of,
            "--limit",
            "5",
            "test cue",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
        timeout=20,
    )

    assert proc.returncode == 0, (
        f"exit={proc.returncode} stderr={proc.stderr!r}"
    )
    payload = json.loads(proc.stdout)
    assert payload.get("_source") == "direct-store"


def test_as_of_mixed_separator_corpus(hermetic_store):
    """Mixed T-form and space-form created_at rows: as_of correctly filters both."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    # Touch the records table so the schema exists, then write two rows by direct SQL
    # with deliberately divergent created_at TEXT shapes (T-form and space-form).
    _ = store.db.open_table("records")
    db = store.db

    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    future = datetime.now(timezone.utc) + timedelta(days=1)

    row_t_form_id = "11111111-1111-1111-1111-111111111111"
    row_space_form_id = "22222222-2222-2222-2222-222222222222"

    with db._conn_lock:
        db._conn.execute(
            "INSERT INTO records (id, tier, literal_surface, aaak_index, embedding,"
            " centrality, detail_level, pinned, stability, difficulty,"
            " never_decay, never_merge, provenance_json, created_at, updated_at,"
            " tags_json, language, schema_version, hv_tier, structure_hv_payload,"
            " embedding_pending)"
            " VALUES (?, 'episodic', '', '', x'', 0.0, 1, 0, 0.0, 0.0, 0, 0,"
            " '[]', ?, ?, '[]', 'en', 1, 'bsc', x'', 0)",
            (row_t_form_id, yesterday.isoformat(), yesterday.isoformat()),
        )
        db._conn.execute(
            "INSERT INTO records (id, tier, literal_surface, aaak_index, embedding,"
            " centrality, detail_level, pinned, stability, difficulty,"
            " never_decay, never_merge, provenance_json, created_at, updated_at,"
            " tags_json, language, schema_version, hv_tier, structure_hv_payload,"
            " embedding_pending)"
            " VALUES (?, 'episodic', '', '', x'', 0.0, 1, 0, 0.0, 0.0, 0, 0,"
            " '[]', ?, ?, '[]', 'en', 1, 'bsc', x'', 0)",
            (row_space_form_id, str(future), str(future)),
        )

    midpoint = datetime.now(timezone.utc)
    out = core_dispatch(
        store,
        "memory_temporal_recall",
        {"as_of": midpoint.isoformat(), "limit": 50},
    )
    ids = {hit.get("id") for hit in out.get("hits", [])}
    assert row_t_form_id in ids
    assert row_space_form_id not in ids


def test_changed_since_strict_gt_semantics(hermetic_store):
    """changed_since uses strict > (not >=): boundary equals returns 0; boundary-1us returns the event."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.events import query_events, write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    write_event(store, kind="record_inserted", data={"x": 1}, severity="info")

    # Read back the ts of the event we just wrote
    rows = query_events(store, kind="record_inserted")
    assert len(rows) >= 1
    ts_val = rows[0]["ts"]
    if not isinstance(ts_val, datetime):
        ts_val = datetime.fromisoformat(str(ts_val).replace("Z", "+00:00").replace(" ", "T"))
    if ts_val.tzinfo is None:
        ts_val = ts_val.replace(tzinfo=timezone.utc)

    # Boundary equal: strict > drops it -> 0 events
    out_eq = core_dispatch(
        store,
        "memory_temporal_recall",
        {"changed_since": ts_val.isoformat(), "limit": 100},
    )
    eq_events = out_eq.get("changed_since_events", [])
    assert all(ev.get("kind") != "record_inserted" or ev.get("data", {}).get("x") != 1 for ev in eq_events)

    # Boundary - 1us: strict > keeps it
    earlier = ts_val - timedelta(microseconds=1)
    out_lt = core_dispatch(
        store,
        "memory_temporal_recall",
        {"changed_since": earlier.isoformat(), "limit": 100},
    )
    lt_events = out_lt.get("changed_since_events", [])
    assert any(
        ev.get("kind") == "record_inserted" and ev.get("data", {}).get("x") == 1
        for ev in lt_events
    )


def test_changed_since_bypasses_events_query_whitelist(hermetic_store):
    """Temporal recall sees non-whitelisted event kinds; events_query still rejects them."""
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    write_event(
        store,
        kind="s5_invariant_update",
        data={"private": True},
        severity="info",
    )

    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    out_temporal = core_dispatch(
        store,
        "memory_temporal_recall",
        {"changed_since": past, "limit": 100},
    )
    events = out_temporal.get("changed_since_events", [])
    kinds = {ev.get("kind") for ev in events}
    assert "s5_invariant_update" in kinds

    out_query = core_dispatch(store, "events_query", {"kind": "s5_invariant_update"})
    assert "error" in out_query


def test_normalize_ts_helper_handles_mixed_formats():
    """_normalize_ts_for_compare canonicalises T-form, space-form, Z-suffix, and aware datetimes."""
    from iai_mcp.store._store import _normalize_ts_for_compare

    t_form = _normalize_ts_for_compare("2026-06-17T09:14:29.542755+00:00")
    space_form = _normalize_ts_for_compare("2026-06-17 09:14:29.542755+00:00")
    dt_aware = _normalize_ts_for_compare(
        datetime(2026, 6, 17, 9, 14, 29, 542755, tzinfo=timezone.utc)
    )

    assert t_form == space_form == dt_aware

    z_suffix = _normalize_ts_for_compare("2026-06-17T09:14:29Z")
    no_micro = _normalize_ts_for_compare("2026-06-17T09:14:29+00:00")
    assert z_suffix == no_micro

    with pytest.raises(ValueError, match="ISO-8601"):
        _normalize_ts_for_compare("not a date")


def test_as_of_same_day_boundary(hermetic_store):
    """Same-UTC-date rows hours apart: as_of correctly excludes a future row and includes a past row.

    This is the bug reproduced by the review: when created_at is stored in space-form and
    as_of is T-form, same-date rows compare as <= the boundary regardless of actual time.
    The fix wraps both sides in SQLite datetime() so the separator byte is irrelevant.

    SQLite datetime() truncates to second resolution; the boundary tests use hour-apart and
    second-apart timestamps. Sub-second (µs) precision at the boundary is not a meaningful
    production invariant and is not tested here.

    Tests:
    - past (10:00) included in as_of=12:00
    - future (14:00) excluded from as_of=12:00
    - boundary-equal (12:00:00) included in as_of=12:00:00 (inclusive bound)
    - T+1s (12:00:01) excluded from as_of=12:00:00
    """
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    _ = store.db.open_table("records")
    db = store.db

    base = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
    past_id      = "aaaa0000-0000-0000-0000-000000000001"
    future_id    = "bbbb0000-0000-0000-0000-000000000002"
    equal_id     = "cccc0000-0000-0000-0000-000000000003"
    after_1s_id  = "dddd0000-0000-0000-0000-000000000004"

    past_ts   = datetime(2026, 6, 23, 10, 0, 0, tzinfo=timezone.utc)
    future_ts = datetime(2026, 6, 23, 14, 0, 0, tzinfo=timezone.utc)
    equal_ts  = base
    after_1s  = base + timedelta(seconds=1)

    def _insert_raw(row_id: str, ts: datetime) -> None:
        # Store in space-form (mimicking Python's sqlite3 datetime adapter) to
        # exercise the separator-mismatch scenario the fix must handle.
        ts_str = str(ts)
        with db._conn_lock:
            db._conn.execute(
                "INSERT INTO records (id, tier, literal_surface, aaak_index, embedding,"
                " centrality, detail_level, pinned, stability, difficulty,"
                " never_decay, never_merge, provenance_json, created_at, updated_at,"
                " tags_json, language, schema_version, hv_tier, structure_hv_payload,"
                " embedding_pending)"
                " VALUES (?, 'episodic', ?, '', x'', 0.0, 1, 0, 0.0, 0.0, 0, 0,"
                " '[]', ?, ?, '[]', 'en', 1, 'bsc', x'', 0)",
                (row_id, row_id, ts_str, ts_str),
            )

    _insert_raw(past_id, past_ts)
    _insert_raw(future_id, future_ts)
    _insert_raw(equal_id, equal_ts)
    _insert_raw(after_1s_id, after_1s)

    out = core_dispatch(
        store,
        "memory_temporal_recall",
        {"as_of": base.isoformat(), "limit": 50},
    )
    ids = {hit.get("id") for hit in out.get("hits", [])}

    assert past_id in ids,        "past row (10:00) must appear in as_of=12:00 view"
    assert future_id not in ids,  "future row (14:00) must NOT appear in as_of=12:00 view"
    assert equal_id in ids,       "boundary-equal row (12:00:00) must appear (inclusive bound)"
    assert after_1s_id not in ids, "T+1s row (12:00:01) must NOT appear in as_of=12:00:00 view"


def test_as_of_tombstoned_after_t_appears(hermetic_store):
    """WR-01: a record tombstoned AFTER as_of still appears in the as_of view (it existed then).

    A record tombstoned BEFORE as_of does NOT appear.
    """
    from iai_mcp.core import dispatch as core_dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=hermetic_store)
    _ = store.db.open_table("records")
    db = store.db

    base = datetime(2026, 6, 23, 12, 0, 0, tzinfo=timezone.utc)
    # row that existed at 10:00, tombstoned at 15:00 — must appear in as_of=12:00 view
    existed_then_id   = "eeee0000-0000-0000-0000-000000000005"
    # row that existed at 10:00, tombstoned at 10:30 — must NOT appear in as_of=12:00 view
    gone_before_t_id  = "ffff0000-0000-0000-0000-000000000006"

    existed_ts   = datetime(2026, 6, 23, 10, 0, 0, tzinfo=timezone.utc)
    tomb_after_t = datetime(2026, 6, 23, 15, 0, 0, tzinfo=timezone.utc)
    tomb_before_t = datetime(2026, 6, 23, 10, 30, 0, tzinfo=timezone.utc)

    def _insert_raw_with_tomb(row_id: str, created: datetime, tombstoned: datetime | None) -> None:
        tomb_str = str(tombstoned) if tombstoned is not None else None
        created_str = str(created)
        with db._conn_lock:
            db._conn.execute(
                "INSERT INTO records (id, tier, literal_surface, aaak_index, embedding,"
                " centrality, detail_level, pinned, stability, difficulty,"
                " never_decay, never_merge, provenance_json, created_at, updated_at,"
                " tombstoned_at, tags_json, language, schema_version,"
                " hv_tier, structure_hv_payload, embedding_pending)"
                " VALUES (?, 'episodic', ?, '', x'', 0.0, 1, 0, 0.0, 0.0, 0, 0,"
                " '[]', ?, ?, ?, '[]', 'en', 1, 'bsc', x'', 0)",
                (row_id, row_id, created_str, created_str, tomb_str),
            )

    _insert_raw_with_tomb(existed_then_id, existed_ts, tomb_after_t)
    _insert_raw_with_tomb(gone_before_t_id, existed_ts, tomb_before_t)

    out = core_dispatch(
        store,
        "memory_temporal_recall",
        {"as_of": base.isoformat(), "limit": 50},
    )
    ids = {hit.get("id") for hit in out.get("hits", [])}

    assert existed_then_id in ids, (
        "record tombstoned AFTER as_of must appear in point-in-time view (it existed at T)"
    )
    assert gone_before_t_id not in ids, (
        "record tombstoned BEFORE as_of must NOT appear (it was gone by T)"
    )
