from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.capture import capture_turn
from iai_mcp.core import dispatch
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import make_tmp_store


def _write_live_event(
    home: Path,
    session_id: str,
    text: str,
    *,
    role: str = "user",
    ts: str | None = None,
    source_uuid: str | None = None,
) -> None:
    from iai_mcp.capture import write_deferred_event
    write_deferred_event(session_id, role, text, ts=ts, source_uuid=source_uuid)


def test_returns_n_most_recent_user_turns_time_desc(tmp_path):
    store = make_tmp_store(tmp_path)

    for i in range(5):
        res = capture_turn(
            store,
            cue=f"user turn alpha{i}",
            text=f"user turn alpha{i} mk59 episodic content for recency test",
            tier="episodic",
            session_id="sess-recency-test",
            role="user",
        )
        assert res["status"] == "inserted", f"insert {i} failed: {res}"

    result = dispatch(store, "episodes_recent", {"n": 3})
    assert "turns" in result, (
        f"episodes_recent response missing 'turns' key; "
        f"dispatch returned: {result!r}"
    )
    turns = result["turns"]
    assert len(turns) == 3, f"expected 3 turns, got {len(turns)}"

    timestamps = [t.get("captured_at") for t in turns]
    assert timestamps == sorted(timestamps, reverse=True), (
        f"turns must be newest-first; got {timestamps}"
    )


def test_session_id_filter(tmp_path):
    store = make_tmp_store(tmp_path)

    for i in range(3):
        capture_turn(
            store,
            cue=f"x turn {i}",
            text=f"session x turn {i} distinctive content mk59 xfilter test",
            tier="episodic",
            session_id="session-X-filter",
            role="user",
        )

    for i in range(2):
        capture_turn(
            store,
            cue=f"y turn {i}",
            text=f"session y turn {i} should not appear in x filter result",
            tier="episodic",
            session_id="session-Y-filter",
            role="user",
        )

    result = dispatch(
        store,
        "episodes_recent",
        {"n": 5, "session_id": "session-X-filter"},
    )
    assert "turns" in result, f"episodes_recent response missing 'turns': {result!r}"
    turns = result["turns"]
    assert len(turns) == 3, f"expected 3 turns for session X, got {len(turns)}"
    for t in turns:
        assert t.get("session_id") == "session-X-filter", (
            f"turn {t.get('record_id')} belongs to wrong session: {t.get('session_id')!r}"
        )
    assert turns[0]["literal_surface"].startswith("session x turn 2"), (
        f"most-recent X turn unexpected: {turns[0]['literal_surface']!r}"
    )


def test_no_filter_returns_global_most_recent(tmp_path):
    store = make_tmp_store(tmp_path)

    for i in range(3):
        capture_turn(
            store,
            cue=f"global turn {i}",
            text=f"global turn {i} earlier content for global recency check",
            tier="episodic",
            session_id="sess-global-A",
            role="user",
        )

    capture_turn(
        store,
        cue="final distinctive turn",
        text="final distinctive turn bxyz9999 globally newest in store",
        tier="episodic",
        session_id="sess-global-B",
        role="user",
    )

    result = dispatch(store, "episodes_recent", {"n": 5})
    assert "turns" in result, f"episodes_recent response missing 'turns': {result!r}"
    turns = result["turns"]
    assert turns, "expected at least one turn"
    assert "bxyz9999" in turns[0]["literal_surface"], (
        f"globally newest turn not first (got {turns[0]['literal_surface']!r})"
    )


def test_negative_n_clamp_returns_empty(tmp_path):
    store = make_tmp_store(tmp_path)
    capture_turn(
        store,
        cue="clamp test",
        text="clamp test content mk59 negative n guard",
        tier="episodic",
        session_id="sess-clamp",
        role="user",
    )

    result = dispatch(store, "episodes_recent", {"n": -5})
    assert "turns" in result, f"episodes_recent missing 'turns': {result!r}"
    turns = result["turns"]
    assert turns == [], f"n=-5 must return empty list; got {turns!r}"
    assert result.get("count") == 0


pytestmark_posix = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths / live-capture helper",
)


@pytest.fixture
def iai_home_60(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    yield tmp_path


def test_no_double_count_after_drain(iai_home_60):
    store = make_tmp_store(iai_home_60)

    session = "dedup-session-60a"
    text = "dedup test turn content that is long enough for capture mk60 test"
    src_uuid = "uuid-dedup-a001"
    ts_str = "2026-05-31T10:00:00.000000+00:00"

    capture_turn(
        store,
        cue="dedup drain cue",
        text=text,
        tier="episodic",
        session_id=session,
        role="user",
        ts=ts_str,
        source_uuid=src_uuid,
    )

    _write_live_event(iai_home_60, session, text, ts=ts_str, source_uuid=src_uuid)

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    assert "turns" in result, f"episodes_recent response missing 'turns': {result!r}"
    turns = result["turns"]
    assert len(turns) == 1, (
        f"expected 1 turn after dedup; got {len(turns)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_dedup_pending_vs_pending(iai_home_60):
    session = "dedup-pend-session-60b"
    text = "pending dedup test turn text long enough for mk60 pending dedup"
    src_uuid = "uuid-pend-dedup-b002"
    ts_str = "2026-05-31T11:00:00.000000+00:00"

    store = make_tmp_store(iai_home_60)

    _write_live_event(iai_home_60, session, text, ts=ts_str, source_uuid=src_uuid)
    _write_live_event(iai_home_60, session, text, ts=ts_str, source_uuid=src_uuid)

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    assert "turns" in result
    turns = result["turns"]
    assert len(turns) == 1, (
        f"re-emitted pending line must appear once (seen_pending_idem dedup); "
        f"got {len(turns)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_live_turn_sorts_before_older_stored(iai_home_60):
    session = "sort-live-session-60c"
    text_old = "older stored turn for recency sort test mk60 content here"
    text_new = "newer pending live turn sort test mk60 content here brand new"

    ts_old = "2026-05-31T08:00:00.000000+00:00"
    ts_new = "2026-05-31T09:00:00.000000+00:00"

    store = make_tmp_store(iai_home_60)
    capture_turn(
        store,
        cue="old stored turn",
        text=text_old,
        tier="episodic",
        session_id=session,
        role="user",
        ts=ts_old,
        source_uuid="uuid-old-stored-60c",
    )

    _write_live_event(
        iai_home_60, session, text_new,
        ts=ts_new, source_uuid="uuid-new-live-60c",
    )

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    assert len(turns) >= 2, f"expected >= 2 turns; got {len(turns)}"
    assert text_new in turns[0]["literal_surface"], (
        f"newer live turn must be first; got {turns[0]['literal_surface']!r}"
    )


def test_distinct_uuid_same_text_both_appear(iai_home_60):
    session = "distinct-uuid-session-60d"
    text = "identical text turn for distinct uuid test content mk60 here"
    ts1 = "2026-05-31T12:00:00.000000+00:00"
    ts2 = "2026-05-31T12:01:00.000000+00:00"

    store = make_tmp_store(iai_home_60)

    _write_live_event(
        iai_home_60, session, text, ts=ts1, source_uuid="uuid-distinct-1"
    )
    _write_live_event(
        iai_home_60, session, text, ts=ts2, source_uuid="uuid-distinct-2"
    )

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    matching = [t for t in turns if t.get("literal_surface") == text]
    assert len(matching) == 2, (
        f"two distinct-uuid events with same text must both appear; "
        f"got {len(matching)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_pending_assistant_excluded_from_user_turns(iai_home_60):
    session = "asst-role-session-60e"
    text = "assistant response text that should not appear in user turns query here"

    store = make_tmp_store(iai_home_60)
    _write_live_event(iai_home_60, session, text, role="assistant")

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    assert len(turns) == 0, (
        f"assistant-role pending turn must NOT appear in episodes_recent; "
        f"got {len(turns)}: {[t.get('literal_surface') for t in turns]!r}"
    )


def test_pending_malformed_role_dropped(iai_home_60):
    session = "system-role-session-60f"

    store = make_tmp_store(iai_home_60)

    deferred = iai_home_60 / ".iai-mcp" / ".deferred-captures"
    deferred.mkdir(parents=True, exist_ok=True)
    path = deferred / f"{session}.live.jsonl"
    header = json.dumps({
        "version": 1, "deferred_at": "2026-05-31T12:00:00+00:00",
        "session_id": session, "cwd": "/tmp",
    })
    ev = json.dumps({
        "text": "system message that must be dropped from user turns query here",
        "role": "system",
        "tier": "episodic",
        "ts": "2026-05-31T12:00:00.000000+00:00",
    })
    path.write_text(header + "\n" + ev + "\n", encoding="utf-8")

    result = dispatch(store, "episodes_recent", {"n": 10, "session_id": session})
    turns = result["turns"]
    assert len(turns) == 0, (
        f"system-role pending event must be dropped by recent_user_turns; got {turns!r}"
    )


def test_pending_record_id_not_literal_none(iai_home_60):
    session_with_uuid = "pending-rid-session-60g1"
    session_no_uuid = "pending-rid-session-60g2"
    text = "pending record id test content long enough for mk60 test here"
    src_uuid = "uuid-rid-test-g001"

    store = make_tmp_store(iai_home_60)

    _write_live_event(iai_home_60, session_with_uuid, text, source_uuid=src_uuid)
    result1 = dispatch(store, "episodes_recent", {"n": 10, "session_id": session_with_uuid})
    turns1 = result1["turns"]
    assert len(turns1) == 1, f"expected 1 turn; got {len(turns1)}"
    rid1 = turns1[0]["record_id"]
    assert rid1 != "None", f"record_id must not be literal 'None'; got {rid1!r}"
    assert rid1.startswith("pending:"), f"pending turn record_id must start with 'pending:'; got {rid1!r}"
    assert src_uuid in rid1, f"source_uuid must be in record_id; got {rid1!r}"

    _write_live_event(iai_home_60, session_no_uuid, text)
    result2 = dispatch(store, "episodes_recent", {"n": 10, "session_id": session_no_uuid})
    turns2 = result2["turns"]
    assert len(turns2) == 1, f"expected 1 turn; got {len(turns2)}"
    rid2 = turns2[0]["record_id"]
    assert rid2 != "None", f"record_id must not be literal 'None'; got {rid2!r}"
    assert rid2.startswith("pending:"), f"must start with 'pending:'; got {rid2!r}"
    suffix = rid2[len("pending:"):]
    assert suffix, f"record_id must not be bare 'pending:'; got {rid2!r}"


# ---------------------------------------------------------------------------
# O(corpus) regression guard for recent_user_turns (RED on HEAD:
# recent_user_turns calls self.all_records() -- full-table materialization).
# ---------------------------------------------------------------------------

def _seed_episodic_role_user_corpus(store: MemoryStore, n: int, *, prefix: str) -> None:
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ca = base_ts + timedelta(seconds=i)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=f"{prefix} corpus scaling probe turn {i}",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": f"{prefix}-scaling-session"}],
            created_at=ca,
            updated_at=ca,
            tags=["capture", "role:user"],
            language="en",
        )
        store.insert(rec)


def _timed_recent_user_turns(store: MemoryStore, *, n: int = 10) -> float:
    t0 = time.perf_counter()
    store.recent_user_turns(n=n)
    return time.perf_counter() - t0


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recent_user_turns_not_o_corpus(driver, tmp_path, monkeypatch):
    """recent_user_turns wall-time must not scale linearly with corpus size.

    The cands read issues a bare ORDER BY created_at DESC LIMIT k (no WHERE),
    riding the ordered-index fast path, with Python-side widening of k until
    n matches are found -- decoupling read cost from total corpus size.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    small_store_path = tmp_path / f"small-{driver}"
    small_store_path.mkdir(parents=True, exist_ok=True)
    small_store = MemoryStore(path=small_store_path)
    try:
        _seed_episodic_role_user_corpus(small_store, 50, prefix="small")
        # Warm-up call excluded from the timed sample (one-time index/open cost).
        _timed_recent_user_turns(small_store, n=10)
        small_time = min(
            _timed_recent_user_turns(small_store, n=10) for _ in range(3)
        )
    finally:
        small_store.close()

    large_store_path = tmp_path / f"large-{driver}"
    large_store_path.mkdir(parents=True, exist_ok=True)
    large_store = MemoryStore(path=large_store_path)
    try:
        _seed_episodic_role_user_corpus(large_store, 1500, prefix="large")
        _timed_recent_user_turns(large_store, n=10)
        large_time = min(
            _timed_recent_user_turns(large_store, n=10) for _ in range(3)
        )
    finally:
        large_store.close()

    ratio = large_time / max(small_time, 1e-6)
    assert ratio <= 4.0, (
        f"[driver={driver}] recent_user_turns scaled {ratio:.1f}x from 50 to "
        f"1500 episodic records (small={small_time*1000:.2f}ms, "
        f"large={large_time*1000:.2f}ms) -- expected a bounded read, not an "
        f"O(corpus) full-table scan"
    )


# ---------------------------------------------------------------------------
# Dedup-branch O(corpus) regression guard (RED on HEAD: the pending_live_events
# branch builds store_idem_set by scanning every episodic record's tags).
# ---------------------------------------------------------------------------

def _seed_episodic_with_idem_tags(store: MemoryStore, n: int, *, prefix: str) -> None:
    from iai_mcp.capture import _idem_tag

    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for i in range(n):
        ca = base_ts + timedelta(seconds=i)
        ts_iso = ca.isoformat()
        text = f"{prefix} corpus scaling probe turn {i}"
        session_id = f"{prefix}-scaling-session"
        source_uuid = f"{prefix}-src-{i}"
        idem = _idem_tag(session_id, "user", ts_iso, text, source_uuid=source_uuid)
        rec = MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=text,
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": session_id}],
            created_at=ca,
            updated_at=ca,
            tags=["capture", "role:user", idem],
            language="en",
        )
        store.insert(rec)


def _timed_recent_user_turns_dedup(
    store: MemoryStore, *, events: list, n: int = 10
) -> float:
    t0 = time.perf_counter()
    store.recent_user_turns(n=n, pending_live_events=events)
    return time.perf_counter() - t0


def _fixed_pending_events(prefix: str) -> list:
    """A small, corpus-size-independent mix of dedup HITs and MISSes.

    HITs match a stored record's idem tag exactly (same session/role/
    source_uuid triple used by ``_seed_episodic_with_idem_tags`` for
    indices 0-3 of the *small* corpus's own session name) -- reusing the
    "small" prefix session/index pairing on purpose so this fixed event
    list produces real HITs against BOTH the small and large seeded
    corpora (both corpora are seeded under their own prefix, but the
    dedup check only depends on whether a matching idem tag exists in
    *this* store, so HIT/MISS composition is intentionally re-derived
    per corpus below via the ``prefix`` argument).
    """
    base_ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events = []
    # 4 events that MATCH stored records (idem HIT -> skipped).
    for i in range(4):
        ca = base_ts + timedelta(seconds=i)
        events.append({
            "text": f"{prefix} corpus scaling probe turn {i}",
            "role": "user",
            "tier": "episodic",
            "session_id": f"{prefix}-scaling-session",
            "ts": ca,
            "ts_iso": ca.isoformat(),
            "source_uuid": f"{prefix}-src-{i}",
        })
    # 4 events that are genuinely NEW (idem MISS -> kept as pending turns).
    for i in range(4):
        ca = base_ts + timedelta(days=1, seconds=i)
        events.append({
            "text": f"{prefix} brand new live turn {i}",
            "role": "user",
            "tier": "episodic",
            "session_id": f"{prefix}-scaling-session",
            "ts": ca,
            "ts_iso": ca.isoformat(),
            "source_uuid": f"{prefix}-live-src-{i}",
        })
    return events


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recent_user_turns_dedup_not_o_corpus(driver, tmp_path, monkeypatch):
    """recent_user_turns(pending_live_events=...) dedup wall-time must not
    scale linearly with corpus size.

    Unlike test_recent_user_turns_not_o_corpus (which never passes
    pending_live_events, so it only exercises the cands scan), this test
    drives the dedup branch directly with a small, FIXED-SIZE pending
    events list at two corpus sizes. The dedup check reuses the existing
    per-tag point lookup (find_record_by_tag), isolating its cost from the
    (bounded) pending-events list size rather than the full episodic corpus.
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    small_store_path = tmp_path / f"dedup-small-{driver}"
    small_store_path.mkdir(parents=True, exist_ok=True)
    small_store = MemoryStore(path=small_store_path)
    try:
        _seed_episodic_with_idem_tags(small_store, 50, prefix="small")
        small_events = _fixed_pending_events("small")
        assert len(small_events) == 8
        # Warm-up call excluded from the timed sample (one-time index/open cost).
        _timed_recent_user_turns_dedup(small_store, events=small_events, n=10)
        small_time = min(
            _timed_recent_user_turns_dedup(small_store, events=small_events, n=10)
            for _ in range(3)
        )
    finally:
        small_store.close()

    large_store_path = tmp_path / f"dedup-large-{driver}"
    large_store_path.mkdir(parents=True, exist_ok=True)
    large_store = MemoryStore(path=large_store_path)
    try:
        _seed_episodic_with_idem_tags(large_store, 1500, prefix="large")
        large_events = _fixed_pending_events("large")
        assert len(large_events) == 8
        _timed_recent_user_turns_dedup(large_store, events=large_events, n=10)
        large_time = min(
            _timed_recent_user_turns_dedup(large_store, events=large_events, n=10)
            for _ in range(3)
        )
    finally:
        large_store.close()

    ratio = large_time / max(small_time, 1e-6)
    assert ratio <= 4.0, (
        f"[driver={driver}] recent_user_turns(pending_live_events=...) "
        f"dedup scaled {ratio:.1f}x from 50 to 1500 episodic records "
        f"(small={small_time*1000:.2f}ms, large={large_time*1000:.2f}ms) "
        f"-- expected a bounded read (fixed pending-events count), not an "
        f"O(corpus) store_idem_set scan"
    )


# ---------------------------------------------------------------------------
# Result/order equivalence anchor (GREEN on HEAD): pins what recent_user_turns
# returns today so the query rewrite cannot silently change it.
# ---------------------------------------------------------------------------

def _make_episodic_record(
    *,
    rid,
    ca,
    role_tag: str | None = "role:user",
    embedding_pending: bool = False,
    session_id: str = "equiv-session",
    tier: str = "episodic",
) -> MemoryRecord:
    tags = ["capture"]
    if role_tag is not None:
        tags.append(role_tag)
    return MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=f"equivalence probe {rid}",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        embedding_pending=embedding_pending,
        provenance=[{"session_id": session_id}],
        created_at=ca,
        updated_at=ca,
        tags=tags,
        language="en",
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_recent_user_turns_equivalence(driver, tmp_path, monkeypatch):
    """Golden id-set + descending-created_at order anchor for the query rewrite.

    Pins EXACTLY what recent_user_turns returns today (both the plain call
    and the pending_live_events call) for a fixed, deterministic seed corpus
    -- including a created_at TIE -- so a future query-shape rewrite is
    checked against a concrete expected result, not just "same as before".
    """
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")

    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    if driver == "lilli":
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    store_path = tmp_path / f"equiv-{driver}"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(path=store_path)
    try:
        base_ts = datetime(2026, 2, 1, tzinfo=timezone.utc)

        # Two user-role episodic records, distinct created_at, newest first.
        rid_user_newest = uuid4()
        rid_user_older = uuid4()
        # Two records sharing the SAME created_at (a tie) -- one user, one
        # non-user -- to pin tie-order behavior (stable sort on equal keys).
        rid_tie_user = uuid4()
        rid_tie_other_role = uuid4()
        # A non-user-role episodic record (must be excluded).
        rid_non_user_role = uuid4()
        # A non-episodic record (must be excluded).
        rid_non_episodic = uuid4()
        # An embedding_pending record with no role:user tag (must be
        # INCLUDED -- recent_user_turns keeps embedding_pending regardless
        # of role tag, per its current `or r.embedding_pending` branch).
        rid_pending = uuid4()

        store.insert(_make_episodic_record(
            rid=rid_user_newest, ca=base_ts + timedelta(seconds=40),
        ))
        pending_ca = base_ts + timedelta(seconds=30)
        pending_ca_iso = pending_ca.isoformat()
        store.insert_pending(
            record_id=str(rid_pending),
            tier="episodic",
            literal_surface=f"equivalence probe {rid_pending}",
            tags_json=json.dumps(["capture"]),
            provenance_json=json.dumps([{"session_id": "equiv-session"}]),
            created_at=pending_ca_iso,
            updated_at=pending_ca_iso,
        )
        store.insert(_make_episodic_record(
            rid=rid_user_older, ca=base_ts + timedelta(seconds=20),
        ))
        tie_ca = base_ts + timedelta(seconds=10)
        store.insert(_make_episodic_record(rid=rid_tie_user, ca=tie_ca))
        store.insert(_make_episodic_record(
            rid=rid_tie_other_role, ca=tie_ca, role_tag="role:assistant",
        ))
        store.insert(_make_episodic_record(
            rid=rid_non_user_role, ca=base_ts + timedelta(seconds=5),
            role_tag="role:assistant",
        ))
        store.insert(_make_episodic_record(
            rid=rid_non_episodic, ca=base_ts + timedelta(seconds=50),
            tier="semantic",
        ))

        # --- Call 1: no pending events. ---
        result_plain = store.recent_user_turns(n=10)
        plain_ids = [r.id for r in result_plain]

        expected_plain_ids = [
            rid_user_newest,
            rid_pending,
            rid_user_older,
            rid_tie_user,
        ]
        assert plain_ids == expected_plain_ids, (
            f"[driver={driver}] recent_user_turns(n=10) id order mismatch: "
            f"got {plain_ids}, expected {expected_plain_ids}"
        )
        plain_cas = [r.created_at for r in result_plain]
        assert plain_cas == sorted(plain_cas, reverse=True), (
            f"[driver={driver}] recent_user_turns(n=10) not descending by "
            f"created_at: {plain_cas}"
        )

        # --- Call 2: with a small pending_live_events list (1 HIT, 1 MISS). ---
        from iai_mcp.capture import _idem_tag

        hit_session = "equiv-session"
        hit_ts = base_ts + timedelta(seconds=1)
        hit_ts_iso = hit_ts.isoformat()
        hit_text = "equivalence dedup hit probe"
        hit_source_uuid = "equiv-hit-src"
        hit_idem = _idem_tag(
            hit_session, "user", hit_ts_iso, hit_text, source_uuid=hit_source_uuid,
        )
        rid_dedup_hit = uuid4()
        store.insert(MemoryRecord(
            id=rid_dedup_hit,
            tier="episodic",
            literal_surface=hit_text,
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.0,
            detail_level=2,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=[{"session_id": hit_session}],
            created_at=hit_ts,
            updated_at=hit_ts,
            tags=["capture", "role:user", hit_idem],
            language="en",
        ))

        miss_ts = base_ts + timedelta(seconds=60)
        pending_events = [
            {
                "text": hit_text,
                "role": "user",
                "tier": "episodic",
                "session_id": hit_session,
                "ts": hit_ts,
                "ts_iso": hit_ts_iso,
                "source_uuid": hit_source_uuid,
            },
            {
                "text": "equivalence dedup miss probe",
                "role": "user",
                "tier": "episodic",
                "session_id": hit_session,
                "ts": miss_ts,
                "ts_iso": miss_ts.isoformat(),
                "source_uuid": "equiv-miss-src",
            },
        ]

        result_pending = store.recent_user_turns(n=10, pending_live_events=pending_events)
        pending_ids = [r.id for r in result_pending]

        assert rid_dedup_hit in pending_ids, (
            f"[driver={driver}] the stored record backing the dedup HIT "
            f"must still appear (it is a real stored record, independent "
            f"of the pending-event dedup check): got {pending_ids}"
        )
        assert None in pending_ids, (
            f"[driver={driver}] the dedup MISS pending event must be kept "
            f"as a synthetic pending turn (id=None): got {pending_ids}"
        )
        pending_texts = [r.literal_surface for r in result_pending]
        assert "equivalence dedup miss probe" in pending_texts, (
            f"[driver={driver}] dedup MISS pending event's text must "
            f"appear in the result: got {pending_texts}"
        )
        assert pending_texts.count(hit_text) == 1, (
            f"[driver={driver}] dedup HIT pending event must be SKIPPED "
            f"(not double-counted alongside its own stored record): "
            f"got {pending_texts}"
        )
        pending_cas = [r.created_at for r in result_pending]
        assert pending_cas == sorted(pending_cas, reverse=True), (
            f"[driver={driver}] recent_user_turns(n=10, pending_live_events=...) "
            f"not descending by created_at: {pending_cas}"
        )
    finally:
        store.close()
