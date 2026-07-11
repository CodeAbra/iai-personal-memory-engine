from __future__ import annotations

import json
import platform
from datetime import datetime, timezone

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)

SESSION_ID = "cccccccc-cccc-cccc-cccc-cccccccccccc"
TRANSCRIPT_TS = "2026-05-31T17:41:01.968Z"
TURN_TEXT = "маркер один — unique marker for re-emission idem test"


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-source-uuid-idem-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def test_same_uuid_twice_is_one_row(iai_home):
    from iai_mcp.capture import capture_turn

    store = _open_store()
    SRC_UUID = "d31c0b76-1111-2222-3333-444455556666"

    r1 = capture_turn(
        store,
        cue="hook emit 1",
        text=TURN_TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TRANSCRIPT_TS,
        source_uuid=SRC_UUID,
    )
    assert r1["status"] == "inserted", f"First emit must insert; got {r1}"

    fresh_ts = datetime.now(timezone.utc).isoformat()
    r2 = capture_turn(
        store,
        cue="hook emit 2 (re-emission)",
        text=TURN_TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=fresh_ts,
        source_uuid=SRC_UUID,
    )
    assert r2["status"] == "reinforced", (
        f"Re-emission of same transcript uuid must be deduplicated (reinforced); "
        f"got {r2!r}.  This is the core regression — a second row would indicate "
        f"the idem key is still now()-based."
    )
    assert r2["record_id"] == r1["record_id"], (
        "Reinforced record_id must equal the original inserted id"
    )

    turns = store.recent_user_turns(n=50, session_id=SESSION_ID)
    matching = [t for t in turns if TURN_TEXT in (t.literal_surface or "")]
    assert len(matching) == 1, (
        f"Expected exactly 1 row for TURN_TEXT; found {len(matching)}.  "
        f"Duplicate rows indicate idem dedup is broken: {[t.literal_surface for t in matching]}"
    )

    rec = matching[0]
    assert rec.created_at is not None
    assert rec.created_at.year == 2026, (
        f"created_at year should be 2026 (transcript time); got {rec.created_at}"
    )
    assert rec.created_at.month == 5
    assert rec.created_at.day == 31
    assert rec.created_at.hour == 17
    assert rec.created_at.minute == 41


def test_distinct_uuids_same_text_are_two_rows(iai_home):
    from iai_mcp.capture import capture_turn

    store = _open_store()
    UUID_1 = "aaaa0000-1111-2222-3333-444455556666"
    UUID_2 = "bbbb0000-1111-2222-3333-444455556666"
    TS_1 = "2026-05-31T10:00:00.000Z"
    TS_2 = "2026-05-31T10:05:00.000Z"
    TEXT = "identical text in two distinct turns for verbatim test"

    r1 = capture_turn(
        store,
        cue="turn 1",
        text=TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TS_1,
        source_uuid=UUID_1,
    )
    r2 = capture_turn(
        store,
        cue="turn 2",
        text=TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TS_2,
        source_uuid=UUID_2,
    )

    assert r1["status"] == "inserted", f"Turn 1 must insert; got {r1}"
    assert r2["status"] == "inserted", (
        f"Turn 2 (distinct uuid, same text) must also insert; got {r2!r}.  "
        f"status=reinforced would indicate the uuid-keyed idem collapsed two "
        f"distinct turns — that is a verbatim-loss bug."
    )
    assert r1["record_id"] != r2["record_id"], "Two distinct turns must get distinct record ids"

    turns = store.recent_user_turns(n=50, session_id=SESSION_ID)
    matching = [t for t in turns if TEXT in (t.literal_surface or "")]
    assert len(matching) == 2, (
        f"Expected 2 rows (verbatim 1:1); found {len(matching)}.  "
        f"Rows: {[t.literal_surface for t in matching]}"
    )


def test_no_uuid_fallback_same_ts_text_is_one_row(iai_home):
    from iai_mcp.capture import capture_turn

    store = _open_store()
    TS = "2026-05-31T09:00:00.000000+00:00"
    TEXT = "fallback key test same ts same text ensures one row here"

    r1 = capture_turn(
        store,
        cue="call 1",
        text=TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TS,
        source_uuid=None,
    )
    r2 = capture_turn(
        store,
        cue="call 2",
        text=TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TS,
        source_uuid=None,
    )

    assert r1["status"] == "inserted", f"First call must insert; got {r1}"
    assert r2["status"] == "reinforced", (
        f"Identical (session, role, ts, text) without uuid must reinforce; got {r2}"
    )

    turns = store.recent_user_turns(n=50, session_id=SESSION_ID)
    matching = [t for t in turns if TEXT in (t.literal_surface or "")]
    assert len(matching) == 1, f"Expected 1 row; found {len(matching)}"


def test_no_uuid_fallback_different_ts_is_two_rows(iai_home):
    from iai_mcp.capture import capture_turn

    store = _open_store()
    TS_1 = "2026-05-31T08:00:00.000000+00:00"
    TS_2 = "2026-05-31T08:05:00.000000+00:00"
    TEXT = "fallback key test different timestamps produce two rows check"

    r1 = capture_turn(
        store,
        cue="call 1",
        text=TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TS_1,
        source_uuid=None,
    )
    r2 = capture_turn(
        store,
        cue="call 2",
        text=TEXT,
        tier="episodic",
        session_id=SESSION_ID,
        role="user",
        ts=TS_2,
        source_uuid=None,
    )

    assert r1["status"] == "inserted"
    assert r2["status"] == "inserted", (
        f"Different ts without uuid must insert (distinct idem key); got {r2}"
    )


def test_drain_deferred_deduplicates_already_inserted_uuid(iai_home):
    from iai_mcp.capture import capture_turn, drain_deferred_captures

    store = _open_store()
    SRC_UUID = "e5f6g7h8-1234-5678-9abc-defabcdef012"
    TS = "2026-05-31T18:00:00.000Z"
    TEXT = "cross-path dedup test: drain_deferred must not re-insert uuid-keyed turn"
    SESSION = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    r1 = capture_turn(
        store,
        cue="drain_active path",
        text=TEXT,
        tier="episodic",
        session_id=SESSION,
        role="user",
        ts=TS,
        source_uuid=SRC_UUID,
    )
    assert r1["status"] == "inserted", f"First insert must succeed; got {r1}"
    record_id_1 = r1["record_id"]

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    drain_file = deferred_dir / f"{SESSION}-1234567890.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-31T18:00:00.000Z",
        "session_id": SESSION,
        "cwd": "/tmp/test",
    }
    event = {
        "text": TEXT,
        "cue": "drain_deferred path",
        "tier": "episodic",
        "role": "user",
        "ts": TS,
        "source_uuid": SRC_UUID,
    }
    with drain_file.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.write(json.dumps(event) + "\n")

    counts = drain_deferred_captures(store)

    assert counts["events_reinforced"] == 1, (
        f"drain_deferred must reinforce (not re-insert) a turn already captured "
        f"by uuid key; got counts={counts!r}.  events_inserted>0 indicates the "
        f"drain_deferred path dropped source_uuid and keyed on (session,role,ts,text), "
        f"computing a different hash than the uuid-keyed idem tag stamped earlier."
    )
    assert counts["events_inserted"] == 0, (
        f"No new inserts expected; got counts={counts!r}"
    )

    turns = store.recent_user_turns(n=50, session_id=SESSION)
    matching = [t for t in turns if TEXT in (t.literal_surface or "")]
    assert len(matching) == 1, (
        f"Expected 1 row after cross-path drain; found {len(matching)}.  "
        f"A second row here is the cross-path duplicate bug."
    )


def test_dedup_with_ts_microsecond_normalization(iai_home):
    from iai_mcp.capture import (
        drain_deferred_captures,
        read_pending_live_events,
        write_deferred_event,
    )

    session = "ts-norm-session-60h5"
    text = "ts microsecond normalization dedup test turn content long enough"
    ts_microsec = "2026-05-31T12:00:00.000000+00:00"

    store = _open_store()

    write_deferred_event(session, "user", text, ts=ts_microsec)

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    deferred_dir.mkdir(parents=True, exist_ok=True)
    drain_file = deferred_dir / f"{session}-drain-1234567890.jsonl"
    header_d = {
        "version": 1,
        "deferred_at": "2026-05-31T12:00:00.000000+00:00",
        "session_id": session,
        "cwd": "/tmp",
    }
    event_d = {
        "text": text,
        "cue": f"session {session} turn",
        "tier": "episodic",
        "role": "user",
        "ts": ts_microsec,
    }
    with drain_file.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header_d) + "\n")
        fh.write(json.dumps(event_d) + "\n")

    counts = drain_deferred_captures(store)
    assert counts["events_inserted"] >= 1, (
        f"drain must insert the turn; got counts={counts!r}"
    )

    pending = read_pending_live_events(session_id=session)
    turns = store.recent_user_turns(10, session_id=session, pending_live_events=pending)

    matching = [t for t in turns if text in (t.literal_surface or "")]
    assert len(matching) == 1, (
        f"same turn (ts normalization path, no uuid) must appear once after drain; "
        f"got {len(matching)}: {[t.literal_surface for t in turns]!r}. "
        f"count>1 means read-time ev['ts_iso'] differs from drain-time ts_iso."
    )
