from __future__ import annotations

import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

pytestmark = pytest.mark.skipif(
    platform.system() == "Windows",
    reason="POSIX paths + UNIX socket semantics",
)


SHORT_TURN = "ok sounds good here"

SESSION_A = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SESSION_B = "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"

TS_A = "2026-05-30T10:00:00.000000+00:00"
TS_B = "2026-05-30T11:00:00.000000+00:00"

TS_TURN1 = "2026-05-30T10:00:00.000000+00:00"
TS_TURN2 = "2026-05-30T10:00:05.000000+00:00"


@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-episodic-dedup-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    import keyring.core
    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None


def _open_store():
    from iai_mcp.store import MemoryStore
    return MemoryStore()


def _write_live_file(
    deferred_dir: Path,
    session_id: str,
    events: list[dict],
) -> Path:
    deferred_dir.mkdir(parents=True, exist_ok=True)
    path = deferred_dir / f"{session_id}.live.jsonl"
    header = {
        "version": 1,
        "deferred_at": "2026-05-30T10:00:00.000000+00:00",
        "session_id": session_id,
        "cwd": "/tmp/test",
    }
    lines = [json.dumps(header, ensure_ascii=False)]
    for ev in events:
        lines.append(json.dumps(ev, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n")
    return path


def _drain(store, session_id_to_drain, exclude_session_id):
    from iai_mcp.capture import drain_active_live_captures
    return drain_active_live_captures(store, exclude_session_id=exclude_session_id)


def _offset_path(iai_home, session_id):
    state_dir = iai_home / ".iai-mcp" / ".capture-state"
    return state_dir / f"{session_id}.drain-offset"


def test_a_two_sessions_distinct_rows(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"

    _write_live_file(deferred_dir, SESSION_A, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_A},
    ])
    _write_live_file(deferred_dir, SESSION_B, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_B},
    ])

    store = _open_store()

    c1 = drain_active_live_captures(store, exclude_session_id=SESSION_B)
    assert c1["events_inserted"] == 1, f"Expected 1 insert for session A; got {c1}"

    c2 = drain_active_live_captures(store, exclude_session_id=SESSION_A)
    assert c2["events_inserted"] == 1, f"Expected 1 insert for session B; got {c2}"

    all_turns = store.recent_user_turns(n=50, session_id=None)
    matching = [r for r in all_turns if r.literal_surface == SHORT_TURN]
    assert len(matching) == 2, (
        f"Expected 2 distinct rows for identical short turn; got {len(matching)}: "
        f"{[r.literal_surface for r in all_turns]}"
    )
    session_ids = {
        (r.provenance or [{}])[0].get("session_id") for r in matching
    }
    assert session_ids == {SESSION_A, SESSION_B}, (
        f"Expected provenance for both sessions; got {session_ids}"
    )
    created_ats = {r.created_at for r in matching}
    assert len(created_ats) == 2, (
        f"Expected distinct created_at values; got {created_ats}"
    )


def test_b_session_a_filter_returns_a_turn(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(deferred_dir, SESSION_A, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_A},
    ])
    _write_live_file(deferred_dir, SESSION_B, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_B},
    ])

    store = _open_store()
    drain_active_live_captures(store, exclude_session_id=SESSION_B)
    drain_active_live_captures(store, exclude_session_id=SESSION_A)

    turns_a = store.recent_user_turns(n=50, session_id=SESSION_A)
    assert len(turns_a) == 1, f"Expected 1 turn for session A; got {len(turns_a)}"
    assert turns_a[0].literal_surface == SHORT_TURN

    expected_ts = datetime.fromisoformat(TS_A)
    actual_ts = turns_a[0].created_at
    assert abs((actual_ts - expected_ts).total_seconds()) < 1, (
        f"Session A created_at should be ~{TS_A}; got {actual_ts}"
    )


def test_c_session_b_filter_returns_b_turn(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(deferred_dir, SESSION_A, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_A},
    ])
    _write_live_file(deferred_dir, SESSION_B, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_B},
    ])

    store = _open_store()
    drain_active_live_captures(store, exclude_session_id=SESSION_B)
    drain_active_live_captures(store, exclude_session_id=SESSION_A)

    turns_b = store.recent_user_turns(n=50, session_id=SESSION_B)
    assert len(turns_b) == 1, f"Expected 1 turn for session B; got {len(turns_b)}"
    assert turns_b[0].literal_surface == SHORT_TURN

    expected_ts = datetime.fromisoformat(TS_B)
    actual_ts = turns_b[0].created_at
    assert abs((actual_ts - expected_ts).total_seconds()) < 1, (
        f"Session B created_at should be ~{TS_B}; got {actual_ts}"
    )


def test_d_exact_key_idempotency_offset_rollback(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(deferred_dir, SESSION_A, [
        {"text": SHORT_TURN, "cue": "turn", "tier": "episodic",
         "role": "user", "ts": TS_A},
    ])

    store = _open_store()
    c1 = drain_active_live_captures(store, exclude_session_id=SESSION_B)
    assert c1["events_inserted"] == 1, f"First drain: {c1}"

    offset_p = _offset_path(iai_home, SESSION_A)
    if offset_p.exists():
        offset_p.write_text("0")

    c2 = drain_active_live_captures(store, exclude_session_id=SESSION_B)
    assert c2["events_inserted"] == 0, (
        f"Second drain after offset reset must not insert duplicates; got {c2}"
    )

    turns = store.recent_user_turns(n=50, session_id=SESSION_A)
    assert len(turns) == 1, (
        f"Expected exactly 1 turn for session A after re-drain; got {len(turns)}"
    )


def test_e_within_session_ts_ordering(iai_home):
    from iai_mcp.capture import drain_active_live_captures

    TURN1 = "first turn message here yes"
    TURN2 = "second turn message here yes"
    SESSION = "cccccccc-cccc-cccc-cccc-cccccccccccc"
    OTHER = "dddddddd-dddd-dddd-dddd-dddddddddddd"

    deferred_dir = iai_home / ".iai-mcp" / ".deferred-captures"
    _write_live_file(deferred_dir, SESSION, [
        {"text": TURN1, "cue": "first", "tier": "episodic",
         "role": "user", "ts": TS_TURN1},
        {"text": TURN2, "cue": "second", "tier": "episodic",
         "role": "user", "ts": TS_TURN2},
    ])

    store = _open_store()
    c = drain_active_live_captures(store, exclude_session_id=OTHER)
    assert c["events_inserted"] == 2, f"Expected 2 inserts; got {c}"

    turns = store.recent_user_turns(n=50, session_id=SESSION)
    assert len(turns) == 2, f"Expected 2 turns; got {len(turns)}"

    by_text = {t.literal_surface: t for t in turns}
    assert TURN1 in by_text and TURN2 in by_text, (
        f"Both turns must be present; got {list(by_text.keys())}"
    )

    ts1_expected = datetime.fromisoformat(TS_TURN1)
    ts2_expected = datetime.fromisoformat(TS_TURN2)
    ts1_actual = by_text[TURN1].created_at
    ts2_actual = by_text[TURN2].created_at
    assert abs((ts1_actual - ts1_expected).total_seconds()) < 1, (
        f"TURN1 created_at mismatch: expected ~{TS_TURN1}, got {ts1_actual}"
    )
    assert abs((ts2_actual - ts2_expected).total_seconds()) < 1, (
        f"TURN2 created_at mismatch: expected ~{TS_TURN2}, got {ts2_actual}"
    )
    assert ts1_actual < ts2_actual, (
        f"TURN1 must be earlier than TURN2; got {ts1_actual} vs {ts2_actual}"
    )


def _make_semantic_record(embedding: list[float]) -> "MemoryRecord":
    from iai_mcp.types import SCHEMA_VERSION_CURRENT, MemoryRecord
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="near identical semantic turn for dedup test",
        aaak_index="",
        embedding=embedding,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": now.isoformat(), "cue": "test",
                     "session_id": "sem-test", "role": "user"}],
        created_at=now,
        updated_at=now,
        tags=["capture", "role:user"],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )


def _make_unit_vector(dim: int, index: int = 0) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def test_f_semantic_gate2_still_merges(iai_home):
    from iai_mcp.store import flush_record_buffer

    store = _open_store()
    dim = store._embed_dim

    emb = _make_unit_vector(dim, index=0)
    rec1 = _make_semantic_record(emb)
    rec2 = _make_semantic_record(list(emb))

    store.insert(rec1)
    flush_record_buffer(store)
    id1 = rec1.id

    store.insert(rec2)
    flush_record_buffer(store)

    assert rec2.id == id1, (
        f"Gate #2: semantic near-dup must be SKIP-merged; "
        f"rec2.id={rec2.id} != rec1.id={id1}"
    )
    all_rec = store.all_records()
    semantic_rows = [r for r in all_rec if r.tier == "semantic"]
    assert len(semantic_rows) == 1, (
        f"Gate #2: expected 1 semantic row after dedup; got {len(semantic_rows)}"
    )


def test_f_semantic_gate1_still_reinforces(iai_home):
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import flush_record_buffer

    store = _open_store()
    SESSION = "ffffffff-ffff-ffff-ffff-ffffffffffff"

    SEMANTIC_TEXT = "this is a longer semantic memory entry for dedup regression test"
    assert len(SEMANTIC_TEXT) >= 12

    r1 = capture_turn(store, cue="test", text=SEMANTIC_TEXT,
                      tier="semantic", session_id=SESSION, role="user")
    assert r1["status"] == "inserted", f"First insert: {r1}"
    flush_record_buffer(store)

    r2 = capture_turn(store, cue="test", text=SEMANTIC_TEXT,
                      tier="semantic", session_id=SESSION, role="user")
    assert r2["status"] == "reinforced", (
        f"Gate #1: semantic duplicate must be reinforced; got {r2}"
    )


def test_g_assistant_distinct_turns_kept(iai_home):
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import flush_record_buffer

    store = _open_store()
    SESSION = "gggggggg-gggg-gggg-gggg-gggggggggggg"
    TURN_1 = "first assistant reply here yes good"
    TURN_2 = "second assistant reply here totally different"
    TS_1 = "2026-05-30T10:00:00.000000+00:00"
    TS_2 = "2026-05-30T10:00:10.000000+00:00"

    r1 = capture_turn(store, cue="test", text=TURN_1, tier="episodic",
                      session_id=SESSION, role="assistant", ts=TS_1)
    assert r1["status"] == "inserted", f"First assistant turn: {r1}"
    flush_record_buffer(store)

    r2 = capture_turn(store, cue="test", text=TURN_2, tier="episodic",
                      session_id=SESSION, role="assistant", ts=TS_2)
    assert r2["status"] == "inserted", (
        f"Second distinct assistant turn must be inserted; got {r2}"
    )
    flush_record_buffer(store)

    all_rec = store.all_records()
    assistant_rows = [
        r for r in all_rec
        if r.tier == "episodic" and "role:assistant" in (r.tags or [])
    ]
    assert len(assistant_rows) == 2, (
        f"Expected 2 assistant rows; got {len(assistant_rows)}: "
        f"{[r.literal_surface for r in assistant_rows]}"
    )

    r3 = capture_turn(store, cue="test", text=TURN_1, tier="episodic",
                      session_id=SESSION, role="assistant", ts=TS_1)
    assert r3["status"] == "reinforced", (
        f"Exact re-drain of TURN_1 must be reinforced; got {r3}"
    )

    all_rec_after = store.all_records()
    assistant_rows_after = [
        r for r in all_rec_after
        if r.tier == "episodic" and "role:assistant" in (r.tags or [])
    ]
    assert len(assistant_rows_after) == 2, (
        f"Expected still 2 assistant rows after re-drain; got {len(assistant_rows_after)}"
    )


def test_h_gate2_idem_skip_direct(iai_home):
    from iai_mcp.capture import _idem_tag
    from iai_mcp.store import flush_record_buffer
    from iai_mcp.types import SCHEMA_VERSION_CURRENT, MemoryRecord

    store = _open_store()
    dim = store._embed_dim
    SESSION = "hhhhhhhh-hhhh-hhhh-hhhh-hhhhhhhhhhhh"
    TEXT = "gate two idem skip direct test turn"
    TS = "2026-05-30T10:00:00.000000+00:00"
    ROLE = "user"

    now = datetime.fromisoformat(TS)
    ts_iso = now.isoformat()
    tag = _idem_tag(SESSION, ROLE, ts_iso, TEXT)

    emb = _make_unit_vector(dim, index=5)

    rec1 = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=TEXT,
        aaak_index="",
        embedding=emb,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": ts_iso, "cue": "test",
                     "session_id": SESSION, "role": ROLE}],
        created_at=now,
        updated_at=now,
        tags=["capture", f"role:{ROLE}", tag],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    store.insert(rec1)
    flush_record_buffer(store)
    id1 = rec1.id

    rec2 = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=TEXT,
        aaak_index="",
        embedding=list(emb),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": ts_iso, "cue": "test",
                     "session_id": SESSION, "role": ROLE}],
        created_at=now,
        updated_at=now,
        tags=["capture", f"role:{ROLE}", tag],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    original_rec2_id = rec2.id

    store.insert(rec2)
    flush_record_buffer(store)

    assert rec2.id == id1, (
        f"Gate #2 idem-skip: rec2.id must be mutated to existing id {id1}; "
        f"got {rec2.id} (original was {original_rec2_id})"
    )

    all_rec = store.all_records()
    idem_rows = [r for r in all_rec if tag in (r.tags or [])]
    assert len(idem_rows) == 1, (
        f"Gate #2 idem-skip: expected 1 row with idem tag; got {len(idem_rows)}"
    )


def test_ts_none_no_crash(iai_home):
    from iai_mcp.capture import capture_turn
    from iai_mcp.store import flush_record_buffer

    store = _open_store()
    SESSION = "00000000-0000-0000-0000-000000000001"
    TEXT = "ts none no crash test text here"

    result = capture_turn(store, cue="test", text=TEXT,
                          tier="episodic", session_id=SESSION,
                          role="user", ts=None)
    assert result["status"] == "inserted", (
        f"capture_turn(ts=None) must insert exactly one row; got {result}"
    )
    flush_record_buffer(store)

    turns = store.recent_user_turns(n=10, session_id=SESSION)
    assert len(turns) == 1, (
        f"Expected 1 row after ts=None capture; got {len(turns)}"
    )
    assert turns[0].literal_surface == TEXT
