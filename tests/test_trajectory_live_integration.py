from __future__ import annotations

from uuid import uuid4

import pytest

from iai_mcp import profile, retrieve
from iai_mcp.events import flush_event_buffer, query_events
from iai_mcp.store import MemoryStore
from iai_mcp.trajectory import (
    m2_precision_at_5_live,
    m4_profile_movement_live,
    m6_context_repeat_rate_live,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _make_record(literal: str, *, lang: str = "en") -> MemoryRecord:
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=[0.5] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
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
        language=lang,
    )

def test_real_recall_emits_retrieval_used_and_m2_lifts_off_zero(tmp_path):
    store = MemoryStore(path=tmp_path)
    for i in range(3):
        store.insert(_make_record(f"hello world {i}"))

    cue_emb = [0.5] * EMBED_DIM
    resp = retrieve.recall(
        store=store,
        cue_embedding=cue_emb,
        cue_text="hello",
        session_id="integration-1",
    )
    assert len(resp.hits) > 0

    flush_event_buffer(store)

    events = query_events(store, kind="retrieval_used", limit=20)
    assert events, (
        "FALSE-GREEN GUARD: retrieve.recall must emit kind='retrieval_used' "
        "in production for M2 to be live; no events found means M2 always "
        "returns 0.0 in real use."
    )

    m2_val = m2_precision_at_5_live(store)
    assert m2_val > 0.0, (
        f"M2 must return >0 when retrieval_used events exist; got {m2_val}"
    )

def test_real_profile_tuned_event_lifts_m4_off_zero(tmp_path):
    from iai_mcp.events import write_event

    store = MemoryStore(path=tmp_path)
    write_event(store, kind="profile_tuned", data={"moved_count": 3}, severity="info")

    events = query_events(store, kind="profile_tuned", limit=20)
    assert events, "profile_tuned events must be readable for M4 to be live."
    m4_val = m4_profile_movement_live(store)
    assert m4_val > 0.0, f"M4 must return >0 with a non-zero moved_count; got {m4_val}"

def test_profile_set_no_op_does_not_emit(tmp_path):
    store = MemoryStore(path=tmp_path)
    state = profile.default_state()
    profile.profile_set("interest_boost", 0.5, state, store=store)
    before = len(query_events(store, kind="profile_updated", limit=100))
    profile.profile_set("interest_boost", 0.5, state, store=store)
    after = len(query_events(store, kind="profile_updated", limit=100))
    assert after == before, "no-op profile_set must not emit"

def test_real_session_start_emits_session_started_and_m6_lifts_off_zero(tmp_path):
    from iai_mcp.session import assemble_session_start

    store = MemoryStore(path=tmp_path)
    store.insert(_make_record("seed"))

    _graph, assignment, rc = retrieve.build_runtime_graph(store)
    assemble_session_start(store, assignment, rc, session_id="sess-A")
    assemble_session_start(store, assignment, rc, session_id="sess-B")

    events = query_events(store, kind="session_started", limit=20)
    assert len(events) >= 2, (
        "FALSE-GREEN GUARD: assemble_session_start must emit "
        "kind='session_started' for M6 to be live."
    )
    m6_val = m6_context_repeat_rate_live(store)
    assert m6_val == pytest.approx(0.5, abs=1e-6), (
        f"two identical session starts must give M6 = 0.5; got {m6_val}"
    )
