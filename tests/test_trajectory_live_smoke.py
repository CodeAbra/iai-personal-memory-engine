from __future__ import annotations

from uuid import uuid4

import pytest

from iai_mcp import profile, retrieve
from iai_mcp.events import write_event
from iai_mcp.session import assemble_session_start
from iai_mcp.store import MemoryStore
from iai_mcp.trajectory import (
    M2_SYNTHETIC_CONSTANT,
    M4_SYNTHETIC_CONSTANT,
    M6_SYNTHETIC_CONSTANT,
    compute_m1_clarifying_questions_per_session,
    compute_m3_token_budget,
    compute_m5_curiosity_frequency,
    compute_session_metrics_snapshot,
    m2_precision_at_5_live,
    m4_profile_variance_live,
    m6_context_repeat_rate_live,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _make_record(literal: str) -> MemoryRecord:
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
        language="en",
    )

def test_m2_live_differs_from_synthetic_when_retrievals_happen(tmp_path):
    store = MemoryStore(path=tmp_path)
    store.insert(_make_record("a"))
    store.insert(_make_record("b"))
    retrieve.recall(
        store=store,
        cue_embedding=[0.5] * EMBED_DIM,
        cue_text="a",
        session_id="smoke",
    )
    live = m2_precision_at_5_live(store)
    assert abs(live - M2_SYNTHETIC_CONSTANT) > 0.001, (
        f"M2 live ({live}) must differ from synthetic ({M2_SYNTHETIC_CONSTANT})"
    )

def test_m4_live_differs_from_synthetic_when_profile_writes_happen(tmp_path):
    store = MemoryStore(path=tmp_path)
    state = profile.default_state()
    profile.profile_set("interest_boost", 0.2, state, store=store)
    profile.profile_set("interest_boost", 0.8, state, store=store)
    live = m4_profile_variance_live(store)
    assert abs(live - M4_SYNTHETIC_CONSTANT) > 0.001, (
        f"M4 live ({live}) must differ from synthetic ({M4_SYNTHETIC_CONSTANT})"
    )

def test_m6_live_differs_from_synthetic_when_session_starts_repeat(tmp_path):
    store = MemoryStore(path=tmp_path)
    store.insert(_make_record("seed"))
    _g, assignment, rc = retrieve.build_runtime_graph(store)
    assemble_session_start(store, assignment, rc, session_id="s1")
    assemble_session_start(store, assignment, rc, session_id="s2")
    live = m6_context_repeat_rate_live(store)
    assert abs(live - M6_SYNTHETIC_CONSTANT) > 0.001, (
        f"M6 live ({live}) must differ from synthetic ({M6_SYNTHETIC_CONSTANT})"
    )

def test_core_trajectory_metrics_remain_live(tmp_path):
    store = MemoryStore(path=tmp_path)
    sid = "smoke"
    write_event(
        store, kind="curiosity_question",
        data={"text": "?"}, severity="info", session_id=sid,
    )
    write_event(
        store, kind="session_started",
        data={"total_cached_tokens": 2500}, severity="info", session_id=sid,
    )
    write_event(
        store, kind="curiosity_silent_log",
        data={"text": "..."}, severity="info", session_id=sid,
    )

    assert compute_m1_clarifying_questions_per_session(store, sid) == 1.0
    assert compute_m3_token_budget(store, sid) == pytest.approx(2500.0, abs=1e-6)
    assert compute_m5_curiosity_frequency(store, sid) == 2.0

def test_compute_session_metrics_snapshot_returns_live_values_for_m2_m4_m6(tmp_path):
    store = MemoryStore(path=tmp_path)
    state = profile.default_state()

    store.insert(_make_record("hello"))
    retrieve.recall(
        store=store, cue_embedding=[0.5] * EMBED_DIM,
        cue_text="hello", session_id="s",
    )
    profile.profile_set("interest_boost", 0.4, state, store=store)
    profile.profile_set("interest_boost", 0.6, state, store=store)
    _g, assignment, rc = retrieve.build_runtime_graph(store)
    assemble_session_start(store, assignment, rc, session_id="x")
    assemble_session_start(store, assignment, rc, session_id="y")

    snap = compute_session_metrics_snapshot(store, "s")
    assert snap["m2"] > 0.0, snap
    assert snap["m4"] > 0.0, snap
    assert snap["m6"] > 0.0, snap
