"""Regression: `tier="procedural"` records must never leak into rendered
recent-thread text -- neither the session-start payload's `recent_thread`
segment nor the per-turn `render_session_delta`.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity
from iai_mcp.session import (
    _compose_session_start_payload,
    _recent_thread_segment,
    format_payload_as_markdown,
    render_session_delta,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

PROC_SENTINEL = "PROC-SENTINEL-2f9c1a-must-never-render"
CONTROL_SENTINEL = "CONTROL-SENTINEL-8b31de-recent-work-entry"


def _mk_record(
    tier: str,
    text: str,
    *,
    salience_level: str = "unflagged",
    created_at: "datetime | None" = None,
):
    now = created_at or datetime.now(timezone.utc)
    rid = uuid4()
    rec = MemoryRecord(
        id=rid,
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.15] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"session_id": "other-session"}],
        created_at=now,
        updated_at=now,
        tags=["role:user"],
        language="en",
        salience_level=salience_level,
    )
    return rec, rid


def _assignment_with_members(member_ids: list) -> CommunityAssignment:
    cid = uuid4()
    return CommunityAssignment(
        node_to_community={m: cid for m in member_ids},
        community_centroids={cid: [0.1] * EMBED_DIM},
        modularity=0.5,
        backend="leiden-networkx",
        top_communities=[cid],
        mid_regions={cid: member_ids},
    )


def _seeded_store(tmp_path, monkeypatch, subdir: str):
    root = tmp_path / subdir
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    (root / "config.json").write_text(
        json.dumps({"identity": {"name": "alice", "languages": "en", "role": "developer"}})
    )
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=root / "store")
    _seed_l0_identity(store)

    control_rec, control_id = _mk_record("semantic", f"{CONTROL_SENTINEL}: normal recent work.")
    store.insert(control_rec)
    proc_rec, proc_id = _mk_record("procedural", f"{PROC_SENTINEL}: chunk internals.")
    store.insert(proc_rec)
    return store, control_id, proc_id


def test_procedural_chunk_absent_from_rendered_session_start_recent_thread(tmp_path, monkeypatch):
    store, control_id, _ = _seeded_store(tmp_path, monkeypatch, "start")
    assignment = _assignment_with_members([control_id])

    from iai_mcp.profile import default_state
    profile_state = {**default_state(), "wake_depth": "standard"}

    payload = _compose_session_start_payload(
        store, assignment, [], session_id="uat-proc-leak", profile_state=profile_state,
    )

    assert CONTROL_SENTINEL in payload.recent_thread, (
        "control (legit semantic recent-work entry) missing from payload.recent_thread "
        "-- test setup is not exercising the recent-thread window"
    )
    assert PROC_SENTINEL not in payload.recent_thread, (
        "procedural chunk leaked into payload.recent_thread"
    )

    served = format_payload_as_markdown(payload)
    assert CONTROL_SENTINEL in served, (
        "control missing from served session-start markdown -- over-filtered"
    )
    assert PROC_SENTINEL not in served, (
        "procedural chunk leaked into served session-start markdown"
    )


def test_procedural_chunk_absent_from_render_session_delta(tmp_path, monkeypatch):
    store, _, _ = _seeded_store(tmp_path, monkeypatch, "delta")

    delta = render_session_delta(
        store, "1970-01-01T00:00:00+00:00", session_id="different-session",
    )

    assert CONTROL_SENTINEL in delta, (
        "control missing from render_session_delta -- test setup is not exercising "
        "the delta window"
    )
    assert PROC_SENTINEL not in delta, (
        "procedural chunk leaked into render_session_delta"
    )


def _isolated_store(tmp_path, monkeypatch, subdir: str) -> MemoryStore:
    root = tmp_path / subdir
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    (root / "config.json").write_text(
        json.dumps({"identity": {"name": "alice", "languages": "en", "role": "developer"}})
    )
    store = MemoryStore(path=root / "store")
    _seed_l0_identity(store)
    return store


def test_recent_thread_salience_composite_outranks_pure_recency(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch, "salience")
    now = datetime.now(timezone.utc)

    older_critical, _ = _mk_record(
        "semantic",
        "OLDER-CRITICAL-SENTINEL: flagged turn.",
        salience_level="critical",
        created_at=now - timedelta(minutes=10),
    )
    store.insert(older_critical)
    newer_unflagged, _ = _mk_record(
        "semantic",
        "NEWER-UNFLAGGED-SENTINEL: plain turn.",
        created_at=now,
    )
    store.insert(newer_unflagged)

    segment = _recent_thread_segment(store, max_records=5)
    assert "OLDER-CRITICAL-SENTINEL" in segment
    assert "NEWER-UNFLAGGED-SENTINEL" in segment
    assert segment.index("OLDER-CRITICAL-SENTINEL") < segment.index("NEWER-UNFLAGGED-SENTINEL"), (
        "a flagged-but-slightly-older turn must outrank a newer unflagged one"
    )


def test_recent_thread_pending_turn_scored_without_salience_slot(tmp_path, monkeypatch):
    store = _isolated_store(tmp_path, monkeypatch, "pending")
    now = datetime.now(timezone.utc)

    control_rec, _ = _mk_record(
        "semantic", f"{CONTROL_SENTINEL}: normal recent work.", created_at=now,
    )
    store.insert(control_rec)

    # _PendingTurn (built internally from this event dict) has no
    # salience_level slot -- the composite scorer must getattr-default it
    # to "unflagged" rather than raise AttributeError.
    segment = _recent_thread_segment(
        store,
        max_records=5,
        pending_live_events=[{
            "role": "user",
            "session_id": "live-session",
            "source_uuid": None,
            "ts_iso": now.isoformat(),
            "ts": now,
            "text": "PENDING-SENTINEL: live turn not yet indexed.",
        }],
    )
    assert "PENDING-SENTINEL" in segment
    assert CONTROL_SENTINEL in segment
