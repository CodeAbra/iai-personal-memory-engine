from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity, dispatch
from iai_mcp.session import (
    L0_RECORD_UUID,
    L2_COMMUNITY_CAP,
    RICH_CLUB_BUDGET_TOKENS,
    TOTAL_CACHED_BUDGET,
    SessionStartPayload,
    _approx_tokens,
    assemble_session_start,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

_STANDARD = {"wake_depth": "standard"}

def _l0_record(store: MemoryStore) -> None:
    _seed_l0_identity(store)

def _pinned_record(
    store: MemoryStore,
    text: str,
    community_id: UUID | None = None,
    tags: list[str] | None = None,
) -> MemoryRecord:
    r = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.5,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language="en",
    )
    store.insert(r)
    return r

def test_empty_store_graceful(tmp_path):
    store = MemoryStore(path=tmp_path)
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert payload.l0 == ""
    assert payload.l1 == ""
    assert payload.l2 == []
    assert payload.rich_club == ""
    assert payload.total_cached_tokens == 0
    assert payload.total_dynamic_tokens > 0

def test_l0_renders_identity(tmp_path, monkeypatch):
    import json

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(
        {"identity": {"name": "alice", "languages": "en", "role": "developer"}}))
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert "alice" in payload.l0

def test_l0_uses_fixed_uuid(tmp_path):
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    assert store.get(L0_RECORD_UUID) is not None
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert payload.l0 != ""

def test_l0_segment_excludes_literal_only_at_cap(tmp_path):
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    payload = assemble_session_start(
        store, CommunityAssignment(), [], profile_state=_STANDARD,
    )
    assert "W:" in payload.l0

def test_total_cached_budget_respected(tmp_path):
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    for i in range(10):
        _pinned_record(store, f"Pinned fact #{i}: short verbatim content here.")
    payload = assemble_session_start(store, CommunityAssignment(), [])
    assert payload.total_cached_tokens <= TOTAL_CACHED_BUDGET

def test_l1_caps_at_max_records(tmp_path):
    store = MemoryStore(path=tmp_path)
    _l0_record(store)
    for i in range(20):
        _pinned_record(store, f"Pinned fact #{i}")
    payload = assemble_session_start(store, CommunityAssignment(), [])
    l1_lines = payload.l1.split("\n") if payload.l1 else []
    assert len(l1_lines) <= 10

def test_l2_capped_at_seven(tmp_path):
    store = MemoryStore(path=tmp_path)
    assignment = CommunityAssignment()
    for i in range(10):
        cid = uuid4()
        rec = _pinned_record(
            store, f"member of community {i}", community_id=cid
        )
        assignment.top_communities.append(cid)
        assignment.mid_regions[cid] = [rec.id]
        assignment.community_centroids[cid] = [0.0] * EMBED_DIM
    payload = assemble_session_start(store, assignment, [])
    assert len(payload.l2) <= L2_COMMUNITY_CAP
    assert L2_COMMUNITY_CAP == 7

def test_rich_club_truncation_under_budget(tmp_path):
    store = MemoryStore(path=tmp_path)
    rich_uuids: list[UUID] = []
    for i in range(50):
        r = _pinned_record(store, f"rich-club entry {i}: " + ("x" * 280))
        rich_uuids.append(r.id)
    payload = assemble_session_start(store, CommunityAssignment(), rich_uuids)
    assert _approx_tokens(payload.rich_club) <= RICH_CLUB_BUDGET_TOKENS

def test_session_start_payload_dispatch_empty(tmp_path):
    store = MemoryStore(path=tmp_path)
    result = dispatch(store, "session_start_payload", {})
    for key in (
        "l0",
        "l1",
        "l2",
        "rich_club",
        "total_cached_tokens",
        "total_dynamic_tokens",
        "breakpoint_marker",
    ):
        assert key in result
    assert result["l0"] == ""
    assert result["total_cached_tokens"] == 0

def test_session_start_payload_dispatch_with_l0(tmp_path, monkeypatch):
    import json
    import iai_mcp.core as core

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(json.dumps(
        {"identity": {"name": "alice", "languages": "en", "role": "developer"}}))
    original = core._profile_state.get("wake_depth", "minimal")
    core._profile_state["wake_depth"] = "standard"
    try:
        store = MemoryStore(path=tmp_path)
        _seed_l0_identity(store)
        result = dispatch(store, "session_start_payload", {})
        assert "alice" in result["l0"]
        assert result["breakpoint_marker"] == "--<cache-breakpoint>--"
    finally:
        core._profile_state["wake_depth"] = original


def test_session_start_payload_minimal_skips_runtime_graph(tmp_path, monkeypatch):
    import iai_mcp.core as core
    import iai_mcp.retrieve as retrieve

    original = core._profile_state.get("wake_depth", "minimal")
    core._profile_state["wake_depth"] = "minimal"
    monkeypatch.setattr(
        retrieve,
        "build_runtime_graph",
        lambda _store: (_ for _ in ()).throw(AssertionError("runtime graph built")),
    )
    try:
        store = MemoryStore(path=tmp_path)
        _pinned_record(store, "keeps the store non-empty")
        result = dispatch(store, "session_start_payload", {})
    finally:
        core._profile_state["wake_depth"] = original

    assert result["wake_depth"] == "minimal"
    assert result["rich_club"] == ""

def test_payload_type_is_session_start_payload(tmp_path):
    store = MemoryStore(path=tmp_path)
    payload = assemble_session_start(store, CommunityAssignment(), [])
    assert isinstance(payload, SessionStartPayload)
    assert payload.breakpoint_marker == "--<cache-breakpoint>--"

@pytest.fixture
def iai_home_assembly(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
    yield tmp_path

def test_recent_thread_includes_live_events_standard(iai_home_assembly):
    store = MemoryStore(path=iai_home_assembly)

    session = "assembly-live-session-60"
    live_text = "pending live turn for session assembly standard test content"

    from iai_mcp.capture import write_deferred_event
    write_deferred_event(session, "user", live_text)

    payload = assemble_session_start(
        store, CommunityAssignment(), [],
        session_id=session,
        profile_state={"wake_depth": "standard"},
    )
    assert live_text in (payload.recent_thread or ""), (
        f"standard payload recent_thread must include pending live turn; "
        f"got: {payload.recent_thread!r}"
    )

def test_recent_thread_skips_live_scan_on_minimal(iai_home_assembly, monkeypatch):
    store = MemoryStore(path=iai_home_assembly)

    call_count = [0]

    import iai_mcp.capture as _cap_mod
    original_fn = _cap_mod.read_pending_live_events

    def spy_fn(*args, **kwargs):
        call_count[0] += 1
        return original_fn(*args, **kwargs)

    monkeypatch.setattr(_cap_mod, "read_pending_live_events", spy_fn)

    payload = assemble_session_start(
        store, CommunityAssignment(), [],
        profile_state={"wake_depth": "minimal"},
    )
    assert call_count[0] == 0, (
        f"read_pending_live_events must NOT be called on minimal wake_depth; "
        f"called {call_count[0]} times"
    )
