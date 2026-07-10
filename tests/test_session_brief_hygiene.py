from __future__ import annotations

import sys
from datetime import datetime, timezone
from uuid import UUID, uuid4


from iai_mcp.community import CommunityAssignment
from iai_mcp.session import (
    SessionStartPayload,
    _clean_surface,
    _compose_session_start_payload,
    _session_state_hash,
    format_payload_as_markdown,
)
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _mk_record(
    store: MemoryStore,
    text: str,
    *,
    pinned: bool = False,
    detail_level: int = 2,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    community_id: UUID | None = None,
) -> MemoryRecord:
    ts = created_at or datetime.now(timezone.utc)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.0] * EMBED_DIM,
        community_id=community_id,
        centrality=0.5,
        detail_level=detail_level,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=ts,
        updated_at=ts,
        tags=list(tags) if tags else [],
        language="en",
    )
    store.insert(rec)
    return rec

def _one_community(
    store: MemoryStore,
    text: str = "community member record",
) -> tuple[UUID, MemoryRecord]:
    cid = uuid4()
    rec = _mk_record(store, text, community_id=cid)
    return cid, rec

def _assembly_standard(store: MemoryStore, assignment=None, rich_club=None):
    if assignment is None:
        assignment = CommunityAssignment()
    if rich_club is None:
        rich_club = []
    return _compose_session_start_payload(
        store,
        assignment,
        rich_club,
        session_id="test-session",
        profile_state={"wake_depth": "standard"},
    )

def test_clean_surface_plain_text_untouched():
    assert _clean_surface("plain text") == "plain text"

def test_clean_surface_strips_ansi_sgr():
    raw = "\x1b[1mbolded\x1b[0m and \x1b[32mgreen\x1b[0m"
    assert _clean_surface(raw) == "bolded and green"

def test_clean_surface_removes_well_formed_command_name():
    raw = "<command-name>/model</command-name> hi"
    assert _clean_surface(raw) == "hi"

def test_clean_surface_removes_unclosed_command_name():
    raw = "<command-name>/model</command-name\n  <command-mes"
    assert _clean_surface(raw) == ""

def test_clean_surface_removes_dangling_local_command_stdout():
    raw = "<local-command-stdout>Set model X"
    assert _clean_surface(raw) == ""

def test_clean_surface_removes_task_notification_block():
    raw = "<task-notification><task-id>1</task-id>Run tests</task-notification>"
    assert _clean_surface(raw) == ""

def test_clean_surface_collapses_internal_whitespace():
    raw = "  word1   word2\n\nword3  "
    assert _clean_surface(raw) == "word1 word2 word3"

def test_clean_surface_empty_string():
    assert _clean_surface("") == ""

def test_clean_surface_only_ansi_returns_empty():
    raw = "\x1b[0m\x1b[1m"
    assert _clean_surface(raw) == ""

def test_sc1_rendered_brief_is_junk_free(tmp_path):
    store = MemoryStore(path=tmp_path)

    junk = "<command-name>/model</command-name\n  <command-mes"
    rec = _mk_record(store, junk)

    payload = _assembly_standard(store)
    rendered = format_payload_as_markdown(payload)

    assert "\x1b[" not in rendered, "ANSI code found in rendered brief"
    assert "<command-name" not in rendered, "<command-name found in rendered brief"
    assert "<local-command-stdout" not in rendered, "<local-command-stdout found in rendered brief"
    assert "<task-notification" not in rendered, "<task-notification found in rendered brief"

    stored = store.get(rec.id)
    assert stored.literal_surface == junk, (
        f"literal_surface mutated; expected {junk!r}, got {stored.literal_surface!r}"
    )

def test_sc1_ansi_in_rich_club_cleaned(tmp_path):
    store = MemoryStore(path=tmp_path)
    ansi_text = "\x1b[33myellow note\x1b[0m"
    rec = _mk_record(store, ansi_text)
    rich_uuids = [rec.id]
    payload = _assembly_standard(store, rich_club=rich_uuids)
    rendered = format_payload_as_markdown(payload)
    assert "\x1b[" not in rendered
    assert "yellow note" in rendered

def test_skip_empty_cleans_no_blank_bullets(tmp_path):
    store = MemoryStore(path=tmp_path)
    _mk_record(
        store, "\x1b[0m\x1b[1m",
        pinned=True, detail_level=5,
    )
    _mk_record(store, "real content here", pinned=True, detail_level=5)

    payload = _assembly_standard(store)
    if payload.l1:
        for line in payload.l1.split("\n"):
            if line.startswith("- "):
                assert line[2:].strip() != "", f"blank bullet found: {line!r}"

def test_sc6_no_llmlingua_after_l2_render(tmp_path):
    for key in list(sys.modules):
        if key == "llmlingua" or key.startswith("llmlingua."):
            del sys.modules[key]

    store = MemoryStore(path=tmp_path)

    cid = uuid4()
    rec = _mk_record(store, "community member record alice", community_id=cid)
    assignment = CommunityAssignment()
    assignment.top_communities.append(cid)
    assignment.mid_regions[cid] = [rec.id]
    assignment.community_centroids[cid] = [0.0] * EMBED_DIM

    _assembly_standard(store, assignment=assignment)

    assert "llmlingua" not in sys.modules, (
        "llmlingua was imported during L2 render — compress call site was not removed"
    )

def test_sc2_recent_thread_section_present(tmp_path):
    from datetime import timedelta
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)

    _mk_record(store, "alice worked on task A earlier", created_at=now - timedelta(hours=2))
    _mk_record(store, "alice switched to task B", created_at=now - timedelta(hours=1))
    _mk_record(store, "alice is now on task C", created_at=now)

    payload = _assembly_standard(store)
    rendered = format_payload_as_markdown(payload)

    assert "recent" in rendered.lower() or "Most recent" in rendered, (
        f"no recent-thread section in rendered brief:\n{rendered}"
    )
    assert "task C" in rendered, f"newest record not in rendered brief:\n{rendered}"

def test_sc2_recent_thread_skips_junk(tmp_path):
    from datetime import timedelta
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)

    _mk_record(store, "\x1b[0m\x1b[1m", created_at=now)
    _mk_record(store, "alice clean work", created_at=now - timedelta(hours=1))

    payload = _assembly_standard(store)
    assert payload.recent_thread != ""
    for line in payload.recent_thread.split("\n"):
        if line.startswith("- "):
            assert line[2:].strip() != "", f"blank bullet in recent_thread: {line!r}"

def test_sc2_recent_thread_default_empty_on_minimal(tmp_path):
    store = MemoryStore(path=tmp_path)
    _mk_record(store, "some record")

    payload = _compose_session_start_payload(
        store, CommunityAssignment(), [],
        profile_state={"wake_depth": "minimal"},
    )
    assert payload.recent_thread == ""

def test_hash_invariant_to_recent_thread(tmp_path):
    p1 = SessionStartPayload(
        l0="ident block",
        l1="fact one\nfact two",
        l2=["community summary"],
        rich_club="hub node",
        recent_thread="most recent work thread A",
    )
    p2 = SessionStartPayload(
        l0="ident block",
        l1="fact one\nfact two",
        l2=["community summary"],
        rich_club="hub node",
        recent_thread="completely different recent thread B",
    )
    assert _session_state_hash(p1) == _session_state_hash(p2)

def test_format_payload_dict_branch_carries_recent_thread():
    payload_dict = {
        "l0": "identity info",
        "l1": "",
        "l2": [],
        "rich_club": "",
        "recent_thread": "- alice worked on plumbing\n- alice debugged the relay",
        "total_cached_tokens": 10,
        "total_dynamic_tokens": 1000,
        "wake_depth": "standard",
    }
    rendered = format_payload_as_markdown(payload_dict)
    assert "plumbing" in rendered, f"recent_thread content missing from dict render:\n{rendered}"
    assert "relay" in rendered, f"recent_thread content missing from dict render:\n{rendered}"

def test_sc6_populated_render_under_cap(tmp_path):
    from datetime import timedelta
    store = MemoryStore(path=tmp_path)
    now = datetime.now(timezone.utc)

    cid = uuid4()
    for i in range(10):
        _mk_record(
            store,
            f"alice record {i}: " + ("x" * 100),
            community_id=cid if i < 5 else None,
            created_at=now - timedelta(minutes=i),
        )

    assignment = CommunityAssignment()
    assignment.top_communities.append(cid)
    assignment.mid_regions[cid] = [rec.id for rec in store.all_records()[:3]]
    assignment.community_centroids[cid] = [0.0] * EMBED_DIM

    payload = _assembly_standard(store, assignment=assignment)
    rendered = format_payload_as_markdown(payload)
    assert len(rendered) <= 10000, f"rendered length {len(rendered)} exceeds 10000-char cap"
