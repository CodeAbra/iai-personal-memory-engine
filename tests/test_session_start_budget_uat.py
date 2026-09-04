"""End-to-end witness for the session-start token budget on the standard
wake branch: a real composed payload, plus the served markdown artifact,
must both stay within the 3000-token ceiling while carrying a non-empty
L0/L1/L2 floor.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity
from iai_mcp.session import _approx_tokens, _compose_session_start_payload, format_payload_as_markdown
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _seed_alice_pinned(store: MemoryStore, n: int = 8) -> list[UUID]:
    now = datetime.now(timezone.utc)
    ids: list[UUID] = []
    for i in range(n):
        rid = uuid4()
        rec = MemoryRecord(
            id=rid,
            tier="semantic",
            literal_surface=f"alice pinned fact {i}: standard-mode session context.",
            aaak_index="",
            embedding=[0.1] * EMBED_DIM,
            community_id=None,
            centrality=0.5,
            detail_level=5,
            pinned=True,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=True,
            never_merge=False,
            provenance=[],
            created_at=now,
            updated_at=now,
            tags=[],
            language="en",
        )
        store.insert(rec)
        ids.append(rid)
    return ids


def _seed_proc_chunk(store: MemoryStore) -> UUID:
    now = datetime.now(timezone.utc)
    rid = uuid4()
    rec = MemoryRecord(
        id=rid,
        tier="procedural",
        literal_surface="procedural chunk should never surface as session-start text",
        aaak_index="",
        embedding=[0.2] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=["chunk", "source:cofire"],
        language="en",
    )
    store.insert(rec)
    return rid


def _seed_pinned_semantic(store: MemoryStore) -> UUID:
    now = datetime.now(timezone.utc)
    rid = uuid4()
    rec = MemoryRecord(
        id=rid,
        tier="semantic",
        literal_surface="sensitivity mutant pinned fact: this record must move the token total.",
        aaak_index="",
        embedding=[0.3] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=5,
        pinned=True,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )
    store.insert(rec)
    return rid


def _assignment_with_members(member_ids: list[UUID]) -> CommunityAssignment:
    cid = uuid4()
    return CommunityAssignment(
        node_to_community={m: cid for m in member_ids},
        community_centroids={cid: [0.1] * EMBED_DIM},
        modularity=0.5,
        backend="leiden-networkx",
        top_communities=[cid],
        mid_regions={cid: member_ids},
    )


def test_standard_payload_and_served_markdown_within_budget(tmp_path, monkeypatch):
    # Isolate the L0 identity seed's config.json read to this scratch store —
    # without this, _seed_l0_identity reads the operator's real ~/.iai-mcp
    # config.json for identity data.
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"identity": {"name": "alice", "languages": "en", "role": "developer"}})
    )

    # Isolate the ambient/pending-events source: _compose_session_start_payload's
    # standard branch calls iai_mcp.capture.read_pending_live_events(), which
    # resolves Path.home()/".iai-mcp" regardless of IAI_MCP_STORE. The import
    # is function-local inside session.py, so the monkeypatch target must be
    # the source module attribute -- patching iai_mcp.session is a no-op.
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=tmp_path / "store")
    _seed_l0_identity(store)
    seeded_ids = _seed_alice_pinned(store, n=8)

    assignment = _assignment_with_members(seeded_ids[:3])
    rich_club = seeded_ids[3:6]

    from iai_mcp.profile import default_state
    profile_state = {**default_state(), "wake_depth": "standard"}

    payload = _compose_session_start_payload(
        store, assignment, rich_club, session_id="uat-token-budget",
        profile_state=profile_state,
    )

    # Proves the standard branch was actually taken -- the minimal branch
    # yields an empty payload and would make every assertion below vacuous.
    assert payload.wake_depth == "standard"
    assert payload.l0, "L0 identity segment must be non-empty on the standard branch"
    assert payload.l1, "L1 pinned-fact segment must be non-empty on the standard branch"
    assert len(payload.l2) >= 1, "at least one L2 community segment must be composed"
    assert payload.rich_club, "rich-club segment must be composed from the seeded ids"

    observed = payload.total_cached_tokens
    assert 0 < observed <= 3000, f"total_cached_tokens={observed} out of budget"
    # Floor tied to the seed (observed ~227 on this fixture): a silently-
    # emptied segment (e.g. l2 or rich_club collapsing to "") would drop the
    # sum well below this, catching the regression instead of passing
    # vacuously on a near-zero total.
    floor = 100
    assert observed >= floor, (
        f"total_cached_tokens={observed} below floor={floor}; a segment "
        f"likely collapsed to empty"
    )

    served = format_payload_as_markdown(payload)
    served_tokens = _approx_tokens(served)
    assert 0 < served_tokens <= 3000, f"served_tokens={served_tokens} out of budget"
    assert served_tokens >= observed, (
        f"served markdown ({served_tokens} tok) should be >= the composed "
        f"sub-sum ({observed} tok) once scaffolding headers are added"
    )


def _isolated_standard_payload(tmp_path, monkeypatch, subdir: str, seed_extra=None):
    root = tmp_path / subdir
    root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(root))
    (root / "config.json").write_text(
        json.dumps({"identity": {"name": "alice", "languages": "en", "role": "developer"}})
    )
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=root / "store")
    _seed_l0_identity(store)
    seeded_ids = _seed_alice_pinned(store, n=8)
    if seed_extra is not None:
        seed_extra(store)

    assignment = _assignment_with_members(seeded_ids[:3])
    rich_club = seeded_ids[3:6]

    from iai_mcp.profile import default_state
    profile_state = {**default_state(), "wake_depth": "standard"}

    return _compose_session_start_payload(
        store, assignment, rich_club, session_id="uat-token-budget-proc",
        profile_state=profile_state,
    )


def test_procedural_chunk_never_grows_session_start_budget(tmp_path, monkeypatch):
    """Pins three true, independently-verified facts about a tier="procedural"
    chunk (pinned=False, detail_level=1): none of the four static segments,
    nor the recent-thread segment, ever surface it.

    TRUE #1 -- total_cached_tokens is structurally blind to the chunk: it is
    summed from l0/l1/l2/rich_club ONLY (session.py computes `cached` before
    _recent_thread_segment is even called), so token-total equality here is
    not a rendering coincidence. The sensitivity mutant below (a pinned,
    detail_level=5 record) proves the metric is not blind to ALL inserts --
    it moves when a record actually qualifies for l1.

    TRUE #2 -- the chunk cannot enter l0, l1 (requires pinned and
    detail_level>=4), l2, or rich_club: those segments draw membership only
    from the assignment/rich_club id lists or the pinned-hi-detail query,
    none of which include the chunk. Breaking mutant: a segment-gate change
    that admits the chunk into one of these four fields.

    TRUE #3 -- the chunk cannot enter recent_thread either:
    _recent_thread_segment drops tier=="procedural" candidates before
    rendering. Breaking mutant: removing that filter re-admits the chunk.
    """
    baseline_payload = _isolated_standard_payload(tmp_path, monkeypatch, "baseline")
    baseline_tokens = baseline_payload.total_cached_tokens

    with_chunk_payload = _isolated_standard_payload(
        tmp_path, monkeypatch, "with_chunk", seed_extra=_seed_proc_chunk,
    )
    assert with_chunk_payload.total_cached_tokens == baseline_tokens, (
        f"procedural chunk changed total_cached_tokens: baseline={baseline_tokens}, "
        f"with_chunk={with_chunk_payload.total_cached_tokens}"
    )

    # Compared as line-sets, not exact strings: l1/rich_club draw from a
    # pinned-hi-detail query whose row order is not stable across two
    # independently-seeded stores with fresh random UUIDs -- only content
    # membership is the claim under test, not row order.
    chunk_surface = "procedural chunk should never surface as session-start text"
    for field_name in ("l0", "l1", "rich_club"):
        with_field = getattr(with_chunk_payload, field_name)
        base_field = getattr(baseline_payload, field_name)
        assert chunk_surface not in with_field, (
            f"procedural chunk leaked into payload.{field_name}"
        )
        assert set(with_field.splitlines()) == set(base_field.splitlines()), (
            f"payload.{field_name} content must be identical (as a line set) "
            f"with vs. without the procedural chunk"
        )
    assert all(chunk_surface not in s for s in with_chunk_payload.l2), (
        "procedural chunk leaked into payload.l2"
    )
    # The "[community <uuid>]" prefix carries a random id minted fresh per
    # CommunityAssignment (test fixture artifact, not store content) --
    # strip it before comparing so the check is on served content only.
    def _l2_body(segment: str) -> str:
        return segment.split("] ", 1)[-1]

    assert len(with_chunk_payload.l2) == len(baseline_payload.l2), (
        "payload.l2 community count must be identical with vs. without the "
        "procedural chunk"
    )
    assert {_l2_body(s) for s in with_chunk_payload.l2} == {
        _l2_body(s) for s in baseline_payload.l2
    }, "payload.l2 content must be identical with vs. without the procedural chunk"

    assert chunk_surface not in with_chunk_payload.recent_thread, (
        "procedural chunk leaked into payload.recent_thread"
    )

    # Sensitivity mutant: a pinned/detail_level=5 record (L1-eligible) DOES
    # change the token total -- proves the metric is not blind to inserts,
    # so TRUE #1's equality above is not a vacuous always-equal check.
    sensitivity_payload = _isolated_standard_payload(
        tmp_path, monkeypatch, "sensitivity", seed_extra=_seed_pinned_semantic,
    )
    assert sensitivity_payload.total_cached_tokens != baseline_tokens, (
        f"sensitivity mutant failed to move total_cached_tokens: "
        f"baseline={baseline_tokens}, sensitivity={sensitivity_payload.total_cached_tokens}"
    )
