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
