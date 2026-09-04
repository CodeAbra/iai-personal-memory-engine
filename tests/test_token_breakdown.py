"""Report-shape unit test for bench/token_breakdown.py on a small synthetic
seeded store -- deterministic, no real-store access. Modeled on
tests/test_session_start_budget_uat.py's seed pattern."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID, uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.core import _seed_l0_identity
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

from bench.token_breakdown import compose_and_measure


def _seed_alice_pinned(store: MemoryStore, n: int = 8) -> list[UUID]:
    now = datetime.now(timezone.utc)
    ids: list[UUID] = []
    for i in range(n):
        rid = uuid4()
        rec = MemoryRecord(
            id=rid,
            tier="semantic",
            literal_surface=f"alice pinned fact {i}: standard-mode session context for token breakdown testing.",
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


def test_report_shape_on_synthetic_store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"identity": {"name": "alice", "languages": "en", "role": "developer"}})
    )
    # _compose_session_start_payload's standard branch calls
    # iai_mcp.capture.read_pending_live_events(), which resolves
    # Path.home()/".iai-mcp" regardless of IAI_MCP_STORE.
    monkeypatch.setattr("iai_mcp.capture.read_pending_live_events", lambda *a, **k: [])

    store = MemoryStore(path=tmp_path / "store")
    _seed_l0_identity(store)
    seeded_ids = _seed_alice_pinned(store, n=8)

    assignment = _assignment_with_members(seeded_ids[:3])
    rich_club = seeded_ids[3:6]

    report = compose_and_measure(store, assignment, rich_club, session_id="test-token-breakdown")

    assert report["mode"] == "B"
    assert report["wake_depth"] == "standard"

    for unit_key in ("chars", "chars4_tokens", "tiktoken_tokens"):
        assert unit_key in report["total"]
        assert isinstance(report["total"][unit_key], int)

    assert isinstance(report["truncated"], bool)
    assert isinstance(report["rich_club_saturation"]["saturated"], bool)

    for name in ("l0", "l1", "l2", "recent_thread", "rich_club"):
        assert name in report["components"], f"missing component: {name}"
        comp = report["components"][name]
        for unit_key in ("chars", "chars4_tokens", "tiktoken_tokens"):
            assert unit_key in comp
            assert isinstance(comp[unit_key], int)
        assert "pct_chars" in comp
        assert "pct_tiktoken_tokens" in comp

    # rich-club membership was seeded non-empty -> the index-vs-content
    # sub-split must be present with both units on both sides.
    assert report["rich_club_split"] is not None
    for side in ("index_age", "content"):
        assert side in report["rich_club_split"]
        for unit_key in ("chars", "chars4_tokens", "tiktoken_tokens"):
            assert unit_key in report["rich_club_split"][side]

    # chars is exact arithmetic (no rounding): the raw-field sum plus the
    # diffed scaffolding must equal the authoritative rendered total exactly.
    sum_component_chars = sum(
        report["components"][name]["chars"]
        for name in ("l0", "l1", "l2", "recent_thread", "rich_club")
    )
    assert sum_component_chars + report["scaffolding"]["chars"] == report["total"]["chars"]

    # token units round per-field; allow a small tolerance for the rounding
    # drift instead of demanding byte-exact additivity across units.
    sum_component_chars4 = sum(
        report["components"][name]["chars4_tokens"]
        for name in ("l0", "l1", "l2", "recent_thread", "rich_club")
    )
    assert abs(
        (sum_component_chars4 + report["scaffolding"]["chars4_tokens"]) - report["total"]["chars4_tokens"]
    ) <= 10

    assert report["component_set"], "component_set must be non-empty on the standard branch"
