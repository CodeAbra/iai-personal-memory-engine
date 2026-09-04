"""Directives always inject, including under the cheapest wake depth."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.community import CommunityAssignment
from iai_mcp.session import assemble_session_start, format_payload_as_markdown
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _directive_record(text: str) -> MemoryRecord:
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )
    rec.directive = True
    return rec


def test_directives_deliver_under_every_wake_depth(tmp_path):
    store = MemoryStore(path=tmp_path)
    rec = _directive_record("standing order that must survive the cheapest boot")
    store.insert(rec)

    for wake_depth in ("minimal", "standard", "deep"):
        payload = assemble_session_start(
            store, CommunityAssignment(), [],
            profile_state={"wake_depth": wake_depth},
        )
        assert payload.wake_depth == wake_depth
        assert payload.directives != "", f"directives empty under wake_depth={wake_depth}"
        assert "standing order that must survive the cheapest boot" in payload.directives

        rendered = format_payload_as_markdown(payload)
        assert "## Standing orders (always active)" in rendered, (
            f"standing orders block missing under wake_depth={wake_depth}"
        )
        assert "standing order that must survive the cheapest boot" in rendered


def test_minimal_wake_depth_still_skips_the_expensive_segments(tmp_path):
    """Directives inject unconditionally at minimal wake depth without
    reopening the existing minimal-mode savings on l0/l1/l2/rich_club."""
    store = MemoryStore(path=tmp_path)
    rec = _directive_record("stay terse")
    store.insert(rec)

    payload = assemble_session_start(
        store, CommunityAssignment(), [],
        profile_state={"wake_depth": "minimal"},
    )
    assert payload.l0 == ""
    assert payload.l1 == ""
    assert payload.l2 == []
    assert payload.rich_club == ""
    assert payload.directives != ""
