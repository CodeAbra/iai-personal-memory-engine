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


def _pinned_hi_detail_record(text: str) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=text,
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )


def test_minimal_wake_depth_still_skips_the_expensive_segments(tmp_path):
    """Minimal wake depth renders a non-empty identity floor and a small
    bounded critical-facts sample, while still skipping the expensive
    topic-community and rich-memory segments the deeper depths add."""
    store = MemoryStore(path=tmp_path)

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    rec = _directive_record("stay terse")
    store.insert(rec)
    store.insert(_pinned_hi_detail_record("Pinned fact: high-detail context."))

    payload = assemble_session_start(
        store, CommunityAssignment(), [],
        profile_state={"wake_depth": "minimal"},
    )
    assert payload.l0 != "", "minimal wake depth must still render an identity floor"
    assert payload.l1 != "", "minimal wake depth must still render a bounded critical-facts floor"
    assert payload.l2 == []
    assert payload.rich_club == ""
    assert payload.directives != ""


def test_minimal_floor_caps_at_the_named_constant(tmp_path):
    """The minimal-mode critical-facts floor never grows past its bound,
    even when far more pinned high-detail records exist than the cap."""
    from iai_mcp.session import MINIMAL_FLOOR_MAX_RECORDS

    store = MemoryStore(path=tmp_path)

    from iai_mcp.core import _seed_l0_identity
    _seed_l0_identity(store)

    for i in range(MINIMAL_FLOOR_MAX_RECORDS + 2):
        store.insert(_pinned_hi_detail_record(f"Pinned fact {i}: high-detail context."))

    payload = assemble_session_start(
        store, CommunityAssignment(), [],
        profile_state={"wake_depth": "minimal"},
    )
    rendered_lines = [line for line in payload.l1.splitlines() if line.strip()]
    assert len(rendered_lines) == MINIMAL_FLOOR_MAX_RECORDS
