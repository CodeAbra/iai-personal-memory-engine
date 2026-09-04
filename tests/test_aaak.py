from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.aaak import (
    enforce_english_raw,
    generate_aaak_index,
    parse_aaak_index,
)
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _make(
    tier: str = "episodic",
    text: str = "hello world",
    tags: list[str] | None = None,
    community_id: UUID | None = None,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=community_id,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=list(tags) if tags else [],
        language=language,
    )


def test_aaak_index_has_exactly_three_slashes():
    r = _make()
    idx = generate_aaak_index(r)
    assert idx.count("/") == 3


def test_aaak_index_starts_with_wing_marker():
    r = _make(tier="semantic")
    idx = generate_aaak_index(r)
    assert idx.startswith("W:S/")


def test_aaak_index_has_four_key_value_segments():
    r = _make(tier="episodic", tags=["entity:alice", "project", "raw:en"])
    idx = generate_aaak_index(r)
    parts = idx.split("/")
    assert len(parts) == 4
    assert parts[0].startswith("W:")
    assert parts[1].startswith("R:")
    assert parts[2].startswith("E:")
    assert parts[3].startswith("T:")


def test_aaak_index_includes_entity_tag_stripped():
    r = _make(tags=["entity:alice", "entity:IAI-MCP", "project"])
    idx = generate_aaak_index(r)
    assert "alice" in idx.split("/E:")[1]
    assert "IAI-MCP" in idx.split("/E:")[1]


def test_aaak_index_deterministic():
    r = _make(tags=["entity:X", "flag"])
    assert generate_aaak_index(r) == generate_aaak_index(r)


def test_aaak_index_does_not_contain_literal_surface():
    verbatim = "alice mentioned the SECRET_PASSWORD_ABC_XYZ on day 3"
    r = _make(text=verbatim, tags=["entity:alice", "project"])
    idx = generate_aaak_index(r)
    assert verbatim not in idx
    assert "SECRET_PASSWORD_ABC_XYZ" not in idx


def test_aaak_index_unknown_community_marker():
    r = _make(community_id=None)
    idx = generate_aaak_index(r)
    assert "R:unknown" in idx


def test_aaak_index_dash_when_no_entities():
    r = _make(tags=["project"])
    idx = generate_aaak_index(r)
    assert "/E:-/" in idx


def test_parse_aaak_index_round_trips_entities_and_tags():
    r = _make(tier="semantic", tags=["entity:alice", "entity:IAI", "project", "urgent"])
    idx = generate_aaak_index(r)
    parsed = parse_aaak_index(idx)
    assert parsed["wing"] == ["S"]
    assert parsed["entities"] == ["alice", "IAI"]
    assert set(parsed["tags"]) == {"project", "urgent"}


def test_parse_aaak_dash_segments_become_empty_lists():
    r = _make(tags=[])
    idx = generate_aaak_index(r)
    parsed = parse_aaak_index(idx)
    assert parsed["entities"] == []
    assert parsed["tags"] == []


def test_enforce_english_raw_accepts_pure_english():
    r = _make(text="alice said the IAI-MCP project is go")
    enforce_english_raw(r)


def test_enforce_english_raw_rejects_cyrillic_without_tag():
    r = _make(text="alice сказал: пусть сохранится точно", tags=["project"])  # non-English fixture data: ru round-trip under test
    with pytest.raises(ValueError) as exc:
        enforce_english_raw(r)
    assert "English raw verbatim" in str(exc.value)


def test_enforce_english_raw_accepts_cyrillic_with_raw_tag():
    r = _make(
        text="alice сказал: пусть сохранится точно",  # non-English fixture data: ru round-trip under test
        tags=["raw:ru", "project"],
    )
    enforce_english_raw(r)


def test_enforce_english_raw_rejects_cjk_without_tag():
    r = _make(text="Hello 世界 verbatim", tags=[])
    with pytest.raises(ValueError):
        enforce_english_raw(r)


def test_enforce_english_raw_rejects_hiragana_without_tag():
    r = _make(text="Hello こんにちは world", tags=[])
    with pytest.raises(ValueError):
        enforce_english_raw(r)


def test_enforce_english_raw_accepts_cjk_with_raw_tag():
    r = _make(text="Hello 世界", tags=["raw:zh"])
    enforce_english_raw(r)


def test_enforce_english_raw_empty_text_passes():
    r = _make(text="")
    enforce_english_raw(r)
