"""cls_summary rich_club render cap: render-only strip-then-cap.

Covers the trigger states for the RICH_CLUB_CLS_SUMMARY_CAP cap: a
minted-format cls_summary line (matches the boilerplate prefix pattern)
strips the prefix then truncates at the cap, a cls_summary line whose
surface does NOT match the full prefix pattern falls back to the legacy
60-char cap, and any non-cls_summary line is unaffected either way.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from iai_mcp.session import RICH_CLUB_CLS_SUMMARY_CAP, _clean_surface, _rich_club_segment_with_budget
from iai_mcp.types import CLS_SUMMARY_PREFIX_MARKER, CLS_SUMMARY_PREFIX_RE, EMBED_DIM, MemoryRecord


def _seed(store, literal_surface: str, tags: list[str]):
    now = datetime.now(timezone.utc) - timedelta(days=1)
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.5,
        detail_level=3,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=True,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=tags,
        language="en",
    )
    store.insert(rec)
    return rec


def _line_content(line: str) -> str:
    _, content = line.split(": ", 1)
    return content


def test_matched_cls_summary_strips_then_caps(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literal_surface = (
        "Cluster summary (3 records, lang=en): "
        + "alice cluster content joined from several member records " + "z" * 40
    )
    rec = _seed(store, literal_surface, ["cls_summary"])
    segment = _rich_club_segment_with_budget(store, [rec.id], budget=5000)
    line = segment.splitlines()[0]

    prefix_match = CLS_SUMMARY_PREFIX_RE.match(rec.literal_surface)
    assert prefix_match is not None
    expected = _clean_surface(rec.literal_surface[prefix_match.end():])[:RICH_CLUB_CLS_SUMMARY_CAP]

    assert _line_content(line) == expected
    assert len(_line_content(line)) == RICH_CLUB_CLS_SUMMARY_CAP
    cleaned_pre = _clean_surface(rec.literal_surface)
    assert len(_line_content(line)) < len(cleaned_pre[:60])
    assert _line_content(line) != cleaned_pre[:60]


def test_matched_cls_summary_hyphenated_lang_strips_then_caps(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literal_surface = (
        "Cluster summary (3 records, lang=pt-BR): "
        + "alice cluster content joined from several member records " + "z" * 40
    )
    rec = _seed(store, literal_surface, ["cls_summary"])

    prefix_match = CLS_SUMMARY_PREFIX_RE.match(rec.literal_surface)
    assert prefix_match is not None

    segment = _rich_club_segment_with_budget(store, [rec.id], budget=5000)
    line = segment.splitlines()[0]
    expected = _clean_surface(rec.literal_surface[prefix_match.end():])[:RICH_CLUB_CLS_SUMMARY_CAP]

    assert _line_content(line) == expected
    assert len(_line_content(line)) == RICH_CLUB_CLS_SUMMARY_CAP


def test_unmatched_cls_summary_falls_back_to_legacy_cap(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literal_surface = (
        f"{CLS_SUMMARY_PREFIX_MARKER}odd non-matching shape): "
        + "alice cluster content that does not fit the full pattern " + "z" * 40
    )
    rec = _seed(store, literal_surface, ["cls_summary"])
    assert CLS_SUMMARY_PREFIX_RE.match(literal_surface) is None

    segment = _rich_club_segment_with_budget(store, [rec.id], budget=5000)
    line = segment.splitlines()[0]
    expected = _clean_surface(rec.literal_surface)[:60]

    assert _line_content(line) == expected
    assert len(_line_content(line)) == 60


def test_non_cls_summary_line_stays_at_legacy_cap(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    rec = _seed(
        store,
        "alice ordinary episodic content untouched by the cap " + "z" * 40,
        ["semantic"],
    )
    segment = _rich_club_segment_with_budget(store, [rec.id], budget=5000)
    line = segment.splitlines()[0]
    expected = _clean_surface(rec.literal_surface)[:60]
    assert _line_content(line) == expected
    assert len(_line_content(line)) == 60


def test_real_content_per_line_non_decreasing(tmp_path):
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    literal_surface = (
        "Cluster summary (3 records, lang=en): "
        + "alice cluster content joined from several member records " + "z" * 40
    )
    rec = _seed(store, literal_surface, ["cls_summary"])
    segment = _rich_club_segment_with_budget(store, [rec.id], budget=5000)
    line = segment.splitlines()[0]

    prefix_len = len(CLS_SUMMARY_PREFIX_RE.match(rec.literal_surface).group(0))
    pre_real_chars = len(_clean_surface(rec.literal_surface)[:60]) - prefix_len
    post_real_chars = len(_line_content(line))

    assert post_real_chars >= pre_real_chars
