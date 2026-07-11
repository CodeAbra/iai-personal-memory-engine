"""Scope (constitutional):
- ALLOWED: L2 community descriptors, session summaries, cls_summary records.
- FORBIDDEN: literal_surface of normal records, pinned, invariant_anchor,
  user-tagged 'raw' records.
- Passthrough when llmlingua package not installed (local-only stays green).
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.events import query_events
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(
    *,
    text: str = "lorem ipsum dolor sit amet consectetur adipiscing elit",
    tags: list[str] | None = None,
    pinned: bool = False,
    detail_level: int = 2,
    s5_trust_score: float = 0.5,
    language: str = "en",
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[1.0] + [0.0] * (EMBED_DIM - 1),
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
        s5_trust_score=s5_trust_score,
    )


def test_is_compressible_rejects_pinned():
    from iai_mcp.compress import is_compressible

    r = _rec(pinned=True)
    ok, reason = is_compressible(r)
    assert ok is False
    assert "pinned" in reason.lower()


def test_is_compressible_rejects_raw_tagged():
    from iai_mcp.compress import is_compressible

    r = _rec(tags=["raw:ru", "project:iai-mcp"])
    ok, reason = is_compressible(r)
    assert ok is False
    assert "raw" in reason.lower()


def test_is_compressible_rejects_invariant_anchor():
    from iai_mcp.compress import is_compressible

    r = _rec(s5_trust_score=0.95)
    ok, reason = is_compressible(r)
    assert ok is False
    assert "invariant" in reason.lower() or "trust" in reason.lower()


def test_is_compressible_allows_cls_summary():
    from iai_mcp.compress import is_compressible

    r = _rec(tags=["semantic", "cls_summary"])
    ok, _reason = is_compressible(r)
    assert ok is True


def test_is_compressible_allows_schema():
    from iai_mcp.compress import is_compressible

    r = _rec(tags=["schema", "auto"])
    ok, _reason = is_compressible(r)
    assert ok is True


def test_is_compressible_rejects_normal_record_by_default():
    """"""
    from iai_mcp.compress import is_compressible

    r = _rec(tags=["project:iai-mcp"])
    ok, reason = is_compressible(r)
    assert ok is False
    assert "literal_surface" in reason.lower() or "constitutional" in reason.lower()


def test_compress_llmlingua2_passes_through_when_pkg_absent(tmp_path, monkeypatch):
    """On ImportError, fall back to passthrough + log llm_health event."""
    from iai_mcp import compress as compress_mod

    # Force the import path to fail.
    monkeypatch.setattr(compress_mod, "_load_llmlingua2", lambda: None)

    store = MemoryStore(path=tmp_path)
    text = "this is a long text that would normally be compressed"
    out = compress_mod.compress_llmlingua2(text, target_ratio=0.5, store=store)
    assert out == text  # passthrough


def test_compress_llmlingua2_logs_fallback_event(tmp_path, monkeypatch):
    from iai_mcp import compress as compress_mod

    monkeypatch.setattr(compress_mod, "_load_llmlingua2", lambda: None)

    store = MemoryStore(path=tmp_path)
    compress_mod.compress_llmlingua2("text", target_ratio=0.5, store=store)
    events = query_events(store, kind="llm_health")
    fallback_events = [e for e in events if e["data"].get("component") == "compress_llmlingua2"]
    assert len(fallback_events) >= 1


def test_compress_l2_descriptor_uses_l2_target_ratio():
    from iai_mcp.compress import COMPRESSION_TARGET_L2, compress_l2_descriptor

    # Passthrough when pkg absent -- just check the function is callable.
    out = compress_l2_descriptor("community summary line")
    assert isinstance(out, str)
    assert COMPRESSION_TARGET_L2 == 0.5


def test_compress_summary_uses_summary_target_ratio():
    from iai_mcp.compress import COMPRESSION_TARGET_SUMMARY, compress_summary

    out = compress_summary("cluster summary line")
    assert isinstance(out, str)
    assert COMPRESSION_TARGET_SUMMARY == 0.3


def test_compress_module_constants():
    from iai_mcp.compress import COMPRESSION_TARGET_L2, COMPRESSION_TARGET_SUMMARY

    assert COMPRESSION_TARGET_L2 == 0.5
    assert COMPRESSION_TARGET_SUMMARY == 0.3
