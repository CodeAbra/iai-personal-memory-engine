from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(text: str, language: str = "", tags: list[str] | None = None) -> MemoryRecord:
    actual_lang = language if language else "XX"
    r = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
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
        language=actual_lang,
    )
    if not language:
        r.language = ""
    return r


def test_enforce_language_tagged_accepts_english_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("hello world", language="en")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_russian_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("привет мир", language="ru")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_japanese_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("こんにちは", language="ja")
    enforce_language_tagged(r)


def test_enforce_language_tagged_accepts_arabic_with_tag():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("مرحبا بالعالم", language="ar")
    enforce_language_tagged(r)


def test_enforce_language_tagged_rejects_missing_language():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("some text", language="")
    with pytest.raises(ValueError) as exc:
        enforce_language_tagged(r)
    assert "record.language" in str(exc.value)


def test_enforce_language_tagged_caller_sets_default_explicitly():
    from iai_mcp.aaak import enforce_language_tagged

    r = _rec("hello world", language="")
    r.language = "en"
    enforce_language_tagged(r)
    assert r.language == "en"


def test_enforce_english_raw_still_importable():
    from iai_mcp.aaak import enforce_english_raw

    assert callable(enforce_english_raw)


def test_enforce_english_raw_with_language_tag_keeps_raw_semantics():
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="ru")
    with pytest.raises(ValueError):
        enforce_english_raw(r)


def test_enforce_english_raw_still_blocks_untagged_cyrillic():
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="")
    with pytest.raises(ValueError) as exc:
        enforce_english_raw(r)
    assert "english raw verbatim" in str(exc.value).lower()


def test_enforce_english_raw_accepts_cyrillic_with_raw_tag():
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("привет мир", language="", tags=["raw:ru"])
    enforce_english_raw(r)


def test_enforce_english_raw_accepts_pure_english():
    from iai_mcp.aaak import enforce_english_raw

    r = _rec("hello world", language="")
    enforce_english_raw(r)
