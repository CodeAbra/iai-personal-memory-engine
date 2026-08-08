from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.types import MemoryRecord

CYRILLIC = re.compile(r"[\u0400-\u04FF]")
HIRAGANA_KATAKANA = re.compile(r"[\u3040-\u30FF]")
CJK = re.compile(r"[\u4E00-\u9FFF]")


_TIER_TO_WING = {
    "working": "W",
    "episodic": "E",
    "semantic": "S",
    "procedural": "P",
    "parametric": "\u03a0",
}


def _wing_from_tier(tier: str) -> str:
    return _TIER_TO_WING.get(tier, "unknown")


def _room_from_community(record: "MemoryRecord") -> str:
    if record.community_id is None:
        return "unknown"
    return str(record.community_id)[:8]


def _entities_from_tags(tags: list[str]) -> str:
    ents = [t[len("entity:"):] for t in tags if t.startswith("entity:")][:16]
    if not ents:
        return "-"
    return ",".join(ents)


def _tagline(tags: list[str]) -> str:
    non_ents = [t for t in tags if not t.startswith("entity:")][:10]
    if not non_ents:
        return "-"
    return ",".join(non_ents)


def generate_aaak_index(record: "MemoryRecord") -> str:
    wing = _wing_from_tier(record.tier)
    room = _room_from_community(record)
    entities = _entities_from_tags(record.tags)
    tags = _tagline(record.tags)
    return f"W:{wing}/R:{room}/E:{entities}/T:{tags}"


def parse_aaak_index(idx: str) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {
        "wing": [],
        "room": [],
        "entities": [],
        "tags": [],
    }
    key_map = {"W": "wing", "R": "room", "E": "entities", "T": "tags"}
    for seg in idx.split("/"):
        if ":" not in seg:
            continue
        k, _, v = seg.partition(":")
        if k not in key_map:
            continue
        name = key_map[k]
        if v == "-" or v == "":
            out[name] = []
        else:
            if name in ("wing", "room"):
                out[name] = [v]
            else:
                out[name] = v.split(",")
    return out


def enforce_language_tagged(record: "MemoryRecord") -> None:
    if record.language and isinstance(record.language, str) and record.language.strip():
        return

    raise ValueError(
        "record.language is required and must be "
        "non-empty. Set record.language='en' (or the appropriate code on "
        "legacy rows) before calling this guard."
    )


def enforce_english_raw(record: "MemoryRecord") -> None:
    text = record.literal_surface or ""
    has_non_english = bool(
        CYRILLIC.search(text)
        or HIRAGANA_KATAKANA.search(text)
        or CJK.search(text)
    )
    if not has_non_english:
        return

    if any(t.startswith("raw:") for t in record.tags):
        return

    raise ValueError(
        "literal_surface contains non-English "
        "characters; storage must be English raw verbatim. "
        "Add 'raw:<lang>' tag to declare explicit raw capture."
    )
