from __future__ import annotations

from typing import TYPE_CHECKING

from iai_mcp.lilli.tiers.bsc import (
    bundle as _bundle,
    filler_hv as _filler_hv,
    role_hv as _role_hv,
    unpack_role as _unpack_role,
)
from iai_mcp.types import STRUCTURE_HV_DIM

if TYPE_CHECKING:
    from iai_mcp.types import MemoryRecord

_D: int = STRUCTURE_HV_DIM


def role_hv(role: str) -> bytes:
    return _role_hv(role, D=_D)


def filler_hv(value: str) -> bytes:
    return _filler_hv(value, D=_D)


def pack_pairs(pairs: list[tuple[str, bytes]]) -> bytes:
    return _bundle(pairs, D=_D)


def unpack_role(hv: bytes, role: str) -> bytes:
    return _unpack_role(hv, role, D=_D)


def _bucket_datetime(dt) -> str:
    try:
        return dt.date().isoformat()
    except (AttributeError, TypeError, ValueError):
        return "unknown"


def bind_structure(record: "MemoryRecord") -> bytes:
    pairs: list[tuple[str, bytes]] = []

    pairs.append(("WHEN", filler_hv(_bucket_datetime(record.created_at))))
    pairs.append(("WHERE", filler_hv(record.tier)))
    pairs.append(("ROLE", filler_hv(record.tier)))
    pairs.append(("PROJECT", filler_hv("iai-mcp")))
    pairs.append(("COMMUNITY_ID", filler_hv(str(record.community_id) if record.community_id else "none")))
    pairs.append(("TEMPORAL_POSITION", filler_hv(_bucket_datetime(record.created_at))))

    pairs.append(("LANG", filler_hv(record.language or "en")))
    pairs.append(("TIER", filler_hv(record.tier)))
    pairs.append(("MODALITY", filler_hv("text")))
    pairs.append(("INTENT", filler_hv("episodic" if record.tier == "episodic" else "semantic")))
    pairs.append(("ACTOR", filler_hv("user")))
    pairs.append(("OBJECT", filler_hv(str(record.id))))
    pairs.append(("VALENCE", filler_hv("neutral")))
    pairs.append(("CERTAINTY", filler_hv(f"trust_{round(record.s5_trust_score, 1)}")))
    pairs.append(("SOURCE", filler_hv("pinned" if record.pinned else "drift")))

    leading_tag = (record.tags[0] if record.tags else "untagged")
    pairs.append(("TOPIC", filler_hv(str(leading_tag))))

    sid = "no-session"
    if record.provenance:
        try:
            sid = str(record.provenance[-1].get("session_id") or "no-session")
        except (AttributeError, TypeError, KeyError, IndexError):
            sid = "no-session"
    pairs.append(("SESSION_ID", filler_hv(sid)))
    pairs.append(("PARENT_ID", filler_hv("root")))

    return pack_pairs(pairs)
