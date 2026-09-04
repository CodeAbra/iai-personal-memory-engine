"""Freshness-verdict synthesis over a memory_recall response.

Pure -- no store/embed/foresight import, no I/O. Reads only the JSON fields
_hit_to_json already emits (score, valid_to, captured_at, community_id).
"""

from __future__ import annotations

from datetime import datetime, timezone

CONTRADICTED = "CONTRADICTED"
CURRENT = "CURRENT"
UNCONFIRMED = "UNCONFIRMED"
NO_EVIDENCE = "NO_EVIDENCE"

# A note untouched while >= 1 week of newer same-topic evidence accrued is
# the staleness signal.
CLAIM_CHECK_STALE_DAYS = 7


def _parse_ts(value: object) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def synthesize_verdict(recall_response: dict, *, now: datetime) -> dict:
    hits = recall_response.get("hits") or []
    anti_hits = recall_response.get("anti_hits") or []

    if anti_hits:
        return {
            "tier": CONTRADICTED,
            "reason": "a contradicts-edge neighbour was found",
        }

    for h in hits:
        if _parse_ts(h.get("valid_to")) is not None:
            return {
                "tier": CONTRADICTED,
                "reason": "a hit carries a superseding valid_to",
            }

    if not hits:
        return {"tier": NO_EVIDENCE, "reason": "no matching memory found"}

    # Anchor by score -- unconditionally populated by _hit_to_json.
    # Staleness pairs on captured_at -- reliably populated on the recall
    # path (backfilled where None), unlike community_id which is None for
    # cache-served hits not community-gated on this call; community_id is
    # therefore a negative filter only, never a required pairing key.
    anchor = max(hits, key=lambda h: float(h.get("score") or 0.0))
    anchor_captured = _parse_ts(anchor.get("captured_at"))

    # anchor.valid_to is guaranteed None here -- any hit with a parseable
    # valid_to already returned CONTRADICTED above.
    if anchor_captured is not None:
        anchor_community = anchor.get("community_id")
        for h in hits:
            if h is anchor:
                continue
            h_captured = _parse_ts(h.get("captured_at"))
            if h_captured is None:
                continue
            age_days = (h_captured - anchor_captured).total_seconds() / 86400.0
            if age_days < CLAIM_CHECK_STALE_DAYS:
                continue
            h_community = h.get("community_id")
            if (
                anchor_community is not None
                and h_community is not None
                and h_community != anchor_community
            ):
                continue
            return {
                "tier": UNCONFIRMED,
                "reason": "newer related evidence found, not formally linked",
            }

    # Accepted false negative: if the newer evidence outscores the stale
    # note, the anchor itself is recent and nothing postdates it by the
    # stale window -- reads as CURRENT, a miss under precision-over-recall,
    # not a bug.
    return {"tier": CURRENT, "reason": "no contradiction or staleness signal found"}
