"""Response DTO serializers (hit/payload -> JSON dict)."""

from __future__ import annotations


def _hit_to_json(h) -> dict:
    _vf = getattr(h, "valid_from", None)
    _vt = getattr(h, "valid_to", None)
    return {
        "record_id": str(h.record_id),
        "score": float(h.score),
        "reason": h.reason,
        "literal_surface": h.literal_surface,
        "adjacent_suggestions": [str(x) for x in h.adjacent_suggestions],
        "valid_from": _vf.isoformat() if _vf is not None else None,
        "valid_to": _vt.isoformat() if _vt is not None else None,
        "session_id": getattr(h, "session_id", None),
        "captured_at": getattr(h, "captured_at", None),
    }


def _payload_to_json(payload) -> dict:
    return {
        "l0": payload.l0,
        "l1": payload.l1,
        "l2": list(payload.l2),
        "rich_club": payload.rich_club,
        "total_cached_tokens": int(payload.total_cached_tokens),
        "total_dynamic_tokens": int(payload.total_dynamic_tokens),
        "breakpoint_marker": payload.breakpoint_marker,
        "identity_pointer": getattr(payload, "identity_pointer", ""),
        "brain_handle": getattr(payload, "brain_handle", ""),
        "topic_cluster_hint": getattr(payload, "topic_cluster_hint", ""),
        "compact_handle": getattr(payload, "compact_handle", ""),
        "wake_depth": getattr(payload, "wake_depth", "minimal"),
    }
