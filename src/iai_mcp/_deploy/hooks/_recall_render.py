"""Stdlib-only renderer for the per-turn ``<iai-mcp-recall>`` hook block.

Daemon-independent by construction: no ``iai_mcp`` import. Consumes the
parsed ``memory_recall`` socket result (``hits``, optionally ``anti_hits``)
and renders the injected block, adding a dated staleness marker for any
superseded hit (``valid_to`` in the past) and a "supersedes prior version
dated ..." line when a corrector is available -- so a superseded record
can never silently outrank an unhedged stale claim already in context.

Kept inside the pre-existing 3-hit / [:400]-char budget: no session-start
or per-turn byte-count regression.
"""
from __future__ import annotations

from datetime import datetime, timezone

MAX_HITS = 3
MAX_CHARS = 400


def _parse_iso(value: object) -> "datetime | None":
    if not isinstance(value, str) or not value:
        return None
    text = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _date_only(value: object) -> "str | None":
    dt = _parse_iso(value)
    return dt.date().isoformat() if dt is not None else None


def _is_past(value: object, *, now: "datetime | None" = None) -> bool:
    dt = _parse_iso(value)
    if dt is None:
        return False
    if now is None:
        now = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return dt <= now


def render_hit_line(hit: dict, *, now: "datetime | None" = None) -> str:
    text = (hit.get("literal_surface") or hit.get("text") or "")[:MAX_CHARS]
    if not text:
        return ""
    valid_to = hit.get("valid_to")
    if valid_to and _is_past(valid_to, now=now):
        marker_date = _date_only(valid_to) or "unknown-date"
        return f"- {text} [superseded — valid until {marker_date}]"
    return f"- {text}"


def _corrector_line(anti_hits: list, *, now: "datetime | None" = None) -> "str | None":
    if not anti_hits:
        return None
    top = anti_hits[0]
    marker_date = _date_only(top.get("valid_to")) or _date_only(top.get("captured_at"))
    if not marker_date:
        return None
    return f"⚠ supersedes prior version dated {marker_date}"


def render_recall_block(result: dict, *, now: "datetime | None" = None) -> str:
    """Pure function: parsed socket result -> rendered block text (or "")."""
    hits = (result.get("hits") or [])[:MAX_HITS]
    lines = [ln for ln in (render_hit_line(h, now=now) for h in hits) if ln]
    if not lines:
        return ""
    body = ["<iai-mcp-recall>", *lines]
    corrector = _corrector_line(result.get("anti_hits") or [], now=now)
    if corrector:
        body.append(corrector)
    body.append("</iai-mcp-recall>")
    return "\n".join(body)
