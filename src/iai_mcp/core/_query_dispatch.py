"""Read-only schema/events store queries."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from iai_mcp.store import MemoryStore

logger = logging.getLogger(__name__)


EVENTS_QUERY_WHITELIST: frozenset[str] = frozenset({
    "s4_contradiction",
    "trajectory_metric",
    "schema_induction_run",
    "llm_health",
    "curiosity_silent_log",
    "curiosity_question",
    "cls_consolidation_run",
    "crypto_key_rotated",
    "session_started",
    "recall_source",
    "embed_construct",
    "rem_cycle_completed",
    "hippo_compacted",
    "recall_timing",
    "first_turn_recall",
    "profile_tuned",
    "retrieval_weight_tuned",
})

# events_query strips free-text insight synthesis from the wire; only
# pass/fail/skip metadata leaves.
_REM_CYCLE_REDACTED_FIELDS: frozenset[str] = frozenset({"main_insight_text"})


def _schema_list_dispatch(store: MemoryStore, params: dict) -> dict:
    import pandas as pd

    confidence_min = float(params.get("confidence_min", 0.0) or 0.0)
    domain_filter = params.get("domain")

    records = store.all_records()
    schema_records = [r for r in records if "schema" in (r.tags or [])]

    edges_df = store.db.open_table("edges").to_pandas()
    if not edges_df.empty:
        schema_edges = edges_df[edges_df["edge_type"] == "schema_instance_of"]
    else:
        schema_edges = pd.DataFrame(columns=["src", "dst", "weight"])

    out: list[dict] = []
    for rec in schema_records:
        pattern = ""
        status = "auto"
        for t in (rec.tags or []):
            if t.startswith("pattern:"):
                pattern = t.split(":", 1)[1]
            elif t in ("auto", "pending_user_approval"):
                status = t
        if not pattern and rec.literal_surface.startswith("Schema: "):
            rest = rec.literal_surface[len("Schema: "):]
            pattern = rest.split(" (confidence=")[0]

        confidence = 0.0
        if "(confidence=" in rec.literal_surface:
            try:
                seg = rec.literal_surface.rsplit("(confidence=", 1)[1]
                num = seg.split(")")[0]
                confidence = float(num)
            except (ValueError, IndexError):
                confidence = 0.0

        if domain_filter is not None:
            domain_tag = f"domain:{domain_filter}"
            if domain_tag not in (rec.tags or []):
                continue

        if confidence < confidence_min:
            continue

        sid = str(rec.id)
        if len(schema_edges) > 0:
            evidence = schema_edges[schema_edges["dst"] == sid]
            evidence_count = int(len(evidence))
            exceptions_count = int(
                len(evidence[evidence["weight"] < 0])
            ) if "weight" in evidence.columns else 0
        else:
            evidence_count = 0
            exceptions_count = 0

        out.append({
            "id": str(rec.id),
            "pattern": pattern,
            "confidence": float(confidence),
            "evidence_count": evidence_count,
            "exceptions_count": exceptions_count,
            "status": status,
            "language": rec.language,
        })

    return {"schemas": out, "total": len(out)}


def _events_query_dispatch(store: MemoryStore, params: dict) -> dict:
    from iai_mcp.events import query_events

    kind = params.get("kind")
    if not kind:
        return {"error": "kind parameter is required"}
    if kind not in EVENTS_QUERY_WHITELIST:
        return {
            "error": (
                f"kind {kind!r} is not user-visible; "
                f"allowed: {sorted(EVENTS_QUERY_WHITELIST)}"
            )
        }

    severity = params.get("severity")
    since_raw = params.get("since")
    since_dt = None
    if since_raw:
        try:
            since_dt = datetime.fromisoformat(str(since_raw).replace("Z", "+00:00"))
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return {"error": f"since must be ISO-8601, got {since_raw!r}"}

    limit = int(params.get("limit", 100) or 100)
    limit = max(1, min(1000, limit))

    events = query_events(
        store,
        kind=kind,
        since=since_dt,
        severity=severity,
        limit=limit,
    )
    out_events: list[dict] = []
    for e in events:
        ts = e["ts"]
        if hasattr(ts, "isoformat"):
            try:
                ts_str = ts.isoformat()
            except (ValueError, TypeError, AttributeError) as exc:
                logger.debug("ts_isoformat_failed: %s", exc)
                ts_str = str(ts)
        else:
            ts_str = str(ts)
        data = e["data"]
        if kind == "rem_cycle_completed" and isinstance(data, dict):
            data = {
                k: v for k, v in data.items()
                if k not in _REM_CYCLE_REDACTED_FIELDS
            }
        out_events.append({
            "id": str(e["id"]),
            "kind": e["kind"],
            "severity": e.get("severity"),
            "domain": e.get("domain"),
            "ts": ts_str,
            "data": data,
            "session_id": e.get("session_id"),
            "source_ids": e.get("source_ids", []),
        })
    return {"events": out_events, "count": len(out_events)}
