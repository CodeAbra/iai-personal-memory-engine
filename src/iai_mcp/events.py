"""Events table: schema_reinforced, trajectory_metric, identity_audit_error, etc."""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

import pandas as pd

from iai_mcp.crypto import (
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.store import EVENTS_TABLE, MemoryStore

TELEMETRY_RANK_DEFICIENCY: str = "rank_deficiency_warning"
TELEMETRY_ROLE_SATURATION: str = "role_saturation_warning"
TELEMETRY_CODEC_MARKER_MISSING: str = "codec_marker_missing"
TELEMETRY_EMBED_NATIVE_FAILURE: str = "embed_native_failure"
TELEMETRY_FRESHNESS_FUSE_TRIPPED: str = "freshness_fuse_tripped"
TELEMETRY_RECALL_SOURCE: str = "recall_source"
TELEMETRY_EMBED_CONSTRUCT: str = "embed_construct"
TELEMETRY_MEMORY_RELIEF: str = "memory_relief"
TELEMETRY_DRAIN_RSS_SOFT_CAP: str = "drain_rss_soft_cap"

DAEMON_WEDGE_KILL: str = "daemon_wedge_kill"
DAEMON_MEMORY_PRESSURE_KILL: str = "daemon_memory_pressure_kill"
DAEMON_WATCHDOG_NEEDS_OPERATOR: str = "daemon_watchdog_needs_operator"
DAEMON_SLEEP_CYCLE_STALE: str = "daemon_sleep_cycle_stale"
CRISIS_MODE_AUTO_EXPIRED: str = "crisis_mode_auto_expired"

TELEMETRY_RGC_WORKER_SUCCESS: str = "rgc_worker_success"
TELEMETRY_RGC_WORKER_JIT_WARMUP: str = "rgc_worker_jit_warmup"
TELEMETRY_RGC_WORKER_TIMEOUT: str = "rgc_worker_timeout"
TELEMETRY_RGC_WORKER_CRASH: str = "rgc_worker_crash"
# Emitted from save() when the encoded snapshot exceeds 80% of MAX_CACHE_BYTES;
# rate-limited to one event per generation window.
TELEMETRY_RGC_SNAPSHOT_NEAR_LIMIT: str = "rgc_snapshot_near_limit"
TELEMETRY_EMBED_NONFINITE: str = "embed_nonfinite_rejected"

_event_buffer: dict[int, list[dict]] = {}

_last_flush_at: dict[int, datetime] = {}

_BUFFER_LOCK = threading.RLock()


def flush_event_buffer(store: MemoryStore) -> int:
    with _BUFFER_LOCK:
        store_id = id(store)
        pending = _event_buffer.pop(store_id, [])
        if not pending:
            return 0
        try:
            store.db.open_table(EVENTS_TABLE).add(pending)
            _last_flush_at[store_id] = datetime.now(timezone.utc)
        except (OSError, RuntimeError, ValueError) as exc:
            logging.getLogger(__name__).warning("flush_event_buffer_failed", extra={"n": len(pending), "err": str(exc)[:120]})
        return len(pending)


def should_flush(store_id: int, max_size: int | None = None) -> bool:
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_EVENT_BUFFER_MAX", "100"))
        except ValueError:
            max_size = 100
    return len(_event_buffer.get(store_id, [])) >= max_size


def should_flush_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    if not _event_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec


def write_event(
    store: MemoryStore,
    kind: str,
    data: dict[str, Any],
    *,
    severity: str | None = None,
    domain: str | None = None,
    session_id: str = "-",
    source_ids: list[UUID] | None = None,
    buffered: bool = False,
) -> UUID:
    event_id = uuid4()
    data_plain = json.dumps(data)
    ad = str(event_id).encode("ascii")
    data_ct = encrypt_field(data_plain, store._key(), associated_data=ad)
    row = {
        "id": str(event_id),
        "kind": kind,
        "severity": severity or "",
        "domain": domain or "",
        "ts": datetime.now(timezone.utc),
        "data_json": data_ct,
        "session_id": session_id,
        "source_ids_json": json.dumps([str(x) for x in (source_ids or [])]),
    }
    if buffered:
        _event_buffer.setdefault(id(store), []).append(row)
        return event_id
    store.db.open_table(EVENTS_TABLE).add([row])

    try:
        from iai_mcp.daemon_config import _load_stc_config
        from iai_mcp.peri_event_buffer import get_buffer

        cfg = _load_stc_config()
        if kind in cfg.strong_event_types:
            buf = get_buffer()
            if buf is not None:
                buf.trigger_stc(store, kind)
    except Exception as exc:  # noqa: BLE001 -- STC trigger must never crash write_event
        logging.getLogger(__name__).warning(
            "stc_trigger_failed",
            extra={
                "kind": kind,
                "err_type": type(exc).__name__,
                "err": str(exc)[:120],
            },
        )
    return event_id


def emit_best_effort(
    store: "MemoryStore | None",
    kind: str,
    data: dict[str, Any],
    *,
    severity: str | None = None,
    session_id: str = "-",
) -> None:
    try:
        if store is None:
            logging.getLogger(__name__).debug("telemetry %s %s", kind, data)
            return
        write_event(store, kind, data, severity=severity,
                    session_id=session_id, buffered=True)
    except Exception:  # noqa: BLE001 -- telemetry must never break the caller
        try:
            logging.getLogger(__name__).debug("telemetry_emit_failed %s", kind)
        except Exception:  # noqa: BLE001
            pass


def query_events(
    store: MemoryStore,
    kind: str | None = None,
    since: datetime | None = None,
    severity: str | None = None,
    limit: int = 100,
    *,
    since_exclusive: bool = False,
) -> list[dict]:
    # Lazy import — avoids a circular at module load between events <-> store/_store.
    from iai_mcp.store._store import _normalize_ts_for_compare

    sql_parts: list[str] = [
        "SELECT id, kind, severity, domain, ts, data_json, session_id, source_ids_json"
        " FROM events"
    ]
    where_parts: list[str] = []
    params: list[Any] = []
    if kind is not None:
        where_parts.append("kind = ?")
        params.append(kind)
    if severity is not None:
        where_parts.append("severity = ?")
        params.append(severity)
    if since is not None:
        # events.ts is always written by write_event via the raw-datetime
        # default adapter (space-form, full-microsecond TEXT). Normalize the
        # boundary to T-form then swap separator to match the stored shape,
        # so lexicographic TEXT compare is correct AND microsecond-precise
        # (SQLite's datetime() func truncates to second resolution).
        op = ">" if since_exclusive else ">="
        where_parts.append(f"ts {op} ?")
        params.append(_normalize_ts_for_compare(since).replace("T", " "))
    if where_parts:
        sql_parts.append("WHERE " + " AND ".join(where_parts))
    sql_parts.append("ORDER BY ts DESC LIMIT ?")
    params.append(int(limit))
    sql = " ".join(sql_parts)

    with store.db._conn_lock:
        rows = store.db._conn.execute(sql, params).fetchall()

    out: list[dict] = []
    for row in rows:
        raw_data = row["data_json"] or "{}"
        if is_encrypted(raw_data):
            ad = str(row["id"]).encode("ascii")
            try:
                raw_data = decrypt_field(raw_data, store._key(), associated_data=ad)
            except Exception as exc:  # noqa: BLE001 -- AEAD raises bare InvalidTag; one bad row must not fail the query
                logging.getLogger(__name__).debug("event_decrypt_failed", extra={"id": row["id"], "err": str(exc)[:80]})
                raw_data = "{}"
        try:
            data = json.loads(raw_data)
        except (TypeError, json.JSONDecodeError):
            data = {}
        try:
            source_ids = json.loads(row["source_ids_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            source_ids = []
        ts_raw = row["ts"]
        if isinstance(ts_raw, datetime):
            ts_value = ts_raw if ts_raw.tzinfo is not None else ts_raw.replace(tzinfo=timezone.utc)
        else:
            try:
                ts_value = datetime.fromisoformat(
                    str(ts_raw).replace("Z", "+00:00").replace(" ", "T")
                )
                if ts_value.tzinfo is None:
                    ts_value = ts_value.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                ts_value = datetime.now(timezone.utc)
        out.append(
            {
                "id": row["id"],
                "kind": row["kind"],
                "severity": row["severity"] or None,
                "domain": row["domain"] or None,
                "ts": ts_value,
                "data": data,
                "session_id": row["session_id"],
                "source_ids": source_ids,
            }
        )
    return out
