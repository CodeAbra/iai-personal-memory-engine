from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.store import MemoryStore

from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE

logger = logging.getLogger(__name__)


_record_buffer: dict[int, list[dict]] = {}
_record_last_flush_at: dict[int, datetime] = {}


def flush_record_buffer(store: "MemoryStore") -> int:
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        # Snapshot WITHOUT removing: records carry verbatim literal_surface and
        # must never be dropped on a write fault. The batch is removed from the
        # buffer ONLY after the write succeeds (clear-after-success); on failure
        # it stays buffered so the next flush retries instead of losing content.
        buffered = _record_buffer.get(store_id)
        if not buffered:
            return 0
        pending = list(buffered)
        n_pending = len(pending)
        try:
            store.db.open_table(RECORDS_TABLE).add(pending)
        except (OSError, RuntimeError, ValueError) as exc:
            # Leave the snapshotted batch in the buffer (do not drop verbatim
            # content). A transient fault is retried by the next flush; a
            # persistent fault keeps the records and stays visible via the log.
            logger.warning(
                "flush_record_buffer_failed",
                extra={"n": n_pending, "err": str(exc)[:120]},
            )
            return 0
        # Write succeeded: remove exactly the snapshotted records by identity,
        # leaving any concurrently-appended record untouched.
        flushed_ids = {id(p) for p in pending}
        remaining = [r for r in _record_buffer.get(store_id, []) if id(r) not in flushed_ids]
        if remaining:
            _record_buffer[store_id] = remaining
        else:
            _record_buffer.pop(store_id, None)
        _record_last_flush_at[store_id] = datetime.now(timezone.utc)
        # The flush's count delta is known exactly (rows are classified by the
        # same predicates the COUNT queries use), so shift the cached values in
        # place — dropping them here forced a full filtered scan per flush.
        # Guarded against a missing or not-yet-constructed cache (e.g. a bare
        # HippoDB access without a MemoryStore).
        try:
            _cc = getattr(store, "_corpus_count_cache", None)
            if _cc is not None:
                _n_active = _n_pending = 0
                for _r in pending:
                    if _r.get("tombstoned_at") is not None:
                        continue
                    if int(_r.get("embedding_pending", 0) or 0):
                        _n_pending += 1
                    else:
                        _n_active += 1
                if _n_active:
                    _cc.adjust("active", _n_active)
                if _n_pending:
                    _cc.adjust("pending", _n_pending)
        except Exception:  # noqa: BLE001 -- cache adjustment MUST NOT crash flush
            pass
        # Upsert each flushed row into the resident exact-cosine matrix. The
        # flushed items are DICTS (the same shape written to SQL above), not
        # record objects, so fields are read by key, never by attribute.
        # Pending rows (embedding_pending == 1) are excluded from the matrix by
        # design and skipped here. Per-item isolation (mirroring the reembed
        # feed): one malformed item must skip only itself, never abort the
        # rest of the batch — the matrix would otherwise silently miss the
        # freshest captures until the next destructive-write invalidate.
        _feed_exact = getattr(store, "_feed_exact_index", None)
        if _feed_exact is not None:
            for _item in pending:
                try:
                    if int(_item.get("embedding_pending", 0) or 0) == 1:
                        continue
                    _feed_exact(_item["id"], _item["embedding"])
                except Exception as exc:  # noqa: BLE001 -- exact-index feed MUST NOT crash flush
                    logger.debug(
                        "exact-index feed failed for %s: %s",
                        _item.get("id") if isinstance(_item, dict) else None,
                        type(exc).__name__,
                    )
                    continue
        # The lexical postings were fed at insert time (plaintext lives only
        # in the in-memory record); the generation restamp belongs HERE,
        # where the rows land and the corpus generation actually moves.
        try:
            _lex = getattr(store, "_lexical_idx", None)
            if _lex is not None and _lex.generation is not None:
                _gen = None
                try:
                    _gen = store._corpus_count_cache.generation()
                except Exception:  # noqa: BLE001
                    pass
                _lex.restamp(_gen)
        except Exception:  # noqa: BLE001 -- lexical restamp MUST NOT crash flush
            pass
        try:
            from iai_mcp.events import write_event
            write_event(
                store,
                "lance_buffer_flush",
                {"table": "records", "count": n_pending},
                severity="info",
                buffered=False,
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
            logger.debug("lance_buffer_flush telemetry failed: %s", str(exc)[:120])
        return n_pending


def should_flush_record_buffer(store_id: int, max_size: int | None = None) -> bool:
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_RECORD_BUFFER_MAX", "500"))
        except ValueError:
            max_size = 500
    return len(_record_buffer.get(store_id, [])) >= max_size


def should_flush_record_buffer_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    if not _record_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec


_edge_buffer: dict[int, list[dict]] = {}
_edge_last_flush_at: dict[int, datetime] = {}


def reset_store_buffers(store_id: int) -> None:
    """Purge every buffer entry keyed to store_id. Buffers key on id(store),
    and CPython reuses freed addresses — a fresh store MUST start clean or
    it inherits a dead store's unflushed rows, which poisons its own writes
    and can land another store's content in its tables."""
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        _record_buffer.pop(store_id, None)
        _record_last_flush_at.pop(store_id, None)
        _edge_buffer.pop(store_id, None)
        _edge_last_flush_at.pop(store_id, None)


def flush_edge_buffer(store: "MemoryStore") -> int:
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        # Clear-after-success: snapshot without removing, write, then remove only
        # the snapshotted edges on success. Edges are rederivable, but the same
        # durable pattern is applied for consistency — a write fault re-buffers
        # the batch for the next flush instead of dropping it.
        buffered = _edge_buffer.get(store_id)
        if not buffered:
            return 0
        pending = list(buffered)
        n_pending = len(pending)
        try:
            store.db.open_table(EDGES_TABLE).merge_insert(["src", "dst", "edge_type"]).execute(pending)
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "flush_edge_buffer_failed",
                extra={"n": n_pending, "err": str(exc)[:120]},
            )
            return 0
        flushed_ids = {id(p) for p in pending}
        remaining = [r for r in _edge_buffer.get(store_id, []) if id(r) not in flushed_ids]
        if remaining:
            _edge_buffer[store_id] = remaining
        else:
            _edge_buffer.pop(store_id, None)
        _edge_last_flush_at[store_id] = datetime.now(timezone.utc)
        # Invalidate the edges count in the corpus-count cache after a
        # successful flush so the next edges_count() call recomputes from SQL.
        try:
            _cc = getattr(store, "_corpus_count_cache", None)
            if _cc is not None:
                _cc.invalidate("edges")
        except Exception:  # noqa: BLE001 -- cache invalidation MUST NOT crash flush
            pass
        try:
            from iai_mcp.events import write_event
            write_event(
                store,
                "lance_buffer_flush",
                {"table": "edges", "count": n_pending},
                severity="info",
                buffered=False,
            )
        except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
            logger.debug("lance_buffer_flush telemetry failed: %s", str(exc)[:120])
        return n_pending


def should_flush_edge_buffer(store_id: int, max_size: int | None = None) -> bool:
    if max_size is None:
        try:
            max_size = int(os.environ.get("IAI_MCP_EDGE_BUFFER_MAX", "500"))
        except ValueError:
            max_size = 500
    return len(_edge_buffer.get(store_id, [])) >= max_size


def should_flush_edge_buffer_by_time(
    store_id: int,
    last_flush_at: datetime | None,
    max_age_sec: float = 5.0,
) -> bool:
    if not _edge_buffer.get(store_id):
        return False
    if last_flush_at is None:
        return True
    return (datetime.now(timezone.utc) - last_flush_at).total_seconds() >= max_age_sec
