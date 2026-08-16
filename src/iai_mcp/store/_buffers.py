from __future__ import annotations

import base64
import json
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


def _record_id_is_durable(store: "MemoryStore", record_id: object) -> bool:
    """True when `records` already holds a row for `record_id`.

    Used ONLY to confirm a UNIQUE-dup row's content is redundant before the
    buffered copy is dropped — never to decide anything about a non-dup
    error.

    Goes through `store.db._conn_lock` / `store.db._conn` directly rather
    than `ro_conn()`: this runs inside `flush_record_buffer`'s per-row retry
    loop, which already holds `_BUFFER_LOCK`. On the lilli driver `ro_conn()`
    borrows from a pooled queue whose `get()` blocks with no timeout when the
    pool is exhausted — that would stall every other holder of `_BUFFER_LOCK`
    behind a pool slot instead of a lock this code path already proves safe.
    `tbl.add([row])` a few lines above already takes `_conn_lock` from
    inside this same `_BUFFER_LOCK`-held loop, so `_BUFFER_LOCK` -> `_conn_lock`
    is the established ordering here — reuse it instead of adding a new one.
    """
    if record_id is None:
        return False
    with store.db._conn_lock:
        row = store.db._conn.execute(
            "SELECT 1 FROM records WHERE id = ? LIMIT 1", (str(record_id),)
        ).fetchone()
    return row is not None


_QUARANTINE_DIRNAME = ".record-quarantine"


def _recoverable_row(row: dict) -> dict:
    """Return a JSON-serializable copy of `row` that round-trips its verbatim
    content: bytes-valued fields (the structure-hv payloads) are base64-wrapped
    so recovery reconstructs the exact bytes. Datetime fields (`created_at` /
    `updated_at`) are dumped by the caller's `default=str` as ISO-8601 text —
    recovery parses them back; every other field is JSON-native.
    """
    def _enc(v: object) -> object:
        if isinstance(v, (bytes, bytearray)):
            return {"__bytes_b64__": base64.b64encode(bytes(v)).decode("ascii")}
        return v

    return {k: _enc(v) for k, v in row.items()}


def _quarantine_record_row(store: "MemoryStore", row: dict, reason: str) -> str | None:
    """Preserve a record row a flush cannot write so its verbatim content is
    recoverable rather than dropped. Appends one recoverable JSON line to a
    dead-letter sink under the store root and returns its path, or None only if
    the sink itself could not be written (logged loud). Recovery reads these
    files offline; the line carries the full row plus the failure reason.
    """
    record_id = row.get("id") if isinstance(row, dict) else None
    try:
        now = datetime.now(timezone.utc)
        qdir = store.root / _QUARANTINE_DIRNAME
        qdir.mkdir(parents=True, exist_ok=True)
        # The sink carries plaintext record fields (aaak_index, tags, language,
        # the raw embedding), so it is owner-only like the other plaintext
        # sinks — never world-readable.
        os.chmod(qdir, 0o700)
        target = qdir / f"{now:%Y%m%dT%H%M%S}-{os.getpid()}.jsonl"
        line = json.dumps(
            {
                "quarantined_at": now.isoformat(),
                "reason": reason,
                "id": record_id,
                "row": _recoverable_row(row),
            },
            ensure_ascii=False,
            default=str,
        )
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(target, 0o600)
        return str(target)
    except Exception as exc:  # noqa: BLE001 -- last-resort sink; never abort the flush
        logger.critical(
            "flush_record_buffer_quarantine_write_failed",
            extra={"id": record_id, "err": str(exc)[:200]},
        )
        return None


def _flush_records_isolating_integrity_errors(
    store: "MemoryStore", pending: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Retry a poisoned batch row-by-row after the whole-batch `.add()` raised
    `IntegrityError` (the batch write is transactional, so nothing landed).

    Returns `(landed, terminal)`. Every row in `terminal` resolved to exactly
    one terminal outcome — landed, dropped as a confirmed-durable duplicate,
    or hard-failed — so the caller removes ONLY `terminal` from the buffer;
    none of those three outcomes should retry the same batch again. `landed`
    (a subset of `terminal`) feeds the caller's downstream side-effects
    (corpus count cache, exact-index feed).

    A row hitting a TRANSIENT fault (`OSError`/`RuntimeError`/`ValueError` —
    the same tuple `flush_record_buffer`'s whole-batch attempt already
    treats as retry-worthy) stops the loop immediately: that row and every
    row after it in `pending` are excluded from `terminal` and stay buffered
    for the next flush, exactly like a transient whole-batch failure. Rows
    already processed before the transient fault keep their terminal outcome
    — a landed or hard-failed row is not replayed.

    A `UNIQUE constraint failed` row is dropped only after confirming its id
    is already durable in `records` (uuid4 ids + the SKIP-merge path
    returning without inserting + distinct batches mean a buffered dup's
    content is always redundant — see `_crypto_mig`/research A1). Any OTHER
    `IntegrityError` (e.g. a NOT NULL violation, which the stdlib driver DOES
    route through this same class) is never treated as a safe drop: the row
    can never be written as is, so it leaves the buffer to avoid an infinite
    retry — but its verbatim content is preserved to a recoverable dead-letter
    sink and surfaced loudly, never silently dropped.
    """
    from iai_mcp.errors import IntegrityError

    tbl = store.db.open_table(RECORDS_TABLE)
    landed: list[dict] = []
    terminal: list[dict] = []
    for row in pending:
        try:
            tbl.add([row])
        except IntegrityError as exc:
            msg = str(exc)
            record_id = row.get("id") if isinstance(row, dict) else None
            if "UNIQUE constraint failed" in msg and _record_id_is_durable(
                store, record_id
            ):
                logger.warning(
                    "flush_record_buffer_dropped_durable_duplicate",
                    extra={"id": record_id, "err": msg[:200]},
                )
                terminal.append(row)
                continue
            # Either a non-UNIQUE integrity violation, or a UNIQUE-shaped
            # message whose id could not actually be confirmed durable (an
            # inconsistency, not a safe drop). The row can never be written as
            # is, so it must leave the buffer to avoid an infinite retry — but
            # its verbatim content is preserved to a recoverable sink and the
            # drop is surfaced loudly, never silent (lossless-recall invariant).
            qpath = _quarantine_record_row(store, row, reason=msg[:200])
            logger.error(
                "flush_record_buffer_integrity_quarantined",
                extra={"id": record_id, "err": msg[:200], "quarantine": qpath},
            )
            try:
                from iai_mcp.events import write_event

                write_event(
                    store,
                    "record_quarantined",
                    {
                        "id": record_id,
                        "reason": msg[:200],
                        "quarantine_path": qpath,
                        "recoverable": qpath is not None,
                    },
                    severity="error",
                    buffered=False,
                )
            except Exception as exc:  # noqa: BLE001 -- telemetry MUST NOT crash flush
                logger.debug("record_quarantined telemetry failed: %s", str(exc)[:120])
            terminal.append(row)
            continue
        except (OSError, RuntimeError, ValueError) as exc:
            record_id = row.get("id") if isinstance(row, dict) else None
            logger.warning(
                "flush_record_buffer_row_retry_transient",
                extra={"id": record_id, "err": str(exc)[:120]},
            )
            break
        landed.append(row)
        terminal.append(row)
    return landed, terminal


def flush_record_buffer(store: "MemoryStore") -> int:
    from iai_mcp.errors import IntegrityError
    from iai_mcp.events import _BUFFER_LOCK

    with _BUFFER_LOCK:
        store_id = id(store)
        # Snapshot WITHOUT removing: records carry verbatim literal_surface and
        # must never be dropped on a write fault. The batch is removed from the
        # buffer ONLY after a terminal outcome (success, a confirmed-durable
        # dup, or a hard-failed row); on a TRANSIENT fault it stays buffered so
        # the next flush retries instead of losing content.
        buffered = _record_buffer.get(store_id)
        if not buffered:
            return 0
        pending = list(buffered)
        n_pending = len(pending)
        try:
            store.db.open_table(RECORDS_TABLE).add(pending)
            landed = pending
            terminal_rows = pending
        except (OSError, RuntimeError, ValueError) as exc:
            # Leave the snapshotted batch in the buffer (do not drop verbatim
            # content). A transient fault is retried by the next flush; a
            # persistent fault keeps the records and stays visible via the log.
            logger.warning(
                "flush_record_buffer_failed",
                extra={"n": n_pending, "err": str(exc)[:120]},
            )
            return 0
        except IntegrityError:
            # The whole-batch write is transactional (executemany rolls back
            # entirely on any row's IntegrityError) — retry per-row so a
            # genuinely poisoned row cannot make the WHOLE batch (including
            # good rows) stay buffered forever. A transient fault surfacing
            # mid-retry stops the per-row loop early: only rows that reached
            # a terminal outcome (landed/dropped-dup/hard-failed) come back
            # in `terminal_rows` — anything after the transient fault stays
            # buffered for the next flush, same as the whole-batch case above.
            landed, terminal_rows = _flush_records_isolating_integrity_errors(
                store, pending
            )
        # Remove ONLY the rows that reached a terminal outcome (landed,
        # confirmed-durable dup, or hard-failed) from the buffer by identity,
        # leaving any concurrently-appended record — and any row stopped short
        # by a transient fault mid-retry — untouched for the next flush.
        flushed_ids = {id(p) for p in terminal_rows}
        remaining = [r for r in _record_buffer.get(store_id, []) if id(r) not in flushed_ids]
        if remaining:
            _record_buffer[store_id] = remaining
        else:
            _record_buffer.pop(store_id, None)
        _record_last_flush_at[store_id] = datetime.now(timezone.utc)
        pending = landed
        n_pending = len(landed)
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
