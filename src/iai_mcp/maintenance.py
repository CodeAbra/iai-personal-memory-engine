from __future__ import annotations

import logging
import multiprocessing
import os
import tempfile
import time
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from typing import Any, Iterator

logger = logging.getLogger(__name__)

HIPPO_COMPACT_INTERVAL_SEC: float = float(
    os.environ.get("IAI_MCP_HIPPO_COMPACT_INTERVAL_SEC", "3600.0"),
)

_HIPPO_TABLES_TO_COMPACT: tuple[str, ...] = ("records", "edges", "events")

# Page-cache budget held during the full-table maintenance scans (count/size).
# The maintenance pass walks every table page through the engine's Python
# pager; without a bound the cache would retain the whole working set resident
# for the duration of the pass. A tight budget keeps the resident page set
# small (~1 MB) during the scans, then the prior budget — the much larger
# steady-state read budget — is restored so normal recall reads are unaffected.
# 256 KiB / 8 KiB-per-page = 32 resident pages — small enough that the
# maintenance pass's peak stays comfortably under the watchdog-relevant bound
# while keeping enough hot pages to avoid pathological re-read thrashing.
_MAINTENANCE_SCAN_CACHE_KIB: int = 256


@contextmanager
def _bounded_scan_cache(db: Any) -> Iterator[None]:
    """Temporarily tighten the engine page-cache budget around a scan-heavy pass.

    Applies only to the in-tree pager connection (which exposes ``_pager``);
    on the native sqlite3 driver the page cache lives in C and is already
    bounded by its own ``cache_size`` PRAGMA, so this is a no-op there. The
    prior budget is captured and restored on exit so the connection's general
    read budget is unchanged after the pass.
    """
    conn = getattr(db, "_conn", None) if db is not None else None
    pager = getattr(conn, "_pager", None) if conn is not None else None
    if conn is None or pager is None:
        yield
        return
    prior = getattr(pager, "_max_cached_pages", None)
    try:
        conn.execute(f"PRAGMA cache_size=-{_MAINTENANCE_SCAN_CACHE_KIB}")
        yield
    finally:
        # Restore the exact prior bound (page count or unbounded).
        try:
            pager.set_max_cached_pages(prior)
        except Exception:
            pass


def _measure_table_size_bytes(store: Any, table_name: str) -> int:
    try:
        db = getattr(store, "db", None)
        if db is None:
            return 0
        conn = getattr(db, "_conn", None)
        if conn is None:
            return 0
        try:
            row = conn.execute(
                "SELECT SUM(pgsize) FROM dbstat WHERE name = ?", (table_name,)
            ).fetchone()
            if row is not None and row[0] is not None:
                return int(row[0])
        except Exception:
            pass
        _COUNT_SQL: dict[str, str] = {
            "records": "SELECT COUNT(*) FROM records",
            "edges":   "SELECT COUNT(*) FROM edges",
            "events":  "SELECT COUNT(*) FROM events",
        }
        count_sql = _COUNT_SQL.get(table_name)
        if count_sql is not None:
            try:
                count_row = conn.execute(count_sql).fetchone()
                if count_row is not None:
                    return int(count_row[0]) * 256
            except Exception:
                pass
        return 0
    except (OSError, TypeError, AttributeError):
        return 0


_HNSW_WORKER_TIMEOUT_S: float = float(
    os.environ.get("IAI_MCP_HNSW_WORKER_TIMEOUT_S", "120.0")
)

_HNSW_WORKER_TIMEOUT_PER_1K_S: float = float(
    os.environ.get("IAI_MCP_HNSW_WORKER_TIMEOUT_PER_1K_S", "5.0")
)


def _worker_entry_hnsw(conn) -> None:  # noqa: ANN001
    """Picklable spawn target for the hnswlib rebuild worker."""
    from iai_mcp.hnsw_rebuild_worker import _worker_entry
    _worker_entry(conn)


def _rebuild_hnsw_in_child(db: Any) -> int:
    """Rebuild the hnswlib ANN index in a spawn-context child worker.

    The full numpy embedding matrix and hnswlib in-process allocation live
    in the child's address space.  The OS reclaims them on child exit, so
    the calling daemon process (i.e. parent) never sees the RSS spike.

    The child saves the rebuilt index to a temp path; the parent loads it
    into `db._hnsw_standby` and atomically swaps under `db._hnsw_lock`,
    matching the double-buffer invariant.

    Returns the number of records indexed (from the child's report).

    Raises RuntimeError on worker failure or timeout.

    AES-key fence: embedding BLOBs are raw float32; the child opens its own
    read-only SQLite connection using only the file path.  No HippoDB, no
    crypto module, and no key material crosses the process boundary.
    """
    from iai_mcp.hippo import (
        HNSW_INITIAL_CAPACITY,
        HNSW_EF_CONSTRUCTION,
        HNSW_M,
        HNSW_EF,
        RECALL_INDEX_EF,
    )

    hippo_dir = getattr(db, "_hippo_dir", None)
    if hippo_dir is None:
        raise RuntimeError("db has no _hippo_dir; cannot locate SQLite file")
    db_path = str(hippo_dir / "brain.sqlite3")

    # Count active records to scale the timeout and set HNSW capacity.
    try:
        with db._conn_lock:
            count_row = db._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        n_active = int(count_row[0]) if count_row else 0
    except Exception:  # noqa: BLE001
        n_active = 0

    cap = max(HNSW_INITIAL_CAPACITY, n_active * 2)
    timeout_s = min(
        _HNSW_WORKER_TIMEOUT_S + _HNSW_WORKER_TIMEOUT_PER_1K_S * (n_active / 1000.0),
        3600.0,
    )

    # Write the rebuilt index to a sibling temp file so the atomic rename is
    # on the same filesystem as the live index.
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(hippo_dir), suffix=".hnsw.child.tmp"
    )
    os.close(tmp_fd)
    # Serialize with in-process rebuilds: the child's DB read -> parent swap is
    # the same build->swap window _rebuild_lock protects (see HippoDB). The
    # journal opened here replays writes committed during the child build into
    # the swapped-in buffer so they are not lost at the swap. EVERY statement
    # after the acquire — journal open included — runs inside the try whose
    # finally releases the lock: a leaked _rebuild_lock hangs every future
    # rebuild (drains + consolidation) until daemon restart.
    rebuild_lock = getattr(db, "_rebuild_lock", None)
    if rebuild_lock is not None:
        rebuild_lock.acquire()
    journal_opened = False
    try:
        if hasattr(db, "_ann_journal"):
            with db._hnsw_lock:
                db._ann_journal = []
            journal_opened = True
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        process = ctx.Process(
            target=_worker_entry_hnsw,
            args=(child_conn,),
            daemon=True,
        )
        process.start()
        child_conn.close()

        parent_conn.send(("params", {
            "db_path": db_path,
            "tmp_path": tmp_path,
            "embed_dim": db._embed_dim,
            "hnsw_m": HNSW_M,
            "ef_construction": HNSW_EF_CONSTRUCTION,
            "ef": max(HNSW_EF, RECALL_INDEX_EF),
            "capacity": cap,
        }))

        deadline = time.monotonic() + timeout_s
        result_kind: str | None = None
        result_payload: Any = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.terminate()
                process.join(timeout=2.0)
                raise RuntimeError(
                    f"hnsw child worker timed out after {timeout_s:.0f}s"
                )
            if parent_conn.poll(min(remaining, 1.0)):
                result_kind, result_payload = parent_conn.recv()
                break

        parent_conn.close()
        process.join(timeout=5.0)

        if result_kind == "error":
            raise RuntimeError(f"hnsw child worker error: {result_payload}")
        if result_kind != "done":
            raise RuntimeError(f"hnsw child worker unexpected envelope: {result_kind!r}")

        if process.exitcode != 0:
            raise RuntimeError(f"hnsw child worker exited with code {process.exitcode}")

        rebuilt_count: int = int((result_payload or {}).get("rebuilt_count", 0))

        # Load the child-built index and prepare the fresh label map OFF the
        # recall lock — the lock below holds only for O(journal) replay plus
        # two reference swaps, never a corpus scan or file I/O.
        import hnswlib
        new_index = hnswlib.Index(space="cosine", dim=db._embed_dim)
        new_index.load_index(tmp_path, max_elements=cap, allow_replace_deleted=True)
        new_index.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
        new_index.set_num_threads(1)
        new_label_map = db._load_label_map_from_sqlite()

        with db._hnsw_lock:
            journal = db._ann_journal if journal_opened else None
            if journal_opened:
                db._ann_journal = None
            if journal:
                db._apply_ann_journal(new_index, journal, new_label_map)
            standby = getattr(db, "_hnsw_standby", None)
            if standby is None:
                # Standby not yet allocated (first post-boot OPTIMIZE_HIPPO
                # before _allocate_standby_index ran): publish directly.
                db._hnsw = new_index
            else:
                # Atomic buffer swap: new_index becomes the active index.
                db._hnsw, db._hnsw_standby = new_index, db._hnsw
            db._label_map = new_label_map

        # Rename the temp file over the authoritative index path (off-lock:
        # file I/O has no business inside the recall-critical section).
        # On failure, log and fall through — the index is already live in
        # memory; the temp file is cleaned up below.
        hnsw_path = str(getattr(db, "_hnsw_path", None) or tmp_path)
        renamed = False
        try:
            os.replace(tmp_path, hnsw_path)
            renamed = True
        except OSError as exc:
            logger.warning("hnsw_rebuild_atomic_rename_failed: %s", exc)

        if not renamed:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        return rebuilt_count

    except Exception:
        # Clean up the temp file on any failure path before re-raising.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    finally:
        if journal_opened:
            with db._hnsw_lock:
                db._ann_journal = None
        if rebuild_lock is not None:
            rebuild_lock.release()


def optimize_hippo_storage(
    store: Any,
    *,
    retention: timedelta | None = None,
    max_versions: int | None = None,
    delete_unverified: bool = False,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    db = getattr(store, "db", None)

    overall_t0 = time.monotonic()

    # The maintenance pass walks every table page through the engine's pager
    # (row counts, size measurement, the child-rebuild's active-record count).
    # Hold a tight page-cache budget for the whole pass so the resident page set
    # stays small instead of retaining the full corpus; the prior (larger) read
    # budget is restored on exit so normal recall reads are unaffected.
    with _bounded_scan_cache(db):
        per_table_before: dict[str, tuple[int, int]] = {}
        for table_name in _HIPPO_TABLES_TO_COMPACT:
            try:
                if db is None:
                    raise RuntimeError("store has no .db attribute")
                tbl = db.open_table(table_name)
                rows = int(tbl.count_rows())
                size = _measure_table_size_bytes(store, table_name)
                per_table_before[table_name] = (rows, size)
            except Exception:
                per_table_before[table_name] = (0, 0)

        # Own the consolidation-intent flag for the checkpoint+VACUUM window:
        # while it exists, client shared opens are refused
        # (lock_protocol.acquire_client_shared_nb) and the watchdog cascade
        # defers its heavy read. Set here, cleared in the finally below —
        # a crash mid-window is healed by the daemon-boot stale-flag cleanup.
        _intent_lock_path = None
        if db is not None:
            hippo_dir = getattr(db, "_hippo_dir", None)
            if hippo_dir is not None:
                from iai_mcp.lock_protocol import (  # noqa: PLC0415
                    clear_consolidation_intent,
                    set_consolidation_intent,
                )
                _intent_lock_path = hippo_dir / ".lock"
                set_consolidation_intent(_intent_lock_path)

        db_compaction_error: str | None = None
        vacuum_elapsed: float = 0.0
        vacuum_skipped_foreground = False
        try:
            # A stdlib VACUUM is monolithic and un-yieldable while it holds
            # _conn_lock: a recall arriving mid-VACUUM fence-exhausts onto the
            # writer lock and stalls behind it. Defer this cycle's VACUUM when
            # a foreground read is IN FLIGHT right now (window_s=0.0 keeps
            # only the in-flight branch of the beacon — a merely-recent read
            # must not defer a night cycle or a user-forced optimize).
            # On the lilli driver VACUUM compacts the meta DDL history (one
            # bounded single-tree pass) and the checkpoint is milliseconds,
            # so the window barely exists there.
            try:
                from iai_mcp.concurrency import foreground_recent  # noqa: PLC0415
                vacuum_skipped_foreground = foreground_recent(window_s=0.0)
            except ImportError:
                vacuum_skipped_foreground = False

            conn = db._conn
            conn_lock = getattr(db, "_conn_lock", None)
            _lock_ctx = conn_lock if conn_lock is not None else nullcontext()
            with _lock_ctx:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                if not vacuum_skipped_foreground:
                    if conn.in_transaction:
                        logger.warning(
                            "hippo_compact_implicit_commit: VACUUM requires no open transaction; "
                            "committing outer transaction before VACUUM."
                        )
                        conn.execute("COMMIT")
                    vac_t0 = time.monotonic()
                    conn.execute("VACUUM")
                    vacuum_elapsed = round(time.monotonic() - vac_t0, 3)
                else:
                    logger.info(
                        "hippo_compact: VACUUM deferred — foreground client "
                        "recently active; next cycle retries."
                    )
        except Exception as exc:
            db_compaction_error = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("hippo_compact_failed: %s", db_compaction_error)
        finally:
            if _intent_lock_path is not None:
                clear_consolidation_intent(_intent_lock_path)

        hnsw_error: str | None = None
        hnsw_elapsed: float = 0.0
        try:
            if db is not None:
                hnsw_t0 = time.monotonic()
                _rebuild_hnsw_in_child(db)
                hnsw_elapsed = round(time.monotonic() - hnsw_t0, 3)
        except Exception as exc:
            hnsw_error = f"{type(exc).__name__}: {exc}"[:500]
            logger.warning("hippo_hnsw_rebuild_failed: %s", hnsw_error)

        for table_name in _HIPPO_TABLES_TO_COMPACT:
            rows_before, size_before = per_table_before.get(table_name, (0, 0))
            per_table: dict[str, Any] = {
                "rows_before": rows_before,
                "rows_after": rows_before,
                "size_bytes_before": size_before,
                "size_bytes_after": 0,
                "vacuum_elapsed_sec": vacuum_elapsed,
                "hnswlib_rebuild_elapsed_sec": hnsw_elapsed,
                "elapsed_sec": round(time.monotonic() - overall_t0, 3),
            }
            try:
                if db is None:
                    raise RuntimeError("store has no .db attribute")
                tbl = db.open_table(table_name)
                per_table["rows_after"] = int(tbl.count_rows())
                per_table["size_bytes_after"] = _measure_table_size_bytes(
                    store, table_name,
                )
            except Exception as exc:
                per_table["error"] = (
                    f"post_compact_measurement_failed: "
                    f"{type(exc).__name__}: {exc}"
                )[:500]
            if db_compaction_error and "error" not in per_table:
                per_table["error"] = f"db_compact_failed: {db_compaction_error}"
            if hnsw_error and "error" not in per_table:
                per_table["error"] = f"hnsw_rebuild_failed: {hnsw_error}"
            report[table_name] = per_table

    return report


# Rows dropped per DELETE batch during aged-tombstone compaction. A single
# unbounded DELETE over the whole aged set drives one pager transaction that
# rebalances every touched leaf page in sequence; bounding each transaction to
# this many rows keeps any one commit's rebalance work small regardless of how
# large the aged-tombstone set has grown.
TOMBSTONE_DROP_BATCH_SIZE: int = int(
    os.environ.get("IAI_MCP_TOMBSTONE_DROP_BATCH_SIZE", "100"),
)


def batched_tombstone_drop(
    tbl: Any,
    drop_where: str,
    *,
    batch_size: int = TOMBSTONE_DROP_BATCH_SIZE,
    interrupt_check: "Any | None" = None,
) -> int:
    """Physically drop rows matching ``drop_where`` in bounded batches.

    Selects up to ``batch_size`` ids matching the predicate, deletes exactly
    those ids, then repeats until no matching row remains. Splitting one
    unbounded ``DELETE ... WHERE <predicate>`` into many bounded
    ``DELETE ... WHERE id IN (...)`` calls removes the identical row set --
    only the transaction boundaries change, so which rows survive is
    unaffected by the batch size.

    ``interrupt_check`` (if given) is polled between batches; when it returns
    True the loop stops early with a partial count rather than raising, so an
    in-progress sleep cycle interrupt never leaves a half-finished drop in an
    inconsistent state (each batch's own transaction always fully commits or
    fully rolls back).

    Fail-safe: a failure reading or deleting a batch stops the loop and
    returns the count dropped so far rather than raising, so a compaction
    hiccup never crashes its caller (the sleep cycle or a migrate finalize).
    """
    total_dropped = 0
    while True:
        if interrupt_check is not None:
            try:
                if interrupt_check():
                    break
            except Exception:  # noqa: BLE001 -- interrupt probe must not abort the drop
                pass
        try:
            rows = (
                tbl.search()
                .select(["id"])
                .where(drop_where)
                .limit(batch_size)
                .to_pandas()
            )
        except Exception as exc:  # noqa: BLE001 -- a read hiccup ends the pass, not the caller
            logger.debug("batched_tombstone_drop select failed: %s", exc)
            break
        if rows is None or len(rows) == 0:
            break
        ids = [str(v) for v in rows["id"].tolist()]
        id_list = ", ".join(f"'{i}'" for i in ids)
        # Keep the original drop predicate in the WHERE clause (scoped by
        # id to a bounded subset) rather than switching to an id-only
        # filter -- the predicate is the correctness anchor (only
        # aged-tombstoned rows are ever candidates for this delete), and
        # the id clause is purely the batch-size bound.
        batch_where = f"({drop_where}) AND id IN ({id_list})"
        try:
            tbl.delete(batch_where)
        except Exception as exc:  # noqa: BLE001 -- a delete hiccup ends the pass, not the caller
            logger.debug("batched_tombstone_drop delete failed: %s", exc)
            break
        total_dropped += len(ids)
        if len(ids) < batch_size:
            break
    return total_dropped


def optimize_lance_storage(store: Any, **kwargs: Any) -> "dict[str, dict[str, Any]]":
    import warnings
    warnings.warn(
        "optimize_lance_storage is deprecated; use optimize_hippo_storage",
        DeprecationWarning,
        stacklevel=2,
    )
    return optimize_hippo_storage(store, **kwargs)


def symmetrize_self_loops(
    store: Any, *, dry_run: bool = True,
) -> dict[str, Any]:
    from uuid import UUID

    db = store.db
    records_tbl = db.open_table("records")
    edges_tbl = db.open_table("edges")

    records_df = records_tbl.to_pandas()
    edges_df = edges_tbl.to_pandas()

    records_ids = set(records_df["id"].astype(str))
    if len(edges_df) > 0:
        self_loops_df = edges_df[
            (edges_df["edge_type"] == "hebbian")
            & (edges_df["src"] == edges_df["dst"])
        ]
        self_loop_ids = set(self_loops_df["src"].astype(str))
    else:
        self_loop_ids = set()

    missing = sorted(records_ids - self_loop_ids)
    result: dict[str, Any] = {
        "records_total": len(records_ids),
        "self_loops_present": len(self_loop_ids & records_ids),
        "self_loops_pending": len(missing),
        "self_loops_inserted": 0,
        "dry_run": bool(dry_run),
    }

    if dry_run or not missing:
        return result

    pairs = [(UUID(rid), UUID(rid)) for rid in missing]
    store.boost_edges(pairs, delta=0.1, edge_type="hebbian")
    result["self_loops_inserted"] = len(missing)
    return result


def optimize_lance_storage(store: Any, **kwargs: Any) -> dict[str, dict[str, Any]]:
    import warnings

    warnings.warn(
        "optimize_lance_storage is deprecated; use optimize_hippo_storage instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return optimize_hippo_storage(store, **kwargs)
