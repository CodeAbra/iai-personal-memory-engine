"""Verbatim SQLite -> lilli store migrator.

Copies a SQLite-format Hippo store into a FRESH lilli-format store row-by-row,
byte-for-byte.  Ciphertext strings, float32 embedding BLOBs, and plaintext
columns are copied AS-IS -- there is NO decrypt / re-encrypt, so the AES key
never crosses the engine boundary (each encrypted cell is self-contained and its
AAD, the record id, travels verbatim with the row).

The copy is RSS-flat: rows are streamed with ``fetchmany`` and written with
batched ``executemany`` + a per-batch commit, so the whole table is never
buffered in memory.  ``vec_label`` is preserved per-row and the lilli
AUTOINCREMENT high-water mark is bumped to ``max(vec_label)`` so the first
post-migration insert cannot collide.  The ANN index is rebuilt from the copied
embeddings before close, so recall works immediately (no stale empty index).

The source is opened READ-ONLY; the migrator physically cannot mutate it.  A
same-path guard refuses to run if the source resolves to the same file as the
destination store.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# The six store tables, copied in this order.  ``_hippo_meta`` uses
# INSERT OR REPLACE because the fresh dest seeds two meta rows on open.
_ALL_TABLES: tuple[str, ...] = (
    "records",
    "edges",
    "events",
    "budget_ledger",
    "ratelimit_ledger",
    "_hippo_meta",
)

# The events timestamp column (telemetry prune predicate target).
_EVENTS_TS_COLUMN = "ts"


@dataclass
class MigrateReport:
    """Outcome of a verbatim migration."""

    rows_copied: dict[str, int] = field(default_factory=dict)
    max_vec_label: int = 0
    elapsed_sec: float = 0.0
    peak_rss_mb: float = 0.0
    # Runtime graph cache pre-warm result.  True when the pre-warm completed
    # successfully and a warm cache was written for the migrated store.  False
    # when the pre-warm was skipped (degenerate empty corpus) or failed.  The
    # elapsed seconds covers only the pre-warm step itself; it is 0.0 on skip or
    # failure (the boot then pays a one-time cold rebuild — correct but slower).
    prewarm_done: bool = False
    prewarm_elapsed_sec: float = 0.0
    # Stale runtime-graph-cache invalidation outcome (fingerprint-checked
    # up-front delete, never lazy).  ``cache_deleted`` is True when a cache
    # file existed at the destination root and was removed because its
    # corpus fingerprint (payload_record_count) fell outside the drift
    # tolerance of the migrated store's active record count, or because the
    # cache could not be fingerprinted at all (corrupt / unreadable / missing
    # field).  ``cache_invalidation_reason`` is one of: "no_cache" (nothing to
    # do), "drift" (fingerprint mismatch beyond tolerance), "unreadable"
    # (corrupt/undecodable cache, deleted defensively), or "within_tolerance"
    # (fingerprint matched close enough, left byte-untouched).
    cache_deleted: bool = False
    cache_invalidation_reason: str = "no_cache"
    # Recall-index sidecar persist outcome (records.colindex). True when the
    # persist ran to completion on a lilli destination. Always False on a
    # stdlib destination (no such call exists there) -- not an error, just
    # not applicable. A persist failure never fails the migration; the boot
    # then pays a one-time cold rebuild (correct but slower).
    col_index_persisted: bool = False
    col_index_persist_error: str = ""
    # Migrate-finalize tombstone compaction. A fresh migrate copies every
    # source row verbatim (including rows already past the tombstone TTL in
    # source time) -- this pass drops those aged-past-TTL rows so the fresh
    # store does not start out re-importing stale bloat, rather than waiting
    # for the first nightly sleep cycle to age them out. Best-effort: a
    # failure here is logged and never fails the migration.
    tombstone_compact_dropped: int = 0
    tombstone_compact_error: str = ""


def _rss_mb() -> float:
    """Current process resident set in MiB, fail-soft to 0 (psutil flakiness)."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:  # noqa: BLE001 -- RSS sampling must never fail the migration
        return 0.0


def _dst_db_path(dst_root: str | Path) -> Path:
    return Path(dst_root) / "hippo" / "brain.sqlite3"


_log = logging.getLogger(__name__)


def _invalidate_stale_runtime_graph_cache(
    dst_root: str | Path,
    migrated_active_count: int,
) -> tuple[bool, str]:
    """Delete the destination root's runtime graph cache if it is stale.

    A migrated store must never sit next to a stdlib-era (or otherwise
    corpus-mismatched) ``runtime_graph_cache.json`` -- an on-disk cache whose
    stored corpus fingerprint (``payload_record_count``) drifts too far from
    the migrated store's ACTIVE record count would only be discovered lazily
    by the recall-adjacent boot probe, which responds by forcing a
    synchronous full-corpus rebuild on the critical path. Checking and
    deleting up front here means that forced rebuild never happens; the
    rebuild still runs later, as ordinary low-priority background work
    against a matching corpus.

    No-raise: any failure while reading, decrypting, or deleting the cache is
    logged as a warning and never fails the migration -- a cache we cannot
    even open is stale by definition and is deleted defensively.

    Returns ``(deleted, reason)`` where ``reason`` is one of ``"no_cache"``,
    ``"drift"``, ``"unreadable"``, ``"within_tolerance"``, or
    ``"check_failed"`` (the check itself raised; the file is left untouched
    defensively rather than risking a wrong delete).
    """
    from iai_mcp.runtime_graph_cache import _cache_path, _load_and_decrypt_cache
    from iai_mcp.retrieve import _within_drift_tolerance

    class _CacheProbeStore:
        """Minimal store-shaped shim for cache path/key resolution.

        ``_load_and_decrypt_cache`` only needs ``root`` (for the cache path
        and as a crypto-key fallback anchor) and a way to resolve the
        encryption key; it never touches record/edge counts, so this shim
        does not need to expose them.
        """

        def __init__(self, root: Path) -> None:
            self.root = root
            self.user_id = "default"
            self._crypto_key = None

    try:
        cache_path = _cache_path(_CacheProbeStore(Path(dst_root)))
        if not cache_path.exists():
            return False, "no_cache"

        probe = _CacheProbeStore(Path(dst_root))
        data = _load_and_decrypt_cache(probe)
        if data is None:
            cache_path.unlink(missing_ok=True)
            return True, "unreadable"

        payload_record_count_raw = data.get("payload_record_count")
        if not isinstance(payload_record_count_raw, int):
            cache_path.unlink(missing_ok=True)
            return True, "unreadable"

        cached_count = int(payload_record_count_raw)
        if not _within_drift_tolerance(cached_count, int(migrated_active_count)):
            cache_path.unlink(missing_ok=True)
            return True, "drift"

        return False, "within_tolerance"
    except Exception:  # noqa: BLE001 -- invalidation must never fail the migrate
        _log.warning(
            "runtime graph cache invalidation check failed; leaving cache "
            "file untouched (first daemon boot will re-check drift lazily)",
            exc_info=True,
        )
        return False, "check_failed"


def _persist_col_index(dst: "object", report: MigrateReport) -> None:
    """Persist the engine's column-index sidecar on the still-open raw dst.

    Lilli-gated (no-op, no error, on a stdlib destination -- the connection
    simply lacks the ``persist_col_indexes`` attribute there). Must run at
    the VERY END of migrate finalize, AFTER ``_prewarm_graph_cache``: prewarm
    opens the migrated store as a full ``MemoryStore``, which runs the
    stamp-gated boot migrations (``migrate_role_column``,
    ``migrate_record_tags_backfill``) that can issue bulk ``UPDATE``s and
    thereby advance the engine's column-generation counter. Persisting AFTER
    those settles means the sidecar's generation is the one that survives to
    the first daemon boot; persisting earlier would be silently invalidated
    by ``load_col_indexes``'s generation check, defeating the whole point.

    Best-effort / fail-safe: a persist failure is logged and recorded on the
    report, never re-raised -- the migrated DATA and MigrateReport must
    survive regardless. The boot then pays a one-time cold build (correct,
    slower), mirroring the sleep-step persist call's own fail-safety
    (``lilli/cycle/sleep_pipeline/_recall_index.py``).
    """
    conn = getattr(dst, "_conn", None)
    persist = getattr(conn, "persist_col_indexes", None)
    if not callable(persist):
        return  # stdlib destination -- no such surface, nothing to do

    from contextlib import nullcontext

    lock = getattr(dst, "_conn_lock", None)
    lock_ctx = lock if lock is not None else nullcontext()
    try:
        with lock_ctx:
            persist()
        report.col_index_persisted = True
    except Exception as exc:  # noqa: BLE001 -- persist must never fail migrate
        _log.warning(
            "records.colindex persist failed at migrate finalize; first "
            "daemon boot will pay a one-time cold rebuild (migrated data is "
            "intact): %s",
            exc,
            exc_info=True,
        )
        report.col_index_persist_error = str(exc)[:200]


def _source_time_now(dst: "object") -> "datetime":
    """The migrated store's own timeline "now", anchored to its own data.

    ``tombstoned_at`` (and ``created_at``) values travel verbatim from the
    source (source-time wall-clock) -- the drop predicate below must
    evaluate the grace window against a "now" on that SAME timeline, not the
    migrate host's real wall-clock, which can be arbitrarily far from the
    source data's own timeline (a migrate can run long after the source data
    was written). ``MAX(created_at)`` over the copied records is the
    freshest timestamp the store itself has recorded -- using it as "now"
    means a row tombstoned recently relative to the source's own activity
    stays within its grace window regardless of how much real wall-clock
    time has elapsed since that data was generated. Falls back to the real
    wall-clock only when the destination has no records at all (nothing to
    anchor to, and nothing to compact either).
    """
    from datetime import datetime, timezone

    from iai_mcp.store import RECORDS_TABLE

    try:
        tbl = dst.open_table(RECORDS_TABLE)
        row = tbl.count_rows()
        if row == 0:
            return datetime.now(timezone.utc)
        conn = getattr(dst, "_conn", None)
        lock = getattr(dst, "_conn_lock", None)
        from contextlib import nullcontext
        lock_ctx = lock if lock is not None else nullcontext()
        with lock_ctx:
            result = conn.execute(
                "SELECT MAX(created_at) FROM records",
            ).fetchone()
        max_created_at = result[0] if result is not None else None
        if max_created_at is None:
            return datetime.now(timezone.utc)
        if isinstance(max_created_at, datetime):
            parsed = max_created_at
        else:
            # Both storage drivers may format the timestamp with or without a
            # UTC offset suffix (e.g. "2026-06-01 12:00:00" vs.
            # "2026-06-01 12:00:00+00:00") -- fromisoformat accepts either.
            parsed = datetime.fromisoformat(str(max_created_at))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    except Exception:  # noqa: BLE001 -- fall back to wall-clock on any read failure
        return datetime.now(timezone.utc)


def _compact_aged_tombstones_at_finalize(dst: "object", report: MigrateReport) -> None:
    """Drop aged-past-TTL tombstoned rows from the freshly migrated dest.

    A fresh migrate is a fresh consolidation event: ``tombstoned_at`` was
    copied VERBATIM from the source (source-time wall-clock), so the drop
    predicate compares that copied source-time value against
    ``source_time_now - tombstone_ttl_sec`` -- the same grace-windowed
    predicate the sleep step's ``step_compact_hippo`` uses, evaluated on the
    migrated store's OWN timeline (see ``_source_time_now``), NOT a clock
    restarted at migrate time (that would treat already-source-aged
    tombstones as freshly tombstoned and drop nothing).

    Pin-protect mirrors the sleep step: pinned/never_decay rows are
    untombstoned before the drop so a pinned record can never be physically
    removed here.

    Best-effort / fail-safe: any failure is logged and recorded on the
    report; the migrated DATA and MigrateReport must survive regardless (the
    aged bloat is then cleaned up by the first nightly sleep cycle instead).
    """
    from datetime import timedelta

    from iai_mcp.daemon_config import _load_erasure_config
    from iai_mcp.maintenance import batched_tombstone_drop
    from iai_mcp.store import RECORDS_TABLE

    try:
        cfg = _load_erasure_config()
        ttl_sec = cfg.tombstone_ttl_sec
        now = _source_time_now(dst)
        drop_cutoff = now - timedelta(seconds=ttl_sec)
        drop_cutoff_str = drop_cutoff.strftime("%Y-%m-%d %H:%M:%S")

        tbl = dst.open_table(RECORDS_TABLE)

        untomb_where = (
            "tombstoned_at IS NOT NULL "
            "AND (pinned = true OR never_decay = true)"
        )
        try:
            count_untombstoned = int(tbl.count_rows(filter=untomb_where))
        except Exception:  # noqa: BLE001 -- best-effort count, never fails migrate
            count_untombstoned = 0
        if count_untombstoned > 0:
            tbl.update(
                where=untomb_where,
                values={"tombstoned_at": None, "live": 1},
            )

        drop_where = (
            "tombstoned_at IS NOT NULL "
            f"AND tombstoned_at < '{drop_cutoff_str}'"
        )
        report.tombstone_compact_dropped = batched_tombstone_drop(
            tbl, drop_where,
        )
    except Exception as exc:  # noqa: BLE001 -- compaction must never fail migrate
        _log.warning(
            "migrate-finalize tombstone compaction failed; migrated data is "
            "intact, aged bloat will be cleaned up by the first sleep cycle: %s",
            exc,
            exc_info=True,
        )
        report.tombstone_compact_error = str(exc)[:200]


def _prewarm_graph_cache(
    dst_root: str | Path,
    report: MigrateReport,
    current_peak_rss: float,
) -> float:
    """Build and persist the runtime graph cache for the migrated store.

    Opens the migrated store as a MemoryStore and calls build_runtime_graph once.
    The cache is built from the migrated data in this process, before any further
    writes, so the payload_record_count in the cache matches the migrated active
    record count exactly — the drift gate on first daemon boot is trivially
    satisfied and no cold rebuild fires.

    Best-effort: any failure is logged as a warning and the migrated data is left
    intact.  The report's prewarm_done and prewarm_elapsed_sec fields are updated
    in-place.  Returns the updated peak_rss (may have grown during the build).
    """
    import time as _time

    try:
        from iai_mcp.store import MemoryStore
        from iai_mcp.retrieve import build_runtime_graph

        # Skip the pre-warm on a degenerate corpus (nothing to build a graph for).
        # A store with zero active records produces an empty centrality map which
        # _runtime_graph_rebuild_needed treats as needing a rebuild — not useful.
        t_pw0 = _time.monotonic()
        store = MemoryStore(dst_root)
        try:
            active = store.active_records_count()
            if active == 0:
                _log.warning(
                    "runtime graph cache pre-warm skipped: migrated store has "
                    "no active records"
                )
                return current_peak_rss
            build_runtime_graph(store)
        finally:
            try:
                store.db.close()
            except Exception:  # noqa: BLE001
                pass

        elapsed = _time.monotonic() - t_pw0
        report.prewarm_done = True
        report.prewarm_elapsed_sec = elapsed
        sampled = _rss_mb()
        if sampled > current_peak_rss:
            current_peak_rss = sampled

    except Exception:  # noqa: BLE001 -- pre-warm must never abort the migration
        _log.warning(
            "runtime graph cache pre-warm failed; first daemon boot will "
            "perform a cold rebuild (migrated data is intact)",
            exc_info=True,
        )

    return current_peak_rss


def migrate_sqlite_to_lilli(
    src_db: str | Path,
    dst_root: str | Path,
    *,
    batch: int = 500,
    prune_telemetry_before: Optional[str] = None,
) -> MigrateReport:
    """Copy a SQLite Hippo store verbatim into a FRESH lilli store at ``dst_root``.

    ``src_db`` is the source store's ``brain.sqlite3`` file (opened RO).
    ``dst_root`` is a NEW store root (its lilli schema is auto-created on open) --
    it MUST NOT be the live store.  ``batch`` is the stream/write batch size.
    ``prune_telemetry_before`` (default None = lossless, copy every event) skips
    ``events`` rows with a timestamp strictly older than the given datetime
    string; it never prunes records or edges.

    Returns a MigrateReport with per-table row counts, ``max_vec_label``, elapsed
    seconds, and peak RSS in MiB.
    """
    src_db = Path(src_db).resolve()
    dst_db = _dst_db_path(dst_root).resolve()

    # Same-path guard: never open the real store as both source AND dest.
    if src_db == dst_db:
        raise ValueError("source and destination resolve to the same store")

    # Fresh-dest guard: the docstring contract requires dst_root to be a NEW
    # store root. LILLI_STORAGE_DRIVER=lilli below only governs a fresh
    # (absent) file -- driver selection is on-disk-magic-first, so an
    # already-existing dest file would silently open in whatever format it
    # was written in (most likely stdlib sqlite3, the migration's own source
    # format), and the migrator would report success while writing into the
    # wrong on-disk format.
    if dst_db.exists():
        raise ValueError(
            f"destination store already exists at {dst_db}: "
            "migrate_sqlite_to_lilli requires a FRESH destination root"
        )

    t0 = time.monotonic()
    report = MigrateReport()
    peak_rss = _rss_mb()

    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    src.row_factory = sqlite3.Row

    # Pitfall 2: IAI_MCP_STORE overrides the explicit path arg in _resolve_root,
    # so force it to the dest root around the open and restore it in finally.
    saved_store = os.environ.get("IAI_MCP_STORE")
    saved_driver = os.environ.get("LILLI_STORAGE_DRIVER")
    os.environ["IAI_MCP_STORE"] = str(dst_root)
    os.environ["LILLI_STORAGE_DRIVER"] = "lilli"

    try:
        from iai_mcp.hippo import HippoDB

        # The dest is opened as a raw HippoDB, NOT through MemoryStore, so the
        # boot-time role backfill does not run on the copy here. The role column
        # is copied verbatim from a V6 source (lossless); a pre-V6 source has no
        # role column, so those dest rows land at role=NULL and are backfilled
        # the first time the dest is opened at runtime via MemoryStore.
        dst = HippoDB(dst_root)
        try:
            for table in _ALL_TABLES:
                cols = [
                    r["name"]
                    for r in src.execute(f"PRAGMA table_info({table})").fetchall()
                ]
                if not cols:
                    continue
                cl = ", ".join(cols)
                ph = ", ".join("?" for _ in cols)
                verb = "INSERT OR REPLACE" if table == "_hippo_meta" else "INSERT"
                ins = f"{verb} INTO {table} ({cl}) VALUES ({ph})"

                # Telemetry prune predicate (events only; lossless by default).
                if (
                    table == "events"
                    and prune_telemetry_before is not None
                    and _EVENTS_TS_COLUMN in cols
                ):
                    cur = src.execute(
                        f"SELECT {cl} FROM {table} "
                        f"WHERE {_EVENTS_TS_COLUMN} >= ?",
                        (prune_telemetry_before,),
                    )
                else:
                    cur = src.execute(f"SELECT {cl} FROM {table}")

                copied = 0
                max_vec = 0
                with dst._conn_lock:
                    while True:
                        rows = cur.fetchmany(batch)
                        if not rows:
                            break
                        dst._conn.executemany(
                            ins, [tuple(r[c] for c in cols) for r in rows]
                        )
                        dst._conn.commit()
                        copied += len(rows)
                        if table == "records" and "vec_label" in cols:
                            for r in rows:
                                if r["vec_label"] is not None:
                                    max_vec = max(max_vec, int(r["vec_label"]))
                        sampled = _rss_mb()
                        if sampled > peak_rss:
                            peak_rss = sampled

                report.rows_copied[table] = copied

                # Pitfall 1: bump the engine's vec_label AUTOINCREMENT HWM to
                # max(vec_label) so the first post-migration insert gets a fresh
                # label and cannot collide with a verbatim-copied row. Driven
                # through one engine-agnostic call that both the pure-Python
                # LilliBrainConnection and the Rust engine.Connection expose; on a
                # stdlib-SQLite destination the connection lacks the method (real
                # AUTOINCREMENT already tracks the max), so the bump is skipped.
                if table == "records" and max_vec > 0:
                    report.max_vec_label = max_vec
                    set_hwm = getattr(dst._conn, "set_vec_label_hwm", None)
                    if callable(set_hwm):
                        set_hwm("records", max_vec)

            # Pitfall 3: rebuild the ANN from the copied embeddings before close
            # so recall works on first open (no stale empty index).
            dst._rebuild_index_from_sqlite()
            sampled = _rss_mb()
            if sampled > peak_rss:
                peak_rss = sampled

            # Compact aged-past-TTL tombstoned rows so the fresh store does not
            # start out re-importing the source's stale tombstone bloat. Runs
            # AFTER the ANN rebuild above (not before): the rebuild populates
            # the hnsw label map from the copied vec_labels, and HippoTable's
            # records-table delete path marks the corresponding hnsw label
            # deleted per removed row -- running compaction before the rebuild
            # would try to mark labels deleted in a still-empty index.
            _compact_aged_tombstones_at_finalize(dst, report)
            sampled = _rss_mb()
            if sampled > peak_rss:
                peak_rss = sampled

            # Fingerprint-checked stale-cache invalidation, up front, never
            # lazy. A cutover may copy an entire prior store directory
            # (including a stdlib-era runtime_graph_cache.json) onto the same
            # root a fresh migrate then writes into -- that stale cache's
            # corpus fingerprint would otherwise be discovered only lazily by
            # the recall-adjacent boot probe, which responds with a forced
            # synchronous full-corpus rebuild on the critical path. Checked
            # BEFORE the pre-warm below so a genuinely stale cache is removed
            # first rather than raced against the pre-warm's own fresh write.
            #
            # Same definition MemoryStore.active_records_count() uses (and the
            # same one _runtime_graph_rebuild_needed compares the cache
            # fingerprint against): active = not tombstoned, not pending an
            # embedding. This is NOT the migrator's total-rows count -- a
            # migrated store's total row count can diverge sharply from its
            # live/active count (tombstoned rows are copied verbatim too).
            with dst._conn_lock:
                _active_row = dst._conn.execute(
                    "SELECT COUNT(*) FROM records"
                    " WHERE tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0"
                ).fetchone()
            migrated_active_count = int(_active_row[0]) if _active_row else 0
            (
                report.cache_deleted,
                report.cache_invalidation_reason,
            ) = _invalidate_stale_runtime_graph_cache(
                dst_root, migrated_active_count
            )

            # Pre-warm the runtime graph cache so the first daemon boot on the
            # migrated store takes the warm cache-hit path (no cold rebuild with
            # a centrality child process on every recall during wake-up).
            #
            # Build from the migrated store itself, in the same process, before
            # any further writes — so the cache's payload_record_count matches
            # the migrated active record count exactly and the drift gate does
            # not force a rebuild on the first open.
            #
            # Best-effort / fail-safe: a pre-warm failure (e.g. a degenerate
            # empty corpus, a worker crash, or a missing community library)
            # logs a warning and leaves the migrated DATA intact.  The boot
            # then pays a one-time cold rebuild — correct but slower.
            peak_rss = _prewarm_graph_cache(dst_root, report, peak_rss)

            # Persist the recall-index sidecar LAST, after the pre-warm's
            # stamp-gated boot migrations have settled the engine's column
            # generation -- see _persist_col_index's docstring for why the
            # ordering is load-bearing. Re-uses the still-open raw dst._conn
            # (the pre-warm's own MemoryStore/HippoDB was already closed by
            # _prewarm_graph_cache above; this dst handle is separate and
            # still open).
            _persist_col_index(dst, report)
        finally:
            dst.close()
    finally:
        src.close()
        if saved_store is None:
            os.environ.pop("IAI_MCP_STORE", None)
        else:
            os.environ["IAI_MCP_STORE"] = saved_store
        if saved_driver is None:
            os.environ.pop("LILLI_STORAGE_DRIVER", None)
        else:
            os.environ["LILLI_STORAGE_DRIVER"] = saved_driver

    report.elapsed_sec = time.monotonic() - t0
    report.peak_rss_mb = peak_rss
    return report
