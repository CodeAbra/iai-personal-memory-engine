from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def step_compact_hippo(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    from iai_mcp.maintenance import optimize_hippo_storage

    if self._check_interrupt(
        SleepStep.OPTIMIZE_HIPPO, 0, interrupt_check,
    ):
        return False, {}

    compact_t0 = time.monotonic()
    report = optimize_hippo_storage(self._store)
    tables_with_errors = [
        t for t, r in (report or {}).items()
        if isinstance(r, dict) and "error" in r
    ]

    from iai_mcp.daemon_config import _load_erasure_config
    cfg = _load_erasure_config()
    ttl_sec = cfg.tombstone_ttl_sec

    now = self._now()
    drop_cutoff = now - timedelta(seconds=ttl_sec)

    from iai_mcp.store import RECORDS_TABLE
    from iai_mcp.events import write_event

    tbl = self._store.db.open_table(RECORDS_TABLE)
    untomb_where = (
        "tombstoned_at IS NOT NULL "
        "AND (pinned = true OR never_decay = true)"
    )
    try:
        count_untombstoned = int(tbl.count_rows(filter=untomb_where))
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("compact_hippo untombstone count failed: %s", exc)
        count_untombstoned = 0
    if count_untombstoned > 0:
        try:
            tbl.update(
                where=untomb_where,
                values={"tombstoned_at": None, "live": 1},
            )
        except (OSError, ValueError, RuntimeError, StoreError) as exc:
            logger.debug("compact_hippo untombstone update failed: %s", exc)
            count_untombstoned = 0
        else:
            # Rows have been restored from tombstoned back to active: active count
            # increased.  Invalidate the cached active count so the next
            # active_records_count() recomputes from the post-untombstone SQL state.
            try:
                _inv = getattr(self._store, "_invalidate_corpus_count", None)
                if _inv is not None:
                    _inv("active")
            except Exception:  # noqa: BLE001 -- invalidation must not crash a sleep step
                pass
            # Rows just returned from tombstoned to active: the resident
            # exact-cosine matrix's warm snapshot predates the restore, so
            # invalidate it and let the next exact_top_k call rebuild from
            # the post-restore SQL state so the restored rows are recallable
            # again.
            try:
                _inv_x = getattr(self._store, "invalidate_exact_index", None)
                if callable(_inv_x):
                    _inv_x()
            except Exception:  # noqa: BLE001 -- invalidation must not crash a sleep step
                pass

    # The drop below removes rows whose tombstoned_at has aged out: those rows were
    # already excluded from the active count by the tombstoned_at predicate, so their
    # removal does NOT change the active count.  No invalidation is needed for the
    # drop path.
    tbl = self._store.db.open_table(RECORDS_TABLE)
    drop_cutoff_str = drop_cutoff.strftime("%Y-%m-%d %H:%M:%S")
    drop_where = (
        "tombstoned_at IS NOT NULL "
        f"AND tombstoned_at < '{drop_cutoff_str}'"
    )
    from iai_mcp.maintenance import batched_tombstone_drop

    def _drop_interrupt_check() -> bool:
        return bool(
            self._check_interrupt(SleepStep.OPTIMIZE_HIPPO, 0, interrupt_check),
        )

    try:
        count_dropped = batched_tombstone_drop(
            tbl, drop_where, interrupt_check=_drop_interrupt_check,
        )
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("compact_hippo batched drop failed: %s", exc)
        count_dropped = 0

    # The untombstone above flips pinned/never_decay rows back to live in
    # SQLite, but a plain UPDATE does not touch the ANN index or its label
    # map — the storage rebuild that ran at the start of this step measured
    # the corpus BEFORE the untombstone, so the freshly-restored rows are
    # present in SQLite yet absent from the recall index. Reconcile the index
    # to SQLite's final tombstone state so a restored pinned record is
    # recallable again. This reuses the canonical double-buffered rebuild
    # (atomic buffer swap + label-map repopulation under the recall lock), so
    # the swap semantics are unchanged. Only runs when an untombstone actually
    # happened; the drop path already removes its rows from the index.
    if count_untombstoned > 0:
        db = getattr(self._store, "db", None)
        rebuild = getattr(db, "_rebuild_index_from_sqlite", None)
        if callable(rebuild):
            try:
                rebuild()
            except (OSError, ValueError, RuntimeError, StoreError) as exc:
                logger.debug(
                    "compact_hippo post-untombstone index rebuild failed: %s", exc,
                )

    try:
        write_event(
            self._store,
            "erasure_optimize_drops",
            {
                "count_dropped": int(count_dropped),
                "count_untombstoned": int(count_untombstoned),
                "ts": now.isoformat(),
            },
            severity="info",
        )
    except (OSError, ValueError, StoreError) as exc:
        logger.debug("best-effort erasure_optimize_drops event failed: %s", exc)

    elapsed = round(time.monotonic() - compact_t0, 3)
    try:
        write_event(
            self._store,
            "hippo_compacted",
            {
                "phase": "sleep_cycle",
                "per_table": report,
                "total_elapsed_sec": elapsed,
            },
            severity="info",
        )
    except Exception:  # noqa: BLE001
        logger.debug("hippo_compacted event emit failed", exc_info=True)

    return True, {
        "tables_optimized": list((report or {}).keys()),
        "tables_with_errors": tables_with_errors,
        "count_dropped_by_erasure": int(count_dropped),
        "count_untombstoned_by_pin_override": int(count_untombstoned),
    }


def step_optimize_hippo(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    return self._step_compact_hippo(interrupt_check)
