from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


def _persist_col_indexes(self, result: dict[str, Any]) -> None:
    """Re-emit the storage engine's persisted column-index sidecar.

    Runs as the final act of the nightly rebuild pass, after every earlier
    step has finished mutating indexed columns, so the on-disk sidecar
    reflects fully-consolidated state and a cold process can load it instead
    of paying a full-scan rebuild on its first indexed read. Lilli-only; a
    failure here is logged and reported but can never fail the step or
    discard the graph-cache outcome already computed above.
    """
    db = getattr(self._store, "db", None)
    if db is None or getattr(db, "_storage_driver", "stdlib") != "lilli":
        return
    conn = getattr(db, "_conn", None)
    if conn is None or not hasattr(conn, "persist_col_indexes"):
        return

    lock = getattr(db, "_conn_lock", None)
    lock_ctx = lock if lock is not None else nullcontext()
    t0 = time.monotonic()
    try:
        with lock_ctx:
            conn.persist_col_indexes()
        result["colindex_persist_elapsed_sec"] = round(time.monotonic() - t0, 3)
    except Exception as exc:  # noqa: BLE001 -- persist must not break the step
        logger.warning("colindex_persist_failed: %s", exc, exc_info=True)
        result["colindex_persist_error"] = str(exc)[:200]


def _recycle_ro_pool(self, result: dict[str, Any]) -> None:
    """Recycle the RO reader pool as part of the nightly rebuild pass.

    Piggybacks the existing RECALL_INDEX_REBUILD cadence rather than
    inventing a new timer — the pool's staleness bound is otherwise only
    "next write", which is fine for correctness but lets long-idle slots
    accumulate parse/col-cache drift over a long WAKE window. Lilli-only; a
    recycle failure is logged and reported but can never fail the step or
    discard the graph-cache/persist-index outcome already computed above.
    """
    db = getattr(self._store, "db", None)
    if db is None or getattr(db, "_storage_driver", "stdlib") != "lilli":
        return
    pool = getattr(db, "_ro_pool", None)
    if pool is None:
        return
    t0 = time.monotonic()
    try:
        pool.recycle()
        result["ro_pool_recycle_elapsed_sec"] = round(time.monotonic() - t0, 3)
    except Exception as exc:  # noqa: BLE001 -- recycle must not break the step
        logger.warning("ro_pool_recycle_failed: %s", exc, exc_info=True)
        result["ro_pool_recycle_error"] = str(exc)[:200]


def step_recall_index_rebuild(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(SleepStep.RECALL_INDEX_REBUILD, 0, interrupt_check):
        return False, {}

    try:
        from iai_mcp import runtime_graph_cache

        result = runtime_graph_cache._rebuild_and_save_rgc(self._store)
        _persist_col_indexes(self, result)
        _recycle_ro_pool(self, result)
        return True, result

    except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
        logger.warning(
            "recall_index_rebuild step failed: %s", exc, exc_info=True,
        )
        return True, {"error": str(exc)[:200], "rebuilt": False}
