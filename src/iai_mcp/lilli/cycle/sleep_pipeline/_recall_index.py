from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Any, Callable

from iai_mcp.lifecycle_event_log import _LIVENESS_SPEC_VERSION
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


def _prewarm_recall_read_model(
    self,
    result: dict[str, Any],
    interrupt_check: Callable[[], bool] | None,
) -> None:
    """Pay the post-rebuild decode and graph build HERE, not on the next recall.

    The rebuild above rewrote the graph-cache snapshot, so the store's decoded
    structural memos and the warm graph bundle all miss on their next
    consultation — a cost the first recall after every cycle otherwise pays
    (whole-map UUID decode plus a full-corpus graph stream). Running the same
    memoizing loaders from the write side leaves the read side a pure memo
    hit. Best-effort: a failure can never fail the step, and recall's own
    lazy paths remain the always-correct fallback.

    Every phase yields to the foreground FIRST: prewarm is pure acceleration,
    and its heavy stretches hold the interpreter while a live read may be
    waiting (worst at boot, when the catch-up cycle overlaps the ANN rebuild
    and every cache is cold — measured to starve the socket outright). A
    skipped phase costs one lazy warm-up on some later recall; a phase that
    refuses to yield costs EVERY concurrent read its latency budget.
    """
    def _foreground_waiting() -> bool:
        return bool(interrupt_check and interrupt_check())

    t0 = time.monotonic()
    try:
        if _foreground_waiting():
            result["structural_prewarm_skipped"] = "foreground"
        else:
            from iai_mcp import runtime_graph_cache

            runtime_graph_cache.load_recall_structural(self._store)
            result["structural_prewarm_elapsed_sec"] = round(time.monotonic() - t0, 3)
    except Exception as exc:  # noqa: BLE001 -- prewarm must not break the step
        logger.warning("structural_prewarm_failed: %s", exc, exc_info=True)
        result["structural_prewarm_error"] = str(exc)[:200]
    t1 = time.monotonic()
    try:
        if _foreground_waiting():
            result["bundle_prewarm_skipped"] = "foreground"
        else:
            from iai_mcp import retrieve

            # Drop the memo first: the rebuild above just rewrote the graph
            # snapshot, so a still-inside-the-fuse memo would ride the
            # stale-while-revalidate branch — build_runtime_graph would
            # return the stale bundle and kick the real build ASYNC, after
            # this step, concurrent with awake recalls: the exact cost this
            # prewarm exists to front-load.
            self._store._warm_graph_bundle = None
            retrieve.build_runtime_graph(self._store)
            result["bundle_prewarm_elapsed_sec"] = round(time.monotonic() - t1, 3)
    except Exception as exc:  # noqa: BLE001 -- prewarm must not break the step
        logger.warning("bundle_prewarm_failed: %s", exc, exc_info=True)
        result["bundle_prewarm_error"] = str(exc)[:200]
    # The optimize/erasure steps earlier in this cycle invalidate the exact
    # authority matrix; this step runs last, so a cold matrix here means no
    # later step will touch it again — rebuild it now and the recall path
    # keeps its exact-authority lane instead of degrading until some future
    # warm-up probe pays the whole-corpus scan.
    t2 = time.monotonic()
    try:
        exact = getattr(self._store, "_exact_index", None)
        if exact is not None and not exact.is_warm:
            if _foreground_waiting():
                result["exact_prewarm_skipped"] = "foreground"
            else:
                self._store._build_exact_index_sync()
                result["exact_prewarm_elapsed_sec"] = round(time.monotonic() - t2, 3)
    except Exception as exc:  # noqa: BLE001 -- prewarm must not break the step
        logger.warning("exact_prewarm_failed: %s", exc, exc_info=True)
        result["exact_prewarm_error"] = str(exc)[:200]


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
        _prewarm_recall_read_model(self, result, interrupt_check)
        try:
            # Full lexical rebuild: reconciles tombstones and updated
            # surfaces the cheap per-insert feed cannot; from here the feed
            # keeps recall's warm BM25 lane current until the next cycle.
            self._store.lexical_search("nightly lexical warm-up", k=1)
            result["lexical_index_warm"] = True
        except Exception as lex_exc:  # noqa: BLE001 -- warm-up is best-effort
            logger.debug("lexical warm-up failed: %s", lex_exc)

        try:
            self._event_log.append({
                "event": "promise_liveness",
                "promise": "warm_bm25_nightly",
                "liveness_candidates": 1,
                "liveness_processed": 1 if result.get("lexical_index_warm") else 0,
                "liveness_spec_version": _LIVENESS_SPEC_VERSION,
            })
        except (OSError, ValueError) as exc:
            logger.debug("best-effort promise_liveness event failed: %s", exc)

        return True, result

    except Exception as exc:  # noqa: BLE001 -- step must not crash the pipeline
        logger.warning(
            "recall_index_rebuild step failed: %s", exc, exc_info=True,
        )
        return True, {"error": str(exc)[:200], "rebuilt": False}
