from __future__ import annotations

import logging
from typing import Any, Callable

from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)


_ORPHAN_SWEEP_EDGE_BATCH_SIZE = 2048
_ORPHAN_SWEEP_DELETE_CHUNK_SIZE = 200


def step_hippo_cleanup_noop(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.HIPPO_CLEANUP, 0, interrupt_check,
    ):
        return False, {}
    return True, {"action": "hippo_cleanup_noop"}


def step_hippo_cleanup(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    if self._check_interrupt(
        SleepStep.HIPPO_CLEANUP, 0, interrupt_check,
    ):
        return False, {}

    try:
        return True, _sweep_orphan_edges(self._store)
    except Exception as exc:  # noqa: BLE001 -- hygiene must never fail the step
        logger.warning("orphan_edge_sweep failed: %s", exc, exc_info=True)
        return True, {"action": "orphan_edge_sweep", "error": str(exc)[:200]}


def _read_live_ids(store: Any) -> set[str]:
    from iai_mcp.store import RECORDS_TABLE

    live_ids: set[str] = set()
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            f"SELECT id FROM {RECORDS_TABLE} WHERE tombstoned_at IS NULL",
        ).fetchall()
    for row in rows:
        live_ids.add(str(row["id"]))
    return live_ids


def _sweep_orphan_edges(store: Any) -> dict[str, Any]:
    from iai_mcp.store import EDGES_TABLE, _uuid_literal

    edges_scanned = 0
    skipped_unparseable = 0
    scanned_pairs: list[tuple[str, str]] = []

    # Read edges FIRST, live-ids SECOND (TOCTOU-safe order). The edge scan
    # runs off a read-only snapshot connection opened inside ``to_batches``;
    # by taking the live-id read AFTER the scan completes, any record (with
    # its seed/hebbian edges) committed concurrently during SLEEP is either
    # absent from this edge snapshot (nothing to misjudge) or present in the
    # later live-id read (so its live edges are never flagged orphan).
    edges_q = (
        store.db.open_table(EDGES_TABLE)
        .search()
        .select(["src", "dst"])
    )
    for batch in edges_q.to_batches(batch_size=_ORPHAN_SWEEP_EDGE_BATCH_SIZE):
        for e in batch.to_pylist():
            edges_scanned += 1
            src = e.get("src")
            dst = e.get("dst")
            try:
                src_lit = _uuid_literal(src)
                dst_lit = _uuid_literal(dst)
            except (ValueError, TypeError):
                skipped_unparseable += 1
                continue
            scanned_pairs.append((src_lit, dst_lit))

    live_ids = _read_live_ids(store)
    dead_ids: set[str] = set()
    for src_lit, dst_lit in scanned_pairs:
        if src_lit not in live_ids:
            dead_ids.add(src_lit)
        if dst_lit not in live_ids:
            dead_ids.add(dst_lit)

    orphan_edges_deleted = 0
    if dead_ids:
        # Belt-and-suspenders: re-read live ids immediately before deleting
        # and re-filter, so a record that became live between the first
        # live-id read and the delete is never swept.
        live_ids_recheck = _read_live_ids(store)
        dead_ids -= live_ids_recheck

    if dead_ids:
        # Delete by dead endpoint id: any edge touching a dead record is
        # orphan regardless of its other endpoint or edge_type, so IN lists
        # over dead ids replace per-triple OR chains (each of which forced a
        # full table scan). Two passes cover the union without double
        # counting — the dst pass runs after the src pass's rows are gone.
        tbl = store.db.open_table(EDGES_TABLE)
        ordered = sorted(dead_ids)
        for column in ("src", "dst"):
            for chunk_start in range(0, len(ordered), _ORPHAN_SWEEP_DELETE_CHUNK_SIZE):
                chunk = ordered[chunk_start:chunk_start + _ORPHAN_SWEEP_DELETE_CHUNK_SIZE]
                in_list = ", ".join(f"'{d}'" for d in chunk)
                where = f"{column} IN ({in_list})"
                # Count the rows this predicate actually matches so the metric
                # reflects real deletions, not the number of ids submitted.
                with store.db._conn_lock:
                    matched = store.db._conn.execute(
                        f"SELECT COUNT(*) FROM {EDGES_TABLE} WHERE {where}",
                    ).fetchone()
                matched_n = int(matched[0]) if matched is not None else 0
                if matched_n == 0:
                    continue
                tbl.delete(where=where)
                orphan_edges_deleted += matched_n

    payload: dict[str, Any] = {
        "action": "orphan_edge_sweep",
        "edges_scanned": edges_scanned,
        "orphan_edges_deleted": orphan_edges_deleted,
        "skipped_unparseable": skipped_unparseable,
    }

    if orphan_edges_deleted > 0:
        try:
            from iai_mcp.events import write_event

            write_event(store, "orphan_edge_sweep", payload, severity="info")
        except Exception as exc:  # noqa: BLE001 -- event write must not fail the step
            logger.debug("orphan_edge_sweep event write failed: %s", exc)

    return payload
