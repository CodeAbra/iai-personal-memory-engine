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


def _orphan_triple_clause(src_lit: str, dst_lit: str, edge_type: Any) -> str:
    # ``edge_type`` is TEXT NOT NULL under the canonical schema, but a
    # permissive driver can still surface a NULL, in which case a
    # ``= 'None'`` predicate would match nothing and leave the row forever.
    # Emit an ``IS NULL`` test for that case so the row is actually swept.
    if edge_type is None:
        return f"(src = '{src_lit}' AND dst = '{dst_lit}' AND edge_type IS NULL)"
    et = str(edge_type).replace("'", "''")
    return f"(src = '{src_lit}' AND dst = '{dst_lit}' AND edge_type = '{et}')"


def _sweep_orphan_edges(store: Any) -> dict[str, Any]:
    from iai_mcp.store import EDGES_TABLE, _uuid_literal

    edges_scanned = 0
    skipped_unparseable = 0
    scanned_triples: list[tuple[str, str, Any]] = []

    # Read edges FIRST, live-ids SECOND (TOCTOU-safe order). The edge scan
    # runs off a read-only snapshot connection opened inside ``to_batches``;
    # by taking the live-id read AFTER the scan completes, any record (with
    # its seed/hebbian edges) committed concurrently during SLEEP is either
    # absent from this edge snapshot (nothing to misjudge) or present in the
    # later live-id read (so its live edges are never flagged orphan).
    edges_q = (
        store.db.open_table(EDGES_TABLE)
        .search()
        .select(["src", "dst", "edge_type"])
    )
    for batch in edges_q.to_batches(batch_size=_ORPHAN_SWEEP_EDGE_BATCH_SIZE):
        for e in batch.to_pylist():
            edges_scanned += 1
            src = e.get("src")
            dst = e.get("dst")
            edge_type = e.get("edge_type")
            try:
                src_lit = _uuid_literal(src)
                dst_lit = _uuid_literal(dst)
            except (ValueError, TypeError):
                skipped_unparseable += 1
                continue
            scanned_triples.append((src_lit, dst_lit, edge_type))

    live_ids = _read_live_ids(store)
    orphan_triples: list[tuple[str, str, Any]] = [
        (src_lit, dst_lit, edge_type)
        for src_lit, dst_lit, edge_type in scanned_triples
        if src_lit not in live_ids or dst_lit not in live_ids
    ]

    orphan_edges_deleted = 0
    if orphan_triples:
        # Belt-and-suspenders: re-read live ids immediately before deleting
        # and re-filter, so a record that became live between the first
        # live-id read and the delete is never swept.
        live_ids_recheck = _read_live_ids(store)
        orphan_triples = [
            (src_lit, dst_lit, edge_type)
            for src_lit, dst_lit, edge_type in orphan_triples
            if src_lit not in live_ids_recheck or dst_lit not in live_ids_recheck
        ]

    if orphan_triples:
        tbl = store.db.open_table(EDGES_TABLE)
        for chunk_start in range(0, len(orphan_triples), _ORPHAN_SWEEP_DELETE_CHUNK_SIZE):
            chunk = orphan_triples[chunk_start:chunk_start + _ORPHAN_SWEEP_DELETE_CHUNK_SIZE]
            where = " OR ".join(
                _orphan_triple_clause(src_lit, dst_lit, edge_type)
                for src_lit, dst_lit, edge_type in chunk
            )
            # Count the rows this predicate actually matches so the metric
            # reflects real deletions, not the number of triples submitted.
            with store.db._conn_lock:
                matched = store.db._conn.execute(
                    f"SELECT COUNT(*) FROM {EDGES_TABLE} WHERE {where}",
                ).fetchone()
            matched_n = int(matched[0]) if matched is not None else 0
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
