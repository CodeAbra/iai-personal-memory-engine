from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable

from iai_mcp.exceptions import StoreError
from iai_mcp.lilli.cycle.sleep_pipeline import SleepStep

logger = logging.getLogger(__name__)

#: Community ids per clear UPDATE: large enough to keep the statement count
#: low, small enough to keep each bound IN-list bounded.
_UPDATE_BATCH = 500

#: Row ids per single-commit transaction of the reassignment updates. Each
#: commit is an F_FULLFSYNC on macOS, so the reassignment batches its
#: distinct-per-row community writes under one commit per chunk, and the
#: chunk boundary is where interrupt checks keep the step deferrable.
_TXN_CHUNK = 2000

_UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)


def step_crisis_recluster(
    self, interrupt_check: Callable[[], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    """Drop the smallest community quartile and re-run Leiden.

    Deferral note: a resumed pass re-runs from the top against the mutated
    landscape and drops another (shrinking) quartile — bounded and
    non-corrupting (community_id is derived data a completed pass fully
    rewrites), but repeated defers erode communities faster than one pass
    would. The drop phase nulls each dropped community's members per batch;
    a record assigned to a dropped community after its batch keeps its
    community_id until the reassignment (or the next pass) rewrites it —
    same derived-data argument, no corruption.
    """
    if self._check_interrupt(
        SleepStep.CRISIS_RECLUSTER, 0, interrupt_check,
    ):
        return False, {}

    state_rec = self._load_state_record()
    if not state_rec.get("crisis_mode", False):
        return True, {"communities_dropped": 0, "crisis_mode": False}

    from iai_mcp.daemon_config import _load_sleep_overhaul_config
    cfg = _load_sleep_overhaul_config()
    drop_quartile = cfg.crisis_drop_quartile
    dry_run = cfg.dry_run

    from iai_mcp.events import write_event
    from iai_mcp.store import RECORDS_TABLE
    tbl = self._store.db.open_table(RECORDS_TABLE)

    # Size communities with a single grouped count instead of materializing the
    # full record corpus. Only community_id is needed here, and the aggregate
    # is O(#communities). Run under the table connection lock, matching the
    # count-rows discipline (every execute/fetch pair guarded).
    sizing_sql = (
        "SELECT community_id, COUNT(*) AS n FROM records "
        "WHERE community_id IS NOT NULL "
        "GROUP BY community_id ORDER BY n ASC, community_id ASC"
    )
    try:
        lock = tbl._db._conn_lock if tbl._db is not None else None
        if lock is not None:
            with lock:
                size_rows = tbl._conn.execute(sizing_sql).fetchall()
        else:
            size_rows = tbl._conn.execute(sizing_sql).fetchall()
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("crisis_recluster sizing query failed: %s", exc)
        size_rows = None

    communities_dropped = 0
    records_reassigned = 0
    new_community_count = 0
    modularity = 0.0
    backend = "flat"
    total_communities = 0

    if size_rows:
        total_communities = len(size_rows)
        n_to_drop = int(total_communities * drop_quartile)
        # Non-UUID community ids never reach a WHERE clause, and the count
        # reports only what a real pass would actually null.
        drop_ids = [
            row[0] for row in size_rows[:n_to_drop]
            if _UUID_RE.fullmatch(str(row[0]))
        ]
        communities_dropped = len(drop_ids)

        if drop_ids and not dry_run:
            # Null each dropped community's members with one bound-parameter
            # IN-UPDATE per batch. The engine serves `community_id IN (?, ...)`
            # through the col-index, so no resolve-then-point-update round trip
            # is needed; each batch commits under one transaction and the batch
            # boundary is where the interrupt check keeps the step deferrable.
            from iai_mcp.hippo import _txn
            chunk_idx = 1
            for i in range(0, len(drop_ids), _UPDATE_BATCH):
                if self._check_interrupt(
                    SleepStep.CRISIS_RECLUSTER, chunk_idx, interrupt_check,
                ):
                    return False, {}
                chunk_idx += 1
                batch = drop_ids[i:i + _UPDATE_BATCH]
                placeholders = ", ".join("?" for _ in batch)
                stmt = (
                    "UPDATE records SET community_id = NULL "
                    f"WHERE community_id IN ({placeholders})"
                )
                params = [str(c) for c in batch]
                lock = tbl._db._conn_lock if tbl._db is not None else None
                try:
                    if lock is not None:
                        with lock:
                            with _txn(tbl._conn):
                                tbl._conn.execute(stmt, params)
                    else:
                        with _txn(tbl._conn):
                            tbl._conn.execute(stmt, params)
                except (OSError, ValueError, RuntimeError, StoreError) as exc:
                    logger.debug("crisis_recluster drop clear failed: %s", exc)
                    continue

        if not dry_run:
            tbl = self._store.db.open_table(RECORDS_TABLE)

            try:
                from iai_mcp.runtime_graph_cache import (
                    compute_assignment_in_child,
                )
                from iai_mcp.graph import MemoryGraph
                from iai_mcp.store import EDGES_TABLE
                import uuid as _uuid

                g = MemoryGraph()
                for row in self._store.iter_record_columns(
                    ["id", "embedding"], batch_size=1024,
                ):
                    try:
                        rid = _uuid.UUID(str(row["id"]))
                        emb = row.get("embedding")
                        emb_list = list(emb) if emb is not None else []
                        g.add_node(rid, None, emb_list)
                    except (ValueError, TypeError, AttributeError):
                        continue

                try:
                    edges_q = (
                        self._store.db.open_table(EDGES_TABLE)
                        .search()
                        .select(["src", "dst", "weight"])
                    )
                    for batch in edges_q.to_batches(batch_size=2048):
                        for e in batch.to_pylist():
                            try:
                                src_u = _uuid.UUID(str(e["src"]))
                                dst_u = _uuid.UUID(str(e["dst"]))
                                g.add_edge(
                                    src_u, dst_u,
                                    weight=float(
                                        e.get("weight", 1.0) or 1.0
                                    ),
                                )
                            except (ValueError, TypeError, KeyError):
                                continue
                except (OSError, ValueError, RuntimeError, StoreError) as exc:
                    logger.debug("crisis_recluster edges query failed: %s", exc)

                _assignment = compute_assignment_in_child(
                    g, prior_mode="cold"
                )
                modularity = float(_assignment.modularity)
                backend = _assignment.backend
                _uuid_to_int: dict[_uuid.UUID, int] = {}
                _next_int = 0
                partition: dict[_uuid.UUID, int] = {}
                for _node_uuid, _comm_uuid in _assignment.node_to_community.items():
                    if _comm_uuid not in _uuid_to_int:
                        _uuid_to_int[_comm_uuid] = _next_int
                        _next_int += 1
                    partition[_node_uuid] = _uuid_to_int[_comm_uuid]
                # Id point-lookup updates in chunked single-commit
                # transactions (see _TXN_CHUNK) — chunks cross community
                # boundaries, so a fragmented partition of thousands of
                # small communities costs the same as a compact one.
                # Interrupt checks between chunks keep the step deferrable —
                # and a FAILING interrupt bookkeeping must defer too, never
                # fall into the broad Leiden handler and report a completed
                # pass over a half-reassigned corpus.
                new_uuids: dict[int, str] = {}
                pairs: list[tuple[str, str]] = []
                for node, lbl in partition.items():
                    if lbl not in new_uuids:
                        new_uuids[lbl] = str(_uuid.uuid4())
                    pairs.append((str(node), new_uuids[lbl]))
                chunk_idx = 1000
                interrupted = False
                for i in range(0, len(pairs), _TXN_CHUNK):
                    try:
                        _stop = self._check_interrupt(
                            SleepStep.CRISIS_RECLUSTER, chunk_idx,
                            interrupt_check,
                        )
                    except Exception:  # noqa: BLE001 -- bookkeeping failure defers
                        logger.warning(
                            "crisis_recluster interrupt bookkeeping "
                            "failed", exc_info=True,
                        )
                        _stop = True
                    if _stop:
                        interrupted = True
                        break
                    chunk_idx += 1
                    chunk = pairs[i:i + _TXN_CHUNK]
                    try:
                        records_reassigned += tbl.update_many_by_id(
                            [(rid, {"community_id": cid}) for rid, cid in chunk]
                        )
                    except (OSError, ValueError, RuntimeError, StoreError):
                        continue
                if interrupted:
                    return False, {}
                new_community_count = len(new_uuids)
            except Exception as exc:  # noqa: BLE001 -- Leiden/graph rebuild
                logger.warning("crisis_recluster Leiden rebuild failed: %s", exc, exc_info=True)

    if not dry_run:
        cleared = self._clear_crisis_mode_via_s2_or_fallback(
            reason="crisis_recluster_complete",
        )
        if not cleared:
            try:
                rec = self._load_state_record()
                rec["crisis_mode"] = False
                rec["crisis_mode_since_ts"] = None
                self._save_state_record(rec)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("crisis_mode clear last-resort write failed: %s", exc)

    write_event(
        self._store,
        "crisis_recluster_pass",
        {
            "communities_dropped": int(communities_dropped),
            "records_reassigned": int(records_reassigned),
            "new_community_count": int(new_community_count),
            "modularity": float(modularity),
            "backend": str(backend),
            "dry_run_mode": bool(dry_run),
        },
        severity="warning" if communities_dropped > 0 else "info",
    )

    return True, {
        "total_communities": int(total_communities),
        "communities_dropped": int(communities_dropped),
        "dry_run": bool(dry_run),
    }
