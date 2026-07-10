from __future__ import annotations

import json
import logging
import os

import numpy as np

from iai_mcp.exceptions import StoreError
from iai_mcp.lifecycle_state import _utc_now_iso

logger = logging.getLogger(__name__)


def clear_crisis_mode_via_s2_or_fallback(self, *, reason: str) -> bool:
    s2 = getattr(self, "_s2_coordinator", None)
    loop = getattr(self, "_loop", None)
    if s2 is None:
        return False
    try:
        import asyncio
        coro = s2.set_crisis_mode(False, reason)
        if loop is not None and loop.is_running():
            fut = asyncio.run_coroutine_threadsafe(coro, loop)
            fut.result(timeout=5.0)
        else:
            asyncio.run(coro)
        return True
    except (OSError, RuntimeError, TimeoutError) as exc:
        logger.debug("S2 clear_crisis_mode failed, falling back: %s", exc)
        return False


def set_crisis_mode_via_s2_or_fallback(
    self, *, value: bool, reason: str,
) -> bool:
    s2 = getattr(self, "_s2_coordinator", None)
    loop = getattr(self, "_loop", None)
    if s2 is not None:
        try:
            import asyncio
            coro = s2.set_crisis_mode(value, reason)
            if loop is not None and loop.is_running():
                fut = asyncio.run_coroutine_threadsafe(coro, loop)
                fut.result(timeout=5.0)
            else:
                asyncio.run(coro)
            return True
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("S2 set_crisis_mode failed, falling back: %s", exc)
    try:
        rec = self._load_state_record()
        rec["crisis_mode"] = bool(value)
        rec["crisis_mode_since_ts"] = _utc_now_iso() if bool(value) else None
        self._save_state_record(rec)
        return False
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("crisis_mode fallback save_state failed: %s", exc)
        return False


def run_essential_variable_tracker_hook(self) -> None:
    from iai_mcp.daemon_config import _load_sleep_overhaul_config
    from iai_mcp.ashby_step import (
        CRISIS_GATING_VARIABLES,
        EssentialVariableTracker,
        TopologySnapshot,
        decide_crisis_transition,
    )
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.events import write_event
    from iai_mcp.store import RECORDS_TABLE, EDGES_TABLE

    cfg = _load_sleep_overhaul_config()
    dry_run = cfg.dry_run

    try:
        recs = (
            self._store.db.open_table(RECORDS_TABLE)
            .search().to_pandas()
        )
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("essential_variable_tracker records query failed: %s", exc)
        return
    if recs.empty:
        return
    # Tombstoned records are dead memories, not graph nodes -- counting them
    # inflates total_nodes against a numerator (live edges) that can never
    # reference them once orphan_edge_sweep runs (HIPPO_CLEANUP), which
    # silently deflates avg_degree and skews giant_component_fraction. Every
    # topology variable in this hook (new and pre-existing) should read the
    # same live graph the orphan sweep maintains.
    if "tombstoned_at" in recs.columns:
        recs = recs[recs["tombstoned_at"].isna()]
    if recs.empty:
        return

    import uuid as _uuid
    g = MemoryGraph()
    community_ids: set = set()
    _community_embeddings: dict[str, list[list[float]]] = {}
    live_node_labels: set[str] = set()
    for _, row in recs.iterrows():
        try:
            rid = _uuid.UUID(str(row["id"]))
            emb = row.get("embedding")
            emb_list = list(emb) if emb is not None else []
            cid_raw = row.get("community_id")
            cid_uuid: _uuid.UUID | None
            if cid_raw is not None:
                try:
                    cid_uuid = _uuid.UUID(str(cid_raw))
                    _cid_str = str(cid_uuid)
                    community_ids.add(_cid_str)
                    if emb_list:
                        _community_embeddings.setdefault(
                            _cid_str, []
                        ).append(emb_list)
                except (ValueError, TypeError):
                    cid_uuid = None
            else:
                cid_uuid = None
            g.add_node(rid, cid_uuid, emb_list)
            live_node_labels.add(str(rid))
        except (ValueError, TypeError, AttributeError):
            continue

    try:
        edges_df = (
            self._store.db.open_table(EDGES_TABLE).search().to_pandas()
        )
        for _, e in edges_df.iterrows():
            try:
                src_s = str(e["src"])
                dst_s = str(e["dst"])
                # An edge referencing a tombstoned/missing record must not
                # fabricate a phantom node -- add_edge() would otherwise
                # silently create adjacency entries for ids that were never
                # add_node()'d, re-inflating total_nodes exactly like the
                # tombstone filter above was meant to prevent. HIPPO_CLEANUP
                # sweeps these edges away periodically; until that sweep
                # runs, the hook itself must not count them.
                if src_s not in live_node_labels or dst_s not in live_node_labels:
                    continue
                src_u = _uuid.UUID(src_s)
                dst_u = _uuid.UUID(dst_s)
                g.add_edge(
                    src_u, dst_u,
                    weight=float(e.get("weight", 1.0) or 1.0),
                )
            except (ValueError, TypeError, KeyError):
                continue
    except (OSError, ValueError, RuntimeError, StoreError) as exc:
        logger.debug("essential_variable_tracker edges query failed: %s", exc)

    total_nodes = g.node_count()
    if total_nodes == 0:
        return

    try:
        rc_ratio = g.rich_club_coefficient()
    except (ValueError, RuntimeError, ZeroDivisionError) as exc:
        logger.debug("rich_club_coefficient failed: %s", exc)
        rc_ratio = 0.0
    # self-loops (if any survive upstream filtering) are counted twice here,
    # same as the pre-existing edge_density numerator -- kept consistent so
    # avg_degree and edge_density agree on what an "edge" is.
    nedges = sum(1 for _ in g.iter_edges_with_weight())
    edge_density = (
        (2.0 * nedges) / (total_nodes * (total_nodes - 1))
        if total_nodes >= 2 else 0.0
    )
    avg_degree = (2.0 * nedges / total_nodes) if total_nodes else 0.0

    try:
        indptr, indices, _data = g.to_csr_arrays()
        # non_isolated must be computed over the SAME edge set that feeds
        # the component search below -- to_csr_arrays() drops self-loops (a
        # node linked only to itself has no real connectivity), so counting
        # degree via g.degrees() (which does count a self-loop as +1 degree)
        # would call a self-loop-only node "non-isolated" while the
        # component search correctly treats it as its own singleton island.
        # Deriving non_isolated from the CSR row pointers keeps both the
        # numerator and denominator of giant_component_fraction defined over
        # the identical connectivity graph.
        non_isolated = sum(
            1 for i in range(total_nodes) if indptr[i + 1] > indptr[i]
        )
        try:
            from iai_mcp_native.graph import connected_components as _native_cc
            components = _native_cc(indptr, indices, total_nodes)
            giant = max((len(c) for c in components), default=0)
        except (ImportError, AttributeError) as exc:
            logger.debug(
                "native connected_components unavailable, using scipy "
                "fallback: %s", exc,
            )
            import scipy.sparse as _sp
            import scipy.sparse.csgraph as _csgraph
            csr = _sp.csr_matrix(
                (
                    np.ones(len(indices), dtype=np.float64)
                    if len(indices)
                    else np.zeros(0, dtype=np.float64),
                    indices,
                    indptr,
                ),
                shape=(total_nodes, total_nodes),
            )
            n_comp, labels = _csgraph.connected_components(
                csr, directed=False,
            )
            if n_comp == 0:
                giant = 0
            else:
                counts = np.bincount(labels, minlength=n_comp)
                giant = int(counts.max())
        # Vacuously healthy when there are no edges at all -- a graph with
        # zero non-isolated nodes has no fragmentation to measure; the
        # avg_degree floor is what catches the edgeless-collapse case.
        giant_component_fraction = (
            (giant / non_isolated) if non_isolated else 1.0
        )
    except (ValueError, RuntimeError, ImportError) as exc:
        # Fail-open: a broken component sensor must never arm crisis. But a
        # PERSISTENT failure silently halves the gating surface, so surface it
        # at warning level to keep it observable in prod logs.
        logger.warning(
            "giant_component_fraction computation failed, "
            "defaulting fail-healthy (1.0): %s",
            exc,
        )
        giant_component_fraction = 1.0

    snapshot = TopologySnapshot(
        rich_club_ratio=float(rc_ratio),
        community_count=int(len(community_ids)),
        edge_density=float(edge_density),
        total_nodes=int(total_nodes),
        avg_degree=float(avg_degree),
        giant_component_fraction=float(giant_component_fraction),
    )
    tracker = EssentialVariableTracker(cfg)
    breaches = tracker.check(snapshot)

    any_gating_breach = any(
        breaches.get(var_name) is not None
        for var_name in CRISIS_GATING_VARIABLES
    )

    state_record = self._load_state_record()
    consecutive_breaches = int(
        state_record.get("essential_variable_consecutive_breaches", 0)
    )
    consecutive_clears = int(
        state_record.get("essential_variable_consecutive_clears", 0)
    )
    decision, new_consecutive_breaches, new_consecutive_clears = (
        decide_crisis_transition(
            any_gating_breach,
            consecutive_breaches,
            consecutive_clears,
            arm_after_n=cfg.ev_arm_after_n,
            disarm_after_n=cfg.ev_disarm_after_n,
        )
    )

    if not dry_run:
        # Persist the hysteresis counters BEFORE any crisis-mode write: the
        # crisis writer (_set_crisis_mode_via_s2_or_fallback fallback path)
        # does its own load-modify-save of the same lifecycle_state.json, so
        # writing counters after it would clobber the counters this cycle
        # just computed with whatever crisis_mode-only view the writer saved.
        state_record["essential_variable_consecutive_breaches"] = (
            new_consecutive_breaches
        )
        state_record["essential_variable_consecutive_clears"] = (
            new_consecutive_clears
        )
        self._save_state_record(state_record)

    crisis_mode_set = False
    if decision is True and not dry_run:
        first_gating_var = next(
            (
                var_name
                for var_name in breaches
                if var_name in CRISIS_GATING_VARIABLES
                and breaches[var_name] is not None
            ),
            "unknown",
        )
        self._set_crisis_mode_via_s2_or_fallback(
            value=True,
            reason=f"essential_variable_breach:{first_gating_var}",
        )
        crisis_mode_set = True
    elif decision is False and not dry_run:
        current_record = self._load_state_record()
        if bool(current_record.get("crisis_mode", False)):
            self._set_crisis_mode_via_s2_or_fallback(
                value=False,
                reason="essential_variable_recovered",
            )

    for var_name, breach in breaches.items():
        if breach is None:
            continue
        gates_crisis = var_name in CRISIS_GATING_VARIABLES
        write_event(
            self._store,
            "essential_variable_breach",
            {
                "variable_name": str(var_name),
                "observed_value": float(breach.observed_value),
                "threshold": float(breach.threshold),
                "direction": str(breach.direction),
                "total_nodes": int(total_nodes),
                "crisis_mode_set": bool(crisis_mode_set and not dry_run),
                "dry_run_mode": bool(dry_run),
                "gates_crisis": bool(gates_crisis),
            },
            severity="warning" if gates_crisis else "info",
        )

    if os.environ.get(
        "IAI_MCP_ORTHO_ENABLED", "",
    ).lower() in {"1", "true"}:
        try:
            from iai_mcp.pattern_separation import detect_hubness
            if _community_embeddings:
                _largest_cid = max(
                    _community_embeddings,
                    key=lambda k: len(_community_embeddings[k]),
                )
                _largest = _community_embeddings[_largest_cid][:100]
                if len(_largest) >= 2:
                    _hubness = detect_hubness(_largest, threshold=0.85)
                    write_event(
                        self._store,
                        "community_hubness_diagnostic",
                        {
                            "community_id": _largest_cid,
                            "mean_similarity": float(
                                _hubness.get("mean_similarity", 0.0)
                            ),
                            "max_similarity": float(
                                _hubness.get("max_similarity", 0.0)
                            ),
                            "is_hub": bool(_hubness.get("is_hub", False)),
                            "size": int(_hubness.get("size", 0)),
                        },
                        severity="info",
                    )
        except Exception as _hub_exc:  # noqa: BLE001 -- diagnostic MUST NOT crash sleep
            logger.debug(
                "detect_hubness diagnostic skipped: %s",
                str(_hub_exc)[:120],
            )
