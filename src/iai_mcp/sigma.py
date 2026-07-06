from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

import numpy as np

from iai_mcp.events import write_event
from iai_mcp_native import graph as lilli_graph

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore


SIGMA_N_FLOOR: int = 200

# Defensive upper bound on graph size for the sigma (small-worldness) audit.
# fast_sigma() runs average_clustering + all-pairs-shortest-path on the largest
# component AND on n_random reference graphs of the same order; that cost is
# roughly O(n_random * (V*E + V^2)) and is unbounded as the live graph grows.
# Above the cap we return None: the regime degrades cleanly to "insufficient_data"
# (sigma None is already handled everywhere) and the tick stays bounded -- the
# watchdog does NOT kill on CPU, so an unbounded sigma compute would spin a core
# uncontained. The default is far above the current live store (~4k nodes) so it
# never trips today; both bounds are env-overridable.
SIGMA_N_CEIL: int = 20000


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


SIGMA_MID_LIFE_THRESHOLD: int = 500

SIGMA_OBSERVATION_KIND: str = "sigma_observation"
SIGMA_DRIFT_KIND: str = "sigma_drift"

HEBBIAN_DEVELOPMENTAL_BOOST_FACTOR: float = 1.3
HEBBIAN_DEVELOPMENTAL_BOOST_TTL_SESSIONS: int = 5

HEBBIAN_RATE_KNOB: str = "hebbian_rate"


def _build_csr_from_edge_lists(
    u_list, v_list, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import scipy.sparse

    m = len(u_list)
    if m == 0 or n_nodes == 0:
        return (
            np.zeros(max(n_nodes + 1, 1), dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
        )
    rows = np.concatenate(
        [
            np.asarray(u_list, dtype=np.int64),
            np.asarray(v_list, dtype=np.int64),
        ]
    )
    cols = np.concatenate(
        [
            np.asarray(v_list, dtype=np.int64),
            np.asarray(u_list, dtype=np.int64),
        ]
    )
    data = np.ones(2 * m, dtype=np.float64)
    coo = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    csr = coo.tocsr()
    return (
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.float64),
    )


def _induced_csr_from_component(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    component_nodes: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    old_to_new = {old: new for new, old in enumerate(component_nodes)}
    node_set = set(component_nodes)
    sub_n = len(component_nodes)
    if sub_n == 0:
        return (
            np.zeros(1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            0,
        )
    sub_rows: list[int] = []
    sub_cols: list[int] = []
    sub_data: list[float] = []
    for new_u, old_u in enumerate(component_nodes):
        start = int(indptr[old_u])
        end = int(indptr[old_u + 1])
        for idx in range(start, end):
            old_v = int(indices[idx])
            if old_v in node_set:
                new_v = old_to_new[old_v]
                sub_rows.append(new_u)
                sub_cols.append(new_v)
                sub_data.append(
                    float(data[idx]) if data is not None and len(data) > idx else 1.0
                )
    if not sub_rows:
        return (
            np.zeros(sub_n + 1, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.float64),
            sub_n,
        )
    import scipy.sparse

    coo = scipy.sparse.coo_matrix(
        (sub_data, (sub_rows, sub_cols)),
        shape=(sub_n, sub_n),
    )
    csr = coo.tocsr()
    return (
        csr.indptr.astype(np.int64),
        csr.indices.astype(np.int64),
        csr.data.astype(np.float64),
        sub_n,
    )


def _largest_component_csr(
    indptr: np.ndarray, indices: np.ndarray, data: np.ndarray, n_nodes: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    if n_nodes == 0:
        return indptr, indices, data, 0
    components = lilli_graph.connected_components(indptr, indices, n_nodes)
    if not components:
        return indptr, indices, data, n_nodes
    if len(components) == 1:
        return indptr, indices, data, n_nodes
    largest = max(components, key=len)
    return _induced_csr_from_component(indptr, indices, data, list(largest))


def fast_sigma(
    graph: "MemoryGraph",
    *,
    n_random: int = 3,
    seed: int = 42,
) -> tuple[float, float, float, float, float]:
    indptr, indices, data = graph.to_csr_arrays()
    n_nodes = len(indptr) - 1
    if n_nodes < 2 or len(indices) == 0:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)

    sub_indptr, sub_indices, sub_data, n = _largest_component_csr(
        indptr, indices, data, n_nodes
    )
    m = int(len(sub_indices) // 2)
    if n < 2 or m == 0:
        return (float("nan"), 0.0, 0.0, 0.0, 0.0)

    C = float(lilli_graph.average_clustering(sub_indptr, sub_indices, n))
    L = float(lilli_graph.average_shortest_path_length(sub_indptr, sub_indices, n))

    Cs: list[float] = []
    Ls: list[float] = []
    for k in range(max(1, n_random)):
        u_list, v_list = lilli_graph.gnm_random_graph(n, m, seed=seed + k)
        ref_indptr, ref_indices, ref_data = _build_csr_from_edge_lists(
            u_list, v_list, n
        )
        ref_n = n
        try:
            ref_connected = lilli_graph.is_connected(ref_indptr, ref_indices, ref_n)
        except ValueError:
            continue
        if not ref_connected:
            ref_indptr, ref_indices, ref_data, ref_n = _largest_component_csr(
                ref_indptr, ref_indices, ref_data, ref_n
            )
        if ref_n < 2 or len(ref_indices) == 0:
            continue
        Cs.append(
            float(lilli_graph.average_clustering(ref_indptr, ref_indices, ref_n))
        )
        Ls.append(
            float(
                lilli_graph.average_shortest_path_length(
                    ref_indptr, ref_indices, ref_n
                )
            )
        )

    if not Cs or not Ls:
        return (float("nan"), C, L, 0.0, 0.0)

    Cr = sum(Cs) / len(Cs)
    Lr = sum(Ls) / len(Ls)
    if Cr <= 0 or Lr <= 0 or L <= 0:
        return (float("nan"), C, L, Cr, Lr)

    sigma_val = (C / Cr) / (L / Lr)
    return (sigma_val, C, L, Cr, Lr)


def compute_sigma(graph: "MemoryGraph", *, seed: int = 42) -> Optional[float]:
    n = graph.node_count()
    floor = _env_int("IAI_MCP_SIGMA_N_FLOOR", SIGMA_N_FLOOR)
    ceil = _env_int("IAI_MCP_SIGMA_N_CEIL", SIGMA_N_CEIL)
    if n < floor:
        return None
    if n > ceil:
        # Above the defensive cap: skip the (unbounded) small-worldness compute
        # so a single tick can never spin a core for minutes on a pathologically
        # large graph. Reported downstream as "insufficient_data".
        logger.warning(
            "sigma_skipped_above_ceiling",
            extra={"node_count": int(n), "ceil": int(ceil)},
        )
        return None
    sigma_val, *_ = fast_sigma(graph, seed=seed)
    if isinstance(sigma_val, float) and math.isnan(sigma_val):
        return None
    return float(sigma_val)


def classify_regime(N: int, sigma: Optional[float]) -> str:
    if sigma is None:
        return "insufficient_data"
    if isinstance(sigma, float) and math.isnan(sigma):
        return "insufficient_data"
    if sigma < 1.0:
        if N < SIGMA_MID_LIFE_THRESHOLD:
            return "developmental"
        return "mid_life_drift"
    return "healthy"


def compute_topology_snapshot(graph, *, assignment=None, rich_club=None) -> dict:
    """Topology metrics for `graph`.

    Computes community detection in-process by default only for callers that
    have not already built the runtime graph. Request-synchronous callers should
    pass the `assignment` and `rich_club` returned by `build_runtime_graph()` so
    this health surface does not re-run the expensive structural passes.
    """
    from iai_mcp.graph import MemoryGraph

    if not isinstance(graph, MemoryGraph):
        raise TypeError(
            f"compute_topology_snapshot expects MemoryGraph, got "
            f"{type(graph).__name__}"
        )

    N = int(graph.node_count())
    if N == 0:
        return {
            "C": 0.0,
            "L": 0.0,
            "sigma": None,
            "community_count": 0,
            "rich_club_ratio": 0.0,
            "N": 0,
            "regime": "insufficient_data",
        }

    indptr, indices, data = graph.to_csr_arrays()
    n_nodes = len(indptr) - 1

    sub_indptr, sub_indices, sub_data, sub_n = _largest_component_csr(
        indptr, indices, data, n_nodes
    )

    C = 0.0
    if sub_n >= 1:
        try:
            C = float(
                lilli_graph.average_clustering(sub_indptr, sub_indices, sub_n)
            )
        except (RuntimeError, ValueError):
            C = 0.0

    L = 0.0
    if sub_n >= 2 and len(sub_indices) > 0:
        try:
            L = float(
                lilli_graph.average_shortest_path_length(
                    sub_indptr, sub_indices, sub_n
                )
            )
        except (RuntimeError, ValueError):
            L = 0.0

    sigma_val = compute_sigma(graph)

    community_count = 0
    rich_club_ratio = 0.0
    try:
        from iai_mcp.community import detect_communities
        from iai_mcp.richclub import rich_club_nodes

        try:
            if assignment is None:
                assignment = detect_communities(
                    graph, prior=None, prior_mode="seeded"
                )
            community_count = int(len(assignment.community_centroids))
        except (RuntimeError, ValueError, TypeError):
            community_count = 0
        try:
            if rich_club is None:
                centrality = {
                    node_id: float(graph.get_centrality(node_id))
                    for node_id in graph.iter_nodes()
                }
                rc = rich_club_nodes(
                    graph, percent=0.10, centrality=centrality
                )
            else:
                rc = rich_club
            rich_club_ratio = (len(rc) / N) if N > 0 else 0.0
        except (RuntimeError, ValueError, TypeError):
            rich_club_ratio = 0.0
    except (ImportError, RuntimeError, TypeError):
        pass

    regime = classify_regime(N, sigma_val)
    return {
        "C": C,
        "L": L,
        "sigma": sigma_val,
        "community_count": community_count,
        "rich_club_ratio": rich_club_ratio,
        "N": N,
        "regime": regime,
    }


def _bump_hebbian_rate_developmental(store: "MemoryStore", N: int) -> None:
    write_event(
        store,
        kind="profile_updated",
        data={
            "knob": HEBBIAN_RATE_KNOB,
            "old": 1.0,
            "new": HEBBIAN_DEVELOPMENTAL_BOOST_FACTOR,
            "ttl_sessions": HEBBIAN_DEVELOPMENTAL_BOOST_TTL_SESSIONS,
            "reason": "sigma_developmental_phase",
            "N": N,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        severity="info",
    )


def compute_and_emit(store: "MemoryStore") -> dict:
    from iai_mcp import retrieve

    graph_bundle = retrieve.build_runtime_graph(store)
    assignment = None
    if isinstance(graph_bundle, tuple):
        graph = graph_bundle[0]
        if len(graph_bundle) > 1:
            assignment = graph_bundle[1]
    else:
        graph = graph_bundle

    # This is the background topology pass (hourly audit + offline sleep step),
    # off the interactive path. The graph build already detects communities in a
    # short-lived child, so the daemon does not accumulate JIT compilation
    # arenas; reuse that assignment. Only when the build did not surface one
    # (degraded bundle) is detection run in a child here, with the same spawn
    # failure degrading to the in-process recompute inside the snapshot.
    if assignment is None:
        try:
            from iai_mcp.runtime_graph_cache import compute_assignment_in_child

            assignment = compute_assignment_in_child(graph, prior_mode="seeded")
        except (RuntimeError, ValueError, TypeError, OSError, AttributeError):
            assignment = None

    snap = compute_topology_snapshot(graph, assignment=assignment)
    regime = snap.get("regime", "insufficient_data")

    base_data = {
        "sigma": snap.get("sigma"),
        "N": snap.get("N", 0),
        "C": snap.get("C", 0.0),
        "L": snap.get("L", 0.0),
        "community_count": snap.get("community_count", 0),
        "rich_club_ratio": snap.get("rich_club_ratio", 0.0),
        "regime": regime,
    }

    if regime == "mid_life_drift":
        write_event(
            store,
            kind=SIGMA_DRIFT_KIND,
            data={**base_data, "phase": "mid_life_drift"},
            severity="warning",
        )
    elif regime == "developmental":
        write_event(
            store,
            kind=SIGMA_OBSERVATION_KIND,
            data={**base_data, "phase": "developmental"},
            severity="info",
        )
        try:
            _bump_hebbian_rate_developmental(store, int(snap.get("N", 0)))
        except (OSError, RuntimeError, ValueError) as exc:
            logger.debug("hebbian_rate_boost_failed", extra={"err": str(exc)[:80]})
            pass
    elif regime == "healthy":
        write_event(
            store,
            kind=SIGMA_OBSERVATION_KIND,
            data={**base_data, "phase": "healthy"},
            severity="info",
        )
    else:
        write_event(
            store,
            kind=SIGMA_OBSERVATION_KIND,
            data={**base_data, "phase": "insufficient_data"},
            severity="info",
        )

    return snap
