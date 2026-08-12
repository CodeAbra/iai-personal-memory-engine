from __future__ import annotations

import math
import time
from typing import Literal
from uuid import UUID, uuid4

import numpy as np
import scipy.sparse
from numba import njit

from iai_mcp.community import CommunityAssignment, _flat_assignment
from iai_mcp.mosaic_lineage import (
    LineageEvent,
    LineageReport,
    LineageTracker,
    init_partitions,
)
from iai_mcp.graph import MemoryGraph

__all__ = [
    "EPSILON",
    "WALL_TIME_HARD_CAP_S",
    "WALL_TIME_WARM_TARGET_S",
    "build_csr_sanitized",
    "compute_sigma_tot",
    "compute_delta_q_cpm",
    "compute_modularity_cpm",
    "_njit_local_move",
    "_subgraph_connected",
    "_split_disconnected_communities",
    "_njit_refine",
    "_aggregate",
    "run_mosaic",
    "_run_one_leiden_pass",
    "multi_objective_gamma_tuner",
    "_super_level_merge",
    "LineageEvent",
    "LineageReport",
    "LineageTracker",
]

EPSILON: float = 1e-9

WALL_TIME_HARD_CAP_S: float = 30.0

WALL_TIME_WARM_TARGET_S: float = 5.0


def build_csr_sanitized(
    graph: MemoryGraph,
) -> tuple[scipy.sparse.csr_matrix, list[UUID], dict[UUID, int]]:
    order: list[UUID] = sorted(graph.iter_nodes(), key=str)
    n = len(order)
    idx_map: dict[UUID, int] = {u: i for i, u in enumerate(order)}

    if n == 0:
        empty = scipy.sparse.csr_matrix((0, 0), dtype=np.float64)
        return empty, order, idx_map

    edge_weights: dict[tuple[int, int], float] = {}
    for u_uuid, v_uuid, w in graph.iter_edges_with_weight():
        if u_uuid == v_uuid:
            continue
        if not math.isfinite(w):
            continue
        if w < 0.0:
            continue
        if u_uuid not in idx_map or v_uuid not in idx_map:
            continue
        a = idx_map[u_uuid]
        b = idx_map[v_uuid]
        key = (a, b) if a <= b else (b, a)
        edge_weights[key] = edge_weights.get(key, 0.0) + w

    if not edge_weights:
        empty = scipy.sparse.csr_matrix((n, n), dtype=np.float64)
        return empty, order, idx_map

    sorted_edges = sorted(edge_weights.items())
    src = np.empty(len(sorted_edges) * 2, dtype=np.int64)
    dst = np.empty(len(sorted_edges) * 2, dtype=np.int64)
    wts = np.empty(len(sorted_edges) * 2, dtype=np.float64)
    for i, ((a, b), w) in enumerate(sorted_edges):
        src[2 * i] = a
        dst[2 * i] = b
        wts[2 * i] = w
        src[2 * i + 1] = b
        dst[2 * i + 1] = a
        wts[2 * i + 1] = w

    coo = scipy.sparse.coo_matrix(
        (wts, (src, dst)), shape=(n, n), dtype=np.float64
    )
    csr = coo.tocsr()
    csr.sort_indices()
    return csr, order, idx_map


def _flat_fallback(
    graph: MemoryGraph, prior: CommunityAssignment | None
) -> tuple[CommunityAssignment, LineageReport]:
    return _flat_assignment(graph, prior), LineageReport(events=())


def run_mosaic(
    graph: MemoryGraph,
    prior: CommunityAssignment | None = None,
    prior_mode: Literal["seeded", "cold"] = "seeded",
    gamma: float | None = None,
    seed: int = 42,
    max_levels: int = 5,
) -> tuple[CommunityAssignment, LineageReport]:
    if graph.node_count() == 0:
        return _flat_fallback(graph, prior)

    t_start = time.monotonic()

    csr, order, _idx_map = build_csr_sanitized(graph)

    if csr.nnz == 0:
        return _flat_fallback(graph, prior)

    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1

    init_partition, init_int_to_uuid, lineage = init_partitions(
        graph, prior, prior_mode
    )
    partition = init_partition.astype(np.int64, copy=False)

    n_communities = int(partition.max()) + 1 if partition.size else 0
    sigma_tot = compute_sigma_tot(indptr, indices, data, partition, n_communities)

    csr_for_tuner = scipy.sparse.csr_matrix(
        (data, indices, indptr), shape=(n, n)
    )
    if gamma is None:
        tuner_partition = np.arange(n, dtype=np.int64)
        tuner_sigma = compute_sigma_tot(
            indptr, indices, data, tuner_partition, n,
        )
        gamma_value, _tuner_diag = multi_objective_gamma_tuner(
            csr_for_tuner, tuner_partition, tuner_sigma, seed,
        )
        if _tuner_diag.get("should_fall_back_to_flat", False):
            return _flat_fallback(graph, prior)
    else:
        gamma_value = float(gamma)


    int_to_uuid: dict[int, UUID] = {
        i: init_int_to_uuid[int(partition[i])] for i in range(n)
    }

    node_to_super_idx = np.arange(n, dtype=np.int64)

    curr_indptr = indptr
    curr_indices = indices
    curr_data = data
    curr_partition = partition
    curr_sigma = sigma_tot
    curr_int_to_uuid = int_to_uuid
    curr_csr = scipy.sparse.csr_matrix(
        (curr_data, curr_indices, curr_indptr), shape=(n, n)
    )

    for level in range(max_levels):
        n_curr = curr_partition.shape[0]

        rng_lm = np.random.Generator(np.random.PCG64(seed + 2 * level))
        visit_lm = rng_lm.permutation(n_curr).astype(np.int64)
        _moved_lm = _njit_local_move(
            curr_indptr, curr_indices, curr_data,
            curr_partition, curr_sigma,
            gamma_value, visit_lm, 20,
        )

        curr_partition, curr_sigma, curr_int_to_uuid = (
            _split_disconnected_communities(
                curr_csr, curr_partition, curr_sigma,
                curr_int_to_uuid, lineage,
            )
        )

        refined = np.arange(n_curr, dtype=np.int64)
        sigma_refined = compute_sigma_tot(
            curr_indptr, curr_indices, curr_data, refined, n_curr,
        )
        rng_ref = np.random.Generator(np.random.PCG64(seed + 2 * level + 1))
        visit_ref = rng_ref.permutation(n_curr).astype(np.int64)
        _moves_ref = _njit_refine(
            curr_indptr, curr_indices, curr_data,
            curr_partition, refined, sigma_refined,
            gamma_value, visit_ref, 1,
        )

        _moves_subgroup = _refinement_subgroup_merge(
            curr_csr, curr_partition, refined, sigma_refined,
            gamma_value, seed + 3 * level + 2,
        )

        if _moved_lm == 0 and _moves_ref == 0 and _moves_subgroup == 0:
            break

        super_csr, super_partition, super_int_to_uuid = _aggregate(
            curr_csr, refined, curr_int_to_uuid, lineage,
            macro_partition=curr_partition,
        )

        unique_refined = np.unique(refined)
        max_refined_label = int(unique_refined.max()) + 1 if unique_refined.size > 0 else 0
        ref_remap = np.full(max_refined_label, -1, dtype=np.int64)
        for new_idx, orig_label in enumerate(unique_refined):
            ref_remap[int(orig_label)] = new_idx
        node_to_super_idx = ref_remap[refined[node_to_super_idx]]

        super_n = super_partition.shape[0]
        super_sigma = compute_sigma_tot(
            np.ascontiguousarray(super_csr.indptr, dtype=np.int64),
            np.ascontiguousarray(super_csr.indices, dtype=np.int64),
            np.ascontiguousarray(super_csr.data, dtype=np.float64),
            super_partition, super_n,
        )

        curr_csr = super_csr
        curr_indptr = np.ascontiguousarray(super_csr.indptr, dtype=np.int64)
        curr_indices = np.ascontiguousarray(super_csr.indices, dtype=np.int64)
        curr_data = np.ascontiguousarray(super_csr.data, dtype=np.float64)
        curr_partition = super_partition
        curr_sigma = super_sigma
        curr_int_to_uuid = super_int_to_uuid

        if super_n <= 1:
            break

        # Budget check at the BOTTOM of the level: the base level always
        # completes (its local move carries the primary structure), and a
        # timeout means "stop refining", never "discard the structure" — the
        # partition built so far finalizes below. The old top-of-loop check
        # returned flat on timeout, which threw away a strong partition on
        # every large-corpus run.
        if time.monotonic() - t_start > WALL_TIME_HARD_CAP_S:
            break

    final_partition_orig = curr_partition[node_to_super_idx].astype(np.int64)
    unique_final = np.unique(final_partition_orig)
    final_remap = {int(lbl): i for i, lbl in enumerate(unique_final)}
    final_partition_compact = np.array(
        [final_remap[int(final_partition_orig[i])] for i in range(n)],
        dtype=np.int64,
    )
    k_final = len(final_remap)
    final_sigma = compute_sigma_tot(
        indptr, indices, data, final_partition_compact, k_final,
    )

    final_label_to_uuid: dict[int, UUID] = {}
    for compact_label, macro_label in enumerate(unique_final):
        macro_int = int(macro_label)
        super_idxs = np.where(curr_partition == macro_int)[0]
        candidates: list[UUID] = []
        seen: set[UUID] = set()
        for s in super_idxs:
            s_int = int(s)
            if s_int in curr_int_to_uuid:
                u = curr_int_to_uuid[s_int]
                if u not in seen:
                    candidates.append(u)
                    seen.add(u)
        if not candidates:
            fresh = uuid4()
            final_label_to_uuid[compact_label] = fresh
            lineage.record_birth(
                fresh, int((final_partition_compact == compact_label).sum())
            )
            continue
        if len(candidates) == 1:
            final_label_to_uuid[compact_label] = candidates[0]
            continue
        surviving = lineage.pick_merge_survivor(candidates)
        final_label_to_uuid[compact_label] = surviving
        lineage.record_merge(
            candidates, surviving,
            int((final_partition_compact == compact_label).sum()),
        )

    _super_level_merge(
        scipy.sparse.csr_matrix((data, indices, indptr), shape=(n, n)),
        final_partition_compact, final_sigma,
        gamma_value, seed,
        lineage_tracker=lineage,
        label_to_uuid=final_label_to_uuid,
    )

    post_merge_unique = np.unique(final_partition_compact)
    if post_merge_unique.size != k_final:
        relabel_post = {int(lbl): i for i, lbl in enumerate(post_merge_unique)}
        new_label_to_uuid: dict[int, UUID] = {}
        for old_lbl, new_lbl in relabel_post.items():
            if old_lbl in final_label_to_uuid:
                new_label_to_uuid[new_lbl] = final_label_to_uuid[old_lbl]
        final_label_to_uuid = new_label_to_uuid
        final_partition_compact = np.array(
            [relabel_post[int(v)] for v in final_partition_compact.tolist()],
            dtype=np.int64,
        )
        k_final = post_merge_unique.size
        final_sigma = compute_sigma_tot(
            indptr, indices, data, final_partition_compact, k_final,
        )

    modularity = float(
        compute_modularity_cpm(
            indptr, indices, data, final_partition_compact, final_sigma, gamma_value,
        )
    )

    # An all-singleton final partition means the local move accepted nothing:
    # no realized structure, and materializing one centroid per node would be
    # pure waste. Flat is the honest answer for exactly this case.
    if k_final >= n:
        return _flat_fallback(graph, prior)

    partition = final_partition_compact
    sigma_tot = final_sigma

    assignment = _build_assignment(
        graph, order, partition, modularity, final_label_to_uuid
    )
    return assignment, lineage.report()


def _build_assignment(
    graph: MemoryGraph,
    order: list[UUID],
    partition: np.ndarray,
    modularity: float,
    label_to_uuid: dict[int, UUID] | None = None,
) -> CommunityAssignment:
    groups: dict[int, list[UUID]] = {}
    for idx, label in enumerate(partition.tolist()):
        groups.setdefault(int(label), []).append(order[idx])

    if label_to_uuid is None:
        label_to_uuid = {}
    label_to_uuid = {
        label: label_to_uuid.get(label, uuid4()) for label in sorted(groups)
    }

    node_to_community: dict[UUID, UUID] = {}
    community_centroids: dict[UUID, list[float]] = {}
    mid_regions: dict[UUID, list[UUID]] = {}
    nonempty_embs: list[list[float]] = []
    for label, members in groups.items():
        u = label_to_uuid[label]
        mid_regions[u] = list(members)
        for n in members:
            node_to_community[n] = u
            emb = graph.get_embedding(n)
            if emb is not None:
                nonempty_embs.append(emb)

    dim = len(nonempty_embs[0]) if nonempty_embs else 0
    for label, members in groups.items():
        u = label_to_uuid[label]
        embs: list[list[float]] = []
        for node in members:
            emb = graph.get_embedding(node)
            embs.append(emb if emb is not None else [0.0] * dim)
        if dim > 0 and embs:
            arr = np.asarray(embs, dtype=np.float32)
            centroid = arr.mean(axis=0)
            norm = float(np.linalg.norm(centroid))
            if norm > 0:
                centroid = centroid / norm
            community_centroids[u] = centroid.tolist()
        else:
            community_centroids[u] = []

    sorted_labels = sorted(groups, key=lambda lbl: -len(groups[lbl]))
    top_communities = [label_to_uuid[lbl] for lbl in sorted_labels[:7]]

    return CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=community_centroids,
        modularity=modularity,
        backend="leiden-custom",
        top_communities=top_communities,
        mid_regions=mid_regions,
    )


@njit(fastmath=False, cache=True)
def compute_sigma_tot(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    n_communities: int,
) -> np.ndarray:
    sigma = np.zeros(n_communities, dtype=np.float64)
    n = partition.shape[0]
    for i in range(n):
        comm = partition[i]
        start = indptr[i]
        end = indptr[i + 1]
        s = 0.0
        for off in range(start, end):
            s += data[off]
        sigma[comm] += s
    return sigma


@njit(fastmath=False, cache=True)
def compute_delta_q_cpm(
    node_idx: int,
    current_comm: int,
    target_comm: int,
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    k_i: float,
    gamma: float,
    two_m: float = 0.0,
) -> float:
    w_to_target = 0.0
    w_to_current_minus_i = 0.0
    start = indptr[node_idx]
    end = indptr[node_idx + 1]
    for off in range(start, end):
        j = indices[off]
        if j == node_idx:
            # A self-loop is node i's own internal mass, not a link to "current
            # community without i" — it belongs in k_i / sigma_tot (degree), not
            # the links-to-community sums. At aggregated levels every merged
            # internal edge becomes such a self-loop, so counting it here would
            # over-state the cost of leaving and make super-nodes artificially
            # sticky to their current community.
            continue
        w = data[off]
        comm_j = partition[j]
        if comm_j == target_comm:
            w_to_target += w
        elif comm_j == current_comm:
            w_to_current_minus_i += w

    sigma_target = sigma_tot[target_comm]
    sigma_current_minus_i = sigma_tot[current_comm] - k_i
    raw_resolution = sigma_target - sigma_current_minus_i
    if two_m > 0.0:
        normalised_resolution = raw_resolution / two_m
    else:
        normalised_resolution = raw_resolution
    return (
        (w_to_target - w_to_current_minus_i)
        - gamma * k_i * normalised_resolution
    )


@njit(fastmath=False, cache=True)
def compute_modularity_cpm(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
) -> float:
    n = partition.shape[0]
    two_m = 0.0
    for off in range(indptr[n]):
        two_m += data[off]
    if two_m <= 0.0:
        return 0.0

    n_comm = sigma_tot.shape[0]
    w_in = np.zeros(n_comm, dtype=np.float64)
    for i in range(n):
        comm_i = partition[i]
        start = indptr[i]
        end = indptr[i + 1]
        for off in range(start, end):
            j = indices[off]
            if partition[j] == comm_i:
                w_in[comm_i] += data[off]

    q = 0.0
    inv_two_m = 1.0 / two_m
    for c in range(n_comm):
        if sigma_tot[c] == 0.0:
            continue
        share = sigma_tot[c] * inv_two_m
        q += (w_in[c] * inv_two_m) - gamma * share * share
    return q


@njit(fastmath=False, cache=True)
def _njit_local_move(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
    visit_order: np.ndarray,
    max_iter: int,
) -> int:
    n = partition.shape[0]
    total_moves = 0
    epsilon = 1e-9

    two_m = 0.0
    nnz = indptr[n]
    for off in range(nnz):
        two_m += data[off]

    for _it in range(max_iter):
        moves_this = 0
        for idx in range(n):
            i = visit_order[idx]
            current = partition[i]
            start = indptr[i]
            end = indptr[i + 1]
            k_i = 0.0
            for off in range(start, end):
                k_i += data[off]

            best_dq = 0.0
            best_comm = current
            for off in range(start, end):
                neighbor_comm = partition[indices[off]]
                if neighbor_comm == current:
                    continue
                dq = compute_delta_q_cpm(
                    i, current, neighbor_comm,
                    indptr, indices, data, partition, sigma_tot,
                    k_i, gamma, two_m,
                )
                if dq > best_dq + epsilon:
                    best_dq = dq
                    best_comm = neighbor_comm

            if best_comm != current:
                sigma_tot[current] -= k_i
                sigma_tot[best_comm] += k_i
                partition[i] = best_comm
                moves_this += 1
        total_moves += moves_this
        if moves_this == 0:
            break
    return total_moves


@njit(fastmath=False, cache=True)
def _subgraph_connected(
    indptr: np.ndarray,
    indices: np.ndarray,
    node_mask: np.ndarray,
) -> bool:
    n = node_mask.shape[0]
    start_idx = -1
    target_count = 0
    for i in range(n):
        if node_mask[i]:
            target_count += 1
            if start_idx == -1:
                start_idx = i
    if start_idx == -1:
        return True
    if target_count == 1:
        return True

    visited = np.zeros(n, dtype=np.bool_)
    queue = np.empty(n, dtype=np.int64)
    queue[0] = start_idx
    visited[start_idx] = True
    head = 0
    tail = 1
    seen_count = 1
    while head < tail:
        u = queue[head]
        head += 1
        start = indptr[u]
        end = indptr[u + 1]
        for off in range(start, end):
            v = indices[off]
            if node_mask[v] and (not visited[v]):
                visited[v] = True
                queue[tail] = v
                tail += 1
                seen_count += 1
    return seen_count == target_count


def _split_disconnected_communities(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    int_to_uuid: dict[int, UUID],
    lineage: "LineageTracker | None",
) -> tuple[np.ndarray, np.ndarray, dict[int, UUID]]:
    import scipy.sparse.csgraph as _csgraph

    n = partition.shape[0]
    new_partition = partition.copy()
    new_int_to_uuid = dict(int_to_uuid)
    next_label = int(new_partition.max()) + 1 if n > 0 else 0

    for label in np.unique(partition):
        members = np.where(partition == label)[0]
        if members.shape[0] <= 1:
            continue
        sub = csr[members, :][:, members]
        n_comp, sub_labels = _csgraph.connected_components(sub, directed=False)
        if n_comp <= 1:
            continue
        sizes = [(int((sub_labels == c).sum()), c) for c in range(n_comp)]
        sizes.sort(key=lambda kv: (-kv[0], kv[1]))
        parent_uuid = new_int_to_uuid.get(int(label))
        child_uuids: list[UUID] = []
        for rank, (_size, comp_id) in enumerate(sizes):
            comp_mask = sub_labels == comp_id
            comp_members = members[comp_mask]
            if rank == 0:
                continue
            new_partition[comp_members] = next_label
            new_uuid = uuid4()
            new_int_to_uuid[next_label] = new_uuid
            child_uuids.append(new_uuid)
            next_label += 1
        if lineage is not None and parent_uuid is not None and child_uuids:
            lineage.record_split(
                parent_uuid, child_uuids, int(members.shape[0])
            )

    new_n_comm = int(new_partition.max()) + 1
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    new_sigma = compute_sigma_tot(
        indptr, indices, data, new_partition, new_n_comm
    )
    return new_partition, new_sigma, new_int_to_uuid


@njit(fastmath=False, cache=True)
def _njit_refine(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    partition: np.ndarray,
    refined: np.ndarray,
    sigma_tot_refined: np.ndarray,
    gamma: float,
    visit_order: np.ndarray,
    max_iter: int,
) -> int:
    n = partition.shape[0]
    epsilon = 1e-9
    total_moves = 0

    two_m = 0.0
    nnz = indptr[n]
    for off in range(nnz):
        two_m += data[off]

    target_mask = np.zeros(n, dtype=np.bool_)
    source_mask = np.zeros(n, dtype=np.bool_)

    for _it in range(max_iter):
        moves_this = 0
        for idx in range(n):
            i = visit_order[idx]
            macro_C = partition[i]
            current = refined[i]

            k_i = 0.0
            start = indptr[i]
            end = indptr[i + 1]
            for off in range(start, end):
                k_i += data[off]

            best_dq = 0.0
            best_comm = current
            for off in range(start, end):
                j = indices[off]
                if partition[j] != macro_C:
                    continue
                neighbor_sub = refined[j]
                if neighbor_sub == current:
                    continue

                dq = compute_delta_q_cpm(
                    i, current, neighbor_sub,
                    indptr, indices, data, refined, sigma_tot_refined,
                    k_i, gamma, two_m,
                )
                if dq <= best_dq + epsilon:
                    continue

                for k in range(n):
                    target_mask[k] = False
                for k in range(n):
                    if partition[k] == macro_C and refined[k] == neighbor_sub:
                        target_mask[k] = True
                target_mask[i] = True
                if not _subgraph_connected(indptr, indices, target_mask):
                    continue

                for k in range(n):
                    source_mask[k] = False
                for k in range(n):
                    if partition[k] == macro_C and refined[k] == current:
                        source_mask[k] = True
                source_mask[i] = False
                source_count = 0
                for k in range(n):
                    if source_mask[k]:
                        source_count += 1
                if source_count > 0:
                    if not _subgraph_connected(indptr, indices, source_mask):
                        continue

                best_dq = dq
                best_comm = neighbor_sub

            if best_comm != current:
                sigma_tot_refined[current] -= k_i
                sigma_tot_refined[best_comm] += k_i
                refined[i] = best_comm
                moves_this += 1

        total_moves += moves_this
        if moves_this == 0:
            break
    return total_moves


def _refinement_subgroup_merge(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    refined: np.ndarray,
    sigma_tot_refined: np.ndarray,
    gamma: float,
    seed: int,
) -> int:
    n = partition.shape[0]
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    two_m = float(data.sum())
    if two_m <= 0.0:
        return 0

    n_refined = int(refined.max()) + 1
    accepted = 0

    sub_to_macro: dict[int, int] = {}
    for i in range(n):
        s = int(refined[i])
        if s not in sub_to_macro:
            sub_to_macro[s] = int(partition[i])

    macro_to_subs: dict[int, list[int]] = {}
    for sub, macro in sub_to_macro.items():
        macro_to_subs.setdefault(macro, []).append(sub)

    k_per_sub = sigma_tot_refined.copy()

    epsilon = 1e-9

    max_subs_per_macro = (
        max(len(s) for s in macro_to_subs.values())
        if macro_to_subs else 0
    )
    if max_subs_per_macro > 50:
        return 0

    for macro in sorted(macro_to_subs.keys()):
        subs = sorted(macro_to_subs[macro])
        if len(subs) < 2:
            continue

        changed = True
        max_inner_iter = max(10, len(subs))
        inner = 0
        while changed and inner < max_inner_iter:
            inner += 1
            changed = False
            current_subs = sorted(set(int(refined[i]) for i in range(n) if partition[i] == macro))
            if len(current_subs) < 2:
                break

            best_dq = epsilon
            best_pair: tuple[int, int] | None = None
            for ii in range(len(current_subs)):
                S_i = current_subs[ii]
                for jj in range(ii + 1, len(current_subs)):
                    S_j = current_subs[jj]
                    w_ij = 0.0
                    members_i = [k for k in range(n) if refined[k] == S_i]
                    for u in members_i:
                        s = int(indptr[u])
                        e = int(indptr[u + 1])
                        for off in range(s, e):
                            v = int(indices[off])
                            if int(refined[v]) == S_j:
                                w_ij += float(data[off])
                    k_Si = float(k_per_sub[S_i])
                    k_Sj = float(k_per_sub[S_j])
                    dq = (w_ij / two_m) - gamma * k_Si * k_Sj / (two_m * two_m)
                    if dq <= best_dq:
                        continue

                    mask = np.zeros(n, dtype=np.bool_)
                    for k in range(n):
                        if int(refined[k]) == S_i or int(refined[k]) == S_j:
                            mask[k] = True
                    if not _subgraph_connected(indptr, indices, mask):
                        continue

                    best_dq = dq
                    best_pair = (S_i, S_j)

            if best_pair is None:
                break
            S_i, S_j = best_pair
            for k in range(n):
                if int(refined[k]) == S_i:
                    refined[k] = S_j
            sigma_tot_refined[S_j] += sigma_tot_refined[S_i]
            sigma_tot_refined[S_i] = 0.0
            k_per_sub[S_j] += k_per_sub[S_i]
            k_per_sub[S_i] = 0.0
            accepted += 1
            changed = True

    return accepted


def _super_level_merge(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
    seed: int,
    lineage_tracker: "LineageTracker | None" = None,
    label_to_uuid: dict[int, UUID] | None = None,
    max_iter: int = 5,
) -> int:
    n = partition.shape[0]
    if n == 0:
        return 0
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    two_m = float(data.sum())
    if two_m <= 0.0:
        return 0
    inv_two_m_sq = 1.0 / (two_m * two_m)

    accepted_total = 0
    _ = seed
    _ = max_iter

    # Aggregate the inter-community weights ONCE (O(E)); every merge decision
    # then reads the super-graph instead of rescanning all nodes per candidate
    # pair. The previous shape — an O(n·d) scan per (ci, cj) pair with a full
    # restart after every accepted merge — was quadratic-and-worse in the
    # community count and unusable at corpus scale (thousands of communities).
    adj: dict[int, dict[int, float]] = {}
    sizes: dict[int, int] = {}
    for u in range(n):
        cu = int(partition[u])
        sizes[cu] = sizes.get(cu, 0) + 1
        row = adj.setdefault(cu, {})
        for off in range(int(indptr[u]), int(indptr[u + 1])):
            cv = int(partition[int(indices[off])])
            if cv != cu:
                row[cv] = row.get(cv, 0.0) + float(data[off])

    alive = set(adj.keys()) | set(sizes.keys())
    if len(alive) < 2:
        return 0

    changed = True
    while changed and len(alive) > 1:
        changed = False
        for ci in sorted(alive):
            if ci not in alive:
                continue
            # Only adjacent communities can gain edge mass from a merge; the
            # best positive-ΔQ neighbor (ties break on the lower label) wins.
            best_dq = EPSILON
            best_cj = -1
            for cj, w_ij in sorted(adj.get(ci, {}).items()):
                if cj == ci or cj not in alive:
                    continue
                delta_q = (
                    2.0 * w_ij / two_m
                    - 2.0 * gamma * float(sigma_tot[ci]) * float(sigma_tot[cj])
                    * inv_two_m_sq
                )
                if delta_q > best_dq:
                    best_dq = delta_q
                    best_cj = cj
            if best_cj < 0:
                continue

            cj = best_cj
            sigma_tot[ci] = float(sigma_tot[ci]) + float(sigma_tot[cj])
            sigma_tot[cj] = 0.0
            sizes[ci] = sizes.get(ci, 0) + sizes.pop(cj, 0)
            row_i = adj.setdefault(ci, {})
            for ck, w in adj.pop(cj, {}).items():
                if ck == ci:
                    continue
                row_i[ck] = row_i.get(ck, 0.0) + w
                nbr = adj.setdefault(ck, {})
                nbr[ci] = nbr.get(ci, 0.0) + nbr.pop(cj, 0.0)
            row_i.pop(cj, None)
            alive.discard(cj)
            partition[partition == cj] = ci

            if (
                lineage_tracker is not None
                and label_to_uuid is not None
                and ci in label_to_uuid
                and cj in label_to_uuid
            ):
                parents = [label_to_uuid[ci], label_to_uuid[cj]]
                surviving = lineage_tracker.pick_merge_survivor(parents)
                lineage_tracker.record_merge(parents, surviving, sizes[ci])
                label_to_uuid[ci] = surviving
                del label_to_uuid[cj]
            elif lineage_tracker is not None:
                placeholder_ci = uuid4()
                placeholder_cj = uuid4()
                lineage_tracker.record_merge(
                    [placeholder_ci, placeholder_cj],
                    surviving=placeholder_ci,
                    member_count=sizes[ci],
                )

            accepted_total += 1
            changed = True

    return accepted_total


def _aggregate(
    csr: scipy.sparse.csr_matrix,
    refined: np.ndarray,
    int_to_uuid: dict[int, UUID],
    lineage: "LineageTracker | None",
    macro_partition: np.ndarray | None = None,
) -> tuple[scipy.sparse.csr_matrix, np.ndarray, dict[int, UUID]]:
    n = refined.shape[0]
    unique_labels = np.unique(refined)
    k = unique_labels.shape[0]
    max_label = int(unique_labels.max()) + 1 if k > 0 else 0
    label_remap = np.full(max_label, -1, dtype=np.int64)
    for new_idx, orig_label in enumerate(unique_labels):
        label_remap[int(orig_label)] = new_idx
    super_idx = label_remap[refined]

    super_to_prior: dict[int, list[int]] = {i: [] for i in range(k)}
    seen_prior: set[int] = set()
    for prior_label in int_to_uuid.keys():
        if prior_label < 0 or prior_label >= n:
            if lineage is not None:
                lineage.record_death(int_to_uuid[prior_label], 0)
            continue
        r = int(refined[prior_label])
        if r < 0 or r >= max_label or label_remap[r] < 0:
            if lineage is not None:
                lineage.record_death(int_to_uuid[prior_label], 0)
            continue
        super_label = int(label_remap[r])
        super_to_prior[super_label].append(prior_label)
        seen_prior.add(prior_label)

    super_int_to_uuid: dict[int, UUID] = {}
    for super_label in range(k):
        contributors = super_to_prior[super_label]
        if len(contributors) == 0:
            new_uuid = uuid4()
            super_int_to_uuid[super_label] = new_uuid
            if lineage is not None:
                member_count = int((super_idx == super_label).sum())
                lineage.record_birth(new_uuid, member_count)
            continue
        parent_uuids_set: dict[UUID, None] = {}
        for c in contributors:
            parent_uuids_set[int_to_uuid[c]] = None
        parent_uuids = list(parent_uuids_set.keys())
        if len(parent_uuids) == 1:
            super_int_to_uuid[super_label] = parent_uuids[0]
            continue
        if lineage is not None:
            surviving = lineage.pick_merge_survivor(parent_uuids)
        else:
            surviving = min(parent_uuids, key=str)
        super_int_to_uuid[super_label] = surviving
        if lineage is not None:
            member_count = int((super_idx == super_label).sum())
            lineage.record_merge(parent_uuids, surviving, member_count)

    indptr = csr.indptr
    indices = csr.indices
    data = csr.data
    src_super = np.empty(indptr[n], dtype=np.int64)
    dst_super = np.empty(indptr[n], dtype=np.int64)
    w_super = np.empty(indptr[n], dtype=np.float64)
    pos = 0
    for u in range(n):
        su = int(super_idx[u])
        s = int(indptr[u])
        e = int(indptr[u + 1])
        for off in range(s, e):
            v = int(indices[off])
            sv = int(super_idx[v])
            src_super[pos] = su
            dst_super[pos] = sv
            w_super[pos] = float(data[off])
            pos += 1
    coo = scipy.sparse.coo_matrix(
        (w_super[:pos], (src_super[:pos], dst_super[:pos])),
        shape=(k, k), dtype=np.float64,
    )
    super_csr = coo.tocsr()
    super_csr.sort_indices()

    if macro_partition is not None:
        super_partition = np.empty(k, dtype=np.int64)
        for orig_label in unique_labels:
            new_super = int(label_remap[int(orig_label)])
            for i in range(n):
                if refined[i] == orig_label:
                    super_partition[new_super] = int(macro_partition[i])
                    break
        unique_super = np.unique(super_partition)
        super_remap = np.full(int(unique_super.max()) + 1, -1, dtype=np.int64)
        for new_idx, orig in enumerate(unique_super):
            super_remap[int(orig)] = new_idx
        super_partition = super_remap[super_partition]
    else:
        super_partition = np.arange(k, dtype=np.int64)
    return super_csr, super_partition, super_int_to_uuid


_DEFAULT_GAMMA_CANDIDATES: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)


def _run_one_leiden_pass(
    csr: scipy.sparse.csr_matrix,
    partition: np.ndarray,
    sigma_tot: np.ndarray,
    gamma: float,
    seed: int,
    max_levels: int = 5,
) -> tuple[np.ndarray, float, dict]:
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    n = indptr.shape[0] - 1

    curr_partition = partition.copy()
    curr_sigma = sigma_tot.copy()
    curr_indptr = indptr
    curr_indices = indices
    curr_data = data
    curr_csr = csr
    curr_int_to_uuid: dict[int, UUID] = {i: uuid4() for i in range(n)}

    node_to_super_idx = np.arange(n, dtype=np.int64)

    total_lm_moves = 0
    total_refine_moves = 0
    levels_run = 0

    _pass_t0 = time.monotonic()
    _pass_budget_s = WALL_TIME_HARD_CAP_S / 4.0

    for level in range(max_levels):
        if time.monotonic() - _pass_t0 > _pass_budget_s:
            break
        levels_run = level + 1
        n_curr = curr_partition.shape[0]

        rng_lm = np.random.Generator(np.random.PCG64(seed + 2 * level))
        visit_lm = rng_lm.permutation(n_curr).astype(np.int64)
        lm_moves = _njit_local_move(
            curr_indptr, curr_indices, curr_data,
            curr_partition, curr_sigma,
            gamma, visit_lm, 20,
        )
        total_lm_moves += int(lm_moves)

        curr_partition, curr_sigma, curr_int_to_uuid = (
            _split_disconnected_communities(
                curr_csr, curr_partition, curr_sigma,
                curr_int_to_uuid, LineageTracker(),
            )
        )

        refined = np.arange(n_curr, dtype=np.int64)
        sigma_refined = compute_sigma_tot(
            curr_indptr, curr_indices, curr_data, refined, n_curr,
        )
        rng_ref = np.random.Generator(np.random.PCG64(seed + 2 * level + 1))
        visit_ref = rng_ref.permutation(n_curr).astype(np.int64)
        ref_moves = _njit_refine(
            curr_indptr, curr_indices, curr_data,
            curr_partition, refined, sigma_refined,
            gamma, visit_ref, 1,
        )
        total_refine_moves += int(ref_moves)


        if lm_moves == 0 and ref_moves == 0:
            break

        super_csr, super_partition, super_int_to_uuid = _aggregate(
            curr_csr, refined, curr_int_to_uuid, LineageTracker(),
            macro_partition=curr_partition,
        )

        unique_refined = np.unique(refined)
        max_refined_label = (
            int(unique_refined.max()) + 1 if unique_refined.size > 0 else 0
        )
        ref_remap = np.full(max_refined_label, -1, dtype=np.int64)
        for new_idx, orig_label in enumerate(unique_refined):
            ref_remap[int(orig_label)] = new_idx
        node_to_super_idx = ref_remap[refined[node_to_super_idx]]

        curr_csr = super_csr
        curr_indptr = np.ascontiguousarray(super_csr.indptr, dtype=np.int64)
        curr_indices = np.ascontiguousarray(super_csr.indices, dtype=np.int64)
        curr_data = np.ascontiguousarray(super_csr.data, dtype=np.float64)
        curr_partition = super_partition
        super_n = super_partition.shape[0]
        curr_sigma = compute_sigma_tot(
            curr_indptr, curr_indices, curr_data, super_partition, super_n,
        )
        curr_int_to_uuid = super_int_to_uuid

        if super_n <= 1:
            break

    final_partition_orig = curr_partition[node_to_super_idx].astype(np.int64)
    unique_final = np.unique(final_partition_orig)
    final_remap = {int(lbl): i for i, lbl in enumerate(unique_final)}
    final_partition_compact = np.array(
        [final_remap[int(final_partition_orig[i])] for i in range(n)],
        dtype=np.int64,
    )
    k_final = len(final_remap)
    final_sigma = compute_sigma_tot(
        indptr, indices, data, final_partition_compact, k_final,
    )
    cpm_q = float(compute_modularity_cpm(
        indptr, indices, data, final_partition_compact, final_sigma, gamma,
    ))
    return final_partition_compact, cpm_q, {
        "lm_moves_total": total_lm_moves,
        "refine_moves_total": total_refine_moves,
        "levels": levels_run,
        "n_communities": k_final,
    }


def multi_objective_gamma_tuner(
    csr: scipy.sparse.csr_matrix,
    initial_partition: np.ndarray,
    initial_sigma_tot: np.ndarray,
    seed: int,
    targets: dict | None = None,
) -> tuple[float, dict]:
    from iai_mcp.mosaic_policy import (
        CPM_MODULARITY_FLOOR,
        all_communities_connected,
        compute_singleton_ratio,
    )

    if targets is None:
        targets = {
            "q_min": CPM_MODULARITY_FLOOR,
            "singleton_ratio_max": 0.30,
        }
    n = int(initial_partition.size)
    candidate_scores: dict[float, float] = {}
    candidate_stats: dict[float, dict] = {}
    best_gamma: float = 1.0
    best_score: float = float("-inf")
    any_satisfied: bool = False
    structure_found: bool = False
    budget_exhausted: bool = False
    tuner_budget_s = WALL_TIME_HARD_CAP_S / 2.0
    t_start = time.monotonic()

    for gamma in _DEFAULT_GAMMA_CANDIDATES:
        gamma_f = float(gamma)
        if time.monotonic() - t_start > tuner_budget_s:
            budget_exhausted = True
            break
        p_test, q, _stats = _run_one_leiden_pass(
            csr, initial_partition, initial_sigma_tot, gamma_f, seed,
        )
        s_ratio = float(compute_singleton_ratio(p_test))
        n_communities = int(len(np.unique(p_test)))
        connected_ok = bool(all_communities_connected(csr, p_test))
        # Secondary criteria RANK candidates; only the modularity floor gates
        # structure-vs-flat. A candidate clearing every secondary bar is
        # preferred, but a strong-Q candidate that misses one must still beat
        # flat — flat disables the community gate outright.
        hard_satisfied = (
            q >= targets["q_min"]
            and s_ratio < targets["singleton_ratio_max"]
            and connected_ok
        )
        composite = q - 0.5 * s_ratio
        candidate_scores[gamma_f] = float(composite)
        candidate_stats[gamma_f] = {
            "q": float(q),
            "singleton_ratio": s_ratio,
            "n_communities": n_communities,
            "connected": connected_ok,
            "hard_satisfied": bool(hard_satisfied),
        }
        if q >= targets["q_min"]:
            structure_found = True
        if hard_satisfied:
            any_satisfied = True
            if composite > best_score:
                best_score = composite
                best_gamma = gamma_f

    if not any_satisfied and candidate_stats:
        # No candidate cleared every secondary bar: prefer the best composite
        # among those with real structure (Q at/above the floor); fall back to
        # the global best composite only when nothing reached the floor.
        structured = {
            g: candidate_scores[g]
            for g, st in candidate_stats.items()
            if st["q"] >= targets["q_min"]
        }
        pool = structured if structured else candidate_scores
        best_gamma = float(max(pool.items(), key=lambda kv: kv[1])[0])

    if candidate_stats and best_gamma in candidate_stats:
        best_gamma_q = float(candidate_stats[best_gamma]["q"])
        best_gamma_s_ratio = float(
            candidate_stats[best_gamma]["singleton_ratio"]
        )
        best_gamma_n_c = int(candidate_stats[best_gamma]["n_communities"])
    else:
        best_gamma_q = 0.0
        best_gamma_s_ratio = 0.0
        best_gamma_n_c = 0

    diagnostics: dict = {
        "all_constraints_satisfied": any_satisfied,
        "candidate_scores": candidate_scores,
        "candidate_stats": candidate_stats,
        "should_fall_back_to_flat": (not structure_found),
        "tuner_budget_exhausted": budget_exhausted,
        "best_gamma_q": best_gamma_q,
        "best_gamma_singleton_ratio": best_gamma_s_ratio,
        "best_gamma_n_communities": best_gamma_n_c,
    }
    return float(best_gamma), diagnostics
