from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph as _csgraph

from iai_mcp.mosaic import compute_modularity_cpm

CPM_MODULARITY_FLOOR: float = 0.1338


def compute_modularity_classical(
    csr: scipy.sparse.csr_matrix, partition: np.ndarray
) -> float:
    if csr.nnz == 0 or partition.size == 0:
        return 0.0
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int64)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int64)
    data = np.ascontiguousarray(csr.data, dtype=np.float64)
    unique = np.unique(partition)
    relabel = np.full(int(unique.max()) + 1 if unique.size else 1, -1, dtype=np.int64)
    for new_idx, lbl in enumerate(unique):
        relabel[int(lbl)] = new_idx
    compact = relabel[partition].astype(np.int64)
    k = int(unique.size)
    sigma = np.zeros(k, dtype=np.float64)
    n = compact.shape[0]
    for i in range(n):
        s = 0.0
        for off in range(int(indptr[i]), int(indptr[i + 1])):
            s += float(data[off])
        sigma[int(compact[i])] += s
    return float(compute_modularity_cpm(indptr, indices, data, compact, sigma, 1.0))


def compute_singleton_ratio(partition: np.ndarray) -> float:
    if partition.size == 0:
        return 0.0
    _, counts = np.unique(partition, return_counts=True)
    singletons = int((counts == 1).sum())
    return float(singletons) / float(partition.size)


def all_communities_connected(
    csr: scipy.sparse.csr_matrix, partition: np.ndarray
) -> bool:
    if partition.size == 0:
        return True
    for lbl in np.unique(partition):
        mask = partition == lbl
        if mask.sum() <= 1:
            continue
        sub = csr[mask][:, mask]
        n_comp, _ = _csgraph.connected_components(sub, directed=False)
        if n_comp > 1:
            return False
    return True


def should_fall_back_to_flat(modularity: float) -> bool:
    """Flat replaces a partition ONLY when it carries no structure (modularity
    below the CPM floor). Secondary signals — singleton ratio, community
    count — rank gamma candidates but must never veto a strong-modularity
    partition down to flat: on a sparse memory graph (average degree ~3)
    thousands of small tight communities ARE the true structure, and a k=1
    flat assignment disables the recall community gate entirely. A former
    count veto (k > n/5) did exactly that on the live corpus, collapsing a
    0.78-modularity partition into one community."""
    return modularity < CPM_MODULARITY_FLOOR
