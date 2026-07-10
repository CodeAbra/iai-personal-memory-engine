from __future__ import annotations

import random
from uuid import UUID

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = random.Random(seed)
    return [rng.random() for _ in range(dim)]


def _sample_community_sizes(
    n: int,
    n_communities: int,
    min_size: int,
    max_size: int,
    rng: random.Random,
) -> list[int]:
    if n_communities <= 0:
        raise ValueError(f"n_communities must be positive, got {n_communities}")
    if n_communities * min_size > n:
        base = n // n_communities
        sizes = [base] * n_communities
        sizes[-1] += n - sum(sizes)
        return sizes

    raw = []
    for _ in range(n_communities):
        u = rng.random()
        size = min_size * ((1.0 - u + u * (max_size / min_size) ** -0.5) ** -2.0)
        raw.append(max(min_size, min(max_size, size)))

    total = sum(raw)
    sizes = [max(min_size, int(round(s * n / total))) for s in raw]
    diff = n - sum(sizes)
    if diff != 0:
        idx = max(range(n_communities), key=lambda i: sizes[i])
        sizes[idx] = max(min_size, sizes[idx] + diff)
    diff = n - sum(sizes)
    if diff != 0:
        order = sorted(range(n_communities), key=lambda i: -sizes[i])
        i = 0
        step = 1 if diff > 0 else -1
        while diff != 0:
            target = order[i % n_communities]
            if step < 0 and sizes[target] <= min_size:
                i += 1
                continue
            sizes[target] += step
            diff -= step
            i += 1
    assert sum(sizes) == n, f"size sum {sum(sizes)} != n {n}"
    return sizes


def _sample_target_degrees(
    n: int,
    avg_degree: float,
    max_degree: int,
    rng: random.Random,
) -> list[int]:
    min_deg = max(1, int(avg_degree // 4))
    if min_deg >= max_degree:
        return [int(avg_degree)] * n

    gamma = 2.5
    raw = []
    for _ in range(n):
        u = rng.random()
        a_pow = min_deg ** (1.0 - gamma)
        b_pow = max_degree ** (1.0 - gamma)
        d_raw = (u * (b_pow - a_pow) + a_pow) ** (1.0 / (1.0 - gamma))
        raw.append(max(min_deg, min(max_degree, d_raw)))

    actual_mean = sum(raw) / n
    if actual_mean > 0:
        scale = avg_degree / actual_mean
        raw = [r * scale for r in raw]

    degrees = [max(1, min(max_degree, int(round(d)))) for d in raw]
    return degrees


def generate_lfr_like(
    n: int,
    avg_degree: float,
    max_degree: int,
    mu: float,
    n_communities: int,
    min_community: int = 10,
    max_community: int = 100,
    seed: int = 42,
) -> tuple[MemoryGraph, list[int]]:
    if not 0.0 <= mu <= 1.0:
        raise ValueError(f"mu must be in [0, 1], got {mu}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if avg_degree < 1:
        raise ValueError(f"avg_degree must be >= 1, got {avg_degree}")
    if max_degree < avg_degree:
        raise ValueError(
            f"max_degree {max_degree} must be >= avg_degree {avg_degree}"
        )

    rng = random.Random(seed)

    uuids: list[UUID] = [UUID(int=seed * 10**12 + i) for i in range(n)]

    sizes = _sample_community_sizes(
        n, n_communities, min_community, max_community, rng
    )
    planted_labels: list[int] = []
    for c_id, size in enumerate(sizes):
        planted_labels.extend([c_id] * size)
    assert len(planted_labels) == n

    community_members: dict[int, list[int]] = {}
    for i, c_id in enumerate(planted_labels):
        community_members.setdefault(c_id, []).append(i)

    target_degrees = _sample_target_degrees(n, avg_degree, max_degree, rng)

    edges: set[tuple[int, int]] = set()
    placed_per_node = [0] * n
    MAX_RETRIES_PER_EDGE = 30

    for i in range(n):
        c_id = planted_labels[i]
        target_k = target_degrees[i]
        target_intra = int(round((1.0 - mu) * target_k))
        target_inter = target_k - target_intra
        intra_pool = [j for j in community_members[c_id] if j != i]
        inter_pool: list[int] = []
        for other_c, members in community_members.items():
            if other_c != c_id:
                inter_pool.extend(members)

        if not intra_pool and not inter_pool:
            continue
        if not intra_pool:
            target_inter += target_intra
            target_intra = 0
        if not inter_pool:
            target_intra += target_inter
            target_inter = 0

        for _ in range(target_intra):
            if placed_per_node[i] >= target_k:
                break
            for retry in range(MAX_RETRIES_PER_EDGE):
                j = rng.choice(intra_pool)
                if j == i:
                    continue
                key = (min(i, j), max(i, j))
                if key in edges:
                    continue
                edges.add(key)
                placed_per_node[i] += 1
                placed_per_node[j] += 1
                break

        for _ in range(target_inter):
            if placed_per_node[i] >= target_k:
                break
            for retry in range(MAX_RETRIES_PER_EDGE):
                j = rng.choice(inter_pool)
                if j == i:
                    continue
                key = (min(i, j), max(i, j))
                if key in edges:
                    continue
                edges.add(key)
                placed_per_node[i] += 1
                placed_per_node[j] += 1
                break

    g = MemoryGraph()
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(seed * 1000 + i))

    sorted_edges = sorted(edges)
    for u, v in sorted_edges:
        g.add_edge(uuids[u], uuids[v], weight=1.0, edge_type="hebbian")

    return g, planted_labels


__all__ = ["generate_lfr_like"]
