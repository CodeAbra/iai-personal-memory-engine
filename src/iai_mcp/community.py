from __future__ import annotations


from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal
from uuid import UUID, uuid4

import numpy as np

from iai_mcp.graph import MemoryGraph

if TYPE_CHECKING:
    from iai_mcp.mosaic_lineage import LineageReport

SMALL_N_FLAT = 200
MID_N_LEIDEN = 500
MODULARITY_FLOOR = 0.2

REFRESH_DELTA = 0.05

UUID_ROTATE_COSINE = 0.7

MAX_TOP_COMMUNITIES = 7


@dataclass
class CommunityAssignment:

    node_to_community: dict[UUID, UUID] = field(default_factory=dict)
    community_centroids: dict[UUID, list[float]] = field(default_factory=dict)
    modularity: float = 0.0
    backend: str = "flat"
    top_communities: list[UUID] = field(default_factory=list)
    mid_regions: dict[UUID, list[UUID]] = field(default_factory=dict)
    lineage_report: "LineageReport | None" = None


def _cosine(a: list[float], b: list[float]) -> float:
    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av))
    nb = float(np.linalg.norm(bv))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(av, bv) / (na * nb))


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    if not embeddings:
        return []
    arr = np.asarray(embeddings, dtype=np.float32)
    centroid = arr.mean(axis=0)
    norm = float(np.linalg.norm(centroid))
    if norm > 0:
        centroid = centroid / norm
    return centroid.tolist()


def _flat_assignment(
    graph: MemoryGraph, prior: CommunityAssignment | None
) -> CommunityAssignment:
    nodes: list[UUID] = []
    valid_embs: list[list[float]] = []
    for u in graph.iter_nodes():
        nodes.append(u)
        emb = graph.get_embedding(u)
        if emb:
            valid_embs.append(emb)
    if not nodes:
        return CommunityAssignment(backend="flat")

    dim = len(valid_embs[0]) if valid_embs else 0
    embs: list[list[float]] = []
    for u in graph.iter_nodes():
        emb = graph.get_embedding(u)
        embs.append(emb if emb else [0.0] * dim)
    centroid = _compute_centroid(embs) if dim else []

    flat_uuid: UUID | None = None
    if prior and len(prior.community_centroids) == 1:
        prior_uuid, prior_cent = next(iter(prior.community_centroids.items()))
        if _cosine(centroid, prior_cent) >= UUID_ROTATE_COSINE:
            flat_uuid = prior_uuid
    if flat_uuid is None:
        flat_uuid = uuid4()

    node_to_community = {n: flat_uuid for n in nodes}
    community_centroids = {flat_uuid: centroid}
    return CommunityAssignment(
        node_to_community=node_to_community,
        community_centroids=community_centroids,
        modularity=0.0,
        backend="flat",
        top_communities=[flat_uuid],
        mid_regions={flat_uuid: nodes},
    )


def detect_communities(
    graph: MemoryGraph,
    prior: CommunityAssignment | None = None,
    prior_mode: Literal["seeded", "cold"] = "seeded",
) -> CommunityAssignment:
    from iai_mcp.mosaic_lineage import LineageReport

    n = graph.node_count()
    if n == 0:
        return CommunityAssignment(
            backend="flat", lineage_report=LineageReport(events=())
        )
    if n < SMALL_N_FLAT:
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = LineageReport(events=())
        return flat

    try:
        # Imported inside the guard, not at function entry: iai_mcp.mosaic
        # imports numba at module scope and mosaic_policy reaches numba
        # through it. A missing or ABI-mismatched numba raises at import
        # time, which must degrade to the flat assignment below rather than
        # propagate out of detect_communities and fail the whole recall.
        from iai_mcp.mosaic import run_mosaic
        from iai_mcp.mosaic_policy import CPM_MODULARITY_FLOOR

        inner_assignment, lineage_report = run_mosaic(
            graph, prior=prior, prior_mode=prior_mode, seed=42
        )
    except (ImportError, RuntimeError, ValueError, TypeError):
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = LineageReport(events=())
        return flat

    if n < MID_N_LEIDEN and inner_assignment.modularity < CPM_MODULARITY_FLOOR:
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = LineageReport(events=())
        return flat

    if len(set(inner_assignment.node_to_community.values())) <= 1:
        flat = _flat_assignment(graph, prior)
        flat.lineage_report = lineage_report
        return flat

    inner_assignment.lineage_report = lineage_report

    counts: dict[UUID, int] = {}
    for c in inner_assignment.node_to_community.values():
        counts[c] = counts.get(c, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[
        :MAX_TOP_COMMUNITIES
    ]
    inner_assignment.top_communities = [u for u, _ in top]

    mid_regions: dict[UUID, list[UUID]] = {}
    for node, comm in inner_assignment.node_to_community.items():
        mid_regions.setdefault(comm, []).append(node)
    inner_assignment.mid_regions = mid_regions

    return inner_assignment


def needs_refresh(
    prior: CommunityAssignment, current_modularity: float
) -> bool:
    return abs(prior.modularity - current_modularity) > REFRESH_DELTA
