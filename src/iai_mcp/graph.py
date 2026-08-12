from __future__ import annotations

import os
from typing import Any, Iterable, Iterator
from uuid import UUID

import numpy as np


AUTO_CACHE_DEFAULT = "on"


def _compact_embedding(embedding: "Iterable[float] | None") -> "np.ndarray | None":
    """Node embeddings live as a contiguous float32 buffer, never a Python
    list: a boxed list of 384 floats costs ~15 KB per node against ~1.6 KB
    for the buffer, and a corpus-scale graph is built in BOTH the parent and
    the community-detection child — the boxed form put the pair near a
    gigabyte and drove the memory watchdog's kill during consolidation.

    Returns None for an absent or empty embedding so callers can keep
    treating "no embedding" as a single condition.
    """
    if embedding is None:
        return None
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.size == 0:
        return None
    # Own the memory: the caller's buffer (a pipe frame, a decrypt buffer)
    # may be reused or freed under us.
    return np.array(arr, dtype=np.float32, copy=True)


class MemoryGraph:

    def __init__(self) -> None:
        self._adj: dict[str, dict[str, dict[str, Any]]] = {}
        self._attrs: dict[UUID, dict[str, Any]] = {}
        self._node_payload: dict[str, dict[str, Any]] = {}
        self._centrality_cache: dict[UUID, float] | None = None
        self._dirty_since_centrality: bool = True
        self._centrality_resolved: bool = False
        self._normalized_pool: tuple[Any, np.ndarray] | None = None
        # Raw collected pool (id sequence + embedding matrix) cached per content
        # version; reused across recalls over the same build so the full node-set
        # iteration + matrix construction runs once, not every recall.
        self._collected_pool: tuple[int, list, np.ndarray] | None = None
        # Monotonic counter bumped by every embedding-affecting mutator. Folded
        # into the normalized-pool cache key so a content change with an
        # unchanged id-sequence still invalidates the cache — making a stale
        # cosine structurally impossible, not invariant-dependent.
        self._pool_content_version: int = 0


    def clear_and_rebuild(
        self,
        nodes: Iterable[tuple[UUID, UUID | None, list[float], dict[str, Any]]],
        edges: Iterable[tuple[UUID, UUID, float, str]],
    ) -> None:
        """Repopulate the adjacency structure in place from scratch.

        The three core containers are cleared in place (their objects are kept
        so freed value sub-dicts are returned to the existing heap arenas for
        reuse) and refilled exclusively through the public mutators, so the end
        state is identical to a fresh instance fed the same mutator sequence.

        All derived/memoized state is invalidated FIRST so a reused instance can
        never serve stale centrality, label order, or community results:
        the centrality cache is dropped, the dirty flag is raised, and the lazily
        built CSR label order is deleted.
        """
        self._centrality_cache = None
        self._dirty_since_centrality = True
        self._centrality_resolved = False
        self._normalized_pool = None
        self._pool_content_version += 1
        if hasattr(self, "_node_ids_csr_order"):
            del self._node_ids_csr_order

        self._adj.clear()
        self._attrs.clear()
        self._node_payload.clear()

        for node_id, community_id, embedding, payload in nodes:
            self.add_node(node_id, community_id=community_id, embedding=embedding)
            self.set_node_payload(node_id, payload)

        for src, dst, weight, edge_type in edges:
            self.add_edge(src, dst, weight=weight, edge_type=edge_type)

    def node_count(self) -> int:
        return len(self._adj)

    def has_node(self, node_id: UUID | str) -> bool:
        return str(node_id) in self._adj


    def add_node(
        self,
        node_id: UUID,
        community_id: UUID | None,
        embedding: list[float],
    ) -> None:
        label = str(node_id)
        self._adj.setdefault(label, {})
        self._attrs[node_id] = {
            "community_id": community_id,
        }
        self._node_payload[label] = {
            "embedding": _compact_embedding(embedding),
        }
        self._dirty_since_centrality = True
        self._normalized_pool = None
        self._pool_content_version += 1

    def set_node_payload(
        self, node_id: UUID | str, payload: dict[str, Any]
    ) -> None:
        key = str(node_id)
        existing = self._node_payload.get(key, {})
        merged = dict(existing)
        for k, v in payload.items():
            # Every entry point normalizes, so no writer can reintroduce the
            # boxed-list form through the payload door.
            merged[k] = _compact_embedding(v) if k == "embedding" else v
        self._node_payload[key] = merged
        # The pool matrix is built from node embeddings; a payload change can
        # alter an embedding, so the cached normalized pool must be dropped.
        self._normalized_pool = None
        self._pool_content_version += 1

    def set_node_centrality(self, node_id: UUID | str, value: float) -> None:
        self.set_node_payload(node_id, {"centrality": float(value)})

    def remove_node(self, node_id: UUID | str) -> None:
        label = str(node_id)
        if label in self._adj:
            for neighbor_label in list(self._adj[label].keys()):
                if neighbor_label == label:
                    continue
                self._adj[neighbor_label].pop(label, None)
            del self._adj[label]
        if isinstance(node_id, UUID):
            self._attrs.pop(node_id, None)
        else:
            try:
                self._attrs.pop(UUID(label), None)
            except (TypeError, ValueError):
                pass
        self._node_payload.pop(label, None)
        self._dirty_since_centrality = True
        self._normalized_pool = None
        self._pool_content_version += 1

    def add_edge(
        self,
        src: UUID,
        dst: UUID,
        weight: float = 1.0,
        edge_type: str = "hebbian",
    ) -> None:
        u, v = str(src), str(dst)
        self._adj.setdefault(u, {})
        self._adj.setdefault(v, {})
        attrs = {"weight": float(weight), "edge_type": str(edge_type)}
        self._adj[u][v] = attrs
        if u != v:
            self._adj[v][u] = attrs
        self._dirty_since_centrality = True
        self._normalized_pool = None
        self._pool_content_version += 1


    def centrality(self) -> dict[UUID, float]:
        env_mode = os.environ.get("IAI_MCP_CENTRALITY_CACHE", "auto").lower()
        effective_mode = (
            AUTO_CACHE_DEFAULT if env_mode == "auto" else env_mode
        )

        if (
            effective_mode == "on"
            and self._centrality_cache is not None
            and not self._dirty_since_centrality
        ):
            return self._centrality_cache

        from iai_mcp_native import graph as _native

        indptr, indices, _data_discarded = self.to_csr_arrays()
        n_nodes = len(indptr) - 1

        self._node_ids_csr_order: list[UUID] = sorted(
            self.iter_nodes(), key=str
        )

        centrality_arr, node_arr = _native.betweenness_centrality(
            indptr, indices, n_nodes, normalized=True
        )
        result: dict[UUID, float] = {
            self._node_ids_csr_order[int(idx)]: float(val)
            for idx, val in zip(node_arr, centrality_arr)
        }
        if effective_mode != "off":
            self._centrality_cache = result
            self._dirty_since_centrality = False
        return result

    TRANSFER_EDGE_TYPES: frozenset = frozenset({"entity_shared"})
    """Edge types that carry activation transfer at rank. Entity anchors are
    the designed associative bridge; letting the dense similarity/hebbian
    mesh carry transfer floods rank with near-neighbours of the seeds and
    displaces true evidence (measured on turn-granularity corpora)."""

    def two_hop_neighborhood(
        self, seeds: list[UUID], top_k: int = 5
    ) -> list[UUID]:
        return sorted(
            self.two_hop_neighborhood_with_provenance(seeds, top_k), key=str
        )

    def two_hop_neighborhood_with_provenance(
        self, seeds: list[UUID], top_k: int = 5
    ) -> "dict[UUID, tuple[UUID, int, bool]]":
        """Same traversal as two_hop_neighborhood, additionally mapping each
        reached node to (originating seed, hop depth, transfer-carrying path).
        A path carries transfer only when EVERY hop is a TRANSFER_EDGE_TYPES
        edge. First discoverer in the deterministic sweep wins the
        attribution — the reached SET must stay byte-identical to the plain
        variant."""
        visited: set[str] = {str(s) for s in seeds}
        frontier: set[str] = {str(s) for s in seeds if str(s) in self._adj}
        origin: dict[str, str] = {str(s): str(s) for s in seeds}
        carries: dict[str, bool] = {str(s): True for s in seeds}
        collected: dict[str, tuple[str, int, bool]] = {}

        for hop in (1, 2):
            next_frontier: set[str] = set()
            # Deterministic frontier order: which neighbours win the top_k cut and
            # the visited race depends on processing order, so a hash-randomized
            # set sweep makes the candidate set differ across processes. Recall
            # reproducibility requires a stable order.
            for node in sorted(frontier):
                if node not in self._adj:
                    continue
                neighbours = [
                    (n, float(attrs.get("weight", 1.0)), attrs)
                    for n, attrs in self._adj[node].items()
                ]
                # Weight descending, then node id ascending, so weight ties break
                # reproducibly rather than by dict insertion order.
                neighbours.sort(key=lambda x: (-x[1], x[0]))
                for n, _, attrs in neighbours[:top_k]:
                    if n not in visited:
                        next_frontier.add(n)
                        hop_carries = (
                            carries[node]
                            and attrs.get("edge_type") in self.TRANSFER_EDGE_TYPES
                        )
                        collected[n] = (origin[node], hop, hop_carries)
                        origin[n] = origin[node]
                        carries[n] = hop_carries
                        visited.add(n)
            frontier = next_frontier
            if not frontier:
                break

        return {
            UUID(n): (UUID(seed), hop, ok)
            for n, (seed, hop, ok) in collected.items()
        }

    def rich_club_coefficient(self, k_threshold: int | None = None) -> float:
        edges_no_selfloop: list[tuple[UUID, UUID]] = [
            (u, v) for u, v, _w in self.iter_edges_with_weight() if u != v
        ]
        if not edges_no_selfloop:
            return 0.0
        degrees: dict[UUID, int] = {nid: 0 for nid in self.iter_nodes()}
        for u, v in edges_no_selfloop:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        if k_threshold is None:
            deg_values = list(degrees.values())
            if not deg_values:
                return 0.0
            k_threshold = int(np.percentile(deg_values, 90))
        n_gt_k = sum(1 for d in degrees.values() if d > k_threshold)
        if n_gt_k < 2:
            return 0.0
        e_gt_k = sum(
            1
            for u, v in edges_no_selfloop
            if degrees.get(u, 0) > k_threshold
            and degrees.get(v, 0) > k_threshold
        )
        return 2.0 * e_gt_k / (n_gt_k * (n_gt_k - 1))


    def iter_nodes(self) -> Iterator[UUID]:
        for label in self._adj:
            yield UUID(label)

    def nodes(self) -> Iterator[UUID]:
        return self.iter_nodes()

    # Edge types INFERRED at insert/consolidation rather than earned by
    # use: they widen spread reach but must not manufacture ranking hubs.
    RANKING_DEGREE_EXCLUDED: "frozenset[str]" = frozenset({
        "pattern_separation_seed",
        "entity_shared",
    })

    def iter_edges_with_weight(
        self,
    ) -> Iterator[tuple[UUID, UUID, float]]:
        for u_label, neighbors in self._adj.items():
            for v_label, attrs in neighbors.items():
                if u_label <= v_label:
                    try:
                        weight = float(attrs.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    yield UUID(u_label), UUID(v_label), weight

    def iter_edges_with_weight_and_type(
        self,
    ) -> Iterator[tuple[UUID, UUID, float, str]]:
        for u_label, neighbors in self._adj.items():
            for v_label, attrs in neighbors.items():
                if u_label <= v_label:
                    try:
                        weight = float(attrs.get("weight", 1.0))
                    except (TypeError, ValueError):
                        weight = 1.0
                    yield (
                        UUID(u_label), UUID(v_label), weight,
                        str(attrs.get("edge_type", "hebbian")),
                    )

    def degrees(
        self,
        edge_types: "frozenset[str] | None" = None,
        exclude_types: "frozenset[str] | None" = None,
    ) -> Iterator[tuple[UUID, int]]:
        if edge_types is None and exclude_types is None:
            for label, neighbors in self._adj.items():
                yield UUID(label), len(neighbors)
            return
        for label, neighbors in self._adj.items():
            count = 0
            for attrs in neighbors.values():
                etype = attrs.get("edge_type")
                if edge_types is not None and etype not in edge_types:
                    continue
                if exclude_types is not None and etype in exclude_types:
                    continue
                count += 1
            yield UUID(label), count

    def to_csr_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        labels: list[str] = sorted(self._adj)
        n = len(labels)
        if n == 0:
            return (
                np.zeros(1, dtype=np.int64),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float64),
            )
        idx_map: dict[str, int] = {label: i for i, label in enumerate(labels)}
        rows: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for u_label, neighbors in self._adj.items():
            a = idx_map[u_label]
            for v_label, attrs in neighbors.items():
                if u_label == v_label:
                    continue
                try:
                    w = float(attrs.get("weight", 1.0))
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(w) or w < 0.0:
                    continue
                b = idx_map.get(v_label)
                if b is None:
                    continue
                rows[a].append((b, w))
        for i in range(n):
            rows[i].sort(key=lambda pair: pair[0])
        indptr = np.zeros(n + 1, dtype=np.int64)
        for i in range(n):
            indptr[i + 1] = indptr[i] + len(rows[i])
        nnz = int(indptr[-1])
        indices = np.empty(nnz, dtype=np.int64)
        data_arr = np.empty(nnz, dtype=np.float64)
        cursor = 0
        for i in range(n):
            for col, w in rows[i]:
                indices[cursor] = col
                data_arr[cursor] = w
                cursor += 1
        return indptr, indices, data_arr


    def get_embedding(self, node_id: UUID | str) -> "np.ndarray | None":
        """The node's embedding as a float32 buffer, or None when absent.

        Callers must test for None (or length) — an array has no truth
        value.
        """
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return None
        emb = payload.get("embedding")
        if emb is None:
            return None
        return emb if len(emb) else None

    def get_centrality(self, node_id: UUID | str) -> float:
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return 0.0
        try:
            return float(payload.get("centrality", 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def get_payload(self, node_id: UUID | str) -> dict[str, Any]:
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return {}
        return dict(payload)
