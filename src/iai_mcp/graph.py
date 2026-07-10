from __future__ import annotations

import os
from typing import Any, Iterable, Iterator
from uuid import UUID

import numpy as np


AUTO_CACHE_DEFAULT = "on"


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
            "embedding": list(embedding),
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
            merged[k] = v
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

    def two_hop_neighborhood(
        self, seeds: list[UUID], top_k: int = 5
    ) -> list[UUID]:
        visited: set[str] = {str(s) for s in seeds}
        frontier: set[str] = {str(s) for s in seeds if str(s) in self._adj}
        collected: set[str] = set()

        for _ in range(2):
            next_frontier: set[str] = set()
            for node in frontier:
                if node not in self._adj:
                    continue
                neighbours = [
                    (n, float(attrs.get("weight", 1.0)))
                    for n, attrs in self._adj[node].items()
                ]
                neighbours.sort(key=lambda x: x[1], reverse=True)
                for n, _ in neighbours[:top_k]:
                    if n not in visited:
                        next_frontier.add(n)
                        collected.add(n)
                        visited.add(n)
            frontier = next_frontier
            if not frontier:
                break

        return [UUID(n) for n in collected]

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

    def degrees(self) -> Iterator[tuple[UUID, int]]:
        for label, neighbors in self._adj.items():
            yield UUID(label), len(neighbors)

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


    def get_embedding(self, node_id: UUID | str) -> list[float] | None:
        payload = self._node_payload.get(str(node_id))
        if not payload:
            return None
        emb = payload.get("embedding")
        return emb if emb else None

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
