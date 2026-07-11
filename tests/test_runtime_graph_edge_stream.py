"""build_runtime_graph's edge stream must never materialize a full-table
pandas DataFrame for the edges table.

Two contracts are pinned here:

  1. No-DataFrame: `HippoTable.to_pandas` is never called for the "edges"
     table while building the runtime graph (spy-asserted).
  2. Equivalence: the graph's edge set built by `build_runtime_graph` is
     identical (same src/dst/edge_type/weight, as a multiset) to a
     `to_pandas().iterrows()` reference computed directly against the same
     store. This holds under both storage drivers.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import retrieve
from iai_mcp.hippo._table import HippoTable
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer
from iai_mcp.types import MemoryRecord


_EMBED_DIM = 32


@pytest.fixture(autouse=True)
def _small_embed_dim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(_EMBED_DIM))


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "store")


def _make_rec(seed: int) -> MemoryRecord:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(_EMBED_DIM).astype(np.float32)
    vec /= np.linalg.norm(vec)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"edge stream fixture record {seed}",
        aaak_index="",
        embedding=vec.tolist(),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _flush(store: MemoryStore) -> None:
    flush_record_buffer(store)
    flush_edge_buffer(store)


def _seed_store_with_edges(store: MemoryStore, n: int = 40) -> list[UUID]:
    """Seed ~n records plus ~60 edges across several edge_types.

    Includes duplicate weights (tie material) and edges touching the same
    node from multiple directions so the graph's undirected symmetrization
    is exercised the same way the pandas path exercises it today.
    """
    ids: list[UUID] = []
    for i in range(n):
        rec = _make_rec(i)
        store.insert(rec)
        ids.append(rec.id)
    _flush(store)

    # hebbian backbone — a connected path, tie-heavy weight (all 1.0).
    backbone = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    store.boost_edges(backbone, delta=1.0, edge_type="hebbian")

    # hebbian chords — extra long-range links, distinct weight (2.0) so both
    # weight values are represented among hebbian edges.
    rng = np.random.default_rng(99)
    chord_pairs: list[tuple[UUID, UUID]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for i in range(len(backbone)):
        seen_pairs.add(tuple(sorted((str(backbone[i][0]), str(backbone[i][1])))))
    while len(chord_pairs) < 8:
        a, b = int(rng.integers(0, n)), int(rng.integers(0, n))
        if a == b:
            continue
        key = tuple(sorted((str(ids[a]), str(ids[b]))))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        chord_pairs.append((ids[a], ids[b]))
    store.boost_edges(chord_pairs, delta=2.0, edge_type="hebbian")

    # contradicts — a distinct edge_type on fresh (unseen) pairs.
    for a_idx, b_idx in [(0, 5), (2, 9), (3, 11)]:
        store.add_contradicts_edge(ids[a_idx], ids[b_idx])

    # consolidated_from — a third edge_type, again on fresh pairs.
    cons_pairs: list[tuple[UUID, UUID]] = [
        (ids[1], ids[20]),
        (ids[4], ids[25]),
        (ids[6], ids[30]),
    ]
    store.boost_edges(cons_pairs, delta=1.5, edge_type="consolidated_from")

    _flush(store)
    return ids


def _norm_pair(a: str, b: str) -> tuple[str, str]:
    """The graph symmetrizes every edge (both directions share the same attrs
    dict), so comparisons must be orientation-independent — normalize by
    sorting the endpoint pair rather than trusting the stored src/dst order."""
    return (a, b) if a <= b else (b, a)


def _pandas_reference_edges(store: MemoryStore) -> list[tuple[str, str, str, float]]:
    """The pre-existing pandas path, computed directly in the test — pins
    today's exact semantics so the streaming rewrite cannot drift."""
    edges_df = store.db.open_table("edges").to_pandas()
    ref: list[tuple[str, str, str, float]] = []
    for _, row in edges_df.iterrows():
        a, b = _norm_pair(str(row["src"]), str(row["dst"]))
        ref.append((a, b, str(row["edge_type"]), float(row["weight"])))
    return sorted(ref)


def _graph_edges(graph) -> list[tuple[str, str, str, float]]:
    """Extract (src, dst, edge_type, weight) triples from a built MemoryGraph.

    ``_adj`` is undirected/symmetrized by ``add_edge`` (both directions carry
    the identical attrs dict), so dedup with the same ``u <= v`` rule the
    graph's own ``iter_edges_with_weight`` uses to avoid double-counting.
    """
    out: list[tuple[str, str, str, float]] = []
    for u_label, neighbors in graph._adj.items():
        for v_label, attrs in neighbors.items():
            if u_label <= v_label and attrs:
                out.append(
                    (u_label, v_label, str(attrs.get("edge_type")), float(attrs.get("weight", 1.0)))
                )
    return sorted(out)


def test_edge_stream_no_pandas_materialization(
    store: MemoryStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_store_with_edges(store)

    called_tables: list[str] = []
    original_to_pandas = HippoTable.to_pandas

    def _spy_to_pandas(self):
        called_tables.append(self._name)
        return original_to_pandas(self)

    monkeypatch.setattr(HippoTable, "to_pandas", _spy_to_pandas)

    retrieve.build_runtime_graph(store)  # graph, assignment, rich_club

    assert "edges" not in called_tables, (
        "build_runtime_graph must not call HippoTable.to_pandas for the "
        f"edges table; observed to_pandas calls: {called_tables}"
    )


def test_edge_stream_equivalence(store: MemoryStore) -> None:
    _seed_store_with_edges(store)

    reference = _pandas_reference_edges(store)
    assert reference, "fixture must seed at least one edge"

    graph, _assignment, _rich_club = retrieve.build_runtime_graph(store)

    built = _graph_edges(graph)
    assert built == reference, (
        "build_runtime_graph's edge set diverges from the pandas reference:\n"
        f"reference-only: {sorted(set(reference) - set(built))}\n"
        f"built-only: {sorted(set(built) - set(reference))}"
    )


def test_edge_stream_empty_edges_table(store: MemoryStore) -> None:
    for i in range(5):
        store.insert(_make_rec(i))
    _flush(store)

    graph, _assignment, _rich_club = retrieve.build_runtime_graph(store)

    assert _graph_edges(graph) == []
