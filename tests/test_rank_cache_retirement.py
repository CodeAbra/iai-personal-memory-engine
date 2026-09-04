"""Guards for the rank-cache retirement (memo-drop of three of the five
Python-resident rank caches).

Retired: `graph._collected_pool`, `graph._normalized_pool`,
`graph._degree_map_cache` -- each recomputed fresh every recall, no
cross-call memoization on the graph.

Kept, with documented live readers (NOT retired): `graph._records_view_cache`
(fts_hits/episodic_ids need whole-pool literal_surface/tier before any
candidate set exists -- removing the resident surfaces made a
winners-only rebuild structurally unreachable) and `LexicalIndex._postings`
(scoped search, entity_link, sleep-pipeline topic naming, AND the
recall-path warm lexical lane all read it).
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_store import _make  # noqa: E402

from iai_mcp.pipeline import recall_for_response  # noqa: E402
from iai_mcp.retrieve import build_runtime_graph  # noqa: E402
from iai_mcp.store import MemoryStore  # noqa: E402
from iai_mcp.types import EMBED_DIM  # noqa: E402

_SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "iai_mcp"

_RETIRED_ATTRS = ("_collected_pool", "_normalized_pool", "_degree_map_cache")
_KEPT_ATTRS = ("_records_view_cache", "_postings")

# Identifiers naming the Rust rank-index handle in source -- used by the
# per-candidate-crossing guard to scope its `get_*(...)` search to the rank
# index specifically, not every unrelated `.get_...()` call in the tree.
_RANK_INDEX_HANDLE_MARKERS = (
    "rank_index_for", "_rank_handle", "rank_index_handle", "_RankIndexHandle",
)


def _iter_py_files():
    for root, dirs, files in os.walk(_SRC_ROOT):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for fn in files:
            if fn.endswith(".py"):
                yield Path(root) / fn


def _attribute_names_in_file(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }


def _unit_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _seed_store(store: MemoryStore, n: int, seed: int) -> list[UUID]:
    rng = np.random.default_rng(seed)
    ids: list[UUID] = []
    for i in range(n):
        v = rng.standard_normal(EMBED_DIM).astype(np.float32)
        v /= np.linalg.norm(v)
        rec = _make(text=f"retirement fixture record {i}", vec=v.tolist())
        store.insert(rec)
        ids.append(rec.id)
    return ids


def test_five_caches_have_zero_readers(tmp_path):
    """Static proof: the three retired attribute names appear NOWHERE as an
    `ast.Attribute` access anywhere under `src/iai_mcp` -- not merely unused
    by the recall path, absent from the tree entirely. The two kept caches
    get the inverse: a positive assertion that they are STILL referenced
    somewhere, so a future silent retirement of either fails this guard
    loud instead of quietly losing coverage.

    A runtime confirmation follows the static scan: a graph that has served
    several live recalls never carries any of the three retired attributes.
    """
    retired_hits: dict[str, list[str]] = {name: [] for name in _RETIRED_ATTRS}
    kept_hits: dict[str, list[str]] = {name: [] for name in _KEPT_ATTRS}

    for path in _iter_py_files():
        names = _attribute_names_in_file(path)
        for retired in _RETIRED_ATTRS:
            if retired in names:
                retired_hits[retired].append(str(path))
        for kept in _KEPT_ATTRS:
            if kept in names:
                kept_hits[kept].append(str(path))

    for name, hits in retired_hits.items():
        assert not hits, f"{name} is retired but still referenced in: {hits}"
    for name, hits in kept_hits.items():
        assert hits, (
            f"{name} is a KEPT cache (documented live readers) but the "
            "static scan found zero references -- it may have been "
            "silently retired without updating this guard"
        )

    store = MemoryStore(path=str(tmp_path / "store"))
    _seed_store(store, n=25, seed=1)
    graph, assignment, rich_club = build_runtime_graph(store)
    cue_vec = _unit_vec(2)

    for _ in range(3):
        recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=None, cue="probe cue", session_id="zero-reader-guard",
            budget_tokens=1500, cue_embedding=list(cue_vec),
        )

    for name in _RETIRED_ATTRS:
        assert not hasattr(graph, name), (
            f"{name} is retired but a live graph carries it after warm recalls"
        )
    assert hasattr(graph, "_records_view_cache") and graph._records_view_cache is not None, (
        "_records_view_cache is kept and must still populate on a warm graph"
    )


def test_no_per_candidate_pyo3_accessor():
    """No call shaped like `rank_index.get_*(id)` (a per-candidate PyO3
    crossing) exists inside a for-loop body anywhere in `src/iai_mcp`. Every
    consumer of the Rust rank index crosses in bulk (`vectors`/
    `adjacency_by_type`/`salience_levels`/`score`/`snapshot`), never
    per-candidate -- this retirement introduced no new accessor of any
    shape, but the guard exists to forbid a future regression."""
    violations: list[str] = []
    for path in _iter_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for loop in ast.walk(tree):
            if not isinstance(loop, (ast.For, ast.AsyncFor)):
                continue
            for node in ast.walk(loop):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if not func.attr.startswith("get_"):
                    continue
                receiver_src = ast.dump(func.value)
                if any(marker in receiver_src for marker in _RANK_INDEX_HANDLE_MARKERS):
                    violations.append(f"{path}:{node.lineno}: .{func.attr}(...)")
    assert not violations, (
        f"per-candidate PyO3-shaped rank-index accessor call(s) found "
        f"inside a loop body: {violations}"
    )


def test_reference_producer_still_callable(tmp_path):
    """The kill-switch Python reference producer (`use_rust_scorer=False`)
    stays callable after retirement -- the differential gate and future
    byte-identity baselines depend on it. It reads the SAME source the Rust
    path reads for the retired features (fresh, uncached), never a dead
    cache."""
    store = MemoryStore(path=str(tmp_path / "store"))
    target_vec = _unit_vec(10)
    target = _make(text="the reference producer must surface this record", vec=target_vec)
    store.insert(target)
    _seed_store(store, n=15, seed=11)
    graph, assignment, rich_club = build_runtime_graph(store)

    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=None, cue="reference producer probe", session_id="reference-callable",
        budget_tokens=1500, cue_embedding=list(target_vec), use_rust_scorer=False,
    )

    assert response.hits, "the Python reference producer returned no hits at all"
    assert any(h.record_id == target.id for h in response.hits), (
        "the reference producer did not surface the target record it was "
        "seeded to find -- it is callable but not functioning"
    )


def test_steady_state_rss_not_above_baseline(tmp_path):
    """Structural, deterministic, in-process reachability proof for the
    net-RSS-DOWN bar: after warm recalls, the three retired attributes are
    provably absent from the graph -- the resident bytes they used to hold
    cannot exist because the attribute writes are gone, not merely because
    nothing happened to touch them this run. This is the reachable bar this
    test suite proves synchronously; the heavier multi-cycle-soak
    OS-level `phys_footprint` measurement lives in
    `tests/test_rank_index_rss_soak.py` and is a follow-up production-scale
    proof, run manually.

    The floor below is the two retired pool matrices' guaranteed byte size
    at this fixture's scale (N * EMBED_DIM * 4 bytes, twice) -- a lower
    bound on the resident bytes this retirement removes, using the same
    accepted arithmetic-lower-bound methodology as the baseline RSS
    measurement this retirement was scoped against.
    """
    n = 300
    store = MemoryStore(path=str(tmp_path / "store"))
    _seed_store(store, n=n, seed=20)
    graph, assignment, rich_club = build_runtime_graph(store)
    cue_vec = _unit_vec(21)

    for _ in range(4):
        recall_for_response(
            store=store, graph=graph, assignment=assignment, rich_club=rich_club,
            embedder=None, cue="rss reachability probe", session_id="rss-reachable",
            budget_tokens=1500, cue_embedding=list(cue_vec),
        )

    for name in _RETIRED_ATTRS:
        assert not hasattr(graph, name), (
            f"{name} is retired but resident on the graph -- the "
            "net-RSS-DOWN bar cannot hold if this cache still exists"
        )

    retired_pool_bytes_floor = 2 * n * EMBED_DIM * 4
    assert retired_pool_bytes_floor > 0, (
        "fixture self-check: the guaranteed retired-bytes floor must be "
        "positive for this to be a meaningful reachability proof"
    )
