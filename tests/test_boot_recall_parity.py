"""Recall-correctness parity around the redundant node_payload_for_cache
removal and the post-boot warm-bundle contract.

`_build_runtime_graph_impl` now hands `graph._node_payload` (the compact,
already-populated payload) straight to `runtime_graph_cache.save()` instead
of building a parallel boxed copy. This module proves the persisted-and-
reloaded graph carries the SAME data recall reads, not just that fields are
present (`test_graph_payload_rank_inputs.py` already pins field presence).

Primary discriminator: per-node persisted-centrality equality across the
warm-served (disk-cache-reloaded) and boot-routed (DB->child streamed)
builds against the cold in-parent rebuild -- mirroring the established proxy
in `tests/test_boot_consolidated_path.py`'s own recall-correctness proof.
This was verified experimentally to be a genuine
discriminator: a scratch mutation harness that corrupts the SAVED payload's
centrality to all-zero (keeping the payload non-empty, so the verbatim-reuse
path this task's `save()` change feeds still activates) and then reloads
from that corrupted disk cache is caught by this comparison -- the same
harness applied to `recall_for_response` top-k was NOT caught (see below).

Secondary check: top-k `recall_for_response` output across the same three
paths. This is NOT a reliable discriminator on its own for a small corpus --
empirically, `recall_for_response`'s final ranked order for a ~12-node
fixture is dominated by terms other than centrality (seed selection /
budget truncation upstream of the cosine+centrality blend), so the SAME
all-zero-centrality mutation above passes through `recall_for_response`
UNCHANGED even though the underlying graph payload was proven corrupted by
the primary check. Kept as an integration smoke test (recall must not
error, must return a stable non-empty response across all three rebuild
paths) but the persisted-centrality comparison above is what actually
proves recall correctness.

Single storage driver: the seam under test is the JSON runtime-graph cache
file (encrypted, filesystem-resident) plus the in-memory graph, neither of
which is SQL-driver-visible -- node_payload never lives in the records
table. Parametrizing over stdlib/lilli would double an already non-trivial
fixture for no seam this task can name.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp import retrieve, runtime_graph_cache
from iai_mcp.pipeline import recall_for_response
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

_N_NODES = 12
_TOP_K = 6
# Asymmetric spider (one hub, arms of DISTINCT lengths) -- a bare path graph
# is mirror-symmetric and produces duplicate betweenness VALUES for
# structurally-equivalent node pairs, which is a weaker fixture than one
# with a real spread of distinct centrality values.
_HUB = 0
_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),   # arm A, length 5
    (0, 6), (6, 7), (7, 8),                     # arm B, length 3
    (0, 9), (9, 10),                            # arm C, length 2
    (0, 11),                                    # arm D, length 1
]


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake


class _FixedVecEmbedder:
    """Returns the SAME cue vector for every cue text."""

    DIM = EMBED_DIM

    def __init__(self, vec: list[float]) -> None:
        self._vec = list(vec)

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _rec(vec: list[float], now: datetime, seed: int) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=f"boot recall parity fixture node {seed}",
        aaak_index="",
        embedding=vec,
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


def _seed_spider_graph(store: MemoryStore, n: int, seed: int = 91) -> tuple[list, list[float]]:
    """An asymmetric spider (one hub, unequal-length arms) with distinct
    random unit-norm embeddings per node, so betweenness centrality carries
    a real spread of distinct values (no mirror-symmetric duplicates)."""
    rng = np.random.default_rng(seed)
    now = datetime.now(timezone.utc)
    recs = []
    for i in range(n):
        v = rng.normal(size=EMBED_DIM)
        v = (v / np.linalg.norm(v)).tolist()
        recs.append(_rec(v, now, i))
    for rec in recs:
        store.insert(rec)
    flush_record_buffer(store)
    ids = [r.id for r in recs]
    store.boost_edges(
        [(ids[a], ids[b]) for a, b in _EDGES], delta=1.0, edge_type="hebbian",
    )
    cue = rng.normal(size=EMBED_DIM)
    cue = (cue / np.linalg.norm(cue)).tolist()
    return recs, cue


def _top_k_ids(
    store: MemoryStore, graph, assignment, rich_club, embedder, session_id: str
) -> list:
    response = recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue="boot recall parity probe", session_id=session_id,
        budget_tokens=1500,
    )
    return [h.record_id for h in response.hits[:_TOP_K]]


def _centrality_map(graph) -> dict[str, float]:
    return {
        str(n): float(graph._node_payload[str(n)].get("centrality", 0.0))
        for n in graph.iter_nodes()
    }


@pytest.fixture
def store(tmp_path: Path) -> MemoryStore:
    s = MemoryStore(path=tmp_path / "store")
    s.root = tmp_path
    return s


def test_boot_recall_parity_across_warm_cold_and_boot_paths(store: MemoryStore):
    _recs, cue = _seed_spider_graph(store, _N_NODES)
    embedder = _FixedVecEmbedder(cue)

    # (1) Cold: from-scratch in-parent rebuild -- the cache-miss streaming
    # path this module exercises. This is the baseline for everything below.
    graph_cold, assignment_cold, rc_cold = retrieve.build_runtime_graph(store)
    assert getattr(store, "_warm_graph_bundle", None) is not None, (
        "build_runtime_graph must memoize a warm bundle after a cold build"
    )
    cold_centrality = _centrality_map(graph_cold)
    distinct_nonzero = {round(v, 6) for v in cold_centrality.values() if v != 0.0}
    assert len(distinct_nonzero) >= 3, (
        f"the asymmetric spider fixture must produce several DISTINCT non-zero "
        f"betweenness values (got {sorted(distinct_nonzero)}) -- otherwise this "
        f"fixture cannot discriminate a corrupted/wiped persisted centrality"
    )
    cold_ids = _top_k_ids(store, graph_cold, assignment_cold, rc_cold, embedder, "cold")
    assert cold_ids, "the cold rebuild must return a non-empty ranked top-k"

    # (2) Warm-served: drop ONLY the in-process memo, forcing the next build
    # to reload the persisted disk cache -- the save()->try_load() round
    # trip of the single-copy `graph._node_payload` this wave introduced.
    # Per-node persisted centrality is the PRIMARY, proven-discriminating
    # check (see module docstring); top-k is a secondary smoke check.
    store._warm_graph_bundle = None
    graph_warm, assignment_warm, rc_warm = retrieve.build_runtime_graph(store)
    warm_centrality = _centrality_map(graph_warm)
    assert warm_centrality == cold_centrality, (
        "warm-served (disk-cache-reloaded) per-node centrality diverges from "
        "the cold-rebuild centrality -- the node_payload round trip through "
        "save()/try_load() lost or altered the persisted centrality"
    )
    warm_ids = _top_k_ids(store, graph_warm, assignment_warm, rc_warm, embedder, "warm")
    assert warm_ids == cold_ids, (
        f"warm-served recall top-k {warm_ids} diverges from the cold-rebuild "
        f"recall top-k {cold_ids}"
    )

    # (3) Full consolidated boot sequence: DB->child streaming rebuild then
    # the in-thread reconstruct, exactly what `daemon._boot_preload_body`
    # runs.
    runtime_graph_cache.invalidate(store)
    assert getattr(store, "_warm_graph_bundle", None) is None

    result = runtime_graph_cache._rebuild_and_save_rgc(store)
    assert result.get("rebuilt") is True, f"boot rebuild did not complete: {result}"
    assert result.get("saved") is True, "boot rebuild must persist the cache"

    graph_boot, assignment_boot, rc_boot = retrieve.build_runtime_graph(store)
    assert getattr(store, "_warm_graph_bundle", None) is not None, (
        "store._warm_graph_bundle must be populated after the boot sequence "
        "completes, before preload_ready would fire in the real daemon"
    )
    boot_centrality = _centrality_map(graph_boot)
    assert boot_centrality == cold_centrality, (
        "post-boot per-node centrality regressed vs the captured pre-boot "
        "baseline on this edge-rich fixture"
    )
    boot_ids = _top_k_ids(store, graph_boot, assignment_boot, rc_boot, embedder, "boot")
    assert boot_ids == cold_ids, (
        f"post-boot recall top-k {boot_ids} regressed vs the captured "
        f"pre-boot baseline {cold_ids} on this edge-rich fixture"
    )
