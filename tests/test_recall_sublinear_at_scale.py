"""Recall's two-hop candidate spread stays bounded independent of corpus size.

Confirmed live seam (traced from the socket dispatch): ``memory_recall`` with a
non-empty, awake store calls ``iai_mcp.pipeline.recall_for_response`` ->
``_recall_core``, which picks a small seed set (base cosine/centrality seeds
plus a confidence-gated substring widen, capped at ``MULTI_SEED_CAP``) and
spreads it through ``graph.two_hop_neighborhood_with_provenance(seeds,
top_k=5)`` -- the SAME function this module drives directly. Its own
per-hop, per-node cap (``neighbours[:top_k]``, two hops) bounds the returned
set by a function of the SEED COUNT, never of the corpus size: this is a
genuine graph-traversal call whose output COULD scale with N if the cap were
removed or broken, so measuring it is a real structural regression test, not
a restatement of a fixed slice.

(The community-gate's top-3 communities and the rich-club set are NOT probed
here: the community gate only biases the rank-fusion SCORE of already-
selected candidates (never restricts the candidate set), and the rich-club
set is DELIBERATELY sized as a fraction of corpus N by design -- neither is
the sub-linear invariant this gap targets.)
"""

from __future__ import annotations

import os
from pathlib import Path

from tests.rss_rca import corpus as rss_corpus
from tests.rss_rca.phys_footprint_harness import (
    _build_synthetic_edges,
    _read_record_ids,
)

_SEED = 42
_N = 1500
_TOP_K = 5
# Matches the live pipeline's own widened-seed ceiling (pipeline.MULTI_SEED_CAP).
_NUM_SEEDS = 6

# Closed-form worst case for a 2-hop BFS capped at top_k neighbours per node
# per hop (graph.py::two_hop_neighborhood_with_provenance): hop 1 admits at
# most _NUM_SEEDS * _TOP_K nodes; hop 2 admits at most _TOP_K more per hop-1
# node. Derived from the traversal's own cap, not measured -- neither term
# involves corpus size. NOTE (measured): at the reused bounded-cluster
# topology below (cluster size 10, from _build_synthetic_edges), the
# traversal saturates the whole 10-node cluster (returns 4 = 10 - 6 seeds)
# well before top_k's per-hop cut ever has anything left to cut -- this
# bound is therefore a non-binding structural CEILING at this scale (a
# regression that widened top_k or dropped the hop limit would still be
# masked by the smaller cluster-size ceiling). The invariant this test
# actually exercises and asserts on is the cross-scale check below: the
# SAME fixed seed cluster returns the SAME spread size at N and 2N,
# regardless of how many additional independent clusters exist elsewhere in
# the graph.
_MAX_SPREAD_SIZE = _NUM_SEEDS * _TOP_K * (1 + _TOP_K)


def _redirect_env(tmp_path: Path) -> dict[str, str | None]:
    prev = {
        k: os.environ.get(k)
        for k in (
            "HOME", "IAI_MCP_STORE", "IAI_MCP_USER_MODEL_PATH",
            "IAI_MCP_CRYPTO_PASSPHRASE", "PYTHON_KEYRING_BACKEND",
        )
    }
    os.environ["HOME"] = str(tmp_path)
    os.environ["IAI_MCP_STORE"] = str(tmp_path / ".iai-mcp" / "hippo")
    os.environ["IAI_MCP_USER_MODEL_PATH"] = str(
        tmp_path / ".iai-mcp" / "user_model.json"
    )
    if not prev["IAI_MCP_CRYPTO_PASSPHRASE"]:
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = (
            "iai-mcp-recall-sublinear-deterministic-2026"
        )
    os.environ["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    (tmp_path / ".iai-mcp").mkdir(parents=True, exist_ok=True)
    return prev


def _restore_env(prev: dict[str, str | None]) -> None:
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _spread_size_at_scale(tmp_path: Path, n: int) -> tuple[int, int]:
    """Build an n-record corpus with bounded-cluster synthetic edges, spread
    a fixed seed set (the first cluster's own members) two hops, and return
    ``(spread_size, corpus_size)``.
    """
    prev_env = _redirect_env(tmp_path)
    store = None
    try:
        from iai_mcp import retrieve
        from iai_mcp.store import MemoryStore
        from iai_mcp.store._buffers import flush_edge_buffer, flush_record_buffer

        store_path = Path(os.environ["IAI_MCP_STORE"])
        store_path.mkdir(parents=True, exist_ok=True)
        store = MemoryStore(path=store_path)

        rss_corpus.build_corpus(
            store, n=n, seed=_SEED, dim=store.embed_dim,
            varied_tags=True, with_hv_payload=True,
        )
        flush_record_buffer(store)
        # Bounded intra-cluster topology (cluster size 10, ~1.3 edges/record,
        # NO cross-cluster edges -- see _build_synthetic_edges docstring):
        # per-node degree stays constant as N grows, only the CLUSTER COUNT
        # grows. This is exactly the shape that isolates "does the traversal
        # around a fixed local neighbourhood scale with unrelated corpus
        # growth" from "does the corpus have more clusters".
        _build_synthetic_edges(store, n, edge_factor=1.3, seed=_SEED)
        flush_edge_buffer(store)

        graph, _assignment, _rich_club = retrieve.build_runtime_graph(store)

        all_ids = _read_record_ids(store)
        seeds = all_ids[:_NUM_SEEDS]
        spread = graph.two_hop_neighborhood_with_provenance(seeds, top_k=_TOP_K)
        return len(spread), len(all_ids)
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001
                pass
        _restore_env(prev_env)


def test_two_hop_spread_bounded_independent_of_corpus_size(tmp_path: Path) -> None:
    """The two-hop candidate spread from a fixed seed set stays the same size
    (and within the closed-form structural cap) whether the background
    corpus has N or 2N records -- growth comes entirely from adding more
    disconnected clusters elsewhere in the graph, never from the traversal
    itself scanning more of the corpus.
    """
    size_n, corpus_n = _spread_size_at_scale(tmp_path / "n", _N)
    size_2n, corpus_2n = _spread_size_at_scale(tmp_path / "twon", 2 * _N)

    assert corpus_n == _N
    assert corpus_2n == 2 * _N
    assert size_n > 0, "the two-hop spread returned nothing at N -- seed topology broken"
    print(
        f"two-hop spread size: N={_N} -> {size_n}, N={2 * _N} -> {size_2n} "
        f"(non-binding closed-form ceiling {_MAX_SPREAD_SIZE})"
    )

    # Non-binding structural ceiling at this topology (see the module-level
    # NOTE above _MAX_SPREAD_SIZE): a true upper bound, still asserted as a
    # sanity floor, but not the discriminating check -- the cross-scale
    # comparison below is.
    assert size_n <= _MAX_SPREAD_SIZE, (
        f"spread size {size_n} at N={_N} exceeds the closed-form structural "
        f"cap {_MAX_SPREAD_SIZE} derived from seeds({_NUM_SEEDS}) x "
        f"top_k({_TOP_K})"
    )
    assert size_2n <= _MAX_SPREAD_SIZE, (
        f"spread size {size_2n} at N={2 * _N} exceeds the closed-form "
        f"structural cap {_MAX_SPREAD_SIZE} derived from "
        f"seeds({_NUM_SEEDS}) x top_k({_TOP_K})"
    )

    # The discriminating check: candidates(2N) must not scale toward 2x
    # candidates(N) -- the seed cluster's own topology is untouched by
    # unrelated corpus growth, so the spread size is expected to be
    # numerically IDENTICAL; a generous margin (1.5x, well under a doubling)
    # keeps the assertion sound if the graph build ever introduces minor
    # cross-cluster noise.
    assert size_2n <= size_n * 1.5 + 1, (
        f"two-hop spread grew from {size_n} (N={_N}) to {size_2n} "
        f"(N={2 * _N}) -- candidate set is scaling with corpus size instead "
        f"of staying bounded by the seed/top_k structural cap"
    )
