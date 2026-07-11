"""Regression: the recall path must not recompute betweenness on every recall.

The build-time graph resolves centrality exactly once (cached-fresh, child map,
or a legitimate all-zero map for an edgeless corpus). A recall against that built
graph must reuse the resolved centrality, never trigger a fresh sampled-betweenness
pass per recall.

On an edgeless corpus every node's betweenness is identically zero, so a guard that
keys the recompute on ``not np.any(centrality_arr)`` fires on EVERY recall — the
O(N) recompute then dominates p95 at scale. The at-most-once-per-build contract is
delivered by a build-stamped "centrality resolved" flag: once a graph is built, the
recall path reads the resolved map and never recomputes.

This test installs a counting spy around the lazily-imported
``centrality_for_runtime`` and asserts it fires at most once across M recalls.
"""
from __future__ import annotations

import pytest

import iai_mcp.centrality_approx as centrality_approx
from iai_mcp.pipeline import recall_for_response


@pytest.fixture(autouse=True)
def _crypto_passphrase(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-passphrase-not-secret")
    yield


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


def test_recall_does_not_recompute_centrality_per_recall(
    tmp_path, monkeypatch: pytest.MonkeyPatch
):
    """Across M recalls against one built graph the betweenness recompute fires
    at most once. RED on the per-recall-recompute guard (fires M times on an
    edgeless corpus); GREEN once the build stamps centrality as resolved."""
    from tests.test_pipeline_perf import _seed_store

    # Edgeless corpus (plain inserts, no boost_edges): betweenness is identically
    # zero, exactly the condition that makes np.any(centrality_arr) False on
    # every recall.
    store, embedder, graph, assignment, rich_club = _seed_store(
        tmp_path, n=300, seed=0,
    )

    real_centrality_for_runtime = centrality_approx.centrality_for_runtime
    call_count = {"n": 0}

    def _counting_centrality_for_runtime(g):
        call_count["n"] += 1
        return real_centrality_for_runtime(g)

    # Patch on the module object so the pipeline's lazy
    # `from iai_mcp.centrality_approx import centrality_for_runtime` binds the spy.
    monkeypatch.setattr(
        centrality_approx,
        "centrality_for_runtime",
        _counting_centrality_for_runtime,
    )

    cues = [
        "what did alice cover about auth yesterday?",
        "explain the db migration plan",
        "how does the web cache invalidation work",
        "summary of the cli subcommand changes",
        "recent network stack bug report",
        "alice notes on the indexing rebuild",
    ]
    m = len(cues)

    for cue in cues:
        recall_for_response(
            store=store,
            graph=graph,
            assignment=assignment,
            rich_club=rich_club,
            embedder=embedder,
            cue=cue,
            session_id="recompute_once",
            budget_tokens=1500,
        )

    assert call_count["n"] <= 1, (
        f"centrality_for_runtime recomputed {call_count['n']} times over {m} "
        f"recalls; the at-most-once-per-build contract requires <= 1. The recall "
        f"path must read the build-resolved centrality, not recompute per recall."
    )
