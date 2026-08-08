"""Degenerate pools must not produce confident orderings, ever.

Two scenarios pinned:

  * a FULLY degenerate pool (every signal flat) serves near-equal scores
    in an order that is stable across runs, flagged flat_cosine — never a
    confident winner picked by noise;
  * a replay of the incident scenario: a corpus clustered on one topic,
    a cue from another. The ranking must flag the flat head and refuse
    to let connectivity pick confident winners.
"""
from __future__ import annotations

import numpy as np

from iai_mcp.types import EMBED_DIM

from tests.test_rank_signal_liveness import CUE_VEC, _build
from tests.test_recall_core_unit import _FakeEmbedder


def _recall(store, graph, recs, assignment, cue_vec, cue, session):
    from iai_mcp.pipeline import _recall_core

    return _recall_core(
        store=store, graph=graph, assignment=assignment,
        rich_club=[], embedder=_FakeEmbedder(vec=list(cue_vec)),
        cue=cue, session_id=session,
    )


def test_fully_degenerate_pool_serves_no_confident_winner(tmp_path, monkeypatch):
    specs = [{"vec": CUE_VEC, "text": f"degenerate twin {i}"} for i in range(10)]

    store, graph, recs, assignment = _build(tmp_path / "a", specs)
    result = _recall(store, graph, recs, assignment, CUE_VEC, "degenerate cue", "s-a")
    scores = [h.score for h in result.scored_hits]
    order_a = [str(h.record_id) for h in result.scored_hits]
    assert max(scores) - min(scores) < 1e-6, (
        f"a fully flat pool must serve near-equal scores, got spread "
        f"{max(scores) - min(scores):.6f}"
    )
    assert any(h.get("kind") == "flat_cosine" for h in result.hints)

    # The same pool served twice must come back in the same order — noise
    # must not pick different winners on different calls.
    result_b = _recall(store, graph, recs, assignment, CUE_VEC, "degenerate cue", "s-b")
    order_b = [str(h.record_id) for h in result_b.scored_hits]
    assert order_a == order_b, "degenerate ordering must be stable across calls"


def test_out_of_corpus_cue_replay(tmp_path, monkeypatch):
    def _hub_gap(dirname: str) -> tuple[float, list[dict]]:
        rng = np.random.default_rng(42)
        topic = np.zeros(EMBED_DIM)
        topic[0] = 1.0
        specs = []
        for _ in range(14):
            v = topic + rng.normal(scale=0.001, size=EMBED_DIM)
            specs.append({"vec": (v / np.linalg.norm(v)).tolist()})
        store, graph, recs, assignment = _build(tmp_path / dirname, specs)
        hub = recs[0]
        for other in recs[1:10]:
            store.boost_edges([(hub.id, other.id)], edge_type="hebbian", delta=1.0)
            graph.add_edge(hub.id, other.id, edge_type="hebbian", weight=1.0)
        cue_vec = np.zeros(EMBED_DIM)
        cue_vec[1] = 1.0
        result = _recall(
            store, graph, recs, assignment, cue_vec.tolist(),
            "how much should i charge for hosting", f"s-replay-{dirname}",
        )
        hits = {h.record_id: h.score for h in result.scored_hits}
        others = [s for rid, s in hits.items() if rid != hub.id]
        return hits[hub.id] - max(others), result.hints

    monkeypatch.setenv("IAI_MCP_COS_SPREAD_MIN", "0")
    gap_unguarded, _ = _hub_gap("off")
    monkeypatch.delenv("IAI_MCP_COS_SPREAD_MIN")
    gap_guarded, hints = _hub_gap("on")

    assert any(h.get("kind") == "flat_cosine" for h in hints), (
        "an out-of-corpus cue must be flagged: the ranking carries no "
        "similarity signal"
    )
    assert gap_unguarded > 1e-3, "fixture must give the hub a real degree edge"
    assert gap_guarded < 0.5 * gap_unguarded, (
        "the hub's connectivity advantage on an uncovered topic must be "
        f"materially dampened: unguarded {gap_unguarded:.4f} vs guarded "
        f"{gap_guarded:.4f}"
    )
