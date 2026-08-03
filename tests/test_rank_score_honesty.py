"""The ranker must not lie about, or hide, what decided a score.

All of it found by comparing a served response against the source on a
topic the corpus barely covers:

  1. `reason` is assembled during scoring — weighted terms plus every
     multiplier that touched the score — so the printed arithmetic
     reconciles with the served number.
  2. When cosine stops discriminating (the candidate head's cos spread
     collapses), the degree term must not silently inherit the decision:
     its weight is dampened by a proportional ramp and the response
     carries a flat-cosine hint instead of confident hub-ranked noise.
  3. `_aaak_overlap` matches a natural-language cue against content
     fields only — entity anchors and doc names — never machine tokens
     like `w:e` or bookkeeping tag values like `role:user`, which would
     bias every record of one role on a common cue word.
"""
from __future__ import annotations

from tests.test_recall_core_unit import (
    _FakeEmbedder,
    _build_store_and_graph,
    _flat_assignment,
)


def test_aaak_overlap_matches_entities_not_machine_tokens() -> None:
    from iai_mcp.pipeline import _aaak_overlap

    machine_only = "W:E/R:0057543b/E:-/T:capture,role:assistant"
    assert _aaak_overlap("how much to charge for hosting on upwork", machine_only) == 0.0

    with_entities = "W:E/R:0057543b/E:upwork,hosting/T:capture,role:assistant"
    got = _aaak_overlap("upwork hosting price question", with_entities)
    assert got > 0.0, (
        "a cue sharing two entity anchors with the index must overlap; "
        f"got {got} — the overlap is matching raw machine tokens"
    )


def test_aaak_overlap_ignores_field_keys_and_hex_room() -> None:
    from iai_mcp.pipeline import _aaak_overlap

    idx = "W:E/R:0057543b/E:alice/T:doc:notes.md"
    # Field keys (w/r/e/t), the wing letter, and the hex room id must never
    # count as matches against a cue that merely mentions those letters.
    assert _aaak_overlap("w e r t 0057543b", idx) == 0.0
    assert _aaak_overlap("what did alice write", idx) > 0.0


def test_aaak_overlap_bookkeeping_tags_never_bias_by_role() -> None:
    from iai_mcp.pipeline import _aaak_overlap

    # "user" is one of the most common cue words; a role:user record must
    # not gain rank from it — bookkeeping tag values are not content.
    user_rec = "W:E/R:0057543b/E:-/T:capture,role:user,idem:9f3a2b1c"
    assistant_rec = "W:E/R:0057543b/E:-/T:capture,role:assistant,idem:9f3a2b1c"
    cue = "how should i charge the user for hosting"
    assert _aaak_overlap(cue, user_rec) == 0.0
    assert _aaak_overlap(cue, assistant_rec) == 0.0
    assert _aaak_overlap("capture the assistant reply", assistant_rec) == 0.0


def test_flat_cosine_pool_does_not_let_degree_decide(tmp_path, monkeypatch):
    from iai_mcp.pipeline import _recall_core

    monkeypatch.setenv("IAI_MCP_W_SPREAD_ACT", "0")
    store, graph, recs = _build_store_and_graph(tmp_path, n=12)
    # Give every record the same embedding as the cue: cosine carries zero
    # information about which record matches.
    embedder = _FakeEmbedder()
    flat_vec = list(embedder.embed("cue"))
    for rec in recs:
        rec.embedding = list(flat_vec)
        store.update(rec)
        graph.add_node(rec.id, community_id=None, embedding=list(flat_vec))
    # Make one record a heavy earned-edge hub.
    hub = recs[0]
    for other in recs[1:9]:
        store.boost_edges([(hub.id, other.id)], edge_type="hebbian", delta=1.0)
        graph.add_edge(hub.id, other.id, edge_type="hebbian", weight=1.0)

    result = _recall_core(
        store=store, graph=graph, assignment=_flat_assignment(recs),
        rich_club=[], embedder=embedder,
        cue="a topic this corpus knows nothing about", session_id="s-flat",
    )

    hits = {h.record_id: h for h in result.scored_hits}
    assert hub.id in hits and len(hits) > 2
    hub_score = hits[hub.id].score
    others = [h.score for rid, h in hits.items() if rid != hub.id]
    # With cosine carrying zero signal, the hub must not rise above its
    # peers on connectivity alone.
    assert hub_score <= max(others) + 1e-6, (
        f"degree decided a flat-cosine pool: hub {hub_score:.4f} "
        f"vs best other {max(others):.4f}"
    )
    assert any(h.get("kind") == "flat_cosine" for h in result.hints), (
        "a flat-cosine pool must be flagged so the caller knows the "
        "ranking carried no similarity signal"
    )


def test_flat_cosine_guard_zeroes_degree_on_a_fully_flat_pool(tmp_path, monkeypatch):
    from iai_mcp.pipeline import _recall_core

    monkeypatch.setenv("IAI_MCP_W_SPREAD_ACT", "0")

    def _hub_gap(dirname: str) -> float:
        store, graph, recs = _build_store_and_graph(tmp_path / dirname, n=12)
        embedder = _FakeEmbedder()
        flat_vec = list(embedder.embed("cue"))
        for rec in recs:
            rec.embedding = list(flat_vec)
            store.update(rec)
            graph.add_node(rec.id, community_id=None, embedding=list(flat_vec))
        hub = recs[0]
        for other in recs[1:9]:
            store.boost_edges([(hub.id, other.id)], edge_type="hebbian", delta=1.0)
            graph.add_edge(hub.id, other.id, edge_type="hebbian", weight=1.0)
        result = _recall_core(
            store=store, graph=graph, assignment=_flat_assignment(recs),
            rich_club=[], embedder=embedder,
            cue="unknown topic", session_id=f"s-{dirname}",
        )
        hits = {h.record_id: h for h in result.scored_hits}
        others = [h.score for rid, h in hits.items() if rid != hub.id]
        return hits[hub.id].score - max(others)

    monkeypatch.setenv("IAI_MCP_COS_SPREAD_MIN", "0")
    gap_unguarded = _hub_gap("off")
    monkeypatch.delenv("IAI_MCP_COS_SPREAD_MIN")
    gap_guarded = _hub_gap("on")

    assert gap_unguarded > 1e-4, "fixture must give the hub a real degree edge"
    assert gap_guarded < 0.5 * gap_unguarded, (
        f"the guard must dampen the degree advantage on a flat pool: "
        f"unguarded {gap_unguarded:.4f} vs guarded {gap_guarded:.4f}"
    )


def test_flat_cosine_damp_is_a_proportional_ramp() -> None:
    from iai_mcp.pipeline import COS_SPREAD_MIN, _flat_cosine_damp

    t = COS_SPREAD_MIN
    assert _flat_cosine_damp(0.0, t) == 0.0
    assert abs(_flat_cosine_damp(t / 4, t) - 0.25) < 1e-12
    assert abs(_flat_cosine_damp(t / 2, t) - 0.5) < 1e-12
    assert _flat_cosine_damp(t, t) == 1.0
    assert _flat_cosine_damp(t * 5, t) == 1.0
    # A disabled threshold means no dampening, never a division error.
    assert _flat_cosine_damp(0.0, 0.0) == 1.0


def test_reason_prints_community_and_spread_terms(tmp_path, monkeypatch):
    from iai_mcp.pipeline import _recall_core

    monkeypatch.setenv("IAI_MCP_COMMUNITY_BIAS", "0.2")
    store, graph, recs = _build_store_and_graph(tmp_path, n=8)
    embedder = _FakeEmbedder()

    result = _recall_core(
        store=store, graph=graph, assignment=_flat_assignment(recs),
        rich_club=[], embedder=embedder,
        cue="cue", session_id="s-reason", mode="concept",
    )

    assert result.scored_hits
    top = result.scored_hits[0]
    # rec0's embedding aligns with the cue (cos 1.0) and every record sits
    # in the single top community, so the community term contributes
    # mode_bias * cos * graded_weight > 0 — the reason must show it.
    assert "community" in top.reason, (
        f"community bias contributed to the score but reason hides it: "
        f"{top.reason!r}"
    )
