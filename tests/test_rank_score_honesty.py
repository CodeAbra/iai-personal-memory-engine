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


def test_every_record_view_construction_carries_rank_fields() -> None:
    # SimpleRecordView defaults every rank column (empty index, now(),
    # 0.5): a construction site that omits them silently kills the aaak,
    # age and stability terms for every recall it serves — and the
    # printed reason stays arithmetically consistent with the wrong
    # inputs. Every site must pass all three explicitly.
    import inspect
    import re

    from iai_mcp import core, pipeline, retrieve, runtime_graph_cache

    def _call_text(src: str, start: int) -> str:
        depth = 0
        for i in range(start - 1, len(src)):
            if src[i] == "(":
                depth += 1
            elif src[i] == ")":
                depth -= 1
                if depth == 0:
                    return src[start - 1:i]
        return src[start - 1:]

    for mod in (pipeline, retrieve):
        src = inspect.getsource(mod)
        for m in re.finditer(r"SimpleRecordView\(", src):
            call = _call_text(src, m.end())
            for field_name in ("aaak_index=", "created_at=", "stability="):
                assert field_name in call, (
                    f"{mod.__name__}: SimpleRecordView construction misses "
                    f"{field_name} — rank term goes silently dead:\n{call[:300]}"
                )

    # Same contract for every full node-payload dict: any dict literal
    # carrying "surface" must carry the rank columns too, in every module
    # that feeds the recall graph (the daemon warm path included). The
    # scan targets DICT LITERALS, not set_node_payload calls — a payload
    # built into a variable and passed by name must not escape it.
    def _dict_text(src: str, brace_start: int) -> str:
        depth = 0
        for i in range(brace_start, len(src)):
            if src[i] == "{":
                depth += 1
            elif src[i] == "}":
                depth -= 1
                if depth == 0:
                    return src[brace_start:i]
        return src[brace_start:]

    payload_dicts = 0
    for mod in (pipeline, retrieve, core, runtime_graph_cache):
        src = inspect.getsource(mod)
        for m in re.finditer(r"\{", src):
            body = _dict_text(src, m.start())
            if '"surface"' not in body or '"embedding"' not in body:
                continue
            payload_dicts += 1
            if "**_rank_fields" in body:
                continue
            for key in ('"aaak_index"', '"created_at"', '"stability"'):
                assert key in body, (
                    f"{mod.__name__}: node payload dict misses {key} — "
                    f"rank term goes silently dead:\n{body[:300]}"
                )
    assert payload_dicts >= 6, (
        f"guard self-check: expected to scan at least 6 payload dicts, "
        f"saw {payload_dicts} — the scan pattern went stale"
    )


def test_aaak_overlap_matches_inflected_name_forms() -> None:
    from iai_mcp.pipeline import _aaak_overlap

    # The anchor is stored as it appeared in text — often an inflected
    # form. The nominative cue must still match it, and vice versa.
    idx = "W:E/R:0057543b/E:зефирбота,parse_kits/T:capture"
    assert _aaak_overlap("зефирбот", idx) == 1.0
    assert _aaak_overlap("parse_kit", idx) == 1.0
    assert _aaak_overlap("что умеет зефирбот?", idx) > 0.0
    # A short stem must not fuzzy-match: "бот" is not "ботлнек".
    assert _aaak_overlap("бот", "W:E/R:x/E:ботлнек/T:capture") == 0.0
    # Nor may a long tail ride a shared prefix.
    assert _aaak_overlap("зефир", idx) == 0.0


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


def _flat_boosted_run(tmp_path, dirname: str):
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path / dirname, n=8)
    embedder = _FakeEmbedder()
    flat_vec = list(embedder.embed("cue"))
    # One record carries a doc tag: on a flat head the knowledge boost is
    # its ONLY advantage.
    boosted = recs[0]
    for rec in recs:
        rec.embedding = list(flat_vec)
        store.update(rec)
        graph.add_node(rec.id, community_id=None, embedding=list(flat_vec))
        graph.set_node_payload(rec.id, {
            "embedding": list(flat_vec),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": ["doc:notes.md"] if rec is boosted else [],
            "language": "en",
        })
    result = _recall_core(
        store=store, graph=graph, assignment=_flat_assignment(recs),
        rich_club=[], embedder=embedder,
        cue="uncovered topic", session_id=f"s-boost-{dirname}",
        mode="concept",
    )
    return boosted, result


def test_flat_head_damps_knowledge_boost(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_COMMUNITY_BIAS", "0.2")

    def _boosted_gap(dirname: str) -> float:
        boosted, result = _flat_boosted_run(tmp_path, dirname)
        hits = {h.record_id: h.score for h in result.scored_hits}
        others = [s for rid, s in hits.items() if rid != boosted.id]
        return hits[boosted.id] - max(others)

    monkeypatch.setenv("IAI_MCP_COS_SPREAD_MIN", "0")
    gap_unguarded = _boosted_gap("off")
    monkeypatch.delenv("IAI_MCP_COS_SPREAD_MIN")
    gap_guarded = _boosted_gap("on")

    assert gap_unguarded > 1e-3, (
        "fixture must give the doc-tagged record a real boost advantage"
    )
    assert gap_guarded < 0.5 * gap_unguarded, (
        "the knowledge boost must not decide a flat head: "
        f"unguarded {gap_unguarded:.4f} vs guarded {gap_guarded:.4f}"
    )


def test_flat_head_silences_community_bias(tmp_path, monkeypatch):
    # The community term is constant across a single-community flat pool,
    # so a score-gap assertion cannot see it — the served reasons can:
    # the term must vanish from every reason when the guard dampens it.
    monkeypatch.setenv("IAI_MCP_COMMUNITY_BIAS", "0.2")

    monkeypatch.setenv("IAI_MCP_COS_SPREAD_MIN", "0")
    _, unguarded = _flat_boosted_run(tmp_path, "comm-off")
    monkeypatch.delenv("IAI_MCP_COS_SPREAD_MIN")
    _, guarded = _flat_boosted_run(tmp_path, "comm-on")

    assert any("community" in h.reason for h in unguarded.scored_hits), (
        "fixture must make the community bias contribute when unguarded"
    )
    assert not any("community" in h.reason for h in guarded.scored_hits), (
        "community bias must not contribute to a fully flat head"
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
