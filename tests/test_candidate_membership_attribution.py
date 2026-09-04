"""Discovery-stage candidate-membership capture and the four-way
miss-cause classifier.

The stubbed-source tests prove per-lane attribution and multi-source
union on a store whose four discovery entry points are directly
controlled. The integration test proves the capture's reported
membership matches the REAL, unmodified production candidate pool on a
seeded store, on both storage drivers, including a genuine cap-drop (a
source returns a gold id that its own per-source cap then excludes
before the pool is assembled) -- the load-bearing distinction the
park-vs-scope-follow-up verdict hinges on.
"""
from __future__ import annotations

import pytest

from bench.fhrr_verdict_membership import (
    ALL_CAUSES,
    CAUSE_DISCOVERY_EXCLUDED,
    CAUSE_LEXICAL_COLD_OR_IDF_GATED,
    CAUSE_RANK_BUDGET_DROPPED,
    CAUSE_SPREAD_DID_NOT_REACH,
    SOURCE_ANN,
    SOURCE_EXACT_AUTHORITY,
    SOURCE_RICH_CLUB,
    SOURCE_TWO_HOP,
    GoldMembership,
    capture_discovery_membership,
    classify_miss_cause,
)
from tests.test_eval_copy_store_warm_baseline import (
    _dense_vec,
    _flush,
    _make_record,
    _reset_graph_cache_generation_epoch,  # noqa: F401 -- autouse fixture
    _select_driver,
)

_ANCHOR_ACTIVE = range(0, 20)


def _stub_sources(
    store,
    monkeypatch,
    *,
    ann_pairs=None,
    exact_pairs=None,
    hop_neighbors=None,
    rich_club_ids=None,
):
    """Directly control all four discovery entry points for one dispatch
    call -- every source not under test returns nothing, so a per-id
    attribution assertion cannot pass by real-embedding coincidence."""
    monkeypatch.setattr(store, "query_similar", lambda *a, **k: list(ann_pairs or []))
    monkeypatch.setattr(store, "exact_top_k", lambda *a, **k: list(exact_pairs or []))

    def _hop_stub(active_store, ids, top_k=5):
        return {"stub_source": list(hop_neighbors or [])}

    monkeypatch.setattr("iai_mcp.core._incident_edges_warm", _hop_stub)

    from iai_mcp import runtime_graph_cache as rgc

    real_load = rgc.load_recall_structural

    def _rc_stub(store_arg):
        assignment, _real_rc, max_degree, source, node_degrees = real_load(store_arg)
        return assignment, list(rich_club_ids or []), max_degree, source, node_degrees

    monkeypatch.setattr(rgc, "load_recall_structural", _rc_stub)


def _dispatch_params(cue_embedding) -> dict:
    return {
        "cue": "alice capture test cue",
        "cue_embedding": cue_embedding,
        "session_id": "membership-capture-gate",
        "budget_tokens": 5000,
    }


# ---------------------------------------------------------------------------
# Stubbed-source attribution (unit, no real cosine/graph dependency)
# ---------------------------------------------------------------------------


def test_capture_attributes_ann_only(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    ann_rec = _make_record("alice ann-only target", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(ann_rec)
    never_rec = _make_record("alice never discovered", _dense_vec(range(100, 120)))
    store.insert(never_rec)
    _flush(store)

    _stub_sources(store, monkeypatch, ann_pairs=[(ann_rec, 0.9)])

    membership = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))

    assert membership.sources_for(ann_rec.id) == {SOURCE_ANN}
    assert membership.survived(ann_rec.id) is True
    assert membership.sources_for(never_rec.id) == set()
    assert membership.survived(never_rec.id) is False
    store.close()


def test_capture_attributes_exact_authority_only(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    exact_rec = _make_record("alice exact-only target", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(exact_rec)
    _flush(store)

    _stub_sources(store, monkeypatch, exact_pairs=[(exact_rec.id, 0.9)])

    membership = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))

    assert membership.sources_for(exact_rec.id) == {SOURCE_EXACT_AUTHORITY}
    assert membership.survived(exact_rec.id) is True
    store.close()


def test_capture_attributes_two_hop_only(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    hop_rec = _make_record("alice two-hop-only target", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(hop_rec)
    _flush(store)

    _stub_sources(
        store, monkeypatch,
        hop_neighbors=[(hop_rec.id, "hebbian", 1.0)],
    )

    membership = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))

    assert membership.sources_for(hop_rec.id) == {SOURCE_TWO_HOP}
    assert membership.survived(hop_rec.id) is True
    store.close()


def test_capture_attributes_rich_club_only(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    rc_rec = _make_record("alice rich-club-only target", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(rc_rec)
    _flush(store)

    _stub_sources(store, monkeypatch, rich_club_ids=[rc_rec.id])

    membership = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))

    assert membership.sources_for(rc_rec.id) == {SOURCE_RICH_CLUB}
    assert membership.survived(rc_rec.id) is True
    store.close()


def test_capture_unions_multi_source_id(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    shared_rec = _make_record("alice ann and exact target", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(shared_rec)
    _flush(store)

    _stub_sources(
        store, monkeypatch,
        ann_pairs=[(shared_rec, 0.9)],
        exact_pairs=[(shared_rec.id, 0.9)],
    )

    membership = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))

    assert membership.sources_for(shared_rec.id) == {SOURCE_ANN, SOURCE_EXACT_AUTHORITY}, (
        "a gold id surfaced by two sources must union, not first-wins"
    )
    assert membership.survived(shared_rec.id) is True
    store.close()


def test_capture_reports_never_discovered_id(tmp_path, monkeypatch) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    ignored_rec = _make_record("alice unreachable", _dense_vec(range(200, 220)))
    store.insert(ignored_rec)
    _flush(store)

    _stub_sources(store, monkeypatch)

    membership = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))

    assert membership.sources_for(ignored_rec.id) == set()
    assert membership.survived(ignored_rec.id) is False
    gm = membership.gold_membership(ignored_rec.id)
    assert gm.in_post_cap_pool is False
    assert gm.capped_out is False
    store.close()


# ---------------------------------------------------------------------------
# Post-cap reconciliation integration proof (both drivers, real dispatch)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_capture_post_cap_membership_matches_real_pool(driver, tmp_path, monkeypatch) -> None:
    """A rich-club id beyond the real production top-50 cap is genuinely
    RETURNED by the source (observed via the capture) but does NOT survive
    into the real, unmodified production candidate pool -- the exact
    reconciliation the miss-cause classifier's discovery_excluded branch
    depends on. A same-source id WITHIN the cap must survive, proving this
    is a real cap boundary, not a broken discovery path."""
    _select_driver(driver, monkeypatch)
    from iai_mcp import runtime_graph_cache as rgc
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    anchor = _make_record("alice rich club anchor", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(anchor)

    rc_records = [
        _make_record(f"alice rich club member {i}", _dense_vec(range(50 + i, 70 + i)))
        for i in range(52)
    ]
    for rec in rc_records:
        store.insert(rec)
    _flush(store)

    gold = rc_records[50]  # index 50 -> the 51st id -> dropped by the top-50 cap
    survivor = rc_records[10]  # well within the cap -- must survive

    monkeypatch.setenv("IAI_MCP_EXACT_AUTHORITY_OFF", "1")
    monkeypatch.setattr("iai_mcp.pipeline.K_CANDIDATES", 1)

    real_load = rgc.load_recall_structural

    def _rc_stub(store_arg):
        assignment, _real_rc, max_degree, source, node_degrees = real_load(store_arg)
        return assignment, [r.id for r in rc_records], max_degree, source, node_degrees

    monkeypatch.setattr(rgc, "load_recall_structural", _rc_stub)

    from bench.recall_accuracy_real import warm_eval_copy_store

    warm_eval_copy_store(store)

    membership = capture_discovery_membership(
        store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)),
    )

    assert gold.id in membership.raw_sources[SOURCE_RICH_CLUB], (
        "the rich-club source must report the gold id as RETURNED (pre-cap)"
    )
    assert gold.id not in membership.post_cap_ids, (
        "a rich-club id beyond the real top-50 cap must not survive into "
        "the post-cap pool -- production's own [:50] slice drops it"
    )
    assert survivor.id in membership.post_cap_ids, (
        "a rich-club id WITHIN the cap must survive, or this is not a "
        "genuine cap boundary proof"
    )

    gm = membership.gold_membership(gold.id)
    assert gm.in_post_cap_pool is False
    assert gm.capped_out is True, (
        "a source-returned-but-cap-dropped id must reconcile as capped "
        "out, not merely absent"
    )
    assert classify_miss_cause(gm, {"in_budget_topk": False}, {}) == CAUSE_DISCOVERY_EXCLUDED

    store.close()


def test_capture_reused_store_leaves_no_instance_shadow(tmp_path, monkeypatch) -> None:
    """A store dispatched against twice (the eval harness's own
    one-store-per-run pattern), with NO test-level source stubs, must
    never accumulate a permanent instance-level shadow of its own class
    methods after capture, and the second call must still correctly
    select the post-cap pool graph -- the branch where the store-resident
    rank-builder graph already exists BEFORE this dispatch (unexercised
    by every single-dispatch test above, which all hit the store's
    first-ever dispatch)."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    rec = _make_record("alice reused-store target", _dense_vec(_ANCHOR_ACTIVE))
    store.insert(rec)
    _flush(store)

    from bench.recall_accuracy_real import warm_eval_copy_store

    warm_eval_copy_store(store)

    assert "query_similar" not in store.__dict__
    assert "exact_top_k" not in store.__dict__

    membership_1 = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))
    assert "query_similar" not in store.__dict__, (
        "restoring after capture must delete the shadow, not merely "
        "re-set it, when no instance-level attribute existed before"
    )
    assert "exact_top_k" not in store.__dict__
    assert membership_1.survived(rec.id) is True

    membership_2 = capture_discovery_membership(store, _dispatch_params(_dense_vec(_ANCHOR_ACTIVE)))
    assert "query_similar" not in store.__dict__
    assert "exact_top_k" not in store.__dict__
    assert membership_2.survived(rec.id) is True, (
        "the second dispatch must still resolve exactly one post-cap "
        "pool graph even though the store-resident rank-builder graph "
        "now already exists from the first call"
    )

    store.close()


# ---------------------------------------------------------------------------
# Four-way miss-cause classifier (synthetic pools)
# ---------------------------------------------------------------------------


def test_classify_discovery_excluded_never_returned() -> None:
    gm = GoldMembership(gold_id="x", in_post_cap_pool=False)
    assert classify_miss_cause(gm, {}, {}) == CAUSE_DISCOVERY_EXCLUDED


def test_classify_discovery_excluded_capped_out() -> None:
    gm = GoldMembership(
        gold_id="x", in_post_cap_pool=False,
        returned_by_sources=frozenset({SOURCE_RICH_CLUB}),
        capped_out=True,
    )
    assert classify_miss_cause(gm, {}, {}) == CAUSE_DISCOVERY_EXCLUDED


def test_classify_spread_did_not_reach() -> None:
    gm = GoldMembership(
        gold_id="x", in_post_cap_pool=False, spread_edge_missing=True,
    )
    assert classify_miss_cause(gm, {}, {}) == CAUSE_SPREAD_DID_NOT_REACH


def test_classify_rank_budget_dropped() -> None:
    gm = GoldMembership(gold_id="x", in_post_cap_pool=True)
    assert classify_miss_cause(gm, {"in_budget_topk": False}, {}) == CAUSE_RANK_BUDGET_DROPPED


def test_classify_lexical_cold_or_idf_gated() -> None:
    gm = GoldMembership(gold_id="x", in_post_cap_pool=False)
    lexical_context = {"lexical_only_rescue": True, "lexical_contributed": False}
    assert classify_miss_cause(gm, {}, lexical_context) == CAUSE_LEXICAL_COLD_OR_IDF_GATED


def test_classify_lexical_rescue_that_contributed_is_not_lexical_gated() -> None:
    """A lexical-only-rescue miss whose lexical lane DID contribute must
    not be misclassified as gated -- it falls through to discovery
    excluded (the lexical lane is not the failing mechanism here)."""
    gm = GoldMembership(gold_id="x", in_post_cap_pool=False)
    lexical_context = {"lexical_only_rescue": True, "lexical_contributed": True}
    assert classify_miss_cause(gm, {}, lexical_context) == CAUSE_DISCOVERY_EXCLUDED


def test_classify_raises_when_not_a_miss() -> None:
    gm = GoldMembership(gold_id="x", in_post_cap_pool=True)
    with pytest.raises(ValueError):
        classify_miss_cause(gm, {"in_budget_topk": True}, {})


def test_classify_capped_out_takes_priority_over_spread_signal() -> None:
    """A capped-out id must classify as discovery_excluded even when a
    (spurious) spread_edge_missing signal is also set -- the cap-drop
    reconciliation is the load-bearing rule, not the edge signal."""
    gm = GoldMembership(
        gold_id="x", in_post_cap_pool=False,
        returned_by_sources=frozenset({SOURCE_TWO_HOP}),
        capped_out=True,
        spread_edge_missing=True,
    )
    assert classify_miss_cause(gm, {}, {}) == CAUSE_DISCOVERY_EXCLUDED


def test_classify_all_four_categories_are_distinct_and_reachable() -> None:
    scenarios = {
        CAUSE_DISCOVERY_EXCLUDED: (
            GoldMembership(gold_id="a", in_post_cap_pool=False), {}, {},
        ),
        CAUSE_SPREAD_DID_NOT_REACH: (
            GoldMembership(gold_id="b", in_post_cap_pool=False, spread_edge_missing=True),
            {}, {},
        ),
        CAUSE_RANK_BUDGET_DROPPED: (
            GoldMembership(gold_id="c", in_post_cap_pool=True),
            {"in_budget_topk": False}, {},
        ),
        CAUSE_LEXICAL_COLD_OR_IDF_GATED: (
            GoldMembership(gold_id="d", in_post_cap_pool=False),
            {}, {"lexical_only_rescue": True, "lexical_contributed": False},
        ),
    }
    seen = set()
    for expected_cause, (gm, rank_ctx, lex_ctx) in scenarios.items():
        cause = classify_miss_cause(gm, rank_ctx, lex_ctx)
        assert cause == expected_cause
        seen.add(cause)
    assert seen == set(ALL_CAUSES), "every canonical scenario must map to a distinct category"


def test_classify_discriminates_on_flipped_input() -> None:
    """Flipping a single membership field changes the classification --
    proving the function actually discriminates rather than returning a
    hard-coded category."""
    gm_excluded = GoldMembership(gold_id="x", in_post_cap_pool=False)
    assert classify_miss_cause(gm_excluded, {}, {}) == CAUSE_DISCOVERY_EXCLUDED

    gm_spread = GoldMembership(gold_id="x", in_post_cap_pool=False, spread_edge_missing=True)
    assert classify_miss_cause(gm_spread, {}, {}) == CAUSE_SPREAD_DID_NOT_REACH

    gm_in_pool = GoldMembership(gold_id="x", in_post_cap_pool=True)
    assert classify_miss_cause(gm_in_pool, {"in_budget_topk": False}, {}) == CAUSE_RANK_BUDGET_DROPPED
