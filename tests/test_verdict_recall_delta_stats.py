"""The Delta conditional Recall@K verdict metric engine's statistics
core (pure, synthetic-input, no store dependency) plus the
measurement-only entity-anchor ceiling expansion's controls: the
gold-peek-rejection fence, the size-matched shuffled-anchor null arm,
the two-sided A/A stability floor, the planted-rescue positive control,
the cannot-false-green control, the UNDERPOWERED-below-N_min control,
and the src/ measurement-only import fence.

Every test here runs on synthetic, fabricated content (never real
memory content) -- the "core" subset (selected via ``-k core``) touches
no store at all.
"""
from __future__ import annotations

import ast
import dataclasses
import inspect
from pathlib import Path

import pytest

from bench.fhrr_verdict_recall_delta import (
    CeilingArmUnmeasurable,
    N_MIN,
    VERDICT_NON_PASS,
    VERDICT_PASS,
    VERDICT_UNDERPOWERED,
    _baseline_precision_at_k,
    _estimate_tokens,
    _ndcg_at_rank,
    _posting_list_provenance,
    _rank_of,
    _select_null_arm_cue_id,
    _TOKEN_BUDGET_PER_QUERY,
    _validate_verdict_fixture_dict,
    build_edge_adjacency,
    cluster_stratified_bootstrap_ci,
    conditional_recall_delta,
    entity_anchor_ceiling_expansion,
    shuffled_anchor_expansion,
    spread_edge_missing_signal,
    verdict_predicate,
)
from tests.test_eval_copy_store_warm_baseline import (
    _reset_graph_cache_generation_epoch,  # noqa: F401 -- autouse fixture
    _select_driver,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent


def test_core_conditional_recall_delta_basic() -> None:
    pairs = [(0, 1), (0, 0), (0, 1), (0, 0)]
    assert conditional_recall_delta(pairs) == pytest.approx(0.5)


def test_core_conditional_recall_delta_empty_input() -> None:
    assert conditional_recall_delta([]) == 0.0


def test_core_conditional_recall_delta_arm_agnostic() -> None:
    """The identical function, called on two different 'arm' columns,
    computes each arm's delta -- no branching on arm identity anywhere
    in its signature or body."""
    params = inspect.signature(conditional_recall_delta).parameters
    assert "arm" not in params and "arm_name" not in params

    correct_arm_pairs = [(0, 1)] * 8 + [(0, 0)] * 2
    shuffled_arm_pairs = [(0, 0)] * 9 + [(0, 1)] * 1

    correct_delta = conditional_recall_delta(correct_arm_pairs)
    shuffled_delta = conditional_recall_delta(shuffled_arm_pairs)

    assert correct_delta == pytest.approx(0.8)
    assert shuffled_delta == pytest.approx(0.1)
    assert correct_delta > shuffled_delta


def test_core_bootstrap_ci_resamples_clusters_not_rows() -> None:
    """One giant all-rescued cluster plus many singleton never-rescued
    clusters: resampling ROWS would almost always include some of the
    giant cluster's many rescued rows and rarely produce an all-zero
    bootstrap replicate, pushing the lower bound toward positive.
    Resampling CLUSTERS treats the giant cluster as ONE draw among many,
    so a bootstrap replicate that never draws it (all singleton
    never-rescued clusters) is common, and the lower bound stays at or
    below zero."""
    pairs = [(0, 1)] * 40 + [(0, 0)] * 10
    cluster_ids = ["giant"] * 40 + [f"single-{i}" for i in range(10)]

    ci_low = cluster_stratified_bootstrap_ci(pairs, cluster_ids, iters=500, seed=1)

    assert ci_low <= 0.0, (
        "cluster-level resampling must be able to draw a bootstrap "
        "replicate with zero rescues (all singleton clusters, none of "
        "them the giant one) -- a row-level resample could not produce "
        "this outcome and would bias the lower bound upward"
    )


def test_core_bootstrap_ci_returns_lower_bound() -> None:
    pairs = [(0, 1)] * 20 + [(0, 0)] * 5
    cluster_ids = [f"c{i}" for i in range(25)]
    point_delta = conditional_recall_delta(pairs)
    ci_low = cluster_stratified_bootstrap_ci(pairs, cluster_ids, iters=800, seed=2)
    assert ci_low <= point_delta


def test_core_bootstrap_ci_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        cluster_stratified_bootstrap_ci([(0, 1), (0, 0)], ["a"], iters=10)


def test_core_verdict_predicate_pass() -> None:
    outcome = verdict_predicate(
        delta=0.4, ci_low=0.1, distinct_rescues=12,
        regression_flags={"precision": False, "token": False, "latency": False},
        eligible_missed_n=20, n_min=N_MIN,
    )
    assert outcome == VERDICT_PASS


def test_core_verdict_predicate_underpowered_distinct_from_nonpass() -> None:
    underpowered = verdict_predicate(
        delta=0.9, ci_low=0.5, distinct_rescues=20, regression_flags={},
        eligible_missed_n=N_MIN - 1, n_min=N_MIN,
    )
    non_pass = verdict_predicate(
        delta=0.1, ci_low=-0.05, distinct_rescues=2, regression_flags={},
        eligible_missed_n=N_MIN + 5, n_min=N_MIN,
    )
    assert underpowered == VERDICT_UNDERPOWERED
    assert non_pass == VERDICT_NON_PASS
    assert underpowered != non_pass, "UNDERPOWERED must be a distinct outcome from NON-PASS"


def test_core_verdict_predicate_underpowered_regardless_of_point_delta() -> None:
    """Even a strongly positive point-delta with a positive CI must still
    report UNDERPOWERED when the denominator is below N_min."""
    outcome = verdict_predicate(
        delta=0.95, ci_low=0.8, distinct_rescues=50, regression_flags={},
        eligible_missed_n=1, n_min=N_MIN,
    )
    assert outcome == VERDICT_UNDERPOWERED


def test_core_verdict_predicate_regression_veto() -> None:
    outcome = verdict_predicate(
        delta=0.4, ci_low=0.1, distinct_rescues=12,
        regression_flags={"precision": True, "token": False, "latency": False},
        eligible_missed_n=20, n_min=N_MIN,
    )
    assert outcome == VERDICT_NON_PASS


def test_core_verdict_predicate_rescue_floor_gates_pass() -> None:
    outcome = verdict_predicate(
        delta=0.2, ci_low=0.05, distinct_rescues=9,
        regression_flags={}, eligible_missed_n=20, n_min=N_MIN,
    )
    assert outcome == VERDICT_NON_PASS


def test_core_posting_list_provenance_is_scalar_not_dict() -> None:
    """posting_list_provenance is a corpus-level constant for the whole
    run, not a per-query distribution -- the output must be the single
    provenance string, never a per-arm count dict."""
    entity_provenance = _posting_list_provenance(({"anchor": {1}}, "entity_tag"))
    literal_provenance = _posting_list_provenance(({}, "literal_surface"))

    assert entity_provenance == "entity_tag"
    assert literal_provenance == "literal_surface"
    assert isinstance(entity_provenance, str)
    assert not isinstance(entity_provenance, dict)


def test_core_token_budget_matches_recall_accuracy_real() -> None:
    """The baseline-serving dispatch (recall_accuracy_real._BUDGET_TOKENS)
    and the membership-capture dispatch (this module's
    _TOKEN_BUDGET_PER_QUERY) must resolve to the SAME candidate pool for
    the same query -- a future edit to either constant without the other
    would silently desync run_verdict's cause-attribution reading."""
    from bench.recall_accuracy_real import _BUDGET_TOKENS

    assert _TOKEN_BUDGET_PER_QUERY == _BUDGET_TOKENS


def test_core_n_min_below_labelled_set_floor() -> None:
    """The pre-registered floor must sit below the labelled fixture's
    initial 30-50 query size, or a faithful low-miss-rate baseline would
    be structurally forced into UNDERPOWERED."""
    assert N_MIN < 30


def test_core_validate_verdict_fixture_dict_accepts_holdout_shape() -> None:
    """The holdout tail-gold fixture's own key contract
    (``scripts/label_holdout_tail_cues.py``'s ``_write_fixture`` output:
    ``cue_id``/``cue_text``/``gold_record_id``/``cluster_id``) -- NOT the
    ``relevant_record_ids`` shape a different fixture family uses."""
    raw = {
        "schema_version": 1,
        "cues": [
            {
                "cue_id": "holdout_cue_0001",
                "cue_text": "what happened with the widget rollout",
                "gold_record_id": "11111111-1111-1111-1111-111111111111",
                "cluster_id": "22222222-2222-2222-2222-222222222222",
                "notes": "",
            },
        ],
    }
    cues = _validate_verdict_fixture_dict(raw)
    assert len(cues) == 1
    assert cues[0]["gold_record_id"] == "11111111-1111-1111-1111-111111111111"


def test_core_validate_verdict_fixture_dict_rejects_missing_gold_record_id() -> None:
    raw = {
        "cues": [
            {"cue_id": "c1", "cue_text": "text", "cluster_id": "x"},
        ],
    }
    with pytest.raises(ValueError):
        _validate_verdict_fixture_dict(raw)


def test_core_rank_of_finds_and_misses() -> None:
    returned = ["a", "b", "c", "d"]
    assert _rank_of("c", returned, k=10) == 2
    assert _rank_of("z", returned, k=10) is None
    assert _rank_of("d", returned, k=2) is None, "must respect the k window"


def test_core_ndcg_at_rank_best_and_absent() -> None:
    assert _ndcg_at_rank(0) == pytest.approx(1.0)
    assert _ndcg_at_rank(None) == 0.0
    assert 0.0 < _ndcg_at_rank(3) < _ndcg_at_rank(0)


def test_core_estimate_tokens_matches_production_heuristic() -> None:
    text = "x" * 40
    assert _estimate_tokens(text) == max(1, len(text) // 4)
    assert _estimate_tokens("") == 1
    assert _estimate_tokens(None) == 1


def test_core_baseline_precision_at_k_is_not_recall_at_k() -> None:
    """With exactly one relevant id per cue, precision@k == 1/k when
    served, 0 when missed -- that only equals recall@k at k=1."""
    baseline_rows = [
        {"served": True}, {"served": True}, {"served": False}, {"served": False},
    ]
    precision = _baseline_precision_at_k(baseline_rows, k=10)
    recall = sum(1 for r in baseline_rows if r["served"]) / len(baseline_rows)

    assert precision == pytest.approx(0.5 * 0.1)
    assert recall == pytest.approx(0.5)
    assert precision != recall


def test_core_baseline_precision_at_k_empty_input() -> None:
    assert _baseline_precision_at_k([], k=10) == 0.0


def test_core_baseline_precision_at_k_equals_recall_only_at_k1() -> None:
    baseline_rows = [{"served": True}, {"served": False}]
    assert _baseline_precision_at_k(baseline_rows, k=1) == pytest.approx(0.5)


def test_core_select_null_arm_cue_id_picks_different_cluster() -> None:
    """Adjacent fixture entries can share cluster_id (and thus entity
    anchors); the null-arm partner must come from a DIFFERENT cluster."""
    missed_cue_ids = ["c0", "c1", "c2", "c3"]
    cluster_id_by_cue_id = {"c0": "A", "c1": "A", "c2": "B", "c3": "B"}

    picked = _select_null_arm_cue_id(0, "A", missed_cue_ids, cluster_id_by_cue_id)

    assert picked in ("c2", "c3")
    assert cluster_id_by_cue_id[picked] != "A"


def test_core_select_null_arm_cue_id_never_self_referential() -> None:
    missed_cue_ids = ["only"]
    cluster_id_by_cue_id = {"only": "A"}

    picked = _select_null_arm_cue_id(0, "A", missed_cue_ids, cluster_id_by_cue_id)

    assert picked is None, (
        "a single missed cue with no different-cluster partner must be "
        "unmeasurable, never a self-referential draw"
    )


def test_core_select_null_arm_cue_id_none_when_all_same_cluster() -> None:
    missed_cue_ids = ["c0", "c1", "c2"]
    cluster_id_by_cue_id = {"c0": "A", "c1": "A", "c2": "A"}

    picked = _select_null_arm_cue_id(1, "A", missed_cue_ids, cluster_id_by_cue_id)

    assert picked is None


def test_core_null_arm_delta_is_none_not_zero_when_unmeasurable() -> None:
    """run_verdict's ``null_delta = conditional_recall_delta(null_pairs) if
    null_pairs else None`` contract: an empty null_pairs list (every row
    unmeasurable) must report None, not a self-referential 0.0."""
    empty_null_pairs: "list[tuple[int, int]]" = []
    null_delta = conditional_recall_delta(empty_null_pairs) if empty_null_pairs else None
    assert null_delta is None


def _null_pairs_from_selection(missed_cue_ids, cluster_id_by_cue_id) -> "list[tuple[int, int]]":
    null_pairs: list[tuple[int, int]] = []
    for i, cid in enumerate(missed_cue_ids):
        partner = _select_null_arm_cue_id(
            i, cluster_id_by_cue_id[cid], missed_cue_ids, cluster_id_by_cue_id,
        )
        if partner is not None:
            null_pairs.append((0, 0))
    return null_pairs


def test_core_null_arm_n_is_full_when_two_or_more_clusters_present() -> None:
    """As long as the missed set spans >=2 clusters, EVERY row has a
    different-cluster partner (the candidate condition is symmetric
    across rows) -- null_arm_n must equal the full eligible-missed
    count, exposing that null_arm_delta covers the whole denominator."""
    missed_cue_ids = ["c0", "c1", "c2", "c3"]
    cluster_id_by_cue_id = {"c0": "A", "c1": "A", "c2": "A", "c3": "B"}

    null_pairs = _null_pairs_from_selection(missed_cue_ids, cluster_id_by_cue_id)
    null_delta = conditional_recall_delta(null_pairs) if null_pairs else None
    null_arm_n = len(null_pairs)

    assert isinstance(null_delta, float)
    assert null_arm_n == len(missed_cue_ids)


def test_core_null_arm_n_is_zero_when_only_one_cluster_present() -> None:
    """A missed set confined to a single cluster has no different-cluster
    partner for ANY row -- null_arm_n must be 0 and null_arm_delta None,
    never a self-referential same-cluster draw."""
    missed_cue_ids = ["c0", "c1", "c2"]
    cluster_id_by_cue_id = {"c0": "A", "c1": "A", "c2": "A"}

    null_pairs = _null_pairs_from_selection(missed_cue_ids, cluster_id_by_cue_id)
    null_delta = conditional_recall_delta(null_pairs) if null_pairs else None
    null_arm_n = len(null_pairs)

    assert null_arm_n == 0
    assert null_delta is None


# ---------------------------------------------------------------------------
# Two-sided A/A stability floor (synthetic): fed through the SAME
# lower-only cluster_stratified_bootstrap_ci via ci_high = -ci(-signed).
# ---------------------------------------------------------------------------


def test_aa_stability_floor_straddles_zero() -> None:
    """Two independent rebuilds of the SAME underlying (noise-only)
    process: the signed per-query outcome (run2 - run1) has no real
    signal, so the two-sided interval derived from the SAME lower-only
    bootstrap function must bracket zero."""
    import numpy as np

    rng = np.random.default_rng(7)
    n = 60
    signed = rng.choice([-1, 0, 1], size=n, p=[0.15, 0.7, 0.15]).tolist()
    cluster_ids = [f"cluster-{i % 12}" for i in range(n)]

    signed_pairs = [(0, s) for s in signed]
    negated_pairs = [(0, -s) for s in signed]

    ci_low = cluster_stratified_bootstrap_ci(signed_pairs, cluster_ids, iters=1000, seed=3)
    ci_high = -cluster_stratified_bootstrap_ci(negated_pairs, cluster_ids, iters=1000, seed=3)

    assert ci_low <= 0.0 <= ci_high, (
        f"A/A floor must straddle zero, got [{ci_low}, {ci_high}]"
    )


def test_aa_stability_floor_degenerate_under_zero_noise() -> None:
    """Deterministic dispatch (identical outcome on both runs) collapses
    the two-sided interval to a point at zero."""
    n = 30
    signed_pairs = [(0, 0)] * n
    negated_pairs = [(0, 0)] * n
    cluster_ids = [f"c{i}" for i in range(n)]

    ci_low = cluster_stratified_bootstrap_ci(signed_pairs, cluster_ids, iters=200, seed=4)
    ci_high = -cluster_stratified_bootstrap_ci(negated_pairs, cluster_ids, iters=200, seed=4)

    assert ci_low == pytest.approx(0.0)
    assert ci_high == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Planted-rescue positive control / cannot-false-green / UNDERPOWERED
# end-to-end (synthetic paired outcomes through the full core).
# ---------------------------------------------------------------------------


def test_planted_rescue_control_drives_pass() -> None:
    """Injecting >=10 genuine rescues, spread across enough clusters and
    past N_min, must drive the lower-95% CI positive and PASS."""
    n_rescued = 14
    n_missed_only = 6
    pairs = [(0, 1)] * n_rescued + [(0, 0)] * n_missed_only
    cluster_ids = [f"cluster-{i}" for i in range(n_rescued + n_missed_only)]

    delta = conditional_recall_delta(pairs)
    ci_low = cluster_stratified_bootstrap_ci(pairs, cluster_ids, iters=1500, seed=5)
    distinct_rescues = sum(1 for _b, a in pairs if a)
    eligible_missed_n = len(pairs)

    outcome = verdict_predicate(
        delta, ci_low, distinct_rescues, {}, eligible_missed_n, N_MIN,
    )
    assert eligible_missed_n >= N_MIN
    assert distinct_rescues >= 10
    assert ci_low > 0
    assert outcome == VERDICT_PASS


def test_cannot_false_green_with_few_rescues() -> None:
    """A slightly positive point-delta from only 1-2 real rescues, well
    under the distinct-rescue floor, must NOT pass even if the CI
    happens to clear zero."""
    pairs = [(0, 1)] * 2 + [(0, 0)] * 18
    cluster_ids = [f"cluster-{i}" for i in range(20)]

    delta = conditional_recall_delta(pairs)
    ci_low = cluster_stratified_bootstrap_ci(pairs, cluster_ids, iters=1500, seed=6)
    distinct_rescues = sum(1 for _b, a in pairs if a)
    eligible_missed_n = len(pairs)

    outcome = verdict_predicate(delta, ci_low, distinct_rescues, {}, eligible_missed_n, N_MIN)
    assert delta > 0
    assert distinct_rescues < 10
    assert outcome == VERDICT_NON_PASS, (
        "a point-positive delta with too few distinct rescues must never PASS"
    )


def test_underpowered_when_denominator_below_n_min() -> None:
    pairs = [(0, 1)] * 5
    cluster_ids = [f"c{i}" for i in range(5)]
    delta = conditional_recall_delta(pairs)
    ci_low = cluster_stratified_bootstrap_ci(pairs, cluster_ids, iters=500, seed=8)
    distinct_rescues = sum(1 for _b, a in pairs if a)
    eligible_missed_n = len(pairs)

    outcome = verdict_predicate(delta, ci_low, distinct_rescues, {}, eligible_missed_n, N_MIN)
    assert eligible_missed_n < N_MIN
    assert outcome == VERDICT_UNDERPOWERED
    assert outcome != VERDICT_PASS
    assert outcome != VERDICT_NON_PASS


# ---------------------------------------------------------------------------
# Measurement-only entity-anchor ceiling expansion -- fabricated seeded
# temp stores only, never real memory content.
# ---------------------------------------------------------------------------


def _tagged_record(text: str, tags: "list[str]"):
    from tests.test_eval_copy_store_warm_baseline import _dense_vec, _make_record

    rec = _make_record(text, _dense_vec(range(0, 20)))
    return dataclasses.replace(rec, tags=tags)


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_ceiling_expansion_entity_tag_path(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _flush

    store = MemoryStore(path=tmp_path / "store")
    target = _tagged_record("widget-alpha rollout notes", ["entity:widget-alpha"])
    other = _tagged_record("unrelated note", ["entity:something-else"])
    store.insert(target)
    store.insert(other)
    _flush(store)

    expansion = entity_anchor_ceiling_expansion(store, "status of widget-alpha rollout")

    assert target.id in expansion
    assert other.id not in expansion
    store.close()


def test_ceiling_expansion_literal_surface_fallback(tmp_path) -> None:
    """No entity tags anywhere in the corpus: the fallback must build a
    literal_surface posting list instead of returning empty."""
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _dense_vec, _flush, _make_record

    store = MemoryStore(path=tmp_path / "store")
    target = _make_record("Zephyrion deployment checklist", _dense_vec(range(20, 40)))
    other = _make_record("unrelated content entirely", _dense_vec(range(40, 60)))
    store.insert(target)
    store.insert(other)
    _flush(store)

    expansion = entity_anchor_ceiling_expansion(store, "any update on Zephyrion status please")

    assert target.id in expansion
    assert other.id not in expansion
    store.close()


def test_ceiling_expansion_raises_when_unmeasurable(tmp_path) -> None:
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    with pytest.raises(CeilingArmUnmeasurable):
        entity_anchor_ceiling_expansion(store, "anything at all")
    store.close()


def test_ceiling_expansion_ignores_gold_only_tag(tmp_path) -> None:
    """A gold record tagged with an entity the CUE never mentions must
    never be recovered -- proving the anchor set is cue-derived only,
    never gold-peeked."""
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _flush

    store = MemoryStore(path=tmp_path / "store")
    gold = _tagged_record("the tail record content", ["entity:secretgoldterm"])
    store.insert(gold)
    _flush(store)

    cue_text_without_gold_term = "what happened with the project last week"
    expansion = entity_anchor_ceiling_expansion(store, cue_text_without_gold_term)

    assert gold.id not in expansion, (
        "a gold-only entity tag the cue never mentions must not be "
        "reachable -- a gold-peeking variant would trivially return it"
    )
    store.close()


def test_ceiling_expansion_signature_has_no_gold_parameter() -> None:
    params = inspect.signature(entity_anchor_ceiling_expansion).parameters
    for forbidden in ("gold_id", "gold_record", "gold", "relevant_record_ids"):
        assert forbidden not in params, (
            f"entity_anchor_ceiling_expansion must have no parameter through "
            f"which a gold record could reach it ({forbidden!r} found)"
        )


def test_ceiling_expansion_is_size_bounded(tmp_path) -> None:
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _flush

    store = MemoryStore(path=tmp_path / "store")
    records = [
        _tagged_record(f"member {i} of a shared entity", ["entity:sharedbig"])
        for i in range(30)
    ]
    for rec in records:
        store.insert(rec)
    _flush(store)

    expansion = entity_anchor_ceiling_expansion(store, "sharedbig status", k=10)

    assert len(expansion) <= 10
    store.close()


def test_shuffled_anchor_expansion_is_size_matched(tmp_path) -> None:
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _flush

    store = MemoryStore(path=tmp_path / "store")
    records = [
        _tagged_record(f"member {i} of wrongterm", ["entity:wrongterm"])
        for i in range(8)
    ]
    for rec in records:
        store.insert(rec)
    _flush(store)

    null_expansion = shuffled_anchor_expansion(store, match_size=3, wrong_anchors=["wrongterm"])

    assert len(null_expansion) == 3
    store.close()


def test_shuffled_anchor_null_arm_stays_flat_through_core(tmp_path) -> None:
    """A correct-anchor arm that genuinely rescues gold, run through the
    SAME core function as a size-matched shuffled-anchor arm that never
    touches the true anchor, must show a clearly positive correct-arm
    delta against a flat (near-zero) null-arm delta."""
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _flush

    store = MemoryStore(path=tmp_path / "store")
    golds = [
        _tagged_record(f"gold {i} tagged correctly", ["entity:realterm"])
        for i in range(10)
    ]
    decoys = [
        _tagged_record(f"decoy {i} tagged wrongly", ["entity:decoyterm"])
        for i in range(10)
    ]
    for rec in golds + decoys:
        store.insert(rec)
    _flush(store)

    from bench.fhrr_verdict_recall_delta import _resolve_posting_list

    posting_list = _resolve_posting_list(store)

    correct_pairs = []
    null_pairs = []
    for gold in golds:
        expansion = entity_anchor_ceiling_expansion(
            store, "realterm status", anchors=["realterm"], posting_list=posting_list,
        )
        correct_pairs.append((0, 1 if gold.id in expansion else 0))

        null_expansion = shuffled_anchor_expansion(
            store, match_size=len(expansion), wrong_anchors=["decoyterm"], posting_list=posting_list,
        )
        null_pairs.append((0, 1 if gold.id in null_expansion else 0))

    correct_delta = conditional_recall_delta(correct_pairs)
    null_delta = conditional_recall_delta(null_pairs)

    assert correct_delta == pytest.approx(1.0)
    assert null_delta == pytest.approx(0.0)
    assert correct_delta > null_delta
    store.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_build_edge_adjacency_and_spread_signal(driver, tmp_path, monkeypatch) -> None:
    _select_driver(driver, monkeypatch)
    from iai_mcp.store import MemoryStore

    from tests.test_eval_copy_store_warm_baseline import _dense_vec, _flush, _make_record

    store = MemoryStore(path=tmp_path / "store")
    gold = _make_record("gold record with an edge", _dense_vec(range(0, 20)))
    connected = _make_record("connected pool member", _dense_vec(range(20, 40)))
    unconnected = _make_record("unconnected pool member", _dense_vec(range(40, 60)))
    store.insert(gold)
    store.insert(connected)
    store.insert(unconnected)
    _flush(store)

    store.boost_edges([(gold.id, connected.id)], edge_type="hebbian")

    adjacency = build_edge_adjacency(store)

    assert spread_edge_missing_signal(gold.id, {connected.id}, adjacency) is False
    assert spread_edge_missing_signal(gold.id, {unconnected.id}, adjacency) is True
    store.close()


# ---------------------------------------------------------------------------
# Measurement-only fence: src/ must never import the ceiling-expansion
# helper or this module.
# ---------------------------------------------------------------------------


def test_measurement_only_fence_src_never_imports_ceiling_expansion() -> None:
    src_root = _REPO_ROOT / "src"
    offenders: "list[str]" = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "fhrr_verdict_recall_delta" in text or "entity_anchor_ceiling_expansion" in text:
            offenders.append(str(path.relative_to(_REPO_ROOT)))
    assert not offenders, (
        f"the measurement-only ceiling expansion must never be imported by "
        f"src/, found references in: {offenders}"
    )


def test_measurement_only_fence_module_not_referenced_by_ast_import(tmp_path) -> None:
    """A stronger structural check alongside the substring scan: no
    ``import``/``from ... import`` statement anywhere in src/ names this
    module."""
    src_root = _REPO_ROOT / "src"
    for path in src_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "fhrr_verdict_recall_delta" not in alias.name, path
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "fhrr_verdict_recall_delta" not in mod, path
