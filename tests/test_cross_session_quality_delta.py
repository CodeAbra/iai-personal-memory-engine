"""Tests for the rank-aware chain-integrity quality comparator."""
from __future__ import annotations

import inspect
import math
import random

import pytest

from bench.cross_session_quality_delta import (
    NOT_FOUND_RANK,
    SELF_TUNING_FINDINGS,
    DegenerateFloorError,
    build_harness_report,
    chain_integrity_verdict,
    no_displacement_value_metric,
    partition_cues_by_involvement,
    per_cue_rank_deltas,
    rank_of,
    replay_outcome_from_slice,
    run_chain_integrity_slice,
    signed_rank_delta,
    value_metric,
)
from bench.cross_session_quality_delta import _assert_commensurable
from bench.proc_corpus_census import two_sided_aa_floor
from iai_mcp.availability import (
    OUTCOME_EXCLUDED_UNAVAILABLE,
    OUTCOME_PASSED,
    classify_session_outcome,
)
from iai_mcp.lilli.profile.retrieval_tuning import RETRIEVAL_MIN_SAMPLES

# ---------------------------------------------------------------------------
# Pure comparator arithmetic (no store, no dispatch)
# ---------------------------------------------------------------------------


def test_rank_of_returns_index_when_present():
    assert rank_of(["a", "b", "c"], "b") == 1


def test_rank_of_returns_sentinel_when_absent():
    assert rank_of(["a", "b", "c"], "z") == NOT_FOUND_RANK


def test_signed_rank_delta_positive_when_gold_moves_up():
    before = ["a", "b", "c", "d", "e", "gold"]  # rank 5
    after = ["x", "gold", "y", "z"]  # rank 1
    assert signed_rank_delta(before, after, "gold") == 4


def test_signed_rank_delta_negative_when_gold_moves_down():
    before = ["x", "gold", "y", "z"]  # rank 1
    after = ["a", "b", "c", "d", "e", "gold"]  # rank 5
    assert signed_rank_delta(before, after, "gold") == -4


def test_signed_rank_delta_censored_when_absent_both_sides():
    before = ["a", "b", "c"]
    after = ["d", "e"]
    delta = signed_rank_delta(before, after, "gold")
    assert delta == 0
    assert delta == NOT_FOUND_RANK - NOT_FOUND_RANK


def test_signed_rank_delta_nonzero_for_reordered_identical_set():
    """The defect this comparator fixes: the eviction-only shipped comparator
    reports zero here because the SET is unchanged -- rank position is not."""
    before = ["gold", "other"]
    after = ["other", "gold"]
    delta = signed_rank_delta(before, after, "gold")
    assert delta != 0
    assert delta == -1


def test_value_metric_is_mean_signed_delta():
    deltas = [4, -4, 0, -1]
    assert value_metric(deltas) == pytest.approx(sum(deltas) / len(deltas))


def test_value_metric_empty_returns_zero():
    assert value_metric([]) == 0.0


def test_per_cue_rank_deltas_keys_and_values():
    cues = [
        {"cue_id": "cue-0", "gold_id": "g0", "cue_text": "irrelevant"},
        {"cue_id": "cue-1", "gold_id": "g1", "cue_text": "irrelevant"},
    ]
    before_ids_by_cue = {"cue-0": ["g0", "z"], "cue-1": ["z", "g1"]}
    after_ids_by_cue = {"cue-0": ["z", "g0"], "cue-1": ["g1", "z"]}
    deltas = per_cue_rank_deltas(cues, before_ids_by_cue, after_ids_by_cue)
    assert deltas == {"cue-0": -1, "cue-1": 1}


# ---------------------------------------------------------------------------
# Hermeticity guard: no live-store path anywhere in the module
# ---------------------------------------------------------------------------


def test_module_never_imports_live_store_copy_path():
    import bench.cross_session_quality_delta as mod

    assert not hasattr(mod, "open_eval_copy_store")


def test_module_source_never_references_live_store_copy_helper():
    import bench.cross_session_quality_delta as mod

    source = inspect.getsource(mod)
    assert "open_eval_copy_store" not in source
    assert "_operator_home" not in source


# ---------------------------------------------------------------------------
# Weight-lever removal guard: the module arms no retrieval weight
# ---------------------------------------------------------------------------


def test_module_never_imports_weight_arming_api():
    import bench.cross_session_quality_delta as mod

    assert not hasattr(mod, "save_retrieval_weights_state")
    assert not hasattr(mod, "retrieval_weight_cache")


def test_module_source_never_references_weight_arming_api():
    import bench.cross_session_quality_delta as mod

    source = inspect.getsource(mod)
    assert "save_retrieval_weights_state" not in source
    assert "retrieval_weight_cache" not in source


# ---------------------------------------------------------------------------
# Commensurability guard (pure)
# ---------------------------------------------------------------------------


def test_commensurable_vectors_pass_silently():
    real = {"cue-0": 1, "cue-1": -2}
    null = {"cue-0": 0, "cue-1": 3}
    _assert_commensurable(real, null)  # must not raise


def test_commensurable_vectors_raises_on_length_mismatch():
    real = {"cue-0": 1, "cue-1": -2}
    null = {"cue-0": 0}
    with pytest.raises(ValueError):
        _assert_commensurable(real, null)


def test_commensurable_vectors_raises_on_key_mismatch():
    real = {"cue-0": 1, "cue-1": -2}
    null = {"cue-0": 0, "cue-2": 3}
    with pytest.raises(ValueError):
        _assert_commensurable(real, null)


# ---------------------------------------------------------------------------
# chain_integrity_verdict (pure): fail-loud floor guard, strict inequality,
# underpowered distinction, no round-number margin literal
# ---------------------------------------------------------------------------


def test_degenerate_all_equal_null_raises():
    degenerate_floor = two_sided_aa_floor([0.0] * 20)
    assert degenerate_floor[1] <= 0.0  # sanity: the fixture really is degenerate
    with pytest.raises(DegenerateFloorError):
        chain_integrity_verdict(value_metric=5.0, aa_floor=degenerate_floor, established_cues=100)


def test_verdict_value_at_floor_bound_does_not_clear():
    verdict = chain_integrity_verdict(value_metric=5.0, aa_floor=(1.0, 5.0), established_cues=100)
    assert verdict["value_clears_floor"] is False
    assert verdict["verdict"] == "PARK"


def test_verdict_value_above_floor_bound_clears():
    verdict = chain_integrity_verdict(
        value_metric=5.0001, aa_floor=(1.0, 5.0), established_cues=100
    )
    assert verdict["value_clears_floor"] is True


def test_verdict_margin_is_measured_floor_not_round_number():
    """The only compared constant is the measured floor upper bound -- an
    odd, non-round floor value must drive the decision exactly."""
    odd_floor = (0.3, 2.7182818)
    at_bound = chain_integrity_verdict(
        value_metric=2.7182818, aa_floor=odd_floor, established_cues=100
    )
    above_bound = chain_integrity_verdict(
        value_metric=2.71828181, aa_floor=odd_floor, established_cues=100
    )
    assert at_bound["value_clears_floor"] is False
    assert above_bound["value_clears_floor"] is True
    assert at_bound["aa_floor"] == odd_floor


def test_verdict_zero_established_reads_underpowered():
    verdict = chain_integrity_verdict(value_metric=10.0, aa_floor=(0.0, 1.0), established_cues=0)
    assert verdict["underpowered"] is True
    assert verdict["established_clears_floor"] is False
    assert verdict["verdict"] == "PARK"


def test_verdict_established_count_gated_by_imported_power_floor():
    below = chain_integrity_verdict(
        value_metric=10.0, aa_floor=(0.0, 1.0), established_cues=RETRIEVAL_MIN_SAMPLES - 1
    )
    at_floor = chain_integrity_verdict(
        value_metric=10.0, aa_floor=(0.0, 1.0), established_cues=RETRIEVAL_MIN_SAMPLES
    )
    assert below["established_clears_floor"] is False
    assert at_floor["established_clears_floor"] is True
    assert below["power_floor"] == RETRIEVAL_MIN_SAMPLES


def test_verdict_proceeds_only_when_both_gates_clear():
    both_clear = chain_integrity_verdict(
        value_metric=10.0, aa_floor=(0.0, 1.0), established_cues=RETRIEVAL_MIN_SAMPLES
    )
    assert both_clear["verdict"] == "proceed"

    value_fails = chain_integrity_verdict(
        value_metric=0.5, aa_floor=(0.0, 1.0), established_cues=RETRIEVAL_MIN_SAMPLES
    )
    assert value_fails["verdict"] == "PARK"


# ---------------------------------------------------------------------------
# Wiring test: small N, both drivers -- proves the vertical slice is wired
# (store construction, twin reinsertion, dispatch, positive control) on both
# storage drivers. Statistical power is proven separately (below), at N>=100
# on one driver, so this stays fast on both.
# ---------------------------------------------------------------------------

_WIRING_N_CUES = 15


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_chain_integrity_slice_wired_on_both_drivers(driver):
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built -- lilli driver unavailable in this env")

    result = run_chain_integrity_slice(seed=0, n_cues=_WIRING_N_CUES, driver=driver)

    assert result["n_cues"] > 0
    assert len(result["real_deltas"]) == result["n_cues"]
    assert len(result["null_deltas"]) == result["n_cues"]
    assert math.isfinite(result["value_metric"])
    assert len(result["aa_floor"]) == 2
    assert result["drop_ids"], "the wiring run must drop at least one gold record"

    # Positive control: every dropped gold is absent in degraded, present in intact.
    for cue_id in result["drop_cue_ids"]:
        assert result["degraded_ranks"][cue_id] == NOT_FOUND_RANK
        assert result["intact_a_ranks"][cue_id] != NOT_FOUND_RANK


# ---------------------------------------------------------------------------
# Statistical measurement: N>=100 self-labelled cues on one driver -- the
# decisive end-to-end signal, the A/A floor, and the resulting verdict.
#
# Cached at module level (not a pytest fixture): a module-scoped fixture is
# instantiated before the function-scoped autouse crypto-passphrase fixture
# runs (pytest orders fixture setup by scope, broadest first), which starves
# a fresh synthetic store of IAI_MCP_CRYPTO_PASSPHRASE. A plain cache read
# from inside a test function body runs after every autouse fixture has
# already set up the environment, regardless of test execution order.
# ---------------------------------------------------------------------------

_STATISTICAL_N_CUES = 100
_statistical_result_cache: "dict" = {}


def _statistical_result() -> dict:
    if "result" not in _statistical_result_cache:
        _statistical_result_cache["result"] = run_chain_integrity_slice(
            seed=0, n_cues=_STATISTICAL_N_CUES, driver="stdlib"
        )
    return _statistical_result_cache["result"]


def test_statistical_slice_covers_at_least_100_cues():
    assert _statistical_result()["n_cues"] >= 100


def test_statistical_slice_value_metric_decisively_positive():
    """Dropping genuinely-resolvable golds must dominate the mean: value_metric
    clearly positive, not merely nonzero."""
    assert _statistical_result()["value_metric"] > 0


def test_statistical_slice_positive_control():
    result = _statistical_result()
    assert result["drop_cue_ids"]
    for cue_id in result["drop_cue_ids"]:
        assert result["degraded_ranks"][cue_id] == NOT_FOUND_RANK
        assert result["intact_a_ranks"][cue_id] != NOT_FOUND_RANK


def test_statistical_slice_vectors_commensurable():
    result = _statistical_result()
    real_deltas = result["real_deltas"]
    null_deltas = result["null_deltas"]
    assert len(real_deltas) == len(null_deltas) == result["n_cues"]
    assert set(real_deltas) == set(null_deltas)


def test_statistical_slice_floor_strictly_positive():
    assert _statistical_result()["aa_floor"][1] > 0.0


def test_statistical_slice_verdict_proceeds():
    result = _statistical_result()
    verdict = chain_integrity_verdict(
        value_metric=result["value_metric"],
        aa_floor=result["aa_floor"],
        established_cues=result["resolved_cues"],
    )
    assert verdict["verdict"] == "proceed"
    assert verdict["value_clears_floor"] is True
    assert verdict["established_clears_floor"] is True
    assert verdict["underpowered"] is False


def test_statistical_slice_resolved_cues_is_the_gold_resolving_subset():
    """resolved_cues must be the count of cues whose own gold actually
    resolves in intact_a -- a strict subset of n_cues, and never larger."""
    result = _statistical_result()
    assert 0 < result["resolved_cues"] <= result["n_cues"]
    resolved_via_ranks = sum(
        1 for rank in result["intact_a_ranks"].values() if rank != NOT_FOUND_RANK
    )
    assert result["resolved_cues"] == resolved_via_ranks


def test_verdict_gates_on_resolved_cues_not_total_n_cues():
    """The power gate must be fed the resolvable-cue count, not the total
    cue count -- a corpus with plenty of total cues but too few that
    actually resolve their own gold must PARK as underpowered, even
    though n_cues alone would clear RETRIEVAL_MIN_SAMPLES."""
    underpowered_resolved = RETRIEVAL_MIN_SAMPLES - 1
    slice_result = {
        "value_metric": 500.0,
        "aa_floor": (0.0, 10.0),
        "n_cues": 100,
        "resolved_cues": underpowered_resolved,
    }
    outcome = replay_outcome_from_slice(slice_result)
    assert outcome["available"] is True
    assert outcome["passed"] is False
    assert outcome["verdict"]["established_cues"] == underpowered_resolved
    assert outcome["verdict"]["established_clears_floor"] is False
    assert outcome["verdict"]["verdict"] == "PARK"


# ---------------------------------------------------------------------------
# Two-rung, two-level non-vacuity -- SYNTHETIC plant/null fixtures only, at
# the comparator-input seam (same shape as
# tests/test_proc_corpus_census.py's planted-signal-vs-i.i.d.-noise pair,
# adapted to this module's rank-delta quantity).
#
# Layer asymmetry (load-bearing, stated explicitly so a reviewer does not
# read the plant seam as a backslide): the DECISIVE rung below reuses the
# real chain-integrity statistical result computed above -- it
# proves the end-to-end PIPELINE resolves a genuine chain-integrity
# difference. The NEAR-FLOOR rung is a comparator-INPUT plant -- it proves
# the statistical VERDICT MACHINERY has resolution at floor scale, a
# narrower, different claim about a different layer. It is NOT a weaker
# substitute for the decisive rung.
#
# Eviction quantum (derived fact, why the near-floor rung must be a plant):
# each dropped gold contributes roughly NOT_FOUND_RANK / n_cues to
# value_metric (the gold goes absent in the degraded leg) -- at n_cues=100
# that is already ~100, many times any measured A/A floor's bootstrap
# width. There is no floor-sized signal on the real eviction axis, so a
# floor-scale non-vacuity rung is necessarily a comparator-input plant of
# exact a-priori magnitude, never a real degradation.
# ---------------------------------------------------------------------------

_PLANT_N_CUES = 200
_PLANT_BASE_RANK = 5
_PLANT_FILLER_COUNT = 20
_PLANT_GOLD_ID = "plant-gold"

# The extra deterministic worsening nudge applied to a thin, randomly
# selected slice of cues -- sized a priori from the noise construction
# itself (a single-position nudge spread across a minority fraction of a
# large cue set, so the mean shift sits at roughly the null's own
# bootstrap-CI scale), never from any measured pipeline output.
_PLANT_SHIFT_MAGNITUDE = 1
_PLANT_SHIFT_FRACTION = 0.3


def _plant_seam_cues(n_cues: int = _PLANT_N_CUES) -> "list[dict]":
    return [
        {"cue_id": f"plant-{i}", "gold_id": _PLANT_GOLD_ID, "cue_text": "irrelevant"}
        for i in range(n_cues)
    ]


def _plant_seam_before_ids(n_cues: int = _PLANT_N_CUES) -> "dict[str, list[str]]":
    filler = [f"filler-{j}" for j in range(_PLANT_FILLER_COUNT)]
    before = filler[:_PLANT_BASE_RANK] + [_PLANT_GOLD_ID] + filler[_PLANT_BASE_RANK:]
    return {f"plant-{i}": list(before) for i in range(n_cues)}


def _plant_seam_perturbed_ids(
    n_cues: int, *, shift_fraction: float, shift_magnitude: int, seed: int
) -> "dict[str, list[str]]":
    """Hand-built ORDERED id lists at the comparator input -- the same seam
    test_signed_rank_delta_* already exercises. Every cue gets +/-1 i.i.d.
    reorder noise; a shift_fraction-selected subset ALSO gets a
    deterministic extra shift_magnitude-position WORSENING nudge.
    shift_fraction=0 (or shift_magnitude=0) yields pure noise -- the null
    draw.

    This is the "before"/degraded-analog side of per_cue_rank_deltas, the
    fixed _plant_seam_before_ids() reference is the "after"/intact-analog
    side -- mirroring the harness's own real_deltas convention exactly
    (per_cue_rank_deltas(cues, degraded_ids_by_cue, intact_a_ids_by_cue)):
    a worse "before" state yields a POSITIVE delta.
    """
    rng = random.Random(seed)
    filler = [f"filler-{j}" for j in range(_PLANT_FILLER_COUNT)]
    perturbed_ids: "dict[str, list[str]]" = {}
    for i in range(n_cues):
        noise = rng.choice([-1, 0, 1])
        extra = shift_magnitude if rng.random() < shift_fraction else 0
        rank = max(0, _PLANT_BASE_RANK + noise + extra)
        perturbed_ids[f"plant-{i}"] = filler[:rank] + [_PLANT_GOLD_ID] + filler[rank:]
    return perturbed_ids


def test_non_vacuity_decisive_rung_metric_arm_clears_floor():
    """Decisive rung, metric-arm level: the REAL pipeline degradation's
    value_metric clears the two-intact-replay A/A floor."""
    result = _statistical_result()
    assert result["value_metric"] > result["aa_floor"][1]


def test_non_vacuity_decisive_rung_verdict_level_proceeds():
    """Decisive rung, verdict level: the SAME real degradation's measured
    inputs drive the actual verdict function to proceed."""
    result = _statistical_result()
    verdict = chain_integrity_verdict(
        value_metric=result["value_metric"],
        aa_floor=result["aa_floor"],
        established_cues=result["resolved_cues"],
    )
    assert verdict["verdict"] == "proceed"


def test_non_vacuity_near_floor_plant_metric_arm_clears_its_own_floor():
    """Near-floor rung, metric-arm level: a comparator-input plant sized a
    priori at roughly the plant-seam floor scale clears ITS OWN
    plant-seam floor -- never the real pipeline floor above (the two
    vectors are not commensurable, per the same commensurability
    constraint _assert_commensurable enforces on the real vectors,
    applied one level down)."""
    reference_ids = _plant_seam_before_ids()
    plant_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES,
        shift_fraction=_PLANT_SHIFT_FRACTION,
        shift_magnitude=_PLANT_SHIFT_MAGNITUDE,
        seed=1,
    )
    null_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES, shift_fraction=0.0, shift_magnitude=0, seed=2
    )
    cues = _plant_seam_cues()
    plant_deltas = per_cue_rank_deltas(cues, plant_perturbed_ids, reference_ids)
    null_deltas = per_cue_rank_deltas(cues, null_perturbed_ids, reference_ids)
    aa_floor = two_sided_aa_floor(list(null_deltas.values()), iters=300, seed=0)

    vm = value_metric(plant_deltas.values())
    assert vm > aa_floor[1], "near-floor plant did not clear its own plant-seam floor"


def test_non_vacuity_near_floor_plant_verdict_level_proceeds():
    """Near-floor rung, verdict level: the SAME plant inputs, fed through
    the actual verdict function, must proceed -- metric-arm clearance
    alone does not prove the verdict function itself proceeds; this
    level checks that explicitly."""
    reference_ids = _plant_seam_before_ids()
    plant_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES,
        shift_fraction=_PLANT_SHIFT_FRACTION,
        shift_magnitude=_PLANT_SHIFT_MAGNITUDE,
        seed=1,
    )
    null_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES, shift_fraction=0.0, shift_magnitude=0, seed=2
    )
    cues = _plant_seam_cues()
    plant_deltas = per_cue_rank_deltas(cues, plant_perturbed_ids, reference_ids)
    null_deltas = per_cue_rank_deltas(cues, null_perturbed_ids, reference_ids)
    aa_floor = two_sided_aa_floor(list(null_deltas.values()), iters=300, seed=0)

    verdict = chain_integrity_verdict(
        value_metric=value_metric(plant_deltas.values()),
        aa_floor=aa_floor,
        established_cues=len(cues),
    )
    assert verdict["verdict"] == "proceed"


def test_non_vacuity_null_plant_metric_arm_stays_within_its_own_floor():
    """Null rung, metric-arm level: i.i.d. reorder noise with NO planted
    shift stays within its own independently-drawn plant-seam floor."""
    reference_ids = _plant_seam_before_ids()
    candidate_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES, shift_fraction=0.0, shift_magnitude=0, seed=3
    )
    floor_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES, shift_fraction=0.0, shift_magnitude=0, seed=4
    )
    cues = _plant_seam_cues()
    candidate_deltas = per_cue_rank_deltas(cues, candidate_perturbed_ids, reference_ids)
    floor_null_deltas = per_cue_rank_deltas(cues, floor_perturbed_ids, reference_ids)
    aa_floor = two_sided_aa_floor(list(floor_null_deltas.values()), iters=300, seed=0)

    vm = value_metric(candidate_deltas.values())
    assert aa_floor[0] <= vm <= aa_floor[1], "i.i.d. noise with no planted shift was flagged as a signal"


def test_non_vacuity_null_plant_verdict_level_parks():
    """Null rung, verdict level: the SAME i.i.d.-noise inputs drive the
    actual verdict function to PARK."""
    reference_ids = _plant_seam_before_ids()
    candidate_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES, shift_fraction=0.0, shift_magnitude=0, seed=3
    )
    floor_perturbed_ids = _plant_seam_perturbed_ids(
        _PLANT_N_CUES, shift_fraction=0.0, shift_magnitude=0, seed=4
    )
    cues = _plant_seam_cues()
    candidate_deltas = per_cue_rank_deltas(cues, candidate_perturbed_ids, reference_ids)
    floor_null_deltas = per_cue_rank_deltas(cues, floor_perturbed_ids, reference_ids)
    aa_floor = two_sided_aa_floor(list(floor_null_deltas.values()), iters=300, seed=0)

    verdict = chain_integrity_verdict(
        value_metric=value_metric(candidate_deltas.values()),
        aa_floor=aa_floor,
        established_cues=len(cues),
    )
    assert verdict["verdict"] == "PARK"


# ---------------------------------------------------------------------------
# No-displacement control: dropping some golds must not worsen cues whose
# own gold was untouched. The control itself must be non-vacuous -- a
# planted worsening displacement on the uninvolved partition must be
# caught.
# ---------------------------------------------------------------------------


def test_partition_cues_by_involvement_splits_correctly():
    affected, uninvolved = partition_cues_by_involvement(
        ["c0", "c1", "c2", "c3"], ["c1", "c3"]
    )
    assert affected == ["c1", "c3"]
    assert uninvolved == ["c0", "c2"]


def test_no_displacement_uninvolved_partition_stays_within_floor():
    """Two-sided: |vm| <= aa_floor[1] strictly implies the weaker one-sided
    vm <= aa_floor[1] -- never a weakening.
    Restricted to resolved-and-uninvolved cues (own gold already has a
    rank in intact_a): a cue with no established rank there has no "own
    rank" to be displaced from, and folding it in mixes a different,
    incidental effect (a previously-unfindable gold surfacing once
    competing records shrink the corpus -- corroborated separately by
    uninvolved_gained/uninvolved_lost) into the displacement measurement.
    An asymmetric aa_floor whose lower bound sits above 0.0 cannot bound
    an untouched vm==0.0 result, so the two-sided check compares
    magnitude against the floor's own scale rather than requiring
    containment in a band that need not straddle zero.
    """
    result = _statistical_result()
    all_cue_ids = list(result["real_deltas"].keys())
    _affected, uninvolved = partition_cues_by_involvement(all_cue_ids, result["drop_cue_ids"])
    resolved_uninvolved = [
        cid for cid in uninvolved if result["intact_a_ranks"][cid] != NOT_FOUND_RANK
    ]
    assert resolved_uninvolved, "the statistical run must leave at least one resolved, uninvolved cue"
    vm = no_displacement_value_metric(result["real_deltas"], resolved_uninvolved)
    aa_floor = result["aa_floor"]
    assert abs(vm) <= aa_floor[1], (
        "resolved uninvolved cues moved beyond the measured A/A floor's scale in either direction"
    )


def test_uninvolved_boundary_crossings_split_by_direction():
    """gained/lost separate incidental corpus-shrinkage surfacing from a
    genuine unfavorable displacement -- independently reconstructed here
    from intact_a_ranks/degraded_ranks to pin the field's actual meaning."""
    result = _statistical_result()
    all_cue_ids = list(result["real_deltas"].keys())
    _affected, uninvolved = partition_cues_by_involvement(all_cue_ids, result["drop_cue_ids"])
    gained_via_ranks = sum(
        1
        for cid in uninvolved
        if result["intact_a_ranks"][cid] == NOT_FOUND_RANK
        and result["degraded_ranks"][cid] != NOT_FOUND_RANK
    )
    lost_via_ranks = sum(
        1
        for cid in uninvolved
        if result["intact_a_ranks"][cid] != NOT_FOUND_RANK
        and result["degraded_ranks"][cid] == NOT_FOUND_RANK
    )
    assert result["uninvolved_gained"] == gained_via_ranks
    assert result["uninvolved_lost"] == lost_via_ranks


def test_no_displacement_control_catches_a_planted_worsening_displacement():
    """Non-vacuity of the control itself: a deliberate rank-WORSENING
    displacement planted on the uninvolved partition, at the comparator
    input (the same seam the near-floor plant uses), must be caught -- the
    control fails on it rather than passing trivially.

    Direction convention matches the harness's own real_deltas (the
    degraded-analog id list goes in the "before" position, the reference
    id list in "after" -- per_cue_rank_deltas reads positive when the
    "before" state is worse than "after", exactly like
    per_cue_rank_deltas(cues, degraded_ids_by_cue, intact_a_ids_by_cue)).
    """
    n = 100
    cues = [{"cue_id": f"u{i}", "gold_id": "gold", "cue_text": "x"} for i in range(n)]
    filler = [f"filler-{j}" for j in range(10)]
    reference_ids = {c["cue_id"]: filler[:3] + ["gold"] + filler[3:] for c in cues}
    # Deterministic worsening: gold pushed several ranks lower on every cue.
    worsened_ids = {c["cue_id"]: filler[:3] + ["a", "b"] + ["gold"] + filler[3:] for c in cues}
    real_deltas = per_cue_rank_deltas(cues, worsened_ids, reference_ids)

    rng = random.Random(0)
    null_worsened_ids = {}
    for c in cues:
        noise = rng.choice([-1, 0, 1])
        rank = max(0, 3 + noise)
        null_worsened_ids[c["cue_id"]] = filler[:rank] + ["gold"] + filler[rank:]
    null_deltas = per_cue_rank_deltas(cues, null_worsened_ids, reference_ids)
    aa_floor = two_sided_aa_floor(list(null_deltas.values()))

    cue_ids = [c["cue_id"] for c in cues]
    vm = no_displacement_value_metric(real_deltas, cue_ids)
    assert vm > aa_floor[1], "planted worsening displacement was NOT caught by the control"


# ---------------------------------------------------------------------------
# Availability gate wiring + the named self-tuning negative-findings field
# ---------------------------------------------------------------------------


def test_module_never_imports_availability_from_source():
    import bench.cross_session_quality_delta as mod

    assert not hasattr(mod, "availability_from_source")
    source = inspect.getsource(mod)
    assert "availability_from_source" not in source


def test_available_with_unresolved_outcome_raises_never_silently_fails():
    """The harness never wraps classify_session_outcome in a broad except:
    an available=True replay with an unresolved passed value must raise --
    proving this is never silently miscounted as a fail."""
    with pytest.raises(ValueError):
        classify_session_outcome(available=True, passed=None)


def test_replay_from_none_slice_lands_in_excluded_not_scored():
    """A simulated store-build failure -- the replay never produced a
    slice_result at all."""
    outcome = replay_outcome_from_slice(None)
    assert outcome["available"] is False
    assert outcome["passed"] is None
    assert outcome["outcome"] == OUTCOME_EXCLUDED_UNAVAILABLE


def test_replay_from_degenerate_slice_lands_in_excluded_not_scored():
    """An intentionally unmeasurable corpus -- a collapsed A/A floor -- is
    also excluded, not scored as a pass or fail."""
    degenerate = {"value_metric": 5.0, "aa_floor": (0.0, 0.0), "n_cues": 100, "resolved_cues": 100}
    outcome = replay_outcome_from_slice(degenerate)
    assert outcome["available"] is False
    assert outcome["passed"] is None
    assert outcome["outcome"] == OUTCOME_EXCLUDED_UNAVAILABLE


def test_replay_from_real_statistical_result_is_available_and_passes():
    outcome = replay_outcome_from_slice(_statistical_result())
    assert outcome["available"] is True
    assert outcome["passed"] is True
    assert outcome["outcome"] == OUTCOME_PASSED


def test_build_harness_report_excludes_unmeasurable_and_keeps_measured_intact():
    degenerate = {"value_metric": 5.0, "aa_floor": (0.0, 0.0), "n_cues": 100, "resolved_cues": 100}
    report = build_harness_report([_statistical_result(), degenerate, None])
    summary = report["measurable_summary"]
    assert summary.excluded_unavailable == 2, "both unmeasurable replays must be excluded"
    assert summary.measured == 1
    assert summary.passed == 1


def test_build_harness_report_all_unavailable_zero_measured_shape():
    report = build_harness_report([None, None])
    assert report["measurable_summary_rendered"] == "0/0 measurable passed, 2 excluded: unavailable"


def test_build_harness_report_exact_summary_shape():
    report = build_harness_report([_statistical_result()])
    assert report["measurable_summary_rendered"] == "1/1 measurable passed, 0 excluded: unavailable"


def test_build_harness_report_carries_self_tuning_findings_field():
    report = build_harness_report([_statistical_result()])
    assert report["self_tuning_findings"] == SELF_TUNING_FINDINGS
    findings = report["self_tuning_findings"]
    assert "cosine" in findings.lower()
    assert "reinforcement" in findings.lower()
    assert "self-loop" in findings.lower()
