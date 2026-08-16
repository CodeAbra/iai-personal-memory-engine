"""The monotropism_depth TuningSpec: pure observe/apply pair over
pre-rank community-gate concentration.
"""

from __future__ import annotations

from iai_mcp.lilli.cycle.sleep_pipeline._knob_tune_specs import (
    DEPTH_DECAY,
    DEPTH_EPSILON,
    K_MIN,
    MAX_AUTO_DEPTH,
    MAX_DEPTH_DELTA,
    MAX_UNOBSERVED_WINDOWS,
    MIN_TOTAL_TOUCHES,
    MIN_TOUCHES_PER_KEY,
    TUNING_SPECS,
    _apply_monotropism_depth,
    _observe_monotropism,
)


def _rows(*, k: int, cids: list[str]) -> list[dict]:
    """One retrieval_used row per cid, in the given order (ts-DESC input is
    not order-sensitive for this observe, only the head row's community_k
    matters)."""
    return [{"cue_community_id": cid, "community_k": k} for cid in cids]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registered_with_min_total_touches_as_min_samples() -> None:
    spec = TUNING_SPECS["monotropism_depth"]
    assert spec.knob == "monotropism_depth"
    assert spec.kinds == ("retrieval_used",)
    assert spec.min_samples == MIN_TOTAL_TOUCHES
    assert spec.observe is _observe_monotropism
    assert spec.apply is _apply_monotropism_depth


# ---------------------------------------------------------------------------
# Observe: guards
# ---------------------------------------------------------------------------


def test_observe_empty_events_returns_no_signal() -> None:
    observed, n, signal = _observe_monotropism({}, current={})
    assert observed is None
    assert n == 0
    assert signal == "implicit"


def test_observe_k_below_floor_returns_no_signal() -> None:
    rows = _rows(k=K_MIN - 1, cids=["a"] * (MIN_TOTAL_TOUCHES + 5))
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": rows}, current={},
    )
    assert observed is None
    assert n == 0


def test_observe_no_row_carries_community_k_returns_no_signal() -> None:
    rows = [{"cue_community_id": "a"} for _ in range(20)]
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": rows}, current={},
    )
    assert observed is None
    assert n == 0


def test_observe_below_min_total_touches_returns_no_signal() -> None:
    rows = _rows(k=K_MIN, cids=["a"] * (MIN_TOTAL_TOUCHES - 1))
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": rows}, current={},
    )
    assert observed is None
    assert n == 0


def test_observe_null_cue_community_id_rows_not_counted() -> None:
    # Flat/small-K gated at emit -- present in the window but contribute
    # nothing to K discovery or the touch count.
    null_rows = [{"cue_community_id": None, "community_k": None} for _ in range(30)]
    real_rows = _rows(k=K_MIN, cids=["a"] * MIN_TOTAL_TOUCHES)
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": null_rows + real_rows}, current={},
    )
    assert observed == {"a": MAX_AUTO_DEPTH}
    assert n == MIN_TOTAL_TOUCHES


def test_observe_flat_distribution_returns_no_signal() -> None:
    # K=4, exactly 1/K share each -- no excess over baseline anywhere.
    per_community = MIN_TOTAL_TOUCHES  # 12 touches per cid, 4 cids -> uniform
    cids = ["a", "b", "c", "d"]
    rows = []
    for cid in cids:
        rows.extend(_rows(k=len(cids), cids=[cid] * per_community))
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": rows}, current={},
    )
    assert observed is None
    assert n == 0


def test_observe_below_min_touches_per_key_excluded() -> None:
    # K=4, total touches clears MIN_TOTAL_TOUCHES, but the "b" community's
    # own count is below MIN_TOUCHES_PER_KEY and must contribute no key,
    # even though its share exceeds 1/K.
    rows = (
        _rows(k=4, cids=["a"] * 10)
        + _rows(k=4, cids=["b"] * (MIN_TOUCHES_PER_KEY - 1))
    )
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": rows}, current={},
    )
    assert observed is not None
    assert "b" not in observed
    assert "a" in observed


# ---------------------------------------------------------------------------
# Observe: a real dominant community
# ---------------------------------------------------------------------------


def test_observe_dominant_community_maps_to_bounded_depth() -> None:
    # K=4, baseline=0.25. 15/20 = 0.75 concentration on "a" ->
    # (0.75-0.25)/(0.75) = 0.667 -> capped at MAX_AUTO_DEPTH.
    rows = (
        _rows(k=4, cids=["a"] * 15)
        + _rows(k=4, cids=["b"] * 5)
    )
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": rows}, current={},
    )
    assert observed == {"a": MAX_AUTO_DEPTH}
    assert n == 20
    assert signal == "implicit"


def test_observe_uses_most_recent_qualifying_community_k() -> None:
    # Rows are fetched ts-DESC; the head row's non-null community_k wins,
    # not a later (older) row's differing K.
    head = [{"cue_community_id": "a", "community_k": 4}]
    stale = [{"cue_community_id": "a", "community_k": 2}] * 20
    observed, n, signal = _observe_monotropism(
        {"retrieval_used": head + stale}, current={},
    )
    # K=4 (head) with total=21 touches all on "a": conc=1.0, well above
    # baseline -- must not be gated by the stale K=2 rows.
    assert observed == {"a": MAX_AUTO_DEPTH}


# ---------------------------------------------------------------------------
# Apply: capping, bounded delta, decay/prune
# ---------------------------------------------------------------------------


def test_apply_new_key_bounded_by_max_delta_from_zero() -> None:
    posterior: dict = {"per_key": {"music": {"mean": MAX_AUTO_DEPTH}}}
    new_value = _apply_monotropism_depth({}, {"music": MAX_AUTO_DEPTH}, posterior)
    assert new_value == {"music": MAX_DEPTH_DELTA}
    assert posterior["per_key"]["music"]["unobserved_windows"] == 0


def test_apply_caps_smoothed_mean_at_max_auto_depth() -> None:
    posterior: dict = {"per_key": {"music": {"mean": 0.95}}}
    new_value = _apply_monotropism_depth(
        {"music": MAX_AUTO_DEPTH - MAX_DEPTH_DELTA}, {"music": 0.95}, posterior,
    )
    assert new_value["music"] == MAX_AUTO_DEPTH


def test_apply_bounds_per_night_delta_both_directions() -> None:
    posterior: dict = {"per_key": {"music": {"mean": 0.0}}}
    new_value = _apply_monotropism_depth(
        {"music": 0.5}, {"music": 0.0}, posterior,
    )
    assert new_value["music"] == round(0.5 - MAX_DEPTH_DELTA, 10)


def test_apply_decays_untouched_key() -> None:
    posterior: dict = {"per_key": {"music": {"unobserved_windows": 0}}}
    new_value = _apply_monotropism_depth({"music": 0.5}, {}, posterior)
    assert new_value["music"] == round(0.5 * DEPTH_DECAY, 10)
    assert posterior["per_key"]["music"]["unobserved_windows"] == 1


def test_apply_drops_untouched_key_below_epsilon() -> None:
    tiny = DEPTH_EPSILON / DEPTH_DECAY - 1e-6  # decays to just under epsilon
    posterior: dict = {"per_key": {"music": {"unobserved_windows": 0}}}
    new_value = _apply_monotropism_depth({"music": tiny}, {}, posterior)
    assert "music" not in new_value
    assert "music" not in posterior["per_key"]


def test_apply_prunes_after_max_unobserved_windows_even_if_above_epsilon() -> None:
    # A high value that decays slowly must still be pruned by the windows
    # backstop, not linger in per_key forever.
    posterior: dict = {
        "per_key": {"music": {"unobserved_windows": MAX_UNOBSERVED_WINDOWS - 1}},
    }
    new_value = _apply_monotropism_depth({"music": MAX_AUTO_DEPTH}, {}, posterior)
    assert "music" not in new_value
    assert "music" not in posterior["per_key"]


def test_apply_never_raises_on_malformed_current_value() -> None:
    posterior: dict = {"per_key": {}}
    new_value = _apply_monotropism_depth({"music": "not-a-float"}, {}, posterior)
    assert new_value == {}


def test_apply_vanished_key_decays_and_eventually_drops_without_raising() -> None:
    posterior: dict = {"per_key": {"music": {}}}
    value = 0.5
    current = {"music": value}
    for _ in range(MAX_UNOBSERVED_WINDOWS + 2):
        current = _apply_monotropism_depth(current, {}, posterior)
    assert "music" not in current
    assert "music" not in posterior["per_key"]
