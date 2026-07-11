from __future__ import annotations

from pathlib import Path

import pytest

from iai_mcp.ashby_step import (
    CRISIS_GATING_VARIABLES,
    BreachInfo,
    EssentialVariableTracker,
    TopologySnapshot,
    decide_crisis_transition,
)
from iai_mcp.daemon_config import _load_sleep_overhaul_config
from iai_mcp.lifecycle_state import (
    _validate_record,
    default_state,
    load_state,
    save_state,
)


class _Cfg:
    rich_club_ratio_floor = 0.05
    community_count_ceiling_ratio = 0.9
    edge_density_floor = 0.001
    avg_degree_floor = 2.0
    giant_component_fraction_floor = 0.5
    ev_min_nodes = 100


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "IAI_MCP_RICH_CLUB_RATIO_FLOOR",
        "IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO",
        "IAI_MCP_EDGE_DENSITY_FLOOR",
        "IAI_MCP_CLUSTER_WINDOW_SEC",
        "IAI_MCP_CRISIS_DROP_QUARTILE",
        "IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT",
        "IAI_MCP_SLEEP_OVERHAUL_DRY_RUN",
        "IAI_MCP_AVG_DEGREE_FLOOR",
        "IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR",
        "IAI_MCP_EV_MIN_NODES",
        "IAI_MCP_EV_ARM_AFTER_N",
        "IAI_MCP_EV_DISARM_AFTER_N",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# TopologySnapshot shape
# ---------------------------------------------------------------------------


def test_topology_snapshot_requires_avg_degree_and_giant_component_fraction() -> None:
    with pytest.raises(TypeError):
        TopologySnapshot(  # type: ignore[call-arg]
            rich_club_ratio=0.5,
            community_count=1,
            edge_density=0.5,
            total_nodes=1000,
        )

    snap = TopologySnapshot(
        rich_club_ratio=0.5,
        community_count=1,
        edge_density=0.5,
        total_nodes=1000,
        avg_degree=4.33,
        giant_component_fraction=0.95,
    )
    assert snap.avg_degree == pytest.approx(4.33)
    assert snap.giant_component_fraction == pytest.approx(0.95)


def test_crisis_gating_variables_frozenset_contents() -> None:
    assert CRISIS_GATING_VARIABLES == frozenset(
        {"avg_degree", "giant_component_fraction", "community_count"}
    )
    assert "rich_club_ratio" not in CRISIS_GATING_VARIABLES
    assert "edge_density" not in CRISIS_GATING_VARIABLES


# ---------------------------------------------------------------------------
# EssentialVariableTracker.check() — 5-key dict, scale invariance
# ---------------------------------------------------------------------------


def test_tracker_check_returns_five_keys() -> None:
    tracker = EssentialVariableTracker(_Cfg())
    snap = TopologySnapshot(
        rich_club_ratio=0.5,
        community_count=1,
        edge_density=0.5,
        total_nodes=1000,
        avg_degree=4.33,
        giant_component_fraction=0.95,
    )
    result = tracker.check(snap)
    assert set(result.keys()) == {
        "rich_club_ratio",
        "community_count",
        "edge_density",
        "avg_degree",
        "giant_component_fraction",
    }


def test_prod_shaped_healthy_snapshot_no_gating_breach() -> None:
    tracker = EssentialVariableTracker(_Cfg())
    snap = TopologySnapshot(
        total_nodes=41836,
        community_count=1,
        edge_density=0.000103,
        rich_club_ratio=0.00245,
        avg_degree=4.33,
        giant_component_fraction=0.95,
    )
    result = tracker.check(snap)

    # Non-gating variables are still reported as breached (telemetry-only).
    assert isinstance(result["rich_club_ratio"], BreachInfo)
    assert isinstance(result["edge_density"], BreachInfo)

    # No CRISIS_GATING_VARIABLES member may report a breach at this scale.
    for var_name in CRISIS_GATING_VARIABLES:
        assert result[var_name] is None, (
            f"{var_name} unexpectedly breached on the prod-shaped healthy snapshot: "
            f"{result[var_name]!r}"
        )


def test_fragmented_snapshot_at_scale_gates_breach() -> None:
    tracker = EssentialVariableTracker(_Cfg())
    snap = TopologySnapshot(
        total_nodes=41836,
        community_count=1,
        edge_density=0.000103,
        rich_club_ratio=0.00245,
        avg_degree=1.2,
        giant_component_fraction=0.3,
    )
    result = tracker.check(snap)

    avg_degree_breach = result["avg_degree"]
    assert isinstance(avg_degree_breach, BreachInfo)
    assert avg_degree_breach.direction == "floor_breach"
    assert avg_degree_breach.observed_value == pytest.approx(1.2)
    assert avg_degree_breach.threshold == pytest.approx(2.0)

    giant_breach = result["giant_component_fraction"]
    assert isinstance(giant_breach, BreachInfo)
    assert giant_breach.direction == "floor_breach"
    assert giant_breach.observed_value == pytest.approx(0.3)
    assert giant_breach.threshold == pytest.approx(0.5)


def test_tiny_corpus_guard_below_min_nodes_all_none() -> None:
    tracker = EssentialVariableTracker(_Cfg())
    snap = TopologySnapshot(
        total_nodes=50,
        community_count=40,
        edge_density=0.0001,
        rich_club_ratio=0.0001,
        avg_degree=0.1,
        giant_component_fraction=0.01,
    )
    result = tracker.check(snap)
    assert set(result.keys()) == {
        "rich_club_ratio",
        "community_count",
        "edge_density",
        "avg_degree",
        "giant_component_fraction",
    }
    assert all(v is None for v in result.values())


def test_empty_snapshot_still_all_none() -> None:
    tracker = EssentialVariableTracker(_Cfg())
    snap = TopologySnapshot(
        total_nodes=0,
        community_count=0,
        edge_density=0.0,
        rich_club_ratio=0.0,
        avg_degree=0.0,
        giant_component_fraction=0.0,
    )
    result = tracker.check(snap)
    assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# decide_crisis_transition — pure hysteresis function
# ---------------------------------------------------------------------------


def test_decide_crisis_transition_arms_after_n_consecutive_breaches() -> None:
    action, breaches, clears = decide_crisis_transition(
        True, 0, 0, arm_after_n=3, disarm_after_n=3
    )
    assert action is None
    assert (breaches, clears) == (1, 0)

    action, breaches, clears = decide_crisis_transition(
        True, breaches, clears, arm_after_n=3, disarm_after_n=3
    )
    assert action is None
    assert (breaches, clears) == (2, 0)

    action, breaches, clears = decide_crisis_transition(
        True, breaches, clears, arm_after_n=3, disarm_after_n=3
    )
    assert action is True
    assert (breaches, clears) == (3, 0)


def test_decide_crisis_transition_breach_streak_broken_resets() -> None:
    action, breaches, clears = decide_crisis_transition(
        True, 0, 0, arm_after_n=3, disarm_after_n=3
    )
    assert (action, breaches, clears) == (None, 1, 0)
    action, breaches, clears = decide_crisis_transition(
        True, breaches, clears, arm_after_n=3, disarm_after_n=3
    )
    assert (action, breaches, clears) == (None, 2, 0)

    # A single clear cycle breaks the breach streak.
    action, breaches, clears = decide_crisis_transition(
        False, breaches, clears, arm_after_n=3, disarm_after_n=3
    )
    assert action is None
    assert (breaches, clears) == (0, 1)


def test_decide_crisis_transition_disarms_after_n_consecutive_clears() -> None:
    action, breaches, clears = decide_crisis_transition(
        False, 0, 0, arm_after_n=3, disarm_after_n=3
    )
    assert (action, breaches, clears) == (None, 0, 1)
    action, breaches, clears = decide_crisis_transition(
        False, breaches, clears, arm_after_n=3, disarm_after_n=3
    )
    assert (action, breaches, clears) == (None, 0, 2)
    action, breaches, clears = decide_crisis_transition(
        False, breaches, clears, arm_after_n=3, disarm_after_n=3
    )
    assert action is False
    assert (breaches, clears) == (0, 3)


def test_decide_crisis_transition_alternating_never_arms_or_disarms() -> None:
    breaches, clears = 0, 0
    for i in range(10):
        breach = i % 2 == 0
        action, breaches, clears = decide_crisis_transition(
            breach, breaches, clears, arm_after_n=3, disarm_after_n=3
        )
        assert action is None
        assert breaches <= 1
        assert clears <= 1


# ---------------------------------------------------------------------------
# daemon_config: 5 new SleepOverhaulConfig fields + env knobs
# ---------------------------------------------------------------------------


def test_load_sleep_overhaul_config_defaults_for_new_knobs() -> None:
    cfg = _load_sleep_overhaul_config()
    assert cfg.avg_degree_floor == pytest.approx(2.0)
    assert cfg.giant_component_fraction_floor == pytest.approx(0.5)
    assert cfg.ev_min_nodes == 100
    assert cfg.ev_arm_after_n == 3
    assert cfg.ev_disarm_after_n == 3


def test_load_sleep_overhaul_config_round_trips_new_env_knobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_AVG_DEGREE_FLOOR", "3.5")
    monkeypatch.setenv("IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR", "0.75")
    monkeypatch.setenv("IAI_MCP_EV_MIN_NODES", "250")
    monkeypatch.setenv("IAI_MCP_EV_ARM_AFTER_N", "5")
    monkeypatch.setenv("IAI_MCP_EV_DISARM_AFTER_N", "7")

    cfg = _load_sleep_overhaul_config()
    assert cfg.avg_degree_floor == pytest.approx(3.5)
    assert cfg.giant_component_fraction_floor == pytest.approx(0.75)
    assert cfg.ev_min_nodes == 250
    assert cfg.ev_arm_after_n == 5
    assert cfg.ev_disarm_after_n == 7


@pytest.mark.parametrize(
    "var_name,bad_value",
    [
        ("IAI_MCP_AVG_DEGREE_FLOOR", "0.0"),
        ("IAI_MCP_AVG_DEGREE_FLOOR", "not_a_float"),
        ("IAI_MCP_AVG_DEGREE_FLOOR", "1001.0"),
        ("IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR", "0.0"),
        ("IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR", "1.1"),
        ("IAI_MCP_EV_MIN_NODES", "0"),
        ("IAI_MCP_EV_MIN_NODES", "not_an_int"),
        ("IAI_MCP_EV_MIN_NODES", "10000001"),
        ("IAI_MCP_EV_ARM_AFTER_N", "0"),
        ("IAI_MCP_EV_ARM_AFTER_N", "101"),
        ("IAI_MCP_EV_DISARM_AFTER_N", "0"),
        ("IAI_MCP_EV_DISARM_AFTER_N", "101"),
    ],
)
def test_load_sleep_overhaul_config_new_knobs_fail_loud_naming(
    monkeypatch: pytest.MonkeyPatch,
    var_name: str,
    bad_value: str,
) -> None:
    monkeypatch.setenv(var_name, bad_value)
    with pytest.raises(ValueError) as exc_info:
        _load_sleep_overhaul_config()
    assert var_name in str(exc_info.value), (
        f"ValueError message {str(exc_info.value)!r} must contain {var_name!r}"
    )


def test_tracker_init_reads_new_cfg_attributes() -> None:
    cfg = _load_sleep_overhaul_config()
    # Must not raise -- tracker reads avg_degree_floor / giant_component_fraction_floor
    # / ev_min_nodes in addition to the three pre-existing floor/ceiling reads.
    EssentialVariableTracker(cfg)


# ---------------------------------------------------------------------------
# lifecycle_state.py — hysteresis counter fields (Task 2)
# ---------------------------------------------------------------------------


def test_default_state_carries_both_counters_at_zero() -> None:
    state = default_state()
    assert state["essential_variable_consecutive_breaches"] == 0
    assert state["essential_variable_consecutive_clears"] == 0


def test_legacy_record_without_counter_keys_validates_as_zero() -> None:
    legacy = default_state()
    del legacy["essential_variable_consecutive_breaches"]  # type: ignore[misc]
    del legacy["essential_variable_consecutive_clears"]  # type: ignore[misc]
    validated = _validate_record(dict(legacy))
    assert validated["essential_variable_consecutive_breaches"] == 0
    assert validated["essential_variable_consecutive_clears"] == 0


def test_save_load_state_round_trips_nonzero_counters(tmp_path: Path) -> None:
    path = tmp_path / "lifecycle_state.json"
    state = default_state()
    state["essential_variable_consecutive_breaches"] = 2
    state["essential_variable_consecutive_clears"] = 0
    save_state(state, path)

    loaded = load_state(path)
    assert loaded["essential_variable_consecutive_breaches"] == 2
    assert loaded["essential_variable_consecutive_clears"] == 0


@pytest.mark.parametrize(
    "key,bad_value",
    [
        ("essential_variable_consecutive_breaches", -1),
        ("essential_variable_consecutive_breaches", "not_an_int"),
        ("essential_variable_consecutive_clears", -1),
        ("essential_variable_consecutive_clears", "not_an_int"),
    ],
)
def test_validate_record_rejects_bad_counter_values(
    key: str, bad_value: object
) -> None:
    record = dict(default_state())
    record[key] = bad_value  # type: ignore[literal-required]
    with pytest.raises(ValueError):
        _validate_record(record)
