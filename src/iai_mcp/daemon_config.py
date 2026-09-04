from __future__ import annotations

import os
from dataclasses import dataclass


_ERASURE_DEFAULT_CENTRALITY_THRESHOLD: float = 0.02
_ERASURE_DEFAULT_AGE_DAYS: int = 30
_ERASURE_DEFAULT_RETRIEVAL_WINDOW_DAYS: int = 30
_ERASURE_DEFAULT_TOMBSTONE_TTL_SEC: int = 604800

_ERASURE_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_ERASURE_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ErasureConfig:

    centrality_threshold: float
    age_days: int
    retrieval_window_days: int
    tombstone_ttl_sec: int
    dry_run: bool


def _load_erasure_config() -> ErasureConfig:
    raw_centrality = os.environ.get(
        "IAI_MCP_ERASURE_CENTRALITY_THRESHOLD",
        str(_ERASURE_DEFAULT_CENTRALITY_THRESHOLD),
    )
    try:
        centrality_threshold = float(raw_centrality)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_CENTRALITY_THRESHOLD: invalid value "
            f"{raw_centrality!r}, expected float"
        )
    if not (0.0 <= centrality_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_ERASURE_CENTRALITY_THRESHOLD: invalid value "
            f"{raw_centrality!r}, expected float in [0.0, 1.0]"
        )

    raw_age_days = os.environ.get(
        "IAI_MCP_ERASURE_AGE_DAYS", str(_ERASURE_DEFAULT_AGE_DAYS)
    )
    try:
        age_days = int(raw_age_days)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_AGE_DAYS: invalid value {raw_age_days!r}, "
            f"expected int"
        )
    if age_days <= 0:
        raise ValueError(
            f"IAI_MCP_ERASURE_AGE_DAYS: invalid value {raw_age_days!r}, "
            f"expected int > 0"
        )

    raw_retrieval_window = os.environ.get(
        "IAI_MCP_ERASURE_RETRIEVAL_WINDOW_DAYS",
        str(_ERASURE_DEFAULT_RETRIEVAL_WINDOW_DAYS),
    )
    try:
        retrieval_window_days = int(raw_retrieval_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_RETRIEVAL_WINDOW_DAYS: invalid value "
            f"{raw_retrieval_window!r}, expected int"
        )
    if retrieval_window_days <= 0:
        raise ValueError(
            f"IAI_MCP_ERASURE_RETRIEVAL_WINDOW_DAYS: invalid value "
            f"{raw_retrieval_window!r}, expected int > 0"
        )

    raw_tombstone_ttl = os.environ.get(
        "IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC",
        str(_ERASURE_DEFAULT_TOMBSTONE_TTL_SEC),
    )
    try:
        tombstone_ttl_sec = int(raw_tombstone_ttl)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC: invalid value "
            f"{raw_tombstone_ttl!r}, expected int"
        )
    if tombstone_ttl_sec <= 0:
        raise ValueError(
            f"IAI_MCP_ERASURE_TOMBSTONE_TTL_SEC: invalid value "
            f"{raw_tombstone_ttl!r}, expected int > 0"
        )

    raw_dry_run = os.environ.get("IAI_MCP_ERASURE_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _ERASURE_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _ERASURE_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_ERASURE_DRY_RUN: invalid value {raw_dry_run!r}, "
                f"expected one of "
                f"{sorted(_ERASURE_DRY_RUN_TRUE_VALUES | _ERASURE_DRY_RUN_FALSE_VALUES)}"
            )

    return ErasureConfig(
        centrality_threshold=centrality_threshold,
        age_days=age_days,
        retrieval_window_days=retrieval_window_days,
        tombstone_ttl_sec=tombstone_ttl_sec,
        dry_run=dry_run,
    )


_PATSEP_DEFAULT_NEAR_DUP_THRESHOLD: float = 0.92
_PATSEP_DEFAULT_LINK_THRESHOLD: float = 0.70
_PATSEP_DEFAULT_LINK_INITIAL_WEIGHT: float = 0.10
_PATSEP_DEFAULT_TOP_K: int = 8

_PATSEP_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_PATSEP_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class PatSepConfig:

    near_dup_threshold: float
    link_threshold: float
    link_initial_weight: float
    top_k: int
    dry_run: bool


def _load_patsep_config() -> PatSepConfig:
    raw_near_dup = os.environ.get(
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        str(_PATSEP_DEFAULT_NEAR_DUP_THRESHOLD),
    )
    try:
        near_dup_threshold = float(raw_near_dup)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD: invalid value "
            f"{raw_near_dup!r}, expected float"
        )
    if not (0.0 < near_dup_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD: invalid value "
            f"{raw_near_dup!r}, expected float in (0.0, 1.0]"
        )

    raw_link = os.environ.get(
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        str(_PATSEP_DEFAULT_LINK_THRESHOLD),
    )
    try:
        link_threshold = float(raw_link)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_THRESHOLD: invalid value "
            f"{raw_link!r}, expected float"
        )
    if not (0.0 < link_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_THRESHOLD: invalid value "
            f"{raw_link!r}, expected float in (0.0, 1.0]"
        )

    if not (link_threshold < near_dup_threshold):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_THRESHOLD: invalid value {raw_link!r}, "
            f"must be strictly less than IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD "
            f"(got link={link_threshold}, near_dup={near_dup_threshold})"
        )

    raw_weight = os.environ.get(
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        str(_PATSEP_DEFAULT_LINK_INITIAL_WEIGHT),
    )
    try:
        link_initial_weight = float(raw_weight)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT: invalid value "
            f"{raw_weight!r}, expected float"
        )
    if not (0.0 < link_initial_weight <= 1.0):
        raise ValueError(
            f"IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT: invalid value "
            f"{raw_weight!r}, expected float in (0.0, 1.0]"
        )

    raw_top_k = os.environ.get(
        "IAI_MCP_PATSEP_TOP_K", str(_PATSEP_DEFAULT_TOP_K)
    )
    try:
        top_k = int(raw_top_k)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PATSEP_TOP_K: invalid value {raw_top_k!r}, "
            f"expected int"
        )
    if not (1 <= top_k <= 64):
        raise ValueError(
            f"IAI_MCP_PATSEP_TOP_K: invalid value {raw_top_k!r}, "
            f"expected int in [1, 64]"
        )

    raw_dry_run = os.environ.get("IAI_MCP_PATSEP_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _PATSEP_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _PATSEP_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_PATSEP_DRY_RUN: invalid value {raw_dry_run!r}, "
                f"expected one of "
                f"{sorted(_PATSEP_DRY_RUN_TRUE_VALUES | _PATSEP_DRY_RUN_FALSE_VALUES)}"
            )

    return PatSepConfig(
        near_dup_threshold=near_dup_threshold,
        link_threshold=link_threshold,
        link_initial_weight=link_initial_weight,
        top_k=top_k,
        dry_run=dry_run,
    )


_S2_DEFAULT_MIN_INTERVAL_SEC: float = 5.0
_S2_DEFAULT_MAX_RETRY: int = 3

_S2_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_S2_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class S2Config:

    min_interval_sec: float
    max_retry: int
    dry_run: bool


def _load_s2_config() -> S2Config:
    raw_min_interval = os.environ.get(
        "IAI_MCP_S2_MIN_INTERVAL_SEC",
        str(_S2_DEFAULT_MIN_INTERVAL_SEC),
    )
    try:
        min_interval_sec = float(raw_min_interval)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_S2_MIN_INTERVAL_SEC: invalid value "
            f"{raw_min_interval!r}, expected float"
        )
    if not (min_interval_sec > 0.0):
        raise ValueError(
            f"IAI_MCP_S2_MIN_INTERVAL_SEC: invalid value "
            f"{raw_min_interval!r}, expected float > 0.0"
        )

    raw_max_retry = os.environ.get(
        "IAI_MCP_S2_MAX_RETRY", str(_S2_DEFAULT_MAX_RETRY)
    )
    try:
        max_retry = int(raw_max_retry)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_S2_MAX_RETRY: invalid value {raw_max_retry!r}, "
            f"expected int"
        )
    if not (0 <= max_retry <= 10):
        raise ValueError(
            f"IAI_MCP_S2_MAX_RETRY: invalid value {raw_max_retry!r}, "
            f"expected int in [0, 10]"
        )

    raw_dry_run = os.environ.get("IAI_MCP_S2_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _S2_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _S2_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_S2_DRY_RUN: invalid value {raw_dry_run!r}, "
                f"expected one of "
                f"{sorted(_S2_DRY_RUN_TRUE_VALUES | _S2_DRY_RUN_FALSE_VALUES)}"
            )

    return S2Config(
        min_interval_sec=min_interval_sec,
        max_retry=max_retry,
        dry_run=dry_run,
    )


_SLEEP_OVERHAUL_DEFAULT_RICH_CLUB_RATIO_FLOOR: float = 0.05
_SLEEP_OVERHAUL_DEFAULT_COMMUNITY_COUNT_CEILING_RATIO: float = 0.9
_SLEEP_OVERHAUL_DEFAULT_EDGE_DENSITY_FLOOR: float = 0.001
_SLEEP_OVERHAUL_DEFAULT_CLUSTER_WINDOW_SEC: int = 300
_SLEEP_OVERHAUL_DEFAULT_CRISIS_DROP_QUARTILE: float = 0.25
_SLEEP_OVERHAUL_DEFAULT_CLUSTER_REPLAY_INITIAL_WEIGHT: float = 0.05
_SLEEP_OVERHAUL_DEFAULT_AVG_DEGREE_FLOOR: float = 2.0
_SLEEP_OVERHAUL_DEFAULT_GIANT_COMPONENT_FRACTION_FLOOR: float = 0.5
_SLEEP_OVERHAUL_DEFAULT_EV_MIN_NODES: int = 100
_SLEEP_OVERHAUL_DEFAULT_EV_ARM_AFTER_N: int = 3
_SLEEP_OVERHAUL_DEFAULT_EV_DISARM_AFTER_N: int = 3

_SLEEP_OVERHAUL_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_SLEEP_OVERHAUL_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class SleepOverhaulConfig:

    rich_club_ratio_floor: float
    community_count_ceiling_ratio: float
    edge_density_floor: float
    cluster_window_sec: int
    crisis_drop_quartile: float
    cluster_replay_initial_weight: float
    dry_run: bool
    avg_degree_floor: float
    giant_component_fraction_floor: float
    ev_min_nodes: int
    ev_arm_after_n: int
    ev_disarm_after_n: int


def _load_sleep_overhaul_config() -> SleepOverhaulConfig:
    raw_rich_club = os.environ.get(
        "IAI_MCP_RICH_CLUB_RATIO_FLOOR",
        str(_SLEEP_OVERHAUL_DEFAULT_RICH_CLUB_RATIO_FLOOR),
    )
    try:
        rich_club_ratio_floor = float(raw_rich_club)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_RICH_CLUB_RATIO_FLOOR: invalid value "
            f"{raw_rich_club!r}, expected float"
        )
    if not (0.0 < rich_club_ratio_floor <= 1.0):
        raise ValueError(
            f"IAI_MCP_RICH_CLUB_RATIO_FLOOR: invalid value "
            f"{raw_rich_club!r}, expected float in (0.0, 1.0]"
        )

    raw_comm_ceiling = os.environ.get(
        "IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO",
        str(_SLEEP_OVERHAUL_DEFAULT_COMMUNITY_COUNT_CEILING_RATIO),
    )
    try:
        community_count_ceiling_ratio = float(raw_comm_ceiling)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO: invalid value "
            f"{raw_comm_ceiling!r}, expected float"
        )
    if not (0.0 < community_count_ceiling_ratio <= 1.0):
        raise ValueError(
            f"IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO: invalid value "
            f"{raw_comm_ceiling!r}, expected float in (0.0, 1.0]"
        )

    raw_edge_density = os.environ.get(
        "IAI_MCP_EDGE_DENSITY_FLOOR",
        str(_SLEEP_OVERHAUL_DEFAULT_EDGE_DENSITY_FLOOR),
    )
    try:
        edge_density_floor = float(raw_edge_density)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_EDGE_DENSITY_FLOOR: invalid value "
            f"{raw_edge_density!r}, expected float"
        )
    if not (0.0 < edge_density_floor <= 1.0):
        raise ValueError(
            f"IAI_MCP_EDGE_DENSITY_FLOOR: invalid value "
            f"{raw_edge_density!r}, expected float in (0.0, 1.0]"
        )

    raw_window = os.environ.get(
        "IAI_MCP_CLUSTER_WINDOW_SEC",
        str(_SLEEP_OVERHAUL_DEFAULT_CLUSTER_WINDOW_SEC),
    )
    try:
        cluster_window_sec = int(raw_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_CLUSTER_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int"
        )
    if not (1 <= cluster_window_sec <= 86400):
        raise ValueError(
            f"IAI_MCP_CLUSTER_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int in [1, 86400]"
        )

    raw_quartile = os.environ.get(
        "IAI_MCP_CRISIS_DROP_QUARTILE",
        str(_SLEEP_OVERHAUL_DEFAULT_CRISIS_DROP_QUARTILE),
    )
    try:
        crisis_drop_quartile = float(raw_quartile)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_CRISIS_DROP_QUARTILE: invalid value "
            f"{raw_quartile!r}, expected float"
        )
    if not (0.0 < crisis_drop_quartile < 1.0):
        raise ValueError(
            f"IAI_MCP_CRISIS_DROP_QUARTILE: invalid value "
            f"{raw_quartile!r}, expected float in (0.0, 1.0)"
        )

    raw_replay_weight = os.environ.get(
        "IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT",
        str(_SLEEP_OVERHAUL_DEFAULT_CLUSTER_REPLAY_INITIAL_WEIGHT),
    )
    try:
        cluster_replay_initial_weight = float(raw_replay_weight)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT: invalid value "
            f"{raw_replay_weight!r}, expected float"
        )
    if not (0.0 < cluster_replay_initial_weight <= 1.0):
        raise ValueError(
            f"IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT: invalid value "
            f"{raw_replay_weight!r}, expected float in (0.0, 1.0]"
        )

    raw_dry_run = os.environ.get("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _SLEEP_OVERHAUL_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _SLEEP_OVERHAUL_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_SLEEP_OVERHAUL_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_SLEEP_OVERHAUL_DRY_RUN_TRUE_VALUES | _SLEEP_OVERHAUL_DRY_RUN_FALSE_VALUES)}"
            )

    raw_avg_degree_floor = os.environ.get(
        "IAI_MCP_AVG_DEGREE_FLOOR",
        str(_SLEEP_OVERHAUL_DEFAULT_AVG_DEGREE_FLOOR),
    )
    try:
        avg_degree_floor = float(raw_avg_degree_floor)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_AVG_DEGREE_FLOOR: invalid value "
            f"{raw_avg_degree_floor!r}, expected float"
        )
    if not (0.0 < avg_degree_floor <= 1000.0):
        raise ValueError(
            f"IAI_MCP_AVG_DEGREE_FLOOR: invalid value "
            f"{raw_avg_degree_floor!r}, expected float in (0.0, 1000.0]"
        )

    raw_giant_component_floor = os.environ.get(
        "IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR",
        str(_SLEEP_OVERHAUL_DEFAULT_GIANT_COMPONENT_FRACTION_FLOOR),
    )
    try:
        giant_component_fraction_floor = float(raw_giant_component_floor)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR: invalid value "
            f"{raw_giant_component_floor!r}, expected float"
        )
    if not (0.0 < giant_component_fraction_floor <= 1.0):
        raise ValueError(
            f"IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR: invalid value "
            f"{raw_giant_component_floor!r}, expected float in (0.0, 1.0]"
        )

    raw_ev_min_nodes = os.environ.get(
        "IAI_MCP_EV_MIN_NODES",
        str(_SLEEP_OVERHAUL_DEFAULT_EV_MIN_NODES),
    )
    try:
        ev_min_nodes = int(raw_ev_min_nodes)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_EV_MIN_NODES: invalid value {raw_ev_min_nodes!r}, "
            f"expected int"
        )
    if not (1 <= ev_min_nodes <= 10000000):
        raise ValueError(
            f"IAI_MCP_EV_MIN_NODES: invalid value {raw_ev_min_nodes!r}, "
            f"expected int in [1, 10000000]"
        )

    raw_ev_arm_after_n = os.environ.get(
        "IAI_MCP_EV_ARM_AFTER_N",
        str(_SLEEP_OVERHAUL_DEFAULT_EV_ARM_AFTER_N),
    )
    try:
        ev_arm_after_n = int(raw_ev_arm_after_n)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_EV_ARM_AFTER_N: invalid value {raw_ev_arm_after_n!r}, "
            f"expected int"
        )
    if not (1 <= ev_arm_after_n <= 100):
        raise ValueError(
            f"IAI_MCP_EV_ARM_AFTER_N: invalid value {raw_ev_arm_after_n!r}, "
            f"expected int in [1, 100]"
        )

    raw_ev_disarm_after_n = os.environ.get(
        "IAI_MCP_EV_DISARM_AFTER_N",
        str(_SLEEP_OVERHAUL_DEFAULT_EV_DISARM_AFTER_N),
    )
    try:
        ev_disarm_after_n = int(raw_ev_disarm_after_n)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_EV_DISARM_AFTER_N: invalid value "
            f"{raw_ev_disarm_after_n!r}, expected int"
        )
    if not (1 <= ev_disarm_after_n <= 100):
        raise ValueError(
            f"IAI_MCP_EV_DISARM_AFTER_N: invalid value "
            f"{raw_ev_disarm_after_n!r}, expected int in [1, 100]"
        )

    return SleepOverhaulConfig(
        rich_club_ratio_floor=rich_club_ratio_floor,
        community_count_ceiling_ratio=community_count_ceiling_ratio,
        edge_density_floor=edge_density_floor,
        cluster_window_sec=cluster_window_sec,
        crisis_drop_quartile=crisis_drop_quartile,
        cluster_replay_initial_weight=cluster_replay_initial_weight,
        dry_run=dry_run,
        avg_degree_floor=avg_degree_floor,
        giant_component_fraction_floor=giant_component_fraction_floor,
        ev_min_nodes=ev_min_nodes,
        ev_arm_after_n=ev_arm_after_n,
        ev_disarm_after_n=ev_disarm_after_n,
    )


_RECONSOLIDATION_DEFAULT_SCHEMA_BYPASS_COS_THRESHOLD: float = 0.85
_RECONSOLIDATION_DEFAULT_LABILE_WINDOW_SEC: int = 21600
_RECONSOLIDATION_DEFAULT_TIER1: bool = False
_RECONSOLIDATION_DEFAULT_ERROR_THRESHOLD: float = 0.5
_RECONSOLIDATION_DEFAULT_DRY_RUN_PRODUCTION: bool = False

_RECONSOLIDATION_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_RECONSOLIDATION_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class ReconsolidationConfig:

    schema_bypass_cos_threshold: float
    labile_window_sec: int
    reconsolidation_tier1: bool
    reconsolidation_error_threshold: float
    dry_run: bool


def _load_reconsolidation_config() -> ReconsolidationConfig:
    raw_cos = os.environ.get(
        "IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD",
        str(_RECONSOLIDATION_DEFAULT_SCHEMA_BYPASS_COS_THRESHOLD),
    )
    try:
        schema_bypass_cos_threshold = float(raw_cos)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD: invalid value "
            f"{raw_cos!r}, expected float"
        )
    if not (0.0 <= schema_bypass_cos_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD: invalid value "
            f"{raw_cos!r}, expected float in [0.0, 1.0]"
        )

    raw_window = os.environ.get(
        "IAI_MCP_LABILE_WINDOW_SEC",
        str(_RECONSOLIDATION_DEFAULT_LABILE_WINDOW_SEC),
    )
    try:
        labile_window_sec = int(raw_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_LABILE_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int"
        )
    if not (labile_window_sec > 0):
        raise ValueError(
            f"IAI_MCP_LABILE_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int > 0"
        )

    raw_tier1 = os.environ.get("IAI_MCP_RECONSOLIDATION_TIER1")
    if raw_tier1 is None:
        reconsolidation_tier1 = _RECONSOLIDATION_DEFAULT_TIER1
    else:
        normalized_tier1 = raw_tier1.strip().lower()
        if normalized_tier1 in _RECONSOLIDATION_TRUE_VALUES:
            reconsolidation_tier1 = True
        elif normalized_tier1 in _RECONSOLIDATION_FALSE_VALUES:
            reconsolidation_tier1 = False
        else:
            raise ValueError(
                f"IAI_MCP_RECONSOLIDATION_TIER1: invalid value "
                f"{raw_tier1!r}, expected one of "
                f"{sorted(_RECONSOLIDATION_TRUE_VALUES | _RECONSOLIDATION_FALSE_VALUES)}"
            )

    raw_err = os.environ.get(
        "IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD",
        str(_RECONSOLIDATION_DEFAULT_ERROR_THRESHOLD),
    )
    try:
        reconsolidation_error_threshold = float(raw_err)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD: invalid value "
            f"{raw_err!r}, expected float"
        )
    if not (0.0 <= reconsolidation_error_threshold <= 1.0):
        raise ValueError(
            f"IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD: invalid value "
            f"{raw_err!r}, expected float in [0.0, 1.0]"
        )

    raw_dry_run = os.environ.get("IAI_MCP_RECONSOLIDATION_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized_dry = raw_dry_run.strip().lower()
        if normalized_dry in _RECONSOLIDATION_TRUE_VALUES:
            dry_run = True
        elif normalized_dry in _RECONSOLIDATION_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_RECONSOLIDATION_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_RECONSOLIDATION_TRUE_VALUES | _RECONSOLIDATION_FALSE_VALUES)}"
            )

    return ReconsolidationConfig(
        schema_bypass_cos_threshold=schema_bypass_cos_threshold,
        labile_window_sec=labile_window_sec,
        reconsolidation_tier1=reconsolidation_tier1,
        reconsolidation_error_threshold=reconsolidation_error_threshold,
        dry_run=dry_run,
    )


@dataclass(frozen=True)
class ReconsolidationValenceConfig:

    enabled: bool


def _load_reconsolidation_valence_config() -> ReconsolidationValenceConfig:
    # Same kill-switch store.raise_valence already honors -- one env var,
    # one meaning, checked in both places.
    return ReconsolidationValenceConfig(
        enabled=os.environ.get("IAI_MCP_VALENCE_WRITE_OFF") != "1",
    )


_STC_DEFAULT_PERI_EVENT_BUFFER_SIZE: int = 20
_STC_DEFAULT_PERI_EVENT_WINDOW_SEC: int = 1800
_STC_DEFAULT_STRONG_EVENT_TYPES: str = "memory_capture,error_trace,user_correction"

_STC_DRY_RUN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_STC_DRY_RUN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class StcConfig:

    peri_event_buffer_size: int
    peri_event_window_sec: int
    strong_event_types: frozenset[str]
    dry_run: bool


def _load_stc_config() -> StcConfig:
    raw_buffer_size = os.environ.get(
        "IAI_MCP_PERI_EVENT_BUFFER_SIZE",
        str(_STC_DEFAULT_PERI_EVENT_BUFFER_SIZE),
    )
    try:
        peri_event_buffer_size = int(raw_buffer_size)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_BUFFER_SIZE: invalid value "
            f"{raw_buffer_size!r}, expected int"
        )
    if not (1 <= peri_event_buffer_size <= 1000):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_BUFFER_SIZE: invalid value "
            f"{raw_buffer_size!r}, expected int in [1, 1000]"
        )

    raw_window = os.environ.get(
        "IAI_MCP_PERI_EVENT_WINDOW_SEC",
        str(_STC_DEFAULT_PERI_EVENT_WINDOW_SEC),
    )
    try:
        peri_event_window_sec = int(raw_window)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int"
        )
    if not (1 <= peri_event_window_sec <= 86400):
        raise ValueError(
            f"IAI_MCP_PERI_EVENT_WINDOW_SEC: invalid value {raw_window!r}, "
            f"expected int in [1, 86400]"
        )

    raw_strong = os.environ.get(
        "IAI_MCP_STC_STRONG_EVENT_TYPES",
        _STC_DEFAULT_STRONG_EVENT_TYPES,
    )
    tokens = [tok.strip().lower() for tok in raw_strong.split(",")]
    if any(tok == "" for tok in tokens):
        raise ValueError(
            f"IAI_MCP_STC_STRONG_EVENT_TYPES: invalid value {raw_strong!r}, "
            f"expected non-empty comma-separated event-type names"
        )
    strong_event_types = frozenset(tokens)
    if not strong_event_types:
        raise ValueError(
            f"IAI_MCP_STC_STRONG_EVENT_TYPES: invalid value {raw_strong!r}, "
            f"expected non-empty comma-separated event-type names"
        )

    raw_dry_run = os.environ.get("IAI_MCP_STC_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _STC_DRY_RUN_TRUE_VALUES:
            dry_run = True
        elif normalized in _STC_DRY_RUN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_STC_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_STC_DRY_RUN_TRUE_VALUES | _STC_DRY_RUN_FALSE_VALUES)}"
            )

    return StcConfig(
        peri_event_buffer_size=peri_event_buffer_size,
        peri_event_window_sec=peri_event_window_sec,
        strong_event_types=strong_event_types,
        dry_run=dry_run,
    )


_USER_MODEL_DEFAULT_AGGREGATION_WINDOW_DAYS: int = 30
_USER_MODEL_DEFAULT_PREFETCH_TOP_K: int = 10
_USER_MODEL_DEFAULT_PATH: str = "~/.iai-mcp/user_model.json"
_USER_MODEL_DEFAULT_DRY_RUN_PRODUCTION: bool = False

_USER_MODEL_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_USER_MODEL_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class UserModelConfig:

    aggregation_window_days: int
    prefetch_top_k: int
    user_model_path: str
    dry_run: bool


def _load_user_model_config() -> UserModelConfig:
    raw_window_days = os.environ.get(
        "IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS",
        str(_USER_MODEL_DEFAULT_AGGREGATION_WINDOW_DAYS),
    )
    try:
        aggregation_window_days = int(raw_window_days)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS: invalid value "
            f"{raw_window_days!r}, expected int"
        )
    if not (1 <= aggregation_window_days <= 365):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_AGGREGATION_WINDOW_DAYS: invalid value "
            f"{raw_window_days!r}, expected int in [1, 365]"
        )

    raw_top_k = os.environ.get(
        "IAI_MCP_USER_MODEL_PREFETCH_TOP_K",
        str(_USER_MODEL_DEFAULT_PREFETCH_TOP_K),
    )
    try:
        prefetch_top_k = int(raw_top_k)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_PREFETCH_TOP_K: invalid value "
            f"{raw_top_k!r}, expected int"
        )
    if not (1 <= prefetch_top_k <= 100):
        raise ValueError(
            f"IAI_MCP_USER_MODEL_PREFETCH_TOP_K: invalid value "
            f"{raw_top_k!r}, expected int in [1, 100]"
        )

    raw_path = os.environ.get("IAI_MCP_USER_MODEL_PATH")
    if raw_path is None or raw_path == "":
        user_model_path = _USER_MODEL_DEFAULT_PATH
    else:
        user_model_path = raw_path

    raw_dry_run = os.environ.get("IAI_MCP_USER_MODEL_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _USER_MODEL_TRUE_VALUES:
            dry_run = True
        elif normalized in _USER_MODEL_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_USER_MODEL_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_USER_MODEL_TRUE_VALUES | _USER_MODEL_FALSE_VALUES)}"
            )

    return UserModelConfig(
        aggregation_window_days=aggregation_window_days,
        prefetch_top_k=prefetch_top_k,
        user_model_path=user_model_path,
        dry_run=dry_run,
    )


_SPATIAL_DEFAULT_AUTO_TAG: bool = False
_SPATIAL_DEFAULT_WING: str = "general"
_SPATIAL_DEFAULT_DRY_RUN_PRODUCTION: bool = False

_SPATIAL_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_SPATIAL_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class SpatialConfig:

    auto_tag: bool
    default_wing: str
    dry_run: bool


def _load_spatial_config() -> SpatialConfig:
    raw_auto_tag = os.environ.get("IAI_MCP_SPATIAL_AUTO_TAG")
    if raw_auto_tag is None:
        auto_tag = _SPATIAL_DEFAULT_AUTO_TAG
    else:
        normalized = raw_auto_tag.strip().lower()
        if normalized in _SPATIAL_TRUE_VALUES:
            auto_tag = True
        elif normalized in _SPATIAL_FALSE_VALUES:
            auto_tag = False
        else:
            raise ValueError(
                f"IAI_MCP_SPATIAL_AUTO_TAG: invalid value "
                f"{raw_auto_tag!r}, expected one of "
                f"{sorted(_SPATIAL_TRUE_VALUES | _SPATIAL_FALSE_VALUES)}"
            )

    raw_wing = os.environ.get("IAI_MCP_SPATIAL_DEFAULT_WING")
    if raw_wing is None or raw_wing == "":
        default_wing = _SPATIAL_DEFAULT_WING
    else:
        default_wing = raw_wing

    raw_dry_run = os.environ.get("IAI_MCP_SPATIAL_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _SPATIAL_TRUE_VALUES:
            dry_run = True
        elif normalized in _SPATIAL_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_SPATIAL_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_SPATIAL_TRUE_VALUES | _SPATIAL_FALSE_VALUES)}"
            )

    return SpatialConfig(
        auto_tag=auto_tag,
        default_wing=default_wing,
        dry_run=dry_run,
    )


_DMN_DEFAULT_REFLECTION_WINDOW_HOURS: int = 24
_DMN_DEFAULT_META_ANALYST_ENABLED: bool = True
_DMN_DEFAULT_DRY_RUN_PRODUCTION: bool = False

_DMN_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_DMN_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class DmnConfig:

    reflection_window_hours: int
    meta_analyst_enabled: bool
    dry_run: bool


def _load_dmn_config() -> DmnConfig:
    raw_window_hours = os.environ.get(
        "IAI_MCP_DMN_REFLECTION_WINDOW_HOURS",
        str(_DMN_DEFAULT_REFLECTION_WINDOW_HOURS),
    )
    try:
        reflection_window_hours = int(raw_window_hours)
    except (TypeError, ValueError):
        raise ValueError(
            f"IAI_MCP_DMN_REFLECTION_WINDOW_HOURS: invalid value "
            f"{raw_window_hours!r}, expected int"
        )
    if not (1 <= reflection_window_hours <= 720):
        raise ValueError(
            f"IAI_MCP_DMN_REFLECTION_WINDOW_HOURS: invalid value "
            f"{raw_window_hours!r}, expected int in [1, 720]"
        )

    raw_meta_enabled = os.environ.get("IAI_MCP_META_ANALYST_ENABLED")
    if raw_meta_enabled is None:
        meta_analyst_enabled = _DMN_DEFAULT_META_ANALYST_ENABLED
    else:
        normalized = raw_meta_enabled.strip().lower()
        if normalized in _DMN_TRUE_VALUES:
            meta_analyst_enabled = True
        elif normalized in _DMN_FALSE_VALUES:
            meta_analyst_enabled = False
        else:
            raise ValueError(
                f"IAI_MCP_META_ANALYST_ENABLED: invalid value "
                f"{raw_meta_enabled!r}, expected one of "
                f"{sorted(_DMN_TRUE_VALUES | _DMN_FALSE_VALUES)}"
            )

    raw_dry_run = os.environ.get("IAI_MCP_DMN_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _DMN_TRUE_VALUES:
            dry_run = True
        elif normalized in _DMN_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_DMN_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_DMN_TRUE_VALUES | _DMN_FALSE_VALUES)}"
            )

    return DmnConfig(
        reflection_window_hours=reflection_window_hours,
        meta_analyst_enabled=meta_analyst_enabled,
        dry_run=dry_run,
    )


_PASK_DEFAULT_ENABLED: bool = True
_PASK_DEFAULT_DRY_RUN_PRODUCTION: bool = False

_PASK_TRUE_VALUES: frozenset[str] = frozenset({"true", "1", "yes", "on"})
_PASK_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no", "off", ""})


@dataclass(frozen=True)
class PaskConfig:

    enabled: bool
    dry_run: bool


def _load_pask_config() -> PaskConfig:
    raw_enabled = os.environ.get("IAI_MCP_PASK_ENABLED")
    if raw_enabled is None:
        enabled = _PASK_DEFAULT_ENABLED
    else:
        normalized = raw_enabled.strip().lower()
        if normalized in _PASK_TRUE_VALUES:
            enabled = True
        elif normalized in _PASK_FALSE_VALUES:
            enabled = False
        else:
            raise ValueError(
                f"IAI_MCP_PASK_ENABLED: invalid value "
                f"{raw_enabled!r}, expected one of "
                f"{sorted(_PASK_TRUE_VALUES | _PASK_FALSE_VALUES)}"
            )

    raw_dry_run = os.environ.get("IAI_MCP_PASK_DRY_RUN")
    if raw_dry_run is None:
        dry_run = bool(os.environ.get("PYTEST_CURRENT_TEST"))
    else:
        normalized = raw_dry_run.strip().lower()
        if normalized in _PASK_TRUE_VALUES:
            dry_run = True
        elif normalized in _PASK_FALSE_VALUES:
            dry_run = False
        else:
            raise ValueError(
                f"IAI_MCP_PASK_DRY_RUN: invalid value "
                f"{raw_dry_run!r}, expected one of "
                f"{sorted(_PASK_TRUE_VALUES | _PASK_FALSE_VALUES)}"
            )

    return PaskConfig(
        enabled=enabled,
        dry_run=dry_run,
    )
