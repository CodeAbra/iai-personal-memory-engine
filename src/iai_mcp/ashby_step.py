
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TopologySnapshot:

    rich_club_ratio: float
    community_count: int
    edge_density: float
    total_nodes: int
    avg_degree: float
    giant_component_fraction: float


@dataclass(frozen=True)
class BreachInfo:

    variable_name: str
    observed_value: float
    threshold: float
    direction: Literal["floor_breach", "ceiling_breach"]


# The essential variables that may actually arm crisis mode. Average degree
# does not decay as the graph grows (wiring-economy principle: real networks
# hold roughly constant average degree at any scale), and giant-component
# fraction is the Molloy-Reed percolation criterion (<k^2>/<k> > 2) for
# whether the graph is still one reachable mass -- a shattered giant
# component is the one condition where recall spreading genuinely breaks.
# community_count's ratio check is already scale-invariant and keeps its
# existing gating role. rich_club_ratio and edge_density decay O(1/N) on any
# large sparse graph and are scale-blind, so they are demoted to
# telemetry-only: still checked and reported for observability, but never
# consulted to arm crisis.
CRISIS_GATING_VARIABLES: frozenset[str] = frozenset(
    {"avg_degree", "giant_component_fraction", "community_count"}
)

_SNAPSHOT_KEYS: tuple[str, ...] = (
    "rich_club_ratio",
    "community_count",
    "edge_density",
    "avg_degree",
    "giant_component_fraction",
)


class EssentialVariableTracker:

    def __init__(self, cfg) -> None:
        self._rich_club_floor: float = float(cfg.rich_club_ratio_floor)
        self._community_count_ceiling_ratio: float = float(
            cfg.community_count_ceiling_ratio
        )
        self._edge_density_floor: float = float(cfg.edge_density_floor)
        self._avg_degree_floor: float = float(cfg.avg_degree_floor)
        self._giant_component_fraction_floor: float = float(
            cfg.giant_component_fraction_floor
        )
        self._ev_min_nodes: int = int(cfg.ev_min_nodes)

    def check(self, snapshot: TopologySnapshot) -> dict[str, "BreachInfo | None"]:
        # Topology statistics are meaningless on tiny graphs -- below the
        # minimum scale, report all-clear so a fresh install can never
        # crisis-arm. This generalizes the prior total_nodes == 0 special
        # case (kept implicitly: 0 < ev_min_nodes always holds).
        if snapshot.total_nodes < self._ev_min_nodes:
            return {key: None for key in _SNAPSHOT_KEYS}

        result: dict[str, BreachInfo | None] = {}

        if snapshot.rich_club_ratio < self._rich_club_floor:
            result["rich_club_ratio"] = BreachInfo(
                variable_name="rich_club_ratio",
                observed_value=float(snapshot.rich_club_ratio),
                threshold=float(self._rich_club_floor),
                direction="floor_breach",
            )
        else:
            result["rich_club_ratio"] = None

        cc_ratio = snapshot.community_count / snapshot.total_nodes
        if cc_ratio > self._community_count_ceiling_ratio:
            result["community_count"] = BreachInfo(
                variable_name="community_count",
                observed_value=float(cc_ratio),
                threshold=float(self._community_count_ceiling_ratio),
                direction="ceiling_breach",
            )
        else:
            result["community_count"] = None

        if snapshot.edge_density < self._edge_density_floor:
            result["edge_density"] = BreachInfo(
                variable_name="edge_density",
                observed_value=float(snapshot.edge_density),
                threshold=float(self._edge_density_floor),
                direction="floor_breach",
            )
        else:
            result["edge_density"] = None

        if snapshot.avg_degree < self._avg_degree_floor:
            result["avg_degree"] = BreachInfo(
                variable_name="avg_degree",
                observed_value=float(snapshot.avg_degree),
                threshold=float(self._avg_degree_floor),
                direction="floor_breach",
            )
        else:
            result["avg_degree"] = None

        if snapshot.giant_component_fraction < self._giant_component_fraction_floor:
            result["giant_component_fraction"] = BreachInfo(
                variable_name="giant_component_fraction",
                observed_value=float(snapshot.giant_component_fraction),
                threshold=float(self._giant_component_fraction_floor),
                direction="floor_breach",
            )
        else:
            result["giant_component_fraction"] = None

        return result


def decide_crisis_transition(
    any_gating_breach: bool,
    consecutive_breaches: int,
    consecutive_clears: int,
    *,
    arm_after_n: int,
    disarm_after_n: int,
) -> tuple[bool | None, int, int]:
    """Pure hysteresis decision over consecutive gating-breach/clear cycles.

    Mirrors the watchdog debounce shape: a trigger only acts once sustained
    for N consecutive cycles, so a single blip never flips crisis state in
    either direction. No I/O, no imports beyond stdlib.

    Returns ``(action, new_consecutive_breaches, new_consecutive_clears)``
    where ``action`` is ``True`` to arm, ``False`` to disarm, or ``None`` if
    the state should remain unchanged this cycle.
    """
    if any_gating_breach:
        new_breaches = consecutive_breaches + 1
        new_clears = 0
        if new_breaches >= arm_after_n:
            return (True, new_breaches, new_clears)
        return (None, new_breaches, new_clears)

    new_clears = consecutive_clears + 1
    new_breaches = 0
    if new_clears >= disarm_after_n:
        return (False, new_breaches, new_clears)
    return (None, new_breaches, new_clears)
