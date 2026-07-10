from __future__ import annotations

import inspect
import math
from dataclasses import FrozenInstanceError
from uuid import uuid4

import numpy as np
import pytest
import scipy.sparse

from iai_mcp.graph import MemoryGraph


def _emb(seed: int, dim: int = 384) -> list[float]:
    rng = np.random.default_rng(seed)
    return rng.random(dim).tolist()


def test_module_imports() -> None:
    from iai_mcp.mosaic import (
        EPSILON,
        WALL_TIME_HARD_CAP_S,
        build_csr_sanitized,
        run_mosaic,
    )

    assert callable(run_mosaic)
    assert callable(build_csr_sanitized)
    assert isinstance(EPSILON, float)
    assert isinstance(WALL_TIME_HARD_CAP_S, float)


def test_lineage_types_importable() -> None:
    from iai_mcp.mosaic_lineage import (
        LineageEvent,
        LineageReport,
        LineageTracker,
    )

    assert LineageEvent is not None
    assert LineageReport is not None
    assert LineageTracker is not None


def test_policy_module_importable() -> None:
    from iai_mcp.mosaic_policy import should_fall_back_to_flat

    assert callable(should_fall_back_to_flat)


def test_csr_canonical_node_order() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    uuids = [uuid4() for _ in range(5)]
    for i, u in enumerate(uuids):
        g.add_node(u, community_id=None, embedding=_emb(i))

    _csr, order, idx_map = build_csr_sanitized(g)
    assert order == sorted(uuids, key=str)
    assert list(idx_map.keys()) == sorted(uuids, key=str)
    for i, u in enumerate(order):
        assert idx_map[u] == i


def test_csr_canonical_edge_order() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_node(c, community_id=None, embedding=_emb(2))
    g.add_edge(a, b, weight=2.0)
    g.add_edge(b, c, weight=3.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.nnz == 4
    csr.sort_indices()
    assert (csr != csr).nnz == 0


def test_csr_strips_nan_weights() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_node(c, community_id=None, embedding=_emb(2))
    g.add_edge(a, b, weight=float("nan"))
    g.add_edge(b, c, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.nnz == 2
    assert all(math.isfinite(float(x)) for x in csr.data)


def test_csr_strips_inf_weights() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c, d = uuid4(), uuid4(), uuid4(), uuid4()
    for i, u in enumerate([a, b, c, d]):
        g.add_node(u, community_id=None, embedding=_emb(i))
    g.add_edge(a, b, weight=float("inf"))
    g.add_edge(b, c, weight=float("-inf"))
    g.add_edge(c, d, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.nnz == 2
    assert all(math.isfinite(float(x)) for x in csr.data)


def test_csr_strips_negative_weights() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_node(c, community_id=None, embedding=_emb(2))
    g.add_edge(a, b, weight=-0.5)
    g.add_edge(b, c, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.nnz == 2
    assert all(float(x) >= 0.0 for x in csr.data)


def test_csr_strips_self_loops() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b = uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_edge(a, a, weight=2.0)
    g.add_edge(a, b, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.nnz == 2
    assert csr.diagonal().sum() == 0.0


def test_csr_dtype_is_float64() -> None:
    from iai_mcp.mosaic import build_csr_sanitized

    g = MemoryGraph()
    a, b = uuid4(), uuid4()
    g.add_node(a, community_id=None, embedding=_emb(0))
    g.add_node(b, community_id=None, embedding=_emb(1))
    g.add_edge(a, b, weight=1.0)

    csr, _order, _idx_map = build_csr_sanitized(g)
    assert csr.data.dtype == np.float64
    assert isinstance(csr, scipy.sparse.csr_matrix)


def test_lineage_event_is_frozen() -> None:
    from datetime import datetime, timezone

    from iai_mcp.mosaic_lineage import LineageEvent

    event = LineageEvent(
        event_type="birth",
        timestamp=datetime.now(timezone.utc),
        parent_uuid=None,
        child_uuids=(uuid4(),),
        member_count=1,
    )
    with pytest.raises(FrozenInstanceError):
        event.member_count = 999  # type: ignore[misc]


def test_lineage_tracker_empty_report_has_zero_events() -> None:
    from iai_mcp.mosaic_lineage import LineageReport, LineageTracker

    tracker = LineageTracker()
    report = tracker.report()
    assert isinstance(report, LineageReport)
    assert report.events == ()


def test_run_mosaic_signature() -> None:
    from iai_mcp.mosaic import run_mosaic

    sig = inspect.signature(run_mosaic)
    params = sig.parameters
    assert set(params.keys()) == {
        "graph",
        "prior",
        "prior_mode",
        "gamma",
        "seed",
        "max_levels",
    }
    assert params["prior"].default is None
    assert params["prior_mode"].default == "seeded"
    assert params["gamma"].default is None
    assert params["seed"].default == 42
    assert params["max_levels"].default == 5


def test_wall_time_constant_is_30s() -> None:
    from iai_mcp.mosaic import WALL_TIME_HARD_CAP_S

    assert WALL_TIME_HARD_CAP_S == 30.0


def test_run_mosaic_returns_assignment_on_nonempty_graph() -> None:
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.mosaic import run_mosaic
    from iai_mcp.mosaic_lineage import LineageReport

    g = MemoryGraph()
    a, b, c = uuid4(), uuid4(), uuid4()
    for i, u in enumerate([a, b, c]):
        g.add_node(u, community_id=None, embedding=_emb(i))
    g.add_edge(a, b, weight=1.0)
    g.add_edge(b, c, weight=1.0)

    assignment, report = run_mosaic(
        g, prior=None, prior_mode="cold", gamma=1.0, seed=42
    )
    assert isinstance(assignment, CommunityAssignment)
    assert assignment.backend == "leiden-custom"
    assert set(assignment.node_to_community.keys()) == {a, b, c}
    assert isinstance(report, LineageReport)


def test_empty_graph_returns_empty_assignment() -> None:
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.mosaic import run_mosaic
    from iai_mcp.mosaic_lineage import LineageReport

    g = MemoryGraph()
    assignment, report = run_mosaic(g, prior=None)
    assert isinstance(assignment, CommunityAssignment)
    assert assignment.backend == "flat"
    assert assignment.node_to_community == {}
    assert isinstance(report, LineageReport)
    assert report.events == ()
