"""The status surface must never trigger the deep topology compute, and the
deep compute must be cached + single-flight.

A status poller once queued one full topology snapshot per probe: each deep
compute walks the whole graph (clustering + APSL + sigma baselines + community
detection), the client timed out long before the answer, and the abandoned
dispatches stacked into an hours-long multi-core burn. These tests pin the
three defenses: a zero-graph-work status RPC, a TTL snapshot cache, and a
non-stacking single-flight guard around the deep compute.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _reset_topology_cache():
    from iai_mcp import core

    with core._topology_state_lock:
        core._topology_cache = None
        core._topology_cache_at = 0.0
        core._topology_cache_key = None


def _warm_topology_cache_for(store, snapshot) -> None:
    from iai_mcp import core

    with core._topology_state_lock:
        core._topology_cache = snapshot
        core._topology_cache_at = time.monotonic()
        core._topology_cache_key = core._topology_store_key(store)


def _fake_snapshot(n: int = 42) -> dict:
    return {
        "N": n, "C": 0.5, "L": 2.0, "sigma": 1.3,
        "community_count": 3, "rich_club_ratio": 0.1,
        "regime": "healthy",
    }


def test_status_light_does_zero_graph_work():
    from iai_mcp import core

    _reset_topology_cache()
    store = MagicMock()
    store.active_records_count.return_value = 123

    with patch.object(core.retrieve, "build_runtime_graph") as build:
        result = core.dispatch(store, "status_light", {})
        build.assert_not_called()

    assert result["N"] == 123
    assert result["regime"] == "unknown"
    assert result["topology_cached"] is False


def test_status_light_serves_cached_regime():
    from iai_mcp import core

    _reset_topology_cache()
    store = MagicMock()
    store.active_records_count.return_value = 42
    _warm_topology_cache_for(store, _fake_snapshot())

    result = core.dispatch(store, "status_light", {})
    assert result["regime"] == "healthy"
    assert result["sigma"] == 1.3
    assert result["topology_cached"] is True
    _reset_topology_cache()


def test_topology_serves_fresh_cache_without_recompute():
    from iai_mcp import core

    _reset_topology_cache()
    store = MagicMock()
    store.active_records_count.return_value = 42

    _warm_topology_cache_for(store, _fake_snapshot())

    with patch.object(core.retrieve, "build_runtime_graph") as build:
        result = core.dispatch(store, "topology", {})
        build.assert_not_called()
    assert result["regime"] == "healthy"
    _reset_topology_cache()


def test_topology_computes_once_and_caches():
    from iai_mcp import core
    from iai_mcp import sigma as sigma_mod

    _reset_topology_cache()
    store = MagicMock()
    store.active_records_count.return_value = 42
    graph = MagicMock()

    with patch.object(
        core.retrieve, "build_runtime_graph", return_value=graph
    ) as build, patch.object(
        sigma_mod, "compute_topology_snapshot", return_value=_fake_snapshot()
    ) as compute:
        first = core.dispatch(store, "topology", {})
        second = core.dispatch(store, "topology", {})
        assert build.call_count == 1, "second call must hit the cache"
        assert compute.call_count == 1
    assert first["regime"] == second["regime"] == "healthy"
    _reset_topology_cache()


def test_topology_concurrent_caller_does_not_stack():
    """While one deep compute runs, a second caller gets a shape-stable
    placeholder immediately instead of queueing another compute."""
    from iai_mcp import core
    from iai_mcp import sigma as sigma_mod

    _reset_topology_cache()
    store = MagicMock()
    store.active_records_count.return_value = 42
    graph = MagicMock()

    entered = threading.Event()
    release = threading.Event()

    def slow_compute(_graph, assignment=None):
        entered.set()
        release.wait(timeout=10)
        return _fake_snapshot()

    results: dict[str, dict] = {}

    with patch.object(
        core.retrieve, "build_runtime_graph", return_value=graph
    ), patch.object(
        sigma_mod, "compute_topology_snapshot", side_effect=slow_compute
    ):
        t = threading.Thread(
            target=lambda: results.__setitem__(
                "deep", core.dispatch(store, "topology", {})
            )
        )
        t.start()
        assert entered.wait(timeout=5), "deep compute never started"
        t0 = time.monotonic()
        overlapped = core.dispatch(store, "topology", {})
        elapsed = time.monotonic() - t0
        release.set()
        t.join(timeout=10)

    assert elapsed < 1.0, "overlapping caller must not wait for the compute"
    assert overlapped["regime"] == "computing"
    assert overlapped["N"] == 42
    assert results["deep"]["regime"] == "healthy"
    _reset_topology_cache()


def test_iai_status_uses_status_light_not_topology():
    import iai_mcp.iai_cli as iai_cli

    calls: list[str] = []

    def fake_rpc(method, params):
        calls.append(method)
        return {"result": {"N": 7, "regime": "healthy"}}

    with patch.object(
        iai_cli, "_send_jsonrpc_request", create=True
    ):
        pass  # module imports the symbol inside cmd_status; patch at source

    with patch(
        "iai_mcp.cli._send_jsonrpc_request", side_effect=fake_rpc
    ), patch(
        "iai_mcp.claude_cli.verify_credentials_subscription",
        return_value={"ok": True, "subscription_type": "max"},
    ):
        rc = iai_cli.cmd_status(SimpleNamespace())

    assert rc == 0
    assert calls == ["status_light"], (
        f"iai status must call status_light only, called {calls}"
    )


def _ring_csr(n: int) -> tuple[np.ndarray, np.ndarray]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u in range(n):
        v = (u + 1) % n
        adj[u].append(v)
        adj[v].append(u)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices: list[int] = []
    for u in range(n):
        adj[u].sort()
        indices.extend(adj[u])
        indptr[u + 1] = indptr[u] + len(adj[u])
    return indptr, np.array(indices, dtype=np.int64)


def test_sampled_apsl_full_sources_equals_exact():
    from iai_mcp_native import graph as lilli_graph

    indptr, indices = _ring_csr(300)
    exact = lilli_graph.average_shortest_path_length(indptr, indices, 300)
    sampled = lilli_graph.average_shortest_path_length_sampled(
        indptr, indices, 300, 300
    )
    assert abs(exact - sampled) < 1e-9


def test_apsl_helper_routes_by_component_size():
    from iai_mcp import sigma as sigma_mod

    calls: list[str] = []
    fake = SimpleNamespace(
        average_shortest_path_length=lambda *a: calls.append("exact") or 1.0,
    )
    indptr = np.zeros(2, dtype=np.int64)
    indices = np.zeros(0, dtype=np.int64)
    n_small = sigma_mod.SIGMA_APSL_EXACT_MAX_N
    n_large = sigma_mod.SIGMA_APSL_EXACT_MAX_N + 1
    indptr_large = np.zeros(n_large + 1, dtype=np.int64)
    with patch.object(sigma_mod, "lilli_graph", fake):
        sigma_mod._average_shortest_path_length(indptr, indices, n_small, seed=1)
        # Above the exact ceiling the sampled scipy path runs — the exact
        # kernel must NOT be called again.
        sigma_mod._average_shortest_path_length(
            indptr_large, indices, n_large, seed=1
        )
    assert calls == ["exact"]


def test_status_light_never_serves_foreign_store_cache():
    from iai_mcp import core

    _reset_topology_cache()
    owner = MagicMock()
    owner.active_records_count.return_value = 42
    _warm_topology_cache_for(owner, _fake_snapshot())

    other = MagicMock()
    other.active_records_count.return_value = 7
    result = core.dispatch(other, "status_light", {})
    assert result["topology_cached"] is False
    assert result["regime"] == "unknown"
    _reset_topology_cache()
