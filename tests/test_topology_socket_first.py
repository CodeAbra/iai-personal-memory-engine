from __future__ import annotations

import argparse
import io
from contextlib import redirect_stdout
from unittest.mock import MagicMock, patch

def _args() -> argparse.Namespace:
    return argparse.Namespace()

def _topology_rpc_response() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "N": 9999,
            "C": 0.3812,
            "L": 3.2100,
            "sigma": 2.1,
            "community_count": 5,
            "rich_club_ratio": 0.17,
            "regime": "healthy",
        },
    }

def test_topology_socket_renders_when_daemon_up():
    from iai_mcp.cli import cmd_topology

    fake_resp = _topology_rpc_response()

    sentinel = MagicMock(side_effect=AssertionError("MemoryStore must not be called via socket path"))

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=fake_resp),
        patch("iai_mcp.store.MemoryStore", sentinel),
        redirect_stdout(buf),
    ):
        rc = cmd_topology(_args())

    assert rc == 0, f"expected rc=0, got {rc}"
    out = buf.getvalue()
    assert "N: 9999" in out, f"N line missing in: {out!r}"
    assert "regime: healthy" in out, f"regime line missing in: {out!r}"
    assert "C: 0.3812" in out, f"C line missing in: {out!r}"
    assert "communities: 5" in out, f"communities line missing in: {out!r}"
    assert "sigma: 2.1000" in out, f"sigma line missing in: {out!r}"

def test_topology_socket_does_not_open_memorystore():
    from iai_mcp.cli import cmd_topology

    called = []

    class _Sentinel:
        def __init__(self, *a, **k):
            called.append(True)

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=_topology_rpc_response()),
        patch("iai_mcp.store.MemoryStore", _Sentinel),
        redirect_stdout(buf),
    ):
        cmd_topology(_args())

    assert not called, "MemoryStore was instantiated on the socket path — lock contention bug"

def test_topology_fallback_when_socket_none():
    from iai_mcp.cli import cmd_topology

    fake_snap = {
        "N": 42,
        "C": 0.5,
        "L": 2.1,
        "sigma": 1.5,
        "community_count": 3,
        "rich_club_ratio": 0.09,
        "regime": "developmental",
    }
    fake_graph = MagicMock()
    fake_assignment = object()
    fake_rich_club = [object()]
    fake_store = MagicMock()
    captured: dict = {}

    def _fake_snapshot(graph, *, assignment=None, rich_club=None):
        captured.update(
            {
                "graph": graph,
                "assignment": assignment,
                "rich_club": rich_club,
            }
        )
        return fake_snap

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=None),
        patch("iai_mcp.store.MemoryStore", return_value=fake_store),
        patch(
            "iai_mcp.retrieve.build_runtime_graph",
            return_value=(fake_graph, fake_assignment, fake_rich_club),
        ),
        patch("iai_mcp.sigma.compute_topology_snapshot", side_effect=_fake_snapshot),
        redirect_stdout(buf),
    ):
        rc = cmd_topology(_args())

    assert rc == 0, f"expected rc=0, got {rc}"
    assert captured == {
        "graph": fake_graph,
        "assignment": fake_assignment,
        "rich_club": fake_rich_club,
    }
    out = buf.getvalue()
    assert "N: 42" in out, f"N line missing in: {out!r}"
    assert "regime: developmental" in out, f"regime line missing in: {out!r}"

def test_topology_degrades_on_hippo_lock_held():
    from iai_mcp.cli import cmd_topology
    from iai_mcp.hippo import HippoLockHeldError

    buf = io.StringIO()
    with (
        patch("iai_mcp.cli._send_jsonrpc_request", return_value=None),
        patch("iai_mcp.store.MemoryStore", side_effect=HippoLockHeldError("test.lock", "test")),
        redirect_stdout(buf),
    ):
        rc = cmd_topology(_args())

    assert rc == 0, f"expected rc=0 on HippoLockHeldError, got {rc}"
    out = buf.getvalue()
    assert "N: insufficient_data" in out, f"N line missing in degraded output: {out!r}"
    assert "regime: insufficient_data" in out, f"regime line missing in degraded output: {out!r}"
