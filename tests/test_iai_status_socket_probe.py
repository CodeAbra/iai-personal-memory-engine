from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout



def _make_args():
    import argparse
    return argparse.Namespace()


def _noop_subscription(monkeypatch):
    monkeypatch.setattr(
        "iai_mcp.claude_cli.verify_credentials_subscription",
        lambda: {"ok": True, "subscription_type": "active"},
    )


def test_status_daemon_up_via_socket(monkeypatch):
    from iai_mcp import iai_cli

    fake_resp = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "N": 4321,
            "C": 0.25,
            "L": 4.1,
            "sigma": 1.8,
            "community_count": 3,
            "rich_club_ratio": 0.12,
            "regime": "healthy",
        },
    }
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)
    _noop_subscription(monkeypatch)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_status(_make_args())

    assert rc == 0
    out = buf.getvalue()
    assert "UP" in out, f"expected UP in output, got: {out!r}"
    assert "4321" in out, f"expected record count 4321 in output, got: {out!r}"
    assert "healthy" in out, f"expected regime 'healthy' in output, got: {out!r}"
    assert "DOWN" not in out, f"DOWN must not appear when daemon is up, got: {out!r}"


def test_status_daemon_down_graceful(monkeypatch):
    from iai_mcp import iai_cli

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: None)
    _noop_subscription(monkeypatch)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_status(_make_args())

    assert rc == 0, f"expected exit code 0, got {rc}"
    out = buf.getvalue()
    assert "DOWN" in out, f"expected DOWN in output, got: {out!r}"
    assert "?" in out, f"expected ? placeholders for records/regime when DOWN, got: {out!r}"
    assert "UP" not in out, f"UP must not appear when daemon is down, got: {out!r}"
