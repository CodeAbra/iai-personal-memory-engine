from __future__ import annotations

import io
import sys
from contextlib import redirect_stdout, redirect_stderr



def _make_args(text: str = "hello world", session_id: str | None = None):
    import argparse
    ns = argparse.Namespace()
    ns.text = text
    ns.session_id = session_id
    return ns


def test_capture_success_prints_record_id(monkeypatch):
    from iai_mcp import iai_cli

    fake_resp = {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {"id": "fed6d868-b1ba-401e-9e57-5e6b44679d44"},
    }
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_capture(_make_args(text="the export format is JSONL"))

    assert rc == 0
    out = buf.getvalue()
    assert "captured" in out.lower()
    assert "fed6d868" in out


def test_capture_daemon_down_returns_nonzero_with_message(monkeypatch):
    from iai_mcp import iai_cli

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: None)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_capture(_make_args())

    assert rc == 1
    msg = err.getvalue().lower()
    assert "daemon" in msg
    assert "iai-mcp daemon" in msg or "iai-mcp daemon start" in msg


def test_capture_daemon_error_response_surfaced(monkeypatch):
    from iai_mcp import iai_cli

    fake_resp = {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32602, "message": "invalid params: empty text"},
    }
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", lambda *a, **k: fake_resp)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    err = io.StringIO()
    with redirect_stderr(err):
        rc = iai_cli.cmd_capture(_make_args(text=""))

    assert rc == 1
    assert "invalid params" in err.getvalue()


def test_capture_passes_session_id_through(monkeypatch):
    from iai_mcp import iai_cli

    captured_params: dict = {}

    def _fake_send(method, params, **kwargs):
        captured_params["method"] = method
        captured_params["params"] = dict(params)
        return {"jsonrpc": "2.0", "id": 1, "result": {"id": "abc-123"}}

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_capture(_make_args(text="hi", session_id="my-shell-session"))

    assert rc == 0
    assert captured_params["method"] == "memory_capture"
    assert captured_params["params"]["session_id"] == "my-shell-session"
    assert captured_params["params"]["text"] == "hi"
    assert captured_params["params"]["tier"] == "episodic"


def test_capture_default_session_id_is_dash(monkeypatch):
    from iai_mcp import iai_cli

    captured_params: dict = {}

    def _fake_send(method, params, **kwargs):
        captured_params["params"] = dict(params)
        return {"jsonrpc": "2.0", "id": 1, "result": {"id": "abc"}}

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)

    buf = io.StringIO()
    with redirect_stdout(buf):
        iai_cli.cmd_capture(_make_args(text="test", session_id=None))

    assert captured_params["params"]["session_id"] == "-"
