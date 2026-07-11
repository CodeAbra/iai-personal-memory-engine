from __future__ import annotations

import argparse



def test_empty_store_yields_empty_stdout_exit_zero(tmp_path, monkeypatch, capsys):
    from iai_mcp import cli as cli_mod
    from iai_mcp.core import dispatch
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)

    def _stub(method, params, **_kw):
        return {"jsonrpc": "2.0", "id": 1, "result": dispatch(store, method, params)}

    monkeypatch.setattr(cli_mod, "_send_jsonrpc_request", _stub)

    rc = cli_mod.cmd_session_start(argparse.Namespace(session_id="-"))

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == ""
