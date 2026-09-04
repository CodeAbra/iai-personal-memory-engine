"""`iai capture --directive` is the genuinely user-typed channel: the user
runs the CLI themselves, so no assistant judgment is in the loop. The
memory_capture RPC path is structurally incapable of minting a directive
(see core.dispatch), so the flag routes straight to the direct-store write
instead of round-tripping through the daemon -- daemon reachable or not.
The no-flag path is unaffected and still goes through the daemon RPC.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID

import pytest

from iai_mcp import iai_cli
from iai_mcp.store import MemoryStore


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _args(text: str, *, directive: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        text=text, session_id="s1", json=False, directive=directive,
    )


@pytest.fixture(autouse=True)
def _redirect_directive_cache(tmp_path, monkeypatch):
    """A directive capture refreshes the global directives cache
    synchronously from inside capture_turn, at a module-level default path
    bound once at import time from the real home directory -- a later HOME
    monkeypatch cannot retarget an already-bound Path constant. Redirect the
    writer itself so this file's --directive tests never touch the real
    ~/.iai-mcp/.directives.cached.md."""
    import iai_mcp.directive_cache as _directive_cache_mod

    real_write = _directive_cache_mod.write_directives_cache
    cache_path = tmp_path / "directive-cache" / ".directives.cached.md"

    def _redirected(store, **kwargs):
        kwargs.setdefault("cache_path", cache_path)
        return real_write(store, **kwargs)

    monkeypatch.setattr(_directive_cache_mod, "write_directives_cache", _redirected)


# --- 1. --directive NEVER touches the daemon RPC; it writes direct -------


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_bypasses_daemon_rpc_entirely(driver, tmp_path, monkeypatch, capsys):
    """--directive must never call the memory_capture RPC -- the RPC path
    cannot mint a directive, so the CLI writes straight to the store
    instead, even when a daemon is reachable."""
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

    rpc_calls: list[tuple] = []

    def _fake_send(method, params, **kwargs):
        rpc_calls.append((method, params))
        return {"result": {"id": "fake-record-id"}}

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)

    rc = iai_cli.cmd_capture(_args("standing directive: reply in English", directive=True))

    assert rc == 0
    assert rpc_calls == [], (
        f"--directive must not call the memory_capture RPC at all: {rpc_calls}"
    )

    store = MemoryStore(path=store_root)
    records = [
        rec for rec in store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")
    ]
    assert len(records) == 1, records
    rec = records[0]
    assert rec.directive is True
    stamps = [p.get("directive_source") for p in rec.provenance if isinstance(p, dict)]
    assert "explicit-command" in stamps, rec.provenance


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_daemon_path_no_flag_directive_absent(driver, monkeypatch, capsys):
    _select_driver(driver, monkeypatch)
    monkeypatch.delenv("IAI_MCP_STORE", raising=False)
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

    captured: dict = {}

    def _fake_send(method, params, **kwargs):
        captured["method"] = method
        captured["params"] = params
        return {"result": {"id": "fake-record-id"}}

    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)

    rc = iai_cli.cmd_capture(_args("just a plain note"))

    assert rc == 0
    assert "directive" not in captured["params"]


# --- 2. Daemon genuinely absent (no RPC mock): --directive still writes --
# ---    direct-store and works identically with no daemon running at all.


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_writes_direct_with_no_daemon_running(
    driver, tmp_path, monkeypatch, capsys
):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

    rc = iai_cli.cmd_capture(
        _args("standing directive: reply in English", directive=True)
    )
    assert rc == 0

    store = MemoryStore(path=store_root)
    records = [
        rec for rec in store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")
    ]
    assert len(records) == 1, records
    rec = records[0]
    assert rec.directive is True
    stamps = [p.get("directive_source") for p in rec.provenance if isinstance(p, dict)]
    assert "explicit-command" in stamps, rec.provenance


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_translates_capture_turn_failure_cleanly(
    driver, tmp_path, monkeypatch, capsys
):
    """A capture_turn exception (e.g. embed failure) on the directive path
    must exit 1 with a clean stderr line, never an unhandled traceback from
    an unbound `result`."""
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("capture encode failed: boom")

    monkeypatch.setattr("iai_mcp.capture.capture_turn", _boom)

    rc = iai_cli.cmd_capture(_args("standing directive: reply in English", directive=True))

    assert rc == 1
    err = capsys.readouterr().err
    assert "capture failed" in err
    assert "boom" in err


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_direct_store_fallback_no_flag_directive_false_no_stamp(
    driver, tmp_path, monkeypatch, capsys
):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)

    rc = iai_cli.cmd_capture(_args("just a plain note about alice"))
    assert rc == 0

    store = MemoryStore(path=store_root)
    live_directives = list(
        store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")
    )
    assert live_directives == []

    all_records = list(store.iter_records(where="tombstoned_at IS NULL"))
    assert len(all_records) == 1, all_records
    rec = all_records[0]
    assert rec.directive is False
    stamps = [p.get("directive_source") for p in rec.provenance if isinstance(p, dict)]
    assert "explicit-command" not in stamps
    assert "explicit-marker" not in stamps
