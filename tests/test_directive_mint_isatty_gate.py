"""`iai capture --directive` refuses a non-interactive caller before any
store open or write -- the invoker (a model on the Bash tool) controls
every CLI flag it types, so the gate has no self-suppliable override.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def _args(text: str, *, directive: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        text=text, session_id="s1", json=False, directive=directive,
    )


@pytest.fixture(autouse=True)
def _redirect_directive_cache(tmp_path, monkeypatch):
    """Same redirect as test_directive_cli_channel.py -- the cache write
    is bound at import time from the real home directory."""
    import iai_mcp.directive_cache as _directive_cache_mod

    real_write = _directive_cache_mod.write_directives_cache
    cache_path = tmp_path / "directive-cache" / ".directives.cached.md"

    def _redirected(store, **kwargs):
        kwargs.setdefault("cache_path", cache_path)
        return real_write(store, **kwargs)

    monkeypatch.setattr(_directive_cache_mod, "write_directives_cache", _redirected)


def _live_directive_count(store_root: Path) -> int:
    s = MemoryStore(path=store_root)
    try:
        return len(list(s.iter_records(where="directive = 1 AND tombstoned_at IS NULL")))
    finally:
        s.close()


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_non_tty_mint_refuses_and_writes_nothing(driver, tmp_path, monkeypatch, capsys):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))

    before = _live_directive_count(store_root)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    rc = iai_cli.cmd_capture(_args("standing directive: reply in English"))

    assert rc == 2
    err = capsys.readouterr().err
    assert "interactive terminal" in err

    after = _live_directive_count(store_root)
    assert after == before == 0


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_tty_mint_still_works(driver, tmp_path, monkeypatch, capsys):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    rc = iai_cli.cmd_capture(_args("standing directive: reply in English"))

    assert rc == 0
    assert _live_directive_count(store_root) == 1


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_no_override_flag_defeats_non_tty_refusal(driver, tmp_path, monkeypatch, capsys):
    """A hypothetical yes/force attribute on args must not be read at all --
    the handler must not read such a flag."""
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    ns = _args("standing directive: reply in English")
    ns.yes = True
    ns.force = True

    rc = iai_cli.cmd_capture(ns)

    assert rc == 2
    assert _live_directive_count(store_root) == 0
