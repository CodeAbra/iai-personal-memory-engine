"""`iai capture --directive` on a full directive budget must land the text
as an ordinary memory record and report that distinctly -- never "capture
failed" / exit 1, which would make the caller re-submit and duplicate the
ordinary-tier record.
"""
from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from iai_mcp import iai_cli
from iai_mcp.capture import capture_turn
from iai_mcp.directive_budget import DIRECTIVE_MAX_COUNT
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


def _args(text: str) -> argparse.Namespace:
    return argparse.Namespace(text=text, session_id="s1", json=False, directive=True)


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


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_budget_full_lands_as_ordinary_and_reports_distinctly(driver, tmp_path, monkeypatch):
    _select_driver(driver, monkeypatch)
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)

    seed_store = MemoryStore(path=store_root)
    try:
        for i in range(DIRECTIVE_MAX_COUNT):
            result = capture_turn(
                store=seed_store, cue="c",
                text=f"standing order number {i}: keep answers under five sentences",
                directive=True, session_id="s1", role="user",
            )
            assert result["status"] == "inserted", result
    finally:
        seed_store.close()

    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = iai_cli.cmd_capture(_args("the eleventh standing order that overflows the count cap"))

    assert rc == 0
    out = buf.getvalue().lower()
    assert "capture failed" not in out
    assert "captured as an ordinary memory record" in out
    assert "not a directive" in out
    assert "count cap" in out

    check_store = MemoryStore(path=store_root)
    try:
        live_directives = list(
            check_store.iter_records(where="directive = 1 AND tombstoned_at IS NULL")
        )
        assert len(live_directives) == DIRECTIVE_MAX_COUNT

        ordinary = [
            rec
            for rec in check_store.iter_records(where="tombstoned_at IS NULL")
            if rec.literal_surface == "the eleventh standing order that overflows the count cap"
        ]
        assert len(ordinary) == 1
        assert ordinary[0].directive is False
    finally:
        check_store.close()
