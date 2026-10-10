"""`iai capture --directive` requires an anchored 'standing directive:'
marker even when the flag is passed -- the flag alone must never be
sufficient to mint a directive from arbitrary text. The isatty() refusal
stays the first guard; the marker requirement is the second.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from iai_mcp import iai_cli
from iai_mcp.store import MemoryStore


@pytest.fixture(autouse=True)
def _simulate_tty(monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True, raising=False)


@pytest.fixture(autouse=True)
def _redirect_directive_cache(tmp_path, monkeypatch):
    import iai_mcp.directive_cache as _directive_cache_mod

    real_write = _directive_cache_mod.write_directives_cache
    cache_path = tmp_path / "directive-cache" / ".directives.cached.md"

    def _redirected(store, **kwargs):
        kwargs.setdefault("cache_path", cache_path)
        return real_write(store, **kwargs)

    monkeypatch.setattr(_directive_cache_mod, "write_directives_cache", _redirected)


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401
        except ImportError:
            pytest.skip("iai_mcp_native not built — lilli driver unavailable in this env")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)


def _args(text: str, *, directive: bool = True) -> argparse.Namespace:
    return argparse.Namespace(
        text=text, session_id="s1", json=False, directive=directive,
    )


def _store_root(tmp_path: Path, monkeypatch) -> Path:
    store_root = tmp_path / "store"
    store_root.mkdir()
    monkeypatch.setenv("IAI_MCP_STORE", str(store_root))
    monkeypatch.delenv("IAI_DAEMON_SOCKET_PATH", raising=False)
    return store_root


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_refuses_arbitrary_non_marker_text(driver, tmp_path, monkeypatch, capsys):
    _select_driver(driver, monkeypatch)
    store_root = _store_root(tmp_path, monkeypatch)

    rc = iai_cli.cmd_capture(_args("reply in English from now on"))

    assert rc == 2
    err = capsys.readouterr().err
    assert "standing directive:" in err

    store = MemoryStore(path=store_root)
    assert list(store.iter_records()) == [], (
        "a refused --directive call must mint no record at all"
    )


@pytest.mark.parametrize("driver", ["stdlib", "lilli"])
def test_directive_flag_with_anchored_marker_still_mints(driver, tmp_path, monkeypatch, capsys):
    _select_driver(driver, monkeypatch)
    store_root = _store_root(tmp_path, monkeypatch)

    rc = iai_cli.cmd_capture(_args("standing directive: reply in English"))

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


def test_isatty_guard_still_precedes_the_marker_guard(monkeypatch, capsys):
    """A non-tty stdin must still refuse at the isatty stage -- with its own
    message -- even when the text carries the anchored marker."""
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False, raising=False)

    rc = iai_cli.cmd_capture(_args("standing directive: reply in English"))

    assert rc == 2
    err = capsys.readouterr().err
    assert "interactive terminal" in err
