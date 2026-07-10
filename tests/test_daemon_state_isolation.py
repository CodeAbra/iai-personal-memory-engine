"""Contracts: the daemon's runtime state file and wake signal follow the
active store root, never the home store, when IAI_MCP_STORE is set.

An isolated daemon (spawned with its own IAI_MCP_STORE) must never read or
write the home store's runtime state file, and must never consume a wake
signal left for the home-store daemon. With the environment variable unset,
resolution must stay byte-identical to the pre-existing home-rooted
defaults — no behavior change for the default store.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_save_state_honors_store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.daemon_state import STATE_PATH, daemon_state_path, save_state

    home_state_path = STATE_PATH
    home_stat_before = None
    if home_state_path.exists():
        st = home_state_path.stat()
        home_stat_before = (st.st_mtime_ns, st.st_size)

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    save_state({"probe": 1})

    resolved = daemon_state_path()
    assert resolved == tmp_path / ".daemon-state.json"
    assert resolved.exists()
    assert json.loads(resolved.read_text()) == {"probe": 1}

    if home_stat_before is not None:
        st = home_state_path.stat()
        assert (st.st_mtime_ns, st.st_size) == home_stat_before, (
            "save_state under IAI_MCP_STORE must never touch the home store's state file"
        )
    else:
        assert not home_state_path.exists(), (
            "save_state under IAI_MCP_STORE must never create the home store's state file"
        )


def test_load_state_honors_store_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.daemon_state import load_state

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    target = tmp_path / ".daemon-state.json"
    target.write_text(json.dumps({"fsm_state": "WAKE", "marker": "isolated"}))

    loaded = load_state()
    assert loaded == {"fsm_state": "WAKE", "marker": "isolated"}


def test_default_paths_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    from iai_mcp.daemon_state import STATE_PATH, daemon_state_path
    from iai_mcp.wake_handler import wake_signal_path

    monkeypatch.delenv("IAI_MCP_STORE", raising=False)

    expected_state = Path.home() / ".iai-mcp" / ".daemon-state.json"
    expected_wake = Path.home() / ".iai-mcp" / "wake.signal"

    assert daemon_state_path() == expected_state
    assert daemon_state_path() == STATE_PATH
    assert wake_signal_path() == expected_wake


def test_wake_signal_path_honors_store_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp.wake_handler import WakeHandler, wake_signal_path

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))

    resolved = wake_signal_path()
    assert resolved == tmp_path / "wake.signal"

    resolved.write_text('{"requested_at":"2026-05-02T15:00:00Z"}')

    handler = WakeHandler(resolved)
    assert handler.consume_wake_signal() is True
    assert not resolved.exists()

    home_wake = Path.home() / ".iai-mcp" / "wake.signal"
    # The isolated resolve/consume cycle above must never have touched the
    # home store's wake signal file.
    assert wake_signal_path() == resolved.parent / "wake.signal"
    assert not (tmp_path / "wake.signal").exists()
    assert home_wake != resolved
