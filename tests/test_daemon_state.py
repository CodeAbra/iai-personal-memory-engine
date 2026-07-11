from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest


@pytest.fixture
def isolated_state_path(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


def test_save_and_load_roundtrip_with_0600_mode(isolated_state_path):
    from iai_mcp.daemon_state import load_state, save_state

    assert load_state() == {}

    state = {
        "fsm_state": "WAKE",
        "daemon_started_at": "2026-04-18T00:00:00+00:00",
        "pending_digest": {"cycles": 4, "insight": "test"},
    }
    save_state(state)

    assert isolated_state_path.exists()
    mode = isolated_state_path.stat().st_mode & 0o777
    assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    loaded = load_state()
    assert loaded == state


def test_save_state_atomic_rename_preserves_old_on_failure(isolated_state_path, monkeypatch):
    from iai_mcp.daemon_state import load_state, save_state

    original = {"fsm_state": "WAKE", "version": 1}
    save_state(original)
    assert load_state() == original

    import iai_mcp.daemon_state as ds
    real_replace = os.replace

    def _boom(src, dst):
        raise OSError("simulated swap failure")

    monkeypatch.setattr(ds.os, "replace", _boom)

    with pytest.raises(OSError):
        save_state({"fsm_state": "SLEEP", "version": 2})

    loaded = load_state()
    assert loaded == original

    leftovers = list(isolated_state_path.parent.glob(".daemon-state.*.tmp"))
    assert leftovers == [], f"temp files not cleaned: {leftovers}"


def test_pending_digest_returned_after_18h(isolated_state_path):
    from iai_mcp.daemon_state import (
        DIGEST_SHOW_THRESHOLD_HOURS,
        get_pending_digest,
        load_state,
        save_state,
    )
    assert DIGEST_SHOW_THRESHOLD_HOURS == 18

    now = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    last_shown = now - timedelta(hours=20)
    state = {
        "last_digest_shown_at": last_shown.isoformat(),
        "pending_digest": {"cycles": 4, "insight": "after-threshold"},
    }
    save_state(state)

    digest = get_pending_digest(state, now)
    assert digest == {"cycles": 4, "insight": "after-threshold"}

    assert "pending_digest" not in state
    assert state["last_digest_shown_at"] == now.isoformat()

    on_disk = load_state()
    assert "pending_digest" not in on_disk
    assert on_disk["last_digest_shown_at"] == now.isoformat()


def test_pending_digest_withheld_before_18h(isolated_state_path):
    from iai_mcp.daemon_state import get_pending_digest

    now = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    last_shown = now - timedelta(hours=4)
    state = {
        "last_digest_shown_at": last_shown.isoformat(),
        "pending_digest": {"cycles": 4, "insight": "too-early"},
    }
    digest = get_pending_digest(state, now)
    assert digest is None

    assert state["pending_digest"] == {"cycles": 4, "insight": "too-early"}
    assert state["last_digest_shown_at"] == last_shown.isoformat()


def test_pending_digest_none_when_not_set(isolated_state_path):
    from iai_mcp.daemon_state import get_pending_digest

    now = datetime(2026, 4, 18, 20, 0, tzinfo=timezone.utc)
    state: dict = {}
    assert get_pending_digest(state, now) is None


def test_prune_evicts_legacy_bool_first_turn_pending():
    from iai_mcp.daemon_state import prune_stale_first_turn

    state = {"first_turn_pending": {"sess-1": True, "sess-2": False, "sess-3": True}}
    removed = prune_stale_first_turn(state)

    assert removed == 3
    assert state["first_turn_pending"] == {}


def test_prune_keeps_fresh_iso_entries_and_evicts_aged():
    from iai_mcp.daemon_state import prune_stale_first_turn

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    fresh = (now - timedelta(hours=1)).isoformat()
    stale = (now - timedelta(hours=48)).isoformat()
    state = {"first_turn_pending": {"fresh": fresh, "stale": stale}}

    removed = prune_stale_first_turn(state, now=now, ttl_hours=24)

    assert removed == 1
    assert "fresh" in state["first_turn_pending"]
    assert "stale" not in state["first_turn_pending"]


def test_prune_caps_max_entries_keeps_newest():
    from iai_mcp.daemon_state import prune_stale_first_turn

    now = datetime(2026, 4, 23, 12, 0, tzinfo=timezone.utc)
    pending = {f"sess-{i}": (now - timedelta(minutes=i)).isoformat() for i in range(10)}
    state = {"first_turn_pending": pending}

    removed = prune_stale_first_turn(state, now=now, ttl_hours=24, max_entries=3)

    assert removed == 7
    kept = state["first_turn_pending"]
    assert len(kept) == 3
    assert set(kept.keys()) == {"sess-0", "sess-1", "sess-2"}


def test_prune_handles_empty_and_missing_pending():
    from iai_mcp.daemon_state import prune_stale_first_turn

    assert prune_stale_first_turn({}) == 0
    assert prune_stale_first_turn({"first_turn_pending": {}}) == 0
    assert prune_stale_first_turn({"first_turn_pending": None}) == 0
