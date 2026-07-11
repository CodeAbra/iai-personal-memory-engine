from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


_EMPTY_DIGEST_EXPECTED = {
    "rem_cycles_completed": 0,
    "episodes_processed": 0,
    "schemas_induced_tier0": 0,
    "claude_call_used": False,
    "quota_used_pct": 0.0,
    "main_insight_text": None,
    "sigma_observed": None,
    "s5_drift_alerts": [],
    "daemon_uptime_hours": 0,
    "timed_out_cycles": 0,
}


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    from iai_mcp import daemon_state
    state_path = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", state_path)
    return state_path


_FULL_DIGEST = {
    "rem_cycles_completed": 4,
    "episodes_processed": 10,
    "schemas_induced_tier0": 3,
    "claude_call_used": True,
    "quota_used_pct": 0.003,
    "main_insight_text": "today's unifying insight",
    "sigma_observed": 1.2,
    "s5_drift_alerts": [],
    "daemon_uptime_hours": 8,
    "timed_out_cycles": 0,
}


def test_first_recall_gets_digest(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    now = datetime.now(timezone.utc)
    save_state({
        "pending_digest": dict(_FULL_DIGEST),
        "last_digest_shown_at": (now - timedelta(hours=20)).isoformat(),
    })

    response: dict = {"hits": [], "anti_hits": [], "activation_trace": [], "budget_used": 0}
    _inject_overnight_digest(response)

    assert "overnight_digest" in response
    dig = response["overnight_digest"]
    assert dig["rem_cycles_completed"] == 4
    assert dig["episodes_processed"] == 10
    assert dig["schemas_induced_tier0"] == 3
    assert dig["claude_call_used"] is True
    assert dig["quota_used_pct"] == 0.003
    assert dig["main_insight_text"] == "today's unifying insight"
    assert dig["sigma_observed"] == 1.2
    assert dig["s5_drift_alerts"] == []
    assert dig["daemon_uptime_hours"] == 8


def test_not_twice(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    now = datetime.now(timezone.utc)
    save_state({
        "pending_digest": dict(_FULL_DIGEST),
        "last_digest_shown_at": (now - timedelta(hours=4)).isoformat(),
    })

    response: dict = {"hits": []}
    _inject_overnight_digest(response)
    assert "overnight_digest" in response
    assert response["overnight_digest"] == _EMPTY_DIGEST_EXPECTED


def test_no_digest_when_none_pending(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import save_state

    save_state({})
    response: dict = {"hits": []}
    _inject_overnight_digest(response)
    assert "overnight_digest" in response
    assert response["overnight_digest"] == _EMPTY_DIGEST_EXPECTED


def test_digest_cleared_after_delivery(isolated_state):
    from iai_mcp.core import _inject_overnight_digest
    from iai_mcp.daemon_state import load_state, save_state

    now = datetime.now(timezone.utc)
    save_state({
        "pending_digest": dict(_FULL_DIGEST),
        "last_digest_shown_at": (now - timedelta(hours=20)).isoformat(),
    })

    response: dict = {"hits": []}
    _inject_overnight_digest(response)
    assert "overnight_digest" in response

    on_disk = load_state()
    assert "pending_digest" not in on_disk
    shown_at = datetime.fromisoformat(on_disk["last_digest_shown_at"])
    if shown_at.tzinfo is None:
        shown_at = shown_at.replace(tzinfo=timezone.utc)
    assert shown_at >= now - timedelta(seconds=5)


def test_exception_is_silent(isolated_state, monkeypatch):
    from iai_mcp import core

    def boom(*args, **kwargs):
        raise RuntimeError("simulated state corruption")

    monkeypatch.setattr("iai_mcp.core.get_pending_digest", boom)

    response: dict = {"hits": [], "existing": True}
    core._inject_overnight_digest(response)
    assert response.get("existing") is True
    assert "overnight_digest" in response
    assert response["overnight_digest"] == _EMPTY_DIGEST_EXPECTED
