from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from iai_mcp import core


def _window_covering_now() -> tuple[int, int]:
    # Same clock as the consumer: the gate reads system-local time (the
    # clock the presence buckets are produced in), not the config tz.
    now_local = datetime.now().astimezone()
    cur_bucket = (now_local.hour * 60 + now_local.minute) // 30
    start = (cur_bucket - 2) % 48
    return (start, 8)


def test_inject_sleep_suggestion_dual_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_state = {"quiet_window": _window_covering_now()}

    def _load() -> dict:
        return dict(fake_state)

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _load)

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" in response, (
        f"expected injection on dual-gate pass, got {response!r}"
    )
    assert response["sleep_suggestion"]["message_hint"] == "user_wind_down_detected"


def test_inject_sleep_suggestion_no_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_state = {"quiet_window": _window_covering_now()}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: dict(fake_state),
    )

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(
        response, cue="how do I configure pytest", language="en",
    )
    assert "sleep_suggestion" not in response


def test_inject_sleep_suggestion_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No learned window: the night default applies (never dark). Force the
    # effective window deterministically away from now, then over it.
    monkeypatch.setattr("iai_mcp.daemon_state.load_state", lambda: {})

    now_local = datetime.now().astimezone()
    away = now_local + timedelta(hours=4)
    away_end = now_local + timedelta(hours=6)
    monkeypatch.setenv(
        "IAI_MCP_CONSOLIDATION_WINDOW",
        f"{away.hour:02d}:00-{away_end.hour:02d}:00",
    )
    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" not in response

    covering = now_local - timedelta(hours=1)
    covering_end = now_local + timedelta(hours=2)
    monkeypatch.setenv(
        "IAI_MCP_CONSOLIDATION_WINDOW",
        f"{covering.hour:02d}:00-{covering_end.hour:02d}:00",
    )
    response2: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response2, cue="good night", language="en")
    assert "sleep_suggestion" in response2


def test_inject_sleep_suggestion_ignores_config_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A store config seeded with a different tz than the system must not
    # suppress the suggestion: producer and consumer share one clock.
    fake_state = {"quiet_window": _window_covering_now()}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state", lambda: dict(fake_state),
    )

    def _boom():
        raise AssertionError("config tz must not be consulted")

    monkeypatch.setattr("iai_mcp.tz.load_user_tz", _boom)

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" in response


def test_inject_sleep_suggestion_detector_raises_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic bedtime failure")

    monkeypatch.setattr("iai_mcp.bedtime.detect_wind_down", _boom)

    response: dict = {"hits": [], "anti_hits": [], "budget_used": 0}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" not in response
    assert response == {"hits": [], "anti_hits": [], "budget_used": 0}
