"""Presence-based quiet-window learning, the deep-idle starvation
override, the sleep entry/exit predicates, and the boot-time reset of a
window the presence learner did not produce."""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from iai_mcp.quiet_window import (
    BUCKET_COUNT,
    MIN_DAYS_FOR_LEARN,
    effective_consolidation_window,
    learn_quiet_window_from_presence,
    local_bucket,
    prune_presence_masks,
    stamp_presence_mask,
    within_window,
)


def _mask_for_buckets(buckets) -> int:
    m = 0
    for b in buckets:
        m |= 1 << b
    return m


def _day_masks(n_days: int, busy_buckets) -> dict:
    base = datetime(2026, 8, 1)
    mask = _mask_for_buckets(busy_buckets)
    return {
        (base + timedelta(days=i)).strftime("%Y-%m-%d"): mask
        for i in range(n_days)
    }


def test_presence_learner_finds_the_truly_quiet_night() -> None:
    # Busy 09:00-23:00 (buckets 18..45), quiet at night.
    busy = list(range(18, 46))
    window = learn_quiet_window_from_presence(_day_masks(10, busy))
    assert window is not None
    start, length = window
    local = datetime(2026, 8, 5, 3, 0)  # 03:00 — deep night
    assert within_window((start, length), local, None)


def test_presence_learner_picks_the_only_quiet_stretch() -> None:
    # Busy 00:00-18:00 every day; the evening is the only quiet stretch
    # and the window must land there, never at night.
    busy = list(range(0, 36))
    window = learn_quiet_window_from_presence(_day_masks(10, busy))
    assert window is not None
    start, length = window
    evening = datetime(2026, 8, 5, 20, 0)
    night = datetime(2026, 8, 5, 3, 0)
    assert within_window((start, length), evening, None)
    assert not within_window((start, length), night, None)


def test_presence_learner_requires_enough_days() -> None:
    busy = list(range(18, 46))
    assert (
        learn_quiet_window_from_presence(_day_masks(MIN_DAYS_FOR_LEARN - 1, busy))
        is None
    )
    assert learn_quiet_window_from_presence(None) is None
    assert learn_quiet_window_from_presence({}) is None


def test_presence_learner_no_quiet_stretch_returns_none() -> None:
    window = learn_quiet_window_from_presence(
        _day_masks(10, range(BUCKET_COUNT))
    )
    assert window is None


def test_presence_learner_busy_day_fraction_boundary() -> None:
    """With 7 observed days the busy threshold is 2 days: a bucket busy on
    a single day stays quiet, a bucket busy on two days turns busy."""
    day_busy = list(range(18, 46))
    two_am = datetime(2026, 8, 5, 2, 0)  # bucket 4

    def masks_with_night_blip(blip_days: int) -> dict:
        masks = _day_masks(7, day_busy)
        keys = sorted(masks)
        blip = 1 << local_bucket(two_am)
        for k in keys[:blip_days]:
            masks[k] |= blip
        return masks

    one_day = learn_quiet_window_from_presence(masks_with_night_blip(1))
    assert one_day is not None
    assert within_window(one_day, two_am, None), (
        "a bucket busy on 1 of 7 days must stay quiet"
    )

    two_days = learn_quiet_window_from_presence(masks_with_night_blip(2))
    assert two_days is not None
    assert not within_window(two_days, two_am, None), (
        "a bucket busy on 2 of 7 days must count as busy"
    )


def test_stamp_presence_mask_sets_and_deduplicates() -> None:
    masks: dict = {}
    ts = datetime(2026, 8, 7, 14, 10)
    assert stamp_presence_mask(masks, ts, busy=True) is True
    key = ts.strftime("%Y-%m-%d")
    assert masks[key] & (1 << local_bucket(ts))
    assert stamp_presence_mask(masks, ts, busy=True) is False
    assert stamp_presence_mask(masks, ts, busy=False) is False


def test_prune_presence_masks_drops_old_days() -> None:
    today = datetime(2026, 8, 7)
    masks = {
        "2026-08-07": 1,
        "2026-08-02": 1,
        "2026-07-24": 1,  # exactly the default keep horizon — retained
        "2026-07-23": 1,
        "2026-06-01": 1,
    }
    assert prune_presence_masks(masks, today) is True
    assert sorted(masks) == ["2026-07-24", "2026-08-02", "2026-08-07"]
    assert prune_presence_masks(masks, today) is False


def _starved_wall() -> float:
    from iai_mcp.daemon import DEEP_IDLE_STARVATION_SEC

    return time.time() - (DEEP_IDLE_STARVATION_SEC + 10.0)


def test_deep_idle_override_thresholds(monkeypatch) -> None:
    from iai_mcp.daemon import DEEP_IDLE_OVERRIDE_ENV, _deep_idle_override

    monkeypatch.delenv(DEEP_IDLE_OVERRIDE_ENV, raising=False)
    wall = _starved_wall()
    # No OS-level idle source: never override, whatever the heartbeat says.
    assert _deep_idle_override(None, wall) is False
    assert _deep_idle_override(89 * 60, wall) is False
    assert _deep_idle_override(90 * 60, wall) is True

    monkeypatch.setenv(DEEP_IDLE_OVERRIDE_ENV, "30")
    assert _deep_idle_override(31 * 60, wall) is True

    monkeypatch.setenv(DEEP_IDLE_OVERRIDE_ENV, "0")
    assert _deep_idle_override(10 ** 9, wall) is False

    for junk in ("garbage", "inf", "nan", "-inf"):
        monkeypatch.setenv(DEEP_IDLE_OVERRIDE_ENV, junk)
        assert _deep_idle_override(90 * 60, wall) is True, junk
        assert _deep_idle_override(89 * 60, wall) is False, junk


def test_deep_idle_override_requires_starvation(monkeypatch) -> None:
    from iai_mcp.daemon import DEEP_IDLE_OVERRIDE_ENV, _deep_idle_override

    monkeypatch.delenv(DEEP_IDLE_OVERRIDE_ENV, raising=False)
    recent = time.time() - 3600.0
    assert _deep_idle_override(10 ** 9, recent) is False, (
        "a clean cycle within the starvation horizon must hold the override"
    )
    assert _deep_idle_override(90 * 60, _starved_wall()) is True
    assert _deep_idle_override(10 ** 9, 0.0) is False, (
        "an unknown last-cycle time must read as NOT starved"
    )


def test_stamp_activity_presence_judges_busy_by_os_idle() -> None:
    from iai_mcp.daemon import PRESENCE_BUSY_IDLE_SEC, _stamp_activity_presence

    # OS says active input — busy, even with no live Claude session.
    state: dict = {}
    assert _stamp_activity_presence(state, 10_000.0, os_idle_sec=0.0) is True
    assert len(state["activity_presence"]) == 1
    assert _stamp_activity_presence(state, 10_000.0, os_idle_sec=0.0) is False

    # OS says nobody at the keyboard — not busy, even with a live session.
    state2: dict = {}
    assert (
        _stamp_activity_presence(
            state2, 0.0, os_idle_sec=PRESENCE_BUSY_IDLE_SEC + 1
        )
        is False
    )
    assert state2.get("activity_presence") == {}

    # OS source unreachable: heartbeat idle is the fallback signal.
    state3: dict = {}
    assert _stamp_activity_presence(state3, 0.0, os_idle_sec=None) is True
    state4: dict = {}
    assert (
        _stamp_activity_presence(
            state4, PRESENCE_BUSY_IDLE_SEC + 1, os_idle_sec=None
        )
        is False
    )


def test_boot_reset_drops_non_presence_window() -> None:
    from iai_mcp.daemon import _reset_unsourced_quiet_window

    state = {"quiet_window": [25, 16]}
    assert _reset_unsourced_quiet_window(state) is True
    assert state["quiet_window"] is None

    state = {"quiet_window": [4, 8], "quiet_window_source": "presence"}
    assert _reset_unsourced_quiet_window(state) is False
    assert state["quiet_window"] == [4, 8]

    state = {"quiet_window": None}
    assert _reset_unsourced_quiet_window(state) is False


def _window_spec(start_local: datetime, end_local: datetime) -> str:
    return f"{start_local.hour:02d}:00-{end_local.hour:02d}:00"


def _env_window(monkeypatch, *, containing: bool) -> None:
    now = datetime.now().astimezone()
    if containing:
        spec = _window_spec(now - timedelta(hours=1), now + timedelta(hours=2))
    else:
        spec = _window_spec(now + timedelta(hours=2), now + timedelta(hours=4))
    monkeypatch.setenv("IAI_MCP_CONSOLIDATION_WINDOW", spec)


def test_may_enter_sleep_truth_table(monkeypatch) -> None:
    from iai_mcp.daemon import (
        DEEP_IDLE_OVERRIDE_ENV,
        SLEEP_HEARTBEAT_IDLE_SEC,
        _may_enter_sleep,
    )

    monkeypatch.delenv(DEEP_IDLE_OVERRIDE_ENV, raising=False)
    idle_ok = SLEEP_HEARTBEAT_IDLE_SEC + 1.0

    _env_window(monkeypatch, containing=True)
    assert _may_enter_sleep(
        idle_ok, True, {}, os_idle_sec=0.0, last_clean_cycle_wall=time.time()
    ) is True
    assert _may_enter_sleep(
        SLEEP_HEARTBEAT_IDLE_SEC - 1.0, True, {},
        os_idle_sec=0.0, last_clean_cycle_wall=time.time(),
    ) is False
    assert _may_enter_sleep(
        idle_ok, False, {}, os_idle_sec=0.0, last_clean_cycle_wall=time.time()
    ) is False

    _env_window(monkeypatch, containing=False)
    assert _may_enter_sleep(
        idle_ok, True, {}, os_idle_sec=0.0, last_clean_cycle_wall=time.time()
    ) is False, "outside the window, no override — must not enter"
    assert _may_enter_sleep(
        idle_ok, True, {},
        os_idle_sec=91 * 60.0, last_clean_cycle_wall=_starved_wall(),
    ) is True, "outside the window a starved deep-idle machine may enter"
    assert _may_enter_sleep(
        idle_ok, True, {},
        os_idle_sec=91 * 60.0, last_clean_cycle_wall=time.time(),
    ) is False, "a recent clean cycle holds the override"
    assert _may_enter_sleep(
        idle_ok, True, {}, os_idle_sec=None, last_clean_cycle_wall=_starved_wall()
    ) is False, "no OS idle source — heartbeat idle alone never overrides"


def test_must_leave_sleep_truth_table(monkeypatch) -> None:
    from iai_mcp.daemon import DEEP_IDLE_OVERRIDE_ENV, _must_leave_sleep

    monkeypatch.delenv(DEEP_IDLE_OVERRIDE_ENV, raising=False)

    _env_window(monkeypatch, containing=True)
    assert _must_leave_sleep(
        {}, os_idle_sec=0.0, last_clean_cycle_wall=time.time()
    ) is False

    _env_window(monkeypatch, containing=False)
    assert _must_leave_sleep(
        {}, os_idle_sec=0.0, last_clean_cycle_wall=time.time()
    ) is True
    assert _must_leave_sleep(
        {}, os_idle_sec=91 * 60.0, last_clean_cycle_wall=_starved_wall()
    ) is False, "an override-entered sleep must not be kicked awake"


def test_idle_transition_event_truth_table(monkeypatch) -> None:
    from iai_mcp.daemon import (
        DEEP_IDLE_OVERRIDE_ENV,
        SLEEP_HEARTBEAT_IDLE_SEC,
        _idle_transition_event,
    )

    monkeypatch.delenv(DEEP_IDLE_OVERRIDE_ENV, raising=False)
    idle_ok = SLEEP_HEARTBEAT_IDLE_SEC + 1.0
    kw = {"os_idle_sec": 0.0, "last_clean_cycle_wall": time.time(),
          "drowsy_after_sec": 300.0}

    _env_window(monkeypatch, containing=True)
    assert _idle_transition_event(True, idle_ok, True, {}, **kw) == "wake_refresh", (
        "an active wrapper always outranks every idle branch"
    )
    assert _idle_transition_event(False, idle_ok, True, {}, **kw) == "sleep"
    assert _idle_transition_event(False, 301.0, True, {}, **kw) == "drowsy"
    assert _idle_transition_event(False, 10.0, True, {}, **kw) is None

    _env_window(monkeypatch, containing=False)
    assert _idle_transition_event(False, idle_ok, True, {}, **kw) == "drowsy", (
        "outside the window a long idle degrades to drowsy, never sleep"
    )


def test_effective_os_idle_tri_state() -> None:
    from iai_mcp.daemon import _effective_os_idle

    assert _effective_os_idle(7200, "HIDIdleTime") == 7200.0
    assert _effective_os_idle(0, "HIDIdleTime") == 0.0
    # Source reachable, session NOT idle (Linux logind active user): the
    # user is at the keyboard — idle 0, never a heartbeat fallback.
    assert _effective_os_idle(None, "logind") == 0.0
    assert _effective_os_idle(None, None) is None


def test_active_linux_user_is_stamped_busy() -> None:
    from iai_mcp.daemon import _effective_os_idle, _stamp_activity_presence

    # logind reports (None, "logind") for an active session; the stamp
    # must mark the bucket busy even with no Claude session alive.
    os_idle = _effective_os_idle(None, "logind")
    state: dict = {}
    assert _stamp_activity_presence(state, 10_000.0, os_idle_sec=os_idle) is True
    assert len(state["activity_presence"]) == 1


def test_tick_routes_through_the_predicates() -> None:
    # The decision function carries the behavior; the tick must call it
    # rather than re-inline the branch (name-token pin, format-insensitive).
    import inspect
    import re

    from iai_mcp import daemon as daemon_mod

    src = inspect.getsource(daemon_mod)
    assert src.count("_idle_transition_event") >= 2
    assert src.count("_must_leave_sleep") >= 2
    # The tick must feed the OS value into the STAMP call specifically —
    # dropping the kwarg would silently revert busy-judgement to the
    # heartbeat (the parameter is keyword-required, but the stamp call
    # sits in an advisory try/except where a TypeError would vanish).
    assert re.search(
        r"_stamp_activity_presence\([^)]*os_idle_sec=os_idle_sec", src, re.S
    ), "the tick must pass the OS idle into the presence stamp"


def test_transition_dispatch_table_is_complete() -> None:
    from iai_mcp.daemon import TRANSITION_DISPATCH
    from iai_mcp.lifecycle import LifecycleEvent

    assert TRANSITION_DISPATCH == {
        "wake_refresh": (
            "HEARTBEAT_REFRESH", "heartbeat_refresh_active_wrapper", {},
        ),
        "sleep": ("IDLE_30MIN", "sleep_on_idle_30min", {"sleep_eligible": True}),
        "drowsy": ("IDLE_5MIN", "drowsy_on_idle_5min", {}),
    }
    for ev_name, _reason, _extra in TRANSITION_DISPATCH.values():
        assert hasattr(LifecycleEvent, ev_name), ev_name


def test_every_transition_label_is_dispatchable(monkeypatch) -> None:
    # Any label the decision function can emit must have a dispatch row.
    from iai_mcp.daemon import (
        DEEP_IDLE_OVERRIDE_ENV,
        SLEEP_HEARTBEAT_IDLE_SEC,
        TRANSITION_DISPATCH,
        _idle_transition_event,
    )

    monkeypatch.delenv(DEEP_IDLE_OVERRIDE_ENV, raising=False)
    kw = {"os_idle_sec": 0.0, "last_clean_cycle_wall": time.time(),
          "drowsy_after_sec": 300.0}
    _env_window(monkeypatch, containing=True)
    seen = {
        _idle_transition_event(True, 0.0, True, {}, **kw),
        _idle_transition_event(False, SLEEP_HEARTBEAT_IDLE_SEC + 1, True, {}, **kw),
        _idle_transition_event(False, 301.0, True, {}, **kw),
        _idle_transition_event(False, 1.0, True, {}, **kw),
    }
    assert seen == {"wake_refresh", "sleep", "drowsy", None}
    for label in seen - {None}:
        assert label in TRANSITION_DISPATCH


def test_effective_window_manual_override_precedence(monkeypatch) -> None:
    monkeypatch.delenv("IAI_MCP_CONSOLIDATION_WINDOW", raising=False)
    assert effective_consolidation_window(
        None, manual=["22:00", "23:30"]
    ) == (44, 3)
    assert effective_consolidation_window(
        [4, 8], manual=["22:00", "23:30"]
    ) == (44, 3), "the operator's manual window outranks the learned one"
    assert effective_consolidation_window(
        [4, 8], manual=["junk"]
    ) == (4, 8), "a malformed manual value falls through to the learned window"
    monkeypatch.setenv("IAI_MCP_CONSOLIDATION_WINDOW", "01:00-03:00")
    assert effective_consolidation_window(
        None, manual=["22:00", "23:30"]
    ) == (2, 4)


def test_wind_down_survives_an_unlearned_window(monkeypatch) -> None:
    from iai_mcp.bedtime import detect_wind_down

    monkeypatch.delenv("IAI_MCP_CONSOLIDATION_WINDOW", raising=False)
    # No learned window: the night default (02:00-06:00) applies and the
    # suggestion must NOT go dark.
    inside = datetime(2026, 8, 7, 3, 0, tzinfo=timezone.utc)
    assert detect_wind_down("good night", "en", {}, inside, timezone.utc)

    outside = datetime(2026, 8, 7, 12, 0, tzinfo=timezone.utc)
    assert (
        detect_wind_down("good night", "en", {}, outside, timezone.utc) is None
    )


def test_wind_down_honors_the_manual_window(monkeypatch) -> None:
    from iai_mcp.bedtime import detect_wind_down

    monkeypatch.delenv("IAI_MCP_CONSOLIDATION_WINDOW", raising=False)
    state = {"quiet_window_manual_override": ["22:00", "23:30"]}
    inside = datetime(2026, 8, 7, 22, 30, tzinfo=timezone.utc)
    assert detect_wind_down("good night", "en", state, inside, timezone.utc)


def test_window_spec_rejects_out_of_range_clock_values() -> None:
    from iai_mcp.quiet_window import parse_window_spec

    assert parse_window_spec("25:99-26:00") is None
    assert parse_window_spec("24:00-02:00") is None
    assert parse_window_spec("02:00-24:00") is None
    assert parse_window_spec("02:60-04:00") is None
    assert parse_window_spec("-1:00-04:00") is None
    assert parse_window_spec("22:00-23:30") == (44, 3)
    assert parse_window_spec("23:00-00:00") == (46, 2)


def test_set_quiet_window_cli_rejects_out_of_range(monkeypatch, capsys) -> None:
    import argparse

    from iai_mcp.cli import _daemon as daemon_cli

    recorded: list = []
    monkeypatch.setattr(
        "iai_mcp.daemon_state.update_state", lambda fn: recorded.append(fn)
    )
    args = argparse.Namespace(key="set-quiet-window", value="25:99-26:00")
    rc = daemon_cli.cmd_daemon_configure(args)
    assert rc == 2
    assert not recorded, "an invalid window must not be stored"
    assert "HH:MM-HH:MM" in capsys.readouterr().err
