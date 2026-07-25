from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import patch


from iai_mcp.idle_detector import IdleDetector, IdleStatus


def _completed_process(
    stdout: str | bytes = b"", returncode: int = 0
) -> subprocess.CompletedProcess[bytes]:
    raw = stdout.encode("utf-8") if isinstance(stdout, str) else stdout
    proc: subprocess.CompletedProcess[bytes] = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=raw, stderr=b""
    )
    return proc


def _ioreg_stdout(idle_ns: int) -> str:
    return (
        "+-o IOHIDSystem  <class IOHIDSystem, id 0x100000abc, registered>\n"
        f'    | "HIDIdleTime" = {idle_ns}\n'
        '    | "DisplayWrangler" = 1\n'
    )


def _pmset_log_stdout(events: list[tuple[str, str]]) -> str:
    lines = []
    for ts, marker in events:
        lines.append(f"{ts} {marker} Notification clientId=foo")
    return "\n".join(lines) + ("\n" if lines else "")


def _now_pmset_ts(offset_min: int) -> str:
    ts = datetime.now(timezone.utc) - timedelta(minutes=offset_min)
    return ts.strftime("%Y-%m-%d %H:%M:%S +0000")


def test_hid_idle_time_sec_parses_ioreg_output() -> None:
    fake = _completed_process(stdout=_ioreg_stdout(idle_ns=612_000_000_000))
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().hid_idle_time_sec()
    assert result == 612


def test_hid_idle_time_sec_returns_none_when_ioreg_missing() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=FileNotFoundError(2, "No such file"),
    ):
        result = IdleDetector().hid_idle_time_sec()
    assert result is None


def test_hid_idle_time_sec_returns_none_on_nonzero_exit() -> None:
    fake = _completed_process(stdout="", returncode=1)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().hid_idle_time_sec()
    assert result is None


def test_hid_idle_time_sec_returns_none_on_timeout() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd=["ioreg"], timeout=5),
    ):
        result = IdleDetector().hid_idle_time_sec()
    assert result is None


def test_pmset_recent_sleep_detects_event_within_window() -> None:
    log = _pmset_log_stdout([
        (_now_pmset_ts(offset_min=2), "System Sleep"),
    ])
    fake = _completed_process(stdout=log)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().pmset_recent_sleep(window_min=5)
    assert result is True


def test_pmset_recent_sleep_detects_display_off_event() -> None:
    log = _pmset_log_stdout([
        (_now_pmset_ts(offset_min=1), "Display is turned off"),
    ])
    fake = _completed_process(stdout=log)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().pmset_recent_sleep(window_min=5)
    assert result is True


def test_pmset_recent_sleep_returns_false_when_no_recent_event() -> None:
    log = _pmset_log_stdout([
        (_now_pmset_ts(offset_min=60), "System Sleep"),
        (_now_pmset_ts(offset_min=120), "Display is turned off"),
    ])
    fake = _completed_process(stdout=log)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().pmset_recent_sleep(window_min=5)
    assert result is False


def test_pmset_recent_sleep_returns_false_when_pmset_missing() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=FileNotFoundError(2, "No such file"),
    ):
        result = IdleDetector().pmset_recent_sleep()
    assert result is False


def test_pmset_recent_sleep_survives_invalid_utf8_in_log() -> None:
    log = _pmset_log_stdout([
        (_now_pmset_ts(offset_min=2), "System Sleep"),
    ])
    corrupted = b"\xd2 garbage line\n" + log.encode("utf-8")
    fake = _completed_process(stdout=corrupted)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().pmset_recent_sleep(window_min=5)
    assert result is True


def test_hid_idle_time_sec_survives_invalid_utf8() -> None:
    corrupted = b"\xff\xfe" + _ioreg_stdout(idle_ns=42_000_000_000).encode("utf-8")
    fake = _completed_process(stdout=corrupted)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().hid_idle_time_sec()
    assert result == 42


def test_sleep_eligible_heartbeat_idle_path() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=AssertionError("must not spawn when heartbeat-idle is True"),
    ):
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=True)
    assert result is True


def test_sleep_eligible_hid_idle_path() -> None:
    fake_ioreg = _completed_process(
        stdout=_ioreg_stdout(idle_ns=1900 * 1_000_000_000)
    )
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Darwin"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        return_value=fake_ioreg,
    ) as run_mock:
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is True
    assert run_mock.call_count == 1


def test_sleep_eligible_pmset_path() -> None:
    fake_ioreg = _completed_process(
        stdout=_ioreg_stdout(idle_ns=10 * 1_000_000_000)
    )
    fake_pmset = _completed_process(
        stdout=_pmset_log_stdout([
            (_now_pmset_ts(offset_min=2), "System Sleep"),
        ])
    )
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Darwin"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_ioreg, fake_pmset],
    ):
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is True


def test_sleep_eligible_all_false() -> None:
    fake_ioreg = _completed_process(
        stdout=_ioreg_stdout(idle_ns=10 * 1_000_000_000)
    )
    fake_pmset = _completed_process(stdout="")
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Darwin"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_ioreg, fake_pmset],
    ):
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is False


def test_status_for_doctor_row_all_signals_available() -> None:
    fake_ioreg = _completed_process(
        stdout=_ioreg_stdout(idle_ns=42 * 1_000_000_000)
    )
    fake_pmset_log = _completed_process(stdout="")
    fake_pmset_g = _completed_process(stdout="Now drawing from 'AC Power'\n")
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Darwin"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_ioreg, fake_pmset_log, fake_pmset_g],
    ):
        status = IdleDetector().status()

    assert isinstance(status, IdleStatus)
    assert status.hid_idle_sec == 42
    assert status.pmset_recent_sleep is False
    assert "HIDIdleTime" in status.available_signals
    assert "pmset" in status.available_signals


def test_status_when_signals_missing() -> None:
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Darwin"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=FileNotFoundError(2, "No such file"),
    ):
        status = IdleDetector().status()
    assert status.hid_idle_sec is None
    assert status.pmset_recent_sleep is False
    assert status.available_signals == []


def test_sleep_eligible_unknown_platform_has_no_secondary_signal() -> None:
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="FreeBSD"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=AssertionError("must not spawn on an unsupported platform"),
    ):
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is False


# ---------------------------------------------------------------------------
# Linux: systemd-logind IdleHint / IdleSinceHint
# ---------------------------------------------------------------------------

_OWN_UID = os.getuid() if hasattr(os, "getuid") else 1000


def _busctl_list_sessions_json(
    sessions: list[tuple[str, int, str, str, str]],
) -> str:
    # Real shape: {"type":"a(susso)","data":[[[id,uid,user,seat,path], ...]]}
    return json.dumps({"type": "a(susso)", "data": [[list(s) for s in sessions]]})


def _busctl_property_json(type_code: str, value: object) -> str:
    return json.dumps({"type": type_code, "data": value})


def test_logind_session_paths_picks_own_uid_seat_session() -> None:
    fake = _completed_process(
        stdout=_busctl_list_sessions_json(
            [("c2", _OWN_UID, "testuser", "seat0", "/org/freedesktop/login1/session/c2")]
        )
    )
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector()._logind_session_paths()
    assert result == ["/org/freedesktop/login1/session/c2"]


def test_logind_session_paths_skips_other_uids_and_headless_sessions() -> None:
    # Another user's graphical session and our own headless (no-seat) SSH
    # session must both be skipped in favor of our own seat-attached one.
    fake = _completed_process(
        stdout=_busctl_list_sessions_json(
            [
                (
                    "c1", _OWN_UID + 1, "otheruser",
                    "seat0", "/org/freedesktop/login1/session/c1",
                ),
                ("c3", _OWN_UID, "testuser", "", "/org/freedesktop/login1/session/c3"),
                (
                    "c2", _OWN_UID, "testuser",
                    "seat0", "/org/freedesktop/login1/session/c2",
                ),
            ]
        )
    )
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector()._logind_session_paths()
    assert result == ["/org/freedesktop/login1/session/c2"]


def test_logind_session_paths_returns_all_matching_seats() -> None:
    # Real multi-seat hardware: two seat-attached sessions for the same
    # user. Both must be returned -- picking only one risks reading the
    # wrong seat's idle state while the other is actively in use.
    fake = _completed_process(
        stdout=_busctl_list_sessions_json(
            [
                (
                    "c2", _OWN_UID, "testuser",
                    "seat0", "/org/freedesktop/login1/session/c2",
                ),
                (
                    "c4", _OWN_UID, "testuser",
                    "seat1", "/org/freedesktop/login1/session/c4",
                ),
            ]
        )
    )
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector()._logind_session_paths()
    assert result == [
        "/org/freedesktop/login1/session/c2",
        "/org/freedesktop/login1/session/c4",
    ]


def test_logind_session_paths_returns_empty_when_no_matching_session() -> None:
    fake = _completed_process(
        stdout=_busctl_list_sessions_json(
            [("c1", _OWN_UID + 1, "otheruser", "seat0", "/org/freedesktop/login1/session/c1")]
        )
    )
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector()._logind_session_paths()
    assert result == []


def test_logind_session_paths_returns_empty_when_busctl_missing() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=FileNotFoundError(2, "No such file"),
    ):
        result = IdleDetector()._logind_session_paths()
    assert result == []


def test_logind_aggregate_idle_returns_none_when_not_idle() -> None:
    fake_hint = _completed_process(stdout=_busctl_property_json("b", False))
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake_hint):
        result = IdleDetector()._logind_aggregate_idle(
            ["/org/freedesktop/login1/session/c2"]
        )
    assert result is None


def test_logind_aggregate_idle_computes_elapsed_when_idle() -> None:
    idle_since_usec = int(
        (datetime.now(timezone.utc) - timedelta(seconds=1800)).timestamp() * 1_000_000
    )
    fake_hint = _completed_process(stdout=_busctl_property_json("b", True))
    fake_since = _completed_process(stdout=_busctl_property_json("t", idle_since_usec))
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_hint, fake_since],
    ):
        result = IdleDetector()._logind_aggregate_idle(
            ["/org/freedesktop/login1/session/c2"]
        )
    assert result is not None
    assert 1795 <= result <= 1805


def test_logind_aggregate_idle_returns_none_for_empty_session_list() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=AssertionError("must not spawn when there are no sessions"),
    ):
        result = IdleDetector()._logind_aggregate_idle([])
    assert result is None


def test_logind_aggregate_idle_not_idle_if_any_seat_is_active() -> None:
    # Two seat-attached sessions: seat0 has been idle for a long time, but
    # seat1 is actively in use. Overall must report "not idle" -- a single
    # active seat means the user isn't idle, no matter how long the other
    # seat has sat untouched.
    idle_since_usec = int(
        (datetime.now(timezone.utc) - timedelta(seconds=7200)).timestamp() * 1_000_000
    )
    fake_seat0_hint = _completed_process(stdout=_busctl_property_json("b", True))
    fake_seat0_since = _completed_process(
        stdout=_busctl_property_json("t", idle_since_usec)
    )
    fake_seat1_hint = _completed_process(stdout=_busctl_property_json("b", False))
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_seat0_hint, fake_seat0_since, fake_seat1_hint],
    ):
        result = IdleDetector()._logind_aggregate_idle(
            [
                "/org/freedesktop/login1/session/c2",
                "/org/freedesktop/login1/session/c4",
            ]
        )
    assert result is None


def test_logind_aggregate_idle_takes_min_across_idle_seats() -> None:
    # Both seats idle, but for different durations -- the aggregate must be
    # the shorter (more conservative) of the two, not the longer.
    seat0_since = int(
        (datetime.now(timezone.utc) - timedelta(seconds=7200)).timestamp() * 1_000_000
    )
    seat1_since = int(
        (datetime.now(timezone.utc) - timedelta(seconds=100)).timestamp() * 1_000_000
    )
    fake_seat0_hint = _completed_process(stdout=_busctl_property_json("b", True))
    fake_seat0_since = _completed_process(
        stdout=_busctl_property_json("t", seat0_since)
    )
    fake_seat1_hint = _completed_process(stdout=_busctl_property_json("b", True))
    fake_seat1_since = _completed_process(
        stdout=_busctl_property_json("t", seat1_since)
    )
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[
            fake_seat0_hint, fake_seat0_since,
            fake_seat1_hint, fake_seat1_since,
        ],
    ):
        result = IdleDetector()._logind_aggregate_idle(
            [
                "/org/freedesktop/login1/session/c2",
                "/org/freedesktop/login1/session/c4",
            ]
        )
    assert result is not None
    assert 95 <= result <= 105


def test_os_idle_time_sec_dispatches_to_logind_on_linux() -> None:
    fake_sessions = _completed_process(
        stdout=_busctl_list_sessions_json(
            [("c2", _OWN_UID, "testuser", "seat0", "/org/freedesktop/login1/session/c2")]
        )
    )
    fake_hint = _completed_process(stdout=_busctl_property_json("b", False))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_sessions, fake_hint],
    ):
        idle_sec, source = IdleDetector().os_idle_time_sec()
    assert idle_sec is None
    assert source == "logind"


def test_os_idle_time_sec_dispatches_to_hid_on_darwin() -> None:
    fake_ioreg = _completed_process(
        stdout=_ioreg_stdout(idle_ns=100 * 1_000_000_000)
    )
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Darwin"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run", return_value=fake_ioreg
    ):
        idle_sec, source = IdleDetector().os_idle_time_sec()
    assert idle_sec == 100
    assert source == "HIDIdleTime"


def test_os_idle_time_sec_none_on_unsupported_platform() -> None:
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="FreeBSD"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=AssertionError("must not spawn on an unsupported platform"),
    ):
        idle_sec, source = IdleDetector().os_idle_time_sec()
    assert idle_sec is None
    assert source is None


def test_sleep_eligible_logind_idle_path() -> None:
    idle_since_usec = int(
        (datetime.now(timezone.utc) - timedelta(seconds=1900)).timestamp() * 1_000_000
    )
    fake_sessions = _completed_process(
        stdout=_busctl_list_sessions_json(
            [("c2", _OWN_UID, "testuser", "seat0", "/org/freedesktop/login1/session/c2")]
        )
    )
    fake_hint = _completed_process(stdout=_busctl_property_json("b", True))
    fake_since = _completed_process(
        stdout=_busctl_property_json("t", idle_since_usec)
    )
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_sessions, fake_hint, fake_since],
    ):
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is True


def test_sleep_eligible_logind_not_idle_has_no_pmset_fallback() -> None:
    fake_sessions = _completed_process(
        stdout=_busctl_list_sessions_json(
            [("c2", _OWN_UID, "testuser", "seat0", "/org/freedesktop/login1/session/c2")]
        )
    )
    fake_hint = _completed_process(stdout=_busctl_property_json("b", False))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_sessions, fake_hint],
    ) as run_mock:
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is False
    # Only the logind calls -- no attempt to shell out to pmset on Linux.
    assert run_mock.call_count == 2


def test_status_reports_logind_signal_on_linux() -> None:
    fake_sessions = _completed_process(
        stdout=_busctl_list_sessions_json(
            [("c2", _OWN_UID, "testuser", "seat0", "/org/freedesktop/login1/session/c2")]
        )
    )
    fake_hint = _completed_process(stdout=_busctl_property_json("b", False))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_sessions, fake_hint],
    ):
        status = IdleDetector().status()
    assert status.available_signals == ["logind"]
    assert status.pmset_recent_sleep is False


def test_status_reports_no_signal_when_logind_unreachable() -> None:
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=FileNotFoundError(2, "No such file"),
    ):
        status = IdleDetector().status()
    assert status.hid_idle_sec is None
    assert status.available_signals == []


def test_subprocess_uses_array_form_not_shell() -> None:
    fake = _completed_process(stdout="")
    captured: list[tuple[tuple, dict]] = []

    def _capture(*args, **kwargs):
        captured.append((args, kwargs))
        return fake

    with patch(
        "iai_mcp.idle_detector.subprocess.run", side_effect=_capture
    ):
        IdleDetector().status()

    assert len(captured) >= 1
    for args, kwargs in captured:
        assert isinstance(args[0], list), (
            f"subprocess.run command must be a list (array form), got: {args[0]!r}"
        )
        assert kwargs.get("shell", False) is False, (
            "subprocess.run must NOT use shell=True (PATH-injection risk)"
        )
        assert "timeout" in kwargs and kwargs["timeout"] > 0, (
            f"subprocess.run must set a finite timeout, got: {kwargs}"
        )
