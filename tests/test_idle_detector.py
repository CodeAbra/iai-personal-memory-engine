from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from iai_mcp.idle_detector import IdleDetector, IdleStatus


def _completed_process(
    stdout: str = "", returncode: int = 0
) -> subprocess.CompletedProcess[str]:
    proc: subprocess.CompletedProcess[str] = subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=""
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


def test_sleep_eligible_unknown_platform_has_no_secondary_signal() -> None:
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="FreeBSD"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=AssertionError("must not spawn on an unsupported platform"),
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


# ---------------------------------------------------------------------------
# Linux: systemd-logind IdleHint / IdleSinceHint
# ---------------------------------------------------------------------------


def _busctl_session_path_stdout(path: str = "/org/freedesktop/login1/session/c2") -> str:
    return f'o "{path}"\n'


def _busctl_bool_stdout(value: bool) -> str:
    return f"b {'true' if value else 'false'}\n"


def _busctl_uint64_stdout(value: int) -> str:
    return f"t {value}\n"


def test_logind_session_path_parses_busctl_call_output() -> None:
    fake = _completed_process(stdout=_busctl_session_path_stdout())
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector()._logind_session_path()
    assert result == "/org/freedesktop/login1/session/c2"


def test_logind_session_path_returns_none_when_busctl_missing() -> None:
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=FileNotFoundError(2, "No such file"),
    ):
        result = IdleDetector()._logind_session_path()
    assert result is None


def test_logind_idle_time_sec_returns_none_when_not_idle() -> None:
    fake_path = _completed_process(stdout=_busctl_session_path_stdout())
    fake_hint = _completed_process(stdout=_busctl_bool_stdout(False))
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_path, fake_hint],
    ):
        result = IdleDetector().logind_idle_time_sec()
    assert result is None


def test_logind_idle_time_sec_computes_elapsed_when_idle() -> None:
    idle_since_usec = int(
        (datetime.now(timezone.utc) - timedelta(seconds=1800)).timestamp() * 1_000_000
    )
    fake_path = _completed_process(stdout=_busctl_session_path_stdout())
    fake_hint = _completed_process(stdout=_busctl_bool_stdout(True))
    fake_since = _completed_process(stdout=_busctl_uint64_stdout(idle_since_usec))
    with patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_path, fake_hint, fake_since],
    ):
        result = IdleDetector().logind_idle_time_sec()
    assert result is not None
    assert 1795 <= result <= 1805


def test_logind_idle_time_sec_returns_none_when_session_not_found() -> None:
    fake = _completed_process(stdout="", returncode=1)
    with patch("iai_mcp.idle_detector.subprocess.run", return_value=fake):
        result = IdleDetector().logind_idle_time_sec()
    assert result is None


def test_os_idle_time_sec_dispatches_to_logind_on_linux() -> None:
    fake_path = _completed_process(stdout=_busctl_session_path_stdout())
    fake_hint = _completed_process(stdout=_busctl_bool_stdout(False))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_path, fake_hint],
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
    fake_path = _completed_process(stdout=_busctl_session_path_stdout())
    fake_hint = _completed_process(stdout=_busctl_bool_stdout(True))
    fake_since = _completed_process(stdout=_busctl_uint64_stdout(idle_since_usec))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_path, fake_hint, fake_since],
    ):
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is True


def test_sleep_eligible_logind_not_idle_has_no_pmset_fallback() -> None:
    fake_path = _completed_process(stdout=_busctl_session_path_stdout())
    fake_hint = _completed_process(stdout=_busctl_bool_stdout(False))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_path, fake_hint],
    ) as run_mock:
        result = IdleDetector().sleep_eligible(heartbeat_idle_30min=False)
    assert result is False
    # Only the logind calls -- no attempt to shell out to pmset on Linux.
    assert run_mock.call_count == 2


def test_status_reports_logind_signal_on_linux() -> None:
    fake_path = _completed_process(stdout=_busctl_session_path_stdout())
    fake_hint = _completed_process(stdout=_busctl_bool_stdout(False))
    with patch(
        "iai_mcp.idle_detector.platform.system", return_value="Linux"
    ), patch(
        "iai_mcp.idle_detector.subprocess.run",
        side_effect=[fake_path, fake_hint],
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
