from __future__ import annotations

import asyncio
import platform
from pathlib import Path

import pytest

from iai_mcp import cli as cli_mod


EXPECTED_WATCHDOG_ENV: dict[str, str] = {
    "IAI_MCP_WATCHDOG_LIVENESS_POLL_SEC": "30.0",
    "IAI_MCP_WATCHDOG_WARN_POLL_SEC": "7.0",
    "IAI_MCP_WATCHDOG_PROBE_TIMEOUT_SEC": "5.0",
    "IAI_MCP_WATCHDOG_FAILURE_DEBOUNCE_N": "3",
    "IAI_MCP_WATCHDOG_RSS_HARD_CAP_BYTES": "4294967296",
    "IAI_MCP_WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES": "1610612736",
    "IAI_MCP_WATCHDOG_MAX_RECOVERIES": "3",
    "IAI_MCP_WATCHDOG_RECOVERY_WINDOW_SEC": "600.0",
    "IAI_MCP_WATCHDOG_COLD_START_GRACE_SEC": "600.0",
}


def _plist_env_value(plist_text: str, key: str) -> str | None:
    import re

    m = re.search(
        rf"<key>{re.escape(key)}</key>\s*<string>(.*?)</string>",
        plist_text,
        re.DOTALL,
    )
    return m.group(1).strip() if m else None


@pytest.fixture
def fake_state_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    fake_home = tmp_path / "home"
    fake_home.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(cli_mod, "LOCK_PATH", fake_home / ".iai-mcp" / ".lock")
    monkeypatch.setattr(
        cli_mod, "SOCKET_PATH", fake_home / ".iai-mcp" / ".daemon.sock",
    )
    monkeypatch.setattr(
        cli_mod, "STATE_PATH", fake_home / ".iai-mcp" / ".daemon-state.json",
    )
    monkeypatch.setattr(
        cli_mod,
        "LAUNCHD_TARGET",
        fake_home / "Library" / "LaunchAgents" / "com.iai-mcp.daemon.plist",
    )
    monkeypatch.setattr(
        cli_mod,
        "SYSTEMD_TARGET",
        fake_home / ".config" / "systemd" / "user" / "iai-mcp-daemon.service",
    )
    return fake_home


@pytest.fixture
def captured_launchctl(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def _fake_run(argv, **kwargs):  # noqa: ANN001 -- mirrors subprocess.run
        calls.append(list(argv))

        class _R:
            returncode = 0
            stdout = ""
            stderr = ""

        return _R()

    monkeypatch.setattr(cli_mod.subprocess, "run", _fake_run)
    return calls


def test_rendered_plist_contains_all_watchdog_env_keys() -> None:
    rendered = cli_mod._render_launchd_plist()
    for key in EXPECTED_WATCHDOG_ENV:
        assert (
            f"<key>{key}</key>" in rendered
        ), f"rendered plist is missing watchdog key {key}"


def test_rendered_plist_watchdog_values_match_code_defaults() -> None:
    import iai_mcp.daemon as daemon_mod

    rendered = cli_mod._render_launchd_plist()
    for key, expected in EXPECTED_WATCHDOG_ENV.items():
        assert _plist_env_value(rendered, key) == expected, (
            f"{key}: rendered value != expected template value {expected}"
        )

    assert daemon_mod.WATCHDOG_LIVENESS_POLL_SEC == 30.0
    assert daemon_mod.WATCHDOG_WARN_POLL_SEC == 7.0
    assert daemon_mod.WATCHDOG_PROBE_TIMEOUT_SEC == 5.0
    assert daemon_mod.WATCHDOG_FAILURE_DEBOUNCE_N == 3
    assert daemon_mod.WATCHDOG_RSS_HARD_CAP_BYTES == 4294967296
    assert daemon_mod.WATCHDOG_RSS_CONTRIBUTOR_FLOOR_BYTES == 1610612736
    assert daemon_mod.WATCHDOG_MAX_RECOVERIES == 3
    assert daemon_mod.WATCHDOG_RECOVERY_WINDOW_SEC == 600.0
    assert daemon_mod.WATCHDOG_COLD_START_GRACE_SEC == 600.0


@pytest.mark.skipif(
    not hasattr(__import__("os"), "getuid"),
    reason="launchd gui/<uid> domain target is macOS-only; os.getuid() is "
    "absent on Windows",
)
def test_reenable_emits_bootout_bootstrap_kickstart_in_order(
    fake_state_dir: Path,
    captured_launchctl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0

    launchctl_subcmds = [
        argv[1] for argv in captured_launchctl
        if argv and argv[0] == "launchctl"
    ]
    assert launchctl_subcmds == ["bootout", "bootstrap", "kickstart"], (
        captured_launchctl
    )

    uid = __import__("os").getuid()
    target = str(cli_mod.LAUNCHD_TARGET)
    label = cli_mod.DAEMON_LABEL

    by_subcmd = {
        argv[1]: argv
        for argv in captured_launchctl
        if argv and argv[0] == "launchctl"
    }
    assert by_subcmd["bootout"] == ["launchctl", "bootout", f"gui/{uid}", target]
    assert by_subcmd["bootstrap"] == [
        "launchctl", "bootstrap", f"gui/{uid}", target,
    ]
    assert by_subcmd["kickstart"] == [
        "launchctl", "kickstart", f"gui/{uid}/{label}",
    ]


def test_reenable_writes_plist_carrying_watchdog_keys(
    fake_state_dir: Path,
    captured_launchctl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0
    assert cli_mod.LAUNCHD_TARGET.exists()

    written = cli_mod.LAUNCHD_TARGET.read_text()
    for key, expected in EXPECTED_WATCHDOG_ENV.items():
        assert f"<key>{key}</key>" in written, f"written plist missing {key}"
        assert _plist_env_value(written, key) == expected, (
            f"{key}: written plist value != expected {expected}"
        )


def test_status_ok_within_bound_returns_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    sent: list[dict] = []

    def _fake_send(req, *, timeout=30.0):  # noqa: ANN001
        sent.append(req)
        assert timeout == 10.0
        return {
            "ok": True,
            "state": "WAKE",
            "uptime_sec": 12.5,
            "version": "0.1.0",
        }

    monkeypatch.setattr(cli_mod, "_send_socket_request", _fake_send)

    rc = cli_mod.main(["daemon", "status"])
    assert rc == 0
    assert sent == [{"type": "status"}]

    out = capsys.readouterr().out
    assert "ok: True" in out
    assert "WAKE" in out


def test_status_timeout_prints_not_responding_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    def _fake_send_timeout(req, *, timeout=30.0):  # noqa: ANN001
        raise asyncio.TimeoutError()

    monkeypatch.setattr(cli_mod, "_send_socket_request", _fake_send_timeout)

    rc = cli_mod.main(["daemon", "status"])
    assert rc != 0

    err = capsys.readouterr().err
    assert "daemon not responding" in err


def test_status_socket_absent_prints_not_running_nonzero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(
        cli_mod, "_send_socket_request", lambda req, *, timeout=30.0: None,
    )

    rc = cli_mod.main(["daemon", "status"])
    assert rc != 0

    out = capsys.readouterr().out
    assert "daemon not running" in out


def test_real_launchctl_is_never_invoked(
    fake_state_dir: Path,
    captured_launchctl: list[list[str]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform, "system", lambda: "Darwin")

    rc = cli_mod.main(["daemon", "install", "--yes"])
    assert rc == 0

    assert all(
        str(cli_mod.LAUNCHD_TARGET).endswith("com.iai-mcp.daemon.plist")
        for _ in [0]
    )
    assert "home" in str(cli_mod.LAUNCHD_TARGET)
    assert len([c for c in captured_launchctl if c and c[0] == "launchctl"]) == 3
