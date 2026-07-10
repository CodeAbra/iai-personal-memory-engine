from __future__ import annotations

import argparse
import subprocess

import pytest


def _make_install_args(**kwargs) -> argparse.Namespace:
    defaults = dict(dry_run=True, yes=True)
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_install_uses_sys_executable_macos(monkeypatch):
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert f"<string>{fake_python}</string>" in rendered, (
        f"plist did not substitute sys.executable; rendered text:\n{rendered[:500]}"
    )
    assert "<string>/usr/local/bin/python3</string>" not in rendered, (
        "plist still contains the unsubstituted /usr/local/bin/python3 placeholder"
    )


def test_install_uses_sys_executable_linux(monkeypatch):
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_systemd_unit

    rendered = _render_systemd_unit()
    assert f"{fake_python} -m iai_mcp.daemon" in rendered or (
        f"{fake_python}" in rendered and "iai_mcp.daemon" in rendered
    ), f"systemd unit did not substitute sys.executable; rendered:\n{rendered[:500]}"
    assert "/usr/bin/python3 -m iai_mcp.daemon" not in rendered, (
        "systemd unit still contains the unsubstituted /usr/bin/python3 placeholder"
    )


def test_plist_keepalive_resurrects_after_kill(monkeypatch):
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert "<key>Crashed</key>" in rendered
    # SuccessfulExit=false is REQUIRED: launchd does not count SIGKILL
    # (jetsam/OOM, force-kill) as Crashed, so without it a killed daemon
    # stays down until a manual start. Graceful paths (hibernation, the
    # SIGTERM stop handler) exit 0 = successful, so this key does NOT
    # respawn a deliberately stopped daemon.
    assert "<key>SuccessfulExit</key>" in rendered
    successful_exit_idx = rendered.index("<key>SuccessfulExit</key>")
    tail = rendered[successful_exit_idx:successful_exit_idx + 80]
    assert "<false/>" in tail, (
        "SuccessfulExit must be FALSE (respawn only after an unsuccessful "
        "death); TRUE would respawn on clean exits and break hibernation."
    )


def test_plist_lifecycle_env_vars_present(monkeypatch):
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert "<key>LIFECYCLE_DROWSY_AFTER_SEC</key>" in rendered
    assert "<key>LIFECYCLE_SLEEP_HEARTBEAT_IDLE_SEC</key>" in rendered
    assert "<key>LIFECYCLE_HIBERNATE_AFTER_SEC</key>" in rendered
    assert "<key>IAI_MCP_SLEEP_QUARANTINE_TTL_HOURS</key>" in rendered


def test_plist_legacy_env_vars_removed(monkeypatch):
    fake_python = "/path/to/venv/bin/python3"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp.cli import _render_launchd_plist

    rendered = _render_launchd_plist()
    assert "<key>IAI_MCP_RSS_RESTART_THRESHOLD_MB</key>" not in rendered, (
        "RSS-watchdog is retired; env var must be gone "
        "from the plist."
    )
    assert "<key>IAI_DAEMON_IDLE_SHUTDOWN_SECS</key>" not in rendered
    assert "<key>IAI_MCP_SKIP_STARTUP_OPTIMIZE</key>" not in rendered


@pytest.mark.xfail(
    reason=(
        "psutil is a mandatory runtime dependency, so the install never needs to "
        "probe for its absence; this contract stays xfail-strict so an accidental "
        "probe implementation fails loudly and forces removal of this marker."
    ),
    strict=True,
)
def test_install_warns_when_sys_executable_lacks_psutil(
    monkeypatch, capsys, tmp_path,
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))

    real_run = subprocess.run

    def _fake_run(cmd, **kwargs):
        if (
            isinstance(cmd, list)
            and len(cmd) >= 3
            and cmd[1] == "-c"
            and cmd[2] == "import psutil"
        ):
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        return real_run(cmd, **kwargs)

    monkeypatch.setattr("subprocess.run", _fake_run)

    from iai_mcp.cli import cmd_daemon_install

    rc = cmd_daemon_install(_make_install_args(dry_run=True, yes=True))
    err = capsys.readouterr().err
    assert rc == 0, f"install must NOT fail on missing psutil; got rc={rc}"
    err_lower = err.lower()
    assert "psutil" in err_lower
    assert "iai-mcp daemon install" in err_lower
    assert "re-run" in err_lower
