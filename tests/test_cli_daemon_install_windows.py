"""Windows daemon install: a per-user Task Scheduler task (logon trigger,
restart-on-failure) registered from a rendered XML, started through a .cmd
wrapper that carries the environment and stream redirects the XML format
cannot express. All platform-neutral: rendering and command wiring are
exercised with schtasks mocked, so the suite runs anywhere."""
from __future__ import annotations

import os

import defusedxml.ElementTree as ET
import pytest


def test_windows_task_xml_renders_and_parses(monkeypatch):
    from iai_mcp import cli

    rendered = cli._render_windows_task_xml()
    root = ET.fromstring(rendered)
    ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"

    cmd = root.find(f"{ns}Actions/{ns}Exec/{ns}Command")
    assert cmd is not None and str(cli.WINDOWS_START_CMD) == cmd.text
    assert "{START_CMD}" not in rendered and "{WORK_DIR}" not in rendered

    assert root.find(f"{ns}Triggers/{ns}LogonTrigger") is not None
    restart = root.find(f"{ns}Settings/{ns}RestartOnFailure/{ns}Count")
    assert restart is not None and int(restart.text) >= 1
    limit = root.find(f"{ns}Settings/{ns}ExecutionTimeLimit")
    assert limit is not None and limit.text == "PT0S", (
        "a long-lived daemon must not be killed by the default 3-day limit"
    )

    # Real test host is non-Windows: no <UserId> element, no leftover token.
    user_id = root.find(f"{ns}Triggers/{ns}LogonTrigger/{ns}UserId")
    assert user_id is None
    assert "{USER_SID}" not in rendered


class _NTNamespacedOS:
    """Proxies the real os module but reports os.name == 'nt'.

    A blanket monkeypatch of the real os module's .name attribute breaks
    pathlib (WindowsPath cannot instantiate on a non-Windows interpreter);
    this proxy is swapped in only as _daemon.py's own module-level `os`
    binding, leaving pathlib's own os reference untouched.
    """

    name = "nt"

    def __getattr__(self, item):
        return getattr(os, item)


def test_windows_task_xml_scopes_logon_trigger_to_user_sid(monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    class _Result:
        stdout = '"DOMAIN\\alice","S-1-5-21-111-222-333-1001"\r\n'
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Result())

    rendered = cli._render_windows_task_xml()
    root = ET.fromstring(rendered)
    ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
    user_id = root.find(f"{ns}Triggers/{ns}LogonTrigger/{ns}UserId")
    assert user_id is not None and user_id.text == "S-1-5-21-111-222-333-1001"
    assert "{USER_SID}" not in rendered


def test_windows_sid_resolution_failure_raises(monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    def _boom(*a, **k):
        raise OSError("whoami not found on PATH")

    monkeypatch.setattr(cli.subprocess, "run", _boom)

    with pytest.raises(RuntimeError):
        cli._render_windows_task_xml()


def test_windows_sid_resolution_rejects_malformed_shape(monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    class _Result:
        # No comma-separated SID field: a locale/banner mismatch, not a
        # whoami failure. Non-emptiness alone would accept this as a SID.
        stdout = "some unexpected banner line\r\n"
        returncode = 0

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Result())

    with pytest.raises(RuntimeError):
        cli._render_windows_task_xml()


def test_windows_sid_resolution_wraps_decode_failure(monkeypatch):
    from iai_mcp import cli
    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    def _boom(*a, **k):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

    monkeypatch.setattr(cli.subprocess, "run", _boom)

    with pytest.raises(RuntimeError):
        cli._render_windows_task_xml()


def test_windows_start_cmd_wraps_the_daemon(monkeypatch):
    fake_python = r"C:\venv\Scripts\python.exe"
    monkeypatch.setattr("iai_mcp.cli.sys.executable", fake_python)
    from iai_mcp import cli

    cmd = cli._render_windows_start_cmd()
    assert fake_python in cmd
    assert "-m iai_mcp.daemon" in cmd
    assert "IAI_MCP_STORE" in cmd, "the wrapper carries the env the XML cannot"
    assert "2>>" in cmd, "stderr must land in a log file, not vanish"


def test_install_dry_run_prints_both_artifacts(capsys):
    from iai_mcp.cli import _install_windows_task

    rc = _install_windows_task(dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "daemon-start.cmd" in out
    assert "-m iai_mcp.daemon" in out
    assert "iai-mcp-daemon" in out
    assert "LogonTrigger" in out


def test_install_registers_and_runs_the_task(monkeypatch, tmp_path):
    from iai_mcp import cli

    monkeypatch.setattr(cli, "WINDOWS_START_CMD", tmp_path / "daemon-start.cmd")
    monkeypatch.setattr(cli, "WINDOWS_TASK_XML", tmp_path / "task.xml")
    monkeypatch.setattr(cli, "_ensure_crypto_key_present", lambda: None)

    calls: list[list[str]] = []

    class _Result:
        returncode = 0
        stderr = ""
        stdout = ""

    def _fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        return _Result()

    monkeypatch.setattr(cli.subprocess, "run", _fake_run)

    rc = cli._install_windows_task(dry_run=False)
    assert rc == 0

    assert (tmp_path / "daemon-start.cmd").exists()
    xml_path = tmp_path / "task.xml"
    assert xml_path.exists()
    # Task Scheduler expects its XML in UTF-16 (its own export encoding).
    ET.fromstring(xml_path.read_text(encoding="utf-16"))

    create = next(c for c in calls if "/Create" in c)
    assert create[:2] == ["schtasks", "/Create"]
    assert "/F" in create and cli.WINDOWS_TASK_NAME in create
    assert any("/Run" in c for c in calls), "install must also start the task"


def test_install_surfaces_schtasks_failure(monkeypatch, tmp_path, capsys):
    from iai_mcp import cli

    monkeypatch.setattr(cli, "WINDOWS_START_CMD", tmp_path / "daemon-start.cmd")
    monkeypatch.setattr(cli, "WINDOWS_TASK_XML", tmp_path / "task.xml")
    monkeypatch.setattr(cli, "_ensure_crypto_key_present", lambda: None)

    class _Fail:
        returncode = 1
        stderr = "ERROR: Access is denied."
        stdout = ""

    monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: _Fail())

    rc = cli._install_windows_task(dry_run=False)
    assert rc == 1
    assert "Access is denied" in capsys.readouterr().err
