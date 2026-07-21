"""Windows daemon install: a per-user Task Scheduler task (logon trigger,
restart-on-failure) registered from a rendered XML, started through a .cmd
wrapper that carries the environment and stream redirects the XML format
cannot express. All platform-neutral: rendering and command wiring are
exercised with schtasks mocked, so the suite runs anywhere."""
from __future__ import annotations

import defusedxml.ElementTree as ET


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
