import argparse
import importlib
import json
import os
from pathlib import Path
from unittest import mock

import pytest

from iai_mcp.cli import _antigravity_hooks


@pytest.fixture(autouse=True)
def _restore_host_constants():
    # The module derives _EXT and _CMD_PREFIX at import time, so a reload under
    # a patched platform leaks into every later test in the session unless it is
    # undone here.
    yield
    importlib.reload(_antigravity_hooks)


@pytest.mark.parametrize(
    ("system", "ext", "prefix"),
    [
        ("Linux", ".sh", "bash "),
        ("Darwin", ".sh", "bash "),
        (
            "Windows",
            ".ps1",
            "powershell.exe -ExecutionPolicy Bypass -NoProfile -File ",
        ),
    ],
)
def test_hook_extension_and_launcher_follow_the_host(
    system: str, ext: str, prefix: str
) -> None:
    with mock.patch("platform.system", return_value=system):
        importlib.reload(_antigravity_hooks)

    assert _antigravity_hooks._EXT == ext
    assert _antigravity_hooks._CMD_PREFIX == prefix
    for _event, marker, _timeout in _antigravity_hooks._EVENT_WIRING:
        assert marker.endswith(ext)


@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_every_required_template_ships_for_the_platform(system: str) -> None:
    # The installer refuses to run when any required template is absent from
    # package data — this must hold for EVERY platform's script list, not
    # just the host's. (The constants alone cannot see a missing file.)
    from importlib import resources as _res

    with mock.patch("platform.system", return_value=system):
        importlib.reload(_antigravity_hooks)

    templates = _res.files("iai_mcp") / "_deploy" / "hooks"
    missing = [
        n + _antigravity_hooks._EXT
        for n in _antigravity_hooks._HOOK_SCRIPTS
        if not (templates / (n + _antigravity_hooks._EXT)).exists()
    ]
    assert not missing, f"{system}: installer would abort, templates missing: {missing}"


def test_windows_copy_set_includes_the_scanner() -> None:
    # The schtasks block runs against hooks_dir/iai-mcp-auto-scan-antigravity.ps1;
    # if the copy loop does not place it there, the installer reports a
    # scheduled scanner that does not exist.
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
    assert "iai-mcp-auto-scan-antigravity" in _antigravity_hooks._HOOK_SCRIPTS


def test_windows_install_end_to_end(tmp_path, monkeypatch) -> None:
    # Full installer run under a mocked Windows host: every template copies,
    # hooks.json wires both events, and the scanner reaches hooks_dir. The
    # schtasks step is exercised up to the subprocess boundary.
    calls: list[list[str]] = []

    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv(
            "IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "config")
        )
        monkeypatch.setenv("HOME", str(tmp_path))

        import subprocess as _sp

        def _fake_run(argv, **kwargs):
            calls.append(list(argv))
            return _sp.CompletedProcess(argv, 0)

        monkeypatch.setattr(_sp, "run", _fake_run)
        rc = _antigravity_hooks.install_antigravity_hooks()

    assert rc == 0, "Windows install must succeed with shipped package data"
    hooks_dir = tmp_path / "config" / "iai-hooks"
    for base in _antigravity_hooks._WIN_HOOK_SCRIPTS:
        assert (hooks_dir / (base + ".ps1")).exists(), base
    assert any("schtasks" in c[0] for c in calls), (
        "the Task Scheduler registration must actually be attempted"
    )


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


def test_windows_install_scopes_logon_trigger_to_user_sid(tmp_path, monkeypatch) -> None:
    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    calls: list[list[str]] = []

    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv(
            "IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "config")
        )
        monkeypatch.setenv("HOME", str(tmp_path))

        import subprocess as _sp

        class _WhoamiResult:
            stdout = '"DOMAIN\\alice","S-1-5-21-111-222-333-1001"\r\n'
            returncode = 0

        def _fake_run(argv, **kwargs):
            calls.append(list(argv))
            if argv[:1] == ["whoami"]:
                return _WhoamiResult()
            return _sp.CompletedProcess(argv, 0)

        monkeypatch.setattr(_sp, "run", _fake_run)
        rc = _antigravity_hooks.install_antigravity_hooks()

    assert rc == 0

    import defusedxml.ElementTree as ET

    xml_out = tmp_path / ".iai-mcp" / "iai-mcp-antigravity-scan.task.xml"
    text = xml_out.read_text(encoding="utf-16")
    root = ET.fromstring(text)
    ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
    user_id = root.find(f"{ns}Triggers/{ns}LogonTrigger/{ns}UserId")
    assert user_id is not None and user_id.text == "S-1-5-21-111-222-333-1001"
    assert "{USER_SID}" not in text


def test_windows_install_sid_resolution_failure_raises_and_leaves_prior_task(
    tmp_path, monkeypatch,
) -> None:
    from iai_mcp.cli import _daemon

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    calls: list[list[str]] = []

    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv(
            "IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "config")
        )
        monkeypatch.setenv("HOME", str(tmp_path))

        import subprocess as _sp

        def _fake_run(argv, **kwargs):
            calls.append(list(argv))
            if argv[:1] == ["whoami"]:
                raise _sp.CalledProcessError(1, argv)
            return _sp.CompletedProcess(argv, 0)

        monkeypatch.setattr(_sp, "run", _fake_run)

        with pytest.raises(RuntimeError):
            _antigravity_hooks.install_antigravity_hooks()

    xml_out = tmp_path / ".iai-mcp" / "iai-mcp-antigravity-scan.task.xml"
    assert not xml_out.exists(), (
        "a failed SID resolution must not write a new task XML"
    )
    assert not any(c[:2] == ["schtasks", "/Create"] for c in calls), (
        "schtasks /Create must never be attempted when SID resolution "
        "fails -- any previously-registered task must be left untouched"
    )


def test_target_all_isolates_antigravity_sid_failure_and_reports_aggregate_failure(
    tmp_path, monkeypatch, capsys,
) -> None:
    """A Windows SID-resolution failure at the antigravity step of
    `capture-hooks install --target all` must not abort the hosts that
    run after it (hermes, openclaw, the core Claude-Code hook block) --
    each host is isolated, and the overall rc reflects the failure
    instead of an uncaught exception (WR-05)."""
    from iai_mcp.cli import _capture, _daemon, _openclaw_mcp

    home = tmp_path / "home"
    (home / ".claude").mkdir(parents=True)
    (home / ".hermes").mkdir()
    (home / ".openclaw").mkdir()
    (home / ".gemini" / "config").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    wrapper = tmp_path / "index.js"
    wrapper.write_text("// fake memory wrapper\n", encoding="utf-8")
    monkeypatch.setenv("IAI_MCP_WRAPPER_PATH", str(wrapper))
    monkeypatch.setenv("IAI_MCP_TOOLS_WRAPPER_PATH", str(wrapper))
    monkeypatch.setattr(_openclaw_mcp.shutil, "which", lambda name: "/usr/bin/node")

    monkeypatch.setattr(_daemon, "os", _NTNamespacedOS())

    calls: list[list[str]] = []

    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)

        import subprocess as _sp

        def _fake_run(argv, **kwargs):
            calls.append(list(argv))
            if argv[:1] == ["whoami"]:
                raise _sp.CalledProcessError(1, argv)
            return _sp.CompletedProcess(argv, 0)

        monkeypatch.setattr(_sp, "run", _fake_run)

        rc = _capture.cmd_capture_hooks_install(argparse.Namespace(target="all"))

    out = capsys.readouterr().out

    assert rc != 0, "an antigravity SID-resolution failure must fail the overall install"
    # A bare "antigravity" substring also matches install-progress lines
    # printed before the failure; assert on the failure-specific report.
    assert any(
        line.startswith("antigravity: ") and "cannot resolve current user SID" in line
        for line in out.splitlines()
    ), out
    assert not any(c[:2] == ["schtasks", "/Create"] for c in calls), (
        "schtasks /Create must never be attempted when SID resolution fails"
    )
    xml_out = home / ".iai-mcp" / "iai-mcp-antigravity-scan.task.xml"
    assert not xml_out.exists()

    hermes_config = home / ".hermes" / "config.yaml"
    assert hermes_config.exists(), "hermes install must still run after antigravity fails"
    assert "iai-mcp-hermes" in hermes_config.read_text(encoding="utf-8")

    openclaw_config = home / ".openclaw" / "openclaw.json"
    assert openclaw_config.exists(), "openclaw install must still run after antigravity fails"

    settings = json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8"))
    stop_entries = settings.get("hooks", {}).get("Stop", [])
    assert any(
        "iai-mcp-session-capture.sh" in (h.get("command") or "")
        for e in stop_entries for h in (e.get("hooks") or [])
    ), "the core Claude-Code Stop hook must still install after antigravity fails"


def test_windows_install_omits_user_id_without_sid(tmp_path, monkeypatch) -> None:
    calls: list[list[str]] = []

    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv(
            "IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "config")
        )
        monkeypatch.setenv("HOME", str(tmp_path))

        import subprocess as _sp

        def _fake_run(argv, **kwargs):
            calls.append(list(argv))
            return _sp.CompletedProcess(argv, 0)

        monkeypatch.setattr(_sp, "run", _fake_run)
        rc = _antigravity_hooks.install_antigravity_hooks()

    assert rc == 0

    import defusedxml.ElementTree as ET

    xml_out = tmp_path / ".iai-mcp" / "iai-mcp-antigravity-scan.task.xml"
    text = xml_out.read_text(encoding="utf-16")
    root = ET.fromstring(text)
    ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
    assert root.find(f"{ns}Triggers/{ns}LogonTrigger/{ns}UserId") is None
    assert "{USER_SID}" not in text


def test_reload_under_a_foreign_platform_does_not_outlive_the_test() -> None:
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
    assert _antigravity_hooks._EXT == ".ps1"

    importlib.reload(_antigravity_hooks)
    import platform

    expected = ".ps1" if platform.system() == "Windows" else ".sh"
    assert _antigravity_hooks._EXT == expected


def test_posix_install_quotes_a_space_bearing_hooks_dir(tmp_path, monkeypatch) -> None:
    home = tmp_path / "John Doe"
    home.mkdir()
    with mock.patch("platform.system", return_value="Linux"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(home / "config"))
        monkeypatch.setenv("HOME", str(home))

        rc = _antigravity_hooks.install_antigravity_hooks()

    assert rc == 0
    hooks_dir = home / "config" / "iai-hooks"
    marker = "iai-mcp-antigravity-capture.sh"
    data = json.loads((home / "config" / "hooks.json").read_text(encoding="utf-8"))
    stop_cmd = data["iai-pme"]["Stop"][0]["hooks"][0]["command"]
    assert stop_cmd == f"bash '{hooks_dir / marker}'"


def test_windows_install_quotes_a_space_bearing_hooks_dir(tmp_path, monkeypatch) -> None:
    home = tmp_path / "John Doe"
    home.mkdir()
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(home / "config"))
        monkeypatch.setenv("HOME", str(home))

        import subprocess as _sp

        monkeypatch.setattr(_sp, "run", lambda argv, **kw: _sp.CompletedProcess(argv, 0))
        rc = _antigravity_hooks.install_antigravity_hooks()

    assert rc == 0
    hooks_dir = home / "config" / "iai-hooks"
    marker = "iai-mcp-antigravity-capture.ps1"
    prefix = "powershell.exe -ExecutionPolicy Bypass -NoProfile -File "
    data = json.loads((home / "config" / "hooks.json").read_text(encoding="utf-8"))
    stop_cmd = data["iai-pme"]["Stop"][0]["hooks"][0]["command"]
    assert stop_cmd == f"{prefix}'{hooks_dir / marker}'"


def test_install_and_status_builders_stay_byte_identical(tmp_path, monkeypatch, capsys) -> None:
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "config"))
        monkeypatch.setenv("HOME", str(tmp_path))

        import subprocess as _sp

        monkeypatch.setattr(_sp, "run", lambda argv, **kw: _sp.CompletedProcess(argv, 0))

        rc_install = _antigravity_hooks.install_antigravity_hooks()
        assert rc_install == 0
        capsys.readouterr()
        rc_status = _antigravity_hooks.status_antigravity_hooks()

    out = capsys.readouterr().out
    assert rc_status == 0, out
    assert "NOT WIRED" not in out, out


def test_heal_on_reinstall_rewrites_a_stale_command_in_place(tmp_path, monkeypatch) -> None:
    home = tmp_path / "John Doe"
    home.mkdir()
    config_dir = home / "config"
    config_dir.mkdir()
    with mock.patch("platform.system", return_value="Linux"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("HOME", str(home))

        hooks_dir = config_dir / "iai-hooks"
        marker = "iai-mcp-antigravity-capture.sh"
        stale_cmd = f"bash {hooks_dir / marker}"  # old, unquoted -- a space-bearing path split target
        hooks_json_path = config_dir / "hooks.json"
        hooks_json_path.write_text(
            json.dumps({
                "iai-pme": {
                    "enabled": True,
                    "Stop": [{"hooks": [{"type": "command", "command": stale_cmd, "timeout": 35}]}],
                }
            }),
            encoding="utf-8",
        )

        rc = _antigravity_hooks.install_antigravity_hooks()

    assert rc == 0
    data = json.loads(hooks_json_path.read_text(encoding="utf-8"))
    stop_entries = data["iai-pme"]["Stop"]
    assert len(stop_entries) == 1, stop_entries  # healed in place, never duplicated
    healed_cmd = stop_entries[0]["hooks"][0]["command"]
    assert healed_cmd == f"bash '{hooks_dir / marker}'"
    assert healed_cmd != stale_cmd


def test_status_reports_not_wired_for_a_stale_entry_then_wired_after_heal(
    tmp_path, monkeypatch, capsys,
) -> None:
    home = tmp_path / "John Doe"
    home.mkdir()
    config_dir = home / "config"
    config_dir.mkdir()
    with mock.patch("platform.system", return_value="Linux"):
        importlib.reload(_antigravity_hooks)
        monkeypatch.setenv("IAI_MCP_ANTIGRAVITY_CONFIG_DIR", str(config_dir))
        monkeypatch.setenv("HOME", str(home))

        hooks_dir = config_dir / "iai-hooks"
        marker = "iai-mcp-antigravity-capture.sh"
        stale_cmd = f"bash {hooks_dir / marker}"
        hooks_json_path = config_dir / "hooks.json"
        hooks_json_path.write_text(
            json.dumps({
                "iai-pme": {
                    "enabled": True,
                    "Stop": [{"hooks": [{"type": "command", "command": stale_cmd, "timeout": 35}]}],
                }
            }),
            encoding="utf-8",
        )

        capsys.readouterr()
        _antigravity_hooks.status_antigravity_hooks()
        out_before = capsys.readouterr().out
        assert any(
            line.startswith("Antigravity hooks.json Stop") and "NOT WIRED" in line
            for line in out_before.splitlines()
        ), out_before

        rc_install = _antigravity_hooks.install_antigravity_hooks()
        assert rc_install == 0

        capsys.readouterr()
        rc_after = _antigravity_hooks.status_antigravity_hooks()
        out_after = capsys.readouterr().out

    assert rc_after == 0, out_after
    assert any(
        line.startswith("Antigravity hooks.json Stop") and "NOT WIRED" not in line
        for line in out_after.splitlines()
    ), out_after
