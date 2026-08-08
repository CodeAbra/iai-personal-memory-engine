import importlib
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


def test_reload_under_a_foreign_platform_does_not_outlive_the_test() -> None:
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
    assert _antigravity_hooks._EXT == ".ps1"

    importlib.reload(_antigravity_hooks)
    import platform

    expected = ".ps1" if platform.system() == "Windows" else ".sh"
    assert _antigravity_hooks._EXT == expected
