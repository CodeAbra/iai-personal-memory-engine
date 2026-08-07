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


def test_reload_under_a_foreign_platform_does_not_outlive_the_test() -> None:
    with mock.patch("platform.system", return_value="Windows"):
        importlib.reload(_antigravity_hooks)
    assert _antigravity_hooks._EXT == ".ps1"

    importlib.reload(_antigravity_hooks)
    import platform

    expected = ".ps1" if platform.system() == "Windows" else ".sh"
    assert _antigravity_hooks._EXT == expected
