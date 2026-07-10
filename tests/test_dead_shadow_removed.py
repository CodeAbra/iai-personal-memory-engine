from __future__ import annotations

from pathlib import Path

import iai_mcp


def _package_root() -> Path:
    return Path(iai_mcp.__file__).resolve().parent


def test_flat_daemon_shadow_absent() -> None:
    flat_shadow = _package_root() / "daemon.py"
    assert not flat_shadow.exists(), (
        "a flat iai_mcp/daemon.py re-appeared and shadows the daemon/ package"
    )


def test_daemon_import_resolves_to_package() -> None:
    import iai_mcp.daemon as d

    resolved = d.__file__.replace("\\", "/")
    assert "daemon/__init__.py" in resolved
