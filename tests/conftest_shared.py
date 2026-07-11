"""NOT a standard pytest conftest — this file is NOT auto-discovered.
Tests import helpers directly:  from tests.conftest_shared import make_tmp_store

The project-wide autouse fixtures in tests/conftest.py (crypto passphrase +
autoflush) already apply to every test here, so store.insert() is synchronous
in tests without any extra setup.

All helpers use tmp_path, never ~/.iai-mcp/ or the live daemon socket.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from iai_mcp.store import MemoryStore


def short_socket_path(prefix: str = "iai-sock-") -> Path:
    """Return a short, bindable AF_UNIX socket path under a new mkdtemp directory.

    Creates a process-unique directory under the system temp root (TMPDIR →
    /var/folders/.../T/ on macOS) whose total path stays well under the platform
    sun_path limit (~104 bytes on macOS).  The caller is responsible for removing
    the directory after use; the companion pytest fixture ``short_socket`` handles
    teardown automatically.

    Use this function (rather than the fixture) only in non-fixture call sites —
    e.g. module-level helpers — where you must manage cleanup via try/finally or
    request.addfinalizer yourself.
    """
    d = Path(tempfile.mkdtemp(prefix=prefix))
    return d / "d.sock"


@pytest.fixture()
def short_socket():
    """Pytest fixture: yields a short, bindable AF_UNIX socket path with cleanup.

    Allocates a process-unique directory under the system temp root (TMPDIR →
    /var/folders/.../T/ on macOS) whose total path stays well under the platform
    sun_path limit (~104 bytes on macOS), avoiding ``OSError: AF_UNIX path too
    long`` on bind() when the pytest tmp_path is deep.

    The store / HOME remain hermetic under ``tmp_path``; only the socket endpoint
    moves to this short root.  The directory and socket file are removed on
    teardown via ``shutil.rmtree``, preventing accumulation in the system temp dir.

    Usage:
        def test_something(short_socket):
            from tests.conftest_shared import short_socket  # import into module
            srv.bind(str(short_socket))
            ...
    """
    d = Path(tempfile.mkdtemp(prefix="iai-sock-"))
    sock = d / "d.sock"
    yield sock
    shutil.rmtree(d, ignore_errors=True)


def make_tmp_store(tmp_path: Path) -> MemoryStore:
    """Construct an isolated MemoryStore rooted at tmp_path.

    Never touches ~/.iai-mcp/ or the live daemon.

    Usage in a test:
        def test_something(tmp_path):
            store = make_tmp_store(tmp_path)
            ...
    """
    store_root = tmp_path / "hippo"
    store_root.mkdir(parents=True, exist_ok=True)
    return MemoryStore(path=store_root)


def set_tmp_env(monkeypatch, tmp_path: Path) -> None:
    """Monkeypatch IAI_MCP_STORE and IAI_DAEMON_SOCKET_PATH to tmp paths.

    Prevents any code path that reads these env vars from touching the
    live store or the live daemon socket.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "hippo"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "test.sock"))
