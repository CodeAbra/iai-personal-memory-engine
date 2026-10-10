"""End-to-end round-trip: a REAL `iai_tools_execd` daemon, spawned by
`tools_exec_client`, driven through its public `run`/`fetch` API.

On the local dev box this exercises the AF_UNIX path (proving the harness
end-to-end); on windows-latest CI the exact same test exercises the
TCP-loopback + mandatory token-handshake path -- the platform branch lives
entirely inside `tools_exec_client`/`iai_tools_execd`, not in this test.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def _e2e_execd_bin() -> str:
    """Resolves the real `iai_tools_execd` binary via the env override only
    -- never an un-suffixed path guess, which would silently self-skip on
    Windows and leave a "green" job that never actually ran anything."""
    env = os.environ.get("IAI_TOOLS_EXECD_BIN")
    if not env:
        pytest.skip(
            "IAI_TOOLS_EXECD_BIN not set; build the daemon and export the path: "
            "cargo build -p iai_mcp_tools_core --bins"
        )
    assert Path(env).exists(), f"IAI_TOOLS_EXECD_BIN={env} does not point at an existing file"
    return env


_TEST_IDLE_SECS = 1.0


@pytest.fixture
def e2e_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "iai-tools-root"
    # AF_UNIX's sun_path is capped at ~104 bytes on macOS -- pytest's own
    # tmp_path nests several directories deep and overflows that, so on
    # POSIX the socket/port/token files live in their own short-path
    # directory directly under /tmp. That directory does not exist on
    # Windows runners (no AF_UNIX length limit there either), so the
    # Windows branch uses the default temp root instead.
    sock_dir = Path(
        tempfile.mkdtemp(
            prefix="iai-tools-e2e-",
            dir=None if sys.platform == "win32" else "/tmp",
        )
    )
    sock_path = sock_dir / ".execd.sock"
    monkeypatch.setenv("IAI_TOOLS_ROOT", str(root))
    monkeypatch.setenv("IAI_TOOLS_SOCKET_PATH", str(sock_path))
    monkeypatch.setenv("IAI_TOOLS_EXECD_BIN", _e2e_execd_bin())
    monkeypatch.setenv("IAI_TOOLS_IDLE_SECS", str(_TEST_IDLE_SECS))
    try:
        yield {"root": root, "sock_path": sock_path}
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


def test_execd_roundtrip_run_and_fetch(e2e_env: dict) -> None:
    from iai_mcp import tools_exec_client

    result = tools_exec_client.run("echo", ["hello-execd-roundtrip"])

    assert result["run_id"]
    assert result["state"] == "exited:0"
    assert "hello-execd-roundtrip" in result["bounded_tail"]["stdout"]

    fetched = tools_exec_client.fetch(result["run_id"])
    assert fetched["run_id"] == result["run_id"]


def test_execd_roundtrip_batch_run(e2e_env: dict) -> None:
    from iai_mcp import tools_exec_client

    results = tools_exec_client.batch_run(
        [
            {"command": "echo", "args": ["batch-one"]},
            {"command": "echo", "args": ["batch-two"]},
        ]
    )

    assert len(results) == 2
    assert "batch-one" in results[0]["bounded_tail"]["stdout"]
    assert "batch-two" in results[1]["bounded_tail"]["stdout"]
