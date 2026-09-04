"""A genuinely fresh process creates a native-engine store by default.

Every guard here spawns a real subprocess with a scrubbed, explicit
environment -- an in-process monkeypatch cannot prove what a fresh process
resolves to, because this suite's own autouse fixture snapshots and restores
``LILLI_STORAGE_DRIVER`` around every test, and a process launched inside
pytest still carries pytest's own ambient environment. The two guards below
are kept as separate test functions so neither the native default nor the
legacy override can silently stop being exercised while the other keeps
passing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from iai_mcp.hippo._db import DEFAULT_STORAGE_DRIVER, _resolve_effective_driver
from iai_mcp.lillibrain.constants import DB_MAGIC

_SQLITE_MAGIC = b"SQLite format 3\x00"
_REPO_SRC = Path(__file__).resolve().parent.parent / "src"


def _scrubbed_env(*, home: Path, store_root: Path, driver: str | None) -> dict[str, str]:
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home),
        "PYTHONPATH": str(_REPO_SRC),
        "IAI_MCP_STORE": str(store_root),
        "IAI_MCP_CRYPTO_PASSPHRASE": "scrubbed-subprocess-test-passphrase",
        "LILLI_FSYNC_MODE": "fast",
    }
    if driver is not None:
        env["LILLI_STORAGE_DRIVER"] = driver
    return env


def _run_scrubbed(code: str, *, home: Path, store_root: Path, driver: str | None):
    home.mkdir(parents=True, exist_ok=True)
    env = _scrubbed_env(home=home, store_root=store_root, driver=driver)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_scrubbed_subprocess_resolves_unset_env_to_native_default(tmp_path: Path) -> None:
    """A subprocess with the driver variable absent resolves a not-yet-existing
    backing path to the native engine (ENGINE-01)."""
    store_root = tmp_path / "store"
    db_path = store_root / "hippo" / "brain.sqlite3"
    code = (
        "from iai_mcp.hippo._db import _resolve_effective_driver\n"
        f"print(_resolve_effective_driver({str(db_path)!r}))\n"
    )
    result = _run_scrubbed(code, home=tmp_path / "home", store_root=store_root, driver=None)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "lilli"


def test_scrubbed_subprocess_unset_env_creates_native_store(tmp_path: Path) -> None:
    """A subprocess with the driver variable absent, told to create a store,
    leaves a backing file whose header is the native engine magic (ENGINE-01).
    """
    store_root = tmp_path / "store"
    code = "from iai_mcp.store import MemoryStore\ns = MemoryStore()\ns.close()\n"
    result = _run_scrubbed(code, home=tmp_path / "home", store_root=store_root, driver=None)
    assert result.returncode == 0, result.stderr

    db_path = store_root / "hippo" / "brain.sqlite3"
    header = db_path.read_bytes()[: len(DB_MAGIC)]
    assert header == DB_MAGIC


def test_scrubbed_subprocess_legacy_override_creates_stdlib_store(tmp_path: Path) -> None:
    """The same subprocess, with the driver variable set to the legacy name,
    still leaves a backing file whose header is the SQLite magic -- the
    override survives the flip (ENGINE-02)."""
    store_root = tmp_path / "store"
    code = "from iai_mcp.store import MemoryStore\ns = MemoryStore()\ns.close()\n"
    result = _run_scrubbed(
        code, home=tmp_path / "home", store_root=store_root, driver="stdlib"
    )
    assert result.returncode == 0, result.stderr

    db_path = store_root / "hippo" / "brain.sqlite3"
    header = db_path.read_bytes()[: len(_SQLITE_MAGIC)]
    assert header == _SQLITE_MAGIC


def test_default_constant_resolves_for_fresh_path_when_env_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-process: with the variable deleted, resolution for a not-yet-existing
    path returns the exported constant, and the constant names the native
    engine."""
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    db_path = tmp_path / "hippo" / "brain.sqlite3"  # not created
    assert DEFAULT_STORAGE_DRIVER == "lilli"
    assert _resolve_effective_driver(str(db_path)) == DEFAULT_STORAGE_DRIVER
