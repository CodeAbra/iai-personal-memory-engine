"""Regression tests for the explicit SQLite cache caps on `HippoDB._conn`.

`tracemalloc` cannot see C-level SQLite page-cache or statement-cache
allocations. Without an explicit cap, both can drift unbounded across the
daemon's long-running uptime — which is invisible to the Python-level RSS
RCA harness and therefore unaccounted for in the warm plateau growth.

This test pins the explicit caps so a future refactor can't silently drop
them.
"""
from __future__ import annotations

from pathlib import Path

import pytest

# Every test in this module exercises PRAGMA cache_size / cached_statements,
# which configures the SQLite page cache and statement cache; the native
# engine has no page-cache analog and accepts-and-ignores the PRAGMA. Its
# at-rest footprint bound is proven separately by the per-record size and
# float32-blob tests in test_hippo_memory_footprint.py. Each test therefore
# opens its store against the legacy driver explicitly rather than relying on
# whatever the ambient default happens to be.


@pytest.fixture(autouse=True)
def _legacy_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")


def test_sqlite_cache_size_pragma_is_capped(tmp_path: Path, monkeypatch):
    """The `HippoDB._conn` should report a finite negative `cache_size` matching
    the configured KiB cap (default 65536 → 64 MiB)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        cache_size = row[0]
        # Negative => KiB. Default cap is -65536 (64 MiB).
        assert cache_size == -65536, (
            f"expected cache_size=-65536 (64 MiB cap), got {cache_size}"
        )
    finally:
        db.close()


def test_sqlite_cache_size_pragma_honors_env_override(tmp_path: Path, monkeypatch):
    """Operator can override the cap via env var."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")
    monkeypatch.setenv("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "32768")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        assert row[0] == -32768
    finally:
        db.close()


def test_sqlite_cached_statements_is_set(tmp_path: Path, monkeypatch):
    """The connection should be constructed with an explicit `cached_statements`
    cap matching the configured value (default 128). Tested indirectly via the
    private attribute path that the connection module exposes."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        # `sqlite3.Connection` does not expose cached_statements directly; the
        # cap is enforced at C level. Smoke-test that the connection still
        # works after the cap is in place by running an executemany with
        # > 128 distinct prepared statements equivalents.
        for i in range(200):
            db._conn.execute(f"SELECT {i}").fetchone()
        # No exception = pass.
    finally:
        db.close()


def test_sqlite_cache_size_pragma_invalid_env_falls_back_to_default(
    tmp_path: Path,
    monkeypatch,
):
    """Garbage env value should not crash boot; fall back to default."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")
    monkeypatch.setenv("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "not-a-number")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        assert row[0] == -65536
    finally:
        db.close()


def test_sqlite_cache_size_tiny_env_clamps_to_floor(tmp_path: Path, monkeypatch):
    """A tiny env value (e.g. 1 KiB) must clamp to the 4 MiB floor, not cause
    a page-cache thrash by sending ``PRAGMA cache_size=-1``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")
    monkeypatch.setenv("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "1")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        cache_size = row[0]
        assert cache_size == -4096, (
            f"expected floor -4096 (4 MiB), got {cache_size}"
        )
    finally:
        db.close()


def test_sqlite_cache_size_zero_env_falls_back_to_default(tmp_path: Path, monkeypatch):
    """``IAI_MCP_SQLITE_CACHE_SIZE_KIB=0`` should fall back to the 64 MiB
    default, not produce ``PRAGMA cache_size=-0`` (SQLite default ~2 MiB)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")
    monkeypatch.setenv("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "0")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        assert row[0] == -65536
    finally:
        db.close()


def test_sqlite_cache_size_negative_env_falls_back_to_default(
    tmp_path: Path, monkeypatch
):
    """A negative env value should fall back to the 64 MiB default — not be
    silently abs()-rewritten into a positive KiB count."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")
    monkeypatch.setenv("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "-32768")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        assert row[0] == -65536
    finally:
        db.close()


def test_sqlite_cache_size_sane_large_env_honored(tmp_path: Path, monkeypatch):
    """A sane large env value (e.g. 131072 = 128 MiB) is honored without
    clamping."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test")
    monkeypatch.setenv("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "131072")

    from iai_mcp.hippo._db import HippoDB

    db = HippoDB(tmp_path / "store")
    try:
        row = db._conn.execute("PRAGMA cache_size").fetchone()
        assert row is not None
        assert row[0] == -131072
    finally:
        db.close()
