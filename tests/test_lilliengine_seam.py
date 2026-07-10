"""The Hippo storage-connection seam routes to the Rust engine under the driver flag.

Asserts that ``_open_storage_connection`` returns the Rust ``engine.Connection``
when ``LILLI_STORAGE_DRIVER=lilli`` (Shape 1), that the connection is registered
so the daemon-down raw path resolves it and calls ``raw_conn(read_only=)`` on it,
that a read-only raw adapter blocks writes with the stdlib error, that the
default stdlib branch still returns a ``sqlite3.Connection``, and that the Python
``LilliBrainConnection`` stays intact as the parity reference.

Also asserts that a registry-hit read-only open uses a dedicated lock-free handle
(``engine.Connection.open_read_only``) rather than borrowing the live writer
connection's shared mutex — so a heavy read scan does not contend with the
writer during concurrent operation.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from iai_mcp.hippo._db import _open_storage_connection


@pytest.fixture
def lilli_driver(monkeypatch):
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")


def _path() -> str:
    return os.path.join(tempfile.mkdtemp(), "t.lilli")


def test_open_returns_rust_engine_connection(lilli_driver):
    from iai_mcp_native import engine

    conn, _owns = _open_storage_connection(_path(), embed_dim=384, cached_statements=128)
    assert type(conn) is engine.Connection


def test_routed_insert_select_roundtrip(lilli_driver):
    conn, _owns = _open_storage_connection(_path(), embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO t (id) VALUES ('a')")
    rows = conn.execute("SELECT id FROM t").fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] == "a"


def test_raw_conn_resolves_and_reads(lilli_driver):
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO t (id) VALUES ('a')")

    raw = get_lilli_raw_conn(path, read_only=True)
    assert raw is not None
    rows = raw.execute("SELECT id FROM t").fetchall()
    assert len(rows) == 1
    raw.close()


def test_raw_conn_read_only_blocks_write(lilli_driver):
    from iai_mcp.lillibrain.connection import get_lilli_raw_conn

    path = _path()
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")

    raw = get_lilli_raw_conn(path, read_only=True)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        raw.execute("INSERT INTO t (id) VALUES ('x')")
    raw.close()


def test_default_stdlib_branch_unchanged(monkeypatch):
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    path = os.path.join(tempfile.mkdtemp(), "b.sqlite3")
    conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    assert isinstance(conn, sqlite3.Connection)


def test_lillibrain_connection_stays_intact():
    from iai_mcp.lillibrain.connection import (
        LilliBrainConnection,
        LilliBrainRawConn,
    )

    assert hasattr(LilliBrainConnection, "raw_conn")
    assert hasattr(LilliBrainRawConn, "execute")


def test_get_lilli_raw_conn_ro_registry_hit_uses_dedicated_open_read_only(lilli_driver):
    """A registry-hit read-only open must use a dedicated lock-free engine handle.

    When the store's writer connection is live in the registry, calling
    get_lilli_raw_conn with read_only=True must open an independent connection
    via engine.Connection.open_read_only — NOT borrow the shared writer handle
    via conn.raw_conn(read_only=True).  That distinction eliminates mutex
    contention between a long read scan and concurrent writer operations.
    """
    from iai_mcp_native import engine as _engine

    import iai_mcp.lillibrain.connection as _conn_mod
    from iai_mcp.lillibrain.connection import (
        _OwnedEngineRawConn,
        get_lilli_raw_conn,
    )

    # Open a store so the path is registered (live writer in the registry).
    path = _path()
    writer_conn, _owns = _open_storage_connection(path, embed_dim=384, cached_statements=128)
    writer_conn.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    writer_conn.execute("INSERT INTO t (id) VALUES ('hello')")

    open_ro_called: list[str] = []
    writer_raw_conn_called_ro: list[bool] = []

    real_open_read_only = _engine.Connection.open_read_only

    def spy_open_read_only(p: str, flags: int):
        open_ro_called.append(p)
        return real_open_read_only(p, flags)

    # Intercept get_lilli_engine_conn to return a proxy that records raw_conn
    # calls — the Rust extension type's raw_conn is read-only and cannot be
    # monkey-patched directly, so the proxy wraps it at the Python layer.
    real_get_lilli_engine_conn = _conn_mod.get_lilli_engine_conn

    def spying_get_lilli_engine_conn(p: str):
        underlying = real_get_lilli_engine_conn(p)
        if underlying is None:
            return None

        class _WriterProxy:
            """Transparent proxy around the writer engine connection.

            Forwards every attribute access to the underlying Rust object and
            records whether raw_conn is invoked with read_only=True, which is
            the writer-mutex-sharing path this test must rule out.
            """

            def raw_conn(self, *, read_only: bool = False):
                if read_only:
                    writer_raw_conn_called_ro.append(True)
                return underlying.raw_conn(read_only=read_only)

            def __getattr__(self, name: str):
                return getattr(underlying, name)

        return _WriterProxy()

    with (
        patch.object(
            _engine.Connection, "open_read_only", staticmethod(spy_open_read_only)
        ),
        patch.object(_conn_mod, "get_lilli_engine_conn", spying_get_lilli_engine_conn),
    ):
        raw = get_lilli_raw_conn(path, read_only=True)

    assert raw is not None
    assert isinstance(raw, _OwnedEngineRawConn), (
        "registry-hit RO must return _OwnedEngineRawConn (dedicated open)"
    )
    # The dedicated open_read_only path must have been called.
    assert open_ro_called, (
        "engine.Connection.open_read_only was not called on registry-hit read_only=True"
    )
    # The writer's raw_conn must NOT have been called with read_only=True.
    assert not writer_raw_conn_called_ro, (
        "writer conn.raw_conn(read_only=True) was called — this shares the writer mutex"
    )
    # The returned adapter must still read committed rows correctly.
    rows = raw.execute("SELECT id FROM t").fetchall()
    assert len(rows) == 1
    assert rows[0]["id"] == "hello"
    raw.close()
