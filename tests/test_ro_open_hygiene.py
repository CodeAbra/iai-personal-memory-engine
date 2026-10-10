"""Two throwaway RO seams (keyset-snapshot scan, no-flock recency read) must
open the stdlib file at the file layer via a `file:...?mode=ro` URI, not via
a post-hoc PRAGMA that can be swallowed on failure. A lilli-format store
short-circuits past the stdlib fallback entirely (`open_store_conn` returns
the engine conn), so the defect class is stdlib-only; every test here forces
the stdlib driver.
"""
from __future__ import annotations

import pytest

from iai_mcp import errors
from iai_mcp.hippo import open_store_conn
from iai_mcp.hippo._recall import _no_flock_recency_rows_from_store
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.capture import capture_turn


def _select_driver(driver: str, monkeypatch) -> None:
    if driver == "lilli":
        try:
            import iai_mcp_native  # noqa: F401, PLC0415
        except ImportError:
            pytest.skip("iai_mcp_native not built")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")
    else:
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")


def _seed_records(store: MemoryStore, n: int) -> None:
    for i in range(n):
        capture_turn(
            store, cue="", text=f"ro open hygiene record number {i} unique",
            tier="episodic", session_id="s", role="user",
        )
    flush_record_buffer(store)


class _ConnectSpy:
    """Records positional args + kwargs of every call, then delegates."""

    def __init__(self, real_connect):
        self._real = real_connect
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._real(*args, **kwargs)


def test_open_keyset_snapshot_opens_mode_ro(tmp_path, monkeypatch):
    _select_driver("stdlib", monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 3)
    db_path = getattr(store.db, "_db_path", None)
    assert db_path, "stdlib store must expose _db_path"
    assert open_store_conn(db_path, read_only=True) is None, (
        "test must exercise the bare-connect stdlib fallback"
    )

    import iai_mcp._sqlite_stdlib as stdlib_mod
    spy = _ConnectSpy(stdlib_mod.connect)
    monkeypatch.setattr(stdlib_mod, "connect", spy)

    conn = store._open_keyset_snapshot(db_path)
    try:
        assert spy.calls, "connect must have been called"
        args, kwargs = spy.calls[0]
        assert args[0] == f"file:{db_path}?mode=ro"
        assert kwargs.get("uri") is True

        conn.execute("PRAGMA query_only=OFF")
        with pytest.raises(errors.OperationalError, match="readonly"):
            conn.execute("CREATE TABLE ro_hygiene_probe (x INTEGER)")
    finally:
        conn.close()
    store.close()


def test_no_flock_recency_opens_mode_ro(tmp_path, monkeypatch):
    _select_driver("stdlib", monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 3)
    db_path = tmp_path / "hippo" / "brain.sqlite3"
    assert open_store_conn(db_path, read_only=True) is None, (
        "test must exercise the bare-connect stdlib fallback"
    )
    store.close()

    import iai_mcp._sqlite_stdlib as stdlib_mod
    spy = _ConnectSpy(stdlib_mod.connect)
    monkeypatch.setattr(stdlib_mod, "connect", spy)

    rows = _no_flock_recency_rows_from_store(db_path)
    assert rows, "helper must still return recency rows"
    assert spy.calls, "connect must have been called"
    args, kwargs = spy.calls[0]
    assert args[0] == f"file:{db_path}?mode=ro"
    assert kwargs.get("uri") is True


def test_open_keyset_snapshot_does_not_swallow_pragma_failure(tmp_path, monkeypatch):
    _select_driver("stdlib", monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 3)
    db_path = getattr(store.db, "_db_path", None)
    assert db_path
    assert open_store_conn(db_path, read_only=True) is None

    import iai_mcp._sqlite_stdlib as stdlib_mod
    real_connect = stdlib_mod.connect

    class _PoisonedConn:
        def __init__(self, real):
            self._real = real

        def execute(self, sql, *a):
            if sql.strip().upper().startswith("PRAGMA QUERY_ONLY=ON"):
                raise errors.OperationalError("forced query_only failure")
            return self._real.execute(sql, *a)

        def __getattr__(self, name):
            return getattr(self._real, name)

        def __setattr__(self, name, value):
            if name == "_real":
                object.__setattr__(self, name, value)
            else:
                setattr(self._real, name, value)

    def _poisoned_connect(*args, **kwargs):
        return _PoisonedConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(stdlib_mod, "connect", _poisoned_connect)

    with pytest.raises(errors.OperationalError, match="forced query_only failure"):
        store._open_keyset_snapshot(db_path)
    store.close()


def test_lilli_iter_record_columns_completes(tmp_path, monkeypatch):
    _select_driver("lilli", monkeypatch)
    store = MemoryStore(path=tmp_path)
    _seed_records(store, 5)
    db_path = getattr(store.db, "_db_path", None)
    assert db_path
    assert open_store_conn(db_path, read_only=True) is not None, (
        "lilli-format store must short-circuit past the stdlib fallback"
    )

    rows = list(store.iter_record_columns(["id"], batch_size=2))
    assert len(rows) == 5
    store.close()
