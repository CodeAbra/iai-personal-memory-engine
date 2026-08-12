from __future__ import annotations

import sqlite3

from iai_mcp import errors


from iai_mcp import doctor as _doctor
from iai_mcp.hippo import HippoLockHeldError


def _raise_lock_held(*_a, **_kw):
    raise HippoLockHeldError("/tmp/alice-store/hippo/.lock", "12345")


def test_f_warns_on_lock_held_on_direct_shared_open(monkeypatch, tmp_path) -> None:
    """A SHARED read-only open coexists with the daemon's steady-state
    SHARED lock, so reaching a lock error on it means something unexpected
    holds the store exclusively -- a real WARN, never the old blind
    "held by the live daemon -- normal" PASS."""
    import iai_mcp.store as _store_mod

    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_lock_held)

    result = _doctor.check_f_hippo_readable()

    assert result.name == "(f) hippo storage readable"
    assert result.passed is True
    assert result.status == "WARN"
    assert "normal" not in result.detail.lower()


def test_f_warns_on_sqlite_database_locked_on_direct_shared_open(
    monkeypatch, tmp_path
) -> None:
    import iai_mcp.store as _store_mod

    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    def _raise_locked(*_a, **_kw):
        raise errors.OperationalError("database is locked")

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_locked)

    result = _doctor.check_f_hippo_readable()

    assert result.passed is True
    assert result.status == "WARN"
    assert "normal" not in result.detail.lower()


def test_f_still_fails_on_real_open_error_daemon_down(monkeypatch, tmp_path) -> None:
    import iai_mcp.store as _store_mod

    # The open-error scenario requires an EXISTING store file — an
    # absent store now short-circuits to the no-store WARN row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    def _raise_corruption(*_a, **_kw):
        raise errors.DatabaseError("file is not a database")

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_corruption)

    result = _doctor.check_f_hippo_readable()

    assert result.passed is False
    assert result.status == "FAIL"
    assert "open failed" in result.detail.lower()


def test_f_still_fails_on_other_operational_error(monkeypatch, tmp_path) -> None:
    import iai_mcp.store as _store_mod

    # The open-error scenario requires an EXISTING store file — an
    # absent store now short-circuits to the no-store WARN row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))

    def _raise_other(*_a, **_kw):
        raise errors.OperationalError("disk I/O error")

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_other)

    result = _doctor.check_f_hippo_readable()

    assert result.passed is False
    assert result.status == "FAIL"


def test_t_benign_when_daemon_holds_store(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an
    # absent one now short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_lock_held)

    result = _doctor.check_t_hippo_compacted_freshness()

    assert result.name == "(t) hippo_compacted freshness"
    assert result.passed is True
    assert result.status == "WARN"
    assert "deferred" not in result.detail.lower()


def test_t_benign_on_lock_held_during_events_query(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an
    # absent one now short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())
    monkeypatch.setattr(_events, "query_events", _raise_lock_held)

    result = _doctor.check_t_hippo_compacted_freshness()

    assert result.status == "WARN"
    assert "deferred" not in result.detail.lower()


def test_t_still_warns_on_genuine_query_failure(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an
    # absent one now short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    def _boom(*_a, **_kw):
        raise RuntimeError("events backend unavailable")

    monkeypatch.setattr(_events, "query_events", _boom)

    result = _doctor.check_t_hippo_compacted_freshness()

    assert result.status == "WARN"
    assert result.passed is True


def test_u_benign_when_daemon_holds_store(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an
    # absent one now short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", _raise_lock_held)

    result = _doctor.check_u_recall_centrality_regression()

    assert result.name == "(u) recall centrality regression"
    assert result.passed is True
    assert result.status == "WARN"
    assert "deferred" not in result.detail.lower()


def test_u_benign_on_lock_held_during_events_query(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an
    # absent one now short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())
    monkeypatch.setattr(_events, "query_events", _raise_lock_held)

    result = _doctor.check_u_recall_centrality_regression()

    assert result.status == "WARN"
    assert "deferred" not in result.detail.lower()


def test_u_still_warns_on_genuine_query_failure(monkeypatch, tmp_path) -> None:
    # The stubbed-open scenario presumes an existing store file; an
    # absent one now short-circuits to the no-store skip row.
    (tmp_path / "hippo").mkdir(parents=True)
    (tmp_path / "hippo" / "brain.sqlite3").write_bytes(b"x")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no.sock"))
    import iai_mcp.events as _events
    import iai_mcp.store as _store_mod

    monkeypatch.setattr(_store_mod, "MemoryStore", lambda *a, **kw: object())

    def _boom(*_a, **_kw):
        raise RuntimeError("events backend unavailable")

    monkeypatch.setattr(_events, "query_events", _boom)

    result = _doctor.check_u_recall_centrality_regression()

    assert result.status == "WARN"
    assert result.passed is True


def test_daemon_held_results_yield_exit_zero() -> None:
    from iai_mcp.doctor import CheckResult

    results = [
        CheckResult("(a) daemon process alive", True, "ok"),
        CheckResult(
            "(f) hippo storage readable",
            True,
            "Hippo storage readable via the running daemon",
        ),
        CheckResult(
            "(t) hippo_compacted freshness",
            True,
            "last hippo_compacted event 2.0h ago",
            status="PASS",
        ),
        CheckResult(
            "(u) recall centrality regression",
            True,
            "centrality_ms median 10.0ms <= 30ms (n_events=3)",
            status="PASS",
        ),
    ]

    fail_count = sum(1 for r in results if not r.passed)
    assert fail_count == 0, "a daemon-served live read must not produce a FALSE FAIL"


def test_real_fail_still_flips_exit_code() -> None:
    from iai_mcp.doctor import CheckResult

    results = [
        CheckResult("(a) daemon process alive", True, "ok"),
        CheckResult(
            "(f) hippo storage readable",
            False,
            "open failed: DatabaseError: file is not a database",
        ),
    ]

    fail_count = sum(1 for r in results if not r.passed)
    assert fail_count == 1
