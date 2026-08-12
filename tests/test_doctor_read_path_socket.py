from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4


def _short_socket_path(tag: str):
    """A unix socket path short enough for sun_path (108 bytes on macOS).

    pytest's own tmp_path is nested too deep for AF_UNIX.
    """
    import os as _os
    import uuid as _uuid
    from pathlib import Path as _Path

    d = _Path(f"/tmp/iai-rps-{_os.getpid()}-{_uuid.uuid4().hex[:8]}-{tag}")
    d.mkdir(parents=True, exist_ok=True)
    return d, d / "d.sock"


def _spawn_socket_server(store, sock_path):
    import asyncio as _asyncio
    import threading

    from iai_mcp.socket_server import SocketServer

    srv = SocketServer(store, idle_secs=99999)
    ready = threading.Event()
    ctx: dict = {}

    def _thread_main() -> None:
        async def _main() -> None:
            ctx["loop"] = _asyncio.get_running_loop()
            task = _asyncio.ensure_future(srv.serve(socket_path=sock_path))
            for _ in range(250):
                if sock_path.exists():
                    break
                await _asyncio.sleep(0.01)
            ready.set()
            await task

        _asyncio.run(_main())

    thread = threading.Thread(target=_thread_main, daemon=True)
    thread.start()
    if not ready.wait(timeout=5.0):
        raise AssertionError("socket server never bound")
    return srv, thread, ctx


def _stop_socket_server(srv, thread, ctx) -> None:
    loop = ctx.get("loop")
    if loop is not None:
        try:
            loop.call_soon_threadsafe(srv.shutdown_event.set)
        except RuntimeError:
            pass
    thread.join(timeout=5.0)


def _refuse_direct_store_open(*args, **kwargs):
    raise AssertionError(
        "the check must not open a direct store handle while the daemon "
        "socket answers -- this proves it did"
    )


def _write_event_at(store, kind: str, data: dict, ts: datetime) -> None:
    """Insert an events row at an explicit timestamp.

    Mirrors ``iai_mcp.events.write_event`` byte-for-byte except for the
    ``ts`` (that helper always stamps ``datetime.now()``).
    """
    from iai_mcp.crypto import encrypt_field
    from iai_mcp.store import EVENTS_TABLE

    event_id = uuid4()
    ad = str(event_id).encode("ascii")
    data_ct = encrypt_field(json.dumps(data), store._key(), associated_data=ad)
    row = {
        "id": str(event_id),
        "kind": kind,
        "severity": "",
        "domain": "",
        "ts": ts,
        "data_json": data_ct,
        "session_id": "-",
        "source_ids_json": "[]",
    }
    store.db.open_table(EVENTS_TABLE).add([row])


def test_check_t_daemon_up_reads_stale_event_through_socket(tmp_path, monkeypatch) -> None:
    """A green WARN verdict here can only mean the socket served the read:
    the direct store open is monkeypatched to raise."""
    from iai_mcp.doctor._storage_checks import check_t_hippo_compacted_freshness
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("t-stale")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        old_ts = datetime.now(timezone.utc) - timedelta(hours=48)
        _write_event_at(store, "hippo_compacted", {"phase": "sleep_cycle"}, old_ts)

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_t_hippo_compacted_freshness()

        assert result.status == "WARN"
        assert result.passed is True
        assert "h ago" in result.detail
    finally:
        if srv is not None:
            _stop_socket_server(srv, thread, ctx)
        store.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_t_daemon_up_reads_fresh_event_through_socket(tmp_path, monkeypatch) -> None:
    from iai_mcp.doctor._storage_checks import check_t_hippo_compacted_freshness
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("t-fresh")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        write_event(store, kind="hippo_compacted", data={"phase": "sleep_cycle"})

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_t_hippo_compacted_freshness()

        assert result.status == "PASS"
        assert result.passed is True
        assert "h ago" in result.detail
    finally:
        if srv is not None:
            _stop_socket_server(srv, thread, ctx)
        store.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_t_malformed_socket_reply_degrades_to_warn_not_pass(monkeypatch) -> None:
    """A wire-shape-malformed reply (result present, no 'events' key) must
    never look like a legitimate answer."""
    from iai_mcp.doctor._storage_checks import check_t_hippo_compacted_freshness

    def _fake_send(method, params, *, connect_timeout=1.0, read_timeout=5.0):
        return {"jsonrpc": "2.0", "id": 1, "result": {"error": "not user-visible"}}

    monkeypatch.setenv("IAI_MCP_STORE", "/tmp/iai-rps-does-not-matter")
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)
    monkeypatch.setattr(
        "iai_mcp.doctor._storage_checks._store_file_present", lambda: True,
    )

    result = check_t_hippo_compacted_freshness()

    assert result.status == "WARN"
    assert result.passed is True


def test_check_u_daemon_up_reads_high_median_through_socket(tmp_path, monkeypatch) -> None:
    from iai_mcp.doctor._storage_checks import check_u_recall_centrality_regression
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("u-high")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        for ms in (60.0, 70.0, 80.0):
            write_event(
                store,
                kind="recall_timing",
                data={
                    "centrality_ms": ms, "sigma_ms": 0.0,
                    "pool_collection_ms": 1.0, "n_nodes": 10,
                },
            )

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_u_recall_centrality_regression()

        assert result.status == "WARN"
        assert result.passed is True
        assert "30ms threshold" in result.detail
    finally:
        if srv is not None:
            _stop_socket_server(srv, thread, ctx)
        store.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_u_daemon_up_reads_low_median_through_socket(tmp_path, monkeypatch) -> None:
    from iai_mcp.doctor._storage_checks import check_u_recall_centrality_regression
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("u-low")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        for ms in (5.0, 10.0, 15.0):
            write_event(
                store,
                kind="recall_timing",
                data={
                    "centrality_ms": ms, "sigma_ms": 0.0,
                    "pool_collection_ms": 1.0, "n_nodes": 10,
                },
            )

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_u_recall_centrality_regression()

        assert result.status == "PASS"
        assert result.passed is True
        assert "<= 30ms" in result.detail
    finally:
        if srv is not None:
            _stop_socket_server(srv, thread, ctx)
        store.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_u_malformed_socket_reply_degrades_to_warn_not_pass(monkeypatch) -> None:
    from iai_mcp.doctor._storage_checks import check_u_recall_centrality_regression

    def _fake_send(method, params, *, connect_timeout=1.0, read_timeout=5.0):
        return {"jsonrpc": "2.0", "id": 1, "result": {"error": "not user-visible"}}

    monkeypatch.setenv("IAI_MCP_STORE", "/tmp/iai-rps-does-not-matter")
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)
    monkeypatch.setattr(
        "iai_mcp.doctor._storage_checks._store_file_present", lambda: True,
    )

    result = check_u_recall_centrality_regression()

    assert result.status == "WARN"
    assert result.passed is True


def test_check_f_daemon_up_proves_live_read_through_socket(tmp_path, monkeypatch) -> None:
    """A green PASS here can only mean the socket served the proof read
    (session_started via events_query): the direct store open is
    monkeypatched to raise."""
    from iai_mcp.doctor import check_f_hippo_readable
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("f-live")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        write_event(store, kind="session_started", data={})

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_f_hippo_readable()

        assert result.status == "PASS"
        assert result.passed is True
        assert "daemon" in result.detail.lower()
    finally:
        if srv is not None:
            _stop_socket_server(srv, thread, ctx)
        store.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_f_daemon_up_proves_live_read_with_no_prior_events(
    tmp_path, monkeypatch
) -> None:
    """Even an EMPTY events reply proves the daemon executed a real store
    read -- absence of session_started rows must still PASS, not WARN."""
    from iai_mcp.doctor import check_f_hippo_readable
    from iai_mcp.events import write_event
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("f-empty")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        # Seed an unrelated event so the store is non-empty, but no
        # session_started row exists.
        write_event(store, kind="hippo_compacted", data={"phase": "sleep_cycle"})

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_f_hippo_readable()

        assert result.status == "PASS"
        assert result.passed is True
    finally:
        if srv is not None:
            _stop_socket_server(srv, thread, ctx)
        store.close()
        try:
            if sock_path.exists():
                sock_path.unlink()
            sock_dir.rmdir()
        except OSError:
            pass


def test_check_f_malformed_socket_reply_degrades_to_warn_not_pass(monkeypatch) -> None:
    from iai_mcp.doctor import check_f_hippo_readable

    def _fake_send(method, params, *, connect_timeout=1.0, read_timeout=5.0):
        return {"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "boom"}}

    monkeypatch.setenv("IAI_MCP_STORE", "/tmp/iai-rps-does-not-matter")
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", _fake_send)
    monkeypatch.setattr(
        "iai_mcp.doctor._storage_checks._store_file_present", lambda: True,
    )

    result = check_f_hippo_readable()

    assert result.status == "WARN"
    assert result.passed is True


