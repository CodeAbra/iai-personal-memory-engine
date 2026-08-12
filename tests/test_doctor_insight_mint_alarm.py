from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from iai_mcp.doctor._lifecycle_checks import check_bb_nightly_insight_mint

_NOW = datetime(2026, 8, 11, 15, 0, 0, tzinfo=timezone.utc)


def _ev(day_offset: int, data: dict, hour: int = 3) -> dict:
    day = _NOW.date() - timedelta(days=day_offset)
    ts = datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)
    return {"ts": ts, "data": data}


def _patch_events(monkeypatch: pytest.MonkeyPatch, events: list[dict]) -> None:
    def _fake_query_events(store, kind=None, since=None, severity=None, limit=100, *, since_exclusive=False):
        return list(events)

    monkeypatch.setattr("iai_mcp.events.query_events", _fake_query_events)


def _raising_query_events(store, kind=None, since=None, severity=None, limit=100, *, since_exclusive=False):
    if since is not None:
        raise RuntimeError("simulated read failure")
    return [{"ts": _NOW, "data": {}}]


def test_fails_after_three_consecutive_insight_less_nights(monkeypatch) -> None:
    events = [
        _ev(1, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
        _ev(2, {"claude_call_used": False, "timed_out": True}),
        _ev(3, {"claude_call_used": False, "insight_skip_reason": "rate_limited"}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "FAIL"
    assert result.passed is False
    assert "3" in result.detail


def test_passes_when_a_recent_night_minted_an_insight(monkeypatch) -> None:
    events = [
        _ev(1, {"claude_call_used": True}),
        _ev(2, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
        _ev(3, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "PASS"
    assert result.passed is True


def test_passes_when_todays_window_already_minted_an_insight(monkeypatch) -> None:
    events = [
        _ev(0, {"claude_call_used": True}, hour=9),
        _ev(1, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
        _ev(2, {"claude_call_used": False, "timed_out": True}),
        _ev(3, {"claude_call_used": False, "insight_skip_reason": "rate_limited"}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "PASS"
    assert result.passed is True


def test_passes_when_only_todays_window_has_any_history(monkeypatch) -> None:
    events = [
        _ev(0, {"claude_call_used": False, "cycle": 1}, hour=10),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "PASS"
    assert result.passed is True


def test_old_mint_does_not_mask_a_fresh_three_night_fail(monkeypatch) -> None:
    events = [
        _ev(10, {"claude_call_used": True}),
        _ev(1, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
        _ev(2, {"claude_call_used": False, "timed_out": True}),
        _ev(3, {"claude_call_used": False, "insight_skip_reason": "rate_limited"}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "FAIL"
    assert result.passed is False


def test_passes_on_a_fresh_store_with_no_nightly_history(monkeypatch) -> None:
    _patch_events(monkeypatch, [])

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "PASS"
    assert result.passed is True
    assert "fresh install" in result.detail.lower() or "no nightly" in result.detail.lower()


def test_warns_on_two_consecutive_insight_less_nights_below_threshold(monkeypatch) -> None:
    events = [
        _ev(1, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
        _ev(2, {"claude_call_used": False, "timed_out": True}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "WARN"
    assert result.passed is True


def test_current_not_yet_elapsed_window_excluded_from_the_count(monkeypatch) -> None:
    events = [
        _ev(0, {"claude_call_used": False, "cycle": 1}, hour=10),
        _ev(1, {"claude_call_used": False, "insight_skip_reason": "auth_failed"}),
        _ev(2, {"claude_call_used": False, "timed_out": True}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "WARN"
    assert result.passed is True


def test_unreadable_event_data_never_produces_a_false_fail(monkeypatch) -> None:
    events = [
        _ev(1, {}),
        _ev(2, {}),
        _ev(3, {}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status != "FAIL"
    assert result.passed is True


def test_query_failure_degrades_to_warn_not_fail(monkeypatch) -> None:
    monkeypatch.setattr("iai_mcp.events.query_events", _raising_query_events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert result.status == "WARN"
    assert result.passed is True


def _write_event_at(store, kind: str, data: dict, ts: datetime) -> None:
    """Insert an events row at an explicit timestamp.

    Mirrors ``iai_mcp.events.write_event`` byte-for-byte except for the ``ts``
    (that helper always stamps ``datetime.now()``, which cannot place a row
    in a specific elapsed nightly bucket).
    """
    import json
    from uuid import uuid4

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


def test_alarm_evaluates_instead_of_deferring_while_daemon_holds_the_store(
    tmp_path, monkeypatch
) -> None:
    """Reproduces the daemon-up steady state: a second process (here, a
    second same-process handle simulating it) holds the store's SHARED lock
    the way the live daemon does after boot (`downgrade_to_shared`). Before
    the fix, the check's self-owned open (`MemoryStore()`, EXCLUSIVE by
    default) collided with that hold and returned a blind
    ``PASS | "deferred -- daemon holds the store (normal)"`` without reading
    a single event. After the fix (SHARED + read-only self-open) the check
    coexists with the held lock and evaluates the real, seeded FAIL
    condition end-to-end -- no monkeypatch of query_events, no injected
    store.

    The socket route is deliberately pointed at a nonexistent path so this
    test exercises the direct-open fallback specifically, independent of
    whatever real daemon may or may not be running on the host machine.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "no-such-daemon.sock"))
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    daemon_store = MemoryStore(tmp_path)
    try:
        for offset in (1, 2, 3):
            day = _NOW.date() - timedelta(days=offset)
            ts = datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc)
            _write_event_at(
                daemon_store,
                "rem_cycle_completed",
                {"claude_call_used": False, "insight_skip_reason": "rate_limited"},
                ts,
            )
        # Steady WAKE state: the daemon downgrades from its boot-time
        # EXCLUSIVE open to SHARED once startup completes.
        daemon_store.db.downgrade_to_shared()
        assert daemon_store.db._access_mode is AccessMode.SHARED

        result = check_bb_nightly_insight_mint(now=_NOW)

        assert result.status == "FAIL", (
            f"expected the alarm to evaluate the real FAIL condition, got "
            f"status={result.status!r} detail={result.detail!r}"
        )
        assert result.passed is False
        assert "3" in result.detail
        assert "deferred" not in result.detail
    finally:
        daemon_store.close()


def test_detail_never_echoes_raw_insight_text_or_skip_reason(monkeypatch) -> None:
    secret_reason = "credential-leak-marker-should-not-surface"
    events = [
        _ev(1, {"claude_call_used": False, "insight_skip_reason": secret_reason}),
        _ev(2, {"claude_call_used": False, "insight_skip_reason": secret_reason}),
        _ev(3, {"claude_call_used": False, "insight_skip_reason": secret_reason}),
    ]
    _patch_events(monkeypatch, events)

    result = check_bb_nightly_insight_mint(store=object(), now=_NOW)

    assert secret_reason not in result.detail


# --- Daemon-UP steady-state coverage: real socket, real dispatch ----------
#
# These reproduce the actual bug this plan fixes: while the daemon is up it
# holds the store's OS lock, so a same-process direct open cannot read it.
# A real `SocketServer` is run in a background thread against a real
# `MemoryStore`, and the check talks to it over a real unix socket using
# the real `events_query` -> `core.dispatch` route -- no mock of the
# check's own decision helpers. `MemoryStore` is monkeypatched to raise so
# a passing assertion can only mean the socket path served the answer, not
# a silent fall-through to the direct-open path.


def _short_socket_path(tag: str):
    """A unix socket path short enough for sun_path (108 bytes on macOS).

    pytest's own tmp_path is nested too deep for AF_UNIX; mirrors the
    pattern already used by test_socket_server_dispatch.py.
    """
    import os as _os
    import uuid as _uuid
    from pathlib import Path as _Path

    d = _Path(f"/tmp/iai-bb-{_os.getpid()}-{_uuid.uuid4().hex[:8]}-{tag}")
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
        "check_bb_nightly_insight_mint must not open a direct store handle "
        "while the daemon socket answers -- this proves it did"
    )


def test_daemon_up_reads_through_socket_and_evaluates_real_fail_verdict(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("fail")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        for offset in (1, 2, 3):
            day = _NOW.date() - timedelta(days=offset)
            ts = datetime(day.year, day.month, day.day, 3, tzinfo=timezone.utc)
            _write_event_at(
                store,
                "rem_cycle_completed",
                {"claude_call_used": False, "insight_skip_reason": "rate_limited"},
                ts,
            )

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_bb_nightly_insight_mint(now=_NOW)

        assert result.status == "FAIL", (
            f"expected the socket-served FAIL verdict, got "
            f"status={result.status!r} detail={result.detail!r}"
        )
        assert result.passed is False
        assert "3" in result.detail
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


def test_daemon_up_socket_reports_fresh_install_with_no_history(
    tmp_path, monkeypatch
) -> None:
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("fresh")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        monkeypatch.setattr("iai_mcp.store.MemoryStore", _refuse_direct_store_open)

        result = check_bb_nightly_insight_mint(now=_NOW)

        assert result.status == "PASS"
        assert result.passed is True
        assert "nightly" in result.detail.lower()
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


def _fake_socket_probe_then(windowed_result: dict):
    """Real probe (history exists), then a stubbed windowed response.

    Mirrors the wire shape `_send_jsonrpc_request` returns for `events_query`.
    """
    calls: list[dict] = []

    def _fake_send(method, params, *, connect_timeout=1.0, read_timeout=5.0):
        calls.append(params)
        if len(calls) == 1:
            return {
                "jsonrpc": "2.0",
                "id": 1,
                "result": {
                    "events": [
                        {
                            "id": "probe",
                            "kind": "rem_cycle_completed",
                            "ts": _NOW.isoformat(),
                            "data": {},
                        }
                    ],
                    "count": 1,
                },
            }
        return {"jsonrpc": "2.0", "id": 1, "result": windowed_result}

    return _fake_send


def test_socket_windowed_row_with_unparseable_ts_degrades_to_warn_not_pass(
    monkeypatch,
) -> None:
    """A shape mismatch that drops every windowed row must never look like
    an empty history and silently resolve to PASS."""
    fake_send = _fake_socket_probe_then(
        {
            "events": [
                {
                    "id": "bad",
                    "kind": "rem_cycle_completed",
                    "ts": "not-a-timestamp",
                    "data": {"claude_call_used": False},
                }
            ],
            "count": 1,
        }
    )
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", fake_send)

    result = check_bb_nightly_insight_mint(now=_NOW)

    assert result.status == "WARN"
    assert result.passed is True


def test_socket_windowed_response_legitimately_empty_still_passes(
    monkeypatch,
) -> None:
    """No drops, just genuinely no rows inside the window -- must stay PASS,
    not be overcorrected into a WARN."""
    fake_send = _fake_socket_probe_then({"events": [], "count": 0})
    monkeypatch.setattr("iai_mcp.cli._send_jsonrpc_request", fake_send)

    result = check_bb_nightly_insight_mint(now=_NOW)

    assert result.status == "PASS"
    assert result.passed is True


def test_events_query_socket_route_redacts_main_insight_text(
    tmp_path, monkeypatch
) -> None:
    """The `events_query` socket dispatch must strip `main_insight_text`
    from the wire payload while pass/fail metadata survives -- this is the
    security control the commit adds, exercised end-to-end over a real
    socket and a real dispatch, not a mock of the redaction itself."""
    import json

    from iai_mcp.cli import _send_jsonrpc_request
    from iai_mcp.store import MemoryStore

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    sock_dir, sock_path = _short_socket_path("redact")
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(sock_path))

    sentinel = "SENTINEL_INSIGHT_PROSE_MUST_NOT_LEAVE_THE_STORE"
    store = MemoryStore(tmp_path)
    srv = None
    thread = None
    try:
        _write_event_at(
            store,
            "rem_cycle_completed",
            {"claude_call_used": True, "main_insight_text": sentinel},
            _NOW,
        )

        srv, thread, ctx = _spawn_socket_server(store, sock_path)

        resp = _send_jsonrpc_request(
            "events_query",
            {"kind": "rem_cycle_completed", "limit": 10},
            connect_timeout=2.0,
            read_timeout=5.0,
        )

        assert resp is not None
        events = resp["result"]["events"]
        assert len(events) == 1
        data = events[0]["data"]
        assert data["claude_call_used"] is True
        assert "main_insight_text" not in data
        assert sentinel not in json.dumps(resp)
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
