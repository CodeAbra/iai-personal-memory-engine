from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from iai_mcp import daemon_state
from iai_mcp.concurrency import (
    _dispatch_socket_request,
    _validate_socket_message,
    serve_control_socket,
)


@pytest.fixture
def tmp_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    candidate = tmp_path / "d.sock"
    if len(str(candidate)) > 100:
        candidate = Path(tempfile.mkdtemp(prefix="iai-sock-")) / "d.sock"
    # Per-test endpoint isolation: serve_control_socket + open_ipc_connection
    # resolve through this (unix socket on POSIX, TCP "<path>.port" on Windows).
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(candidate))
    return candidate


@pytest.fixture
def tmp_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    p = tmp_path / ".daemon-state.json"
    monkeypatch.setattr(daemon_state, "STATE_PATH", p)
    return p


def test_validate_session_open_accepts_valid_message() -> None:
    ok, err = _validate_socket_message(
        {"type": "session_open", "session_id": "s1", "ts": "2026-04-19T00:00:00Z"}
    )
    assert ok is True
    assert err is None


def test_validate_session_open_accepts_missing_session_id() -> None:
    ok, err = _validate_socket_message({"type": "session_open"})
    assert ok is True
    assert err is None


def test_validate_session_open_rejects_non_string_session_id() -> None:
    ok, err = _validate_socket_message(
        {"type": "session_open", "session_id": 123, "ts": "x"}
    )
    assert ok is False
    assert err is not None
    assert "session_id" in err


def test_validate_session_open_rejects_non_string_ts() -> None:
    ok, err = _validate_socket_message(
        {"type": "session_open", "session_id": "s1", "ts": 42}
    )
    assert ok is False
    assert err is not None
    assert "ts" in err


def _make_fake_store() -> Any:
    return MagicMock()


def test_dispatch_session_open_queues_first_turn_and_cascade(
    tmp_state: Path,
) -> None:
    state: dict = {"fsm_state": "WAKE"}
    req = {
        "type": "session_open",
        "session_id": "sess-abc",
        "ts": "2026-04-19T12:00:00Z",
    }
    resp = asyncio.run(
        _dispatch_socket_request(req, _make_fake_store(), state)
    )
    assert resp == {"ok": True, "reason": "session_open_queued"}
    pending = state.get("first_turn_pending")
    assert isinstance(pending, dict)
    stamp = pending.get("sess-abc")
    assert isinstance(stamp, str) and stamp
    cascade = state.get("hippea_cascade_request")
    assert isinstance(cascade, dict)
    assert cascade.get("pending") is True
    assert cascade.get("session_id") == "sess-abc"
    last = state.get("last_session_open")
    assert isinstance(last, dict)
    assert last.get("session_id") == "sess-abc"
    assert tmp_state.exists()
    on_disk = json.loads(tmp_state.read_text())
    assert on_disk.get("hippea_cascade_request", {}).get("pending") is True


def test_dispatch_session_open_missing_session_id_ok(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    req = {"type": "session_open", "ts": "2026-04-19T12:00:00Z"}
    resp = asyncio.run(
        _dispatch_socket_request(req, _make_fake_store(), state)
    )
    assert resp.get("ok") is True
    assert resp.get("reason") == "session_open_queued"


def test_dispatch_session_open_clips_long_session_id(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    long_id = "a" * 1000
    req = {"type": "session_open", "session_id": long_id, "ts": "x"}
    resp = asyncio.run(
        _dispatch_socket_request(req, _make_fake_store(), state)
    )
    assert resp["ok"] is True
    last = state.get("last_session_open") or {}
    assert len(last.get("session_id", "")) <= 128


def test_dispatch_force_wake_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "force_wake", "ts": "x"},
            _make_fake_store(),
            state,
        )
    )
    assert resp == {"ok": True, "reason": "wake_queued"}


def test_dispatch_force_rem_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "force_rem", "ts": "x"},
            _make_fake_store(),
            state,
        )
    )
    assert resp == {"ok": True, "reason": "rem_queued"}


def test_dispatch_pause_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "pause"},
            _make_fake_store(),
            state,
        )
    )
    assert resp == {"ok": True, "paused": True}
    assert state["scheduler_paused"] is True


def test_dispatch_resume_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE", "scheduler_paused": True}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "resume"},
            _make_fake_store(),
            state,
        )
    )
    assert resp == {"ok": True, "paused": False}
    assert state["scheduler_paused"] is False


def test_dispatch_user_initiated_sleep_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "user_initiated_sleep", "reason": "night", "ts": "x"},
            _make_fake_store(),
            state,
        )
    )
    assert resp.get("ok") is True
    assert resp.get("state") == "TRANSITIONING"


def test_dispatch_status_still_works(tmp_state: Path) -> None:
    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "status"},
            _make_fake_store(),
            state,
        )
    )
    assert resp.get("ok") is True
    assert resp.get("state") == "WAKE"
    assert "version" in resp


def test_dispatch_status_prefers_canonical_lifecycle_state(
    tmp_state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp import lifecycle_state

    canonical_path = tmp_path / "lifecycle_state.json"
    monkeypatch.setattr(lifecycle_state, "LIFECYCLE_STATE_PATH", canonical_path)
    lifecycle_state.save_state(
        {
            "current_state": "DROWSY",
            "since_ts": "2026-07-05T08:22:48+00:00",
            "last_activity_ts": "2026-07-05T08:17:27+00:00",
            "wrapper_event_seq": 1,
            "sleep_cycle_progress": None,
            "quarantine": None,
            "shadow_run": False,
            "crisis_mode": False,
            "crisis_mode_since_ts": None,
        },
        canonical_path,
    )

    state: dict = {"fsm_state": "WAKE"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "status"},
            _make_fake_store(),
            state,
        )
    )

    assert resp.get("ok") is True
    assert resp.get("state") == "DROWSY"
    assert resp.get("fsm_state") == "DROWSY"


def test_dispatch_status_ignores_invalid_canonical_lifecycle_state(
    tmp_state: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from iai_mcp import lifecycle_state

    canonical_path = tmp_path / "lifecycle_state.json"
    canonical_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(lifecycle_state, "LIFECYCLE_STATE_PATH", canonical_path)

    state: dict = {"fsm_state": "TRANSITIONING"}
    resp = asyncio.run(
        _dispatch_socket_request(
            {"type": "status"},
            _make_fake_store(),
            state,
        )
    )

    assert resp.get("ok") is True
    assert resp.get("state") == "TRANSITIONING"
    assert resp.get("fsm_state") == "TRANSITIONING"


class _ThreadedDaemon:

    def __init__(self, path: Path, state: dict) -> None:
        self.path = path
        self.state = state
        self.store = MagicMock()
        self.shutdown = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self.shutdown = asyncio.Event()

            async def _serve() -> None:
                async def _dispatcher(req: dict) -> dict:
                    return await _dispatch_socket_request(
                        req, self.store, self.state
                    )

                task = asyncio.create_task(
                    serve_control_socket(
                        self.store,
                        self.state,
                        self.shutdown,  # type: ignore[arg-type]
                        dispatcher=_dispatcher,
                        socket_path=self.path,
                    )
                )
                await asyncio.sleep(0.1)
                self._ready.set()
                await task

            try:
                self._loop.run_until_complete(_serve())
            except Exception:
                pass
            finally:
                try:
                    self._loop.close()
                except Exception:
                    pass

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0), "threaded daemon failed to start"

    def stop(self) -> None:
        if self._loop is None:
            return
        if self.shutdown is not None:
            self._loop.call_soon_threadsafe(self.shutdown.set)
        self._thread and self._thread.join(timeout=5.0)


async def _send(path: Path, msg: dict, *, timeout: float = 5.0) -> dict:
    from iai_mcp._ipc import open_ipc_connection

    reader, writer = await open_ipc_connection(timeout=timeout)
    try:
        writer.write((json.dumps(msg) + "\n").encode("utf-8"))
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        return json.loads(line)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass


def test_session_open_end_to_end_round_trip(
    tmp_socket: Path, tmp_state: Path,
) -> None:
    state: dict = {"fsm_state": "WAKE"}
    daemon = _ThreadedDaemon(tmp_socket, state)
    daemon.start()
    try:
        resp = asyncio.run(
            _send(
                tmp_socket,
                {
                    "type": "session_open",
                    "session_id": "e2e-sess-1",
                    "ts": "2026-04-19T12:00:00Z",
                },
            )
        )
        assert resp == {"ok": True, "reason": "session_open_queued"}
        pending = state.get("first_turn_pending")
        assert isinstance(pending, dict)
        stamp = pending.get("e2e-sess-1")
        assert isinstance(stamp, str) and stamp
        cascade = state.get("hippea_cascade_request")
        assert isinstance(cascade, dict)
        assert cascade.get("pending") is True
    finally:
        daemon.stop()


def test_session_open_does_not_regress_other_6_types(
    tmp_socket: Path, tmp_state: Path,
) -> None:
    state: dict = {"fsm_state": "WAKE"}
    daemon = _ThreadedDaemon(tmp_socket, state)
    daemon.start()
    try:
        r = asyncio.run(_send(tmp_socket, {"type": "force_wake", "ts": "x"}))
        assert r == {"ok": True, "reason": "wake_queued"}
        r = asyncio.run(_send(tmp_socket, {"type": "force_rem", "ts": "x"}))
        assert r == {"ok": True, "reason": "rem_queued"}
        r = asyncio.run(_send(tmp_socket, {"type": "pause"}))
        assert r.get("ok") is True
        r = asyncio.run(_send(tmp_socket, {"type": "resume"}))
        assert r.get("ok") is True
        r = asyncio.run(_send(tmp_socket, {"type": "status"}))
        assert r.get("ok") is True
        r = asyncio.run(
            _send(
                tmp_socket,
                {"type": "user_initiated_sleep", "reason": "night", "ts": "x"},
            )
        )
        assert r.get("ok") is True
    finally:
        daemon.stop()
