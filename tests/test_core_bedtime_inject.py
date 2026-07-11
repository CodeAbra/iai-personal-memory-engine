from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path

import pytest

from iai_mcp import core


class _ThreadedFakeDaemon:

    def __init__(self, path: Path, captured: list, reply: dict) -> None:
        self.path = path
        self.captured = captured
        self.reply = reply
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)

            async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
                try:
                    line = await reader.readline()
                    if line:
                        self.captured.append(json.loads(line.decode("utf-8")))
                    writer.write((json.dumps(self.reply) + "\n").encode("utf-8"))
                    await writer.drain()
                finally:
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass

            async def _serve() -> None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._server = await asyncio.start_unix_server(_handle, path=str(self.path))
                self._ready.set()
                async with self._server:
                    await self._server.serve_forever()

            try:
                self._loop.run_until_complete(_serve())
            except asyncio.CancelledError:
                pass
            finally:
                self._loop.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        assert self._ready.wait(timeout=5.0), "fake daemon failed to start within 5s"

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return

        async def _shutdown() -> None:
            if self._server is not None:
                self._server.close()
                await self._server.wait_closed()

        fut = asyncio.run_coroutine_threadsafe(_shutdown(), loop)
        try:
            fut.result(timeout=5.0)
        except Exception:
            pass
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)


@pytest.fixture
def tmp_socket(tmp_path: Path) -> Path:
    candidate = tmp_path / "d.sock"
    if len(str(candidate)) > 100:
        candidate = Path(tempfile.mkdtemp(prefix="iai-sock-")) / "d.sock"
    return candidate


async def _run_fake_server(
    sock: Path,
    captured: list,
    reply: dict,
    *,
    delay_before_reply: float = 0.0,
) -> asyncio.AbstractServer:

    async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if line:
                captured.append(json.loads(line.decode("utf-8")))
            if delay_before_reply > 0:
                await asyncio.sleep(delay_before_reply)
            writer.write((json.dumps(reply) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    sock.parent.mkdir(parents=True, exist_ok=True)
    return await asyncio.start_unix_server(_handle, path=str(sock))


def test_consent_false_short_circuits_no_socket_touch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    async def _explode(*args, **kwargs):
        raise AssertionError(
            "C2 violation: asyncio.open_unix_connection reached with consent=False"
        )

    monkeypatch.setattr(asyncio, "open_unix_connection", _explode)

    result = asyncio.run(
        core.handle_initiate_sleep_mode({"consent": False, "reason": "not ready"})
    )
    assert result == {"ok": False, "reason": "consent_declined"}


def test_consent_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="consent"):
        asyncio.run(core.handle_initiate_sleep_mode({"reason": "missing"}))


def test_consent_wrong_type_raises_value_error() -> None:
    for bad in ["true", 1, 0, None, [True]]:
        with pytest.raises(ValueError):
            asyncio.run(
                core.handle_initiate_sleep_mode({"consent": bad, "reason": "x"})
            )


def test_reason_missing_raises_value_error() -> None:
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(core.handle_initiate_sleep_mode({"consent": True}))


def test_reason_wrong_type_raises_value_error() -> None:
    with pytest.raises(ValueError, match="reason"):
        asyncio.run(
            core.handle_initiate_sleep_mode({"consent": True, "reason": 42})
        )


def test_consent_true_opens_socket_and_returns_reply(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    async def _runner() -> dict:
        server = await _run_fake_server(
            tmp_socket, captured, {"ok": True, "state": "TRANSITIONING"},
        )
        try:
            async with server:
                monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
                return await core.handle_initiate_sleep_mode(
                    {"consent": True, "reason": "good night"},
                )
        finally:
            server.close()
            await server.wait_closed()

    result = asyncio.run(_runner())
    assert result == {"ok": True, "state": "TRANSITIONING"}
    assert len(captured) == 1
    sent = captured[0]
    assert sent["type"] == "user_initiated_sleep"
    assert sent["reason"] == "good night"
    assert "ts" in sent


def test_consent_true_daemon_unreachable_returns_graceful_error(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not tmp_socket.exists()
    monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
    result = asyncio.run(
        core.handle_initiate_sleep_mode(
            {"consent": True, "reason": "night"},
        )
    )
    assert result["ok"] is False
    assert result["reason"] == "daemon_not_running"


def test_force_wake_sends_correct_message(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []

    async def _runner() -> dict:
        server = await _run_fake_server(
            tmp_socket, captured, {"ok": True, "state": "WAKE"},
        )
        try:
            async with server:
                monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
                return await core.handle_force_wake({})
        finally:
            server.close()
            await server.wait_closed()

    result = asyncio.run(_runner())
    assert result == {"ok": True, "state": "WAKE"}
    assert len(captured) == 1
    assert captured[0]["type"] == "force_wake"
    assert "ts" in captured[0]


def test_force_wake_daemon_unreachable_graceful(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert not tmp_socket.exists()
    monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
    result = asyncio.run(core.handle_force_wake({}))
    assert result["ok"] is False
    assert result["reason"] == "daemon_not_running"


def test_force_wake_timeout_is_fifteen_minutes() -> None:
    assert core.FORCE_WAKE_TIMEOUT_SEC == 900


def _window_covering_now() -> tuple[int, int]:
    from iai_mcp.tz import load_user_tz

    tz = load_user_tz()
    now_local = datetime.now(timezone.utc).astimezone(tz)
    cur_bucket = (now_local.hour * 60 + now_local.minute) // 30
    start = (cur_bucket - 2) % 48
    return (start, 8)


def test_inject_sleep_suggestion_dual_gate_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_state = {"quiet_window": _window_covering_now()}

    def _load() -> dict:
        return dict(fake_state)

    monkeypatch.setattr("iai_mcp.daemon_state.load_state", _load)

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" in response, (
        f"expected injection on dual-gate pass, got {response!r}"
    )
    assert response["sleep_suggestion"]["message_hint"] == "user_wind_down_detected"


def test_inject_sleep_suggestion_no_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_state = {"quiet_window": _window_covering_now()}
    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: dict(fake_state),
    )

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(
        response, cue="how do I configure pytest", language="en",
    )
    assert "sleep_suggestion" not in response


def test_inject_sleep_suggestion_no_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("iai_mcp.daemon_state.load_state", lambda: {})

    response: dict = {"hits": [], "anti_hits": []}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" not in response


def test_inject_sleep_suggestion_detector_raises_is_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args, **kwargs):
        raise RuntimeError("synthetic bedtime failure")

    monkeypatch.setattr("iai_mcp.bedtime.detect_wind_down", _boom)

    response: dict = {"hits": [], "anti_hits": [], "budget_used": 0}
    core._inject_sleep_suggestion(response, cue="good night", language="en")
    assert "sleep_suggestion" not in response
    assert response == {"hits": [], "anti_hits": [], "budget_used": 0}


def test_dispatch_routes_initiate_sleep_mode(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(tmp_socket, captured, {"ok": True})
    daemon.start()
    try:
        monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
        result = core.dispatch(
            None,
            "initiate_sleep_mode",
            {"consent": True, "reason": "test"},
        )
        assert result == {"ok": True}
        assert captured[0]["type"] == "user_initiated_sleep"
    finally:
        daemon.stop()


def test_dispatch_routes_force_wake(
    tmp_socket: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict] = []
    daemon = _ThreadedFakeDaemon(tmp_socket, captured, {"ok": True, "state": "WAKE"})
    daemon.start()
    try:
        monkeypatch.setattr(core, "SOCKET_PATH", tmp_socket)
        result = core.dispatch(None, "force_wake", {})
        assert result == {"ok": True, "state": "WAKE"}
        assert captured[0]["type"] == "force_wake"
    finally:
        daemon.stop()
