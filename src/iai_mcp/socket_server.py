
from __future__ import annotations

import asyncio
import inspect
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from iai_mcp.concurrency import SOCKET_PATH, cleanup_stale_socket
from iai_mcp.core import UnknownMethodError

ERR_DAEMON_INTERNAL = -32001
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_PARSE_ERROR = -32700

IDLE_SECS_DEFAULT = 1800


def _inherit_activated_socket() -> socket.socket | None:
    listen_fds = os.environ.get("LISTEN_FDS")
    listen_pid = os.environ.get("LISTEN_PID")
    if listen_fds is None or listen_pid is None:
        return None
    try:
        if int(listen_pid) != os.getpid():
            return None
        if int(listen_fds) < 1:
            return None
    except ValueError:
        return None
    inherited_fd = 3
    sock = socket.socket(fileno=inherited_fd)
    sock.setblocking(False)
    return sock


def _validate_jsonrpc_envelope(req: Any) -> tuple[bool, str | None]:
    if not isinstance(req, dict):
        return False, "request must be a JSON object"
    if req.get("jsonrpc") != "2.0":
        return False, "jsonrpc must be '2.0'"
    if "id" not in req or req["id"] is None:
        return False, "id required and non-null"
    if not isinstance(req.get("method"), str):
        return False, "method must be a string"
    if "params" in req and not isinstance(req["params"], (dict, list)):
        return False, "params must be object or array"
    return True, None


class SocketServer:

    CONTROL_MSG_TYPES = frozenset({
        "status", "user_initiated_sleep", "force_wake", "force_rem",
        "pause", "resume", "session_open", "embed_cue",
    })

    def __init__(
        self,
        store: Any,
        idle_secs: int | None = None,
        *,
        state: dict | None = None,
    ) -> None:
        self.store = store
        if idle_secs is None:
            idle_secs = IDLE_SECS_DEFAULT
        self.idle_secs = idle_secs
        self.last_activity_ts: float = time.monotonic()
        self.active_connections: int = 0
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self._state = state

    async def handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self.active_connections += 1
        try:
            while not reader.at_eof():
                line = await reader.readline()
                if not line:
                    break
                req_id: Any = None
                try:
                    req = json.loads(line)
                except json.JSONDecodeError as e:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": None,
                        "error": {"code": ERR_PARSE_ERROR, "message": str(e)},
                    }
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()
                    continue

                if (
                    isinstance(req, dict)
                    and req.get("type") in self.CONTROL_MSG_TYPES
                    and "jsonrpc" not in req
                ):
                    if self._state is None:
                        result = {
                            "ok": False,
                            "reason": "control_plane_unwired",
                            "error": (
                                "SocketServer constructed without state; "
                                "control-plane fork unavailable in this context"
                            ),
                        }
                    else:
                        try:
                            from iai_mcp.concurrency import _dispatch_socket_request
                            result = await _dispatch_socket_request(
                                req, self.store, self._state,
                            )
                        except Exception as e:  # noqa: BLE001
                            result = {"ok": False, "reason": "control_plane_error",
                                      "error": str(e)[:200]}
                    if result is not None:
                        writer.write((json.dumps(result) + "\n").encode("utf-8"))
                        await writer.drain()
                    continue

                # A genuine external MCP memory operation. Only these count as
                # "the user is actively working" for the sleep pipeline's
                # interrupt-check -- internal health/liveness probes and other
                # control-plane messages (handled above) must NOT, or the
                # daemon's own liveness watchdog would perpetually reset this
                # timestamp and starve consolidation. Dashboard READ verbs are
                # observation, not work — a watching brain view polling every
                # few seconds must not defer consolidation forever either.
                _p = req.get("params") if isinstance(req, dict) else None
                # The observation set is imported from brainview as the single
                # source of truth — a hand-copied tuple here would drift and let
                # a dashboard polling a read verb reset the activity clock,
                # starving consolidation.
                from iai_mcp.brainview import BRAIN_VIEW_OBSERVATION_VERBS
                _is_observation = (
                    isinstance(req, dict)
                    and req.get("method") == "brain_view"
                    and isinstance(_p, dict)
                    and str(_p.get("verb")) in BRAIN_VIEW_OBSERVATION_VERBS
                )
                if not _is_observation:
                    self.last_activity_ts = time.monotonic()

                ok, err = _validate_jsonrpc_envelope(req)
                req_id = req.get("id") if isinstance(req, dict) else None
                if not ok:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": ERR_INVALID_REQUEST, "message": err},
                    }
                    writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                    await writer.drain()
                    continue
                method = req["method"]
                params = req.get("params") or {}
                try:
                    from iai_mcp.core import dispatch
                    # Foreground-priority marker: while a live recall is in
                    # flight, polite background loops (deferred-capture drain
                    # and friends) yield the shared connection and the GIL.
                    _fg = method in (
                        "memory_recall",
                        "memory_recall_structural",
                        # A live search is foreground work too: a relayed
                        # teach must yield to it, not just to recalls.
                        "memory_search",
                    )
                    if _fg:
                        try:
                            from iai_mcp.concurrency import foreground_begin
                            foreground_begin()
                        except Exception:  # noqa: BLE001 -- beacon is advisory
                            _fg = False
                    try:
                        result = await asyncio.to_thread(
                            dispatch, self.store, method, params,
                        )
                    finally:
                        if _fg:
                            try:
                                from iai_mcp.concurrency import foreground_end
                                foreground_end()
                            except Exception:  # noqa: BLE001 -- beacon is advisory
                                pass
                    resp = {"jsonrpc": "2.0", "id": req_id, "result": result}
                except UnknownMethodError as e:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": ERR_METHOD_NOT_FOUND,
                            "message": f"unknown method '{e.args[0]}'",
                        },
                    }
                except KeyError as e:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {
                            "code": ERR_INVALID_PARAMS,
                            "message": f"missing required param: {e.args[0]!r}",
                        },
                    }
                except TypeError as e:
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": ERR_INVALID_PARAMS, "message": str(e)},
                    }
                except Exception as e:  # noqa: BLE001 -- socket must never crash daemon
                    resp = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": ERR_DAEMON_INTERNAL, "message": str(e)},
                    }
                writer.write((json.dumps(resp) + "\n").encode("utf-8"))
                await writer.drain()
        except ValueError:
            # readline() raises when a single line exceeds the 64MB limit;
            # reply with a parse error instead of dropping the task uncaught.
            try:
                writer.write((json.dumps({
                    "jsonrpc": "2.0", "id": None,
                    "error": {"code": ERR_PARSE_ERROR,
                              "message": "request line exceeds size limit"},
                }) + "\n").encode("utf-8"))
                await writer.drain()
            except (OSError, ConnectionError):
                pass
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass
        finally:
            self.active_connections -= 1
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):  # noqa: BLE001 -- cleanup is best-effort
                pass


    async def serve(self, socket_path: Path | None = None) -> None:
        if socket_path is None:
            env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
            socket_path = Path(env_path) if env_path else SOCKET_PATH

        sig = inspect.signature(asyncio.start_unix_server)
        supports_cleanup_socket = "cleanup_socket" in sig.parameters

        # One JSON-RPC request is one line; a relayed document upload
        # (base64, ≤25 MB raw) must fit the StreamReader line buffer.
        _line_limit = 64 * 1024 * 1024
        from iai_mcp._ipc import IS_WINDOWS, start_ipc_server, shutdown_ipc
        if IS_WINDOWS:
            server, _actual_addr, _needs_cleanup = await start_ipc_server(
                self.handle, socket_path, limit=_line_limit,
            )
            try:
                async with server:
                    await self.shutdown_event.wait()
                    server.close()
                    await server.wait_closed()
            finally:
                shutdown_ipc()
            return
        inherited = _inherit_activated_socket()
        if inherited is not None:
            server = await asyncio.start_unix_server(
                self.handle,
                sock=inherited,
                limit=_line_limit,
            )
        else:
            cleanup_stale_socket(socket_path)
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            server_kwargs: dict[str, Any] = (
                {"cleanup_socket": True} if supports_cleanup_socket else {}
            )
            server = await asyncio.start_unix_server(
                self.handle,
                path=str(socket_path),
                limit=_line_limit,
                **server_kwargs,
            )
            try:
                os.chmod(str(socket_path), 0o600)
            except OSError:
                pass

        try:
            async with server:
                await self.shutdown_event.wait()
                server.close()
                await server.wait_closed()
        finally:
            if inherited is None and not supports_cleanup_socket:
                try:
                    socket_path.unlink()
                except (FileNotFoundError, OSError):
                    pass
