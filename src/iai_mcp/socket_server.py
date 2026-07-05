
from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from pathlib import Path
from typing import Any

from iai_mcp._ipc import IS_WINDOWS, cleanup_ipc_address, shutdown_ipc, start_ipc_server
from iai_mcp.concurrency import SOCKET_PATH
from iai_mcp.core import UnknownMethodError

ERR_DAEMON_INTERNAL = -32001
ERR_INVALID_REQUEST = -32600
ERR_METHOD_NOT_FOUND = -32601
ERR_INVALID_PARAMS = -32602
ERR_PARSE_ERROR = -32700

IDLE_SECS_DEFAULT = 1800
RECALL_BUSY_REASON = "recall_busy"
RECALL_CONCURRENCY_DEFAULT = 2
RECALL_SLOT_WAIT_SEC_DEFAULT = 0.25


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


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
        self._recall_slots = asyncio.Semaphore(
            _env_int("IAI_MCP_RECALL_CONCURRENCY", RECALL_CONCURRENCY_DEFAULT)
        )
        self._recall_slot_wait_sec = _env_float(
            "IAI_MCP_RECALL_SLOT_WAIT_SEC",
            RECALL_SLOT_WAIT_SEC_DEFAULT,
        )
        self.last_activity_ts: float = time.monotonic()
        self.active_connections: int = 0
        self.shutdown_event: asyncio.Event = asyncio.Event()
        self._state = state

    async def _dispatch_jsonrpc_method(self, method: str, params: dict) -> Any:
        from iai_mcp.core import dispatch

        if method != "memory_recall" or "cue" not in params:
            return await asyncio.to_thread(dispatch, self.store, method, params)

        try:
            await asyncio.wait_for(
                self._recall_slots.acquire(),
                timeout=max(self._recall_slot_wait_sec, 0.001),
            )
        except asyncio.TimeoutError:
            return {
                "hits": [],
                "anti_hits": [],
                "activation_trace": [],
                "budget_used": 0,
                "_degraded": True,
                "_reason": RECALL_BUSY_REASON,
            }

        try:
            return await asyncio.to_thread(dispatch, self.store, method, params)
        finally:
            self._recall_slots.release()

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
                # NB: last_activity_ts is intentionally NOT updated for every
                # inbound line. It must track REAL memory traffic (recall/capture)
                # only — never control-plane messages, in particular the watchdog's
                # periodic {"type":"status"} liveness probe (every 7-30s). Counting
                # those probes as activity kept _interrupt_check (daemon) perpetually
                # True, so the sleep cycle never completed and never hibernated ->
                # the 221% CPU churn. It is set below, only for dispatched JSON-RPC
                # method calls.
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
                # Real memory traffic: mark activity only for dispatched JSON-RPC
                # method calls so background consolidation defers while the user is
                # actively recalling/capturing. Control/status probes never reach
                # here, so they no longer keep the daemon awake.
                self.last_activity_ts = time.monotonic()
                try:
                    result = await self._dispatch_jsonrpc_method(method, params)
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
        if IS_WINDOWS:
            # Windows: TCP server on loopback; socket_path is unused
            server, actual_addr, needs_cleanup = await start_ipc_server(self.handle)
            try:
                async with server:
                    await self.shutdown_event.wait()
                    server.close()
                    await server.wait_closed()
            finally:
                if needs_cleanup:
                    shutdown_ipc(actual_addr)
            return

        if socket_path is None:
            env_path = os.environ.get("IAI_DAEMON_SOCKET_PATH")
            socket_path = Path(env_path) if env_path else SOCKET_PATH

        inherited = _inherit_activated_socket()
        if inherited is not None:
            server = await asyncio.start_unix_server(self.handle, sock=inherited)
            needs_cleanup = False
            actual_addr: Any = str(socket_path)
        else:
            cleanup_ipc_address(socket_path)
            socket_path.parent.mkdir(parents=True, exist_ok=True)
            server, actual_addr, needs_cleanup = await start_ipc_server(self.handle, socket_path)
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
            if inherited is None and needs_cleanup:
                shutdown_ipc(actual_addr)
