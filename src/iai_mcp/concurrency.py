from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"


def cleanup_stale_socket(path: Path = SOCKET_PATH) -> None:
    from iai_mcp._ipc import cleanup_ipc_address
    cleanup_ipc_address(path)


def _validate_socket_message(req: dict) -> tuple[bool, str | None]:
    req_type = req.get("type")
    if not isinstance(req_type, str):
        return False, "type must be a string"

    if req_type == "status":
        return True, None

    if req_type == "user_initiated_sleep":
        reason = req.get("reason")
        ts = req.get("ts")
        if not isinstance(reason, str):
            return False, "reason must be a string"
        if not isinstance(ts, str):
            return False, "ts must be a string"
        return True, None

    if req_type in ("force_wake", "force_rem"):
        ts = req.get("ts")
        if not isinstance(ts, str):
            return False, "ts must be a string"
        return True, None

    if req_type in ("pause", "resume"):
        if "seconds" in req:
            seconds = req.get("seconds")
            if not isinstance(seconds, int) or isinstance(seconds, bool):
                return False, "seconds must be an int"
        return True, None

    if req_type == "session_open":
        if "session_id" in req and not isinstance(req["session_id"], str):
            return False, "session_id must be a string"
        if "ts" in req and not isinstance(req["ts"], str):
            return False, "ts must be a string"
        return True, None

    if req_type == "embed_cue":
        cue = req.get("cue")
        if not isinstance(cue, str):
            return False, "cue must be a string"
        return True, None

    return True, None


async def _dispatch_socket_request(
    req: dict,
    store: Any,
    state: dict,
) -> dict:
    if not isinstance(req, dict):
        return {
            "ok": False,
            "reason": "invalid_message",
            "error": "request must be a JSON object",
        }

    ok, err = _validate_socket_message(req)
    if not ok:
        return {
            "ok": False,
            "reason": "invalid_message",
            "error": err or "schema_validation_failed",
        }

    req_type = req.get("type")

    from datetime import datetime, timezone

    from iai_mcp import __version__ as pkg_version
    from iai_mcp.daemon_state import save_state

    if req_type == "status":
        fsm_state = state.get("fsm_state", "WAKE")
        try:
            from iai_mcp.lifecycle_state import LIFECYCLE_STATE_PATH, LifecycleState

            lifecycle_raw = json.loads(
                LIFECYCLE_STATE_PATH.read_text(encoding="utf-8")
            )
            lifecycle_state = (
                lifecycle_raw.get("current_state")
                if isinstance(lifecycle_raw, dict)
                else None
            )
            if lifecycle_state in {item.value for item in LifecycleState}:
                fsm_state = lifecycle_state
        except Exception:  # noqa: BLE001 -- status must stay best-effort
            pass
        started_at = state.get("daemon_started_at")
        uptime_sec: float | None = None
        if started_at:
            try:
                start_dt = datetime.fromisoformat(started_at)
                uptime_sec = (datetime.now(timezone.utc) - start_dt).total_seconds()
            except (TypeError, ValueError):
                uptime_sec = None

        pending_digest = state.get("pending_digest")
        if isinstance(pending_digest, dict):
            truncated_digest = {
                "rem_cycles_completed": pending_digest.get("rem_cycles_completed", 0),
                "episodes_processed": pending_digest.get("episodes_processed", 0),
                "schemas_induced_tier0": pending_digest.get(
                    "schemas_induced_tier0", 0,
                ),
                "claude_call_used": pending_digest.get("claude_call_used", False),
            }
        else:
            truncated_digest = None

        return {
            "ok": True,
            "state": fsm_state,
            "uptime_sec": uptime_sec,
            "version": pkg_version,
            "fsm_state": fsm_state,
            "last_tick_at": state.get("last_tick_at"),
            "quiet_window": state.get("quiet_window"),
            "pending_digest": truncated_digest,
            "daemon_started_at": started_at,
            "scheduler_paused": bool(state.get("scheduler_paused", False)),
        }

    if req_type == "user_initiated_sleep":
        current_fsm = state.get("fsm_state", "WAKE")
        if current_fsm in ("SLEEP", "DREAMING", "TRANSITIONING"):
            return {"ok": False, "reason": "already_sleeping"}

        reason = str(req.get("reason", ""))[:500]
        ts = str(req.get("ts", ""))
        state["user_sleep_request"] = {
            "reason": reason,
            "ts": ts,
            "pending": True,
        }
        try:
            await asyncio.to_thread(save_state, state)
        except Exception as exc:  # noqa: BLE001 -- socket must never crash daemon
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "state": "TRANSITIONING"}

    if req_type == "force_wake":
        ts = str(req.get("ts", ""))
        state["force_wake_request"] = {"ts": ts, "pending": True}
        try:
            await asyncio.to_thread(save_state, state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "wake_queued"}

    if req_type == "force_rem":
        ts = str(req.get("ts", ""))
        state["force_rem_request"] = {"ts": ts, "pending": True}
        try:
            await asyncio.to_thread(save_state, state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "rem_queued"}

    if req_type == "pause":
        state["scheduler_paused"] = True
        try:
            await asyncio.to_thread(save_state, state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "paused": True}

    if req_type == "resume":
        state["scheduler_paused"] = False
        try:
            await asyncio.to_thread(save_state, state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "paused": False}

    if req_type == "session_open":
        session_id = str(req.get("session_id", ""))[:128]
        ts = str(req.get("ts", ""))
        state["last_session_open"] = {"session_id": session_id, "ts": ts}
        first_turn = state.setdefault("first_turn_pending", {})
        now_iso = datetime.now(timezone.utc).isoformat()
        if isinstance(first_turn, dict):
            first_turn[session_id] = now_iso
        else:
            state["first_turn_pending"] = {session_id: now_iso}
        state["hippea_cascade_request"] = {
            "session_id": session_id,
            "ts": ts,
            "pending": True,
        }
        try:
            await asyncio.to_thread(save_state, state)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "session_open_queued"}

    if req_type == "embed_cue":
        cue = str(req.get("cue", ""))
        try:
            from iai_mcp.embed import embedder_for_store
            embedder = embedder_for_store(store)
            vec = await asyncio.to_thread(embedder.embed, cue)
            if len(vec) != embedder.DIM:
                return {
                    "ok": False,
                    "reason": "embed_dim_mismatch",
                    "error": f"embedder returned {len(vec)} dims, expected {embedder.DIM}",
                }
            return {"ok": True, "embedding": list(vec)}
        except Exception as exc:  # noqa: BLE001 -- embedder not ready / cold
            return {"ok": False, "reason": "daemon_not_ready", "error": str(exc)[:200]}

    return {
        "ok": False,
        "reason": "unknown_message_type",
        "type": req_type,
    }


async def serve_control_socket(
    store: Any,
    state: dict,
    shutdown: asyncio.Event,
    *,
    dispatcher: Callable[[dict], Awaitable[dict]] | None = None,
    socket_path: Path = SOCKET_PATH,
) -> None:
    from iai_mcp._ipc import IS_WINDOWS, cleanup_ipc_address, start_ipc_server, shutdown_ipc

    cleanup_ipc_address(socket_path)
    if not IS_WINDOWS:
        socket_path.parent.mkdir(parents=True, exist_ok=True)

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            line = await reader.readline()
            if not line:
                return
            try:
                req = json.loads(line)
            except (TypeError, ValueError) as exc:
                writer.write((json.dumps({"error": f"invalid_json: {exc}"}) + "\n").encode("utf-8"))
                await writer.drain()
                return
            try:
                if dispatcher is not None:
                    resp = await dispatcher(req)
                else:
                    resp = await _dispatch_socket_request(req, store, state)
            except Exception as exc:  # noqa: BLE001 -- socket must never crash daemon
                logger.warning("socket_dispatch_error", extra={"err": str(exc)[:200]})
                resp = {"error": str(exc)}
            writer.write((json.dumps(resp) + "\n").encode("utf-8"))
            await writer.drain()
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except (OSError, ConnectionError):  # noqa: BLE001 -- cleanup is best-effort
                pass

    server, actual_addr, needs_cleanup = await start_ipc_server(handle, socket_path)
    if not IS_WINDOWS:
        try:
            os.chmod(str(socket_path), 0o600)
        except OSError:
            pass

    try:
        async with server:
            await shutdown.wait()
    finally:
        if needs_cleanup:
            shutdown_ipc(actual_addr)
