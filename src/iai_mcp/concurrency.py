from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)

SOCKET_PATH: Path = Path.home() / ".iai-mcp" / ".daemon.sock"

# ---------------------------------------------------------------------------
# Foreground-activity beacon
# ---------------------------------------------------------------------------
# A foreground read (a live recall serving a user) and background ingestion
# (the deferred-capture drain, other maintenance loops) contend for the same
# shared writer connection and the GIL. The beacon gives background loops a
# cheap way to yield: the recall path stamps it at dispatch, and a polite
# background loop pauses between work items while the stamp is fresh. Purely
# advisory — a loop that never checks it is merely impolite, never wrong —
# and every waiter bounds its total pause so background work always completes
# even under a continuous stream of foreground reads (fail-open).

_foreground_last_touch: float = 0.0
_foreground_inflight: int = 0
_foreground_touch_lock = threading.Lock()

#: How long after the last foreground stamp the beacon still reads "active"
#: when no in-flight marker is held (a stamp-only caller).
FOREGROUND_RECENT_WINDOW_S = 0.75

#: Ceiling on how long a stuck in-flight marker can hold the beacon active: a
#: read that failed to clear its marker (a crashed thread) must not silence
#: background work forever. Past this age since the last stamp, the beacon
#: reads quiet regardless of the in-flight count.
FOREGROUND_INFLIGHT_CEILING_S = 30.0


def foreground_touch() -> None:
    """Stamp the beacon: a foreground read is being served right now."""
    global _foreground_last_touch  # noqa: PLW0603
    with _foreground_touch_lock:
        _foreground_last_touch = time.monotonic()


def foreground_begin() -> None:
    """Mark a foreground read in flight (paired with `foreground_end`)."""
    global _foreground_inflight, _foreground_last_touch  # noqa: PLW0603
    with _foreground_touch_lock:
        _foreground_inflight += 1
        _foreground_last_touch = time.monotonic()


def foreground_end() -> None:
    """Clear one in-flight foreground marker. Never goes below zero."""
    global _foreground_inflight  # noqa: PLW0603
    with _foreground_touch_lock:
        if _foreground_inflight > 0:
            _foreground_inflight -= 1


def foreground_recent(window_s: float = FOREGROUND_RECENT_WINDOW_S) -> bool:
    """True while a foreground read is in flight (bounded by the stuck-marker
    ceiling) or one was stamped within the last `window_s`."""
    with _foreground_touch_lock:
        last = _foreground_last_touch
        inflight = _foreground_inflight
    if last <= 0.0:
        return False
    age = time.monotonic() - last
    if inflight > 0:
        return age < FOREGROUND_INFLIGHT_CEILING_S
    return age < window_s


def foreground_backoff(max_wait_s: float = 2.0, step_s: float = 0.05) -> None:
    """Pause while the beacon is fresh, up to `max_wait_s` total.

    For background loops between work items: yields the GIL and the shared
    connection to in-flight foreground reads. Returns as soon as the beacon
    goes quiet or the bound is reached — never an indefinite stall.
    """
    waited = 0.0
    while waited < max_wait_s and foreground_recent():
        time.sleep(step_s)
        waited += step_s


def cleanup_stale_socket(path: Path = SOCKET_PATH) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        try:
            path.unlink()
        except OSError:
            pass


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


def _status_embed_identity(store: Any) -> "dict | None":
    """runtime-vs-stored vector identity, computed ONCE at daemon boot.

    The status handler serves this from the daemon's state dict — never
    inline: the status path is the liveness watchdog's signal, and this
    computation takes _conn_lock, which a consolidating writer can hold
    for seconds. `match` is True on an unstamped store — protection
    begins at the first stamp.
    """
    try:
        from iai_mcp.embed import (
            EMBED_IDENTITY_META_KEY,
            effective_model_identity,
            embedder_for_store,
        )

        runtime_id = effective_model_identity(
            embedder_for_store(store, allow_identity_mismatch=True)
        )
        stored_id = None
        db = getattr(store, "db", None)
        if db is not None:
            with db._conn_lock:
                row = db._conn.execute(
                    "SELECT value FROM _hippo_meta WHERE key = ?",
                    (EMBED_IDENTITY_META_KEY,),
                ).fetchone()
            stored_id = row["value"] if row is not None else None
        return {
            "runtime": runtime_id,
            "stored": stored_id,
            "match": stored_id is None or stored_id == runtime_id,
        }
    except Exception as exc:  # noqa: BLE001 -- status must never fail on this
        return {"error": str(exc)[:160]}


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
    from iai_mcp.daemon_state import update_state

    # Handlers mutate the daemon's shared in-memory `state` for coherent
    # same-process reads, but persistence goes through update_state with a
    # targeted mutator: a whole-dict save of the long-held shared dict
    # erases every key another writer persisted since daemon boot.
    def _persist(*keys: str) -> None:
        def _apply(d: dict) -> None:
            for k in keys:
                d[k] = state[k]

        update_state(_apply)

    if req_type == "status":
        fsm_state = state.get("fsm_state", "WAKE")
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
            "quiet_window_source": state.get("quiet_window_source"),
            "pending_digest": truncated_digest,
            "daemon_started_at": started_at,
            "scheduler_paused": bool(state.get("scheduler_paused", False)),
            "ann_pool_health": state.get("ann_pool_health"),
            # Coerced to None unless it is a plain dict: a wrong-typed value
            # (a stray coroutine from a broken boot wire) would make
            # json.dumps fail OUTSIDE the dispatcher's guard and take every
            # status reply down with it.
            "embed_identity": (
                state.get("embed_identity")
                if isinstance(state.get("embed_identity"), dict)
                else None
            ),
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
            await asyncio.to_thread(_persist, "user_sleep_request")
        except Exception as exc:  # noqa: BLE001 -- socket must never crash daemon
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "state": "TRANSITIONING"}

    if req_type == "force_wake":
        ts = str(req.get("ts", ""))
        state["force_wake_request"] = {"ts": ts, "pending": True}
        try:
            await asyncio.to_thread(_persist, "force_wake_request")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "wake_queued"}

    if req_type == "force_rem":
        ts = str(req.get("ts", ""))
        state["force_rem_request"] = {"ts": ts, "pending": True}
        try:
            await asyncio.to_thread(_persist, "force_rem_request")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "rem_queued"}

    if req_type == "pause":
        state["scheduler_paused"] = True
        try:
            await asyncio.to_thread(_persist, "scheduler_paused")
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "paused": True}

    if req_type == "resume":
        state["scheduler_paused"] = False
        try:
            await asyncio.to_thread(_persist, "scheduler_paused")
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

        def _persist_session_open(d: dict) -> None:
            d["last_session_open"] = state["last_session_open"]
            d["hippea_cascade_request"] = state["hippea_cascade_request"]
            # Merge only this session's entry: the on-disk map may hold
            # prunes/consumes this process has not observed.
            fresh_ft = d.get("first_turn_pending")
            if not isinstance(fresh_ft, dict):
                fresh_ft = {}
            fresh_ft[session_id] = now_iso
            d["first_turn_pending"] = fresh_ft

        try:
            await asyncio.to_thread(update_state, _persist_session_open)
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "reason": "state_write_failed", "error": str(exc)[:200]}
        return {"ok": True, "reason": "session_open_queued"}

    if req_type == "embed_cue":
        cue = str(req.get("cue", ""))
        try:
            from iai_mcp.embed import embed_query as _embed_query, embedder_for_store
            embedder = embedder_for_store(store)
            vec = await asyncio.to_thread(_embed_query, embedder, cue)
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
    cleanup_stale_socket(socket_path)
    socket_path.parent.mkdir(parents=True, exist_ok=True)

    _supports_cleanup_socket = False
    try:
        import inspect as _inspect
        import asyncio as _asyncio_mod
        _loop_sig = _inspect.signature(
            _asyncio_mod.get_event_loop_policy().new_event_loop().create_unix_server
        )
        _supports_cleanup_socket = "cleanup_socket" in _loop_sig.parameters
    except (TypeError, ValueError, AttributeError):
        _supports_cleanup_socket = False

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

    from iai_mcp._ipc import IS_WINDOWS, start_ipc_server
    if IS_WINDOWS:
        server, _actual_addr, _needs_cleanup = await start_ipc_server(
            handle, socket_path,
        )
    else:
        _server_kwargs = {"cleanup_socket": True} if _supports_cleanup_socket else {}
        server = await asyncio.start_unix_server(
            handle, path=str(socket_path), **_server_kwargs,
        )
        try:
            os.chmod(str(socket_path), 0o600)
        except OSError:
            pass

    try:
        async with server:
            await shutdown.wait()
    finally:
        if not _supports_cleanup_socket:
            try:
                socket_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
