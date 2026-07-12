from __future__ import annotations

import argparse
import os
import subprocess
import sys
from typing import Any

from iai_mcp import __version__


_CYAN = "\x1b[96m"
_DIM = "\x1b[2m"
_RESET = "\x1b[0m"


def _color(text: str, *, color: str = _CYAN) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"{color}{text}{_RESET}"


_LOGO_LINES: tuple[str, ...] = (
    "  ██╗ █████╗ ██╗     ██████╗██╗     ██╗",
    "  ██║██╔══██╗██║    ██╔════╝██║     ██║",
    "  ██║███████║██║    ██║     ██║     ██║",
    "  ██║██╔══██║██║    ██║     ██║     ██║",
    "  ██║██║  ██║██║    ╚██████╗███████╗██║",
    "  ╚═╝╚═╝  ╚═╝╚═╝     ╚═════╝╚══════╝╚═╝",
)


def _print_logo() -> None:
    for line in _LOGO_LINES:
        print(_color(line))
    print(_color(f"  iai-cli · terminal memory for your agent · v{__version__}  ", color=_DIM))
    print()


def _format_hits(hits: list[dict], *, max_surface_chars: int = 200) -> str:
    lines: list[str] = []
    for h in hits:
        surface = (h.get("literal_surface") or h.get("surface") or "")[:max_surface_chars]
        score_raw = h.get("score") or h.get("final_score") or 0.0
        try:
            score = float(score_raw)
        except (TypeError, ValueError):
            score = 0.0
        lines.append(f"  {score:.3f}  {surface}")
    return "\n".join(lines)


def cmd_recall(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.cli import _send_jsonrpc_request

    cue = args.cue
    limit = max(1, int(args.limit))
    json_mode = bool(getattr(args, "json", False))

    _raw_rt = os.environ.get("IAI_RECALL_READ_TIMEOUT", "")
    try:
        _recall_read_timeout = max(0.5, float(_raw_rt))
    except (ValueError, TypeError):
        # Must survive a consolidating daemon: real hits measured at 5-8s
        # during a heavy sleep window, and a slow true answer beats an
        # instant recency-junk fallback. Warm-path recall stays ~20ms —
        # this ceiling only bites while the daemon is grinding.
        _recall_read_timeout = 10.0

    resp = _send_jsonrpc_request(
        "memory_recall",
        {"cue": cue, "budget_tokens": limit * 300},
        read_timeout=_recall_read_timeout,
    )
    if isinstance(resp, dict) and "result" in resp and isinstance(resp["result"], dict):
        result = resp["result"]
        hits_raw = result.get("hits") or []
        if isinstance(hits_raw, list):
            hits = hits_raw[:limit]
            if json_mode:
                payload = {"hits": hits, "_source": "daemon", "count": len(hits)}
                print(_json.dumps(payload))
                return 0
            if not hits:
                print(_color("(no hits)"))
                return 0
            print(_color(f"via daemon  [N={len(hits)}]"))
            print(_format_hits(hits))
            return 0


    _store_root_direct: "str | None" = None
    _store_reached: bool = False
    try:
        from pathlib import Path as _Path
        from iai_mcp.semantic_recall import recall_semantic_warm as _recall_warm

        _store_env = os.environ.get("IAI_MCP_STORE")
        store_root = _Path(_store_env) if _store_env else _Path.home() / ".iai-mcp"
        _store_root_direct = str(store_root)

        _hippo_db = store_root / "hippo" / "brain.sqlite3"
        if not _hippo_db.exists():
            raise FileNotFoundError(f"store absent: {_hippo_db}")

        _store_reached = True
        degraded_hits = _recall_warm(store_root, cue, n=limit)

        if json_mode:
            _src = (degraded_hits[0].get("_source", "direct-store") if degraded_hits else "direct-store")
            payload = {
                "hits": degraded_hits,
                "_source": _src,
                "count": len(degraded_hits),
            }
            print(_json.dumps(payload))
            return 0
        if not degraded_hits:
            print(_color("(no hits)", color=_DIM))
            return 0
        print(_color("(daemon unreachable — store recall)", color=_DIM), file=sys.stderr)
        for h in degraded_hits:
            surface = (h.get("literal_surface") or "")[:200]
            score_raw = h.get("score") or 0.0
            try:
                score = float(score_raw)
            except (TypeError, ValueError):
                score = 0.0
            print(f"  {score:.3f}  {surface}")
        return 0
    except Exception:  # noqa: BLE001
        if _store_reached:
            pass

    print(_color("(daemon unreachable — bank-fallback)", color=_DIM), file=sys.stderr)
    completed = subprocess.run(  # noqa: S603 -- argv list, no shell
        ["iai-mcp", "bank-recall", "--query", cue, "--limit", str(limit)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode == 0:
        sys.stdout.write(completed.stdout)
        return 0
    sys.stderr.write(completed.stderr or "recall failed\n")
    return 1


def cmd_temporal_recall(args: argparse.Namespace) -> int:
    """Time-bounded recall: as_of (records) + changed_since (events).

    Daemon-up path: JSON-RPC `memory_temporal_recall`.
    Daemon-down path: opens HippoDB SHARED 0.25s read-only in-process and
    materialises the AES key via CryptoKey(store_root).get_or_create().
    """
    import json as _json

    from iai_mcp.cli import _send_jsonrpc_request

    cue = (getattr(args, "cue", "") or "").strip()
    as_of_raw = getattr(args, "as_of", None) or None
    changed_since_raw = getattr(args, "changed_since", None) or None
    limit = max(1, int(getattr(args, "limit", 10)))
    json_mode = bool(getattr(args, "json", False))

    _raw_rt = os.environ.get("IAI_RECALL_READ_TIMEOUT", "")
    try:
        _recall_read_timeout = max(0.5, float(_raw_rt))
    except (ValueError, TypeError):
        # Must survive a consolidating daemon: real hits measured at 5-8s
        # during a heavy sleep window, and a slow true answer beats an
        # instant recency-junk fallback. Warm-path recall stays ~20ms —
        # this ceiling only bites while the daemon is grinding.
        _recall_read_timeout = 10.0

    rpc_params: dict[str, Any] = {"limit": limit}
    if cue:
        rpc_params["cue"] = cue
    if as_of_raw is not None:
        rpc_params["as_of"] = as_of_raw
    if changed_since_raw is not None:
        rpc_params["changed_since"] = changed_since_raw

    resp = _send_jsonrpc_request(
        "memory_temporal_recall",
        rpc_params,
        read_timeout=_recall_read_timeout,
    )
    if isinstance(resp, dict) and "result" in resp and isinstance(resp["result"], dict):
        result = resp["result"]
        hits = result.get("hits") or []
        events = result.get("changed_since_events") or []
        if isinstance(hits, list) and isinstance(events, list):
            if json_mode:
                payload = {
                    "hits": hits[:limit],
                    "changed_since_events": events,
                    "_source": "daemon",
                    "count": len(hits[:limit]),
                }
                scope = result.get("_scope")
                if scope is not None:
                    payload["_scope"] = scope
                print(_json.dumps(payload))
                return 0
            shown = hits[:limit]
            if not shown and not events:
                print(_color("(no temporal hits)"))
                return 0
            print(_color(f"via daemon  [records={len(shown)} events={len(events)}]"))
            print(_format_hits(shown))
            return 0

    return _cmd_temporal_recall_direct(
        cue=cue,
        as_of_raw=as_of_raw,
        changed_since_raw=changed_since_raw,
        limit=limit,
        json_mode=json_mode,
    )


def _cmd_temporal_recall_direct(
    *,
    cue: str,
    as_of_raw: str | None,
    changed_since_raw: str | None,
    limit: int,
    json_mode: bool,
) -> int:
    """Daemon-down direct rail for cmd_temporal_recall.

    Records: degraded_temporal_recall(store_root, cue, as_of, limit).
    Events: open HippoDB SHARED 0.25s read-only, parameterised SELECT under
    _conn_lock; AES key via CryptoKey(store_root).get_or_create(); ascii AAD
    for the events table.
    """
    import json as _json
    import logging as _logging
    from pathlib import Path as _Path

    from iai_mcp.store._store import _normalize_ts_for_compare

    store_env = os.environ.get("IAI_MCP_STORE")
    store_root = _Path(store_env) if store_env else _Path.home() / ".iai-mcp"

    as_of_norm: str | None = None
    if as_of_raw is not None:
        try:
            as_of_norm = _normalize_ts_for_compare(as_of_raw)
        except ValueError as exc:
            sys.stderr.write(f"as_of must be ISO-8601: {exc}\n")
            return 1

    changed_since_norm: str | None = None
    if changed_since_raw is not None:
        try:
            changed_since_norm = _normalize_ts_for_compare(changed_since_raw)
        except ValueError as exc:
            sys.stderr.write(f"changed_since must be ISO-8601: {exc}\n")
            return 1

    hits_out: list[dict] = []
    if as_of_norm is not None:
        try:
            from iai_mcp.hippo._recall import degraded_temporal_recall
            raw_hits = degraded_temporal_recall(
                store_root=store_root,
                cue=cue,
                as_of=as_of_norm,
                limit=limit,
            )
            hits_out = list(raw_hits or [])
        except Exception as exc:  # noqa: BLE001 -- direct-rail must never crash
            _logging.getLogger(__name__).debug(
                "direct_temporal_records_failed err=%s", str(exc)[:120],
            )

    events_out: list[dict] = []
    if changed_since_norm is not None:
        events_out = _read_events_direct(
            store_root=store_root,
            changed_since_norm=changed_since_norm,
            limit=limit,
        )

    if json_mode:
        payload: dict[str, Any] = {
            "hits": hits_out,
            "changed_since_events": events_out,
            "_source": "direct-store",
            "count": len(hits_out),
        }
        if as_of_norm is not None or changed_since_norm is not None:
            payload["_scope"] = "committed_by_time_t"
        print(_json.dumps(payload))
        return 0

    print(_color("(daemon unreachable — direct store rail)", color=_DIM), file=sys.stderr)
    if not hits_out and not events_out:
        print(_color("(no temporal hits)"))
        return 0
    if hits_out:
        print(_color(f"records  [N={len(hits_out)}]"))
        print(_format_hits(hits_out))
    if events_out:
        print(_color(f"events  [N={len(events_out)}]"))
        for ev in events_out:
            ts = (ev.get("ts") or "")[:19]
            kind = ev.get("kind") or "?"
            print(f"  [{ts}] {kind}")
    return 0


def _read_events_direct(
    *,
    store_root,
    changed_since_norm: str,
    limit: int,
) -> list[dict]:
    """Daemon-down events read: SHARED HippoDB + parameterised SELECT + per-row
    AES decrypt with ascii AAD. Mirrors events.py::query_events inline loop but
    sources the AES key from CryptoKey(store_root).get_or_create() because no
    MemoryStore instance exists in this code path.
    """
    import logging as _logging

    from iai_mcp.crypto import CryptoKey, decrypt_field, is_encrypted
    from iai_mcp.hippo import AccessMode, HippoDB

    # events.ts is written by write_event() via the default datetime adapter
    # (space-form TEXT). The events.py read path normalises the boundary to
    # T-form then swaps to space-form for raw lexicographic compare; mirror
    # that here so the direct-rail matches the daemon-up semantics.
    boundary = changed_since_norm.replace("T", " ")

    sql = (
        "SELECT id, kind, severity, domain, ts, data_json, session_id,"
        " source_ids_json FROM events WHERE ts > ?"
        " ORDER BY ts DESC LIMIT ?"
    )

    raw_rows: list[dict] = []
    db: "HippoDB | None" = None
    try:
        db = HippoDB(
            store_root,
            access_mode=AccessMode.SHARED,
            read_only=True,
            _lock_timeout_override=0.25,
        )
        with db._conn_lock:
            cursor = db._conn.execute(sql, (boundary, int(limit)))
            raw_rows = [dict(r) for r in cursor.fetchall()]
    except Exception as exc:  # noqa: BLE001 -- direct-rail fail-soft
        _logging.getLogger(__name__).debug(
            "direct_temporal_events_open_failed err=%s", str(exc)[:120],
        )
        raw_rows = []
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:  # noqa: BLE001
                pass

    if not raw_rows:
        return []

    crypto_key: bytes | None = None
    try:
        crypto_key = CryptoKey(store_root=store_root).get_or_create()
    except Exception as exc:  # noqa: BLE001 -- no key available: leave ciphertext as-is
        _logging.getLogger(__name__).debug(
            "direct_temporal_events_key_unavailable err=%s", str(exc)[:80],
        )

    import json as _json

    out: list[dict] = []
    for row in raw_rows:
        raw_data = row.get("data_json") or "{}"
        if is_encrypted(raw_data) and crypto_key is not None:
            try:
                aad = str(row.get("id") or "").encode("ascii")
                raw_data = decrypt_field(raw_data, crypto_key, associated_data=aad)
            except (OSError, ValueError, RuntimeError) as exc:
                _logging.getLogger(__name__).debug(
                    "event_decrypt_failed id=%s err=%s",
                    row.get("id"), str(exc)[:80],
                )
                raw_data = "{}"
        try:
            data = _json.loads(raw_data)
        except (TypeError, _json.JSONDecodeError):
            data = {}
        try:
            source_ids = _json.loads(row.get("source_ids_json") or "[]")
        except (TypeError, _json.JSONDecodeError):
            source_ids = []
        ts_raw = row.get("ts")
        ts_str = ts_raw if isinstance(ts_raw, str) else (
            ts_raw.isoformat() if ts_raw is not None and hasattr(ts_raw, "isoformat") else str(ts_raw or "")
        )
        out.append({
            "id": row.get("id"),
            "kind": row.get("kind"),
            "severity": row.get("severity") or None,
            "domain": row.get("domain") or None,
            "ts": ts_str,
            "data": data,
            "session_id": row.get("session_id"),
            "source_ids": source_ids,
            "_source": "direct-store",
        })
    return out


def cmd_ask(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.cli import _send_jsonrpc_request

    question = args.question
    limit = max(1, int(args.limit))

    recall_resp = _send_jsonrpc_request(
        "memory_recall",
        {"cue": question, "budget_tokens": limit * 300},
    )
    hits: list[dict] = []
    if isinstance(recall_resp, dict) and isinstance(recall_resp.get("result"), dict):
        raw = recall_resp["result"].get("hits") or []
        if isinstance(raw, list):
            hits = raw[:limit]

    if not hits:
        print(
            "(no memories matched the question — daemon down, or empty store)",
            file=sys.stderr,
        )
        return 1

    memories = [
        {
            "id": str(h.get("id") or h.get("record_id") or "?"),
            "surface": (h.get("literal_surface") or h.get("surface") or "")[:500],
        }
        for h in hits
    ]
    prompt = (
        "Answer the question grounded in the memories. Cite by id. "
        "Reply in 1-3 sentences, plain text.\n\n"
        f"Question: {question}\n"
        f"Memories: {_json.dumps(memories)}"
    )

    from iai_mcp.claude_cli import invoke_claude_sync

    result = invoke_claude_sync(prompt, model="haiku", timeout_sec=60.0)
    if not result.get("ok"):
        reason = result.get("reason", "unknown")
        print(f"ask failed: {reason}", file=sys.stderr)
        return 1

    data = result.get("data") or {}
    answer = ""
    if isinstance(data, dict):
        answer = str(data.get("result") or data.get("text") or "").strip()

    if not answer:
        print("ask failed: empty answer from claude -p", file=sys.stderr)
        return 1

    print(answer)
    ids = ", ".join(m["id"] for m in memories)
    print(_color(f"\nSources: {ids}", color=_DIM))
    return 0


def cmd_status(args: argparse.Namespace) -> int:  # noqa: ARG001 -- argparse contract
    from iai_mcp.claude_cli import verify_credentials_subscription
    from iai_mcp.cli import _send_jsonrpc_request

    daemon_state = "DOWN"
    record_count = "?"
    regime = "?"
    resp = _send_jsonrpc_request("topology", {})
    if isinstance(resp, dict):
        result = resp.get("result")
        if isinstance(result, dict):
            daemon_state = "UP"
            n = result.get("N")
            record_count = str(n) if n is not None else "?"
            regime = str(result.get("regime") or "?")

    creds = verify_credentials_subscription()
    if creds.get("ok"):
        sub_label = creds.get("subscription_type") or creds.get("billing_type") or "active"
    else:
        sub_label = f"missing ({creds.get('reason', 'unknown')})"

    print(_color("iai status"))
    print(f"  daemon         {daemon_state}")
    print(f"  records        {record_count}")
    print(f"  regime         {regime}")
    print(f"  subscription   {sub_label}")
    return 0


def cmd_capture(args: argparse.Namespace) -> int:
    from iai_mcp.cli import _send_jsonrpc_request

    text = args.text
    session_id = getattr(args, "session_id", None) or "-"
    use_json = getattr(args, "json", False)

    resp = _send_jsonrpc_request(
        "memory_capture",
        {
            "text": text,
            "session_id": session_id,
            "tier": "episodic",
        },
    )
    if not isinstance(resp, dict):
        store_root_env = os.environ.get("IAI_MCP_STORE")
        if store_root_env:
            try:
                from pathlib import Path as _Path
                from iai_mcp.direct_write import write_turn_direct
                store_root = _Path(store_root_env)
                result = write_turn_direct(
                    store_root=store_root,
                    text=text,
                    session_id=session_id,
                    role="user",
                    deferred_embedding=True,
                )
                rid = result.get("record_id") or "?"
                if use_json:
                    import json as _json
                    print(_json.dumps({"id": rid, "status": result.get("status"), "_source": "direct-store"}))
                else:
                    print(_color(f"captured  id={rid}  [direct-store]"))
                return 0
            except Exception as exc:
                print(f"capture failed (direct write): {exc}", file=sys.stderr)
                return 1
        print(
            "capture failed: daemon unreachable. "
            "Start the daemon with `iai-mcp daemon start`.",
            file=sys.stderr,
        )
        return 1
    if "error" in resp:
        err = resp["error"]
        msg = err.get("message", "unknown") if isinstance(err, dict) else str(err)
        print(f"capture failed: {msg}", file=sys.stderr)
        return 1
    if "result" in resp:
        result = resp["result"]
        rid: Any = "?"
        if isinstance(result, dict):
            rid = result.get("id") or result.get("record_id") or "?"
        if use_json:
            import json as _json
            print(_json.dumps({"id": rid, "status": "inserted", "_source": "daemon"}))
        else:
            print(_color(f"captured  id={rid}"))
        return 0
    print("capture failed: malformed daemon reply", file=sys.stderr)
    return 1


def _resolve_store_root():
    from pathlib import Path

    env = os.environ.get("IAI_MCP_STORE")
    return Path(env) if env else Path.home() / ".iai-mcp"


def _open_store_shared(store_root=None):
    """Awake-path store open: SHARED coexists with the daemon's lock, and a
    non-owning handle never persists the hnsw index (disk hnsw is derived
    data owned by the daemon's rebuild)."""
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore

    return MemoryStore(
        store_root if store_root is not None else _resolve_store_root(),
        access_mode=AccessMode.SHARED,
        persist_index=False,
    )


_STORE_BUSY_HINT = (
    "memory is busy — the daemon owns the store right now. "
    "Retry in a moment, or stop the daemon (iai-mcp daemon stop) "
    "for direct access."
)


def _daemon_socket_path():
    return _resolve_store_root() / ".daemon.sock"


def _daemon_alive() -> bool:
    try:
        return _daemon_socket_path().is_socket()
    except OSError:
        return False


def _relay_rpc(method: str, params: dict, timeout: float = 900.0):
    """One JSON-RPC call over the daemon socket (the same wire brainview's
    relay uses). Returns the result payload, or {"status": "daemon_down"|
    "error", ...} — transport failures never raise into a CLI command."""
    import json as _json
    import socket as _socket
    import time as _time

    req = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    deadline = _time.monotonic() + timeout
    try:
        from iai_mcp._ipc import make_sync_ipc_socket, send_sync_auth_token
        s, _addr = make_sync_ipc_socket(addr=str(_daemon_socket_path()))
        s.settimeout(timeout)
        s.connect(_addr)
        send_sync_auth_token(s)
        s.sendall((_json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n") and len(buf) < (64 << 20):
            remaining = deadline - _time.monotonic()
            if remaining <= 0:
                raise _socket.timeout("relay budget exhausted")
            s.settimeout(remaining)
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            buf += chunk
        s.close()
        resp = _json.loads(buf or b"{}")
    except (OSError, ValueError) as exc:
        return {"status": "daemon_down", "reason": str(exc)[:200]}
    if isinstance(resp, dict) and "error" in resp:
        return {"status": "error", "reason": str(resp["error"])[:200]}
    return resp.get("result") if isinstance(resp, dict) else resp


def _relay_teach_verb(kwargs: dict, timeout: float = 900.0):
    return _relay_rpc(
        "brain_view", {"verb": "teach", "kwargs": kwargs}, timeout=timeout,
    )


def _is_relay_failure(res) -> bool:
    return isinstance(res, dict) and res.get("status") in ("daemon_down", "error")


def _open_store_shared_or_fail(purpose: str):
    """Direct store open with the engine/lock rejections translated into a
    clean actionable line on stderr. Returns None on failure — the caller
    exits 1 instead of dumping a traceback from a headline user command."""
    try:
        return _open_store_shared()
    except Exception as exc:  # noqa: BLE001 -- CLI boundary: translate, never traceback
        msg = str(exc)
        name = type(exc).__name__
        if "locked" in msg.lower() or name in (
            "ConsolidationPendingError",
            "HippoLockHeldError",
        ):
            print(f"{purpose} failed: {_STORE_BUSY_HINT}", file=sys.stderr)
        else:
            print(
                f"{purpose} failed: cannot open memory store "
                f"({name}: {msg[:200]})",
                file=sys.stderr,
            )
        return None


def cmd_last(args: argparse.Namespace) -> int:
    import json as _json

    from iai_mcp.direct_recency import read_recent_user_turns_direct

    n = max(0, int(args.n))
    session_id = args.session or None
    emit_json = getattr(args, "json", False)

    store_root = _resolve_store_root()

    turns_direct = read_recent_user_turns_direct(store_root, n=n, session_id=session_id)

    from iai_mcp.capture import read_pending_live_events

    pending = read_pending_live_events(session_id=session_id)

    if turns_direct or pending:
        from iai_mcp.capture import _idem_tag as _cap_idem_tag

        store_idem_set: set[str] = set()
        for r in turns_direct:
            for tag in (r.tags or []):
                if tag.startswith("idem:"):
                    store_idem_set.add(tag)

        seen_pending: set[str] = set()
        pending_user = []
        for ev in pending:
            if ev.get("role") != "user":
                continue
            ev_session = ev.get("session_id", "-")
            if session_id and ev_session != session_id:
                continue
            src_uuid = ev.get("source_uuid")
            ts_iso = ev.get("ts_iso", "")
            text = ev.get("text", "")
            idem = _cap_idem_tag(ev_session, "user", ts_iso, text, source_uuid=src_uuid)
            if idem in store_idem_set or idem in seen_pending:
                continue
            seen_pending.add(idem)
            pending_user.append(ev)

        turn_dicts = []
        for r in turns_direct:
            turn_dicts.append({
                "record_id": str(r.id),
                "literal_surface": r.literal_surface,
                "session_id": (r.provenance or [{}])[0].get("session_id"),
                "captured_at": r.created_at.isoformat() if r.created_at else None,
            })
        for ev in pending_user:
            src_uuid = ev.get("source_uuid")
            idem = _cap_idem_tag(
                ev.get("session_id", "-"),
                "user",
                ev.get("ts_iso", ""),
                ev.get("text", ""),
                source_uuid=src_uuid,
            )
            idem_hex = idem[5:] if idem.startswith("idem:") else idem
            rid = f"pending:{src_uuid}" if src_uuid else (f"pending:{idem_hex}" if idem_hex else "pending:unknown")
            turn_dicts.append({
                "record_id": rid,
                "literal_surface": ev.get("text", ""),
                "session_id": ev.get("session_id"),
                "captured_at": ev.get("ts_iso"),
            })

        from datetime import datetime, timezone

        def _ts_key(d: dict):
            raw = d.get("captured_at")
            if not raw:
                return datetime.min.replace(tzinfo=timezone.utc)
            try:
                dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except (TypeError, ValueError):
                return datetime.min.replace(tzinfo=timezone.utc)

        turn_dicts.sort(key=_ts_key, reverse=True)
        turn_dicts = turn_dicts[:n]

        if emit_json:
            print(_json.dumps({"turns": turn_dicts, "count": len(turn_dicts), "_source": "direct-store"}))
            return 0

        if not turn_dicts:
            print(_color("(no user turns found)"))
            return 0
        for t in turn_dicts:
            captured = (t.get("captured_at") or "?")[:19]
            sid = (t.get("session_id") or "?")[:8]
            surface = (t.get("literal_surface") or "")[:120]
            print(_color(f"[{captured}] {sid}: {surface}"))
        return 0

    from iai_mcp.cli import _send_jsonrpc_request

    resp = _send_jsonrpc_request(
        "episodes_recent",
        {"n": n, "session_id": session_id},
    )
    if isinstance(resp, dict) and "result" in resp and isinstance(resp["result"], dict):
        result = resp["result"]
        sock_turns = result.get("turns") or []
        if emit_json:
            print(_json.dumps({"turns": sock_turns, "count": len(sock_turns)}))
            return 0
        if not sock_turns:
            print(_color("(no user turns found)"))
            return 0
        for t in sock_turns:
            captured = (t.get("captured_at") or "?")[:19]
            sid = (t.get("session_id") or "?")[:8]
            surface = (t.get("literal_surface") or "")[:120]
            print(_color(f"[{captured}] {sid}: {surface}"))
        return 0

    if emit_json:
        print(_json.dumps({"turns": [], "count": 0, "_source": "direct-store"}))
        return 0
    print(_color("(no user turns found)"))
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Ingest a single file into memory in-process.

    Reads the bytes off disk, extracts text via the suffix dispatcher,
    chunks into ~256-token windows with ~20% overlap, and captures each
    chunk through the shared spine with a content-bound source_uuid so
    re-uploading the same file is idempotent (the second run reinforces
    every chunk instead of inserting). Provenance carries the source
    filename and chunk index so retrieval can scope to uploaded content.

    Open the store directly — never through the daemon socket — because
    the socket capture handler does not forward source_uuid, which would
    silently defeat the content-hash dedup contract.
    """
    import hashlib
    from pathlib import Path

    from iai_mcp.capture import capture_turn
    from iai_mcp.ingest import chunk_text, extract_text

    raw_path = getattr(args, "path", None)
    if not raw_path:
        print("upload failed: missing file path", file=sys.stderr)
        return 1
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        print(f"upload failed: not a file: {path}", file=sys.stderr)
        return 1

    size = path.stat().st_size
    if size > 100 * 1024 * 1024:
        print(
            f"upload failed: file too large ({size} bytes; max 100 MB)",
            file=sys.stderr,
        )
        return 1

    try:
        text = extract_text(path)
    except NotImplementedError as exc:
        print(f"upload failed: {exc}", file=sys.stderr)
        return 1
    except (RuntimeError, ValueError) as exc:
        print(f"upload failed: {exc}", file=sys.stderr)
        return 1

    if not text.strip():
        print(
            f"upload skipped: no extractable text in {path.name}",
            file=sys.stderr,
        )
        return 1

    # Cap extracted text size to guard against PDF decompression bombs and
    # large plaintext amplification; operator-overridable via env var.
    _MAX_EXTRACTED_CHARS = int(
        os.environ.get("IAI_UPLOAD_MAX_CHARS", 20_000_000)
    )
    if len(text) > _MAX_EXTRACTED_CHARS:
        print(
            f"upload failed: extracted text too large "
            f"({len(text):,} chars; max {_MAX_EXTRACTED_CHARS:,}). "
            f"Split the file and upload in parts.",
            file=sys.stderr,
        )
        return 1

    chunks = chunk_text(text)
    if not chunks:
        print(
            f"upload skipped: no chunks emitted from {path.name} (too short)",
            file=sys.stderr,
        )
        return 1

    # Cap chunk count to prevent unbounded embed+insert work per upload call;
    # operator-overridable via env var.
    _MAX_CHUNKS = int(os.environ.get("IAI_UPLOAD_MAX_CHUNKS", 5000))
    if len(chunks) > _MAX_CHUNKS:
        print(
            f"upload failed: {len(chunks):,} chunks exceeds the "
            f"{_MAX_CHUNKS:,}-chunk per-upload cap. "
            f"Split the file and upload in parts.",
            file=sys.stderr,
        )
        return 1

    # Shared session_id across all uploads so identical chunks across
    # different files dedup at the exact-key path (idem tag derives from
    # session_id + role + source_uuid). The filename + chunk_index travel
    # on provenance_extra for traceability; the optional --session-id
    # override lets callers scope a single upload to a custom session.
    session_id = getattr(args, "session_id", None) or "upload"

    store = _open_store_shared_or_fail("upload")
    if store is None:
        print(
            "hint: `iai teach` relays through a running daemon; "
            "upload needs direct store access.",
            file=sys.stderr,
        )
        return 1
    try:
        tally: dict[str, int] = {"inserted": 0, "reinforced": 0, "skipped": 0}
        skipped_reasons: list[str] = []
        for i, chunk in enumerate(chunks):
            chunk_sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            result = capture_turn(
                store,
                cue="",
                text=chunk,
                tier="episodic",
                session_id=session_id,
                role="user",
                source_uuid=chunk_sha,
                provenance_extra={
                    "source": "upload",
                    "filename": path.name,
                    "chunk_index": i,
                },
                # Bulk ingest needs the cosine near-dup gate, same as study:
                # the idem key only dedups byte-identical chunks, so an
                # overlapping re-upload would otherwise insert cos>=0.95 twins.
                near_dup_gate=True,
            )
            status = result.get("status", "skipped")
            tally[status] = tally.get(status, 0) + 1
            if status == "skipped":
                skipped_reasons.append(result.get("reason", "?"))
    finally:
        try:
            store.close()
        except Exception:  # noqa: BLE001 -- close-after-upload must not fail user
            pass

    if getattr(args, "json", False):
        import json as _json

        payload = {
            "file": path.name,
            "session_id": session_id,
            "chunks_total": len(chunks),
            "inserted": tally["inserted"],
            "reinforced": tally["reinforced"],
            "skipped": tally["skipped"],
            "skipped_reasons": skipped_reasons[:10],
        }
        print(_json.dumps(payload))
    else:
        print(_color(
            f"uploaded {path.name}  chunks={len(chunks)}  "
            f"inserted={tally['inserted']}  "
            f"reinforced={tally['reinforced']}  "
            f"skipped={tally['skipped']}"
        ))
    return 0


def _teach_relay_file(path, session_id: str, reconcile: bool, source_name=None):
    """Relay one file to the live daemon's teach verb (the daemon is the
    single writer on the production engine — while it is alive, IT studies)."""
    import base64 as _b64

    content_b64 = _b64.b64encode(path.read_bytes()).decode("ascii")
    return _relay_teach_verb({
        "filename": str(source_name or path.name),
        "content_b64": content_b64,
        "session_id": session_id,
        "reconcile": bool(reconcile),
    })


def _teach_directory_relay(root, session_id: str, reconcile: bool) -> "dict | None":
    """Relay every studyable file under root through the daemon, aggregating
    the same totals study_directory reports. Returns None if the relay died
    mid-run (caller falls back to the clean error path)."""
    from iai_mcp.study import iter_study_files

    totals: dict = {
        "root": str(root), "files": 0, "files_skipped": 0,
        "chunks_total": 0, "inserted": 0, "reinforced": 0, "skipped": 0,
        "edges_formed": 0, "contradictions_resolved": 0,
        "schemas_induced": 0, "superseded_chunks": 0,
        "recall_tested": 0, "recall_verified": 0, "replay_queued": 0,
    }
    for path in iter_study_files(root):
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path.name
        try:
            report = _teach_relay_file(
                path, session_id, reconcile, source_name=str(rel),
            )
        except OSError:
            totals["files_skipped"] += 1
            continue
        if _is_relay_failure(report):
            if report.get("status") == "daemon_down":
                return None
            totals["files_skipped"] += 1
            continue
        totals["files"] += 1
        for key in (
            "chunks_total", "inserted", "reinforced", "skipped",
            "edges_formed", "contradictions_resolved", "schemas_induced",
            "superseded_chunks", "recall_tested", "recall_verified",
            "replay_queued",
        ):
            totals[key] += int(report.get(key) or 0)
    return totals


def cmd_teach(args: argparse.Namespace) -> int:
    """Study a file: ingest through the shared spine, then weave it into
    existing memory (Hebbian links to related prior records, contradiction
    reconciliation, schema induction). While the daemon is alive it is the
    single writer on the production engine, so teach RELAYS through its
    socket; daemon-down opens the store directly — the awake path works
    either way."""
    from pathlib import Path

    from iai_mcp.study import study_document

    raw_path = getattr(args, "path", None)
    if not raw_path:
        print("teach failed: missing file path", file=sys.stderr)
        return 1
    path = Path(raw_path).expanduser().resolve()
    session_id = getattr(args, "session_id", None) or "study"
    reconcile = bool(getattr(args, "reconcile", False))

    if path.is_dir():
        from iai_mcp.study import study_directory

        totals = None
        if _daemon_alive():
            totals = _teach_directory_relay(path, session_id, reconcile)
        if totals is None:
            critic = None
            if reconcile:
                from iai_mcp.reconsolidation_critic import (
                    evaluate_batch_reconsolidation,
                )
                critic = evaluate_batch_reconsolidation
            store = _open_store_shared_or_fail("teach")
            if store is None:
                return 1
            try:
                totals = study_directory(
                    store, path, session_id=session_id, critic=critic,
                )
            finally:
                try:
                    store.close()
                except Exception:  # noqa: BLE001 -- close-after-study must not fail user
                    pass
        if getattr(args, "json", False):
            import json as _json
            print(_json.dumps(totals))
        else:
            print(_color(
                f"studied {totals['files']} files under {path.name}/  "
                f"new={totals['inserted']}  reinforced={totals['reinforced']}  "
                f"edges={totals['edges_formed']}  "
                f"superseded={totals['superseded_chunks']}  "
                f"recall-check={totals['recall_verified']}/{totals['recall_tested']}"
            ))
        return 0

    if not path.is_file():
        print(f"teach failed: not a file: {path}", file=sys.stderr)
        return 1
    size = path.stat().st_size
    if size > 100 * 1024 * 1024:
        print(f"teach failed: file too large ({size} bytes; max 100 MB)", file=sys.stderr)
        return 1

    report = None
    if _daemon_alive():
        # The daemon-side teach verb caps decoded content at 25 MB; a bigger
        # file needs the direct path, which needs the writer the daemon holds.
        if size > 25 * 1024 * 1024:
            print(
                f"teach failed: {path.name} is {size // (1024 * 1024)} MB — the "
                f"live-daemon relay accepts up to 25 MB. Stop the daemon "
                f"(iai-mcp daemon stop) to teach large files directly.",
                file=sys.stderr,
            )
            return 1
        try:
            relayed = _teach_relay_file(path, session_id, reconcile)
        except OSError as exc:
            print(f"teach failed: cannot read {path}: {exc}", file=sys.stderr)
            return 1
        if not _is_relay_failure(relayed):
            report = relayed
        elif relayed.get("status") == "error":
            print(
                f"teach failed: {relayed.get('reason', 'daemon error')}",
                file=sys.stderr,
            )
            return 1
        # daemon_down mid-call: fall through to the direct path below.

    if report is None:
        try:
            from iai_mcp.ingest import extract_text
            text = extract_text(path)
        except (NotImplementedError, RuntimeError, ValueError) as exc:
            print(f"teach failed: {exc}", file=sys.stderr)
            return 1
        if not text.strip():
            print(f"teach skipped: no extractable text in {path.name}", file=sys.stderr)
            return 1
        _MAX_EXTRACTED_CHARS = int(os.environ.get("IAI_UPLOAD_MAX_CHARS", 20_000_000))
        if len(text) > _MAX_EXTRACTED_CHARS:
            print(
                f"teach failed: extracted text too large "
                f"({len(text):,} chars; max {_MAX_EXTRACTED_CHARS:,}).",
                file=sys.stderr,
            )
            return 1

        critic = None
        if reconcile:
            from iai_mcp.reconsolidation_critic import evaluate_batch_reconsolidation
            critic = evaluate_batch_reconsolidation

        store = _open_store_shared_or_fail("teach")
        if store is None:
            return 1
        try:
            report = study_document(
                store,
                text=text,
                source_name=path.name,
                session_id=session_id,
                critic=critic,
            )
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- close-after-study must not fail user
                pass

    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps(report))
    else:
        print(_color(
            f"studied {report['source']}  chunks={report['chunks_total']}  "
            f"new={report['inserted']}  reinforced={report['reinforced']}  "
            f"edges={report['edges_formed']}  "
            f"contradictions={report['contradictions_resolved']}"
            f"/{report['contradiction_candidates']}  "
            f"schemas={report['schemas_induced']}  "
            f"recall-check={report['recall_verified']}/{report['recall_tested']}"
        ))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Hybrid lexical+semantic scoped search. While the daemon is alive it
    answers over its socket (same memory_search dispatch, live store);
    daemon-down opens the store in-process — the awake hippocampus answers
    either way."""
    from iai_mcp import core as _core

    query = (getattr(args, "query", None) or "").strip()
    if not query:
        print("search failed: empty query", file=sys.stderr)
        return 1
    k = getattr(args, "limit", None) or 8

    resp = None
    if _daemon_alive():
        relayed = _relay_rpc(
            "memory_search", {"query": query, "k": k}, timeout=30.0,
        )
        if not _is_relay_failure(relayed) and isinstance(relayed, dict):
            resp = relayed
        # daemon_down/error mid-call: fall through to the direct path.

    if resp is None:
        store = _open_store_shared_or_fail("search")
        if store is None:
            return 1
        try:
            resp = _core.dispatch(
                store, "memory_search", {"query": query, "k": k},
            )
        finally:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- close-after-search must not fail user
                pass
    if getattr(args, "json", False):
        import json as _json
        print(_json.dumps(resp, ensure_ascii=False))
    else:
        for h in resp.get("hits", []):
            lane = h.get("lane", "?")
            print(_color(f"[{lane:8}] {h.get('surface', '')[:120]}"))
        if not resp.get("hits"):
            print(_color("(no hits)"))
    return 0


def cmd_watch(args: argparse.Namespace) -> int:
    """Keep studying a directory: restudy changed files, fade deleted ones.
    The live counterpart of `iai teach <dir>` — memory follows the tree.
    Each tick picks its backend: a live daemon gets the relay (it is the
    single writer); a dead one gets a direct store, yielded back the moment
    the daemon returns."""
    import time as _time
    from pathlib import Path

    from iai_mcp.study import watch_scan_once

    raw_path = getattr(args, "path", None)
    path = Path(raw_path or ".").expanduser().resolve()
    if not path.is_dir():
        print(f"watch failed: not a directory: {path}", file=sys.stderr)
        return 1
    interval = max(5, int(getattr(args, "interval", None) or 30))
    session_id = getattr(args, "session_id", None) or "study"

    def _relay_study_fn(**kw):
        if "path" in kw:
            res = _teach_relay_file(
                kw["path"], session_id, False,
                source_name=kw.get("source_name"),
            )
        else:
            res = _relay_teach_verb({
                "filename": str(kw.get("source_name") or "deleted"),
                "content_b64": "",
                "session_id": session_id,
                "fade": True,
            })
        if _is_relay_failure(res):
            raise RuntimeError(f"daemon relay failed: {res.get('reason', '?')}")
        return res

    store = None
    seen: dict = {}
    print(_color(f"watching {path}  (every {interval}s; Ctrl+C to stop)"))
    try:
        while True:
            if _daemon_alive():
                if store is not None:
                    # Yield the single-writer engine back to the daemon.
                    try:
                        store.close()
                    except Exception:  # noqa: BLE001 -- yield must not stop the watch
                        pass
                    store = None
                study_fn = _relay_study_fn
            else:
                if store is None:
                    store = _open_store_shared_or_fail("watch")
                    if store is None:
                        _time.sleep(interval)
                        continue
                study_fn = None
            seen, changes = watch_scan_once(
                store, path, seen, session_id=session_id, study_fn=study_fn,
            )
            if changes["studied"] or changes["faded"]:
                print(_color(
                    f"  studied={changes['studied']}  faded={changes['faded']}  "
                    f"superseded={changes['superseded_chunks']}"
                ))
            _time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:  # noqa: BLE001 -- close-after-watch must not fail user
                pass
    return 0


def cmd_brain(args: argparse.Namespace) -> int:
    """Serve the local brain-view dashboard (loopback only, daemon-free)."""
    from iai_mcp.brainview import BRAINVIEW_DEFAULT_PORT, serve

    port = getattr(args, "port", None) or BRAINVIEW_DEFAULT_PORT
    if not (0 <= int(port) <= 65535):
        print(f"brain view failed: port {port} out of range (0-65535)", file=sys.stderr)
        return 1
    try:
        serve(
            _resolve_store_root(),
            port=int(port),
            open_browser=not getattr(args, "no_open", False),
            app_window=bool(getattr(args, "app", False)),
        )
    except (OSError, OverflowError) as exc:
        print(f"brain view failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="iai",
        description="Terminal memory for your agent — recall, capture, ask, status.",
        add_help=True,
    )
    parser.add_argument("--version", action="version", version=f"iai {__version__}")

    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    p_recall = sub.add_parser(
        "recall",
        help="Recall memories by natural-language cue",
        description="Recall memories. Uses the daemon when alive; falls back "
        "to the offline bank scan when daemon is down.",
    )
    p_recall.add_argument("cue", help="Natural-language query")
    p_recall.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum hits to print (default 5)",
    )
    p_recall.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Print result as a JSON payload for programmatic use (MCP wrapper)",
    )
    p_recall.set_defaults(func=cmd_recall)

    p_temporal = sub.add_parser(
        "temporal-recall",
        help="Time-travel recall (as-of records + changed-since events)",
        description=(
            "Recall what was committed to memory by a point in time, "
            "optionally scoped to the records-ledger changed window. "
            "Honest scope: as_of returns what was committed by time t, "
            "NOT what was true on date t. Daemon-down operation supported."
        ),
    )
    p_temporal.add_argument(
        "cue",
        nargs="?",
        default="",
        help="Optional natural-language query; omit for range-only recall",
    )
    p_temporal.add_argument(
        "--as-of",
        default=None,
        help="ISO-8601 timestamp; records.created_at <= as_of",
    )
    p_temporal.add_argument(
        "--changed-since",
        default=None,
        help="ISO-8601 timestamp; events.ts > changed_since (strict)",
    )
    p_temporal.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum items to return (default 10)",
    )
    p_temporal.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit result as JSON on stdout (for programmatic use)",
    )
    p_temporal.set_defaults(func=cmd_temporal_recall)

    p_upload = sub.add_parser(
        "upload",
        help="Ingest a file into memory (txt/md/csv/pdf)",
        description=(
            "Chunks the file (~256 tokens, ~20% overlap) and captures "
            "each chunk as an episodic memory tagged with the source "
            "filename and chunk index. Re-uploading the same file is "
            "idempotent (content-hash exact-key dedup). Daemon-down "
            "operation supported (the store opens in-process)."
        ),
    )
    p_upload.add_argument("path", help="Path to file to ingest")
    p_upload.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id to scope this upload (default: shared 'upload' "
             "namespace so identical chunks across files dedup).",
    )
    p_upload.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit result as JSON on stdout (for programmatic use)",
    )
    p_upload.set_defaults(func=cmd_upload)

    p_teach = sub.add_parser(
        "teach",
        help="Study a file and weave it into existing memory",
        description=(
            "Ingests the file like `upload` (same chunking + dedup gate), "
            "then integrates it: Hebbian links form to related existing "
            "records, contradicted prior beliefs gain a contradicts edge to "
            "the correcting chunk (with --reconcile), and document-level "
            "schemas are induced. Re-teaching the same file is idempotent. "
            "Daemon-down operation supported."
        ),
    )
    p_teach.add_argument("path", help="Path to file to study (txt/md/csv/pdf)")
    p_teach.add_argument(
        "--session-id",
        default=None,
        help="Optional session_id scope (default: shared 'study' namespace)",
    )
    p_teach.add_argument(
        "--reconcile",
        action="store_true",
        default=False,
        help="Run the LLM reconsolidation critic over related existing "
             "records and mark contradicted ones (subscription-billed "
             "claude -p; off by default)",
    )
    p_teach.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit the study report as JSON on stdout",
    )
    p_teach.set_defaults(func=cmd_teach)

    p_search = sub.add_parser(
        "search",
        help="Hybrid lexical+semantic search (daemon-free)",
    )
    p_search.add_argument("query", help="Identifiers, phrases, or a question")
    p_search.add_argument("--limit", type=int, default=8)
    p_search.add_argument("--json", action="store_true", default=False)
    p_search.set_defaults(func=cmd_search)

    p_watch = sub.add_parser(
        "watch",
        help="Keep studying a directory (restudy changes, fade deletions)",
    )
    p_watch.add_argument("path", nargs="?", default=".", help="Directory to watch")
    p_watch.add_argument("--interval", type=int, default=30, help="Scan period, sec")
    p_watch.add_argument("--session-id", default=None)
    p_watch.set_defaults(func=cmd_watch)

    p_brain = sub.add_parser(
        "brain",
        help="Open the live brain-view dashboard (local, daemon-free)",
        description=(
            "Serves a loopback-only page visualizing the memory graph live: "
            "records as glowing nodes (tier-colored, community-ringed, "
            "pinned-marked), Hebbian and contradicts synapses, lifecycle "
            "state, and the event feed. You can capture a new memory "
            "(through the shared dedup-gated spine) or hint a record into "
            "the forgetting queue — there is deliberately NO delete."
        ),
    )
    p_brain.add_argument(
        "--port", type=int, default=None,
        help="Port to bind on 127.0.0.1 (default 4477)",
    )
    p_brain.add_argument(
        "--no-open", action="store_true", default=False,
        help="Do not auto-open the browser",
    )
    p_brain.add_argument(
        "--app", action="store_true", default=False,
        help="Open as a chromeless desktop window (Chrome app mode)",
    )
    p_brain.set_defaults(func=cmd_brain)

    p_capture = sub.add_parser(
        "capture",
        help="Capture one episodic memory",
        description="Write one episodic record to the store via the daemon.",
    )
    p_capture.add_argument("text", help="Memory text to store")
    p_capture.add_argument(
        "--session-id",
        default=None,
        help="Session identifier (default '-')",
    )
    p_capture.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit result as JSON on stdout (for programmatic use)",
    )
    p_capture.set_defaults(func=cmd_capture)

    p_ask = sub.add_parser(
        "ask",
        help="Recall + LLM synthesis grounded in memories",
        description="Recall the top-K memories matching the question, then "
        "synthesize an answer via `claude -p` (subscription-billed). "
        "Prints the answer + a Sources: footer with the cited record ids.",
    )
    p_ask.add_argument("question", help="Natural-language question")
    p_ask.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max memories to ground the answer (default 5)",
    )
    p_ask.set_defaults(func=cmd_ask)

    p_status = sub.add_parser(
        "status",
        help="Short health summary (daemon + records + subscription)",
        description="User-tier health summary. For the 17-row operator "
        "checklist run `iai-mcp doctor` instead.",
    )
    p_status.set_defaults(func=cmd_status)

    p_last = sub.add_parser(
        "last",
        help="Show the most-recent user-turn records, time-descending",
        description="Return the N most-recent role:user turns from the store. "
        "Optionally filter to a single session with --session.",
    )
    p_last.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of turns to return (default 5)",
    )
    p_last.add_argument(
        "--session",
        default=None,
        metavar="SESSION_ID",
        help="Filter to a specific session UUID",
    )
    p_last.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Emit turns as a JSON object on stdout (for programmatic use)",
    )
    p_last.set_defaults(func=cmd_last)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.cmd is None:
        _print_logo()
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
