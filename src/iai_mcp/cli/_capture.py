"""Session, capture, and hook commands for the iai-mcp operator CLI."""

from __future__ import annotations

import argparse
import importlib.resources as _res
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_LOCK_POLL_TRIES = 50
_LOCK_POLL_INTERVAL = 0.2


def _write_state_file(path: Path, body: str) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        f = os.fdopen(fd, "w", encoding="utf-8")
    except Exception:
        os.close(fd)
        raise
    with f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())

_HOOK_TRUNCATION_TRAILER = "[... payload truncated to fit Claude Code 10000-char limit ...]"


def _truncate_for_claude_code_hook(text: str, cap: int = 10000) -> str:
    if len(text) <= cap:
        return text
    head_len = cap - len(_HOOK_TRUNCATION_TRAILER)
    if head_len <= 0:
        return _HOOK_TRUNCATION_TRAILER[:cap]
    return text[:head_len] + _HOOK_TRUNCATION_TRAILER


def _is_custom_store() -> bool:
    env_store = os.environ.get("IAI_MCP_STORE")
    if not env_store:
        return False
    from iai_mcp.store import DEFAULT_STORAGE_PATH as _DEFAULT

    try:
        custom = Path(env_store).expanduser().resolve()
        default = Path(_DEFAULT).expanduser().resolve()
        return custom != default
    except Exception:
        return False


def cmd_session_start(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli

    try:
        from iai_mcp.session import format_payload_as_markdown
        session_id = getattr(args, "session_id", "-") or "-"
        resp = _cli._send_jsonrpc_request(
            "session_start_payload", {"session_id": session_id}
        )
        if not isinstance(resp, dict) or "result" not in resp:
            return 0
        result = resp.get("result")
        if not isinstance(result, dict):
            return 0
        rendered = format_payload_as_markdown(result)
        if not rendered:
            return 0
        _cli.sys.stdout.write(_truncate_for_claude_code_hook(rendered, cap=10000))
        return 0
    except Exception as exc:
        logger.error("session-start failed: %s", exc)
        return 0


def get_other_sessions_live_size(session_id: str) -> int:
    try:
        deferred_dir = Path.home() / ".iai-mcp" / ".deferred-captures"
        if not deferred_dir.exists():
            return 0
        own_name = f"{session_id}.live.jsonl"
        total = 0
        for entry in deferred_dir.iterdir():
            if not entry.is_file():
                continue
            if not entry.name.endswith(".live.jsonl"):
                continue
            if entry.name == own_name:
                continue
            try:
                total += entry.stat().st_size
            except OSError:
                pass
        return total
    except Exception:
        return 0


def read_live_fingerprint(session_id: str) -> int | None:
    p = Path.home() / ".iai-mcp" / ".capture-state" / f"{session_id}.live-fingerprint"
    try:
        if not p.exists():
            return None
        raw = p.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        return int(raw)
    except (OSError, ValueError):
        return None


def write_live_fingerprint(session_id: str, total_size: int) -> None:
    d = Path.home() / ".iai-mcp" / ".capture-state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session_id}.live-fingerprint.tmp{os.getpid()}"
    tmp.write_text(str(total_size), encoding="utf-8")
    os.replace(tmp, d / f"{session_id}.live-fingerprint")


def get_max_created_at() -> str | None:
    from iai_mcp import _sqlite_stdlib
    from iai_mcp.hippo._raw_open import open_store_conn
    from iai_mcp.store_watermark import read as _read_watermark

    store_root = Path.home() / ".iai-mcp" / "hippo"
    stamped = _read_watermark(store_root)
    if stamped:
        return stamped

    db_path = store_root / "brain.sqlite3"
    if not db_path.exists():
        return None
    try:
        _eng = open_store_conn(db_path, read_only=True)
        if _eng is not None:
            conn = _eng
        else:
            conn = _sqlite_stdlib.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT MAX(created_at) FROM records WHERE tombstoned_at IS NULL"
            ).fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()
    except Exception:
        return None


def _utc_iso(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.isoformat()
    except (TypeError, ValueError):
        return ts


def read_watermark(session_id: str) -> str | None:
    p = Path.home() / ".iai-mcp" / ".capture-state" / f"{session_id}.watermark"
    try:
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def write_watermark(session_id: str, ts: str) -> None:
    d = Path.home() / ".iai-mcp" / ".capture-state"
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{session_id}.watermark.tmp{os.getpid()}"
    tmp.write_text(_utc_iso(ts), encoding="utf-8")
    os.replace(tmp, d / f"{session_id}.watermark")


def cmd_session_refresh_if_stale(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli

    try:
        session_id: str = (getattr(args, "session_id", None) or "-")

        current = get_max_created_at()
        if current is None:
            return 0

        wm = read_watermark(session_id)
        live_size = get_other_sessions_live_size(session_id)

        if wm is None:
            write_watermark(session_id, current)
            write_live_fingerprint(session_id, live_size)
            return 0

        store_advanced = _utc_iso(current) > _utc_iso(wm)

        fp = read_live_fingerprint(session_id)
        if fp is None:
            write_live_fingerprint(session_id, live_size)
            fp = live_size
        live_grew = live_size > fp

        if not store_advanced and not live_grew:
            return 0

        resp = _cli._send_jsonrpc_request(
            "session_refresh_if_stale",
            {"watermark": wm, "session_id": session_id},
            connect_timeout=5.0,
            read_timeout=30.0,
        )
        if resp is None:
            return 0

        result = resp.get("result") or {}
        rendered: str = result.get("rendered") or ""
        new_max: str = result.get("new_max_ts") or current

        if rendered:
            payload = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": rendered,
                }
            }
            _cli.sys.stdout.write(json.dumps(payload, ensure_ascii=False))
            write_watermark(session_id, new_max)
            write_live_fingerprint(session_id, live_size)

        return 0
    except Exception:
        return 0


def cmd_capture_transcript(args: argparse.Namespace) -> int:
    import json
    import sys as _sys

    no_spawn = bool(getattr(args, "no_spawn", False))

    if no_spawn:
        from iai_mcp.capture import write_deferred_captures

        try:
            out = write_deferred_captures(
                session_id=args.session_id,
                transcript_path=args.transcript_path,
                cwd=os.getcwd(),
                max_turns=args.max_turns,
            )
            print(json.dumps({"status": "deferred", "path": str(out)}, ensure_ascii=False))
            return 0
        except Exception as e:
            logger.error("capture-transcript --no-spawn failed: %s", e)
            print(
                f"capture-transcript --no-spawn: failed {type(e).__name__}: {e}",
                file=_sys.stderr,
            )
            return 0

    # Default path
    from iai_mcp.capture import capture_transcript
    from iai_mcp.store import MemoryStore

    try:
        with MemoryStore() as store:
            counts = capture_transcript(
                store,
                args.transcript_path,
                session_id=args.session_id,
                max_turns=args.max_turns,
            )
            _stamp_capture_batch_liveness(counts)
            print(json.dumps(counts, ensure_ascii=False))
        return 0
    except Exception as e:
        logger.error("capture-transcript inline failed: %s", e)
        print(f"capture-transcript: failed {type(e).__name__}: {e}", file=_sys.stderr)
        return 0


def _stamp_capture_batch_liveness(counts: dict) -> None:
    # A "reason" key means capture_transcript never opened a transcript at
    # all (e.g. the path was missing) -- there were no genuine candidates,
    # not a batch that ran and produced zero. Skip rather than fabricate a
    # liveness row from an on-open failure.
    if counts.get("reason"):
        return
    try:
        from iai_mcp.lifecycle_event_log import _LIVENESS_SPEC_VERSION, LifecycleEventLog

        log_root = os.environ.get("IAI_MCP_STORE")
        log_dir = (Path(log_root) if log_root else Path.home() / ".iai-mcp") / "logs"
        turns = sum(counts.get(k, 0) for k in ("inserted", "reinforced", "skipped"))
        records = counts.get("inserted", 0) + counts.get("reinforced", 0)
        LifecycleEventLog(log_dir=log_dir).append({
            "event": "promise_liveness",
            "promise": "capture_batch",
            "liveness_candidates": turns,
            "liveness_processed": records,
            "liveness_spec_version": _LIVENESS_SPEC_VERSION,
        })
    except (OSError, ValueError) as exc:
        logger.debug("capture_batch_liveness_append_failed: %s", exc)


def _trajectory_last_recorded(store, session_id: str, metric: str) -> float | None:
    # session_id-scoped at the SQL level -- an unrelated writer flooding
    # trajectory_metric under other sessions must never evict this
    # session's own dedup key from a fixed-size global window (the same
    # horizon-eviction failure mode this phase's detector fix addresses).
    from iai_mcp.events import query_events

    for e in query_events(
        store, kind="trajectory_metric", session_id=session_id, limit=200,
    ):
        data = e.get("data") or {}
        if data.get("metric") != metric:
            continue
        try:
            return float(data.get("value"))
        except (TypeError, ValueError):
            return None
    return None


def _drive_trajectory_metrics(session_id: str, state_dir: Path) -> None:
    from iai_mcp.store import MemoryStore
    from iai_mcp.trajectory import (
        compute_m1_clarifying_questions_per_session,
        compute_m3_token_budget,
        compute_m5_curiosity_frequency,
        m2_precision_at_5_live,
        m4_profile_movement_live,
        m6_context_repeat_rate_live,
        record_session_metrics,
    )

    with MemoryStore() as store:
        per_session = {
            "m1": compute_m1_clarifying_questions_per_session(store, session_id),
            "m3": compute_m3_token_budget(store, session_id),
            "m5": compute_m5_curiosity_frequency(store, session_id),
        }
        to_record = {
            m: v for m, v in per_session.items()
            if v != _trajectory_last_recorded(store, session_id, m)
        }
        if to_record:
            record_session_metrics(store, session_id=session_id, metrics=to_record)

        # Globals are store-wide, not session-scoped -- computed at most once per
        # session so the per-turn cost stays a single cheap per-session write,
        # not a repeated 10k-row scan (m6) plus buffer flush (m2) every turn.
        globals_marker = state_dir / f"{session_id}.trajectory-globals-done"
        if globals_marker.exists():
            return

        globals_vals = {
            "m2": m2_precision_at_5_live(store),
            "m4": m4_profile_movement_live(store),
            "m6": m6_context_repeat_rate_live(store),
        }
        globals_to_record = {
            m: v for m, v in globals_vals.items()
            if v != _trajectory_last_recorded(store, "-", m)
        }
        if globals_to_record:
            record_session_metrics(store, session_id="-", metrics=globals_to_record)

        try:
            globals_marker.touch(exist_ok=True)
        except OSError:
            pass


def cmd_capture_turn_deferred(args: argparse.Namespace) -> int:
    import sys as _sys

    _lock_fd = -1
    try:
        from iai_mcp.capture import (
            _parse_transcript_obj,
            _ToolTrailerState,
            write_deferred_event,
        )
        import json as _json

        transcript = Path(args.transcript_path).expanduser()
        if not transcript.exists():
            return 0

        # Session id is host-controlled text that lands in file paths.
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", str(args.session_id)):
            logger.warning("capture-turn-deferred: invalid session_id, skipping")
            return 0

        state_dir = Path.home() / ".iai-mcp" / ".capture-state"
        state_dir.mkdir(parents=True, exist_ok=True)
        offset_path = state_dir / f"{args.session_id}.offset"
        # Offset and pending tool names live in ONE snapshot file published
        # by a single atomic rename: any crash or carrier race can only ever
        # surface an older coherent pair. Its replay is deterministic, and
        # lines carrying native identity (uuid or timestamp) land on the
        # same idem keys; the cosine dedup gate absorbs the rest. Two
        # separate files can pair a fresh offset with stale pending and
        # fabricate a trailer.
        turnstate_path = state_dir / f"{args.session_id}.turnstate.json"

        from iai_mcp import _flock as _fcntl

        _lock_fd = os.open(
            str(state_dir / f"{args.session_id}.capture.lock"),
            os.O_WRONLY | os.O_CREAT, 0o600,
        )
        # Bounded wait, not skip-on-contention: the Stop hook rotates the
        # live spool unconditionally after this call, so skipping would
        # orphan the transcript tail. Bounded, because the holder may be
        # walking a large transcript and non-Claude hosts give this process
        # no external wall-clock cap. Only contention retries — any other
        # lock failure must surface, not burn the budget silently.
        acquired = False
        for _ in range(_LOCK_POLL_TRIES):
            try:
                _fcntl.flock(_lock_fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(_LOCK_POLL_INTERVAL)
        if not acquired:
            logger.warning(
                "capture-turn-deferred: session lock still contended after "
                "bounded wait, skipping this pass"
            )
            os.close(_lock_fd)
            _lock_fd = -1
            return 0

        snap_offset = -1
        prev_pending: list = []
        snap_fp = ""
        if turnstate_path.exists():
            try:
                snap = _json.loads(turnstate_path.read_text(encoding="utf-8"))
                if isinstance(snap, dict):
                    snap_offset = int(snap.get("offset", 0))
                    loaded = snap.get("pending")
                    if isinstance(loaded, list):
                        prev_pending = loaded[:256]
                    loaded_fp = snap.get("fp")
                    if isinstance(loaded_fp, str):
                        snap_fp = loaded_fp
            except (ValueError, TypeError, OSError):
                snap_offset = -1
                prev_pending = []
                snap_fp = ""

        legacy_offset = 0
        if offset_path.exists():
            try:
                legacy_offset = int(
                    offset_path.read_text(encoding="utf-8").strip() or "0"
                )
            except ValueError:
                legacy_offset = 0

        if snap_offset >= legacy_offset:
            prev_offset = max(snap_offset, 0)
        else:
            # A writer that knows only the legacy offset advanced past the
            # snapshot: adopt its position and drop pending — trailer loss,
            # never fabrication.
            prev_offset = legacy_offset
            prev_pending = []

        with transcript.open(encoding="utf-8") as fh:
            all_lines = fh.readlines()
        total = len(all_lines)

        import hashlib as _hashlib

        # Fingerprint the RAW first-line bytes: the fp is shared between
        # carriers through the snapshot, and hashing a decoded string would
        # couple it to each carrier's decode settings.
        with transcript.open("rb") as fh_raw:
            first_raw = fh_raw.readline()
        stream_fp = (
            _hashlib.sha256(first_raw).hexdigest()[:16] if first_raw else ""
        )
        # A changed first line means a different stream at the same path:
        # the stored offset and pending belong to a dead transcript, even
        # when the new one has already regrown past the old length — the
        # length fence below stays as the backstop for fingerprint-less
        # legacy snapshots and same-stream truncation.
        if (snap_fp and stream_fp and snap_fp != stream_fp) or prev_offset > total:
            prev_offset = 0
            prev_pending = []

        new_lines = all_lines[prev_offset:]
        consumed = 0
        emitted = 0
        max_emit = int(getattr(args, "max_turns_per_call", 200))
        cwd = os.getcwd()
        trailers = _ToolTrailerState(prev_pending)
        for line in new_lines:
            if emitted >= max_emit:
                break
            consumed += 1
            try:
                _obj = _json.loads(line)
            except (ValueError, TypeError):
                _obj = {}
            if not isinstance(_obj, dict):
                _obj = {}
            parsed = trailers.feed(
                _obj, _parse_transcript_obj(_obj) if _obj else None
            )
            if parsed is None:
                continue
            role, text, src_uuid, src_ts = parsed
            write_deferred_event(
                args.session_id, role, text,
                cwd=cwd,
                ts=src_ts,
                source_uuid=src_uuid,
            )
            emitted += 1

        new_offset = prev_offset + consumed
        # Snapshot first, legacy offset second: a crash between them leaves
        # the legacy offset older, and the snapshot (>=) stays authoritative.
        snap_tmp = turnstate_path.parent / (
            f"{turnstate_path.name}.tmp{os.getpid()}"
        )
        snap_body = _json.dumps(
            {
                "offset": new_offset,
                "pending": trailers.pending[:256],
                "fp": stream_fp,
            }
        )
        _write_state_file(snap_tmp, snap_body)
        os.replace(snap_tmp, turnstate_path)

        # A pre-snapshot reader pairs the mirrored offset with the split
        # pending file and can fabricate a trailer from it — remove it
        # BEFORE the offset publish, so a reader that sees the fresh offset
        # can never find it.
        try:
            (state_dir / f"{args.session_id}.pending-tools").unlink()
        except OSError:
            pass

        tmp_path = offset_path.parent / (f"{offset_path.name}.tmp{os.getpid()}")
        _write_state_file(tmp_path, str(new_offset))
        os.replace(tmp_path, offset_path)
        # Rename durability needs the directory entry flushed too.
        dfd = os.open(str(state_dir), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)

        try:
            _drive_trajectory_metrics(args.session_id, state_dir)
        except Exception as exc:  # noqa: BLE001 -- best-effort, never blocks the turn capture
            logger.debug("capture-turn-deferred trajectory driver failed: %s", exc)

        return 0
    except Exception as e:
        logger.error("capture-turn-deferred failed: %s", e)
        print(
            f"capture-turn-deferred: failed {type(e).__name__}: {e}",
            file=_sys.stderr,
        )
        return 0
    finally:
        if _lock_fd >= 0:
            try:
                os.close(_lock_fd)
            except OSError:
                pass


def _capture_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-session-capture.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-session-capture.sh"
    settings = Path.home() / ".claude" / "settings.json"
    return src, dst, settings


def _turn_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-turn-capture.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-turn-capture.sh"
    return src, dst


def _resolve_wrapper_path() -> Path:
    import iai_mcp as _pkg

    env_val = os.environ.get("IAI_MCP_WRAPPER_PATH")
    if env_val:
        p = Path(env_val)
        if p.exists():
            return p
        raise FileNotFoundError(
            f"IAI_MCP_WRAPPER_PATH={env_val!r} is set but the file does not exist."
        )

    try:
        pkg_p = Path(str(_res.files("iai_mcp") / "_wrapper" / "index.js"))
        if pkg_p.exists():
            return pkg_p
    except (TypeError, FileNotFoundError):
        pass

    src_file = Path(_pkg.__file__).resolve()
    repo_root = src_file.parent.parent.parent
    editable_path = repo_root / "mcp-wrapper" / "dist" / "index.js"
    if editable_path.exists():
        return editable_path

    raise FileNotFoundError(
        "MCP wrapper (index.js) not found. Checked locations:\n"
        f"  1. IAI_MCP_WRAPPER_PATH env var (not set)\n"
        f"  2. Package data: {str(_res.files('iai_mcp') / '_wrapper' / 'index.js')}\n"
        f"  3. Editable source: {editable_path}\n"
        "To build: cd mcp-wrapper && npm run build\n"
        "Or run: bash scripts/install.sh\n"
        "For packaged installs: reinstall the wheel (it should include the wrapper)."
    )


def _build_iai_mcp_server_entry() -> dict:
    from iai_mcp import cli as _cli

    wrapper = _resolve_wrapper_path()
    return {
        "command": "node",
        "args": [str(wrapper)],
        "env": {
            "IAI_MCP_PYTHON": _cli.sys.executable,
            "IAI_MCP_STORE": str(Path.home() / ".iai-mcp"),
            "TRANSFORMERS_VERBOSITY": "error",
            "TOKENIZERS_PARALLELISM": "false",
        },
    }


def _patch_claude_desktop_config(action: str) -> str:
    from iai_mcp import cli as _cli
    import json as _json

    cfg_path = _cli._claude_desktop_config_path()
    if cfg_path is None:
        return "Claude Desktop: not installed (no config dir) — skipped"

    if not cfg_path.exists():
        if action == "uninstall":
            return f"Claude Desktop: {cfg_path} absent — skipped"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"mcpServers": {"iai-mcp": _build_iai_mcp_server_entry()}}
        cfg_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        return f"Claude Desktop: created {cfg_path} with iai-mcp registered"

    try:
        data = _json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"Claude Desktop: {cfg_path} unreadable ({type(e).__name__}) — skipped"

    servers = data.setdefault("mcpServers", {})

    if action == "uninstall":
        if "iai-mcp" in servers:
            servers.pop("iai-mcp", None)
            cfg_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
            return f"Claude Desktop: removed iai-mcp from {cfg_path}"
        return f"Claude Desktop: iai-mcp not in config — no change"

    new_entry = _build_iai_mcp_server_entry()
    if servers.get("iai-mcp") == new_entry:
        return f"Claude Desktop: {cfg_path} already has iai-mcp — no change"
    servers["iai-mcp"] = new_entry
    cfg_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    return f"Claude Desktop: patched {cfg_path} (iai-mcp registered)"


def _patch_claude_code_config(action: str) -> str:
    from iai_mcp import cli as _cli
    import json as _json

    cfg_path = Path.home() / ".claude.json"

    if action == "uninstall":
        if not cfg_path.exists():
            return "Claude Code: ~/.claude.json absent — skipped"
        try:
            data = _json.loads(cfg_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            return f"Claude Code: ~/.claude.json unreadable ({type(e).__name__}) — skipped"
        servers = data.get("mcpServers", {})
        if "iai-mcp" in servers:
            servers.pop("iai-mcp")
            data["mcpServers"] = servers
            cfg_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
            return "Claude Code: removed iai-mcp from ~/.claude.json"
        return "Claude Code: iai-mcp not in ~/.claude.json — no change"

    try:
        entry = _build_iai_mcp_server_entry()
    except FileNotFoundError as exc:
        entry = {
            "type": "stdio",
            "command": "node",
            "args": ["<run: cd mcp-wrapper && npm run build>"],
            "env": {
                "IAI_MCP_PYTHON": _cli.sys.executable,
                "IAI_MCP_STORE": str(Path.home() / ".iai-mcp"),
                "TRANSFORMERS_VERBOSITY": "error",
                "TOKENIZERS_PARALLELISM": "false",
            },
        }
        print(
            f"WARN: MCP wrapper not found — ~/.claude.json entry written with "
            f"placeholder args. Build it first: cd mcp-wrapper && npm run build. "
            f"({exc})",
            file=_cli.sys.stderr,
        )
    else:
        entry.setdefault("type", "stdio")

    if not cfg_path.exists():
        cfg_path.write_text(_json.dumps({"mcpServers": {"iai-mcp": entry}}, indent=2), encoding="utf-8")
        return "Claude Code: created ~/.claude.json with iai-mcp registered"

    try:
        data = _json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"Claude Code: ~/.claude.json unreadable ({type(e).__name__}) — skipped"

    servers = data.setdefault("mcpServers", {})
    if servers.get("iai-mcp") == entry:
        return "Claude Code: ~/.claude.json already has iai-mcp — no change"
    servers["iai-mcp"] = entry
    cfg_path.write_text(_json.dumps(data, indent=2), encoding="utf-8")
    return "Claude Code: patched ~/.claude.json (iai-mcp registered)"


_CAPTURE_HOOK_MARKER = "iai-mcp-session-capture.sh"
_TURN_HOOK_MARKER = "iai-mcp-turn-capture.sh"
_SESSION_RECALL_HOOK_MARKER = "iai-mcp-session-recall.sh"
_PER_TURN_RECALL_HOOK_MARKER = "iai-mcp-per-turn-recall.sh"


def _session_recall_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-session-recall.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-session-recall.sh"
    settings = Path.home() / ".claude" / "settings.json"
    return src, dst, settings


def _per_turn_recall_hook_paths() -> tuple:
    src = _res.files("iai_mcp") / "_deploy" / "hooks" / "iai-mcp-per-turn-recall.sh"
    dst = Path.home() / ".claude" / "hooks" / "iai-mcp-per-turn-recall.sh"
    return src, dst


def _load_settings(path):
    import json as _json
    if not path.exists():
        return {}
    try:
        return _json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cmd_capture_hooks_install(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import json as _json
    import stat

    target = getattr(args, "target", "claude")
    # Under "all", a host whose config dir is absent is skipped — installing
    # would fabricate ~/.cursor / ~/.gemini/config / ~/.hermes / ~/.openclaw
    # on machines that never had the host. Explicit targets install anyway.
    host_rcs: list[int] = []
    if target in ("codex", "all"):
        from iai_mcp.cli._codex_hooks import install_codex_hooks
        rc = install_codex_hooks()
        if target == "codex":
            return rc
        host_rcs.append(rc)
    if target in ("cursor", "all"):
        from iai_mcp.cli._cursor_hooks import _cursor_home, install_cursor_hooks
        if target == "cursor":
            return install_cursor_hooks()
        if _cursor_home().exists():
            host_rcs.append(install_cursor_hooks())
        else:
            print(f"cursor: {_cursor_home()} absent — skipped")
    if target in ("antigravity", "all"):
        from iai_mcp.cli._antigravity_hooks import (
            _antigravity_config_dir,
            install_antigravity_hooks,
        )
        if target == "antigravity":
            return install_antigravity_hooks()
        if _antigravity_config_dir().exists():
            host_rcs.append(install_antigravity_hooks())
        else:
            print(f"antigravity: {_antigravity_config_dir()} absent — skipped")
    if target in ("hermes", "all"):
        from iai_mcp.cli._hermes_hooks import _hermes_home, install_hermes_hooks
        if target == "hermes":
            return install_hermes_hooks()
        if _hermes_home().exists():
            host_rcs.append(install_hermes_hooks())
        else:
            print(f"hermes: {_hermes_home()} absent — skipped")
    if target in ("openclaw", "all"):
        from iai_mcp.cli._openclaw_mcp import _openclaw_home, install_openclaw_mcp
        if target == "openclaw":
            return install_openclaw_mcp()
        if _openclaw_home().exists():
            host_rcs.append(install_openclaw_mcp())
        else:
            print(f"openclaw: {_openclaw_home()} absent — skipped")

    src, dst, settings = _capture_hook_paths()
    turn_src, turn_dst = _turn_hook_paths()

    if not src.exists():
        print(f"ERROR: hook template missing in package data: {src}", file=_cli.sys.stderr)
        return 1
    if not turn_src.exists():
        print(f"ERROR: turn-hook template missing in package data: {turn_src}", file=_cli.sys.stderr)
        return 1

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())
    dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"installed: {dst}")

    turn_dst.parent.mkdir(parents=True, exist_ok=True)
    turn_dst.write_bytes(turn_src.read_bytes())
    turn_dst.chmod(turn_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    print(f"installed: {turn_dst}")

    settings.parent.mkdir(parents=True, exist_ok=True)
    data = _load_settings(settings)
    data.setdefault("hooks", {})
    stop_list = data["hooks"].setdefault("Stop", [])
    submit_list = data["hooks"].setdefault("UserPromptSubmit", [])

    hook_cmd = f"bash {dst}"
    turn_cmd = f"bash {turn_dst}"

    already_stop = any(
        any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    if already_stop:
        print(f"settings.json already has Stop hook — no change")
    else:
        stop_list.append({"hooks": [{"type": "command", "command": hook_cmd, "timeout": 35}]})
        print(f"patched: {settings} (Stop hook registered)")

    already_turn = any(
        any(_TURN_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in submit_list
    )
    if already_turn:
        print(f"settings.json already has UserPromptSubmit hook — no change")
    else:
        submit_list.append({"hooks": [{"type": "command", "command": turn_cmd, "timeout": 5}]})
        print(f"patched: {settings} (UserPromptSubmit hook registered)")

    pt_src, pt_dst = _per_turn_recall_hook_paths()
    if pt_src.exists():
        pt_dst.parent.mkdir(parents=True, exist_ok=True)
        pt_dst.write_bytes(pt_src.read_bytes())
        pt_dst.chmod(pt_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"installed: {pt_dst}")

        pt_cmd = f"bash {pt_dst}"
        already_pt = any(
            any(_PER_TURN_RECALL_HOOK_MARKER in (h.get("command") or "")
                for h in (entry.get("hooks") or []))
            for entry in submit_list
        )
        if already_pt:
            print("settings.json already has per-turn recall hook — no change")
        else:
            submit_list.append(
                {"hooks": [{"type": "command", "command": pt_cmd, "timeout": 5}]}
            )
            print(f"patched: {settings} (per-turn recall hook registered)")
    else:
        print(f"WARN: per-turn recall hook template missing in package data: {pt_src}")

    src_recall, dst_recall, _ = _session_recall_hook_paths()
    if src_recall.exists():
        dst_recall.parent.mkdir(parents=True, exist_ok=True)
        dst_recall.write_bytes(src_recall.read_bytes())
        dst_recall.chmod(dst_recall.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        print(f"installed: {dst_recall}")

        ss_list = data["hooks"].setdefault("SessionStart", [])
        recall_cmd = f"bash {dst_recall}"
        already_recall = any(
            any(_SESSION_RECALL_HOOK_MARKER in (h.get("command") or "")
                for h in (entry.get("hooks") or []))
            for entry in ss_list
        )
        if already_recall:
            print("settings.json already has SessionStart hook — no change")
        else:
            ss_list.append({
                "matcher": "startup|resume|clear|compact",
                "hooks": [{"type": "command", "command": recall_cmd, "timeout": 30}],
            })
            print(f"patched: {settings} (SessionStart hook registered)")
    else:
        print(f"WARN: recall hook template missing in package data: {src_recall}")

    settings.write_text(_json.dumps(data, indent=2), encoding="utf-8")

    code_msg = _patch_claude_code_config("install")
    print(code_msg)
    desktop_msg = _patch_claude_desktop_config("install")
    print(desktop_msg)

    print("\nNext: fully quit + relaunch Claude Code AND Claude Desktop")
    print("      so both pick up the registration (macOS: `killall Claude`).")
    print("Verify: iai-mcp capture-hooks status")
    return max(host_rcs, default=0)


def cmd_capture_hooks_uninstall(args: argparse.Namespace) -> int:
    import json as _json

    target = getattr(args, "target", "claude")
    host_rcs: list[int] = []
    if target in ("codex", "all"):
        from iai_mcp.cli._codex_hooks import uninstall_codex_hooks
        rc = uninstall_codex_hooks()
        if target == "codex":
            return rc
        host_rcs.append(rc)
    if target in ("cursor", "all"):
        from iai_mcp.cli._cursor_hooks import uninstall_cursor_hooks
        rc = uninstall_cursor_hooks()
        if target == "cursor":
            return rc
        host_rcs.append(rc)
    if target in ("antigravity", "all"):
        from iai_mcp.cli._antigravity_hooks import uninstall_antigravity_hooks
        rc = uninstall_antigravity_hooks()
        if target == "antigravity":
            return rc
        host_rcs.append(rc)
    if target in ("hermes", "all"):
        from iai_mcp.cli._hermes_hooks import uninstall_hermes_hooks
        rc = uninstall_hermes_hooks()
        if target == "hermes":
            return rc
        host_rcs.append(rc)
    if target in ("openclaw", "all"):
        from iai_mcp.cli._openclaw_mcp import uninstall_openclaw_mcp
        rc = uninstall_openclaw_mcp()
        if target == "openclaw":
            return rc
        host_rcs.append(rc)

    _, dst, settings = _capture_hook_paths()
    _, turn_dst = _turn_hook_paths()
    _, dst_recall, _ = _session_recall_hook_paths()
    _, pt_dst = _per_turn_recall_hook_paths()

    if dst.exists():
        dst.unlink()
        print(f"removed: {dst}")
    else:
        print(f"(not present) {dst}")

    if turn_dst.exists():
        turn_dst.unlink()
        print(f"removed: {turn_dst}")
    else:
        print(f"(not present) {turn_dst}")

    if dst_recall.exists():
        dst_recall.unlink()
        print(f"removed: {dst_recall}")
    else:
        print(f"(not present) {dst_recall}")

    if pt_dst.exists():
        pt_dst.unlink()
        print(f"removed: {pt_dst}")
    else:
        print(f"(not present) {pt_dst}")

    if settings.exists():
        data = _load_settings(settings)
        changed = False
        for key, marker in (
            ("Stop", _CAPTURE_HOOK_MARKER),
            ("UserPromptSubmit", _TURN_HOOK_MARKER),
            ("UserPromptSubmit", _PER_TURN_RECALL_HOOK_MARKER),
        ):
            entries = data.get("hooks", {}).get(key, [])
            kept = [
                entry for entry in entries
                if not any(marker in (h.get("command") or "")
                           for h in (entry.get("hooks") or []))
            ]
            if len(kept) != len(entries):
                if kept:
                    data["hooks"][key] = kept
                else:
                    data["hooks"].pop(key, None)
                changed = True
                print(f"patched: {settings} ({key} entry removed)")
        if changed:
            settings.write_text(_json.dumps(data, indent=2), encoding="utf-8")
        else:
            print(f"(no hook entry to remove) {settings}")

        data = _load_settings(settings)
        ss_list = data.get("hooks", {}).get("SessionStart", [])
        kept_ss = [
            entry for entry in ss_list
            if not any(_SESSION_RECALL_HOOK_MARKER in (h.get("command") or "")
                       for h in (entry.get("hooks") or []))
        ]
        if len(kept_ss) != len(ss_list):
            if kept_ss:
                data["hooks"]["SessionStart"] = kept_ss
            else:
                data["hooks"].pop("SessionStart", None)
            settings.write_text(_json.dumps(data, indent=2), encoding="utf-8")
            print(f"patched: {settings} (SessionStart entry removed)")
        else:
            print(f"(no SessionStart entry to remove) {settings}")

    code_msg = _patch_claude_code_config("uninstall")
    print(code_msg)
    desktop_msg = _patch_claude_desktop_config("uninstall")
    print(desktop_msg)

    return max(host_rcs, default=0)


def cmd_capture_hooks_status(args: argparse.Namespace) -> int:
    from iai_mcp import cli as _cli
    import json as _json

    target = getattr(args, "target", "claude")
    # Under "all", a host whose config dir is absent is skipped — an
    # uninstalled host is not a failure of this machine's wiring.
    if target in ("codex", "all"):
        from iai_mcp.cli._codex_hooks import status_codex_hooks
        rc = status_codex_hooks()
        if target == "codex":
            return rc
        print()
    if target in ("cursor", "all"):
        from iai_mcp.cli._cursor_hooks import _cursor_home, status_cursor_hooks
        if target == "cursor":
            return status_cursor_hooks()
        if _cursor_home().exists():
            status_cursor_hooks()
        else:
            print(f"cursor: {_cursor_home()} absent — skipped")
        print()
    if target in ("antigravity", "all"):
        from iai_mcp.cli._antigravity_hooks import (
            _antigravity_config_dir,
            status_antigravity_hooks,
        )
        if target == "antigravity":
            return status_antigravity_hooks()
        if _antigravity_config_dir().exists():
            status_antigravity_hooks()
        else:
            print(f"antigravity: {_antigravity_config_dir()} absent — skipped")
        print()
    if target in ("hermes", "all"):
        from iai_mcp.cli._hermes_hooks import _hermes_home, status_hermes_hooks
        if target == "hermes":
            return status_hermes_hooks()
        if _hermes_home().exists():
            status_hermes_hooks()
        else:
            print(f"hermes: {_hermes_home()} absent — skipped")
        print()
    if target in ("openclaw", "all"):
        from iai_mcp.cli._openclaw_mcp import _openclaw_home, status_openclaw_mcp
        if target == "openclaw":
            return status_openclaw_mcp()
        if _openclaw_home().exists():
            status_openclaw_mcp()
        else:
            print(f"openclaw: {_openclaw_home()} absent — skipped")
        print()

    src, dst, settings = _capture_hook_paths()
    turn_src, turn_dst = _turn_hook_paths()
    src_recall, dst_recall, _ = _session_recall_hook_paths()
    pt_src, pt_dst = _per_turn_recall_hook_paths()

    def _installed_state(template, installed) -> str:
        # PRESENT alone hides a stale install: an old hook body silently
        # lacks the features the daemon expects until reinstall.
        if not installed.exists():
            return "MISSING"
        if not template.exists():
            return "PRESENT"
        try:
            if installed.read_bytes() == Path(str(template)).read_bytes():
                return "PRESENT (matches template)"
        except OSError:
            return "PRESENT (unreadable)"
        return "STALE (differs from packaged template — rerun capture-hooks install)"

    print(f"Stop template:        {src}  {'PRESENT' if src.exists() else 'MISSING'}")
    print(f"Stop installed:       {dst}  {_installed_state(src, dst)}")
    print(f"Turn template:        {turn_src}  {'PRESENT' if turn_src.exists() else 'MISSING'}")
    print(f"Turn installed:       {turn_dst}  {_installed_state(turn_src, turn_dst)}")
    print(f"Recall template:      {src_recall}  {'PRESENT' if src_recall.exists() else 'MISSING'}")
    print(f"Recall installed:     {dst_recall}  {_installed_state(src_recall, dst_recall)}")
    print(f"Per-turn template:    {pt_src}  {'PRESENT' if pt_src.exists() else 'MISSING'}")
    print(f"Per-turn installed:   {pt_dst}  {_installed_state(pt_src, pt_dst)}")

    data = _load_settings(settings)
    stop_list = data.get("hooks", {}).get("Stop", [])
    submit_list = data.get("hooks", {}).get("UserPromptSubmit", [])
    wired = any(
        any(_CAPTURE_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in stop_list
    )
    turn_wired = any(
        any(_TURN_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in submit_list
    )
    ss_list = data.get("hooks", {}).get("SessionStart", [])
    recall_wired = any(
        any(_SESSION_RECALL_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in ss_list
    )
    pt_wired = any(
        any(_PER_TURN_RECALL_HOOK_MARKER in (h.get("command") or "")
            for h in (entry.get("hooks") or []))
        for entry in submit_list
    )
    print(f"Claude Code settings.json Stop:             {settings}  {'WIRED' if wired else 'NOT WIRED'}")
    print(f"Claude Code settings.json UserPromptSubmit: {settings}  {'WIRED' if turn_wired else 'NOT WIRED'}")
    print(f"Claude Code settings.json SessionStart:     {settings}  {'WIRED' if recall_wired else 'NOT WIRED'}")
    print(f"Claude Code settings.json per-turn recall:  {settings}  {'WIRED' if pt_wired else 'NOT WIRED'}")

    desktop_cfg = _cli._claude_desktop_config_path()
    if desktop_cfg is None:
        desktop_line = "Claude Desktop: not installed"
        desktop_wired = False
    elif not desktop_cfg.exists():
        desktop_line = f"Claude Desktop: {desktop_cfg} MISSING"
        desktop_wired = False
    else:
        try:
            d = _json.loads(desktop_cfg.read_text(encoding="utf-8"))
            desktop_wired = "iai-mcp" in d.get("mcpServers", {})
            desktop_line = f"Claude Desktop: {desktop_cfg}  {'WIRED' if desktop_wired else 'NOT WIRED'}"
        except (OSError, ValueError):
            desktop_line = f"Claude Desktop: {desktop_cfg} (unreadable)"
            desktop_wired = False
    print(desktop_line)

    ok = (
        dst.exists() and wired
        and turn_dst.exists() and turn_wired
        and dst_recall.exists() and recall_wired
        and pt_dst.exists() and pt_wired
    )
    desktop_problem = desktop_cfg is not None and desktop_cfg.exists() and not desktop_wired

    if ok and not desktop_problem:
        print(f"\nstatus: ACTIVE — Stop + UserPromptSubmit + SessionStart + per-turn hooks wired "
              f"(Claude Code{'; Desktop also wired' if desktop_wired else ''})")
        return 0
    msg = []
    if not ok:
        msg.append("Claude Code not fully wired")
    if desktop_problem:
        msg.append("Claude Desktop present but iai-mcp not registered")
    print(f"\nstatus: INACTIVE — {'; '.join(msg)}. Run: iai-mcp capture-hooks install")
    return 1
