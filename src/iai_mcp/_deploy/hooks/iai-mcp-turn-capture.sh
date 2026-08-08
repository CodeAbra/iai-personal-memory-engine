#!/bin/sh
# IAI-MCP UserPromptSubmit hook — per-turn ambient capture.
#
# Pure file IO: appends one JSONL event line per new transcript turn to
# ~/.iai-mcp/.deferred-captures/{session_id}.live.jsonl. Inline system
# python3 (stdlib only) so cold-start stays under the per-turn latency
# budget; the equivalent `iai-mcp capture-turn-deferred` CLI exists for
# manual / debugging use.
#
# At-rest contract: this writer STAGES PLAINTEXT at mode 0600 (stdlib
# python carries no AES-GCM); the daemon encrypts staged lines on its next
# drain pass (capture.reencrypt_plaintext_spool_lines). Readers accept
# both wire formats — never assume this file is uniformly either.
#
# Fail-safe: any error exits 0. Hard 5s wall-clock timeout.

set -u
# Spool files must never be world-readable.
umask 077
input=$(cat 2>/dev/null || true)

# Extract session_id and transcript_path in a single subprocess call.
_extract_tmp=$(mktemp 2>/dev/null || echo "/tmp/iai-mcp-turn-extract-$$.tmp")
if command -v jq >/dev/null 2>&1; then
  printf '%s' "$input" | jq -r '(.session_id // "") + "\t" + (.transcript_path // "")' >"$_extract_tmp" 2>/dev/null
else
  printf '%s' "$input" | /usr/bin/python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print((d.get('session_id') or '') + '\t' + (d.get('transcript_path') or ''))
except Exception:
    print('\t')
" >"$_extract_tmp" 2>/dev/null
fi
_TAB=$(printf '\t')
IFS="$_TAB" read -r session_id transcript_path < "$_extract_tmp"
rm -f "$_extract_tmp" 2>/dev/null || true

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/turn-capture-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

if [ -z "$session_id" ] || [ -z "$transcript_path" ]; then
  echo "$ts skipped: missing session_id or transcript_path" >> "$log" 2>/dev/null
  exit 0
fi

PY_SCRIPT='
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MAX_TURNS = 100_000

session_id = sys.argv[1]
stdin_path = Path(sys.argv[2]).expanduser()
home = Path(os.environ.get("HOME", str(Path.home())))

# Resolve the canonical transcript for this session.
#
# Claude Code sometimes passes a transcript_path via hook stdin that is stale,
# points to an empty file, or belongs to a different session entirely.  The
# stdin path is the result of whatever the host process had on hand at fire
# time, which may be empty or wrong even when it physically exists on disk.
#
# Strategy: ALWAYS scan ~/.claude/projects/*/{session_id}.jsonl first.  If the
# canonical file exists and is non-empty, use it — it is guaranteed to contain
# this session only.  Fall back to the stdin path only when the canonical file
# is absent or empty (early first-fire timing race).  If neither source has
# content, exit cleanly — the Stop hook will capture at session end.
#
# This makes offset accounting safe: the offset is always relative to one
# consistent file (canonical > stdin), preventing line-number skew across fires.

def _scan_canonical(home: Path, session_id: str):
    projects_dir = home / ".claude" / "projects"
    if not projects_dir.is_dir():
        return None
    target = f"{session_id}.jsonl"
    for project_dir in projects_dir.iterdir():
        candidate = project_dir / target
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None

canonical = _scan_canonical(home, session_id)
if canonical is not None:
    transcript_path = canonical
elif stdin_path.exists() and stdin_path.stat().st_size > 0:
    transcript_path = stdin_path
else:
    sys.exit(0)

deferred_dir = home / ".iai-mcp" / ".deferred-captures"
state_dir = home / ".iai-mcp" / ".capture-state"
deferred_dir.mkdir(parents=True, exist_ok=True)
state_dir.mkdir(parents=True, exist_ok=True)
live = deferred_dir / f"{session_id}.live.jsonl"
offset = state_dir / f"{session_id}.offset"

prev = 0
if offset.exists():
    try:
        prev = int(offset.read_text().strip() or "0")
    except ValueError:
        prev = 0

with transcript_path.open() as fh:
    lines = fh.readlines()
total = len(lines)
if prev > total:
    # Transcript is shorter than the stored offset — it was rotated or
    # replaced.  Preserve the existing offset and skip capture to avoid
    # re-emitting old turns or clobbering a valid large offset.
    tmp = state_dir / f"{session_id}.offset.tmp"
    tmp.write_text(str(prev))
    os.replace(tmp, offset)
    sys.exit(0)

cwd = os.getcwd()
emitted = 0
consumed = 0

_NOISE_STARTSWITH = (
    "<command-message>",
    "<command-name>",
    "Base directory for this skill:",
    "<task-notification>",
)
_NOISE_EQUALS = ("[Request interrupted by user]",)

def _is_noise(text):
    for prefix in _NOISE_STARTSWITH:
        if text.startswith(prefix):
            return True
    for exact in _NOISE_EQUALS:
        if text == exact:
            return True
    return False

def parse_line(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if "step_index" in obj and "source" in obj:
        # Antigravity transcript_full lines: conversational turns only.
        src, typ = obj.get("source"), obj.get("type")
        if src == "USER_EXPLICIT" and typ == "USER_INPUT":
            agy_role = "user"
        elif src == "MODEL" and typ == "PLANNER_RESPONSE":
            agy_role = "assistant"
        else:
            return None
        agy_text = str(obj.get("content") or "").strip()
        if not agy_text or _is_noise(agy_text):
            return None
        return agy_role, agy_text, None, obj.get("created_at"), []
    msg = obj.get("message") if isinstance(obj.get("message"), dict) else obj
    # Claude puts the role in top-level "type", Cursor in top-level "role"
    # with the content nested under "message".
    role = obj.get("type") or msg.get("role") or obj.get("role", "")
    if role not in {"user", "assistant"}:
        return None
    content = msg.get("content", "")
    tools = []
    if isinstance(content, list):
        parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        text = "\n".join(parts).strip()
        if role == "assistant":
            tools = [
                str(b.get("name") or "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
                and b.get("name")
            ]
    else:
        text = str(content).strip()
    src_uuid = obj.get("uuid") or None
    src_ts = obj.get("timestamp") or None
    if not text:
        # Action-only assistant entries carry the mechanics of the episode;
        # their tool names attach to the nearest text turn of the response.
        # A text-less user entry is a tool RESULT, not a conversational
        # boundary — it must not reach the caller at all.
        if role == "assistant" and tools:
            return role, "", src_uuid, src_ts, tools
        return None
    if _is_noise(text):
        # Noise text never becomes a record, but an assistant noise entry
        # may still carry tool calls, and a user noise entry is still a
        # conversational boundary; both signal via empty text.
        if role == "assistant" and tools:
            return role, "", src_uuid, src_ts, tools
        if role == "user":
            return role, "", src_uuid, src_ts, []
        return None
    # Extract transcript-native identity when present.  Real Claude Code
    # JSONL lines carry top-level "uuid" and "timestamp" fields.  Test
    # fixtures may lack these; None is the safe fallback so capture.py
    # _idem_tag falls back to the (session, role, ts, text) key.
    return role, text, src_uuid, src_ts, tools


def _tools_trailer(names):
    seen = []
    for n in names:
        if n not in seen:
            seen.append(n)
    if not seen:
        return ""
    extra = f" +{len(seen) - 8}" if len(seen) > 8 else ""
    return "\n[tools: " + ", ".join(seen[:8]) + extra + "]"

if total > prev:
    need_header = (not live.exists()) or live.stat().st_size == 0
    with live.open("a") as out:
        if need_header:
            header = {
                "version": 1,
                "deferred_at": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "cwd": cwd,
            }
            out.write(json.dumps(header, ensure_ascii=False) + "\n")
        pending_tools = []
        for raw in lines[prev:]:
            if emitted >= MAX_TURNS:
                break
            consumed += 1
            parsed = parse_line(raw)
            if parsed is None:
                continue
            role, text, src_uuid, src_ts, tools = parsed
            if role == "assistant" and not text:
                # Action-only entry: hold the tool names for the next text
                # turn of the response so the episode records its mechanics.
                pending_tools.extend(tools)
                continue
            if role == "user" and not text:
                # Conversational user boundary with no recordable text (a
                # noise line, not a tool result): mechanics of the previous
                # response must not leak across it.
                pending_tools = []
                continue
            if role == "assistant":
                # The floor guards the BARE text: a trailer must never turn
                # an otherwise-skipped stub into a record (mirror of
                # capture.MIN_CAPTURE_LEN — keep the two in lockstep).
                if len(text) >= 12:
                    text = text + _tools_trailer(pending_tools + tools)
                    pending_tools = []
                else:
                    pending_tools.extend(tools)
            else:
                # A user turn closes the previous response; tool names that
                # never met a text turn must not leak across the boundary.
                pending_tools = []
            event = {
                "text": text,
                "cue": f"session {session_id} turn",
                "tier": "episodic",
                "role": role,
                "ts": src_ts if src_ts else datetime.now(timezone.utc).isoformat(),
            }
            if src_uuid:
                event["source_uuid"] = src_uuid
            out.write(json.dumps(event, ensure_ascii=False) + "\n")
            emitted += 1

new_offset = prev + consumed
tmp = state_dir / f"{session_id}.offset.tmp"
with open(tmp, "w") as _f:
    _f.write(str(new_offset))
    _f.flush()
    os.fsync(_f.fileno())
os.replace(tmp, offset)

# --- Freshness gate (best-effort, capture-first) ---
#
# Check whether new cross-session memory has arrived since the last prompt.
# This runs AFTER the capture write and offset persist above are already
# complete.  Any exception in the gate is swallowed — the capture is always
# independent of the gate outcome.
#
# Two trigger signals:
#   Signal A: MAX(created_at) in the Hippo SQLite file advanced past the
#             per-session watermark (another session was drained into the store).
#   Signal B: total byte-size of OTHER sessions live files grew since last look
#             (another session is still open and wrote new turns not yet drained).
#
# When triggered, the gate contacts the daemon via a raw stdlib AF_UNIX socket
# and asks for a refreshed session-start brief.  On a non-empty rendered brief,
# the gate emits additionalContext JSON to stdout — the channel Claude Code reads
# for per-prompt context injection.
#
# Custom-store isolation: when IAI_MCP_STORE points to a non-default location
# and no IAI_DAEMON_SOCKET_PATH override is set, the gate skips the socket RPC
# entirely.  Contacting the default daemon socket would surface the wrong
# store context.  Explicit IAI_DAEMON_SOCKET_PATH always wins.
#
# Sidecar and watermark reads are HOME-based regardless of IAI_MCP_STORE,
# matching the behavior of the manual cmd_session_refresh_if_stale command in
# cli.py. Only the socket RPC is store-guarded.
#
# Hosts whose submit event cannot inject context set
# IAI_MCP_TURN_INJECT_DISABLED — the gate (and its socket RPC) is skipped;
# the capture above already completed.
if os.environ.get("IAI_MCP_TURN_INJECT_DISABLED"):
    sys.exit(0)
try:
    # --- Helper functions (all HOME-based, no IAI_MCP_STORE awareness) ---

    def _gate_get_max_created_at():
        # The store-advance sidecar is stamped by the writers on every record
        # write. The hook must never open the store file itself — its on-disk
        # engine format is not a hook contract.
        try:
            raw = (home / ".iai-mcp" / "hippo" / ".max-created-at").read_text(
                encoding="utf-8"
            ).strip()
            return raw or None
        except OSError:
            return None

    def _gate_utc_iso(ts):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.isoformat()
        except (TypeError, ValueError):
            return ts

    def _gate_read_watermark(sid):
        p = home / ".iai-mcp" / ".capture-state" / f"{sid}.watermark"
        try:
            if not p.exists():
                return None
            return p.read_text().strip() or None
        except OSError:
            return None

    def _gate_write_watermark(sid, ts):
        d = home / ".iai-mcp" / ".capture-state"
        d.mkdir(parents=True, exist_ok=True)
        tmp_wm = d / f"{sid}.watermark.tmp"
        tmp_wm.write_text(_gate_utc_iso(ts))
        os.replace(tmp_wm, d / f"{sid}.watermark")

    def _gate_get_other_live_size(sid):
        try:
            dd = home / ".iai-mcp" / ".deferred-captures"
            if not dd.exists():
                return 0
            own = f"{sid}.live.jsonl"
            total_sz = 0
            for entry in dd.iterdir():
                if not entry.is_file():
                    continue
                if not entry.name.endswith(".live.jsonl"):
                    continue
                if entry.name == own:
                    continue
                try:
                    total_sz += entry.stat().st_size
                except OSError:
                    pass
            return total_sz
        except Exception:
            return 0

    def _gate_read_fingerprint(sid):
        p = home / ".iai-mcp" / ".capture-state" / f"{sid}.live-fingerprint"
        try:
            if not p.exists():
                return None
            raw = p.read_text().strip()
            if not raw:
                return None
            return int(raw)
        except (OSError, ValueError):
            return None

    def _gate_write_fingerprint(sid, total_sz):
        d = home / ".iai-mcp" / ".capture-state"
        d.mkdir(parents=True, exist_ok=True)
        tmp_fp = d / f"{sid}.live-fingerprint.tmp"
        tmp_fp.write_text(str(total_sz))
        os.replace(tmp_fp, d / f"{sid}.live-fingerprint")

    # --- Gate control flow (mirrors cmd_session_refresh_if_stale in cli.py) ---

    current = _gate_get_max_created_at()
    if current is not None:
        wm = _gate_read_watermark(session_id)
        live_size = _gate_get_other_live_size(session_id)

        if wm is None:
            # First prompt of this session: set baselines without triggering.
            _gate_write_watermark(session_id, current)
            _gate_write_fingerprint(session_id, live_size)
        else:
            store_advanced = _gate_utc_iso(current) > _gate_utc_iso(wm)
            fp = _gate_read_fingerprint(session_id)
            if fp is None:
                _gate_write_fingerprint(session_id, live_size)
                fp = live_size
            live_grew = live_size > fp

            if store_advanced or live_grew:
                # New memory exists — check custom-store isolation before opening socket.
                env_store = os.environ.get("IAI_MCP_STORE")
                is_custom = False
                if env_store:
                    try:
                        is_custom = (
                            Path(env_store).expanduser().resolve()
                            != (home / ".iai-mcp").resolve()
                        )
                    except Exception:
                        is_custom = False

                sock_env = os.environ.get("IAI_DAEMON_SOCKET_PATH")
                if not sock_env and is_custom:
                    # Custom store with no explicit socket — do not contact the
                    # default daemon socket (it would surface a different store
                    # context).  Leave sidecars unchanged and skip the RPC.
                    pass
                else:
                    sock_path = sock_env or str(home / ".iai-mcp" / ".daemon.sock")

                    # Lazy import: only in the trigger branch to keep common-path import-light.
                    import socket as _sock
                    import sys as _sys
                    import time as _time

                    # Unresponsive-daemon backoff: a reply that never comes
                    # costs the full socket timeout on EVERY prompt, so one
                    # timed-out attempt opens a cooldown window during which
                    # the RPC is skipped. A successful reply clears it.
                    _cooldown = home / ".iai-mcp" / ".capture-state" / f"{session_id}.refresh-cooldown"
                    _COOLDOWN_SEC = 600
                    _skip_rpc = False
                    try:
                        if (_time.time() - _cooldown.stat().st_mtime) < _COOLDOWN_SEC:
                            _skip_rpc = True
                    except OSError:
                        pass
                    if _skip_rpc:
                        raise SystemExit(0)

                    _MAX_REPLY = 2 * 1024 * 1024  # 2 MB buffer cap
                    _replied = False
                    _conn = None
                    try:
                        _conn = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
                        _conn.settimeout(3.0)
                        _conn.connect(sock_path)
                        req_bytes = (
                            json.dumps({
                                "jsonrpc": "2.0",
                                "id": 1,
                                "method": "session_refresh_if_stale",
                                "params": {"watermark": wm, "session_id": session_id},
                            }) + "\n"
                        ).encode("utf-8")
                        _conn.sendall(req_bytes)

                        # Accumulate recv chunks until a newline byte arrives,
                        # the socket closes, or the 3 s settimeout fires.
                        # A single recv call silently truncates large briefs.
                        buf = b""
                        while len(buf) < _MAX_REPLY:
                            try:
                                chunk = _conn.recv(16384)
                            except Exception:
                                break
                            if not chunk:
                                break
                            buf += chunk
                            if b"\n" in buf:
                                break

                        nl_pos = buf.find(b"\n")
                        frame = buf[:nl_pos] if nl_pos >= 0 else buf
                        if frame:
                            _replied = True
                            try:
                                resp_obj = json.loads(frame.decode("utf-8"))
                                result_obj = resp_obj.get("result") or {}
                                rendered = result_obj.get("rendered") or ""
                                new_max = result_obj.get("new_max_ts") or current
                                if rendered:
                                    payload = {
                                        "hookSpecificOutput": {
                                            "hookEventName": "UserPromptSubmit",
                                            "additionalContext": rendered,
                                        }
                                    }
                                    _sys.stdout.write(
                                        json.dumps(payload, ensure_ascii=False)
                                    )
                                    _gate_write_watermark(session_id, new_max)
                                    _gate_write_fingerprint(session_id, live_size)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    finally:
                        if _conn is not None:
                            try:
                                _conn.close()
                            except Exception:
                                pass
                        try:
                            if _replied:
                                _cooldown.unlink(missing_ok=True)
                            else:
                                _cooldown.parent.mkdir(parents=True, exist_ok=True)
                                _cooldown.touch()
                        except OSError:
                            pass
except Exception:
    pass
'

if command -v timeout >/dev/null 2>&1; then
  timeout 5 /usr/bin/python3 -c "$PY_SCRIPT" "$session_id" "$transcript_path" 2>/dev/null
elif command -v gtimeout >/dev/null 2>&1; then
  gtimeout 5 /usr/bin/python3 -c "$PY_SCRIPT" "$session_id" "$transcript_path" 2>/dev/null
else
  /usr/bin/python3 -c "$PY_SCRIPT" "$session_id" "$transcript_path" 2>/dev/null
fi
rc=$?

echo "$ts session=$session_id rc=$rc" >> "$log" 2>/dev/null

exit 0
