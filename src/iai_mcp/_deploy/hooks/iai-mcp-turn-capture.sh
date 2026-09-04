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

# The full stdin payload is staged to a file for the immediate-capture block
# below to read .prompt from: prompt text may contain newlines/tabs, and a
# `read` line-split (used for session_id/transcript_path/prompt_id below)
# would truncate or shift fields on such input.
_payload_tmp=$(mktemp 2>/dev/null || echo "/tmp/iai-mcp-turn-payload-$$.tmp")
_extract_tmp=""
# Registered before either tmp file is populated so every exit path below
# (including the early guards) cleans up: a plaintext prompt copy must
# never be orphaned in TMPDIR.
trap 'rm -f "$_payload_tmp" "$_extract_tmp" 2>/dev/null || true' EXIT
printf '%s' "$input" > "$_payload_tmp" 2>/dev/null || true

# Extract session_id, transcript_path and prompt_id in a single subprocess
# call. prompt_id is UUID-ish (no newlines/tabs expected); its shape is
# re-validated in PY_SCRIPT before use.
_extract_tmp=$(mktemp 2>/dev/null || echo "/tmp/iai-mcp-turn-extract-$$.tmp")
if command -v jq >/dev/null 2>&1; then
  printf '%s' "$input" | jq -r '(.session_id // "") + "\t" + (.transcript_path // "") + "\t" + (.prompt_id // "")' >"$_extract_tmp" 2>/dev/null
else
  printf '%s' "$input" | /usr/bin/python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print((d.get('session_id') or '') + '\t' + (d.get('transcript_path') or '') + '\t' + (d.get('prompt_id') or ''))
except Exception:
    print('\t\t')
" >"$_extract_tmp" 2>/dev/null
fi
_TAB=$(printf '\t')
IFS="$_TAB" read -r session_id transcript_path prompt_id < "$_extract_tmp"

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/turn-capture-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
channel="settings"
[ -n "${CLAUDE_PLUGIN_ROOT:-}" ] && channel="plugin"

if [ -z "$session_id" ] || [ -z "$transcript_path" ]; then
  echo "$ts skipped: missing session_id or transcript_path channel=$channel" >> "$log" 2>/dev/null
  exit 0
fi

case "$session_id" in
  *[!A-Za-z0-9._-]*)
    echo "$ts skipped: invalid session_id channel=$channel" >> "$log" 2>/dev/null
    exit 0
    ;;
esac

PY_SCRIPT='
import fcntl
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

MAX_TURNS = 100_000

session_id = sys.argv[1]
# Session id is host-controlled text that lands in file paths.
if not re.fullmatch(r"[A-Za-z0-9._-]{1,128}", session_id):
    print("skipped: invalid session_id", file=sys.stderr)
    sys.exit(0)
stdin_path = Path(sys.argv[2]).expanduser()
home = Path(os.environ.get("HOME", str(Path.home())))

# Resolve the canonical transcript for this session.
#
# Claude Code sometimes passes a transcript_path via hook stdin that is stale,
# points to an empty file, or belongs to a different session entirely.  The
# stdin path is the result of whatever the host process had on hand at fire
# time, which may be empty or wrong even when it physically exists on disk.
#
# Strategy: ALWAYS scan {claude_root}/projects/*/{session_id}.jsonl first,
# where claude_root is the Claude configuration directory named by the
# environment, defaulting to the shared home when unset, equal, or rejected
# by validation.  If the canonical file exists and is non-empty, use it — it
# is guaranteed to contain this session only.  Fall back to the stdin path
# only when the canonical file is absent or empty (early first-fire timing
# race).  If neither source has content, there is nothing to walk yet — the
# immediate stdin-prompt capture below still runs (it needs no transcript)
# and the Stop hook covers the rest at session end.
#
# This makes offset accounting safe: the offset is always relative to one
# consistent file (canonical > stdin), preventing line-number skew across fires.

def _resolve_claude_root(home: Path):
    # CLAUDE_CONFIG_DIR is host-supplied, untrusted input: it must be an
    # absolute path, an existing directory, free of parent-directory
    # traversal segments and restricted to letters/digits/space/"._/-"
    # (space allowed: real install paths carry one; the traversal check
    # below covers the actual attack surface), or the shared home is used
    # and the rejection is logged. Returns (root, refused).
    shared = home / ".claude"
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "")
    if not raw:
        return shared, False
    candidate = Path(raw)
    if candidate == shared:
        return shared, False
    if not candidate.is_absolute():
        return shared, True
    if not re.fullmatch(r"[A-Za-z0-9._/ -]+", raw):
        return shared, True
    if ".." in candidate.parts:
        return shared, True
    if not candidate.is_dir():
        return shared, True
    return candidate, False


def _scan_canonical(claude_root: Path, session_id: str):
    projects_dir = claude_root / "projects"
    if not projects_dir.is_dir():
        return None
    target = f"{session_id}.jsonl"
    for project_dir in projects_dir.iterdir():
        candidate = project_dir / target
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None

claude_root, _config_dir_refused = _resolve_claude_root(home)
if _config_dir_refused:
    try:
        _cfg_channel = "plugin" if os.environ.get("CLAUDE_PLUGIN_ROOT") else "settings"
        _cfg_log_dir = home / ".iai-mcp" / "logs"
        _cfg_log_dir.mkdir(parents=True, exist_ok=True)
        _cfg_log_path = _cfg_log_dir / f"turn-capture-{datetime.now(timezone.utc):%Y-%m-%d}.log"
        with open(_cfg_log_path, "a") as _cfg_lf:
            _cfg_lf.write(
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                + f" config-dir-refused channel={_cfg_channel}\n"
            )
    except Exception:
        pass

canonical = _scan_canonical(claude_root, session_id)
if canonical is not None:
    transcript_path = canonical
elif stdin_path.exists() and stdin_path.stat().st_size > 0:
    transcript_path = stdin_path
else:
    transcript_path = None

deferred_dir = home / ".iai-mcp" / ".deferred-captures"
state_dir = home / ".iai-mcp" / ".capture-state"
deferred_dir.mkdir(parents=True, exist_ok=True)
state_dir.mkdir(parents=True, exist_ok=True)
live = deferred_dir / f"{session_id}.live.jsonl"
offset = state_dir / f"{session_id}.offset"
# Offset and pending tool names live in ONE snapshot file published by a
# single atomic rename: any crash or carrier race can only surface an older
# coherent pair, whose replay is deterministic and lands on the same idem
# keys. Two separate files can pair a fresh offset with stale pending and
# fabricate a trailer. The legacy offset file is still mirrored for older
# readers; when it is ahead, its position wins and pending drops.
turnstate = state_dir / f"{session_id}.turnstate.json"

_lock_fd = os.open(
    str(state_dir / f"{session_id}.capture.lock"),
    os.O_WRONLY | os.O_CREAT, 0o600,
)
try:
    fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except OSError:
    # Another carrier is draining this session right now; its walk covers
    # these lines.
    sys.exit(0)

snap_offset = -1
pending_tools = []
snap_fp = ""
if turnstate.exists():
    try:
        _snap = json.loads(turnstate.read_text())
        if isinstance(_snap, dict):
            snap_offset = int(_snap.get("offset", 0))
            _loaded = _snap.get("pending")
            if isinstance(_loaded, list):
                pending_tools = [
                    n for n in _loaded[:256] if isinstance(n, str) and n
                ]
            _loaded_fp = _snap.get("fp")
            if isinstance(_loaded_fp, str):
                snap_fp = _loaded_fp
    except Exception:
        snap_offset = -1
        pending_tools = []
        snap_fp = ""

legacy = 0
if offset.exists():
    try:
        legacy = int(offset.read_text().strip() or "0")
    except ValueError:
        legacy = 0

if snap_offset >= legacy:
    prev = max(snap_offset, 0)
else:
    # A writer that knows only the legacy offset advanced past the
    # snapshot: adopt its position and drop pending — trailer loss,
    # never fabrication.
    prev = legacy
    pending_tools = []

if transcript_path is not None:
    with transcript_path.open(encoding="utf-8") as fh:
        lines = fh.readlines()
    total = len(lines)
    # Fingerprint the RAW first-line bytes: the fp is shared between carriers
    # through the snapshot, and hashing a decoded string would couple it to
    # each carrier decode setting.
    with transcript_path.open("rb") as fh_raw:
        _first_raw = fh_raw.readline()
    stream_fp = hashlib.sha256(_first_raw).hexdigest()[:16] if _first_raw else ""
    # A changed first line means a different stream at the same path: the
    # stored offset and pending belong to a dead transcript even when the new
    # one already regrew past the old length. The length fence stays as the
    # backstop for fingerprint-less legacy snapshots and same-stream
    # truncation. Restarting at zero re-walks the new stream; replays dedup
    # on source-uuid idempotency keys where the host provides them and are
    # absorbed by the cosine gate for text-keyed hosts.
    if (snap_fp and stream_fp and snap_fp != stream_fp) or prev > total:
        prev = 0
        pending_tools = []
else:
    # First fire of a session: no transcript resolvable yet. The immediate
    # stdin-prompt capture below runs regardless; the walk below is a no-op
    # (total stays 0) and the next fire covers this prompt via its own walk.
    lines = []
    total = 0
    stream_fp = snap_fp

cwd = os.getcwd()
emitted = 0
consumed = 0

# Mirror of capture._NOISE_PATTERNS — keep the two in lockstep.
_NOISE_STARTSWITH = (
    "<command-message>",
    "<command-name>",
    "<command-args>",
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<local-command-stderr>",
    "Base directory for this skill:",
    "<task-notification>",
    "<system-reminder>",
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

def _clean_tool_name(value):
    # Mirror of capture._clean_tool_name — keep the two in lockstep.
    if not isinstance(value, str):
        return None
    name = "".join(
        ch for ch in value if ch.isprintable() and ch not in ",[]"
    ).strip()
    return name[:80].strip() or None

def parse_line(raw):
    try:
        obj = json.loads(raw)
    except Exception:
        return None
    if obj.get("type") == "response_item":
        # Codex rollout lines. function_call carries the tool name;
        # function_call_output is plumbing and must not reach the caller.
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            return None
        ptype = payload.get("type")
        if ptype == "function_call":
            name = _clean_tool_name(payload.get("name"))
            return ("assistant", "", None, obj.get("timestamp"),
                    [name] if name else [])
        if ptype != "message":
            return None
        cdx_role = payload.get("role", "")
        if cdx_role not in {"user", "assistant"}:
            return None
        content = payload.get("content", [])
        if isinstance(content, list):
            parts = [
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and isinstance(b.get("text"), str)
            ]
            cdx_text = "\n".join(p for p in parts if p).strip()
        else:
            cdx_text = str(content or "").strip()
        if cdx_text.lstrip().startswith("<environment_context>") or _is_noise(cdx_text):
            cdx_text = ""
        cdx_uuid = payload.get("id") or obj.get("uuid") or None
        if not cdx_text:
            # A filtered user turn is still a conversational boundary.
            if cdx_role == "user":
                return "user", "", cdx_uuid, obj.get("timestamp"), []
            return None
        return cdx_role, cdx_text, cdx_uuid, obj.get("timestamp"), []
    if "step_index" in obj and "source" in obj:
        # Antigravity transcript_full lines: conversational turns only.
        src, typ = obj.get("source"), obj.get("type")
        if src == "USER_EXPLICIT" and typ != "USER_INPUT":
            # Any explicit-user-sourced step is a conversational boundary.
            return "user", "", None, obj.get("created_at"), []
        if src == "USER_EXPLICIT":
            agy_role = "user"
        elif src == "MODEL" and typ == "PLANNER_RESPONSE":
            agy_role = "assistant"
        else:
            return None
        agy_tools = []
        if agy_role == "assistant":
            calls = obj.get("tool_calls")
            if isinstance(calls, list):
                agy_tools = [
                    n for n in (
                        _clean_tool_name(c.get("name"))
                        for c in calls if isinstance(c, dict)
                    ) if n
                ]
        agy_text = str(obj.get("content") or "").strip()
        if _is_noise(agy_text):
            agy_text = ""
        if not agy_text:
            if agy_role == "user":
                return "user", "", None, obj.get("created_at"), []
            if agy_tools:
                return "assistant", "", None, obj.get("created_at"), agy_tools
            return None
        return agy_role, agy_text, None, obj.get("created_at"), agy_tools
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
                n for n in (
                    _clean_tool_name(b.get("name"))
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                    and b.get("name")
                ) if n
            ]
    else:
        text = str(content).strip()
    # role:user keys by promptId, the immediate stdin capture join key;
    # role:assistant stays on the per-line uuid — assistant lines of one
    # prompt share promptId, so keying them by it would collapse a whole
    # response into one idem.
    src_uuid = ((obj.get("promptId") if role == "user" else None) or obj.get("uuid")) or None
    src_ts = obj.get("timestamp") or None
    if not text:
        # Action-only assistant entries carry the mechanics of the episode;
        # their tool names attach to the nearest text turn of the response.
        # A text-less user entry is a tool RESULT, not a conversational
        # boundary — it must not reach the caller at all.
        if role == "assistant" and tools:
            return role, "", src_uuid, src_ts, tools
        return None
    # isMeta marks host-injected pseudo-turns (skill bodies, local-command
    # caveats); they are never dialogue whatever their text says.
    if obj.get("isMeta") is True or _is_noise(text):
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
        if n and n not in seen:
            seen.append(n)
    if not seen:
        return ""
    extra = f" +{len(seen) - 8}" if len(seen) > 8 else ""
    return "\n[tools: " + ", ".join(seen[:8]) + extra + "]"

# Immediate stdin-prompt capture, keyed by prompt_id: stamped as
# source_uuid=prompt_id so the later transcript walk (role:user keyed by
# promptId above) collapses onto the same idem tag instead of
# double-inserting. Runs regardless of whether a transcript is resolvable
# yet — it needs none. Wrapped so a bad prompt can never abort the offset
# publish below (capture is best-effort by contract).
try:
    _raw_prompt_id = sys.argv[4] if len(sys.argv) > 4 else ""
    if _raw_prompt_id and re.fullmatch(r"[A-Za-z0-9._-]{1,128}", _raw_prompt_id):
        _immediate_prompt_id = _raw_prompt_id
    else:
        _immediate_prompt_id = ""
    _immediate_prompt_text = ""
    if _immediate_prompt_id and len(sys.argv) > 3 and sys.argv[3]:
        try:
            _payload_obj = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
            if isinstance(_payload_obj, dict):
                _immediate_prompt_text = str(_payload_obj.get("prompt") or "")
        except Exception:
            _immediate_prompt_text = ""
    if (
        _immediate_prompt_id
        and _immediate_prompt_text
        and not os.environ.get("IAI_MCP_IMMEDIATE_PROMPT_DISABLED")
        and not _immediate_prompt_text.lstrip().startswith("/")
        and not _is_noise(_immediate_prompt_text)
        and len(_immediate_prompt_text) >= 12
    ):
        _need_header_immediate = (not live.exists()) or live.stat().st_size == 0
        with live.open("a") as _out_immediate:
            if _need_header_immediate:
                _header_immediate = {
                    "version": 1,
                    "deferred_at": datetime.now(timezone.utc).isoformat(),
                    "session_id": session_id,
                    "cwd": cwd,
                }
                _out_immediate.write(json.dumps(_header_immediate, ensure_ascii=False) + "\n")
            _event_immediate = {
                "text": _immediate_prompt_text,
                "cue": f"session {session_id} turn",
                "tier": "episodic",
                "role": "user",
                "ts": datetime.now(timezone.utc).isoformat(),
                "source_uuid": _immediate_prompt_id,
            }
            _out_immediate.write(json.dumps(_event_immediate, ensure_ascii=False) + "\n")
except Exception:
    pass

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

# Snapshot first, legacy offset second: a crash between them leaves the
# legacy offset older, and the snapshot (>=) stays authoritative.
new_offset = prev + consumed
tmp_s = state_dir / f"{session_id}.turnstate.json.tmp{os.getpid()}"
with open(tmp_s, "w") as _f:
    _f.write(json.dumps(
        {"offset": new_offset, "pending": pending_tools[:256], "fp": stream_fp}
    ))
    _f.flush()
    os.fsync(_f.fileno())
os.replace(tmp_s, turnstate)

# A pre-snapshot reader pairs the mirrored offset with the split pending
# file and can fabricate a trailer from it — remove it BEFORE the offset
# publish, so a reader that sees the fresh offset can never find it.
try:
    (state_dir / f"{session_id}.pending-tools").unlink()
except OSError:
    pass

tmp = state_dir / f"{session_id}.offset.tmp{os.getpid()}"
with open(tmp, "w") as _f:
    _f.write(str(new_offset))
    _f.flush()
    os.fsync(_f.fileno())
os.replace(tmp, offset)
# Rename durability needs the directory entry flushed too.
_dfd = os.open(str(state_dir), os.O_RDONLY)
try:
    os.fsync(_dfd)
finally:
    os.close(_dfd)
# Capture state is published — the freshness gate below must not run
# inside the capture critical section.
os.close(_lock_fd)

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
        tmp_wm = d / f"{sid}.watermark.tmp{os.getpid()}"
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
        tmp_fp = d / f"{sid}.live-fingerprint.tmp{os.getpid()}"
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
                                new_max_ts = result_obj.get("new_max_ts") or ""
                                # Sidecars advance ONLY on an explicit caught_up
                                # reply: the daemon promoted everything pending
                                # and did not suppress a delta. A missing flag
                                # (older daemon), a debounced reply or a partial
                                # drain keeps them, or the still-pending turns
                                # would fall below the watermark for good.
                                caught_up = result_obj.get("caught_up") is True
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
                                if caught_up and new_max_ts:
                                    if _gate_utc_iso(new_max_ts) > _gate_utc_iso(wm):
                                        _gate_write_watermark(session_id, new_max_ts)
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
  timeout 5 /usr/bin/python3 -c "$PY_SCRIPT" "$session_id" "$transcript_path" "$_payload_tmp" "$prompt_id" 2>/dev/null
elif command -v gtimeout >/dev/null 2>&1; then
  gtimeout 5 /usr/bin/python3 -c "$PY_SCRIPT" "$session_id" "$transcript_path" "$_payload_tmp" "$prompt_id" 2>/dev/null
else
  /usr/bin/python3 -c "$PY_SCRIPT" "$session_id" "$transcript_path" "$_payload_tmp" "$prompt_id" 2>/dev/null
fi
rc=$?
# _payload_tmp / _extract_tmp cleanup runs via the EXIT trap above.

echo "$ts session=$session_id rc=$rc channel=$channel" >> "$log" 2>/dev/null

exit 0
