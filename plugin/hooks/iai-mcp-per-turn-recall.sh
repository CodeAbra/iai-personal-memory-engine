#!/bin/bash
# Per-turn context injection for Claude Code (UserPromptSubmit hook).
#
# Daemon-independent by construction: reads ONLY daemon-emitted cache files
# (the working-tier snapshot). No socket round-trip, no iai_mcp import — an
# absent or sleeping daemon costs nothing and blocks nothing. A stale
# snapshot (older than the freshness window) is ignored so a dead daemon can
# never inject yesterday's task as the active one — with one exception: the
# live-state emitter (goal/next-action only) ignores this freshness window
# by design, so session continuity survives a daemon restart even when the
# snapshot is stale.
#
# Live accelerator: IAI_MCP_PER_TURN_SOCKET_ACCEL additionally attempts a
# live memory_recall over the daemon socket with a hard sub-second timeout,
# via python3 stdlib only (still no iai_mcp import). Default ON — measured
# warm round-trip overhead sits well under the 0.8s socket timeout, so the
# current-turn cue takes priority over the lagged cache pack. Set to "0" to
# opt back out; the cache path alone still honors the awake-memory invariant
# when the accelerator is off or the daemon socket is absent.
# IAI_MCP_RECALL_SOCKET_TIMEOUT overrides the 0.8s socket timeout (float
# seconds); unset or unparseable falls back to 0.8.
#
# Always exits 0: context injection is best-effort, never a turn blocker.

set -u

# IAI_MCP_STORE is the canonical store-root variable; IAI_MCP_ROOT is kept
# as a legacy fallback for environments installed before the rename.
IAI_ROOT="${IAI_MCP_STORE:-${IAI_MCP_ROOT:-$HOME/.iai-mcp}}"
CHANNEL="settings"
[ -n "${CLAUDE_PLUGIN_ROOT:-}" ] && CHANNEL="plugin"
FRESH_SEC="${IAI_MCP_WORKING_TIER_FRESH_SEC:-7200}"
PACK_FRESH_SEC="${IAI_MCP_FORESIGHT_FRESH_SEC:-2700}"
# Mirrors daemon_state.RUNNING_AGENT_TTL_HOURS (6h = 21600s); hardcoded,
# never shelled out to python -- keep this value in sync by hand.
RUNNING_AGENT_TTL_SEC="${IAI_MCP_RUNNING_AGENT_TTL_SEC:-21600}"
# Same-shell gate: set by emit_working_tier only on the path where it
# actually emits a block, read by emit_live_state to suppress a duplicate.
# Explicit local init (never an inherited/exported value) under set -u.
_WORKING_TIER_EMITTED=""

json_field() {
    printf '%s' "$2" | sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1
}

session_scope_blocks() {
    # Shared session-scope comparison for a shared cache/state-file pair.
    # Blocks (returns 0) ONLY when both sides are known and differ; unknown
    # on either side -- no incoming id, no state file, no recorded id, or a
    # recorded "-" (this codebase's own sentinel for "no known session id")
    # -- fails open (returns 1). Used by every emitter reading a file
    # another session may have written last.
    local state_file="$1"
    local incoming_sid="$2"
    local recorded_sid
    [ -n "$incoming_sid" ] || return 1
    [ -f "$state_file" ] || return 1
    recorded_sid=$(json_field session_id "$(head -c 4096 "$state_file")")
    [ -n "$recorded_sid" ] && [ "$recorded_sid" != "-" ] || return 1
    [ "$recorded_sid" != "$incoming_sid" ] || return 1
    return 0
}

_sid_safe_bash() {
    printf '%s' "$1" | tr -cd 'A-Za-z0-9_-' | cut -c1-64
}

clear_continuation_marker_fresh() {
    # Proves the incoming sid was born via /clear, not WHICH window it
    # continues -- rare concurrent-window cross-focus is an accepted gap.
    local sid_safe marker age
    sid_safe=$(_sid_safe_bash "$SESS_IN")
    [ -n "$sid_safe" ] || return 1
    marker="$IAI_ROOT/.session-clear-continuation.$sid_safe"
    [ -f "$marker" ] || return 1
    age=$(file_age "$marker")
    case "$age" in
        ''|*[!0-9-]*) return 1 ;;
    esac
    [ "$age" -ge 0 ] && [ "$age" -le "$RUNNING_AGENT_TTL_SEC" ]
}

# At most two python3 spawns per run (this preamble + emit_socket_recall) --
# never add a third.
#
# Malformed/duplicate-key stdin JSON -> session_id stays empty (fail open),
# never a fallback to the greedy sed extraction below, which a duplicate-key
# payload could steer.
#
# Every session-derived cache/state file is opened O_NOFOLLOW with a
# post-open fstat regular-file re-check, then copied into SAFE_DIR (mode
# 0700, removed on exit, original mtime preserved). Emitters read only the
# shadow copy, never the original path -- a rejected/failed read just leaves
# that shadow absent, so its emitter degrades silently and the turn still
# renders.
_SAFE_DIR=$(mktemp -d 2>/dev/null) || _SAFE_DIR=""
[ -n "$_SAFE_DIR" ] && trap 'rm -rf "$_SAFE_DIR"' EXIT

_STDIN_PREAMBLE=$(cat <<'PYEOF'
import base64, fcntl, json, os, re, select, stat, sys, time


def _read_stdin_bounded(limit, seconds):
    deadline = time.monotonic() + seconds
    buf = bytearray()
    while len(buf) < limit:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([0], [], [], remaining)[0]:
            break
        chunk = os.read(0, min(8192, limit - len(buf)))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def _unique_pairs(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError("duplicate key")
        seen[key] = value
    return seen


raw = _read_stdin_bounded(65536, 1.0)
session_id = ""
try:
    payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs)
    if isinstance(payload, dict):
        candidate = payload.get("session_id", "")
        if isinstance(candidate, str) and not any(
            ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in candidate
        ):
            session_id = candidate
except (UnicodeDecodeError, ValueError, RecursionError):
    session_id = ""

IAI_ROOT = os.environ["IAI_ROOT"]
sock = os.environ.get("IAI_DAEMON_SOCKET_PATH") or (IAI_ROOT + "/.daemon.sock")

_SID_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
)


def _sanitize_sid(value):
    return "".join(ch for ch in value if ch in _SID_CHARS)[:64]


def _read_regular(path, limit):
    # O_NOFOLLOW rejects a symlinked final path component at open time; the
    # post-open fstat S_ISREG re-check rejects any other non-regular type
    # (fifo, device) that O_NOFOLLOW alone does not cover. O_NOFOLLOW
    # unavailable on this platform -> fail closed on this one guard only.
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None or not path:
        return b"", 0
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | nofollow)
    except OSError:
        return b"", 0
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return b"", 0
        data = bytearray()
        while len(data) < limit:
            chunk = os.read(fd, min(8192, limit - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        return bytes(data), int(info.st_mtime) or 1
    except OSError:
        return b"", 0
    finally:
        os.close(fd)


def _write_shadow(safe_dir, name, data, mtime):
    if not safe_dir or mtime <= 0:
        return
    path = os.path.join(safe_dir, name)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.utime(path, (mtime, mtime))
    except OSError:
        pass


def _extract_session_id_field(data):
    matches = re.findall(rb'"session_id"\s*:\s*"([^"]*)"', data)
    return matches[-1].decode("utf-8", errors="ignore") if matches else ""


def _int_env(name, default):
    try:
        return int(os.environ.get(name) or default)
    except ValueError:
        return default


_LEDGER_LOCK_BUDGET_SEC = 0.5


def _write_ledger(root, byte_count, channel, deadline):
    # Best-effort telemetry: LOCK_NB retried to a wall-clock deadline, never
    # a blocking LOCK_EX -- a contended/slow lock must skip this write, not
    # block the turn.
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        return
    try:
        log_dir = os.path.join(root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        fd = os.open(
            os.path.join(log_dir, "foresight-served.jsonl"),
            os.O_RDWR | os.O_CREAT | os.O_NONBLOCK | nofollow, 0o600,
        )
    except OSError:
        return
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    return
                time.sleep(0.01)
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            return
        max_bytes = 524288
        if info.st_size > max_bytes:
            return
        data = bytearray()
        while len(data) <= max_bytes:
            chunk = os.read(fd, min(8192, max_bytes + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > max_bytes:
            return
        record = (json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bytes": byte_count, "channel": channel,
        }, separators=(",", ":")) + "\n").encode("ascii")
        lines = bytes(data).splitlines(keepends=True)
        lines.append(record)
        if len(lines) > 4000:
            output = b"".join(lines[-2000:])
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
        else:
            output = record
            os.lseek(fd, 0, os.SEEK_END)
        remaining = memoryview(output)
        while remaining:
            written = os.write(fd, remaining)
            if written <= 0:
                break
            remaining = remaining[written:]
    except OSError:
        pass
    finally:
        os.close(fd)


safe_dir = os.environ.get("SAFE_DIR", "")
sid = _sanitize_sid(session_id) if session_id else ""

wt_override = os.environ.get("IAI_MCP_WORKING_TIER_CACHE", "")
if wt_override:
    wt_path = wt_override
elif sid:
    wt_path = os.path.join(IAI_ROOT, ".working-tier." + sid + ".cached.md")
else:
    wt_path = ""
wt_data, wt_mtime = _read_regular(wt_path, 65536)
_write_shadow(safe_dir, "wt.md", wt_data, wt_mtime)

fs_override = os.environ.get("IAI_MCP_FORESIGHT_PACK", "")
if fs_override:
    fs_path = fs_override
elif sid:
    candidate = os.path.join(IAI_ROOT, ".next-turn-pack." + sid + ".cached.md")
    fs_path = candidate if os.path.exists(candidate) else os.path.join(
        IAI_ROOT, ".next-turn-pack.cached.md"
    )
else:
    fs_path = os.path.join(IAI_ROOT, ".next-turn-pack.cached.md")
fs_state_path = (
    fs_path[: -len(".cached.md")] if fs_path.endswith(".cached.md") else fs_path
) + ".state.json"
fs_data, fs_mtime = _read_regular(fs_path, 6144)
fss_data, fss_mtime = _read_regular(fs_state_path, 4096)
_write_shadow(safe_dir, "fs.md", fs_data, fs_mtime)
_write_shadow(safe_dir, "fss.json", fss_data, fss_mtime)

cc_path = os.path.join(IAI_ROOT, ".session-continuity.cached.md")
cs_path = cc_path[: -len(".cached.md")] + ".state.json"
cc_data, cc_mtime = _read_regular(cc_path, 65536)
cs_data, cs_mtime = _read_regular(cs_path, 4096)
_write_shadow(safe_dir, "cc.md", cc_data, cc_mtime)
_write_shadow(safe_dir, "cs.json", cs_data, cs_mtime)

# Best-effort accounting, mirrored against the same freshness/session check
# bash applies to this shadow below.
if fs_mtime > 0:
    now = int(time.time())
    fresh_sec = _int_env("IAI_MCP_FORESIGHT_FRESH_SEC", 2700)
    recorded_sid = _extract_session_id_field(fss_data) if fss_data else ""
    blocked = bool(session_id) and bool(recorded_sid) and recorded_sid != "-" and recorded_sid != session_id
    if (now - fs_mtime) <= fresh_sec and not blocked:
        channel = "plugin" if os.environ.get("CLAUDE_PLUGIN_ROOT") else "settings"
        _write_ledger(
            IAI_ROOT, len(fs_data), channel,
            time.monotonic() + _LEDGER_LOCK_BUDGET_SEC,
        )

out = sys.stdout.buffer
out.write(base64.b64encode(sock.encode("utf-8")) + b"\n")
out.write(session_id.encode("utf-8") + b"\n")
out.write(raw)
PYEOF
)

STDIN_JSON=""
SESS_IN=""
SOCK_SELECTED=""
if command -v python3 >/dev/null 2>&1; then
    _SOCK_B64=""
    {
        IFS= read -r _SOCK_B64
        IFS= read -r SESS_IN
        STDIN_JSON=$(cat)
    } < <(IAI_ROOT="$IAI_ROOT" SAFE_DIR="$_SAFE_DIR" python3 -c "$_STDIN_PREAMBLE" 2>/dev/null)
    SOCK_SELECTED=$(printf '%s' "$_SOCK_B64" | base64 -d 2>/dev/null)
fi
[ -n "$SOCK_SELECTED" ] || SOCK_SELECTED="${IAI_DAEMON_SOCKET_PATH:-$IAI_ROOT/.daemon.sock}"

WT_SHADOW=""
PACK=""
PACK_STATE=""
CONTINUITY_CACHE=""
CONTINUITY_STATE=""
if [ -n "$_SAFE_DIR" ]; then
    WT_SHADOW="$_SAFE_DIR/wt.md"
    PACK="$_SAFE_DIR/fs.md"
    PACK_STATE="$_SAFE_DIR/fss.json"
    CONTINUITY_CACHE="$_SAFE_DIR/cc.md"
    CONTINUITY_STATE="$_SAFE_DIR/cs.json"
fi

file_age() {
    case "$(uname)" in
        Darwin) m=$(stat -f %m "$1" 2>/dev/null || echo 0) ;;
        *)      m=$(stat -c %Y "$1" 2>/dev/null || echo 0) ;;
    esac
    echo $(( $(date +%s) - m ))
}

emit_foresight() {
    [ -f "$PACK" ] || return 0
    [ "$(file_age "$PACK")" -le "$PACK_FRESH_SEC" ] || return 0
    # Session scope: a pack anticipated for one conversation must not leak
    # into another running in parallel. Unknown on either side -> fail open.
    session_scope_blocks "$PACK_STATE" "$SESS_IN" && return 0
    echo "<iai-mcp-foresight>"
    head -c 6144 "$PACK"
    echo "</iai-mcp-foresight>"
}

emit_working_tier() {
    # Session scope: the snapshot layout is per-session; the shadow was
    # resolved from ONLY this session's cache (or the explicit env override),
    # so another conversation's task can never be injected here.
    [ -f "$WT_SHADOW" ] || return 0
    [ "$(file_age "$WT_SHADOW")" -le "$FRESH_SEC" ] || return 0
    echo "<iai-mcp-working-tier>"
    head -c 4096 "$WT_SHADOW"
    echo "</iai-mcp-working-tier>"
    _WORKING_TIER_EMITTED=1
}

emit_live_state() {
    # Suppressed once emit_working_tier already carried the same goal/
    # next-action lines verbatim in this same turn's output.
    [ -z "$_WORKING_TIER_EMITTED" ] || return 0
    # Same per-session shadow emit_working_tier reads, but temporal — no
    # freshness/mtime gate, so this reflects the last-known live state
    # regardless of cache age WHENEVER a real next_action has been folded.
    # Silent when next_action is still the "(none)" placeholder — avoids a
    # zero-information block on every turn nothing has been folded yet.
    # Extraction is by line PREFIX (goal: / next action:), never fixed
    # position, so an added snapshot section can never silently break it.
    [ -f "$WT_SHADOW" ] || return 0
    next_val=$(sed -n 's/^next action:[[:space:]]*//p' "$WT_SHADOW" | head -1)
    [ -n "$next_val" ] || return 0
    [ "$next_val" != "(none)" ] || return 0
    block=$(sed -n -e '/^goal:/p' -e '/^next action:/p' "$WT_SHADOW")
    [ -n "$block" ] || return 0
    echo "<iai-mcp-live-state>"
    printf '%s\n' "$block" | head -c 4096
    echo "</iai-mcp-live-state>"
}

emit_agent_registry() {
    # Session-agnostic: the eager continuity file carries the pending
    # running-agent registry under a fixed name, no session id in the path,
    # so a /clear that mints a new session id still reconstructs the
    # pending agent on the very next turn. Whole-file mtime bound: older
    # than RUNNING_AGENT_TTL_SEC emits nothing (an abandoned agent with no
    # subsequent write must not surface forever).
    [ -f "$CONTINUITY_CACHE" ] || return 0
    [ "$(file_age "$CONTINUITY_CACHE")" -le "$RUNNING_AGENT_TTL_SEC" ] || return 0
    block=$(sed -n '/<iai-mcp-agent-registry>/,/<\/iai-mcp-agent-registry>/p' "$CONTINUITY_CACHE" | sed '1d;$d')
    [ -n "$block" ] || return 0
    echo "<iai-mcp-agent-registry>"
    printf '%s\n' "$block" | head -c 4096
    echo "</iai-mcp-agent-registry>"
}

emit_live_state_fallback() {
    # A fresh /clear-continuation marker for the incoming sid overrides the
    # double-fire gate too -- turn 2's thin scoped snapshot already set
    # _WORKING_TIER_EMITTED, but its next_action is still "(none)", so the
    # thin-snapshot check just below must stay reachable.
    if [ -n "$_WORKING_TIER_EMITTED" ]; then
        clear_continuation_marker_fresh || return 0
    fi
    # Fires only when this session's own scoped snapshot has no real
    # next_action yet (mirrors emit_live_state's "(none)" check) -- not
    # merely absent, since a fresh /clear entry exists with next_action
    # still "(none)". At most one <iai-mcp-live-state> block fires across
    # emit_live_state and this fallback.
    if [ -f "$WT_SHADOW" ]; then
        scoped_next=$(sed -n 's/^next action:[[:space:]]*//p' "$WT_SHADOW" | head -1)
        [ -n "$scoped_next" ] && [ "$scoped_next" != "(none)" ] && return 0
    fi

    [ -f "$CONTINUITY_CACHE" ] || return 0
    [ "$(file_age "$CONTINUITY_CACHE")" -le "$RUNNING_AGENT_TTL_SEC" ] || return 0
    # Session scope: the shared eager file's live-state block was last
    # written by whichever session folded most recently -- a different
    # session reading it here must never surface that other session's goal.
    # Unknown on either side -> fail open (old daemon, absent sidecar, fresh
    # session with no prior writer). A fresh /clear-continuation marker for
    # the incoming sid overrides a proven block -- the one external signal a
    # concurrent second window can never produce for its own new session id.
    if session_scope_blocks "$CONTINUITY_STATE" "$SESS_IN"; then
        clear_continuation_marker_fresh || return 0
    fi
    block=$(sed -n '/<iai-mcp-live-state>/,/<\/iai-mcp-live-state>/p' "$CONTINUITY_CACHE" | sed '1d;$d')
    [ -n "$block" ] || return 0
    echo "<iai-mcp-live-state>"
    printf '%s\n' "$block" | head -c 4096
    echo "</iai-mcp-live-state>"
}

emit_directives() {
    # Global, no session gate, no freshness gate: unlike the emitters above,
    # this must inject regardless of session id or cache age.
    [ "${IAI_MCP_DIRECTIVES_OFF:-}" != "1" ] || return 0
    cache="$IAI_ROOT/.directives.cached.md"
    [ -f "$cache" ] || return 0
    echo "<iai-mcp-directives>"
    head -c 4096 "$cache"
    echo "</iai-mcp-directives>"
}

emit_socket_recall() {
    [ "${IAI_MCP_PER_TURN_SOCKET_ACCEL:-1}" = "1" ] || return 0
    sock="$SOCK_SELECTED"
    [ -S "$sock" ] || return 0
    command -v python3 >/dev/null 2>&1 || return 0
    # _safe_socket_path re-verifies ownership at connect time -- the bash -S
    # test above is advisory only, a stat race away from stale. One
    # wall-clock deadline spans connect+send+recv. HOOK_DIR points the child
    # at _recall_render.py, deployed next to this script in lockstep by the
    # capture-hooks installer.
    HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
    SOCK="$sock" HOOK_STDIN="$STDIN_JSON" HOOK_DIR="$HOOK_DIR" SOCK_TIMEOUT="${IAI_MCP_RECALL_SOCKET_TIMEOUT:-}" python3 - <<'PYEOF' 2>/dev/null || true
import json, os, signal, socket, stat, sys, time


def _remaining(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError
    return remaining


def _safe_socket_path(path):
    # Reject a socket (or parent dir) owned by another account, or one
    # either is group/other-writable -- planted by a sibling local uid.
    try:
        socket_stat = os.stat(path)
        parent_stat = os.stat(os.path.dirname(path) or ".")
    except OSError:
        return False
    uid = os.geteuid()
    return (
        stat.S_ISSOCK(socket_stat.st_mode)
        and socket_stat.st_uid == uid
        and parent_stat.st_uid == uid
        and not socket_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        and not parent_stat.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _recall_once(sock_path, cue, deadline):
    if not _safe_socket_path(sock_path):
        return None
    req = {"jsonrpc": "2.0", "id": 1, "method": "memory_recall",
           "params": {"cue": cue, "limit": 3}}
    payload = (json.dumps(req) + "\n").encode()
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(_remaining(deadline))
        conn.connect(sock_path)
        conn.settimeout(_remaining(deadline))
        conn.sendall(payload)
        buf = bytearray()
        while not buf.endswith(b"\n") and len(buf) < 65536:
            conn.settimeout(_remaining(deadline))
            chunk = conn.recv(min(8192, 65536 - len(buf)))
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        conn.close()
    reply = json.loads(buf)
    return reply.get("result") if isinstance(reply, dict) else None


def _render_bounded(result, hook_dir):
    # A slow/huge renderer cannot block the turn -- SIGALRM preempts it
    # regardless of what it is doing (import, regex, string building).
    def _on_alarm(signum, frame):
        raise TimeoutError

    previous_handler = signal.signal(signal.SIGALRM, _on_alarm)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 1.0)
    started = time.monotonic()
    try:
        if hook_dir and hook_dir not in sys.path:
            sys.path.insert(0, hook_dir)
        from _recall_render import render_recall_block
        return render_recall_block(result)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        delay, interval = previous_timer
        if delay:
            delay = max(0.000001, delay - (time.monotonic() - started))
        signal.setitimer(signal.ITIMER_REAL, delay, interval)


sock_path = os.environ["SOCK"]
try:
    raw = os.environ.get("HOOK_STDIN", "")[:65536]
    try:
        payload = json.loads(raw)
        prompt = payload.get("prompt") if isinstance(payload, dict) else None
        cue = prompt[:512] if isinstance(prompt, str) else ""
    except ValueError:
        cue = raw[:512].strip()
    if not cue:
        raise SystemExit(0)
    try:
        sock_timeout = float(os.environ.get("SOCK_TIMEOUT") or 0.8)
    except (TypeError, ValueError):
        sock_timeout = 0.8
    deadline = time.monotonic() + sock_timeout
    result = _recall_once(sock_path, cue, deadline)
    block = _render_bounded(result, os.environ.get("HOOK_DIR", ""))
    if block:
        print(block)
except Exception:
    pass
PYEOF
}

emit_foresight
emit_working_tier
emit_live_state
emit_live_state_fallback
emit_agent_registry
emit_directives
emit_socket_recall
exit 0
