#!/bin/bash
# Per-turn context injection for Claude Code (UserPromptSubmit hook).
#
# Daemon-independent by construction: reads ONLY daemon-emitted cache files
# (the working-tier snapshot). No socket round-trip, no python interpreter,
# no iai_mcp import — an absent or sleeping daemon costs nothing and blocks
# nothing. A stale snapshot (older than the freshness window) is ignored so
# a dead daemon can never inject yesterday's task as the active one — with
# one exception: the live-state emitter (goal/next-action only) ignores
# this freshness window by design, so session continuity survives a
# daemon restart even when the snapshot is stale.
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
PACK="${IAI_MCP_FORESIGHT_PACK:-$IAI_ROOT/.next-turn-pack.cached.md}"
PACK_STATE="${PACK%.cached.md}.state.json"
PACK_FRESH_SEC="${IAI_MCP_FORESIGHT_FRESH_SEC:-2700}"
# Mirrors daemon_state.RUNNING_AGENT_TTL_HOURS (6h = 21600s); hardcoded,
# never shelled out to python -- keep this value in sync by hand.
RUNNING_AGENT_TTL_SEC="${IAI_MCP_RUNNING_AGENT_TTL_SEC:-21600}"
CONTINUITY_CACHE="$IAI_ROOT/.session-continuity.cached.md"
# Same-shell gate: set by emit_working_tier only on the path where it
# actually emits a block, read by emit_live_state to suppress a duplicate.
# Explicit local init (never an inherited/exported value) under set -u.
_WORKING_TIER_EMITTED=""

# stdin is read once; downstream consumers get it from the variable.
STDIN_JSON=$(head -c 65536)

json_field() {
    printf '%s' "$2" | sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1
}

# The hook stdin carries the USER PROMPT next to session_id; the greedy sed
# above takes the LAST occurrence on the line, so pasted text containing a
# "session_id" key could redirect pack selection. Parse real JSON for the
# session id; the sed stays as fallback for our own trusted state files.
SESS_IN=$(printf '%s' "$STDIN_JSON" | /usr/bin/python3 -c '
import json, sys
try:
    print(str(json.load(sys.stdin).get("session_id", "") or ""))
except Exception:
    print("")
' 2>/dev/null)
[ -n "$SESS_IN" ] || SESS_IN=$(json_field session_id "$STDIN_JSON")

file_age() {
    case "$(uname)" in
        Darwin) m=$(stat -f %m "$1" 2>/dev/null || echo 0) ;;
        *)      m=$(stat -c %Y "$1" 2>/dev/null || echo 0) ;;
    esac
    echo $(( $(date +%s) - m ))
}

emit_foresight() {
    # This session's own pack wins over the shared global one: parallel
    # sessions each read their own anticipation instead of racing for the
    # last writer's. An explicit env pack bypasses the per-session layout.
    sess_in="$SESS_IN"
    if [ -z "${IAI_MCP_FORESIGHT_PACK:-}" ] && [ -n "$sess_in" ]; then
        sid=$(printf '%s' "$sess_in" | tr -cd 'A-Za-z0-9_-' | cut -c1-64)
        if [ -n "$sid" ] && [ -f "$IAI_ROOT/.next-turn-pack.$sid.cached.md" ]; then
            PACK="$IAI_ROOT/.next-turn-pack.$sid.cached.md"
            PACK_STATE="$IAI_ROOT/.next-turn-pack.$sid.state.json"
        fi
    fi
    [ -f "$PACK" ] || return 0
    [ "$(file_age "$PACK")" -le "$PACK_FRESH_SEC" ] || return 0
    # Session scope: a pack anticipated for one conversation must not leak
    # into another running in parallel. Unknown on either side -> fail open.
    if [ -n "$sess_in" ] && [ -f "$PACK_STATE" ]; then
        sess_pack=$(json_field session_id "$(head -c 4096 "$PACK_STATE")")
        if [ -n "$sess_pack" ] && [ "$sess_pack" != "$sess_in" ]; then
            return 0
        fi
    fi
    echo "<iai-mcp-foresight>"
    head -c 6144 "$PACK"
    echo "</iai-mcp-foresight>"
    # Serve ledger: one line per pack actually delivered to the agent — the
    # numerator of the anticipation economy (searches the agent never made).
    # Rotated in place past 4000 lines (keep the newest 2000) so a long-lived
    # install never grows the file unbounded; readers already frame their
    # totals as a lower-bound estimate.
    {
        mkdir -p "$IAI_ROOT/logs" 2>/dev/null
        LEDGER="$IAI_ROOT/logs/foresight-served.jsonl"
        printf '{"ts":"%s","bytes":%s,"channel":"%s"}\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(wc -c < "$PACK" | tr -d ' ')" "$CHANNEL" \
            >> "$LEDGER"
        if [ "$(wc -l < "$LEDGER" | tr -d ' ')" -gt 4000 ]; then
            tail -n 2000 "$LEDGER" > "$LEDGER.rot.$$" 2>/dev/null \
                && mv "$LEDGER.rot.$$" "$LEDGER"
        fi
    } 2>/dev/null || true
}

emit_working_tier() {
    # Session scope: the snapshot layout is per-session; this session reads
    # ONLY its own file, so another conversation's task can never be injected
    # here. The env override is an explicit single-file setup and bypasses
    # the per-session layout (tests / single-consumer roots).
    if [ -n "${IAI_MCP_WORKING_TIER_CACHE:-}" ]; then
        cache="$IAI_MCP_WORKING_TIER_CACHE"
    else
        sess_in="$SESS_IN"
        [ -n "$sess_in" ] || return 0
        sid=$(printf '%s' "$sess_in" | tr -cd 'A-Za-z0-9_-' | cut -c1-64)
        [ -n "$sid" ] || return 0
        cache="$IAI_ROOT/.working-tier.$sid.cached.md"
    fi
    [ -f "$cache" ] || return 0
    [ "$(file_age "$cache")" -le "$FRESH_SEC" ] || return 0
    echo "<iai-mcp-working-tier>"
    head -c 4096 "$cache"
    echo "</iai-mcp-working-tier>"
    _WORKING_TIER_EMITTED=1
}

emit_live_state() {
    # Suppressed once emit_working_tier already carried the same goal/
    # next-action lines verbatim in this same turn's output.
    [ -z "$_WORKING_TIER_EMITTED" ] || return 0
    # Same per-session cache emit_working_tier reads, but temporal — no
    # freshness/mtime gate, so this reflects the last-known live state
    # regardless of cache age WHENEVER a real next_action has been folded.
    # Silent when next_action is still the "(none)" placeholder — avoids a
    # zero-information block on every turn nothing has been folded yet.
    # Extraction is by line PREFIX (goal: / next action:), never fixed
    # position, so an added snapshot section can never silently break it.
    if [ -n "${IAI_MCP_WORKING_TIER_CACHE:-}" ]; then
        cache="$IAI_MCP_WORKING_TIER_CACHE"
    else
        sess_in="$SESS_IN"
        [ -n "$sess_in" ] || return 0
        sid=$(printf '%s' "$sess_in" | tr -cd 'A-Za-z0-9_-' | cut -c1-64)
        [ -n "$sid" ] || return 0
        cache="$IAI_ROOT/.working-tier.$sid.cached.md"
    fi
    [ -f "$cache" ] || return 0
    next_val=$(sed -n 's/^next action:[[:space:]]*//p' "$cache" | head -1)
    [ -n "$next_val" ] || return 0
    [ "$next_val" != "(none)" ] || return 0
    block=$(sed -n -e '/^goal:/p' -e '/^next action:/p' "$cache")
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
    # Fires whenever the session-scoped working-tier snapshot for this
    # session id has NO real next_action yet -- not merely when the file is
    # absent. A /clear's first captured turn opens a thin fresh entry (its
    # own scoped file exists but next_action is still the "(none)"
    # placeholder), so a presence-only guard would go silent on turn 2
    # without the new session ever having acquired its own state. Mirrors
    # emit_live_state's own "(none)" check, so this fires only when the
    # scoped snapshot has no real next_action yet. At most one
    # <iai-mcp-live-state> block fires across emit_live_state and this
    # fallback, and the count may be zero by design once emit_working_tier
    # already carried the same content. Same path resolution as
    # emit_working_tier/emit_live_state, session-agnostic eager source,
    # whole-file mtime bound (mirrors emit_agent_registry).
    if [ -n "${IAI_MCP_WORKING_TIER_CACHE:-}" ]; then
        scoped="$IAI_MCP_WORKING_TIER_CACHE"
    else
        sess_in="$SESS_IN"
        [ -n "$sess_in" ] || return 0
        sid=$(printf '%s' "$sess_in" | tr -cd 'A-Za-z0-9_-' | cut -c1-64)
        [ -n "$sid" ] || return 0
        scoped="$IAI_ROOT/.working-tier.$sid.cached.md"
    fi
    if [ -f "$scoped" ]; then
        scoped_next=$(sed -n 's/^next action:[[:space:]]*//p' "$scoped" | head -1)
        [ -n "$scoped_next" ] && [ "$scoped_next" != "(none)" ] && return 0
    fi

    [ -f "$CONTINUITY_CACHE" ] || return 0
    [ "$(file_age "$CONTINUITY_CACHE")" -le "$RUNNING_AGENT_TTL_SEC" ] || return 0
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
    sock="$IAI_ROOT/.daemon.sock"
    [ -S "$sock" ] || return 0
    command -v python3 >/dev/null 2>&1 || return 0
    # The hook receives {"prompt": ...} JSON on stdin; the cue is extracted in
    # python (stdlib only). `timeout` is optional on macOS — the socket's own
    # sub-second timeouts bound the call either way. HOOK_DIR points the
    # child at _recall_render.py, deployed next to this script in lockstep
    # by the capture-hooks installer (still no iai_mcp import — the render
    # helper is stdlib-only too).
    HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
    _runner="python3"
    command -v timeout >/dev/null 2>&1 && _runner="timeout 1 python3"
    SOCK="$sock" HOOK_STDIN="$STDIN_JSON" HOOK_DIR="$HOOK_DIR" SOCK_TIMEOUT="${IAI_MCP_RECALL_SOCKET_TIMEOUT:-}" $_runner - <<'PYEOF' 2>/dev/null || true
import json, os, socket, sys
sock_path = os.environ["SOCK"]
try:
    raw = os.environ.get("HOOK_STDIN", "")[:65536]
    try:
        cue = str(json.loads(raw).get("prompt") or "")[:512]
    except (ValueError, AttributeError):
        cue = raw[:512].strip()
    if not cue:
        raise SystemExit(0)
    try:
        sock_timeout = float(os.environ.get("SOCK_TIMEOUT") or 0.8)
    except (TypeError, ValueError):
        sock_timeout = 0.8
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(sock_timeout)
    s.connect(sock_path)
    req = {"jsonrpc": "2.0", "id": 1, "method": "memory_recall",
           "params": {"cue": cue, "limit": 3}}
    s.sendall((json.dumps(req) + "\n").encode())
    buf = b""
    while not buf.endswith(b"\n") and len(buf) < 65536:
        chunk = s.recv(8192)
        if not chunk:
            break
        buf += chunk
    result = json.loads(buf).get("result") or {}
    hook_dir = os.environ.get("HOOK_DIR", "")
    if hook_dir and hook_dir not in sys.path:
        sys.path.insert(0, hook_dir)
    from _recall_render import render_recall_block
    block = render_recall_block(result)
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
