#!/bin/bash
# --- iai-pme: portable Python interpreter resolution (Windows/macOS/Linux) ---
# Honors IAI_MCP_PYTHON when set by the installer; otherwise resolves a real
# python3/python, skipping the Windows Store "python3" App-Execution-Alias stub
# (WindowsApps), and finally falls back to the Windows "py" launcher. The result
# is an absolute interpreter path usable as "$PYBIN" -c '...'.
PYBIN="${IAI_MCP_PYTHON:-}"
if [ -z "$PYBIN" ]; then
  for _c in python3 python; do
    _p=$(command -v "$_c" 2>/dev/null) || continue
    case "$_p" in *WindowsApps*) continue ;; esac
    PYBIN="$_p"; break
  done
fi
if [ -z "$PYBIN" ] && command -v py >/dev/null 2>&1; then
  PYBIN=$(py -3 -c "import sys; print(sys.executable)" 2>/dev/null)
fi
[ -n "$PYBIN" ] || PYBIN=/usr/bin/python3
# --- end iai-pme interpreter resolution ---
# Per-turn context injection for Claude Code (UserPromptSubmit hook).
#
# Daemon-independent by construction: reads ONLY daemon-emitted cache files
# (the working-tier snapshot). No socket round-trip, no python interpreter,
# no iai_mcp import — an absent or sleeping daemon costs nothing and blocks
# nothing. A stale snapshot (older than the freshness window) is ignored so
# a dead daemon can never inject yesterday's task as the active one.
#
# Optional accelerator: IAI_MCP_PER_TURN_SOCKET_ACCEL=1 additionally attempts
# a live memory_recall over the daemon socket with a hard sub-second timeout,
# via python3 stdlib only (still no iai_mcp import). Default OFF — the cache
# path alone honors the awake-memory invariant.
#
# Always exits 0: context injection is best-effort, never a turn blocker.

set -u

# IAI_MCP_STORE is the canonical store-root variable; IAI_MCP_ROOT is kept
# as a legacy fallback for environments installed before the rename.
IAI_ROOT="${IAI_MCP_STORE:-${IAI_MCP_ROOT:-$HOME/.iai-mcp}}"
FRESH_SEC="${IAI_MCP_WORKING_TIER_FRESH_SEC:-7200}"
PACK="${IAI_MCP_FORESIGHT_PACK:-$IAI_ROOT/.next-turn-pack.cached.md}"
PACK_STATE="${PACK%.cached.md}.state.json"
PACK_FRESH_SEC="${IAI_MCP_FORESIGHT_FRESH_SEC:-2700}"

# stdin is read once; downstream consumers get it from the variable.
STDIN_JSON=$(head -c 65536)

json_field() {
    printf '%s' "$2" | sed -n 's/.*"'"$1"'"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1
}

# The hook stdin carries the USER PROMPT next to session_id; the greedy sed
# above takes the LAST occurrence on the line, so pasted text containing a
# "session_id" key could redirect pack selection. Parse real JSON for the
# session id; the sed stays as fallback for our own trusted state files.
SESS_IN=$(printf '%s' "$STDIN_JSON" | "$PYBIN" -c '
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
        printf '{"ts":"%s","bytes":%s}\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(wc -c < "$PACK" | tr -d ' ')" \
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
}

emit_socket_recall() {
    [ "${IAI_MCP_PER_TURN_SOCKET_ACCEL:-0}" = "1" ] || return 0
    sock="$IAI_ROOT/.daemon.sock"
    [ -S "$sock" ] || return 0
    command -v python3 >/dev/null 2>&1 || return 0
    # The hook receives {"prompt": ...} JSON on stdin; the cue is extracted in
    # python (stdlib only). `timeout` is optional on macOS — the socket's own
    # sub-second timeouts bound the call either way.
    _runner="python3"
    command -v timeout >/dev/null 2>&1 && _runner="timeout 1 python3"
    SOCK="$sock" HOOK_STDIN="$STDIN_JSON" $_runner - <<'PYEOF' 2>/dev/null || true
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
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(0.8)
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
    hits = (json.loads(buf).get("result") or {}).get("hits") or []
    if hits:
        print("<iai-mcp-recall>")
        for h in hits[:3]:
            text = (h.get("literal_surface") or h.get("text") or "")[:400]
            if text:
                print(f"- {text}")
        print("</iai-mcp-recall>")
except Exception:
    pass
PYEOF
}

emit_foresight
emit_working_tier
emit_socket_recall
exit 0
