#!/usr/bin/env bash
# Antigravity PreInvocation adapter. Payload key casing is unstable across
# builds (camelCase per official docs, snake_case seen in the wild) — accept
# both. First invocation gets the session-start recall as a durable
# userMessage; later invocations get the per-turn slice as an
# ephemeralMessage. Fail-safe: empty stdout + exit 0 on any error.
set -u
input=$(cat 2>/dev/null || true)
here="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
[ -n "$here" ] || exit 0
PY_SCRIPT='
import json, subprocess, sys
from pathlib import Path

here = sys.argv[1]
try:
    payload = json.loads(sys.stdin.read() or "{}")
except Exception:
    sys.exit(0)
if not isinstance(payload, dict):
    sys.exit(0)

def pick(*keys):
    for k in keys:
        v = payload.get(k)
        if v not in (None, ""):
            return v
    return ""

session_id = str(pick("conversationId", "conversation_id", "session_id") or "")
invocation = pick("invocationNum", "invocation_num", "invocationNumber")

# "First" must fail toward the ephemeral slice, never toward re-injecting
# the durable brief every turn: the counter is a hint, the per-conversation
# marker file is the authority when the counter is absent or unparseable.
marker = None
if session_id:
    state_dir = Path.home() / ".iai-mcp" / ".capture-state"
    marker = state_dir / f"antigravity-{session_id}.recalled"
try:
    first = int(invocation) == 1
except (TypeError, ValueError):
    first = marker is not None and not marker.exists()
if marker is not None:
    try:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
    except OSError:
        pass

core = f"{here}/iai-mcp-session-recall.sh" if first else f"{here}/iai-mcp-per-turn-recall.sh"
core_stdin = json.dumps({
    "session_id": session_id or "-",
    "source": "startup",
    "cwd": str(pick("cwd") or ""),
    "transcript_path": "",
})
try:
    proc = subprocess.run(
        ["bash", core], input=core_stdin, capture_output=True,
        text=True, timeout=25,
    )
    text = proc.stdout if proc.returncode == 0 else ""
except Exception:
    text = ""
if text.strip():
    key = "userMessage" if first else "ephemeralMessage"
    print(json.dumps({"injectSteps": [{key: text}]}))
'
printf '%s' "$input" | /usr/bin/python3 -c "$PY_SCRIPT" "$here" 2>/dev/null || true
exit 0
