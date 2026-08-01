#!/usr/bin/env bash
# Hermes pre_llm_call adapter: first turn gets the session-start recall,
# later turns the per-turn slice; output rides Hermes {"context": ...}
# envelope, which appends to the user message without touching the system
# prompt. Fail-safe: empty stdout + exit 0 on any error.
set -u
input=$(cat 2>/dev/null || true)
here="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
[ -n "$here" ] || exit 0
PY_SCRIPT='
import json, subprocess, sys

here = sys.argv[1]
try:
    payload = json.loads(sys.stdin.read() or "{}")
except Exception:
    sys.exit(0)
if not isinstance(payload, dict):
    sys.exit(0)

session_id = str(payload.get("session_id") or "-")
first = bool(payload.get("is_first_turn"))
core = f"{here}/iai-mcp-session-recall.sh" if first else f"{here}/iai-mcp-per-turn-recall.sh"
core_stdin = json.dumps({
    "session_id": session_id,
    "source": "startup",
    "cwd": str(payload.get("cwd") or ""),
    "transcript_path": "",
    "prompt": str(payload.get("user_message") or ""),
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
    print(json.dumps({"context": text}))
'
printf '%s' "$input" | /usr/bin/python3 -c "$PY_SCRIPT" "$here" 2>/dev/null || true
exit 0
