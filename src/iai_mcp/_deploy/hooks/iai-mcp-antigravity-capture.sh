#!/usr/bin/env bash
# Antigravity Stop adapter: translate the payload to the core capture
# contract. The short transcript.jsonl truncates fields — when the _full
# sibling exists it MUST be used or verbatim capture silently loses text.
# Fail-safe: empty stdout + exit 0 on any error.
set -u
input=$(cat 2>/dev/null || true)
here="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
[ -n "$here" ] || exit 0
core="$here/iai-mcp-session-capture.sh"
[ -f "$core" ] || exit 0
translated=$(printf '%s' "$input" | /usr/bin/python3 -c '
import json, os, sys
try:
    payload = json.loads(sys.stdin.read() or "{}")
except Exception:
    sys.exit(0)

def pick(*keys):
    for k in keys:
        v = payload.get(k)
        if v not in (None, ""):
            return v
    return ""

session_id = str(pick("conversationId", "conversation_id", "session_id") or "")
transcript = str(pick("transcriptPath", "transcript_path") or "")
if transcript.endswith("transcript.jsonl"):
    full = transcript[: -len("transcript.jsonl")] + "transcript_full.jsonl"
    if os.path.isfile(full):
        transcript = full
print(json.dumps({
    "session_id": session_id,
    "transcript_path": transcript,
    "cwd": str(pick("cwd") or ""),
}))
' 2>/dev/null || true)
[ -n "$translated" ] || exit 0
printf '%s' "$translated" | bash "$core" >/dev/null 2>&1 || true
exit 0
