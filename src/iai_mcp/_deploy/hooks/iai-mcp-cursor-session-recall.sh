#!/usr/bin/env bash
# Cursor sessionStart adapter: run the core recall script on the untouched
# stdin (payload already carries session_id) and wrap its markdown into the
# {"additional_context": ...} envelope Cursor injects. Fail-safe: any error
# degrades to empty stdout + exit 0 — a hook must never break a session.
set -u
input=$(cat 2>/dev/null || true)
core="$(cd "$(dirname "$0")" 2>/dev/null && pwd)/iai-mcp-session-recall.sh"
[ -f "$core" ] || exit 0
out=$(printf '%s' "$input" | bash "$core" 2>/dev/null || true)
[ -n "$out" ] || exit 0
printf '%s' "$out" | /usr/bin/python3 -c '
import json, sys
text = sys.stdin.read()
if text.strip():
    print(json.dumps({"additional_context": text}))
' 2>/dev/null || true
exit 0
