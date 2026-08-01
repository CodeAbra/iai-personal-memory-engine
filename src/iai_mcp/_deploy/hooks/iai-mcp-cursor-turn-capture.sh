#!/usr/bin/env bash
# Cursor beforeSubmitPrompt adapter: this event cannot inject context, so
# run the core turn capture with the injection gate disabled (no socket RPC
# spent) and swallow stdout — Cursor must never see a foreign envelope.
# Fail-safe: exit 0 always.
set -u
here="$(cd "$(dirname "$0")" 2>/dev/null && pwd)"
core="$here/iai-mcp-turn-capture.sh"
if [ -z "$here" ] || [ ! -f "$core" ]; then
  cat >/dev/null 2>&1 || true
  exit 0
fi
IAI_MCP_TURN_INJECT_DISABLED=1 bash "$core" >/dev/null 2>&1 || true
exit 0
