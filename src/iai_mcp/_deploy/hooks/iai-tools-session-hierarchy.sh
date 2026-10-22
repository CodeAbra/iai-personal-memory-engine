#!/usr/bin/env bash
# SessionStart hook -- injects the iai-tools tool-selection hierarchy into
# the main session's additionalContext once per lifecycle event (startup,
# resume, clear, compact).
#
# Sibling to iai-tools-session-index.sh, which stays side-effect-only (no
# stdout) -- this hook is the dedicated context injector instead.
#
# Fail-open by design: missing jq or unparseable stdin exits 0 with no
# output so a broken hook never blocks session start.

input=$(cat 2>/dev/null || true)

command -v jq >/dev/null 2>&1 || exit 0
printf '%s' "$input" | jq -e . >/dev/null 2>&1 || exit 0

block='Tool selection for this session: prefer mcp__iai-tools__code_search over Grep/grep, mcp__iai-tools__execute or mcp__iai-tools__batch_execute for command output (raw bytes stay in the sandbox), mcp__iai-tools__session_search for past conversations, mcp__iai-tools__stats for usage insight. Read is fine for a single known file you are about to edit.'

out=$(jq -n --arg block "$block" '{
  hookSpecificOutput: {
    hookEventName: "SessionStart",
    additionalContext: $block
  }
}' 2>/dev/null)

[ -n "$out" ] && printf '%s' "$out"
exit 0
