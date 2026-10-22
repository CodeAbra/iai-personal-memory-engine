#!/usr/bin/env bash
# PreToolUse hook -- injects the iai-tools tool-selection hierarchy into a
# spawned Agent/Task subagent's prompt-shaped input field.
#
# Field detection scans an ORDERED candidate-key list -- prompt, request,
# objective, question, query, task -- and appends to the FIRST key the
# payload actually has. Never fabricates a key the payload lacks.
#
# Fail-open by design: no candidate key present, a non-string matched
# value, an already-injected value, malformed stdin, or missing jq all
# exit 0 with no output, leaving the spawn's tool_input untouched.

input=$(cat 2>/dev/null || true)

command -v jq >/dev/null 2>&1 || exit 0

block='<iai_tools_routing>First call ToolSearch(query: "select:mcp__iai-tools__code_search,mcp__iai-tools__execute,mcp__iai-tools__batch_execute,mcp__iai-tools__session_search,mcp__iai-tools__stats") to load these deferred tools, then use them: code_search instead of grep/Grep, execute/batch_execute for command output so raw bytes stay in the sandbox, Read only for a single known file you are about to edit.</iai_tools_routing>'

key=""
for candidate in prompt request objective question query task; do
  has=$(printf '%s' "$input" | jq -r --arg k "$candidate" \
    'if has("tool_input") and (.tool_input | has($k)) then "yes" else "no" end' 2>/dev/null)
  if [ "$has" = "yes" ]; then
    key="$candidate"
    break
  fi
done
[ -n "$key" ] || exit 0

# has() is presence, not truthiness -- a present-but-null or non-string
# value must fail the guard explicitly rather than degrade on string concat.
is_string=$(printf '%s' "$input" | jq -r --arg k "$key" \
  '(.tool_input[$k] | type) == "string"' 2>/dev/null)
[ "$is_string" = "true" ] || exit 0

already=$(printf '%s' "$input" | jq -r --arg k "$key" \
  '.tool_input[$k] | contains("<iai_tools_routing>")' 2>/dev/null)
[ "$already" = "true" ] && exit 0

out=$(printf '%s' "$input" | jq --arg k "$key" --arg block "$block" '
  .tool_input[$k] as $orig
  | {
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "allow",
        updatedInput: (.tool_input + {($k): ($orig + "\n\n" + $block)})
      }
    }
' 2>/dev/null)
[ -n "$out" ] && printf '%s' "$out"
exit 0
