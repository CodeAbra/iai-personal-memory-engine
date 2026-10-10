#!/usr/bin/env bash
# PreToolUse hook -- ships to every end user via `iai-mcp capture-hooks install`.
# Blocks the native Grep/Glob tools plus Bash search shapes and routes the
# caller to mcp__iai-tools__code_search / mcp__iai-tools__execute. Also
# silently rewrites bare noisy Bash commands (curl, gradle, ...) to run
# through the iai-tools sandbox executor, so raw output stays out of context.
#
# Fail-open by design: any internal error in this script (jq missing,
# malformed payload) must NOT silently block the world -- we exit 0 with
# no output and the tool call proceeds normally.

input=$(cat 2>/dev/null || true)

tool=$(printf '%s' "$input" | jq -r '.tool_name // ""' 2>/dev/null || echo "")

emit_deny() {
  local reason="$1"
  jq -n --arg r "$reason" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "deny",
      permissionDecisionReason: $r
    }
  }' 2>/dev/null
  exit 0
}

# Resolution order for the iai-tools CLI (trusted sources only):
#   1. IAI_TOOLS_RUN_CLI env override (test/dev only -- distinct from the
#      session-index hook's IAI_MCP_TOOLS_INDEX_CLI, a different feature).
#   2. ~/.iai-mcp/.tools-cli-path cache file, written at install time.
#   3. `command -v iai-tools` PATH lookup.
#   4. Baked-in candidate list.
# Unresolvable -> empty string, caller must fail open (no rewrite).
resolve_tools_cli() {
  local cli=""
  if [ -n "${IAI_TOOLS_RUN_CLI:-}" ] && [ -x "${IAI_TOOLS_RUN_CLI:-}" ]; then
    printf '%s' "$IAI_TOOLS_RUN_CLI"
    return 0
  fi
  local cache="$HOME/.iai-mcp/.tools-cli-path"
  if [ -f "$cache" ]; then
    local cached
    cached=$(cat "$cache" 2>/dev/null || true)
    if [ -n "$cached" ] && [ -x "$cached" ]; then
      printf '%s' "$cached"
      return 0
    fi
  fi
  local resolved
  resolved=$(command -v iai-tools 2>/dev/null || true)
  if [ -n "$resolved" ] && [ -x "$resolved" ]; then
    printf '%s' "$resolved"
    return 0
  fi
  local candidate
  for candidate in \
    "$HOME/.pyenv/shims/iai-tools" \
    "$HOME/.local/bin/iai-tools" \
    "$HOME/.local/pipx/venvs/iai-mcp/bin/iai-tools" \
    "/opt/homebrew/bin/iai-tools" \
    "/usr/local/bin/iai-tools"
  do
    if [ -x "$candidate" ]; then
      printf '%s' "$candidate"
      return 0
    fi
  done
  printf ''
}

# Appends one best-effort JSONL line to the redirect savings log. Plain
# atomic `>>` append only -- never a read-modify-write -- so concurrent
# hook invocations cannot race each other. Must never block or fail the
# rewrite: every failure is swallowed.
append_savings_log() {
  local cmd_class="$1"
  local root="${IAI_TOOLS_ROOT:-$HOME/.iai-tools}"
  mkdir -p "$root" 2>/dev/null || true
  local line
  line=$(jq -nc --arg ts "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --arg cls "$cmd_class" \
    '{ts: $ts, cmd_class: $cls}' 2>/dev/null)
  if [ -n "$line" ]; then
    printf '%s\n' "$line" >> "$root/tools-savings.jsonl" 2>/dev/null || true
  fi
}

# Emits the Form B allow+updatedInput.command rewrite. The whole rewritten
# command is passed through jq --arg as a single string value -- never
# built by bash string interpolation into a JSON literal.
emit_redirect() {
  local original="$1" cli="$2"
  jq -n --arg cmd "$cli run -- $original" '{
    hookSpecificOutput: {
      hookEventName: "PreToolUse",
      permissionDecision: "allow",
      updatedInput: { command: $cmd }
    }
  }' 2>/dev/null
}

case "$tool" in
  Grep|Glob)
    emit_deny "BLOCKED (iai-tools-enforce): $tool forbidden -- use mcp__iai-tools__code_search / mcp__iai-tools__execute instead"
    ;;
  Bash)
    cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null || echo "")

    # Command-position boundary: start of the command line, or right after
    # a separator that starts a new command (; | & ( ` or a real newline).
    # A bare space never counts -- it only separates one command's own
    # arguments, and must never be mistaken for the start of a new command
    # (that false match denies e.g. `rm cat`, `git checkout -b find`).
    # Newline stays in the class -- unlike space it IS a real separator
    # (a multi-line Bash command's second line is its own command).
    sep=$'[;|&(`\n]'
    cmdpos="(^|${sep})[[:space:]]*"
    # rtk's subcommand slot: rtk itself must be in command position.
    rtkpos="${cmdpos}rtk[[:space:]]+"

    if [[ "$cmd" =~ ${cmdpos}(grep|rg|ripgrep|find|fd)([[:space:]]|$) ]] \
       || [[ "$cmd" =~ ${rtkpos}(grep|rg|ripgrep|find|fd)([[:space:]]|$) ]]; then
      emit_deny "BLOCKED (iai-tools-enforce): grep/rg/ripgrep/find/fd in Bash forbidden -- use mcp__iai-tools__code_search / mcp__iai-tools__execute instead"
    fi

    # Bulk cat/head/tail, only when actually invoked (command position or
    # rtk's subcommand slot) -- also catches `git log | head` / `some_cmd
    # | tail -50` via a pipe boundary -- as long as no output redirect
    # makes it a write shape instead of a bulk read.
    if { [[ "$cmd" =~ ${cmdpos}(cat|head|tail)([[:space:]]|$) ]] \
         || [[ "$cmd" =~ ${rtkpos}(cat|head|tail)([[:space:]]|$) ]]; } \
       && [[ "$cmd" != *">"* ]]; then
      emit_deny "BLOCKED (iai-tools-enforce): bulk cat/head/tail floods context -- use the Read tool for a single known file, or mcp__iai-tools__execute for command output"
    fi

    # rtk's verb-substituted bulk read (`cat file` -> `rtk read file`).
    # Matched only at rtk's own subcommand position -- the bash `read`
    # builtin (`while read -r line`) has no preceding `rtk` token and is
    # never denied.
    if [[ "$cmd" =~ ${rtkpos}read([[:space:]]|$) ]] \
       && [[ "$cmd" != *">"* ]]; then
      emit_deny "BLOCKED (iai-tools-enforce): bulk cat/head/tail floods context -- use the Read tool for a single known file, or mcp__iai-tools__execute for command output"
    fi

    # Raw gh read shapes: pr/issue/repo/run/release + a read verb, or a
    # bare `gh api` call with no explicit -X/--method (defaults to GET).
    # Write verbs (create/merge/close/comment/edit/review) are untouched.
    if [[ "$cmd" =~ ${cmdpos}gh[[:space:]]+(pr|issue|repo|run|release)[[:space:]]+(view|list|show|diff|log)([[:space:]]|$) ]] \
       || [[ "$cmd" =~ ${rtkpos}gh[[:space:]]+(pr|issue|repo|run|release)[[:space:]]+(view|list|show|diff|log)([[:space:]]|$) ]]; then
      emit_deny "BLOCKED (iai-tools-enforce): raw gh read floods context -- use mcp__iai-tools__execute or mcp__iai-tools__batch_execute to keep the output bounded"
    fi
    # -X/--method must be a real, standalone flag token (whitespace-bounded
    # on both sides) -- a substring match let `-f name=-Xtra` through.
    if { [[ "$cmd" =~ ${cmdpos}gh[[:space:]]+api([[:space:]]|$) ]] \
         || [[ "$cmd" =~ ${rtkpos}gh[[:space:]]+api([[:space:]]|$) ]]; } \
       && ! [[ "$cmd" =~ (^|[[:space:]])(-X|--method)([[:space:]]|$) ]]; then
      emit_deny "BLOCKED (iai-tools-enforce): raw gh read floods context -- use mcp__iai-tools__execute or mcp__iai-tools__batch_execute to keep the output bounded"
    fi

    # Noisy-but-legitimate commands: rewrite to run through the sandbox
    # executor so raw output stays bounded. Fires only in command position,
    # only with no shell operator -- a piped/redirected/chained command is
    # excluded to avoid mis-wrapping only part of it; its output is not
    # otherwise bounded, that is a known scope limit, not a solved case.
    # Double-wrap is structurally impossible: an already-wrapped
    # `<cli> run -- ...` command never has the noisy word at cmdpos/rtkpos
    # -- it sits after a bare space (the one after `run --`), never a
    # separator, so the outer match here never fires on it.
    noisy_pattern='(curl|wget|gradle|gradlew|\.\/gradlew|mvn|mvnw|\.\/mvnw|sbt)'
    if [[ "$cmd" =~ ${cmdpos}${noisy_pattern}([[:space:]]|$) ]] \
       || [[ "$cmd" =~ ${rtkpos}${noisy_pattern}([[:space:]]|$) ]]; then
      noisy_class="${BASH_REMATCH[2]:-noisy}"
      # A bare --version probe is structurally-bounded (fixed, tiny output)
      # -- never worth sandboxing.
      version_probe=0
      [[ "$cmd" =~ (^|[[:space:]])--version([[:space:]]|$) ]] && version_probe=1

      if [ "$version_probe" -eq 0 ] \
         && [[ "$cmd" != *'|'* ]] && [[ "$cmd" != *'>'* ]] && [[ "$cmd" != *'<'* ]] \
         && [[ "$cmd" != *'&&'* ]] && [[ "$cmd" != *';'* ]] && [[ "$cmd" != *'`'* ]] \
         && [[ "$cmd" != *'('* ]] && [[ "$cmd" != *$'\n'* ]]; then
        # No separator survived the exclusion above, so a leading rtk token
        # can only be at string start -- strip it before wrapping, or the
        # sandbox executor double-nests rtk's own output condenser.
        redirect_cmd="$cmd"
        [[ "$cmd" =~ ^[[:space:]]*rtk[[:space:]]+(.*)$ ]] && redirect_cmd="${BASH_REMATCH[1]}"
        tools_cli=$(resolve_tools_cli)
        if [ -n "$tools_cli" ]; then
          emit_redirect "$redirect_cmd" "$tools_cli"
          append_savings_log "$noisy_class"
          exit 0
        fi
      fi
    fi
    ;;
esac

exit 0
