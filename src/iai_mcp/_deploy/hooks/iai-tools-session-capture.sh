#!/bin/sh
# IAI-MCP PostToolUse/UserPromptSubmit hook -- session-event capture producer.
#
# Reads raw hook JSON from stdin and pipes it, unmodified, to
# `iai-tools capture --source-hook <lifecycle>`, where <lifecycle> is this
# script's first positional argument, set by the settings.json command line
# (e.g. `bash iai-tools-session-capture.sh PostToolUse`). The lifecycle
# label is carried this way rather than read from the payload itself, so
# classification never depends on the host supplying a reliable
# event-identifying JSON field.
#
# Side-effect hook, not a context injector: emits nothing on stdout.
#
# Fail-safe by design: every error path (missing binary, non-zero CLI,
# timeout, empty/missing lifecycle argument, unreadable stdin) exits 0 so a
# broken/missing iai-tools binary never blocks a host turn.

set -u  # no -e: fail-safe is paramount

source_hook="${1:-}"
input=$(cat 2>/dev/null || true)

[ -n "$source_hook" ] || exit 0

# Locate the iai-tools executable. Resolution order (trusted sources only):
#   1. IAI_MCP_SESSION_CAPTURE_CLI environment variable (developer override).
#   2. ~/.iai-mcp/.tools-cli-path cache file -- written by `iai-mcp
#      capture-hooks install` at install time, shared with the code-index
#      SessionStart hook (both resolve the same console script). This hook
#      only ever reads the cache, never writes it -- self-caching is the
#      installer's job.
#   3. `command -v iai-tools` -- PATH lookup.
#   4. Baked-in candidate list -- checked when PATH has no entry.
cli_cache="$HOME/.iai-mcp/.tools-cli-path"
tools_cli=""
if [ -n "${IAI_MCP_SESSION_CAPTURE_CLI:-}" ] && [ -x "$IAI_MCP_SESSION_CAPTURE_CLI" ]; then
  tools_cli="$IAI_MCP_SESSION_CAPTURE_CLI"
fi
if [ -z "$tools_cli" ] && [ -f "$cli_cache" ]; then
  cached=$(cat "$cli_cache" 2>/dev/null || true)
  [ -x "$cached" ] && tools_cli="$cached"
fi
if [ -z "$tools_cli" ]; then
  resolved=$(command -v iai-tools 2>/dev/null || true)
  if [ -n "$resolved" ] && [ -x "$resolved" ]; then
    tools_cli="$resolved"
  fi
fi
if [ -z "$tools_cli" ]; then
  for candidate in \
    "$HOME/.pyenv/shims/iai-tools" \
    "$HOME/.local/bin/iai-tools" \
    "$HOME/.local/pipx/venvs/iai-mcp/bin/iai-tools" \
    "/opt/homebrew/bin/iai-tools" \
    "/usr/local/bin/iai-tools"
  do
    if [ -x "$candidate" ]; then
      tools_cli="$candidate"
      break
    fi
  done
fi
[ -n "$tools_cli" ] || exit 0

hook_timeout="${IAI_MCP_SESSION_CAPTURE_HOOK_TIMEOUT:-5}"

set -- "$tools_cli" capture --source-hook "$source_hook"
if command -v timeout >/dev/null 2>&1; then
  printf '%s' "$input" | timeout "$hook_timeout" "$@" >/dev/null 2>&1
elif command -v gtimeout >/dev/null 2>&1; then
  printf '%s' "$input" | gtimeout "$hook_timeout" "$@" >/dev/null 2>&1
else
  # POSIX watchdog when coreutils is absent: launch the CLI in background,
  # poll for completion, kill on cap-exceed.
  printf '%s' "$input" | "$@" >/dev/null 2>&1 &
  cli_pid=$!
  i=0
  max_iter=$((hook_timeout * 10))
  while [ "$i" -lt "$max_iter" ]; do
    if ! kill -0 "$cli_pid" 2>/dev/null; then break; fi
    sleep 0.1
    i=$((i + 1))
  done
  if kill -0 "$cli_pid" 2>/dev/null; then
    kill -TERM "$cli_pid" 2>/dev/null
    sleep 0.2
    kill -KILL "$cli_pid" 2>/dev/null
  fi
  wait "$cli_pid" 2>/dev/null
fi

exit 0
