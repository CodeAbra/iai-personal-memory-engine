#!/bin/sh
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
# IAI-MCP Stop hook — turn-boundary checkpoint for ambient capture.
#
# Claude Code fires Stop after EVERY assistant response, not at session end.
# This hook therefore does two cheap incremental things per response:
#   1. `iai-mcp capture-turn-deferred` — appends only the transcript lines
#      newer than the per-session offset (the response that just finished)
#      to the session's live spool. Same offset the UserPromptSubmit hook
#      maintains; it is NEVER deleted here — wiping it forces the next
#      capture to re-read the whole transcript from line 0, and a full
#      re-capture per response floods the deferred spool by gigabytes on
#      long-running sessions.
#   2. Rotates {session_id}.live.jsonl so the drain can claim the turns.
#
# Full-transcript capture (`iai-mcp capture-transcript`) stays a manual
# import/recovery tool — it must not run on a per-response event.
#
# Fail-safe by design: any error exits 0 so the response is never blocked.
# Logs go to ~/.iai-mcp/logs/capture-YYYY-MM-DD.log for audit.
#
# Hook payload (stdin JSON from Claude Code) contains:
#   - session_id       (UUID of the active session)
#   - transcript_path  (absolute path to the session JSONL) — available in
#                      newer Claude Code builds; we fall back to scanning the
#                      per-project transcript dir for the matching session_id.
#   - cwd              (working directory at fire time)

set -u  # no -e: we must not abort on errors, fail-safe is paramount
input=$(cat 2>/dev/null || true)

# Best-effort jq; fall back to Python if jq missing.
extract() {
  key=$1
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r ".${key} // empty" 2>/dev/null
  else
    printf '%s' "$input" | "$PYBIN" -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('${key}', '') or '')
except Exception:
    print('')
" 2>/dev/null
  fi
}

session_id=$(extract "session_id")
transcript_path=$(extract "transcript_path")
cwd=$(extract "cwd")

# Fallback: locate transcript if the hook payload didn't include its path.
# Claude Code stores transcripts under ~/.claude/projects/{cwd-hash}/{uuid}.jsonl
if [ -z "$transcript_path" ] && [ -n "$session_id" ]; then
  projects_dir="$HOME/.claude/projects"
  if [ -d "$projects_dir" ]; then
    # Look for the most recent file whose basename starts with session_id.
    # ls -t (mtime newest first). Avoid `find` per the project's no-grep hook.
    for d in "$projects_dir"/*/; do
      candidate="${d}${session_id}.jsonl"
      if [ -f "$candidate" ]; then
        transcript_path="$candidate"
        break
      fi
    done
  fi
fi

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/capture-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

{
  echo "---"
  echo "$ts session=$session_id cwd=$cwd transcript=$transcript_path"
} >> "$log" 2>/dev/null

# Skip if we couldn't find anything to capture.
if [ -z "$transcript_path" ] || [ ! -f "$transcript_path" ]; then
  echo "$ts skipped: no transcript found" >> "$log" 2>/dev/null
  exit 0
fi

# Locate the `iai-mcp` CLI. Resolution order:
#   1. IAI_MCP_SESSION_CAPTURE_CLI environment variable — highest priority;
#      set in your shell init for non-standard install locations.
#   2. ~/.iai-mcp/.cli-path cache — written on first successful resolution
#      so subsequent session-ends skip the scan entirely.
#   3. `command -v iai-mcp` — PATH lookup; picks up pyenv shims, pipx
#      wrappers, and any other PATH-managed install transparently.
#   4. Baked-in candidate list — checked when PATH has no entry; covers
#      common install locations (pyenv shims, pipx, homebrew, user-site,
#      dev venv).
# Only generic $HOME-relative or system paths belong here; install-specific
# paths belong in the env var or the cache.
cli_cache="$HOME/.iai-mcp/.cli-path"
iai_cli=""
if [ -n "${IAI_MCP_SESSION_CAPTURE_CLI:-}" ] && [ -x "$IAI_MCP_SESSION_CAPTURE_CLI" ]; then
  iai_cli="$IAI_MCP_SESSION_CAPTURE_CLI"
fi
if [ -z "$iai_cli" ] && [ -f "$cli_cache" ]; then
  cached=$(cat "$cli_cache" 2>/dev/null || true)
  [ -x "$cached" ] && iai_cli="$cached"
fi
if [ -z "$iai_cli" ]; then
  resolved=$(command -v iai-mcp 2>/dev/null || true)
  if [ -n "$resolved" ] && [ -x "$resolved" ]; then
    iai_cli="$resolved"
    printf '%s' "$iai_cli" > "$cli_cache" 2>/dev/null || true
  fi
fi
if [ -z "$iai_cli" ]; then
  for candidate in \
    "$HOME/.pyenv/shims/iai-mcp" \
    "$HOME/.local/bin/iai-mcp" \
    "$HOME/.local/pipx/venvs/iai-mcp/bin/iai-mcp" \
    "/opt/homebrew/bin/iai-mcp" \
    "$HOME/IAI-MCP/.venv/bin/iai-mcp" \
    "/usr/local/bin/iai-mcp"
  do
    if [ -x "$candidate" ]; then
      iai_cli="$candidate"
      printf '%s' "$iai_cli" > "$cli_cache" 2>/dev/null || true
      break
    fi
  done
fi

if [ -z "$iai_cli" ]; then
  echo "$ts skipped: iai-mcp CLI not found" >> "$log" 2>/dev/null
  exit 0
fi

# Incremental catch-up FIRST: the assistant response that just finished is in
# the transcript past the offset but not yet in the live spool (the per-turn
# hook only fires on the NEXT user prompt, which may never come). 30s hard
# timeout; `timeout` is in coreutils (macOS: brew install coreutils).
if command -v timeout >/dev/null 2>&1; then
  result=$(timeout 30 "$iai_cli" capture-turn-deferred \
    --session-id "$session_id" \
    --transcript-path "$transcript_path" \
    --max-turns-per-call 1000 2>&1)
elif command -v gtimeout >/dev/null 2>&1; then
  result=$(gtimeout 30 "$iai_cli" capture-turn-deferred \
    --session-id "$session_id" \
    --transcript-path "$transcript_path" \
    --max-turns-per-call 1000 2>&1)
else
  result=$("$iai_cli" capture-turn-deferred \
    --session-id "$session_id" \
    --transcript-path "$transcript_path" \
    --max-turns-per-call 1000 2>&1)
fi
rc=$?

# THEN atomically rename the active-writer marker so the drain can claim the
# fully-captured turn on its next pass. Target name uses
# `.live-${epoch}-${pid}.jsonl`: the `.live-` prefix keeps it clear of the
# bulk-import output shape, and the pid keeps two capture events firing in
# the same wall-clock second from overwriting each other. The per-session
# offset state is deliberately left in place — it is line-count-relative to
# the transcript, not to the live file, and stays valid across rotations.
if [ -n "$session_id" ]; then
  live_file="$HOME/.iai-mcp/.deferred-captures/${session_id}.live.jsonl"
  if [ -f "$live_file" ]; then
    mv "$live_file" "$HOME/.iai-mcp/.deferred-captures/${session_id}.live-$(date +%s)-$$.jsonl" 2>/dev/null || true
  fi
fi

{
  echo "$ts rc=$rc result=$result"
} >> "$log" 2>/dev/null

exit 0
