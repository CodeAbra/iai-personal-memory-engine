#!/bin/sh
# IAI-MCP SessionStart hook — iai-tools code-index refresh.
#
# Fires on Claude Code session start (sources: startup, resume, clear,
# compact). Reads the stdin JSON for the project directory (`cwd`, falling
# back to $PWD) and shells to `iai-tools index <dir>` to refresh the
# brain-independent code-index store for that project.
#
# Side-effect hook, not a context injector: emits nothing on stdout on
# success.
#
# Fail-safe by design: every error path (missing binary, non-zero indexer,
# timeout) exits 0 with empty stdout so a broken/missing iai-tools binary
# never blocks session start. Logs go to
# ~/.iai-mcp/logs/tools-index-YYYY-MM-DD.log for audit.
#
# Freshness-gated: re-indexing an unchanged tree appends a new segment
# whose file entries duplicate the prior segment's — compaction merges
# segment COUNT but never dedupes by file path, so an ungated per-session
# call grows the store with unreclaimed duplicate postings. A watermark
# keyed by project path skips the index call once nothing has changed
# since the last run: git HEAD+porcelain-content when available, else a
# top-level directory-listing checksum. A cap-exceed attempt backs off
# instead of re-timing-out on every session start.

set -u  # no -e: fail-safe is paramount
input=$(cat 2>/dev/null || true)

extract() {
  key=$1
  if command -v jq >/dev/null 2>&1; then
    printf '%s' "$input" | jq -r ".${key} // empty" 2>/dev/null
  else
    printf '%s' "$input" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get('${key}', '') or '')
except Exception:
    print('')
" 2>/dev/null
  fi
}

# Reads a directory's own mtime (set atomically by `mkdir`, unlike a file
# written inside it after the fact) as an epoch integer, or empty on any
# failure -- callers must treat empty as "staleness unknown," never as zero,
# so a lock whose age cannot be determined is never falsely stolen.
dir_mtime() {
  m=""
  if command -v stat >/dev/null 2>&1; then
    m=$(stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null || true)
    case "$m" in ''|*[!0-9]*) m="" ;; esac
  fi
  if [ -z "$m" ]; then
    m=$(python3 -c "
import os, sys
try:
    print(int(os.stat(sys.argv[1]).st_mtime))
except Exception:
    pass
" "$1" 2>/dev/null || true)
    case "$m" in ''|*[!0-9]*) m="" ;; esac
  fi
  printf '%s' "$m"
}

project_dir=$(extract "cwd")
[ -n "$project_dir" ] || project_dir="$PWD"
[ -d "$project_dir" ] || exit 0

mkdir -p "$HOME/.iai-mcp/logs" 2>/dev/null || true
log="$HOME/.iai-mcp/logs/tools-index-$(date -u +%Y-%m-%d).log"
ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)

# $HOME is never a project to index.
[ "$project_dir" = "$HOME" ] && { echo "$ts skipped: home dir not indexed dir=$project_dir" >> "$log" 2>/dev/null; exit 0; }

# Default 20s; IAI_MCP_TOOLS_INDEX_HOOK_TIMEOUT overrides. Read here (not
# just at the CLI-run site below) because the lock-staleness check also
# needs it.
hook_timeout="${IAI_MCP_TOOLS_INDEX_HOOK_TIMEOUT:-20}"

# Freshness watermark: keyed by a checksum of the project path so distinct
# repos never collide. When the store root cannot be created, degrade to
# "no watermark" (signature stays unset below) rather than fail the hook.
watermark_dir="$HOME/.iai-mcp/tools/.watermarks"
mkdir -p "$watermark_dir" 2>/dev/null || true
path_hash=$(printf '%s' "$project_dir" | cksum | cut -d' ' -f1)
watermark_file="$watermark_dir/$path_hash"

signature=""
if command -v git >/dev/null 2>&1 && git -C "$project_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  head_sha=$(git -C "$project_dir" rev-parse HEAD 2>/dev/null || true)
  dirty_hash=$(git -C "$project_dir" status --porcelain 2>/dev/null | cksum | cut -d' ' -f1)
  signature="git:${head_sha}:${dirty_hash}"
else
  # Non-git fallback: top-level entry names plus file content hashes.
  # Coarser than the git-porcelain signature (top level only, misses nested
  # changes) but keeps the gate convergent for non-git trees instead of
  # re-indexing on every single session start. Hashes CONTENT rather than a
  # directory listing's metadata columns -- `ls -l` renders mtime at minute
  # granularity, so a same-size edit inside the same calendar minute would
  # otherwise alias to an unchanged signature (the same failure class fixed
  # for the git-porcelain path above). The entry-name listing is combined
  # in so a top-level directory's own add/remove still moves the
  # signature, since only regular files are content-hashed.
  signature="nogit:$(
    {
      find "$project_dir" -maxdepth 1 2>/dev/null | sort
      find "$project_dir" -maxdepth 1 -type f 2>/dev/null | sort | xargs -I{} cksum "{}" 2>/dev/null
    } | cksum | cut -d' ' -f1
  )"
fi

# Lock the check-run-write sequence: `mkdir` is atomic and POSIX-portable,
# so two SessionStart hooks racing for the same project cannot both pass
# the freshness check and both re-index. Fail-safe over fail-strict: if the
# lock is already held, this session just skips its own reindex rather than
# waiting (session start must never block on another session's run) --
# convergence still holds because the lock holder's own write leaves a
# fresh watermark behind for the next hook invocation to read. A lock older
# than 2x hook_timeout has outlived any run its own `timeout` wrapper would
# have allowed, so it is stolen (its holder is presumed dead, e.g.
# SIGKILLed) rather than left to block every future session forever.
lock_dir="$watermark_file.lock"
lock_age_limit=$((hook_timeout * 2))
have_lock=0
if mkdir "$lock_dir" 2>/dev/null; then
  have_lock=1
elif [ -d "$lock_dir" ]; then
  # Staleness is read from the lock DIRECTORY's own mtime, set atomically by
  # `mkdir` -- never from a file written inside it afterwards. A stamp-file
  # signal has a window where a live, just-acquired lock (mkdir succeeded,
  # stamp not yet written) reads identically to a dead holder's lock (stamp
  # will never come); the directory's own mtime has no such window, since
  # `mkdir` sets it in the same syscall that creates the lock.
  lock_mtime=$(dir_mtime "$lock_dir")
  now_ts_lock=$(date -u +%s)
  if [ -n "$lock_mtime" ] && [ $((now_ts_lock - lock_mtime)) -gt "$lock_age_limit" ]; then
    # Steal via rename, not rm+mkdir: `rm -rf` then `mkdir` is two steps, so
    # two contenders that both observe "stale" can both remove and both
    # recreate, defeating the lock entirely. `mv` on a directory is a single
    # atomic rename -- only one contender's `mv` can still find the
    # directory at $lock_dir, so only one contender can ever win a steal.
    steal_tmp="$lock_dir.stale.$$"
    rm -rf "$steal_tmp" 2>/dev/null || true
    if mv "$lock_dir" "$steal_tmp" 2>/dev/null; then
      rm -rf "$steal_tmp" 2>/dev/null || true
      mkdir "$lock_dir" 2>/dev/null && have_lock=1
    fi
  fi
  # No readable mtime means staleness cannot be established: do not steal.
  # A wrong steal (live double-index) is worse than a wedge on a system
  # missing both `stat` and `python3`, a broken-system precondition already
  # out of scope elsewhere in this hook.
fi
if [ "$have_lock" -ne 1 ]; then
  echo "$ts skipped: lock held by concurrent session dir=$project_dir" >> "$log" 2>/dev/null
  exit 0
fi
# A lock instance is tagged with a nonce; the release trap removes the
# directory only if the nonce it reads back still matches. Without this, a
# process whose lock was legitimately stolen (it overran 2x hook_timeout
# before its own trap ran) would later delete the new holder's live lock on
# its own exit, handing the next arrival a free, uncontended `mkdir` --
# unbounded past two processes.
nonce="$$-$(date -u +%s%N 2>/dev/null || date -u +%s)"
trap 'owner=$(cat "$lock_dir/owner" 2>/dev/null || true); if [ "$owner" = "$nonce" ]; then rm -rf "$lock_dir" 2>/dev/null || true; fi' EXIT INT TERM
printf '%s' "$nonce" > "$lock_dir/owner" 2>/dev/null || true
owner_check=$(cat "$lock_dir/owner" 2>/dev/null || true)
if [ "$owner_check" != "$nonce" ]; then
  # Our own stamp write failed. Only clean up an EMPTY/unreadable owner
  # file -- a foreign nonce means another process already owns this
  # instance (e.g. it stole this exact lock under a misconfigured near-zero
  # timeout), and deleting it would be the exact cascade the nonce exists
  # to prevent.
  if [ -z "$owner_check" ]; then
    rm -rf "$lock_dir" 2>/dev/null || true
  fi
  echo "$ts skipped: lock unstampable dir=$project_dir" >> "$log" 2>/dev/null
  exit 0
fi

if [ -n "$signature" ] && [ -f "$watermark_file" ]; then
  prior=$(cat "$watermark_file" 2>/dev/null || true)
  if [ "$prior" = "$signature" ]; then
    echo "$ts skipped: unchanged (freshness gate) dir=$project_dir" >> "$log" 2>/dev/null
    exit 0
  fi
fi

# Timeout back-off: keyed separately from the success watermark so a
# cap-exceed attempt can never be read back as a successful signature. When
# the last attempt at this exact signature timed out, back off instead of
# re-attempting (and re-timing-out) on every session start; a signature
# change (real tree change) or an elapsed backoff window still retries.
attempt_file="$watermark_dir/$path_hash.attempted"
attempt_ts_file="$watermark_dir/$path_hash.attempted-ts"
backoff_seconds="${IAI_MCP_TOOLS_INDEX_BACKOFF_SECONDS:-3600}"
if [ -n "$signature" ] && [ -f "$attempt_file" ]; then
  attempt_sig=$(cat "$attempt_file" 2>/dev/null || true)
  if [ "$attempt_sig" = "$signature" ]; then
    attempt_ts=$(cat "$attempt_ts_file" 2>/dev/null || true)
    case "$attempt_ts" in ''|*[!0-9]*) attempt_ts=0 ;; esac
    now_ts=$(date -u +%s)
    elapsed=$((now_ts - attempt_ts))
    if [ "$elapsed" -lt "$backoff_seconds" ]; then
      echo "$ts skipped: backoff after prior timeout (${elapsed}s<${backoff_seconds}s) dir=$project_dir" >> "$log" 2>/dev/null
      exit 0
    fi
  fi
fi

# Locate the iai-tools executable. $project_dir is only ever the index
# TARGET argument below -- never a source of an executable path. Resolution
# order (trusted sources only):
#   1. IAI_MCP_TOOLS_INDEX_CLI environment variable (developer override).
#   2. ~/.iai-mcp/.tools-cli-path cache file — written by `iai-mcp
#      capture-hooks install` at install time, or self-cached below on a
#      successful PATH/candidate resolution.
#   3. `command -v iai-tools` — PATH lookup.
#   4. Baked-in candidate list — checked when PATH has no entry.
cli_cache="$HOME/.iai-mcp/.tools-cli-path"
tools_cli=""
if [ -n "${IAI_MCP_TOOLS_INDEX_CLI:-}" ] && [ -x "$IAI_MCP_TOOLS_INDEX_CLI" ]; then
  tools_cli="$IAI_MCP_TOOLS_INDEX_CLI"
fi
if [ -z "$tools_cli" ] && [ -f "$cli_cache" ]; then
  cached=$(cat "$cli_cache" 2>/dev/null || true)
  [ -x "$cached" ] && tools_cli="$cached"
fi
if [ -z "$tools_cli" ]; then
  resolved=$(command -v iai-tools 2>/dev/null || true)
  if [ -n "$resolved" ] && [ -x "$resolved" ]; then
    tools_cli="$resolved"
    printf '%s' "$tools_cli" > "$cli_cache" 2>/dev/null || true
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
      printf '%s' "$tools_cli" > "$cli_cache" 2>/dev/null || true
      break
    fi
  done
fi
if [ -z "$tools_cli" ]; then
  echo "$ts skipped: iai-tools CLI not found dir=$project_dir" >> "$log" 2>/dev/null
  exit 0
fi

# Hard cap on the CLI call. On cap-exceed the index call is killed, not
# left hanging -- session start is never blocked.
set -- "$tools_cli" index "$project_dir"
if command -v timeout >/dev/null 2>&1; then
  timeout "$hook_timeout" "$@" >>"$log" 2>>"$log"
  rc=$?
elif command -v gtimeout >/dev/null 2>&1; then
  gtimeout "$hook_timeout" "$@" >>"$log" 2>>"$log"
  rc=$?
else
  # POSIX watchdog when coreutils is absent: launch the CLI in background,
  # poll for completion, kill on cap-exceed.
  "$@" >>"$log" 2>>"$log" &
  cli_pid=$!
  killed=0
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
    killed=1
  fi
  wait "$cli_pid" 2>/dev/null
  rc=$?
  [ "$killed" -eq 1 ] && rc=124
fi

if [ "$rc" -eq 0 ] && [ -n "$signature" ]; then
  printf '%s' "$signature" > "$watermark_file" 2>/dev/null || true
  rm -f "$attempt_file" "$attempt_ts_file" 2>/dev/null || true
elif [ "$rc" -ne 0 ] && [ -n "$signature" ]; then
  printf '%s' "$signature" > "$attempt_file" 2>/dev/null || true
  date -u +%s > "$attempt_ts_file" 2>/dev/null || true
fi
echo "$ts rc=$rc dir=$project_dir" >> "$log" 2>/dev/null
exit 0
