#!/usr/bin/env bash
# scripts/uninstall.sh — LaunchAgent + daemon teardown.
#
# Usage:
#   bash scripts/uninstall.sh                  # remove LaunchAgent + kill daemon
#   bash scripts/uninstall.sh --purge-state    # also remove ~/.iai-mcp/.daemon.sock,
#                                              #             .daemon-state.json, .lock
#   bash scripts/uninstall.sh --purge-data     # also remove ~/.iai-mcp/lancedb +
#                                              #             runtime_graph_cache.json
#                                              # DESTRUCTIVE — wipes user's brain.
#
# Idempotent: safe to re-run. Always exits 0 (best-effort).
# DRY_RUN=1 env skips real launchctl + kill + rm calls (used by tests).
#
# Inverse of scripts/install.sh section 6 (LaunchAgent registration).

# NOTE on shell flags: we deliberately use only `set -u`, NOT `set -e`.
# Uninstall must NEVER abort mid-flow — partial cleanup is worse than
# best-effort full cleanup. Each step prints its own outcome via ok/warn/die.
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[0;33m!\033[0m %s\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. parse flags
# ---------------------------------------------------------------------------
PURGE_STATE=0
PURGE_DATA=0
for arg in "$@"; do
    case "${arg}" in
        --purge-state) PURGE_STATE=1 ;;
        --purge-data)  PURGE_DATA=1 ;;
        -h|--help)
            sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            warn "unknown flag '${arg}' (ignored — expected --purge-state, --purge-data, --help)"
            ;;
    esac
done

step "iai-mcp uninstall"
if [[ "${PURGE_DATA}" == "1" ]]; then
    warn "--purge-data is DESTRUCTIVE: ~/.iai-mcp/lancedb (your brain) will be deleted"
fi

LA_PATH="${HOME}/Library/LaunchAgents/com.iai-mcp.daemon.plist"

# ---------------------------------------------------------------------------
# 2. launchctl unload (Darwin only)
# ---------------------------------------------------------------------------
step "launchctl unload"
if [[ "$(uname)" != "Darwin" ]]; then
    warn "non-Darwin OS — skipping launchctl unload"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    ok "DRY_RUN=1 — skipping launchctl unload (test mode)"
else
    if [ -f "${LA_PATH}" ]; then
        if launchctl unload -w "${LA_PATH}" 2>/dev/null; then
            ok "LaunchAgent unloaded"
        else
            ok "LaunchAgent was not registered (already clean)"
        fi
    else
        ok "no LaunchAgent plist at ${LA_PATH} (already clean)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. remove plist file (Darwin only)
# ---------------------------------------------------------------------------
step "remove plist"
if [[ "$(uname)" != "Darwin" ]]; then
    warn "non-Darwin OS — skipping plist removal"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    ok "DRY_RUN=1 — skipping rm of ${LA_PATH} (test mode)"
else
    rm -f "${LA_PATH}"
    ok "${LA_PATH} removed (or never existed)"
fi

# ---------------------------------------------------------------------------
# 4. kill any lingering daemon by cmdline match
#
# Defense against pgrep regex misfire: pgrep -f matches on substring of
# the full command line. We re-verify each PID's cmdline contains the
# literal "iai_mcp.daemon" via `ps -p PID -o command=` BEFORE killing.
# ---------------------------------------------------------------------------
step "kill lingering daemon"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    ok "DRY_RUN=1 — skipping pgrep + kill (test mode)"
else
    pids="$(pgrep -f "iai_mcp\.daemon" 2>/dev/null || true)"
    if [[ -n "${pids}" ]]; then
        warn "found pids: ${pids}"
        for pid in ${pids}; do
            # Verify cmdline really contains iai_mcp.daemon (defense against pgrep regex misfire).
            if ps -p "${pid}" -o command= 2>/dev/null | grep -q "iai_mcp.daemon"; then
                kill -TERM "${pid}" 2>/dev/null || true
            fi
        done
        sleep 3
        # SIGKILL stragglers
        for pid in ${pids}; do
            if kill -0 "${pid}" 2>/dev/null; then
                warn "pid ${pid} still alive — sending SIGKILL"
                kill -KILL "${pid}" 2>/dev/null || true
            fi
        done
        ok "lingering daemon(s) terminated"
    else
        ok "no lingering iai_mcp.daemon processes"
    fi
fi

# ---------------------------------------------------------------------------
# 5. --purge-state: remove socket + state + lock
# ---------------------------------------------------------------------------
if [[ "${PURGE_STATE}" == "1" ]]; then
    step "--purge-state: remove ~/.iai-mcp/.daemon.sock + .daemon-state.json + .lock"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        ok "DRY_RUN=1 — skipping rm of state files (test mode)"
    else
        rm -f "${HOME}/.iai-mcp/.daemon.sock" \
              "${HOME}/.iai-mcp/.daemon-state.json" \
              "${HOME}/.iai-mcp/.lock"
        ok "state files removed (or never existed)"
    fi
fi

# ---------------------------------------------------------------------------
# 6. --purge-data: remove lancedb + runtime cache (DESTRUCTIVE)
# ---------------------------------------------------------------------------
if [[ "${PURGE_DATA}" == "1" ]]; then
    step "--purge-data: remove ~/.iai-mcp/lancedb + runtime_graph_cache.json"
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        ok "DRY_RUN=1 — skipping rm of data files (test mode)"
    else
        # Confirmation prompt — only if attached to a tty (skip in non-interactive
        # subprocess to avoid hanging under set -u). bash 3.2 compatible.
        confirmed=0
        if [ -t 0 ]; then
            printf "   \033[0;33m!\033[0m really delete ~/.iai-mcp/lancedb? [y/N] "
            read -r REPLY || REPLY=N
            if [[ "${REPLY}" =~ ^[Yy]$ ]]; then
                confirmed=1
            fi
        else
            warn "non-interactive stdin — skipping --purge-data confirmation (no deletion)"
        fi
        if [[ "${confirmed}" == "1" ]]; then
            rm -rf "${HOME}/.iai-mcp/lancedb" \
                   "${HOME}/.iai-mcp/runtime_graph_cache.json"
            ok "data files removed"
        else
            ok "user declined — data preserved"
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 7. verify
# ---------------------------------------------------------------------------
step "verify"
if [[ "$(uname)" != "Darwin" ]]; then
    warn "non-Darwin OS — skipping launchctl verify"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    ok "DRY_RUN=1 — skipping launchctl list verify (test mode)"
else
    if launchctl list 2>/dev/null | grep -q "com.iai-mcp.daemon"; then
        warn "com.iai-mcp.daemon still appears in launchctl list — manual cleanup may be needed"
    else
        ok "com.iai-mcp.daemon no longer in launchctl list"
    fi
fi

# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------
step "done"
ok "iai-mcp uninstalled — re-run scripts/install.sh to restore."
exit 0
