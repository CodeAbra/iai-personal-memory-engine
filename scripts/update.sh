#!/usr/bin/env bash
# scripts/update.sh — pull + rebuild + restart daemon for collaborators
#
# Usage (from repo root or anywhere inside the clone):
#   bash scripts/update.sh
#
# Idempotent. Aborts on a dirty working tree so local changes are never
# clobbered. Re-runs safely — each step detects whether it is needed.

set -euo pipefail

# Resolve repo root no matter where the script is invoked from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[0;33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# 0. Preconditions
# ---------------------------------------------------------------------------
step "preflight"
[ -d .git ] || die "not a git repository (run from an iai-mcp clone)"

# Require a clean working tree — never trample local edits.
if [ -n "$(git status --porcelain)" ]; then
    git status --short
    die "working tree is dirty. commit or stash first, then re-run."
fi
ok "working tree clean"

VENV_PY="${REPO_ROOT}/.venv/bin/python"
[ -x "${VENV_PY}" ] || die ".venv/bin/python not found — run 'python3 -m venv .venv && .venv/bin/pip install -e .' once, then rerun"
ok "venv detected"

# ---------------------------------------------------------------------------
# 1. git pull (fast-forward only — never merge surprises)
# ---------------------------------------------------------------------------
step "git pull --ff-only origin main"
BEFORE="$(git rev-parse HEAD)"
git fetch --quiet origin main
git pull --ff-only --quiet origin main
AFTER="$(git rev-parse HEAD)"
if [ "${BEFORE}" = "${AFTER}" ]; then
    ok "already at $(git rev-parse --short HEAD) — no upstream commits"
    NOOP=1
else
    ok "advanced $(git rev-parse --short "${BEFORE}") → $(git rev-parse --short "${AFTER}")"
    NOOP=0
fi

# ---------------------------------------------------------------------------
# 2. Python package (editable reinstall — picks up deps or entry-point drift)
# ---------------------------------------------------------------------------
step "python package refresh (editable)"
"${VENV_PY}" -m pip install --quiet -e . || die "pip install -e failed"
ok "iai-mcp python package up to date"

# ---------------------------------------------------------------------------
# 3. TypeScript MCP wrapper
# ---------------------------------------------------------------------------
step "TS wrapper build"
if [ -d mcp-wrapper ]; then
    pushd mcp-wrapper >/dev/null
    if [ -f package-lock.json ]; then
        npm ci --silent --no-audit --no-fund
    else
        npm install --silent --no-audit --no-fund
    fi
    npm run build --silent
    popd >/dev/null
    ok "mcp-wrapper/dist rebuilt"
else
    warn "mcp-wrapper/ missing — skipping"
fi

# ---------------------------------------------------------------------------
# 4. Global CLI symlink (idempotent — ensures ~/.local/bin/iai-mcp exists)
# ---------------------------------------------------------------------------
step "global CLI symlink"
LOCAL_BIN="${HOME}/.local/bin"
LINK_PATH="${LOCAL_BIN}/iai-mcp"
TARGET="${REPO_ROOT}/.venv/bin/iai-mcp"
if [ -e "${LINK_PATH}" ] && [ ! -L "${LINK_PATH}" ]; then
    warn "${LINK_PATH} exists as a regular file — skipping symlink refresh"
else
    mkdir -p "${LOCAL_BIN}"
    ln -sf "${TARGET}" "${LINK_PATH}"
    ok "${LINK_PATH} -> ${TARGET}"
fi

# ---------------------------------------------------------------------------
# 5. Daemon (restart only if currently running; plist drift advisory)
# ---------------------------------------------------------------------------
step "daemon lifecycle"
IAI_MCP="${REPO_ROOT}/.venv/bin/iai-mcp"

# Check template drift using a python one-liner (avoids shell grep, which is
# hook-blocked in this repo's dev env).
TEMPLATE_CHECK="$("${VENV_PY}" - <<'PY'
import pathlib, sys
home = pathlib.Path.home()
installed = home / "Library/LaunchAgents/com.iai-mcp.daemon.plist"
template  = pathlib.Path.cwd() / "src" / "iai_mcp" / "_deploy" / "launchd" / "com.iai-mcp.daemon.plist"
if not installed.exists() or not template.exists():
    print("none"); sys.exit(0)
# Substitute USERNAME placeholder and compare env-var + args payload.
rendered = template.read_text().replace("{USERNAME}", home.name)
a_env = "IAI_MCP_STORE" in installed.read_text() and home.as_posix() + "/.iai-mcp" in installed.read_text()
b_env = "IAI_MCP_STORE" in rendered and home.as_posix() + "/.iai-mcp" in rendered
print("drift" if a_env != b_env else "same")
PY
)"

if [ "${TEMPLATE_CHECK}" = "drift" ]; then
    warn "launchd plist template drift detected"
    warn "run: '${IAI_MCP} daemon uninstall --yes && ${IAI_MCP} daemon install --yes' to pick up the new plist"
fi

if "${IAI_MCP}" daemon status >/dev/null 2>&1; then
    # daemon status exits 0 only when running
    "${IAI_MCP}" daemon stop >/dev/null 2>&1 || true
    sleep 2
    "${IAI_MCP}" daemon start >/dev/null 2>&1 || warn "daemon start returned non-zero; check 'iai-mcp daemon status'"
    ok "daemon restarted on new code"
else
    ok "daemon not running — nothing to restart"
fi

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
step "done"
if [ "${NOOP}" = "1" ]; then
    ok "no-op — everything already current"
else
    ok "updated to $(git rev-parse --short HEAD)"
    echo
    git log --oneline "${BEFORE}..${AFTER}"
fi
