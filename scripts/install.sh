#!/usr/bin/env bash
# scripts/install.sh — first-time setup for collaborators.
#
# Usage (from repo root or anywhere inside the clone):
#   bash scripts/install.sh
#
# Does:
#   1. creates .venv if missing
#   2. installs iai-mcp editable into the venv
#   3. builds the TS MCP wrapper
#   4. symlinks ~/.local/bin/iai-mcp -> .venv/bin/iai-mcp so the CLI is
#      callable from anywhere without activating the venv
#   5. optionally installs the sleep daemon (launchd on macOS, systemd on Linux)
#
# Idempotent. Safe to re-run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[0;33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Sections 1-4: build / venv / pip / npm / symlink.
#
# IAI_TEST_SKIP_BUILD=1 short-circuits the whole bootstrap so the LaunchAgent
# section (6) can be exercised in isolation by tests/test_install_uninstall.py
# (Plan 07.1-03 Task 3) without spending ~30s on venv + npm.
# ---------------------------------------------------------------------------
if [[ "${IAI_TEST_SKIP_BUILD:-0}" == "1" ]]; then
    step "build skip (IAI_TEST_SKIP_BUILD=1)"
    ok "skipping sections 1-4 (venv/pip/npm/symlink) — test mode"
else
    # -----------------------------------------------------------------------
    # 1. venv
    # -----------------------------------------------------------------------
    step "python venv"
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        ok ".venv created"
    else
        ok ".venv already exists"
    fi

    # -----------------------------------------------------------------------
    # 2. editable install
    # -----------------------------------------------------------------------
    step "editable install (pip -e .)"
    .venv/bin/pip install --quiet --upgrade pip
    .venv/bin/pip install --quiet -e .
    ok "iai-mcp python package installed into venv"

    # -----------------------------------------------------------------------
    # 3. TS wrapper build
    # -----------------------------------------------------------------------
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
        ok "mcp-wrapper/dist built"
    else
        warn "mcp-wrapper/ missing — skipping"
    fi

    # -----------------------------------------------------------------------
    # 4. global symlink into ~/.local/bin
    # -----------------------------------------------------------------------
    step "global CLI symlink"
    LOCAL_BIN="${HOME}/.local/bin"
    LINK_PATH="${LOCAL_BIN}/iai-mcp"
    TARGET="${REPO_ROOT}/.venv/bin/iai-mcp"

    [ -x "${TARGET}" ] || die "venv entry point not found at ${TARGET}"

    mkdir -p "${LOCAL_BIN}"

    # `ln -sf` overwrites any existing symlink safely (idempotent).
    # Refuse to clobber a regular file the user put there themselves.
    if [ -e "${LINK_PATH}" ] && [ ! -L "${LINK_PATH}" ]; then
        die "${LINK_PATH} exists and is NOT a symlink. move it aside and re-run."
    fi
    ln -sf "${TARGET}" "${LINK_PATH}"
    ok "${LINK_PATH} -> ${TARGET}"

    # PATH sanity check using python (grep is hook-blocked in this dev env).
    PATH_HAS_LOCAL_BIN="$(.venv/bin/python - <<PY
import os
print("1" if "${LOCAL_BIN}" in os.environ.get("PATH", "").split(":") else "0")
PY
)"
    if [ "${PATH_HAS_LOCAL_BIN}" != "1" ]; then
        warn "${LOCAL_BIN} is NOT in your PATH"
        warn "add this to ~/.zshrc or ~/.bashrc and restart your shell:"
        warn "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
    else
        ok "${LOCAL_BIN} is in PATH"
    fi
fi

# ---------------------------------------------------------------------------
# 5. optional daemon install
# ---------------------------------------------------------------------------
step "sleep daemon (optional)"
if command -v iai-mcp >/dev/null 2>&1; then
    INSTALLED_PATH="$(command -v iai-mcp)"
    ok "iai-mcp globally reachable at ${INSTALLED_PATH}"
    echo
    echo "   to run the background sleep daemon (recommended — REM cycles +"
    echo "   overnight consolidation on your local Claude subscription):"
    echo
    echo "     iai-mcp daemon install --yes"
    echo "     iai-mcp daemon start"
    echo
    echo "   or skip for now and install later."
else
    warn "iai-mcp not on PATH yet — add ~/.local/bin to PATH first, then run:"
    warn "  iai-mcp daemon install --yes"
fi

# ---------------------------------------------------------------------------
# 6. LaunchAgent registration (socket-activated singleton)
#
# Section 6 — daemon service registration.
# macOS: renders plist template, registers with launchctl.
# Linux: delegates to `iai-mcp daemon install --yes` (systemd user unit).
# Idempotent on both platforms.
# ---------------------------------------------------------------------------
step "daemon service registration"
if [[ "$(uname)" == "Linux" ]]; then
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        ok "DRY_RUN=1 — skipping systemd registration (test mode)"
    else
        if [ ! -f "${HOME}/.iai-mcp/.crypto.key" ] && [ -z "${IAI_MCP_CRYPTO_PASSPHRASE:-}" ]; then
            if "${REPO_ROOT}/.venv/bin/iai-mcp" crypto init >/dev/null 2>&1; then
                ok "crypto key generated (~/.iai-mcp/.crypto.key)"
            else
                warn "crypto init failed — run \`iai-mcp crypto init\` manually"
            fi
        fi
        if "${REPO_ROOT}/.venv/bin/iai-mcp" daemon install --yes 2>&1; then
            ok "systemd user service installed and started"
        else
            warn "systemd registration failed — run \`iai-mcp daemon install\` manually"
        fi
    fi
elif [[ "$(uname)" != "Darwin" ]]; then
    warn "unsupported OS ($(uname)) — skipping daemon registration"
elif [[ "${DRY_RUN:-0}" == "1" ]]; then
    ok "DRY_RUN=1 — skipping launchctl calls (test mode)"
else
    PYTHON_PATH="${REPO_ROOT}/.venv/bin/python"
    if [ ! -x "${PYTHON_PATH}" ]; then
        warn "venv python not found at ${PYTHON_PATH} — falling back to $(command -v python3)"
        PYTHON_PATH="$(command -v python3)"
    fi
    LA_DIR="${HOME}/Library/LaunchAgents"
    LA_PATH="${LA_DIR}/com.iai-mcp.daemon.plist"
    TEMPLATE="${REPO_ROOT}/scripts/com.iai-mcp.daemon.plist.template"
    [ -f "${TEMPLATE}" ] || die "plist template missing at ${TEMPLATE}"
    mkdir -p "${LA_DIR}" "${HOME}/.iai-mcp/logs" "${HOME}/.iai-mcp"
    # Substitute placeholders using sed; HOME/PYTHON_PATH may contain forward
    # slashes so we use `|` as the sed separator (not `/`).
    sed -e "s|{PYTHON_PATH}|${PYTHON_PATH}|g" -e "s|{HOME}|${HOME}|g" "${TEMPLATE}" > "${LA_PATH}"
    if [ ! -f "${HOME}/.iai-mcp/.crypto.key" ] && [ -z "${IAI_MCP_CRYPTO_PASSPHRASE:-}" ]; then
        if "${REPO_ROOT}/.venv/bin/iai-mcp" crypto init >/dev/null 2>&1; then
            ok "crypto key generated (~/.iai-mcp/.crypto.key)"
        else
            warn "crypto init failed — run \`iai-mcp crypto init\` manually"
        fi
    fi
    # Idempotent: unload prior registration if any, then load fresh. -w persists across reboots.
    launchctl unload -w "${LA_PATH}" 2>/dev/null || true
    if ! launchctl load -w "${LA_PATH}"; then
        warn "launchctl load reported non-zero — checking registration anyway"
    fi
    if launchctl list | grep -q "com.iai-mcp.daemon"; then
        ok "LaunchAgent registered (daemon starts automatically at login)"
    else
        die "LaunchAgent NOT registered after launchctl load — investigate ${HOME}/.iai-mcp/logs/launchd-stderr.log"
    fi
fi

# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------
step "done"
ok "iai-mcp installed at $(git rev-parse --short HEAD)"
echo
echo "   next:   bash scripts/uninstall.sh    (to roll back; preserves data unless --purge-data)"
echo "   update: bash scripts/update.sh        (pull + rebuild + restart daemon)"
