#!/usr/bin/env bash
# Linux systemd install/uninstall idempotency.
#
# Verifies:
#   - DAEMON-01: unit installed under ~/.config/systemd/user
#   - DAEMON-10: silent install (--yes bypasses the consent banner)
#   - cleanup: uninstall removes unit + ~/.iai-mcp/.lock +
#     ~/.iai-mcp/.daemon.sock + ~/.iai-mcp/.daemon-state.json
#   - Idempotency: install twice / uninstall twice -> no error
#
# Skipped on non-Linux (returns 0). macOS equivalent lives in
# tests/shell/test_launchd_install.sh.
#
# Skipped if systemctl --user is not usable (headless CI without an active
# user-systemd session, e.g. GitHub Actions ubuntu-latest by default).
# DAEMON-12 cross-platform parity is enforced by CI matrix; this script is
# a smoke test that runs FULL flow when a user session exists.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "SKIP: not Linux"
    exit 0
fi

# Skip on CI without user systemd session.
if ! systemctl --user status >/dev/null 2>&1; then
    echo "SKIP: no user systemd session available (expected on headless CI without loginctl enable-linger)"
    exit 0
fi

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi
CLI=( "$PY" -m iai_mcp.cli )

UNIT="$HOME/.config/systemd/user/iai-mcp-daemon.service"
STATE_DIR="$HOME/.iai-mcp"
LOCK="$STATE_DIR/.lock"
SOCK="$STATE_DIR/.daemon.sock"
STATE="$STATE_DIR/.daemon-state.json"

BACKUP_DIR="$(mktemp -d -t iai-mcp-shtest-XXXXXX)"
PRE_EXISTING_UNIT=0
PRE_EXISTING_LOCK=0
PRE_EXISTING_SOCK=0
PRE_EXISTING_STATE=0
if [[ -f "$UNIT" ]]; then
    PRE_EXISTING_UNIT=1
    cp "$UNIT" "$BACKUP_DIR/unit.bak"
fi
if [[ -f "$LOCK" ]]; then
    PRE_EXISTING_LOCK=1
    cp "$LOCK" "$BACKUP_DIR/lock.bak"
fi
if [[ -f "$SOCK" ]]; then
    PRE_EXISTING_SOCK=1
    cp "$SOCK" "$BACKUP_DIR/sock.bak" 2>/dev/null || true
fi
if [[ -f "$STATE" ]]; then
    PRE_EXISTING_STATE=1
    cp "$STATE" "$BACKUP_DIR/state.bak"
fi

cleanup() {
    "${CLI[@]}" daemon uninstall --yes >/dev/null 2>&1 || true
    if [[ "$PRE_EXISTING_UNIT" == "1" ]]; then
        mkdir -p "$(dirname "$UNIT")"
        cp "$BACKUP_DIR/unit.bak" "$UNIT"
    fi
    mkdir -p "$STATE_DIR"
    if [[ "$PRE_EXISTING_LOCK" == "1" ]]; then
        cp "$BACKUP_DIR/lock.bak" "$LOCK"
    fi
    if [[ "$PRE_EXISTING_SOCK" == "1" && -f "$BACKUP_DIR/sock.bak" ]]; then
        cp "$BACKUP_DIR/sock.bak" "$SOCK" 2>/dev/null || true
    fi
    if [[ "$PRE_EXISTING_STATE" == "1" ]]; then
        cp "$BACKUP_DIR/state.bak" "$STATE"
    fi
    rm -rf "$BACKUP_DIR"
    systemctl --user daemon-reload >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ "$PRE_EXISTING_UNIT" == "1" ]]; then
    echo "SKIP: existing unit at $UNIT -- not clobbering user data"
    exit 0
fi

echo "[1/6] First install (--yes bypasses consent banner)..."
"${CLI[@]}" daemon install --yes
if [[ ! -f "$UNIT" ]]; then
    echo "FAIL: unit not created at $UNIT"
    exit 1
fi
# Pitfall 5 sanity: rendered unit has absolute python path
if ! grep -q "$PY" "$UNIT"; then
    echo "FAIL: unit does not contain absolute sys.executable ($PY)"
    cat "$UNIT"
    exit 1
fi

echo "[2/6] Verify systemctl shows the unit as enabled..."
if ! systemctl --user is-enabled iai-mcp-daemon.service 2>/dev/null | grep -q enabled; then
    echo "WARN: unit not enabled (may be expected on minimal CI sessions)"
fi

echo "[3/6] Second install -- must be idempotent..."
if ! "${CLI[@]}" daemon install --yes; then
    echo "FAIL: install #2 returned non-zero"
    exit 1
fi
if [[ ! -f "$UNIT" ]]; then
    echo "FAIL: unit missing after install #2"
    exit 1
fi

# Seed state files so we can verify C4 cleanup actually removes them.
mkdir -p "$STATE_DIR"
touch "$LOCK" "$SOCK"
echo "{}" > "$STATE"

echo "[4/6] First uninstall (C4: remove unit + 3 state files)..."
"${CLI[@]}" daemon uninstall --yes
if [[ -f "$UNIT" ]]; then
    echo "FAIL: unit not removed"
    exit 1
fi
if [[ -f "$LOCK" ]]; then
    echo "FAIL: lock file not removed (C4 violation)"
    exit 1
fi
if [[ -f "$SOCK" ]]; then
    echo "FAIL: socket file not removed (C4 violation)"
    exit 1
fi
if [[ -f "$STATE" ]]; then
    echo "FAIL: state file not removed (C4 violation)"
    exit 1
fi

echo "[5/6] Second uninstall -- must be idempotent..."
if ! "${CLI[@]}" daemon uninstall --yes; then
    echo "FAIL: uninstall #2 returned non-zero"
    exit 1
fi

echo "[6/6] Dry-run on Linux prints unit content + does NOT write..."
"${CLI[@]}" daemon install --dry-run --yes | grep -q "iai_mcp.daemon" || {
    echo "FAIL: dry-run did not print unit content"
    exit 1
}
if [[ -f "$UNIT" ]]; then
    echo "FAIL: dry-run wrote $UNIT -- it must be a no-write preview"
    exit 1
fi

echo "PASS: systemd install/uninstall idempotency + C4 + Pitfall 5"
exit 0
