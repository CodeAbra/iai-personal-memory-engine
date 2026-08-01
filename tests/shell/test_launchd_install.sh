#!/usr/bin/env bash
# macOS launchd install/uninstall idempotency.
#
# Verifies:
#   - DAEMON-01: plist installed under ~/Library/LaunchAgents
#   - DAEMON-10: silent install (--yes bypasses the consent banner)
#   - uninstall removes plist + ~/.iai-mcp/.lock +
#     ~/.iai-mcp/.daemon.sock + ~/.iai-mcp/.daemon-state.json
#   - Idempotency: install twice / uninstall twice -> no error
#
# Skipped on non-macOS (returns 0). Linux equivalent lives in
# tests/shell/test_systemd_install.sh.
#
# This script does NOT actually invoke launchctl in CI environments where it
# would fail (GitHub Actions macos-latest runners have launchd but no UI
# session for `gui/$UID` bootstrap to succeed). The CLI itself uses
# `check=False` on launchctl so a non-zero return there does not abort the
# install -- the plist file write + state file removal still happens.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
    echo "SKIP: not macOS"
    exit 0
fi

# Resolve which Python + iai-mcp module to use. Prefer venv, else system.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
    PY="$ROOT/.venv/bin/python"
else
    PY="${PYTHON:-python3}"
fi
CLI=( "$PY" -m iai_mcp.cli )

PLIST="$HOME/Library/LaunchAgents/com.iai-mcp.daemon.plist"
STATE_DIR="$HOME/.iai-mcp"
LOCK="$STATE_DIR/.lock"
SOCK="$STATE_DIR/.daemon.sock"
STATE="$STATE_DIR/.daemon-state.json"

# Snapshot pre-existing state so cleanup restores real user data.
# Backup directory in /tmp scoped to this run.
BACKUP_DIR="$(mktemp -d -t iai-mcp-shtest-XXXXXX)"
PRE_EXISTING_PLIST=0
PRE_EXISTING_LOCK=0
PRE_EXISTING_SOCK=0
PRE_EXISTING_STATE=0
if [[ -f "$PLIST" ]]; then
    PRE_EXISTING_PLIST=1
    cp "$PLIST" "$BACKUP_DIR/plist.bak"
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
    # Always restore the user's pre-existing state, even if the test failed.
    "${CLI[@]}" daemon uninstall --yes >/dev/null 2>&1 || true
    if [[ "$PRE_EXISTING_PLIST" == "1" ]]; then
        mkdir -p "$(dirname "$PLIST")"
        cp "$BACKUP_DIR/plist.bak" "$PLIST"
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
}
trap cleanup EXIT

# If the user already has a real plist installed, refuse to run -- this
# script would clobber their service state (separate from file restore).
if [[ "$PRE_EXISTING_PLIST" == "1" ]]; then
    echo "SKIP: existing plist at $PLIST -- not clobbering user data"
    exit 0
fi

echo "[1/6] First install (--yes bypasses consent banner)..."
"${CLI[@]}" daemon install --yes
if [[ ! -f "$PLIST" ]]; then
    echo "FAIL: plist not created at $PLIST"
    exit 1
fi
# Pitfall 5 sanity: rendered plist has absolute python path, not /usr/local/bin/python3
if ! grep -q "$PY" "$PLIST"; then
    echo "FAIL: plist does not contain absolute sys.executable ($PY)"
    cat "$PLIST"
    exit 1
fi

echo "[2/6] Second install -- must be idempotent..."
if ! "${CLI[@]}" daemon install --yes; then
    echo "FAIL: install #2 returned non-zero"
    exit 1
fi
if [[ ! -f "$PLIST" ]]; then
    echo "FAIL: plist missing after install #2"
    exit 1
fi

# Seed state files so we can verify C4 cleanup actually removes them.
mkdir -p "$STATE_DIR"
touch "$LOCK" "$SOCK"
echo "{}" > "$STATE"

echo "[3/6] First uninstall (C4: remove plist + 3 state files)..."
"${CLI[@]}" daemon uninstall --yes
if [[ -f "$PLIST" ]]; then
    echo "FAIL: plist not removed"
    exit 1
fi
# C4 invariant: lock + sock + state file all gone
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

echo "[4/6] Second uninstall -- must be idempotent (no error on missing files)..."
if ! "${CLI[@]}" daemon uninstall --yes; then
    echo "FAIL: uninstall #2 returned non-zero"
    exit 1
fi

echo "[5/6] Cross-platform: dry-run install on macOS prints plist..."
if ! "${CLI[@]}" daemon install --dry-run --yes | grep -q "com.iai-mcp.daemon"; then
    echo "FAIL: dry-run did not print plist content"
    exit 1
fi

echo "[6/6] Cross-platform: dry-run does NOT write plist..."
"${CLI[@]}" daemon install --dry-run --yes >/dev/null
if [[ -f "$PLIST" ]]; then
    echo "FAIL: dry-run wrote $PLIST -- it must be a no-write preview"
    exit 1
fi

echo "PASS: launchd install/uninstall idempotency + C4 + Pitfall 5"
exit 0
