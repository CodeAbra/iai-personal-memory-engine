#!/usr/bin/env bash
# scripts/bootstrap.sh — install iai-pme from nothing, in one command.
#
#   curl -fsSL https://raw.githubusercontent.com/CodeAbra/iai-personal-memory-engine/main/scripts/bootstrap.sh | bash
#
# Clones the repo, then runs scripts/install.sh (venv, package, MCP wrapper,
# CLI symlink, background engine), wires ambient capture, registers the MCP
# server with Claude Code, and finishes with a health check.
#
# Options:
#   --dir PATH         where to keep the source tree (default ~/.local/share/iai-pme)
#   --branch NAME      branch to install (default main)
#   --preflight-only   check prerequisites and exit, changing nothing
#   --dry-run          print every step that would run, changing nothing
#
# Environment: IAI_SRC_DIR, IAI_BRANCH, IAI_REPO_URL override the defaults.
# Idempotent: re-running updates an existing install in place.

set -euo pipefail

REPO_URL="${IAI_REPO_URL:-https://github.com/CodeAbra/iai-personal-memory-engine.git}"
SRC_DIR="${IAI_SRC_DIR:-${HOME}/.local/share/iai-pme}"
BRANCH="${IAI_BRANCH:-main}"
PREFLIGHT_ONLY=0
DRY_RUN=0

while [ $# -gt 0 ]; do
    case "$1" in
        --dir) SRC_DIR="${2:?--dir needs a path}"; shift 2 ;;
        --branch) BRANCH="${2:?--branch needs a name}"; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,19p' "$0" | cut -c3-; exit 0 ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
ok()   { printf '   \033[0;32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[0;33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[0;31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
run()  {
    if [ "${DRY_RUN}" = "1" ]; then
        printf '   \033[0;36m→\033[0m would run: %s\n' "$*"
    else
        "$@"
    fi
}

DOCS_URL="https://github.com/CodeAbra/iai-personal-memory-engine#quick-start"

# ---------------------------------------------------------------------------
# 1. prerequisites
# ---------------------------------------------------------------------------
step "checking prerequisites"

OS="$(uname -s)"
case "${OS}" in
    Darwin) ok "macOS $(uname -m)" ;;
    Linux)  ok "Linux $(uname -m)" ;;
    *) die "unsupported platform: ${OS}. Windows install is manual — see ${DOCS_URL}" ;;
esac

command -v git >/dev/null 2>&1 || die "git not found. Install it, then re-run."
ok "git $(git --version | awk '{print $3}')"

# The package supports 3.11 and 3.12 only; find one of those, whatever it's called.
PYTHON=""
for candidate in python3.12 python3.11 python3; do
    if command -v "${candidate}" >/dev/null 2>&1; then
        if "${candidate}" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)' 2>/dev/null; then
            PYTHON="$(command -v "${candidate}")"
            break
        fi
    fi
done
[ -n "${PYTHON}" ] || die "need Python 3.11 or 3.12 (found: $(python3 --version 2>&1 || echo none)). See ${DOCS_URL}"
ok "$("${PYTHON}" --version) at ${PYTHON}"

command -v node >/dev/null 2>&1 || die "Node.js 18+ not found — the MCP wrapper is TypeScript. See ${DOCS_URL}"
NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]')"
[ "${NODE_MAJOR}" -ge 18 ] || die "Node.js ${NODE_MAJOR} is too old; 18+ required."
command -v npm >/dev/null 2>&1 || die "npm not found (it ships with Node.js)."
ok "Node $(node -v)"

# No prebuilt wheels are published yet, so the native engine always compiles here.
if command -v cargo >/dev/null 2>&1; then
    ok "Rust $(cargo --version | awk '{print $2}')"
else
    die "Rust toolchain not found — the native engine builds from source.
     Install it with:  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
     then open a new shell and re-run this script."
fi

if [ "${PREFLIGHT_ONLY}" = "1" ]; then
    step "preflight only"
    ok "all prerequisites satisfied — nothing was changed"
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. source tree
# ---------------------------------------------------------------------------
step "source tree at ${SRC_DIR}"
if [ -d "${SRC_DIR}/.git" ]; then
    run git -C "${SRC_DIR}" fetch --quiet origin "${BRANCH}"
    run git -C "${SRC_DIR}" checkout --quiet "${BRANCH}"
    if [ "${DRY_RUN}" = "1" ]; then
        printf '   \033[0;36m→\033[0m would run: git -C %s merge --ff-only origin/%s\n' "${SRC_DIR}" "${BRANCH}"
    elif git -C "${SRC_DIR}" merge --ff-only "origin/${BRANCH}" >/dev/null 2>&1; then
        ok "updated to $(git -C "${SRC_DIR}" rev-parse --short HEAD)"
    else
        warn "local changes in ${SRC_DIR} — keeping them, skipping the update"
    fi
elif [ -e "${SRC_DIR}" ]; then
    die "${SRC_DIR} exists but is not a git clone. Move it aside, or pass --dir PATH."
else
    run mkdir -p "$(dirname "${SRC_DIR}")"
    run git clone --quiet --depth 1 --branch "${BRANCH}" "${REPO_URL}" "${SRC_DIR}"
    ok "cloned ${BRANCH}"
fi

# ---------------------------------------------------------------------------
# 3. build and install (delegated — install.sh owns venv, package, wrapper,
#    CLI symlink and background-engine registration)
# ---------------------------------------------------------------------------
step "building (a few minutes: the Rust engine compiles from source)"
if [ "${DRY_RUN}" = "1" ]; then
    printf '   \033[0;36m→\033[0m would run: IAI_PYTHON=%s bash %s/scripts/install.sh\n' "${PYTHON}" "${SRC_DIR}"
else
    [ -f "${SRC_DIR}/scripts/install.sh" ] || die "scripts/install.sh missing from the clone"
    IAI_PYTHON="${PYTHON}" bash "${SRC_DIR}/scripts/install.sh"
fi

IAI_BIN="${SRC_DIR}/.venv/bin/iai-mcp"

# ---------------------------------------------------------------------------
# 4. ambient capture
# ---------------------------------------------------------------------------
step "ambient capture hooks"
if [ "${DRY_RUN}" = "1" ]; then
    printf '   \033[0;36m→\033[0m would run: %s capture-hooks install\n' "${IAI_BIN}"
elif [ -x "${IAI_BIN}" ]; then
    if "${IAI_BIN}" capture-hooks install; then
        ok "capture and recall hooks installed"
    else
        warn "hook install failed — run \`iai-mcp capture-hooks install\` yourself"
    fi
else
    warn "CLI not found at ${IAI_BIN} — skipping hooks"
fi

# ---------------------------------------------------------------------------
# 5. register the MCP server with Claude Code
# ---------------------------------------------------------------------------
step "MCP host registration"
WRAPPER="${SRC_DIR}/mcp-wrapper/dist/index.js"
if [ "${DRY_RUN}" = "1" ]; then
    printf '   \033[0;36m→\033[0m would run: claude mcp add iai-mcp -- node %s\n' "${WRAPPER}"
elif ! command -v claude >/dev/null 2>&1; then
    warn "the \`claude\` CLI is not installed — register manually with:"
    warn "  claude mcp add iai-mcp -- node ${WRAPPER}"
elif [ ! -f "${WRAPPER}" ]; then
    warn "wrapper not built at ${WRAPPER} — see the build output above"
elif claude mcp list 2>/dev/null | grep -q "iai-mcp"; then
    ok "already registered with Claude Code"
elif claude mcp add iai-mcp -- node "${WRAPPER}"; then
    ok "registered with Claude Code"
else
    warn "registration failed — run it yourself:"
    warn "  claude mcp add iai-mcp -- node ${WRAPPER}"
fi

# ---------------------------------------------------------------------------
# 6. health check
# ---------------------------------------------------------------------------
step "health check"
if [ "${DRY_RUN}" = "1" ]; then
    printf '   \033[0;36m→\033[0m would run: %s doctor\n' "${IAI_BIN}"
elif [ -x "${IAI_BIN}" ]; then
    "${IAI_BIN}" doctor || warn "doctor reported problems — read the lines above"
else
    warn "CLI not found at ${IAI_BIN} — skipping"
fi

# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------
step "done"
echo
echo "   Restart your assistant, then work normally — capture and recall are automatic."
echo
echo "   watch the brain:  iai brain"
echo "   check anytime:    iai-mcp doctor"
echo "   update:           bash ${SRC_DIR}/scripts/update.sh"
echo "   remove:           bash ${SRC_DIR}/scripts/uninstall.sh"
echo
if ! printf '%s' ":${PATH}:" | grep -q ":${HOME}/.local/bin:"; then
    warn "add ~/.local/bin to your PATH so \`iai\` and \`iai-mcp\` work everywhere:"
    warn "  export PATH=\"\${HOME}/.local/bin:\${PATH}\""
fi
