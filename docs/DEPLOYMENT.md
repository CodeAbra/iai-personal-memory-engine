# iai-mcp Deployment

**Audience:** Self-hosters running iai-mcp locally on macOS or Linux as a
personal memory layer for Claude Code, Claude Desktop, or another MCP host.
For the architecture overview see [`README.md`](../README.md); for the
per-release log see [`CHANGELOG.md`](../CHANGELOG.md).

## Requirements

| Requirement | Version / notes |
|---|---|
| Python | 3.11–3.12 (CPython) |
| OS | macOS or Linux. The daemon uses `fcntl.flock` and Unix-socket IPC. Windows support is in beta. WSL2 works as a Linux target. |
| RAM | 8+ GB comfortable. The `bge-small-en-v1.5` embedder occupies ~600 MB resident once loaded. |
| Disk | ~5 GB free for model weights + store + WAL. Model weights live in `~/.cache/huggingface/` (~130 MB). |
| Toolchain (source build only) | A Rust toolchain is needed when compiling the native extension from source. On Linux, `libssl-dev` and `pkg-config` (or your distro's equivalents). |

The native extension (`iai_mcp_native` — the embedder, graph algorithms, and
storage engine) is compiled during `pip install` when a prebuilt wheel is not
available for your platform.

## Install

```bash
pip install iai-mcp
```

Or from a checkout, for development:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

`scripts/install.sh` is the canonical installer. It is idempotent and safe to
re-run:

```bash
bash scripts/install.sh
```

The installer creates the venv if missing, installs `iai-mcp`, builds the
native Rust extension and the TypeScript MCP wrapper, symlinks the CLI onto
your `PATH`, and optionally installs the sleep daemon (launchd on macOS,
systemd user unit on Linux).

## Register with an MCP host

After install, register iai-mcp with your host. For Claude Code:

```bash
claude mcp add iai-mcp \
  --command node \
  --args "$(pwd)/mcp-wrapper/dist/index.js" \
  --env IAI_MCP_PYTHON="$(pwd)/.venv/bin/python" \
  --env IAI_MCP_STORE="$HOME/.iai-mcp" \
  --env TRANSFORMERS_VERBOSITY=error \
  --env TOKENIZERS_PARALLELISM=false
```

Restart the host and ask it to list MCP tools — you should see the iai-mcp
memory tools (see the tools table in `README.md`).

## Capture hooks (ambient memory)

```bash
iai-mcp capture-hooks install     # SessionStart + UserPromptSubmit + Stop
iai-mcp capture-hooks status      # verify all three are active
iai-mcp capture-hooks uninstall   # remove them (preserves ~/.iai-mcp/)
```

After install you never need to say "save" / "recall" / "remember". The Stop
hook captures the session transcript on exit; the UserPromptSubmit hook
appends each prompt plus the preceding assistant turn to a deferred-capture
buffer; the SessionStart hook injects the daemon's pre-cached session-start
payload as `additionalContext` before the first prompt of every session.

## Desktop dashboard

A local, loopback-only dashboard to watch the memory graph live, add a
memory, or hint one into the forgetting queue:

```bash
iai brain          # serve the dashboard (127.0.0.1, daemon-independent)
```

It binds an ephemeral port and opens in your browser; reads go to the awake
store, writes through the capture spine. There is no delete — a forget hint
lets a record age out through the normal decay path, and pinned records
refuse the hint.

## Encryption

Records are written under AES-256-GCM. The key lives in
`~/.iai-mcp/.crypto.key` and is auto-generated on first run. To use a
passphrase-derived key instead:

```bash
export IAI_MCP_CRYPTO_PASSPHRASE='<your-passphrase>'
iai-mcp crypto init
```

`IAI_MCP_CRYPTO_PASSPHRASE` is the documented fallback when the key file is
missing or unreadable.

## Daemon lifecycle

```bash
iai-mcp daemon status     # FSM state, last heartbeat, socket health
iai-mcp daemon logs --tail 20
iai-mcp daemon force-rem  # manually trigger a consolidation cycle
iai-mcp daemon install    # launchd / systemd install (idempotent)
iai-mcp daemon start
iai-mcp daemon stop
iai-mcp doctor            # full health audit
```

## Troubleshooting

**Daemon won't start / "socket not bound".** Run `iai-mcp daemon status`, then
`iai-mcp daemon logs --tail 50`, then `iai-mcp doctor`. A common cause is a
stale lock file: the lifecycle lock cross-checks the holding PID's command
line and rejects a recycled PID automatically, but if a stale
`~/.iai-mcp/.locked` remains, delete it and retry `iai-mcp daemon start`.

**`CryptoKeyError` on start.** Check `iai-mcp crypto status` and that
`~/.iai-mcp/.crypto.key` exists and is mode `0600`. If it is missing or
corrupted, regenerate from a passphrase:

```bash
export IAI_MCP_CRYPTO_PASSPHRASE='<your-passphrase>'
iai-mcp crypto init
iai-mcp daemon start
```

Regenerating a fresh key on a non-empty store renders previously-encrypted
records unreadable — restore from a backup first if there is data to keep.

**Wrapper reports "daemon degraded".** The wrapper probes `iai-mcp doctor` on
session start and warns when the daemon is degraded. MCP calls still work
against the recent-memory transit window; nightly consolidation resumes once
the daemon is healthy.

**`iai-mcp doctor` red on a headless host.** `iai-mcp doctor --headless` (auto-
detected when `DISPLAY` and `WAYLAND_DISPLAY` are both absent on Linux)
downgrades HID-idle and display-dependent rows from FAIL to WARN.

## Uninstall

```bash
bash scripts/uninstall.sh           # removes the service + CLI symlink, preserves ~/.iai-mcp/
bash scripts/uninstall.sh --purge   # also deletes ~/.iai-mcp/
```

Run `iai-mcp capture-hooks uninstall` before `--purge` so the host does not
hit stale hook references on the next session.
