# Linux port probe — `iai-mcp` on Debian 13

Built the **unmodified** fork from source on Linux with the system GNU toolchain.
**It works:** the native Rust engine builds, loads, and embeds; the systemd daemon
runs; and after installing the daemon `iai-mcp doctor` reports **24 PASS / 1 WARN /
0 FAIL**. A live capture→recall round-trip through the MCP tools succeeded.

## Environment

| | |
|---|---|
| OS | Debian GNU/Linux 13 (trixie), x86_64, inside an LXC |
| Python | 3.12.13 (Debian 13 ships only 3.13, which the project pins out: `requires-python = ">=3.11,<3.13"`) |
| Rust | 1.96.0 (rustup, default stable) |
| C toolchain | gcc 14.2.0, make 4.4.1, pkg-config, OpenSSL 3.5.6 |
| Node | 22.x |

## Required Linux packages

On Debian/Ubuntu:

```bash
apt-get install -y build-essential pkg-config libssl-dev
```

Plus a **Python 3.11 or 3.12** interpreter (Debian 13 packages only 3.13).

Why each is needed (verified during the build):

- **`build-essential`** (gcc/g++/make/ld) — `tokenizers 0.23` is pulled with its
  default features, which compile C/C++ (`onig_sys`, `esaxx-rs`); `ring` (in the
  TLS stack) compiles C as well.
- **`pkg-config` + `libssl-dev`** — `hf-hub`'s default `default-tls =
  ["native-tls"]` resolves to `openssl-sys` on Linux, which needs `pkg-config` and
  OpenSSL headers. (On macOS `native-tls` uses the Security.framework, so this
  isn't hit there.)

## Build

With those packages present, `pip install .` builds the native Rust engine
(`iai_mcp_native`) via `setuptools-rust` with **no source changes**. The compiled
extension imports cleanly (`iai_mcp_native.cpython-312-x86_64-linux-gnu.so`).

> Optional dependency cleanup (not required): pinning
> `hf-hub = { default-features = false, features = ["ureq"] }` in `rust/Cargo.toml`
> drops the `native-tls`/`openssl-sys` + `reqwest`/`tokio` pull-in (the manifest
> comment already calls `ureq` "the blocking HTTP backend"), which would let the
> build succeed without `libssl-dev`/`pkg-config` and trim the dependency tree on
> macOS too.

## `iai-mcp doctor` (daemon installed and running)

```
24 PASS / 1 WARN / 0 FAIL
[PASS] (h) crypto key file state   file-backed key present (mode 0o600)
[PASS] (o) Claude subscription     valid team subscription with inference scope
[PASS] (p) anthropic SDK absent    ImportError as expected (subscription-only path)
[PASS] (v) native Rust embedder    encode ok, backend=rust, 384-dim
[PASS] (z) AVX2 CPU support        AVX2 available
[WARN] (n) HID idle source         HIDIdleTime/pmset unavailable → falls back to heartbeat-idle
... (remaining 19 checks PASS)
```

Note: the first `(v)` embedder run downloads the bge-small-en-v1.5 model from
Hugging Face, so it requires network on first encode.

## macOS assumptions observed — behavior on Linux

| Area | macOS path | On Linux | Notes |
|---|---|---|---|
| Credential store | Keychain | File-backed key at `~/.iai-mcp/.crypto.key` | works; created by `daemon install` / `crypto init` |
| Idle detection `(n)` | IOKit `HIDIdleTime` / `pmset` | Unavailable → falls back to heartbeat-idle | the single remaining WARN; graceful |
| Service manager | launchd plist | systemd user unit | works (see below) |

The codebase already carries Linux paths: `_is_linux()`, systemd unit handling,
`/proc/cpuinfo` AVX2 probing (`cpu_features.py`), and non-root-Linux `/proc/<pid>/fd`
fallbacks in the doctor checks.

## systemd daemon — works

`iai-mcp daemon install --yes` on Linux:

- Renders the systemd user unit with the running venv interpreter substituted into
  `ExecStart` (not the template's static `/usr/bin/python3`); writes
  `~/.config/systemd/user/iai-mcp-daemon.service`.
- Runs `loginctl enable-linger`, `systemctl --user daemon-reload`, `enable --now`.
  The daemon comes up `active (running)` and binds its socket.
- With the daemon up, doctor reports 24 PASS / 1 WARN / 0 FAIL.

**Unit-template bug** (`_deploy/systemd/iai-mcp-daemon.service`): the journal logs
`Unknown key 'StartLimitIntervalSec' in section [Service], ignoring`.
`StartLimitIntervalSec` and `StartLimitBurst` belong in the `[Unit]` section, not
`[Service]`, on current systemd — as written, the restart rate-limit is ignored.
Fix: move both keys to `[Unit]`. (Same template on macOS-vs-Linux; affects the
Linux unit.)

Heads-up for users: install sets `Linger=yes` (daemon survives logout) and enables
the unit at boot. Removal: `iai-mcp daemon uninstall --yes` (does not disable
linger — `loginctl disable-linger $USER` separately if desired).

## MCP wrapper packaging bug (not Linux-specific)

`iai-mcp capture-hooks install` registers the MCP server in `~/.claude.json`
pointing at the staged wrapper inside the wheel
(`…/site-packages/iai_mcp/_wrapper/index.js`). `setup.py` builds that wrapper with
`npm run build` (plain `tsc`, unbundled) and stages **`*.js` only — no
`node_modules`**. `_resolve_wrapper_path()` prefers this staged copy over the repo
`mcp-wrapper/dist/index.js`. Result: `node …/_wrapper/index.js` fails with
`ERR_MODULE_NOT_FOUND: Cannot find package '@modelcontextprotocol/sdk'`, so the
MCP server doesn't start.

Scope (verified): occurs on a non-editable `pip install .` (the staged copy is
present and preferred). Editable installs (`pip install -e .`) skip npm-staging in
`setup.py`, so the resolver falls through to `mcp-wrapper/dist/index.js`, which has
its `node_modules`. The capture/recall *hooks* are unaffected (they run on system
`python3` + the daemon socket, not the wrapper); only the MCP **tools**
(`memory_recall`, `memory_capture`, …) depend on the wrapper.

Fix options (any one):
- Register a wrapper that has its deps alongside it (`mcp-wrapper/dist/index.js`
  after `npm install`) — done here; verified `initialize` + `tools/list` over
  stdio return the full toolset.
- Bundle the wrapper to one self-contained file (esbuild `--bundle
  --platform=node`) and ship that as the staged `_wrapper/index.js`.
- Vendor `node_modules` into the wheel's `_wrapper/`.

## README manual steps vs `scripts/install.sh`

We installed by following the README's individual commands (`pip install .`,
`capture-hooks install`, `daemon install`) rather than `scripts/install.sh`.
`scripts/install.sh` additionally:

- **symlinks `~/.local/bin/iai-mcp -> .venv/bin/iai-mcp`** (its step 4), putting the
  CLI on `PATH`. The capture/recall hooks resolve `iai-mcp` via `PATH` / the shared
  `~/.iai-mcp/.cli-path` cache the capture hook writes on first resolution; without
  the symlink and with the venv off `PATH`, the hooks log `iai-mcp CLI not found`.
- **runs `iai-mcp daemon install --yes`** (its step 5), starting the daemon.

A manual README install must perform those two steps itself. On this box we
reproduced them by symlinking `iai-mcp`/`iai` into `~/.local/bin` (and writing the
venv path to `~/.iai-mcp/.cli-path`) and running `daemon install`.

## Status on this box

Daemon (systemd) + capture/recall hooks + MCP tools are installed and working.
Verified a live capture→recall round-trip via the MCP tools (semantic match on a
zero-keyword-overlap cue, cosine 0.57, ~95 ms, hnswlib ANN path). The MCP server
loads after a Claude Code restart (`/mcp`).
