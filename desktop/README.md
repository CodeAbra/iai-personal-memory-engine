# IAI Brain — desktop app

Native desktop window onto the brain dashboard (Tauri, Rust). Starts the
local `iai brain` server if one is not already running, then shows it.
Daemon start/stop/restart buttons live in the dashboard's Autonomic panel.

## Requirements

- The `iai` CLI installed (the app finds it via `$IAI_BIN`, the
  `~/.iai-mcp/.cli-path` cache, or common install locations).
- Rust toolchain per `src-tauri/rust-toolchain.toml` plus
  [`tauri-cli`](https://crates.io/crates/tauri-cli) (`cargo install tauri-cli --locked`).

## Build

```bash
cd desktop/src-tauri
cargo tauri build --config tauri.local.conf.json
```

`tauri.local.conf.json` is a git-ignored overlay carrying the machine's
signing identity (`{"bundle": {"macOS": {"signingIdentity": "..."}}}`).
Signing with a STABLE identity matters: an ad-hoc-signed rebuild gets a new
code identity, macOS revokes the app's folder-access grants, and the spawned
server blocks on a permission prompt every deploy. Without the overlay
(plain `cargo tauri build`) the bundle is ad-hoc signed — expect macOS to
re-ask for folder access after each install.

macOS output: `target/release/bundle/macos/IAI Brain.app` (+ `.dmg`).

Linux output: `.deb` / `.AppImage` under `target/release/bundle/`.
System packages needed before building:

```bash
# Debian/Ubuntu
sudo apt install libwebkit2gtk-4.1-dev build-essential curl wget file \
  libxdo-dev libssl-dev librsvg2-dev
# Fedora
sudo dnf install webkit2gtk4.1-devel openssl-devel curl wget file \
  libappindicator-gtk3-devel librsvg2-devel
```

## Environment

- `IAI_BRAIN_PORT` — dashboard port (default 4477).
- `IAI_BIN` — explicit path to the `iai` executable.

Closing the window stops the dashboard server the app itself started; the
sleep daemon is not affected — memory works with or without it.
