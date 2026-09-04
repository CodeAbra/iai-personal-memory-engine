# iai Deployment

**Version:** v7.1
**Audience:** Self-hosters running iai locally on macOS or Linux.
**Last updated:** 2026-05-20

This document covers the practical setup for running iai locally as
the operator's personal memory layer for Claude Code / Claude Desktop /
other MCP hosts. For the architectural overview, see `README.md`. For the
empirical-falsifiability stance and ship-gate evidence, see
`FALSIFIABILITY_REPORT.md`. For the per-release shipping log, see
`CHANGELOG.md`.

## Hardware requirements

- **AVX2 is NOT required (post-v8.0).** The storage backend is now
  **Hippo** (SQLite + hnswlib), which has no AVX2 dependency —
  `iai_mcp.store.CPU_HAS_AVX2` is hardcoded `True`. The legacy AVX2
  hard-requirement was a LanceDB Rust-core constraint and is gone since
  the LanceDB → Hippo storage move. iai runs on any modern Intel / AMD /
  Apple-silicon host. The `iai-mcp doctor` row `(z) AVX2 CPU support`
  is retained as an informational check (PASS, or N/A on ARM) — it no
  longer gates installation. LanceDB only enters via the optional
  `pip install -e ".[migration]"` extra, used once to migrate a legacy
  store into Hippo; that one-time path is the only place AVX2 still
  matters, and only on the migrating host.
- **RAM:** 8+ GB comfortable (16+ GB if running parallel test executors).
  The `bge-small-en-v1.5` embedder occupies ~600 MB resident once loaded;
  the Hippo backend (SQLite + hnswlib) adds far less than the old LanceDB
  native layer. Post-Hippo `bench/memory_footprint.py` at `N = 1000`
  dropped to ~292 MB RSS (was ~1.4 GB under LanceDB — a 4.5× reduction),
  and `N = 10000` completes in ~3.5 min / ~611 MB (was a 40+ min hang).
  The production daemon runs at ~500 MB–1.5 GB idle and ~2 GB during
  nightly REM consolidation.
- **Disk:** ~5 GB free for the model weights + Hippo store + WAL.
  Model weights live in `~/.cache/huggingface/` (~ 130 MB);
  the Hippo store (`~/.iai-mcp/hippo/brain.sqlite3` + hnswlib index)
  grows ≈ 1 MB per 100 records steady-state after compaction.
- **OS:** macOS or Linux. The daemon uses `fcntl.flock` and Unix-socket
  IPC; there is no Windows port. WSL2 works as a Linux target.

## Python venv preparation

Python 3.11–3.12 supported. Python 3.13 currently breaks some torch
wheels — pin to 3.12.

```bash
git clone <your-iai-mcp-fork>.git iai-mcp
cd iai-mcp
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
```

### Build the native extension (required)

The `iai_mcp_native` Rust extension is the sole embed and graph runtime — the
daemon refuses to start without it. Build it once after install (and after any
`git pull` that changes the Rust source):

```bash
cd rust/iai_mcp_native && maturin develop --release && cd ../..
```

`maturin` must be installed (it is included in `.[dev]`). The build compiles the
BERT embedder and graph algorithms as a single shared library in the active venv.
No internet access needed after the initial HuggingFace model cache is populated.

```bash
pytest                              # 2100+ tests; expect 4 skipped / 1 legacy fail
```

### Worktree editable-install gotcha (DEF-25-A predecessor)

If the operator uses `git worktree` to run multiple branches in parallel,
**do not create per-worktree venvs for the production daemon.** A
`pip install -e .` from worktree A's `.venv` registers an editable install
pointing at worktree A's `src/iai_mcp/`. Switching to worktree B and
invoking the bench / daemon from B's checkout can still import worktree
A's source, because the editable-install registration is per-venv and
the venv lookup walks up from the running script. Symptoms include "I
fixed the bug in worktree B but the bench still hits the old code path".

**Workaround in the v7.1 bench harness**: every script in
`bench/` includes a defensive `sys.path` insertion that resolves to the
script's own worktree-local `src/iai_mcp/` before falling back to the
editable install. Production code (the daemon, the CLI, the MCP wrapper)
does not need this shim because it always runs from a single canonical
checkout via the `~/.local/bin/iai-mcp` symlink created by
`scripts/install.sh`.

**Recommended:** one main checkout + one `.venv` for the daemon; use
worktrees for experimentation; run benches with the shim defaults; never
swap the daemon between worktrees mid-session.

## Daemon install (launchd / systemd)

`scripts/install.sh` is the canonical installer. It is idempotent and
safe to re-run.

```bash
bash scripts/install.sh
```

The installer:

1. Creates `.venv` if missing.
2. Installs `iai-mcp` editable into the venv.
3. Builds the native Rust extension: `cd rust/iai_mcp_native && maturin develop --release`.
   This step is **required** — `iai_mcp_native` is the sole embed and graph runtime.
4. Builds the TypeScript MCP wrapper (`cd mcp-wrapper && npm install &&
   npm run build`).
5. Symlinks `~/.local/bin/iai-mcp → .venv/bin/iai-mcp` so the CLI is
   callable from anywhere without activating the venv.
6. Optionally installs the sleep daemon: **launchd** on macOS (via
   `scripts/com.iai-mcp.daemon.plist.template`), **systemd** on Linux
   (via `src/iai_mcp/_deploy/systemd/`).

### Crypto initialisation

The daemon writes records under AES-256-GCM. The encryption key lives in
`~/.iai-mcp/.crypto.key` and is auto-generated by
`iai-mcp daemon install` on first run. Idempotent. If the operator
prefers a passphrase-derived key:

```bash
export IAI_MCP_CRYPTO_PASSPHRASE='<your-passphrase>'
iai-mcp crypto init
```

`IAI_MCP_CRYPTO_PASSPHRASE` is the documented fallback when the key file
is missing or unreadable.

### Daemon lifecycle commands

```bash
iai-mcp daemon status               # FSM state, last heartbeat, socket health
iai-mcp daemon logs --tail 20       # recent daemon activity
iai-mcp daemon force-rem            # manually trigger an REM cycle
iai-mcp daemon stats                # session_start_tokens_p90 + counts
iai-mcp daemon install              # launchd / systemd install (idempotent)
iai-mcp daemon start                # explicit start
iai-mcp daemon stop                 # graceful stop
iai-mcp health                      # last llm_health event
iai-mcp crypto status               # verify key backend (keychain / file / passphrase)
iai-mcp doctor                      # full health audit (AVX2, socket, FSM, etc.)
```

### Capture hooks (ambient memory)

`iai-mcp capture-hooks install` wires three hooks into
`~/.claude/settings.json`:

```bash
iai-mcp capture-hooks install       # SessionStart + UserPromptSubmit + Stop
iai-mcp capture-hooks status        # verify all three are active
iai-mcp capture-hooks uninstall     # remove all three (preserves ~/.iai-mcp/)
```

After install, the operator never needs to say "save" / "recall" / "remember".
The Stop hook batch-captures the session transcript on exit; the
UserPromptSubmit hook appends each prompt + the preceding assistant turn
to a deferred-captures buffer; the SessionStart hook reads the daemon's
pre-cached session-start payload and injects it as `additionalContext`
before the first prompt of every session.

## MCP host registration

After `scripts/install.sh` completes, register iai with the operator's
MCP host. For Claude Code:

```bash
claude mcp add iai-mcp \
  --command node \
  --args "$(pwd)/mcp-wrapper/dist/index.js" \
  --env IAI_MCP_PYTHON="$(pwd)/.venv/bin/python" \
  --env IAI_MCP_STORE="$HOME/.iai-mcp" \
  --env TRANSFORMERS_VERBOSITY=error \
  --env TOKENIZERS_PARALLELISM=false
```

Restart Claude Code. Ask "list MCP tools" — expect 12 iai-mcp tools (see
`README.md`'s MCP tools table).

## HF_TOKEN setup (bench rerun)

`bench/longmemeval_blind.py` downloads the LongMemEval-S dataset from
HuggingFace and requires a personal access token even though the dataset
is public.

```bash
export HF_TOKEN='<your-huggingface-token>'
# or persist to your shell rc
echo 'export HF_TOKEN="<your-huggingface-token>"' >> ~/.zshrc
```

Then:

```bash
source .venv/bin/activate
python bench/longmemeval_blind.py --rows 500
```

The bench resumes from a JSONL checkpoint if a prior run was killed (the
run was killed at row 339/500 and the next run picks up from
row 340 automatically).

Other benches in `bench/` do not require external network access and
should run with `python bench/<name>.py` once the venv is active and the
daemon is running.

### Bench scripts need `IAI_MCP_CRYPTO_PASSPHRASE`

Three bench scripts construct ephemeral per-row tmp `MemoryStore()`
instances and consequently need a crypto passphrase even when the
operator's main store is happy reading `~/.iai-mcp/.crypto.key`. Plan
25-04 added a defensive default (`iai-mcp-bench-falsifiability-deterministic-2026`)
so the benches are self-contained, but any operator-set value is
respected.

## Troubleshooting

### Daemon won't start / "socket not bound"

```bash
iai-mcp daemon status            # is it really down, or just unresponsive?
iai-mcp daemon logs --tail 50    # last actions before the issue
iai-mcp doctor                   # systematic health audit
```

Common causes:

- **Stale `.locked` PID-reuse.** The lifecycle lock cross-checks
  `psutil.Process(pid).cmdline()`; if a recycled PID is now used by a
  non-IAI process the lock file is rejected automatically. If this fails,
  delete `~/.iai-mcp/.locked` and retry `iai-mcp daemon start`.
- **Hippo store growth / WAL bloat.** Post-Hippo the daemon checkpoints
  the SQLite WAL + VACUUMs + rebuilds the hnswlib index during the SLEEP
  cycle (`OPTIMIZE_LANCE` step — name preserved for back-compat). Force it
  manually: `iai-mcp maintenance compact-records --apply --yes`.
- **Hippo lock held (`HippoLockHeldError`).** Hippo is single-writer
  (exclusive `~/.iai-mcp/hippo/.lock`). A separate `iai-mcp doctor` run
  while the daemon is alive cannot open the store directly — doctor rows
  `(f)/(t)/(u)` report the lock-held state. This is expected, not a fault;
  stop the daemon if you need a separate process to open Hippo.

For a step-by-step diagnostic / recovery flow, the **`iai-mcp-recovery`
skill** ships under `~/.claude/skills/iai-mcp-recovery/`. From the
operator's shell, type `iai-mcp-recovery` (or ask Claude "run iai-mcp
recovery"); the skill walks through socket-fresh checks, lifecycle FSM
status, crypto-gate validation, deferred-captures buffer drain, and the
launchd / systemd restart procedure.

### AVX2 / `import lancedb` SIGILL (legacy — no longer applies)

Pre-v8.0 the LanceDB Rust core required AVX2 and SIGILL'd on Celeron /
Atom. The runtime backend is **Hippo** (SQLite + hnswlib),
which has no AVX2 dependency — the daemon runs on any modern CPU,
including AVX2-less hosts and Apple silicon. AVX2 only matters if you run
the optional one-time `[migration]` extra (which still pulls LanceDB) on
the migrating host; the steady-state daemon never imports lancedb.

### `CryptoKeyError` on daemon start

```bash
iai-mcp crypto status            # which backend is in use?
ls -la ~/.iai-mcp/.crypto.key    # exists? readable? mode 0600?
```

If the file is missing or corrupted:

```bash
export IAI_MCP_CRYPTO_PASSPHRASE='<your-passphrase>'
iai-mcp crypto init              # regenerates the key from the passphrase
iai-mcp daemon start
```

**Warning:** regenerating a fresh key on a non-empty store renders all
previously-encrypted records unreadable. Restore from
`iai-mcp restore <backup>` first if there is data to keep.

### TypeScript wrapper reports "daemon degraded"

The wrapper probes `iai-mcp doctor` on session start and emits a one-line
stderr warning when the daemon is in degraded health. The operator can
still issue MCP calls — the wrapper falls back to `bank-recall` against
the encrypted bank/recent transit window (last 30 days of memories) — but
nightly REM consolidation, schema mining, and rich-club spreading
activation are paused until the daemon is back.

### `iai-mcp doctor` red on headless / VPS

`iai-mcp doctor --headless` (or auto-detected when `DISPLAY` and
`WAYLAND_DISPLAY` are both absent on Linux) downgrades HID-idle and
socket-fresh rows from FAIL to WARN. Headless deployments no longer
light up red on legitimately-absent display hardware.

### Bench script crashes at "crypto key not found"

Apply the same `IAI_MCP_CRYPTO_PASSPHRASE` default as the production daemon, OR
set your own:

```bash
export IAI_MCP_CRYPTO_PASSPHRASE='iai-mcp-bench-falsifiability-deterministic-2026'
python bench/<name>.py
```

The bench scripts that need this default already set it themselves (Phase
25-04 fix); only invocations from non-standard wrappers or one-off
scripts inherit the gap.

## Verification

After install, verify the deployment with the official sanity-check
sequence:

```bash
iai-mcp daemon status               # expect FSM=WAKE, heartbeat fresh
iai-mcp doctor                      # all rows PASS (except possibly headless WARN)
iai-mcp daemon stats                # session_start_tokens_p90 visible (may be empty on a fresh install)
pytest -x                           # 2100+ tests; expect 4 skipped / 1 legacy fail
```

For empirical-falsifiability claims (e.g. "Rescue@10 = 1.000 on
Contradiction-longitudinal Regime 2"), see
the falsifiability report (internal archive). For the
per-release shipping log, see
[`CHANGELOG.md`](../CHANGELOG.md).

## Uninstall

`scripts/uninstall.sh` removes the launchd / systemd unit, the
`~/.local/bin/iai-mcp` symlink, and (optionally) the venv. By default it
**preserves `~/.iai-mcp/`** so the operator's memory store survives the
uninstall; pass `--purge` to also delete the store directory.

```bash
bash scripts/uninstall.sh           # preserves ~/.iai-mcp/
bash scripts/uninstall.sh --purge   # also deletes ~/.iai-mcp/
```

`iai-mcp capture-hooks uninstall` removes the three Claude Code hooks
from `~/.claude/settings.json`. Always run it before `uninstall.sh
--purge` to avoid Claude Code hitting stale hook references on the next
session.

## Common pitfalls

- **Skipping the AVX2 check.** Save yourself the SIGILL crash; verify
  AVX2 before `pip install`.
- **Running multiple daemons against the same store.** The `fcntl.flock`
  lock at `~/.iai-mcp/.locked` prevents this, but stale lock files can
  fail open. One operator → one daemon → one store directory.
- **Editing source inside a worktree, expecting the production daemon to
  pick it up.** It will not — the daemon was started against the main
  checkout's `~/.local/bin/iai-mcp` symlink. Either restart the daemon
  against the new code (`launchctl kickstart -k gui/$(id -u)/com.iai-mcp.daemon`
  on macOS) or run code from the worktree as a one-shot CLI invocation.
- **Forgetting `HF_TOKEN`.** `bench/longmemeval_blind.py` will exit
  early without an explicit diagnostic if `HF_TOKEN` is unset; the
  HuggingFace SDK simply 401s on the dataset download.
- **Pinning Python 3.13.** Some torch wheels are not yet available;
  the project pins to 3.11–3.12.
- **Pushing the private repo to a public remote.** The default
  development branch may contain developer-only annotations and
  references to private experiments; the public-release scrub is a
  separate release-hardening workstream. Treat the current repo as
  confidential until that scrub completes.

## See also

- [`README.md`](../README.md) — architecture overview + MCP tools table.
- [`CHANGELOG.md`](../CHANGELOG.md) — per-release shipping log.
- The falsifiability report (internal archive) — empirical
  ship-gate evidence + refuted predictions.
- `~/.claude/skills/iai-mcp-recovery` — operator-facing recovery skill
  (step-by-step daemon revive flow).
