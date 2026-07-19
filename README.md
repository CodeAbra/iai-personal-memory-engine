**English** | [中文](./README_zh-CN.md)

<p align="center">
  <img src="logo.png" alt="iai-pme" width="600">
</p>

<h3 align="center">The best open-source personal memory engine for AI coding assistants.</h3>
<p align="center">Every claim ships with the harness that proves it. Run the benchmarks yourself.</p>
<p align="center"><b>Persistent memory for any MCP client.</b> Local, encrypted, verbatim recall via vector search + knowledge graph. MIT.</p>

<p align="center">
  <img src="https://img.shields.io/badge/release-v2.4.1-1f6feb?style=flat-square" alt="Release v2.4.1">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1f6feb?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%20|%203.12-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 | 3.12">
  <img src="https://img.shields.io/badge/platform-macOS%20|%20Linux-555?style=flat-square" alt="Platform: macOS and Linux">
  <img src="https://img.shields.io/badge/Windows-beta-dbab09?style=flat-square&logo=windows&logoColor=white" alt="Windows: beta">
  <img src="https://img.shields.io/badge/engine-Rust%20native-dea584?style=flat-square&logo=rust&logoColor=black" alt="Rust-native engine">
</p>
<p align="center">
  <img src="https://img.shields.io/badge/LongMemEval%20R%405-0.962-2ea043?style=flat-square" alt="LongMemEval R@5 0.962">
  <img src="https://img.shields.io/badge/Rescue%4010-1.000-2ea043?style=flat-square" alt="Rescue@10 1.000">
  <img src="https://img.shields.io/badge/at%20rest-AES--256--GCM-2ea043?style=flat-square" alt="AES-256-GCM">
  <img src="https://img.shields.io/badge/local--only-no%20telemetry-2ea043?style=flat-square" alt="Local only, no telemetry">
  <img src="https://img.shields.io/badge/MCP-compatible-8957e5?style=flat-square" alt="MCP compatible">
  <a href="https://glama.ai/mcp/servers/CodeAbra/iai-mcp"><img src="https://glama.ai/mcp/servers/CodeAbra/iai-mcp/badges/score.svg" alt="Glama MCP score"></a>
</p>

---

# iai-pme

**Your AI assistant forgets you every session. iai-pme gives it a memory that doesn't.**

*Independent Autistic Intelligence — a personal memory engine. Fully local, ambient. Works with Claude Code, Claude Desktop, Cursor, Codex CLI, Gemini CLI, Cline, Continue.dev, Zed, Cherry Studio, Goose, Aider, Hermes, OpenClaw, Le Chat, Kimi — anything that speaks MCP-over-stdio.*

## Table of contents

- [What it is](#what-it-is)
- [Quick start](#quick-start)
- [Usage](#usage)
- [How it works](#how-it-works)
- [Built our own](#built-our-own)
- [Benchmarks](#benchmarks)
- [Configuration](#configuration)
- [Doctor](#doctor)
- [Notes for AI assistants](#notes-for-ai-assistants-helping-with-installation)
- [Status and limitations](#status-and-limitations)
- [Compatibility](#compatibility)
- [About the name](#about-the-name)
- [Authors](#authors)
- [License](#license)
- [Contributing](#contributing)

---

## What it is

A local server that speaks the [MCP protocol](https://modelcontextprotocol.io) and gives Claude, and any other MCP-compatible assistant, a long-term memory. It captures every turn of every session verbatim, organizes those captures over time into a personal map of who you are, and serves a small slice of relevant memory back at the start of each new conversation. You never have to say *"remember this"* or *"what did we say last time?"*.

<p align="center"><img src="docs/assets/slides/slide-02.jpg" width="850" alt="iai-pme"></p>

I built this for myself. It worked. I've been running it daily for months, and now I'm sharing it. The benchmarks were mostly for my own curiosity. I wanted to know if it actually works or if I'd just gotten used to it.

Under the hood it's not a wrapper around someone else's vector store and graph library — the parts that matter are my own code: the storage engine, the community-detection algorithm, the hyperdimensional memory substrate, and a native engine that makes it fast. More on that in [Built our own](#built-our-own).

And unlike cloud memory services, there's no API key, no account, and no telemetry: the engine, the store, and the embeddings all run locally. The only thing that leaves your machine is the normal model call your CLI already makes.

<p align="center"><img src="docs/assets/slides/slide-04.jpg" width="850" alt="iai-pme"></p>

---

## Pick your path

|  |  |  |
|---|---|---|
| **🟢 Just want it to work?** | **🔵 Want the numbers?** | **🟣 Want the internals?** |
| Install once, then forget it's there — no commands, fully local. | Every claim ships with the harness that proves it — run them yourself. | We built our own storage engine, clustering, HD substrate and Rust core. |
| → [Quick start](#quick-start) | → [Benchmarks](#benchmarks) | → [Built our own](#built-our-own) |

---

## Quick start

<p align="center"><img src="docs/assets/slides/slide-15.jpg" width="850" alt="iai-pme"></p>

### Prerequisites

- macOS (Apple Silicon), Linux, or Windows (beta)
- Python 3.11 or 3.12
- Node.js 18+
- A Rust toolchain — the native engine builds from source
- An MCP-compatible CLI host — [Claude Code](https://docs.claude.com/en/docs/claude-code/overview), Codex CLI, Gemini CLI, Cursor CLI, and others
- ~500 MB free disk

macOS and Linux are fully supported. **Windows support is in beta** — the runtime is ported and validated on Windows 11, but the test suite is still being ported, so treat it as experimental. Contributions are very welcome: if you hit a Windows issue, open an issue or PR and I'll help however I can.

### Install

```bash
git clone https://github.com/CodeAbra/iai-personal-memory-engine.git
cd iai-personal-memory-engine
python3.12 -m venv .venv && source .venv/bin/activate
pip install .
```

`pip install` builds the native Rust engine (`iai_mcp_native` — the embedder + graph kernels) automatically, as part of the package build, via `setuptools-rust`. There's no separate build script. If you change the Rust source later and need to rebuild it by hand, there's an escape hatch:

```bash
iai-mcp build-native        # rebuild the native engine in place
```

Then build the MCP wrapper and set up the local engine (it runs in the background):

```bash
cd mcp-wrapper && npm install && npm run build && cd ..
iai-mcp daemon install      # launchd on macOS, systemd on Linux, Task Scheduler on Windows
iai --version
```

### Install the capture + recall hooks

This is what makes memory ambient. Without these hooks iai-mcp reads memory but never writes conversation content and never injects recall at session start. One command wires all three:

```bash
iai-mcp capture-hooks install       # copies all three hooks + patches ~/.claude/settings.json
iai-mcp capture-hooks status        # verify: should print "status: ACTIVE"
iai-mcp capture-hooks uninstall     # clean removal if ever needed
```

For Codex:

```bash
iai-mcp capture-hooks install --target codex
```

To install both:

```bash
iai-mcp capture-hooks install --target all
```

What the install does:

- Copies three hook scripts bundled with the package to `~/.claude/hooks/` (chmod +x):
  - `iai-mcp-turn-capture.sh` (`UserPromptSubmit`, timeout 5s) — appends each prompt + the preceding assistant turn(s) to a per-session buffer as pure file IO. Zero engine RPC during the session.
  - `iai-mcp-session-capture.sh` (`Stop`, timeout 35s) — at session end, rolls the buffer over for the local engine to drain, and runs `iai-mcp capture-transcript --no-spawn` as a safety net.
  - `iai-mcp-session-recall.sh` (`SessionStart`, timeout 30s) — calls `iai-mcp session-start` and pipes the assembled memory prefix to stdout, which Claude Code injects as `additionalContext` before the first prompt. Fail-safe: empty store or unreachable local engine yields empty stdout — session start is never blocked.
- Registers iai-mcp in Claude Desktop's config if installed.
- Idempotent — re-running detects existing entries and makes no changes.
- No secrets, no tokens, no network calls.

What happens at runtime:

- **Every prompt** (per-turn hook): appends new transcript turns to the session buffer. ~5 ms per turn, no embedding, no engine socket.
- **Every session end** (Stop hook): rolls the buffer over, captures any remaining turns. Fail-safe exit 0.
- **Every session start** (recall hook): assembles the cached memory prefix and pipes it to Claude. Empty store or unreachable local engine → empty stdout.
- **When idle** (local engine): drains the buffer through the shield → embed → dedup → encrypted insert pipeline on the WAKE → DROWSY edge (5-min idle) and after every REM cycle.

### Connect your MCP host

Claude Code:

```bash
claude mcp add iai-mcp -- node "$(pwd)/mcp-wrapper/dist/index.js"
```

Or edit `~/.claude.json` directly:

```json
{
  "mcpServers": {
    "iai-mcp": {
      "command": "node",
      "args": ["/absolute/path/to/iai-mcp/mcp-wrapper/dist/index.js"]
    }
  }
}
```

Use the absolute path. `~` and `$HOME` won't expand here.

For Claude Desktop, edit `~/Library/Application Support/Claude/claude_desktop_config.json`.

Codex CLI:

```toml
[mcp_servers.iai-mcp]
command = "node"
args = ["/absolute/path/to/iai-mcp/mcp-wrapper/dist/index.js"]

[mcp_servers.iai-mcp.env]
IAI_MCP_PYTHON = "/absolute/path/to/iai-mcp/.venv/bin/python"
IAI_MCP_STORE = "/Users/you/.iai-mcp"
```

Codex hooks are stable in current Codex CLI builds. If hooks are disabled by
local policy or an older install, enable `[features].hooks = true` in
`~/.codex/config.toml`.

### Verify

```bash
iai-mcp doctor
iai-mcp daemon status
```

Restart Claude Code. Start a session, do some work, exit. Then:

```bash
tail ~/.iai-mcp/logs/capture-$(date -u +%Y-%m-%d).log
```

You should see a `rc=0` line. That's your first memory.

---

## Usage

You do not call `iai-mcp` directly during a session. Once it's connected:

Capture is automatic. Every turn, yours and the assistant's, is recorded verbatim with timestamps and session metadata. You don't say *"remember this."*

Recall is automatic. When a new session starts, the local engine assembles a small relevant slice of your history and injects it into the conversation prefix. You don't say *"what did we say."*

Consolidation runs idle. Between sessions, the local engine merges duplicates, strengthens recall pathways for things retrieved often, and prunes weak edges. The system gets quietly better at remembering you over time.

After a few weeks of regular use the difference becomes noticeable. The assistant stops asking the same orientation questions, references things you mentioned in passing, and adapts to your style without being told.

There's also a CLI — you don't need it for normal use, but when you want to query or add to your memory straight from the terminal, `iai` is there: `recall`, `capture`, `ask` (LLM synthesis grounded in your memory), `status`, and `last`.

<p align="center">
  <img src="docs/assets/iai-cli.png" alt="iai — terminal memory for your agent" width="600">
</p>

---

## How it works

The local engine is a Python process that runs in the background — it sleeps when idle and wakes when your assistant needs it, so it isn't always-on or constantly using CPU. Your MCP client connects to it via a Unix socket. No network exposure.

Recall doesn't depend on the engine being awake. The store itself is always available: when the engine is asleep or not running, your assistant (and the `iai` CLI) read memory directly from the local store. The engine handles the fast LLM-free recall path when it's up, plus the nightly consolidation pass — it's never a gatekeeper on your memory.

Memory is stored in three tiers:

*Episodic* is verbatim, timestamped fragments of what was said. Write-once, never overwritten or rewritten.

*Semantic* is summaries induced from clusters of related episodes during idle-time consolidation.

*Procedural* is a small set of stable parameters about you, learned over time: preferences, style cues, recurring patterns. Eleven sealed knobs that shift based on what works.

The three tiers are backed by a hyperdimensional memory substrate — each kind of memory gets its own representation, so episodic detail, semantic gist, and procedural patterns don't collapse into one undifferentiated blob.

<p align="center"><img src="docs/assets/slides/slide-05.jpg" width="850" alt="iai-pme"></p>

A background pass runs periodically (sleep cycles): it clusters episodes with my own community-detection algorithm, builds semantic summaries, decays old unreinforced connections, and reinforces frequently co-retrieved paths. Things you haven't revisited fade naturally. One step per night can make a single LLM call **through your existing Claude subscription** (`claude -p`) — no separate API key, capped at ≤1% of your daily quota. (`iai-mcp doctor` row (p) verifies there's no API-key SDK path installed at all.)

<p align="center"><img src="docs/assets/slides/slide-08.jpg" width="850" alt="iai-pme"></p>

Recall combines three signals: semantic similarity, graph-link strength, and recency. All ranked together. The hot path runs entirely locally with no LLM in the loop.

<p align="center"><img src="docs/assets/slides/slide-06.jpg" width="850" alt="iai-pme"></p>
<p align="center"><img src="docs/assets/slides/slide-07.jpg" width="850" alt="iai-pme"></p>

All records are encrypted at rest with AES-256-GCM. The key lives in `~/.iai-mcp/.key` (mode 0600). Back it up. Lose the key, lose the memories.

Everything lives at `~/.iai-mcp/`. Embeddings are computed locally. The only data that leaves the machine is your normal conversation with whatever LLM API your client uses.

<p align="center"><img src="docs/assets/slides/slide-10.jpg" width="850" alt="iai-pme"></p>

---

## Built our own

Most memory projects are a thin layer over an off-the-shelf vector store and someone else's graph library. This one isn't. The load-bearing pieces are my own code, written for this exact workload — a small memory graph that mutates every night and gets queried on every recall:

| Piece | What it is |
|---|---|
| **Hippo** | The storage engine — encrypted records, the vector index, and the graph in one local store. |
| **MOSAIC** | My community-detection algorithm — a Leiden-family method (CPM objective), rewritten in pure MIT-licensed Python to replace the GPL-licensed `leidenalg`/`igraph` dependency. It clusters the memory graph so recall spreads through the right neighbourhood and sleep can replay coherent episodes — tuned for a small, heterogeneously-weighted graph that changes every cycle, with stable community identity across splits and merges. |
| **Lilli HD** | The hyperdimensional memory substrate — distinct representations for episodic / semantic / procedural memory, with structural recall (retrieve by the *shape* of a memory, not just its embedding). |
| **Native engine** | A Rust core — the embedder and the graph kernels. This is where the latency comes from. |

These sit on a thin layer of proven, permissive primitives — SQLite, the `candle` tensor library, NumPy, and the audited `cryptography` AES implementation. I build the engine and the algorithms; I don't reinvent a database, a tensor kernel, or — deliberately — a crypto primitive. The interesting bricks are mine; the foundation under them is boring, battle-tested, and permissive. MIT throughout.

I wrote these because the off-the-shelf options were built for a different problem — large static graphs, multi-tenant clouds, gist-style summarization — and they were slower and a worse fit for "one person's memory on one machine, reorganized every night." Mine are faster *on this shape of problem*, which is the only shape I care about.

<p align="center"><img src="docs/assets/slides/slide-09.jpg" width="850" alt="iai-pme"></p>

---

## Benchmarks

I built these because I wanted honest numbers, not a leaderboard. Every harness ships in `bench/` with a one-line reproduce command — run them and get your own results. Where a number missed its target or regressed, it says so. Full detail in [`BENCHMARKS.md`](BENCHMARKS.md).

### LongMemEval-S — the one head-to-head arena

Validated in a single harness against [mempalace](https://github.com/MemPalace/mempalace) on the identical 500 cleaned questions, session granularity, `recall_any@k`, raw (no rerank):

| System | Embedder | R@5 | R@10 |
|---|---|---|---|
| **iai** (product) | bge-small-en-v1.5 | **0.962** | 0.978 |
| iai (matched embedder) | all-MiniLM-L6-v2 | 0.966 | 0.978 |
| mempalace v3.3.6 | all-MiniLM-L6-v2 | 0.966 | 0.978 |

On raw retrieval — the headline both projects ship — it's an **exact tie** on the matched embedder — R@5 0.966 = 0.966 and R@10 0.978 = 0.978. Our product embedder scores 0.962 R@5, a 2-question-in-500 difference (noise). No win claimed — an honest tie is the strong, defensible statement. LongMemEval is a *cold, one-shot* retrieval test; it doesn't exercise cross-session memory, which is where the design's real edge is.

### Where it actually leads — longitudinal memory

| Benchmark | Result | What it measures |
|---|---|---|
| **Rescue@10** (post-contradiction) | **1.000** | After a fact is updated/contradicted, the *current* fact still ranks top-10 — where flat-vector stores collapse on the more-similar stale fact. |
| Historical-verbatim (hit@10) | **1.000** | The *superseded/archived* wording of an updated fact also ranks top-10 — both current and prior versions stay retrievable. Flat-cosine baseline ~0.71. |
| Personal-fact drift (recall@10) | 0.9933 | Retention across 50 facts / 50 sessions / 30 intervening sessions. |
| Sleep-consolidation (recall@10) | 1.000 → 1.000 | Recall survives a full consolidation cycle. |
| Session-start tokens | 1,629 min / 2,993 std | Under the ≤3,000-token budget. |
| MOSAIC parity | 36/36 LFR + 10/10 | NMI vs ground-truth, deterministic. |

### Cost & footprint (honest disclosure — not a brag)

| Metric | Measured | Note |
|---|---|---|
| Recall p95 latency | 77 ms @1k · 368 ms @10k | Above the <100 ms@10k target at scale; the rank/centrality stage dominates — a known optimization candidate. |
| Memory (RSS) | 589 MB @10k records | Embedder + graph runtime; well under the 2 GB budget. |
| Rust embedder | p50 70 ms / p95 253 ms | bge-small-en-v1.5, 384-dim. |

**Historical-verbatim retrieval** (pulling the *superseded/archived* wording of an updated fact, not just its current value) ranks top-10 at **hit@10 = 1.000** on the honest-scale contradiction bench (1000 sessions × 3 seeds), versus ~0.71 for a flat-cosine baseline — so both the current fact (Rescue@10) and its earlier wording stay retrievable.

```bash
python -m bench.longmemeval_blind            # LongMemEval-S (raw)
python -m bench.contradiction_longitudinal   # Rescue@10 / longitudinal
python -m bench.personal_fact_drift          # drift / retention
python bench/sleep_ablation.py               # sleep-consolidation recall
python -m bench.tokens                       # session-start token cost
python -m bench.neural_map                   # recall latency
python -m bench.memory_footprint             # RAM footprint
```

Measured on an Apple M2 Max (64 GB). The harnesses are the proof — run them yourself.

---

## Configuration

| Variable | Default | What it does |
|---|---|---|
| `IAI_MCP_STORE` | `~/.iai-mcp/` | Data directory |
| `IAI_MCP_PYTHON` | — | Absolute path to the venv Python (for the MCP host config) |
| `IAI_MCP_RECALL_CONCURRENCY` | `2` | Maximum cued `memory_recall` calls dispatched concurrently by the socket daemon |
| `IAI_MCP_RECALL_SLOT_WAIT_SEC` | `0.25` | How long an overflow cued recall waits for a slot before returning `_degraded: recall_busy` |
| `IAI_MCP_EMBED_PROVIDER` | `native` | `native` for built-in BGE or `http` for a replaceable loopback provider |
| `IAI_MCP_EMBED_URL` | — | Loopback endpoint or base URL for the `http` provider |
| `IAI_MCP_EMBED_DIM` | `384` | Vector dimension; required for the `http` provider |
| `IAI_MCP_EMBED_MODEL_ID` | — | Model identifier; required for the `http` provider |
| `IAI_MCP_EMBED_TIMEOUT_SEC` | `30` | Local provider request timeout |

The built-in Rust BGE model remains the zero-configuration default. Setting the
provider to `http` replaces it completely: the native model is not constructed,
downloaded, or run. This makes multilingual and domain-specific embedders
possible without adding a Python ML stack to iai-mcp. See
[`docs/EMBEDDERS.md`](docs/EMBEDDERS.md) for the protocol and migration steps.

---

## Doctor

`iai-mcp doctor` runs 25 checks against the local engine, the store, the native engine, and the runtime state. Output is one line per check: PASS, WARN, or FAIL.

<p align="center"><img src="docs/assets/slides/slide-13.jpg" width="850" alt="iai-pme"></p>

```bash
iai-mcp doctor
```

What it checks:

| # | Check | What it means |
|---|---|---|
| a | daemon process alive | Is the daemon process running? |
| b | socket file fresh | Can the UNIX socket accept a connection? |
| c | lock file healthy | Is the process lock held correctly? |
| d | no orphan core procs | No leftover stdio core process without a daemon |
| e | daemon state file valid | State file parses and has expected fields |
| f | hippo storage readable | Can the store be opened and queried? |
| g | no dup binders | Only one process is bound to the socket |
| h | crypto key file state | Encryption key exists, correct permissions (0600) |
| i | hippo db size | Store size is within healthy bounds |
| j | lifecycle current state | Current FSM state is valid |
| k | lifecycle history 24h | Recent lifecycle transitions look sane |
| l | sleep cycle quarantine | No sleep cycle is stuck or quarantined |
| m | heartbeat scanner | Wrapper heartbeat files are fresh |
| n | HID idle source | Idle detection source is available |
| o | Claude subscription credentials | Subscription creds present for the nightly LLM step |
| p | anthropic SDK absent | Confirms no API-key SDK path is installed |
| q | iai CLI reachable | The `iai` user CLI is on `PATH` |
| r | hippo hnsw index | The ANN index is loadable |
| s | hippo schema version | Store schema is current |
| t | hippo_compacted freshness | Compaction has run recently |
| u | recall centrality regression | Recall ranking hasn't regressed |
| v | native Rust embedder | The Rust embedder is built and produces vectors |
| w | no permanent-failed captures | No capture is stuck after exhausting its retries |
| x | timestamps not collapsed | Record timestamps span a real range, not all-identical |
| z | AVX2 CPU support | CPU supports the instructions the native libs need |

A full-PASS run is healthy. Dropping (b) during a sleep cycle is normal (the socket is busy during consolidation). Multiple FAILs, or a FAIL on (a)/(f)/(v), means something is actually wrong.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Local engine refuses to start, error ends in a build command | The native Rust engine isn't built (mandatory — no fallback) | Run the build command the error prints (the installer normally does this). |
| `keyring.errors.NoKeyringError` on first run | Storage is file-backed at `~/.iai-mcp/.crypto.key`. Older setups referenced a Keychain-only path. | `iai-mcp crypto init` (idempotent). `iai-mcp daemon install` calls this automatically on fresh installs. |
| Daemon crashes on first start with `CryptoKeyError` | Fresh install bypassed `daemon install` — no `.crypto.key` exists yet. | `iai-mcp crypto init`, then restart the daemon. |
| `iai-mcp daemon install` says "launchd bootstrap failed" | Existing plist from previous install | `iai-mcp daemon uninstall` first, then `install` again. |
| Daemon "active" but no tick events | First-week bootstrap (no quiet-window data yet) | Wait 2 h of MCP idle, or force: `iai-mcp daemon force-rem` |
| Claude Code doesn't show iai-mcp tools after `claude mcp add` | Forgot to fully quit — "reload window" is not enough | `killall Claude` then relaunch. Check `~/Library/Logs/Claude/*.log` for MCP stderr. |

---

## Notes for AI assistants helping with installation

If you're an AI model helping a user install or debug iai-mcp, read this before diagnosing anything.

Terminology note: this doc calls the background component the **local engine**. Its process, CLI subcommands and `doctor` check labels still use the original name `daemon` — e.g. `iai-mcp daemon status`, `pgrep -f iai_mcp.daemon`, and the `(a) daemon process alive` check. "Local engine" and "daemon" refer to the same process.

The local engine has four FSM states: `WAKE`, `TRANSITIONING`, `SLEEP`, `DREAMING`. Separately, the wrapper-layer lifecycle has four states: `WAKE`, `DROWSY`, `SLEEP`, `HIBERNATION`. These are two different state machines in two different files. Don't confuse them.

Things that look broken but are not:

- `last_tick_at` hasn't changed in hours. This is a sleep-cycle tick timestamp, not a process heartbeat. The daemon can be alive and healthy with a stale `last_tick_at`. Check `iai-mcp daemon status` instead.
- `fsm_state: DREAMING` for hours. Normal. The local engine is running a sleep cycle. Wake transitions are event-driven (a new session arriving wakes it), not time-driven.
- `status: deferred` in the capture log. This is the only path the Stop hook takes. It writes captures to `~/.iai-mcp/.deferred-captures/` and the local engine picks them up on its next tick. This is by design, not a failure.
- `iai-mcp doctor` drops the socket check during SLEEP or DREAMING. The socket is intentionally unresponsive during consolidation passes. This is healthy.
- Cold start takes a few seconds. The local engine initializes the store and loads the embedding model into memory. Don't kill it because it hasn't responded in 5 seconds.

Actual problems look like: `iai-mcp daemon status` returning `ok: False`, doctor showing multiple FAIL results, or the daemon process not existing at all (`pgrep -f iai_mcp.daemon` returns nothing).

When in doubt, run `iai-mcp doctor` and read what it says. The output is self-explanatory.

---

## Status and limitations

**Out of experimental.** I built this for myself and ran it daily for months; it's now a stable release with a committed public surface. The MCP tool set and the on-disk store stay stable across `1.x` — breaking changes go through the changelog with a deprecation window. It's still a solo-maintained project with no enterprise SLA, but it's no longer a moving target.

Limitations worth knowing about:

<p align="center"><img src="docs/assets/slides/slide-14.jpg" width="850" alt="iai-pme"></p>

- English-only by design. The assistant translates to English on the way into memory; the store and the embedder are English-only on purpose.
- No cross-machine sync. The data lives where the local engine runs. Backup is `cp -a ~/.iai-mcp/` somewhere safe.
- No GUI. Inspection happens through CLI subcommands (`iai-mcp doctor`, `iai-mcp daemon status`, `iai-mcp topology`).
- Cold start on a freshly booted machine takes a few seconds while the local engine initializes caches.
- Recall quality on the first ~10 sessions is mediocre. The system needs material to consolidate before it gets useful.

---

## Compatibility

iai-mcp talks to its host over **MCP-over-stdio** — the same protocol every MCP-compatible CLI speaks. So the memory tools (recall, capture, ask, status) work with **any MCP CLI**:

<p align="center"><img src="docs/assets/slides/slide-12.jpg" width="850" alt="iai-pme"></p>

- **Claude Code** — primary host, validated in daily use.
- **Codex CLI** — supported, with ambient capture through a `Stop` hook.
- **Gemini CLI**, **Cursor CLI**, and other MCP-over-stdio CLIs — connect through the same standard protocol; the MCP tools work out of the box.
- **Claude Desktop** — works; uses `claude_desktop_config.json` instead of `~/.claude.json`.

Ambient capture (the hooks that record and recall automatically) ships for Claude Code and Codex today. On other CLIs the MCP tools work directly; wiring up their native hooks for fully automatic capture is a great first contribution — open an issue or PR.

---

## About the name

The project is **iai** — a *personal memory engine*. The short name is an acronym; the descriptor says what it is.

**IAI — Independent Autistic Intelligence** (the memory style):

- **Independent.** Fully local. The local engine runs on your machine, embeddings are computed locally, no telemetry, no cloud dependency. Your memory is your data and stays your data — and it tunes itself over time without you steering it.
- **Autistic.** Describes the memory style, not a diagnosis or a metaphor. The memory is built around verbatim recall, attention to specific cues, and a refusal to smooth rare events into typical ones. Most memory systems compress and summarize aggressively, aiming to give the assistant a *gist* of the past. This one preserves what was actually said and surfaces it on a precise cue. In practice that shows up as: literal preservation over paraphrase; deep focus on the current thread rather than diffuse association; direct, unmasked output; a stable identity that doesn't drift. The trade-off is intentional: more storage and a stricter retrieval interface, in exchange for not losing details.
- **Intelligence.** Used in the systems sense — something that observes, adapts, and stays viable over time — not the marketing sense.

**Personal memory engine** (what it is): not a chatbot feature or a cloud add-on, but a memory *engine* — its own storage, clustering, hyperdimensional substrate and native core — that belongs to one person and runs on one machine. See [Built our own](#built-our-own).

It's an operational design choice about how memory should behave, not a clinical claim.

---

## Authors

By Areg Aramovich Noya and Lilli Noya, in collaboration with the team at [lcgc.dev](https://lcgc.dev).

I built this because I needed it. It works for me. If it works for you, take it.

## License

[MIT](LICENSE)

## Contributing

Issues and PRs welcome. If your change touches retrieval, capture, or consolidation, include bench re-runs. See [CONTRIBUTING.md](CONTRIBUTING.md) for scope, development setup, and how to run the tests, and please follow the [Code of Conduct](CODE_OF_CONDUCT.md).

Found a security issue? Please report it privately as described in [SECURITY.md](SECURITY.md).
