**English** | [中文](./README_zh-CN.md)

<p align="center">
  <img src="docs/assets/iai-memory-banner.png" alt="iai-memory — a personal memory engine for your AI coding workflow" width="100%">
</p>

<p align="center">
  <b>Captures conversations verbatim, recalls relevant context across sessions,<br>
  and keeps both old and current wording retrievable when facts change.</b>
</p>

<p align="center">
  <img src="docs/assets/iai-brain-demo.gif" alt="iai-memory searching, recalling, pinning, fading, rescuing, and learning a file" width="850">
</p>

<p align="center">
  <a href="https://pypi.org/project/iai-pme/"><img src="https://img.shields.io/pypi/v/iai-pme?style=flat-square&color=1f6feb&label=pypi" alt="iai-memory on PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-1f6feb?style=flat-square" alt="MIT License"></a>
  <img src="https://img.shields.io/badge/python-3.11%20%7C%203.12-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or 3.12">
  <img src="https://img.shields.io/badge/macOS%20%7C%20Linux-supported-555?style=flat-square" alt="macOS and Linux supported">
  <img src="https://img.shields.io/badge/Windows-beta-dbab09?style=flat-square&logo=windows&logoColor=white" alt="Windows beta">
  <img src="https://img.shields.io/badge/MCP-compatible-8957e5?style=flat-square" alt="MCP compatible">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Rescue%4010-1.000-2ea043?style=flat-square" alt="Rescue@10 1.000">
  <img src="https://img.shields.io/badge/LongMemEval%20R%405-0.962-2ea043?style=flat-square" alt="LongMemEval R@5 0.962">
  <img src="https://img.shields.io/badge/historical--verbatim-1.000-2ea043?style=flat-square" alt="Historical-verbatim hit@10 1.000">
  <img src="https://img.shields.io/badge/at%20rest-AES--256--GCM-2ea043?style=flat-square" alt="AES-256-GCM at rest">
</p>

<p align="center">
  <a href="#quick-start"><b>Quick start</b></a> ·
  <a href="#how-it-works"><b>How it works</b></a> ·
  <a href="#benchmarks"><b>Benchmarks</b></a> ·
  <a href="#compatibility"><b>Compatibility</b></a> ·
  <a href="docs/REFERENCE.md"><b>Technical reference</b></a>
</p>

---

## What it is

iai-memory gives the coding assistant you already use a persistent memory on your
machine. With ambient hooks enabled, it records both sides of a conversation,
keeps the captured wording, and supplies a bounded slice of relevant history
when a session starts or advances. You do not maintain a memory file or keep
saying “remember this.”

Corrections do not rewrite history. A changed fact becomes a new record linked
to the superseded one, so both the current statement and the earlier wording
remain queryable. Recall can return contradictory or superseded records beside
matching ones instead of letting an obsolete fact pass as current.

This is a **personal engine for an assistant you already use**, not a
multi-tenant memory API for an application. Episodic capture is write-once and
verbatim; storage, embeddings, retrieval, graph operations, and the dashboard
run locally. No external vector or graph database is required.

**The memory style is autistic by design:** verbatim over paraphrase, precise
cues, sustained focus, and rare events kept rare. [Why the name](#about-the-name).

---

## Quick start

### Claude Code

```bash
python3.12 -m pip install -U iai-pme
```

Then run inside Claude Code:

```text
/plugin marketplace add CodeAbra/iai-personal-memory-engine
/plugin install iai-memory@iai-pme
```

Restart the session, then verify:

```bash
iai --version
iai-mcp daemon status
iai-mcp doctor
```

Python 3.11 is also supported.

### macOS or Linux: all-in-one source install

```bash
curl -fsSL https://raw.githubusercontent.com/CodeAbra/iai-personal-memory-engine/main/scripts/bootstrap.sh | bash
```

This builds the Rust engine and TypeScript wrapper, installs the background
service and hooks, registers Claude Code, and runs the health check. It requires
Git, Python 3.11/3.12, Node.js 18+, and Rust. To inspect the steps without
changing anything:

```bash
curl -fsSL https://raw.githubusercontent.com/CodeAbra/iai-personal-memory-engine/main/scripts/bootstrap.sh | bash -s -- --dry-run
```

### Other hosts

```bash
python3.12 -m pip install -U iai-pme
iai-mcp crypto init
iai-mcp daemon install
iai-mcp capture-hooks install --target codex
```

Replace `codex` with `cursor`, `antigravity`, `hermes`, `openclaw`, or `all`.
MCP tools work with any MCP-over-stdio client; automatic capture and context
injection depend on the hooks exposed by the host. See the
[technical reference](docs/REFERENCE.md).

New stores use the native engine format by default; an existing store keeps its
current format on upgrade. To move an existing legacy SQLite store onto the
native engine, run `iai-mcp migrate-to-lilli` — `iai-mcp doctor` prints the exact
command, and the [technical reference](docs/REFERENCE.md) documents the full flow.

---

## What happens after installation

| Event | Action |
|---|---|
| Prompt | New turns are appended to a session buffer as file IO; no embedding or engine RPC is needed on the capture path |
| Session end | Remaining transcript content is rolled over for ingestion; hook failures do not block the host |
| Session start | A bounded memory prefix is exposed as host context; an empty store or unavailable engine yields empty output |
| Later turns | Supported hosts receive a small foresight or delta pack with age and revision markers |
| Idle time | Captures are embedded, deduplicated, encrypted, inserted, clustered, consolidated, reinforced, and decayed |

The background process is called the `daemon` in the CLI. The MCP wrapper and
`iai` can still read the local store directly when it is asleep or temporarily
unavailable.

---

## How it works

### Memory model

| Tier | Contains |
|---|---|
| **Episodic** | Timestamped, write-once fragments of what was said |
| **Semantic** | Summaries induced from related episodes during idle consolidation |
| **Procedural** | Ten bounded behavioural parameters learned over time |

Distinct hyperdimensional representations keep literal detail, semantic
structure, and behavioural tendencies from collapsing into one vector surface.

The local, LLM-free recall path combines semantic similarity, graph evidence,
recency, temporal validity, and lexical evidence. `memory_recall` returns both
`hits` and `anti_hits`; `memory_contradict` closes the old record's validity
interval, creates a new record, and links the two.

While idle, the engine groups related episodes, induces semantic memory,
reinforces useful paths, and decays weak unreviewed edges. One optional REM step
may invoke `claude -p` through the user's existing Claude subscription, capped
at no more than 1% of the daily quota. No Anthropic API key is required.

### First-party components

| Component | Role |
|---|---|
| **Hippo** | Encrypted records, vector index, and graph in one local store |
| **MOSAIC** | Leiden-family community detection with stable community identity |
| **Lilli HD** | Hyperdimensional substrate and structural recall |
| **Native engine** | Rust embedder and graph kernels |

---

## Dashboard and CLI

```bash
iai brain
```

The local dashboard searches the store, exposes graph neighbourhoods and
contradictions, pins or fades memories, ingests files, controls the background
engine, and reports token-use estimates from your own store.

```text
iai recall · temporal-recall · search · ask · capture · teach · upload
iai watch · brain · status · last
```

`iai upload` accepts documents, Office files, e-books, source code,
configuration files, and directories. Full formats and administrative commands
are listed in [`docs/REFERENCE.md`](docs/REFERENCE.md).

---

## Benchmarks

Every harness ships in `bench/`; methodology and reproduce commands are in
[`BENCHMARKS.md`](BENCHMARKS.md).

| Benchmark | Result |
|---|---:|
| Rescue@10 after contradiction | **1.000** |
| Historical-verbatim hit@10 | **1.000** |
| LongMemEval-S R@5, product embedder | **0.962** |
| LongMemEval-S R@10, product embedder | **0.978** |

Historical-verbatim retrieval uses a flat-cosine baseline of about 0.71. With
the matched `all-MiniLM-L6-v2` embedder, iai-memory and mempalace v3.3.6 both
score R@5 `0.966` and R@10 `0.978`; no win is claimed.

On the author's store, an automatically injected memory pack averaged about
350 tokens versus about 2,850 tokens for the agent-search round trip it
replaced: approximately 88% cheaper on that measured workload. This does not
apply to explicit `memory_recall`, whose default response budget is 1,500
tokens.

---

## MCP tools

```text
memory_recall              memory_temporal_recall
memory_recall_structural   memory_search
memory_capture             memory_contradict
memory_reinforce           memory_consolidate
profile_get_set            topology
schema_list                events_query
episodes_recent            curiosity_pending
```

Fourteen tools cover cue, temporal, structural, and lexical recall; capture and
correction; reinforcement and consolidation; behavioural-profile control; and
store introspection.

---

## Compatibility

| Host | Ambient behaviour |
|---|---|
| **Claude Code** | Session-start recall, per-turn updates, turn capture, and session capture |
| **Codex CLI** | Full integration through Codex hooks |
| **Cursor** | Session-start recall and capture; no per-turn text injection |
| **Antigravity** | Recall per invocation and lossless transcript capture |
| **Hermes 0.5.0+** | Recall before model calls and capture from its message store |
| **OpenClaw** | MCP tools on request; no ambient shell hooks |
| **Gemini CLI and other MCP hosts** | MCP tools; no bundled host-specific hooks unless listed above |
| **Claude Desktop** | MCP tools; plain Chat does not expose Claude Code-style ambient hooks |

---

## Privacy and limitations

- Records are encrypted at rest with AES-256-GCM. The store and key live under
  `~/.iai-mcp/`; back them up together.
- macOS and Linux use a Unix socket. Windows uses an ephemeral loopback port
  with a per-user token.
- There is no iai-memory account, telemetry pipeline, hosted dashboard, or
  cross-machine sync.
- Optional iai-memory network activity is the REM `claude -p` step and a daily
  PyPI version check. Set `IAI_MCP_VERSION_CHECK=0` to disable the check.
- The store refuses to mix incompatible embedding generations; changing the
  embedder requires an explicit migration.
- Recall is usually mediocre during roughly the first ten sessions, and quality
  and latency depend on corpus size, language, embedder, and stored history.
- The default store is English-first. Raw non-English records require an
  explicit `raw:<lang>` tag and a multilingual or custom embedder.
- Windows support is beta. Ambient behaviour varies with host hook support.
- The project is solo-maintained and has no enterprise SLA.

Health and updates:

```bash
iai-mcp doctor          # 36 checks
iai-mcp daemon status
iai-mcp self-update
```

---

## About the name

**IAI — Independent Autistic Intelligence** describes the memory design.

- **Independent:** the engine, store, embeddings, and dashboard run locally.
- **Autistic:** literal preservation, precise cues, sustained focus, and rare
  events retained as rare rather than smoothed into a typical summary. This is
  an operational design description, not a diagnosis or casual metaphor.
- **Intelligence:** used in the systems sense — a process that observes,
  adapts, reorganizes itself, and remains viable over time.

“Personal memory engine” describes the scope: one person's memory, on one
machine, used by the assistant they already have.

---

## Documentation

- [`docs/REFERENCE.md`](docs/REFERENCE.md) — technical and operational reference
- [`BENCHMARKS.md`](BENCHMARKS.md) — methodology and reproduce commands
- [`docs/EMBEDDERS.md`](docs/EMBEDDERS.md) — providers, languages, and migrations
- [`CHANGELOG.md`](CHANGELOG.md) — release history
- [`CONTRIBUTING.md`](CONTRIBUTING.md) — development and test setup
- [`SECURITY.md`](SECURITY.md) — private vulnerability reporting

Issues and pull requests are welcome. Changes to retrieval, capture,
contradiction handling, or consolidation should include relevant benchmark
reruns.

## Authors

By Areg Aramovich Noya and Lilli Noya, in collaboration with the team at
[lcgc.dev](https://lcgc.dev).

## License

[MIT](LICENSE)
