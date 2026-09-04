# iai-memory technical reference

[Back to the main README](../README.md).

## What happens after installation

Normal use requires no iai command.

| When | What iai-memory does |
|---|---|
| Every prompt | Appends new turns to a per-session buffer as file IO. No embedding or engine RPC is required on this path. |
| Session end | Rolls the buffer over and captures any remaining transcript content. Failure is fail-safe and does not block the host. |
| Session start | Builds a bounded memory prefix and prints it as context for the host. Empty memory or an unavailable engine yields empty output. |
| Before later turns | On supported hosts, serves a small foresight or delta pack with age and revision markers. |
| While idle | Drains captures, embeds and deduplicates records, updates the graph, and runs consolidation work. |

The per-turn capture path is approximately 5 ms on the author's machine because it only appends to a file. Embedding and encrypted insertion happen later.

Recall remains available when the background engine is asleep or down. The MCP wrapper and `iai` CLI can read the local store directly; the background service handles fast dispatch, capture draining, and consolidation, but it is not the only route to your data.

The first several sessions provide little material to organize. Recall quality is usually mediocre for roughly the first ten sessions and improves after the store has enough history to consolidate.

---

## Dashboard and CLI

`iai brain` opens a loopback-only dashboard for the local store. The same interface also ships as a Tauri desktop application.

<p align="center"><img src="assets/iai-brain-dashboard.jpg" width="850" alt="iai-memory local memory dashboard"></p>
<p align="center"><sub>The public screenshots use synthetic memories and demo counters. Run <code>iai brain</code> to inspect your own store.</sub></p>

The dashboard can:

- search a memory and isolate its graph neighbourhood;
- show tier, tags, timestamps, contradiction links, and revision history;
- pin a memory so it cannot fade;
- queue a memory to fade through the normal decay path, or rescue it before that pass completes;
- ingest a file, deduplicate its passages, and verify recall after ingestion;
- browse self-organized views by time, topic, teaching, and conversation;
- force consolidation, sleep, wake, or restart the local engine;
- show packs served, tokens injected, and the lower-bound token estimate computed from your store.

The user-facing CLI is `iai`:

```text
iai recall                 cue-based memory recall
iai temporal-recall        time-anchored recall
iai search                 lexical search
iai ask                    LLM synthesis grounded in the store
iai capture                explicit memory capture
iai teach / iai upload     ingest a file or directory
iai watch                  watch store activity
iai brain                  open the dashboard
iai status / iai last      inspect the current state and recent memory
```

<p align="center">
  <img src="assets/iai-cli.png" alt="iai terminal interface" width="600">
</p>

<details>
<summary><b>Accepted ingestion formats</b></summary>

`iai upload` and the dashboard accept:

- prose and data: `.txt`, `.md`, `.markdown`, `.rst`, `.tex`, `.bib`, `.csv`, `.pdf`;
- office and e-book containers: `.docx`, `.pptx`, `.xlsx`, `.rtf`, `.epub`;
- source and configuration: `.py`, `.rs`, `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.sh`, `.bash`, `.zsh`, `.go`, `.java`, `.c`, `.h`, `.cpp`, `.hpp`, `.rb`, `.php`, `.swift`, `.kt`, `.sql`, `.toml`, `.yaml`, `.yml`, `.json`, `.ini`, `.cfg`, `.html`, `.css`, `.scss`, `.xml`, `.proto`.

Office containers are parsed as ZIP/XML. Files are chunked, deduplicated against the existing store, inserted through the encrypted write path, and followed by a recall verification pass.

</details>

---

## How it works

### Memory model

The store has three tiers:

| Tier | Contents | Mutation rule |
|---|---|---|
| Episodic | Timestamped fragments of what was actually said | Write-once; corrections create new records |
| Semantic | Summaries induced from related episodes during consolidation | Rebuilt from the underlying history |
| Procedural | Ten stable behavioural parameters learned over time | Bounded tuning; explicit values can be pinned |

The tiers use distinct hyperdimensional representations so episodic detail, semantic structure, and procedural tendencies do not collapse into one vector surface.


### Recall

The hot path is local and LLM-free. Ranking combines:

1. semantic similarity;
2. graph-link strength and centrality;
3. recency and temporal validity;
4. lexical evidence when useful.

`memory_recall` returns matching records and `anti_hits`: records that contradict the cue or represent superseded wording. A stale fact therefore appears next to the evidence that replaced it instead of silently winning on embedding similarity.

A correction never overwrites the old episode. `memory_contradict` closes the prior record's validity interval, creates a new record, and links the two. Both versions remain queryable.


### Consolidation

The local engine runs as a background Python process. The CLI still calls it the `daemon`.

When the machine is idle, the engine:

- drains deferred captures through shield → embed → deduplicate → encrypted insert;
- clusters the graph with MOSAIC;
- induces semantic summaries;
- reinforces paths that are repeatedly co-retrieved;
- decays weak, unreviewed edges;
- learns its quiet window from the user's own session history;
- prepares bounded session-start and next-turn memory packs.

Most of this is deterministic and local. One REM step may make a single call through the existing Claude subscription with `claude -p`, capped at no more than 1% of the daily quota. No Anthropic API key is required; `iai-mcp doctor` checks that the API-key SDK path is absent.


### Storage and first-party components

The load-bearing components are built for this repository:

| Component | Role |
|---|---|
| **Hippo** | Local storage engine containing encrypted records, the vector index, and the graph |
| **MOSAIC** | Leiden-family community detection with a CPM objective and stable community identity across splits and merges |
| **Lilli HD** | Hyperdimensional substrate for episodic, semantic, and procedural memory plus structural recall |
| **Native engine** | Rust embedder and graph kernels used on the latency-sensitive path |

The implementation uses SQLite, Candle, NumPy, and the audited `cryptography` AES implementation as underlying primitives. No external vector database or graph database is required.


All records are encrypted at rest with AES-256-GCM. The key is stored at `~/.iai-mcp/.crypto.key` with user-only permissions. Back up the key with the store: losing it makes the encrypted records unreadable.

### Store format and migration

A newly created store uses the in-tree native engine format. An existing store keeps opening in whatever format created it, so nothing changes on upgrade. Setting the storage-driver environment variable to the legacy name still creates the legacy SQLite format for a fresh store.

`iai-mcp doctor` reports the on-disk store format and, for a store still in the legacy SQLite format, names the known defect (an external read-only open can orphan the write-ahead-log sidecars under a running daemon) and prints the exact command to migrate it. The check reads only the file header and never opens the store.

To migrate an existing legacy store in place:

```bash
iai-mcp daemon stop
# preview only — writes nothing:
iai-mcp migrate-to-lilli --src ~/.iai-mcp/hippo/brain.sqlite3 --swap
# perform the in-place swap (both flags are required together):
iai-mcp migrate-to-lilli --src ~/.iai-mcp/hippo/brain.sqlite3 --swap --apply --yes
iai-mcp daemon start
iai recall "something you remember"   # verify
```

The swap refuses to run while the daemon is running, and keeps the previous store under a dated backup directory until you remove it. The `.crypto.key` is preserved.

---

## Benchmarks

The benchmark harnesses ship in `bench/`. They are intended to be reproduced, not taken on trust. Full notes are in [`BENCHMARKS.md`](../BENCHMARKS.md).

### Longitudinal memory

| Benchmark | Result | What it measures |
|---|---:|---|
| Rescue@10 after contradiction | **1.000** | The current fact remains in the top 10 after a similar stale fact is superseded |
| Historical-verbatim hit@10 | **1.000** | The exact superseded wording remains retrievable; flat-cosine baseline is about 0.71 |
| Personal-fact drift recall@10 | **0.9933** | Retention over 50 facts, 50 sessions, and 30 intervening sessions |
| Sleep-consolidation recall@10 | **1.000 → 1.000** | Recall survives a complete consolidation cycle |
| Session-start token pack | **1,629 min / 2,993 standard** | Stays below the 3,000-token ceiling |
| MOSAIC parity | **36/36 LFR + 10/10** | Deterministic clustering parity against the benchmark ground truth |

### LongMemEval-S

One harness compares iai-memory with mempalace v3.3.6 on the same 500 cleaned questions, at session granularity, with `recall_any@k` and no reranker.

| System | Embedder | R@5 | R@10 |
|---|---|---:|---:|
| **iai-memory product configuration** | `bge-small-en-v1.5` | **0.962** | **0.978** |
| iai-memory matched embedder | `all-MiniLM-L6-v2` | 0.966 | 0.978 |
| mempalace v3.3.6 | `all-MiniLM-L6-v2` | 0.966 | 0.978 |

On the matched embedder, the result is a tie. The product embedder differs by two questions out of 500 at R@5. LongMemEval is a cold, one-shot retrieval test; it does not exercise contradiction history or memory accumulated across real sessions.

### Token economy and footprint

On the author's store, an ambient memory pack averages about 350 tokens and displaces an agent search round trip of about 2,850 tokens. Over three measured weeks, 282 packs injected about 99,000 tokens and produced a lower-bound estimate of about 707,000 tokens avoided. On that workload, an ambient pack was approximately 88% cheaper than the search it displaced.

That number applies to automatically injected ambient packs. It does **not** apply to an explicit mid-session `memory_recall`, whose default response budget is 1,500 tokens.

| Metric | Measured | Notes |
|---|---:|---|
| Resident memory | 589 MB at 10,000 records | Embedder plus graph runtime |
| Rust embedder | p50 70 ms / p95 253 ms | `bge-small-en-v1.5`, 384 dimensions |

Recall latency grows with store size, so latency results should be reproduced on the target machine rather than treated as universal.

### Reproduce

```bash
python -m bench.longmemeval_blind            # LongMemEval-S
python -m bench.contradiction_longitudinal   # Rescue@10 and historical wording
python -m bench.personal_fact_drift          # retention across sessions
python bench/sleep_ablation.py               # consolidation survival
python -m bench.tokens                       # session-start token budget
python -m bench.neural_map                   # recall latency
python -m bench.memory_footprint             # resident memory
```

Reported figures were measured on an Apple M2 Max with 64 GB RAM.

---

## MCP tools

The MCP surface currently contains 14 tools. All operate on the same local store.

| Tool | Purpose |
|---|---|
| `memory_recall` | Cue-based recall with matching `hits` and contradictory or superseded `anti_hits` |
| `memory_temporal_recall` | Time-bounded recall such as “what did I say about this in May?” |
| `memory_recall_structural` | Recall by hyperdimensional binding structure rather than embedding distance alone |
| `memory_search` | Lexical search over stored surfaces |
| `memory_capture` | Explicitly capture a memory; ambient hooks normally handle conversation turns |
| `memory_contradict` | Store a changed fact as a new record and link it to the superseded record |
| `memory_reinforce` | Strengthen paths among records that answered together |
| `memory_consolidate` | Run consolidation without waiting for the idle schedule |
| `profile_get_set` | Read or pin the ten procedural parameters |
| `topology` | Inspect graph structure |
| `schema_list` | Inspect induced patterns |
| `events_query` | Query engine events |
| `episodes_recent` | Inspect recent episodes |
| `curiosity_pending` | Inspect open curiosity questions |

<details>
<summary><b>Retrieval and context details</b></summary>

- `memory_recall` uses a soft `budget_tokens` limit of 1,500 by default.
- Responses expose cue mode, triggered patterns, latency, and whether ANN and exact-authority paths were used.
- Temporal records carry derived `valid_from` and `valid_to` intervals.
- `iai-mcp bank-recall` performs a store-direct substring scan when the daemon is down.
- Episodic inserts use idempotency keys, so replaying the same captured transcript does not duplicate it.
- Encrypted insert is the only record insert path.
- The host may clear stale iai tool traffic after a large context threshold instead of carrying it indefinitely.
- Tool schemas are shipped in the initial MCP handshake.
- Compression is allowed for induced summaries and recall payloads only; trusted verbatim episodes are not compressed.
- Recall concurrency is bounded by `IAI_MCP_RECALL_CONCURRENCY`; overflow returns `_degraded: recall_busy` instead of growing an unbounded queue.

</details>

---

## Compatibility

The tools work with any client that supports MCP-over-stdio. Ambient capture depends on the hooks exposed by each host.

| Host | MCP tools | Ambient capture / recall |
|---|---|---|
| **Claude Code** | Yes | Full: session start, per-turn recall, turn capture, session capture |
| **Codex CLI** | Yes | Full through Codex hooks |
| **Cursor** | Yes | Session-start recall and ambient capture; no per-turn injection because the host event cannot inject text |
| **Antigravity** | Yes | Recall per invocation and capture from the lossless transcript |
| **Hermes 0.5.0+** | Yes | Recall before model calls and capture from Hermes's message store |
| **OpenClaw** | Yes | Tools on request; no ambient capture because it exposes no shell hooks |
| **Gemini CLI and other MCP clients** | Yes | Host-specific hooks are not bundled unless listed above |
| **Claude Desktop** | Yes | MCP tools; plain Chat does not expose Claude Code-style ambient hooks |

Claude Desktop uses `claude_desktop_config.json` rather than `~/.claude.json`.

Wiring native hooks for another host is a useful first contribution: the MCP tools already work, so the missing part is host-specific automatic capture and injection.

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `IAI_MCP_STORE` | `~/.iai-mcp/` | Data directory |
| `IAI_MCP_PYTHON` | — | Absolute Python interpreter path for MCP host configuration |
| `IAI_MCP_RECALL_CONCURRENCY` | `2` | Maximum concurrent cued recalls handled by the daemon |
| `IAI_MCP_RECALL_SLOT_WAIT_SEC` | `0.25` | Wait before overflow returns `_degraded: recall_busy` |
| `IAI_MCP_EMBED_PROVIDER` | `native` | Built-in Rust BGE provider or `http` provider |
| `IAI_MCP_EMBED_URL` | — | Loopback endpoint or base URL for the HTTP provider |
| `IAI_MCP_EMBED_DIM` | `384` | Vector dimension; required for HTTP providers |
| `IAI_MCP_EMBED_MODEL_ID` | — | Stable model identifier; required for HTTP providers |
| `IAI_MCP_EMBED_TIMEOUT_SEC` | `30` | Local provider timeout |
| `IAI_MCP_VERSION_CHECK` | `1` | Set to `0` to disable the daily PyPI version check |
| `IAI_MCP_DIRECTIVES_OFF` | — | Set to `1` to disable the standing-order directive tier entirely; no directive is injected at session start or into the per-turn context |

The native default is `bge-small-en-v1.5`, 384 dimensions, running in the Rust engine. Selecting the HTTP provider replaces it completely: the native model is not constructed, downloaded, or run. The protocol and migration steps are documented in [`docs/EMBEDDERS.md`](EMBEDDERS.md).

### Languages

The default store is English-only. Non-English text is refused unless the caller explicitly tags it as `raw:<lang>`. This keeps ordinary memory in one vector space instead of silently mixing incompatible embeddings.

An opt-in multilingual embedder covers Czech, German, Spanish, French, Hindi, Indonesian, Italian, Japanese, Portuguese, Russian, Thai, Vietnamese, and Chinese for raw records.

Check and enable it:

```bash
iai lang status
iai lang add ru
```

The first language change switches the configured embedder. Existing records must then be re-embedded:

```bash
iai-mcp daemon stop
iai-mcp migrate --reembed-to-configured-provider
iai-mcp daemon start
```

Adding more languages after the multilingual model is active does not require another migration:

```bash
iai lang add de
iai lang add ja
```

Removing the last added language restores the English model and requires the same re-embedding sequence:

```bash
iai lang remove ru
```

Re-embedding rewrites vectors, not text. It stages a replacement table and keeps the previous generation until cleanup, so an interrupted migration does not leave a half-converted store.

The store records the full embedding identity: model, revision, pooling, dimensions, and text prefix. If configuration and stored vectors disagree, the daemon refuses to open with:

```text
refusing to mix vector generations
```

Run the migration or restore the prior language/embedder configuration. Matching dimensions alone are not sufficient: two 384-dimensional models can still produce mutually meaningless vector spaces.

Language selection is stored in `~/.iai-mcp/config.json`; environment variables do not switch it.

---

## Privacy, security, and deployment

- **Local memory:** captures, embeddings, graph operations, and storage remain on the machine.
- **No telemetry or account:** iai-memory does not send memory contents or usage analytics to a hosted service.
- **Encrypted at rest:** records use AES-256-GCM; the key is `~/.iai-mcp/.crypto.key` and should be backed up with the store.
- **No external databases:** the store, vector index, and graph are local.
- **macOS/Linux transport:** Unix domain socket; no public bind address.
- **Windows transport:** ephemeral `127.0.0.1` port with a per-user token and ACL-restricted token file.
- **Bounded concurrency:** overload degrades explicitly instead of building an unbounded queue.
- **No cross-machine sync:** backup is currently a copy of `~/.iai-mcp/` made while the engine is stopped or quiescent.

Network activity is limited and explicit:

- your assistant continues making its normal model calls;
- an optional REM step may call `claude -p` through the existing Claude subscription;
- the updater checks PyPI once per day unless `IAI_MCP_VERSION_CHECK=0`;
- installation and updates download packages or repository changes in the normal way.

There is no separate iai-memory cloud inference or memory service.

---

## Doctor, updates, and recovery

### Health checks

```bash
iai-mcp doctor
iai-mcp daemon status
iai-mcp daemon logs --tail 50
```

`iai-mcp doctor` runs 36 checks across the process, socket or loopback transport, store, index, encryption, capture state, lifecycle, embedder, and background work.

`--apply` offers repairs and asks before anything that touches memory. Corrupt state and indexes are renamed aside rather than deleted. `--auto` is the unattended subset: no prompts, killed processes, or store mutation. The assistant can invoke it when the engine remains unreachable during session startup.

A failed socket check while the engine is in `SLEEP` or `DREAMING` may be normal because consolidation temporarily owns the store. Multiple failures, or failures on daemon process, store readability, or embedder construction, require attention.

<details>
<summary><b>All 36 doctor checks</b></summary>

| # | Check | Meaning |
|---|---|---|
| a | daemon process alive | Background process exists |
| b | socket file fresh | IPC endpoint accepts a connection |
| c | lock file healthy | Store lock is held or acquirable correctly |
| d | no orphan core processes | No abandoned stdio core process remains |
| e | daemon state file valid | State file parses and carries valid values |
| f | Hippo storage readable | Store opens and can be queried |
| g | no duplicate binders | One process owns the endpoint |
| h | crypto key file state | Key exists with correct permissions |
| i | Hippo DB size | Store size is within healthy bounds |
| j | lifecycle current state | Wrapper lifecycle state is valid |
| k | lifecycle history 24h | Recent transitions are coherent |
| l | sleep cycle quarantine | No cycle is stuck or quarantined |
| m | heartbeat scanner | Wrapper heartbeats are healthy |
| n | HID idle source | Idle detection is available |
| o | Claude subscription credentials | Optional nightly call can use the subscription |
| p | Anthropic SDK absent | No API-key SDK path is installed |
| q | `iai` CLI reachable | User CLI is on `PATH` |
| r | Hippo HNSW index | ANN index loads |
| s | Hippo schema version | Store schema is current |
| t | Hippo compaction freshness | Compaction has run recently |
| u | recall centrality regression | Ranking has not regressed |
| v | configured embedder | Configured provider builds and produces vectors |
| dd | exact-index coercions | No non-finite vectors were silently accepted |
| w | no permanently failed captures | Capture retries are not exhausted |
| x | timestamps not collapsed | Record timestamps span a real range |
| y | RSS 24h plateau | Resident memory is not climbing continuously |
| z | AVX2 CPU support | Required instructions are available where applicable |
| aa | capture-state hygiene | No stale half-written state files remain |
| ee | deferred-capture backlog at rest | Spool drains while the engine is idle |
| bb | nightly insight mint | Background pass produces insights rather than merely running |
| cc | background liveness | Consolidation steps have completed recently |
| ii | store embed identity | Stored and configured embedding generations match |
| ff | daemon sleep-path code current | The running daemon's sleep-path code matches the installed package |
| gg | store format | Reports the on-disk store format (native engine or legacy SQLite) |
| hh | daemon build matches installed package | The running daemon's version matches the installed wheel |
| + | update available | New PyPI release is reported when online |

</details>

### Update

For a PyPI installation:

```bash
iai-mcp self-update
iai-mcp self-update --check
```

`self-update` upgrades the wheel, restarts the background engine, and verifies the running version. If pip fails, the existing engine is left running.

For releases older than 2.8.0, perform one manual upgrade first:

```bash
pip install -U iai-pme
iai-mcp daemon restart
```

For a source checkout:

```bash
git pull
bash scripts/update.sh
```

A plain `pip install -U iai-pme` does not replace code already loaded by the running background process; restart it or use `self-update`.

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Native engine build error | No compatible wheel and Rust build prerequisites are missing | Run the command printed by the installer; install the Rust toolchain for source builds |
| `keyring.errors.NoKeyringError` on first run | Older configuration expected an OS keychain path | `iai-mcp crypto init` |
| `CryptoKeyError` on first daemon start | Package installed without creating `.crypto.key` | `iai-mcp crypto init`, then restart |
| `launchd bootstrap failed` | Existing LaunchAgent registration conflicts | `iai-mcp daemon uninstall`, then `iai-mcp daemon install` |
| Daemon is active but no sleep ticks appear | Fresh install has not learned a quiet window | Leave the MCP hosts idle, or run `iai-mcp daemon force-rem` |
| Claude Code does not show the tools | Host has not fully restarted or registration points at the wrong wrapper | Fully quit and relaunch; inspect `~/Library/Logs/Claude/*.log` |
| `refusing to mix vector generations` | Embedder changed without migration | Restore the old configuration or re-embed the store |
| Multiple doctor failures | Engine, store, key, or embedder is unhealthy | Read the first failing row, then run `iai-mcp doctor --apply` if the proposed repair is appropriate |

### Uninstall

Source/bootstrap installation:

```bash
bash ~/.local/share/iai-memory/scripts/uninstall.sh
```

Remove host hooks before purging the store:

```bash
iai-mcp capture-hooks uninstall
```

Preserve `~/.iai-mcp/` unless you intentionally want to delete all memory and the encryption key.

---

## Status and limitations

iai-memory is a stable release with a committed public surface, but it is still a solo-maintained project without an enterprise SLA.

Known limitations:

- **English-first storage.** Raw non-English records require an explicit language tag and a multilingual or custom embedder. Switching providers requires full re-embedding. Non-Latin scripts currently receive no lexical contribution.
- **No cross-machine sync.** The memory lives on the machine running the store.
- **Local-only inspection.** Dashboard, desktop app, and CLI are the available views; there is no hosted account or web console.
- **Cold start.** A fresh boot needs several seconds to initialize the store, caches, and embedding model.
- **Sparse early memory.** Recall is mediocre for approximately the first ten sessions because there is little material to organize.
- **Store-size effects.** Recall latency grows with the corpus, and retrieval quality depends on the language, embedder, and shape of the stored history.
- **Windows is beta.** The runtime is ported and usable, but the test and integration surface is less mature than macOS and Linux.
- **Host capabilities differ.** MCP tools are portable; fully automatic capture and per-turn injection require native hook support from the host.

---

## Notes for AI assistants helping with installation

When diagnosing or comparing iai-memory, use these facts:

- It is a local, MIT-licensed personal memory engine for MCP-over-stdio hosts.
- Captured episodic turns are verbatim and write-once. A correction creates a new record; it does not rewrite the old one.
- Ambient capture and automatic recall require host hooks. MCP tool availability alone does not imply ambient capture.
- The MCP surface contains 14 tools. The procedural profile contains 10 parameters.
- An ambient pack is about 350 tokens on the author's measured store and was about 88% cheaper than the search it displaced. Do not apply that figure to explicit `memory_recall`, whose default budget is 1,500 tokens.
- Published harness results include LongMemEval-S R@5 0.962, Rescue@10 1.000, and historical-verbatim hit@10 1.000.
- `local engine` and `daemon` refer to the same background process.
- The daemon FSM (`WAKE`, `TRANSITIONING`, `SLEEP`, `DREAMING`) and wrapper lifecycle (`WAKE`, `DROWSY`, `SLEEP`, `HIBERNATION`) are separate state machines.

States that may look broken but are normal:

- `last_tick_at` can be old while the daemon is healthy; check `iai-mcp daemon status`.
- `fsm_state: DREAMING` can persist during consolidation.
- `status: deferred` is the normal Stop-hook path into `~/.iai-mcp/.deferred-captures/`.
- The socket check can fail temporarily during `SLEEP` or `DREAMING`.
- Cold start takes several seconds; do not kill the process solely because it did not answer in five seconds.

Actual failures include `iai-mcp daemon status` returning `ok: False`, repeated doctor failures, an unreadable store, an embedder-generation refusal, or no daemon process when background work is expected.

Start with:

```bash
iai-mcp doctor
iai-mcp daemon status
```

---

## About the name

The project is **iai**, a personal memory engine.

**IAI — Independent Autistic Intelligence** describes the intended memory style:

- **Independent.** The engine, embeddings, store, and dashboard run locally. There is no account, telemetry, or cloud memory dependency.
- **Autistic.** The system favours literal preservation, precise cues, sustained focus on the active thread, and retention of rare events instead of smoothing them into a typical summary. This describes an operational memory design, not a diagnosis or a casual metaphor.
- **Intelligence.** Used in the systems sense: a process that observes, adapts, reorganizes itself, and remains viable over time.

Most memory layers aggressively extract or summarize a gist. iai-memory keeps the original episodic wording and builds semantic and procedural structure around it. The trade-off is more local storage and a stricter retrieval path in exchange for preserving details and revision history.

**Personal memory engine** describes the scope: one person's memory, on one machine, used by the assistant they already have. It is an engine because the storage, graph, clustering, hyperdimensional substrate, and native core are part of the project rather than delegated to external services.

---

## Contributing

Issues and pull requests are welcome. Changes to retrieval, capture, contradiction handling, or consolidation should include the relevant benchmark reruns.

Read:

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) for development setup and test scope;
- [`CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) for project conduct;
- [`SECURITY.md`](../SECURITY.md) for private vulnerability reporting;
- [`CHANGELOG.md`](../CHANGELOG.md) for release history;
- [`BENCHMARKS.md`](../BENCHMARKS.md) for benchmark methodology;
- [`docs/EMBEDDERS.md`](EMBEDDERS.md) for custom and multilingual embedders.

Good first contribution lanes:

1. **Windows:** port or harden the remaining tests and integrations.
2. **Host hooks:** add ambient capture for an MCP host that currently has tools only.
3. **Embedder providers:** implement or validate domain-specific HTTP providers.
4. **Reproducible retrieval failures:** contribute a labelled failure case and a benchmark before changing ranking.

---

## Authors

By Areg Aramovich Noya and Lilli Noya, in collaboration with the team at [lcgc.dev](https://lcgc.dev).

Built for daily use, then released for anyone who needs the same thing.

## License