# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.3.0] — 2026-07-13

### Added

- **Pluggable embedding provider (opt-in).** You can now point iai at your own
  local embedding model over a loopback HTTP service instead of the built-in
  English BGE, which enables multilingual or domain-specific recall. The native
  BGE stays the zero-config default with unchanged weight and speed; the provider
  is strictly opt-in, loopback-only (`127.0.0.1`), with strict response
  validation and a resumable, zero-failure re-embedding migration. See
  `docs/EMBEDDERS.md`. Contributed by Marsu6996.

## [2.2.2] — 2026-07-12

### Fixed

- **No duplicate startup work.** The daemon no longer warms its dispatch surface
  twice at boot, and a cancelled startup cleans up its locks properly.
- **Cleaner graph.** A stale edge can no longer resurrect a deleted record as an
  empty graph node.
- **Path classification.** IAI-MCP checkouts are classified by exact path
  component instead of a loose substring match.

### Changed

- **Faster topology on large graphs.** Shortest-path work is bounded above ~2,500
  nodes with a deterministic estimator (within 0.2% of exact), avoiding a large
  N×N allocation.
- **Reproducible test suite.** The default suite no longer depends on
  machine-specific state, so it runs cleanly in fresh environments.

All of the above contributed by Marsu6996.

## [2.2.1] — 2026-07-12

### Fixed

- **Right context, not stale replays.** Backlog replay no longer injects old
  conversations as the active task, per-message context injection works on the
  Rust engine again, and captures still feed the freshness surfaces when the
  daemon is down.
- **Lighter capture.** The Stop hook captures one response per turn instead of
  re-spooling the whole transcript every time.
- **Steadier daemon under load.** The boot background fleet defers instead of
  stampeding (which could push memory past the watchdog's limit and kill the
  daemon), the watchdog honors cold-start grace and no longer races its own
  shutdown, recent-window writes survive torn lines across processes, and a
  slow-exiting worker keeps its result instead of discarding it.

## [2.2.0] — 2026-07-11

### Added

- **Linux idle sensing.** On Linux the daemon now reads idle state from
  systemd-logind (IdleHint / IdleSinceHint), mirroring the macOS HID-idle check
  and working across X11 and Wayland. Contributed by joerybka.

### Fixed

- **More accurate timestamp re-derivation.** `migrate --rederive-timestamps` no
  longer matches an internal transcript event (e.g. a queue operation) ahead of
  the real user or assistant turn, so re-derivation stops silently doing nothing
  on collapsed-timestamp groups. Contributed by joerybka.

## [2.1.0] — 2026-07-11

### Added

- **Windows support.** The engine now runs on Windows end to end. The Rust storage
  engine compiles and runs there through a cross-platform positional-I/O layer, and
  the daemon talks over a local TCP loopback with a per-start token handshake in
  place of a Unix socket. Community port contributed by danielhertz.

### Fixed

- **Reliable liveness checks.** Every process-alive check now goes through one
  canonical probe, so a healthy daemon is never misreported as dead (previously
  possible on Windows).
- **Foresight on live turns only.** The anticipation pack now refreshes only on a
  live turn, not while replaying historical events, so replay no longer thrashes
  next-turn anticipation.

## [2.0.3] — 2026-07-11

### Fixed

- **Windows daemon compatibility.** The background daemon now starts, stops, and
  reports its own liveness correctly on Windows — shutdown terminates the whole
  process tree so the store lock is released, the encryption key file is written
  in binary mode and replaced atomically, and file locking uses a Windows-native
  region lock so the store package imports and runs.
- **Duplicate-daemon race.** The lifecycle lock is now claimed atomically, so two
  processes starting at the same time can no longer both become the daemon.
- **Schema confidence gate.** Mid-confidence memory-schema patterns now correctly
  reach the user-approval path instead of being dropped.

## [2.0.2] — 2026-07-11

### Added

- **Desktop app downloads.** Prebuilt desktop binaries for macOS, Linux, and
  Windows are now attached to each release (unsigned; on first launch macOS and
  Windows may ask you to confirm an unidentified developer).

## [2.0.1] — 2026-07-10

### Changed

- **Lower idle CPU.** Corpus counters now update incrementally instead of
  rescanning, fresh readers inherit the writer's sort index instead of
  rebuilding it, and the graph cache no longer re-streams the whole corpus on
  ambient captures — cutting background rebuild churn substantially.

## [2.0.0] — 2026-07-10

### Changed

- **New Rust storage and graph engine.** The memory store and graph now run on a
  native Rust engine, replacing the previous Python/hnswlib backend. Existing stores
  migrate automatically on first run — no manual steps.

### Added

- **Desktop app.** A local dashboard to watch the memory graph live, add a memory,
  and hint records into the forgetting queue.
- **Document study/teach**, anticipatory foresight packs, and combined lexical +
  semantic search lanes.

### Removed

- The last GPL-adjacent dependencies; the project now ships its own storage and
  graph engines under MIT.

### Fixed

- **Dashboard graph view** stays centered on zoom and pan, with a recenter
  control that appears when the view leaves its home position.

## [1.2.1] — 2026-06-30

### Fixed

- `capture-hooks install` no longer wires the wheel-bundled MCP wrapper that ships
  without its Node dependencies. That copy could not resolve `@modelcontextprotocol/sdk`,
  so the `iai-mcp` MCP entry failed to start after install; install now prefers a
  runnable wrapper. (#26)
- HNSW boot-integrity check compares the live index element count instead of the
  already-repopulated label-map size, so a stale on-disk index is rebuilt rather
  than silently kept.
- The SessionStart cache refreshes during active (WAKE) operation, not only after a
  sleep cycle, so long-lived daemons serve current context.
- macOS test suite: socket integration tests use short paths (Darwin `AF_UNIX`
  limit) and corrected fixtures, restoring a green macOS CI run.

### Added

- `capture-hooks install --components` to opt out of re-registering the SessionStart
  recall hook instead of always putting all three hooks back. (#26)

Thanks to @danielhertz1999-bit and @Marsu6996.

## [1.2.0] — 2026-06-26

### Added

- **Windows support (beta).** The daemon and CLI now run on Windows: Unix-domain
  sockets fall back to authenticated TCP loopback, `fcntl` / `resource` / `signal`
  calls are shimmed or guarded, the daemon installs via Task Scheduler, and
  crypto-key permissions are set with `icacls`. The MCP wrapper dials the daemon
  over the same transport. The runtime is ported and validated on Windows 11; the
  test suite is still being ported, so Windows is beta for now. Thanks to
  @danielhertz1999-bit for the port, and to @warplayer for thorough Windows 11
  validation.

## [1.1.7] — 2026-06-22

### Added

- `scripts/install-linux.sh` — a one-shot Fedora / RHEL / Debian setup helper:
  prerequisite checks (Python ≥3.11, Rust, Node ≥18), venv creation, editable
  install, MCP-wrapper build, crypto-key init, and systemd user-service install.
  Idempotent and safe to re-run. Thanks to @MoppelMat.

### Fixed

- Dropped two ineffective directives (`StartLimitIntervalSec`, `StartLimitBurst`)
  from the systemd unit's `[Service]` section. systemd only honors those keys in
  `[Unit]`, so in `[Service]` they had no effect and emitted a warning on modern
  systemd. Thanks to @MoppelMat.

## [1.1.6] — 2026-06-21

### Fixed

- **Daemon memory and CPU under sustained load.** On large stores the background
  daemon's warm state and nightly consolidation could climb in resident memory
  and spin the CPU. This release isolates the runtime-graph rebuild in a
  spawn-context worker, computes graph centrality with a bounded sampled
  estimator (so it never recomputes exact betweenness in-process at scale),
  streams record reads instead of materializing the whole corpus, drains the
  deferred-capture backlog in two phases (insert first, embed later) with
  self-limiting safety rails, and grades the memory watchdog against the kernel's
  physical-footprint metric rather than raw resident set. Warm memory now stays
  well under the cap and the consolidation CPU storm is gone. No changes to the
  public API, CLI, MCP tools, or on-disk store format.

## [1.1.5] — 2026-06-21

### Security

- **Deferred-embed pending rows are now encrypted at rest.** On an encrypted
  store, a record awaiting background embedding briefly held its text
  (`literal_surface`) and provenance in plaintext during the embed window.
  Pending rows are now encrypted on write and decrypted just before embedding,
  matching the rest of the at-rest encryption. Unencrypted stores are
  unaffected.

### Fixed

- **Sleep daemon WAKE/idle CPU storm.** The consolidation cycle could spin the
  CPU and never settle; the daemon now serves recall on wake instead of leaving
  it unserved. Thanks to @Marsu6996.
- **Consolidation and recall correctness.** Tombstoned records are excluded from
  the runtime graph, crisis-mode topology is built on the live graph, recall
  scores are clamped to a valid range, and reflection embedding plus crash
  recovery are hardened — closing the remaining root causes behind the
  crisis-mode loop. Thanks to @Marsu6996.

## [1.1.4] — 2026-06-21

### Fixed

- **`migrate --reembed-from-text` repaired nothing on bulk-loaded stores.** The
  version added in v1.1.3 fetched each record through a path that returned
  nothing on stores populated in bulk, so it re-embedded zero records and exited
  reporting success — a silent no-op. It now reads records directly, actually
  re-embeds from the stored text, and is resumable with bounded memory use.
  **If you ran the migration on v1.1.3, run it again on v1.1.4** — your vectors
  were not repaired. Throughput is embedder-bound (no batch speedup yet), so a
  large store takes a while; the run is resumable and reports progress.

## [1.1.3] — 2026-06-21

### Fixed

- **Ambient capture embedded the cue label instead of the message.** The
  session-capture path embedded a positional provenance label
  (`"session <id> turn <n>"`) rather than the message content, so stored
  vectors collapsed and semantic recall degraded on any store built through the
  session hook. Capture now embeds the message content; the stored text
  (`literal_surface`) was never affected. Run
  `iai-mcp migrate --reembed-from-text` once after upgrading to repair vectors
  written before this fix. Thanks to @Marsu6996 for the report and fix.
- **Data loss under parallel transcript imports.** `write_deferred_captures`
  wrote in place to the final filename, so a concurrent drain could read a
  half-written file and quarantine it as permanently failed. Writes are now
  atomic (temp file + `os.replace`). Thanks to @gardinermichael for the report.
- **Sleep daemon could stall in crisis mode.** Interrupted consolidation steps
  now record the underlying error instead of a bare deferred marker; recall
  degrades honestly instead of serving stale schema-dominated results while a
  cycle is stuck; and crisis mode auto-clears after 72 hours. A watchdog now
  emits an alert when the sleep cycle stops completing.

### Added

- `iai-mcp migrate --reembed-from-text` — re-embeds existing episodic records
  from their stored text to repair vectors written before the capture fix above.
  Idempotent; supports `--dry-run`, `--resume`, `--rollback`, and
  `--reembed-batch-size`.
- `iai-mcp migrate --salvage-torn-permanent-failed` — recovers complete records
  from torn `.permanent-failed-*.jsonl` capture files and quarantines the
  originals.
- `iai-mcp deferred-unlock-dead-pids` — releases deferred-capture files left
  locked by a process that is no longer running. Run while the daemon is
  stopped.

## [1.1.2] — 2026-06-17

### Fixed

- **macOS Keychain credentials for nightly consolidation.** When `claude /login`
  stores OAuth credentials in the macOS login Keychain instead of
  `~/.claude/.credentials.json` (the file is absent on a normal desktop-app
  setup), the subscription check now falls back to the Keychain item, so the
  nightly `claude -p` path is found. Added `IAI_MCP_CLAUDE_BARE=0` to drop the
  `--bare` flag for setups where `claude --bare -p` reports "Not logged in"
  while plain `claude -p` authenticates. Default behavior unchanged.

## [1.1.1] — 2026-06-15

### Fixed

- Linux runtime fixes for the experimental Linux path: daemon binder detection
  gains an `ss` fallback, ANN query distances are clamped to a valid range, and
  the multiprocess store path no longer races on table visibility. Linux remains
  experimental and unvalidated end-to-end — testing and port feedback are welcome.

### Changed

- Internal cleanup only — no changes to the public API, the `iai-mcp` / `iai`
  CLI, the MCP tool set, or the on-disk store format: the sleep-pipeline
  compatibility shim was removed in favour of the canonical module, sleep-step
  names were made consistent, and more wall-clock benches are gated out of the
  default test run.

## [1.1.0] — 2026-06-14

### Added

- **Experimental Linux support.** The native engine now builds on Linux (the
  Rust extension no longer hard-depends on a macOS-only acceleration backend),
  the daemon installs as a systemd user service, and the capture/recall hooks
  run on POSIX shells. `scripts/install.sh` handles the Linux path and the
  README documents the extra build prerequisites. Validated on macOS; Linux is
  code-complete but not yet validated end-to-end — testing and port feedback
  are welcome.

### Changed

- **Source restructured into focused packages.** The largest modules — `cli`,
  `store`, `daemon`, `hippo`, `doctor`, `migrate`, and `core` — are now packages
  with concern-grouped sub-modules instead of single large files. The public API,
  the `iai-mcp` / `iai` CLI surface, the MCP tool set, and the on-disk store
  format are unchanged; this is an internal reorganization that makes the storage,
  daemon, community-detection, and migration layers easier to read and navigate.
- Background daemon and graph-cache rebuild paths gained additional
  resource-isolation and reliability hardening.

## [1.0.3] — 2026-06-11

### Fixed

- MCP `tools/list` no longer stalls ~5 seconds when the daemon is down: the
  wrapper connects the MCP transport first and wakes the daemon in the
  background, so tool discovery answers from the static registry immediately.
- Shell scripts (`scripts/install.sh` and siblings) ship with the executable
  bit set; `./scripts/install.sh` works without a `bash` prefix.
- The wrapper test runner no longer hangs after the suite finishes (reconnect
  socket and timer are unref'd; teardown reconnects are suppressed).
- Store teardown is more deterministic: a reference cycle between the store
  and its database handle was broken.
- On machines without the optional LongMemEval dataset or a freshly built
  native extension, the affected tests now skip instead of failing.

### Added

- Session capture keeps full transcripts: the per-session turn ceiling was
  raised to 100 000 turns.
- `iai-mcp migrate --rederive-timestamps` repairs legacy records whose
  timestamps collapsed to a single import time.
- Doctor: a new check flags time-collapsed episodic sessions, and the daemon
  writes an audit event when it was respawned by the doctor.
- Typed stubs for the native extension (`iai_mcp_native.pyi`) ship in the
  wheel and the source tree.

### Changed

- launchd: the daemon installs as always-on (`RunAtLoad=true`, restart on
  crash) instead of socket-activated. The daemon starts at login and is
  immediately available to the first session.

### Removed

- The experimental summary-compression module and its optional `[compress]`
  extra. The path was a transparent passthrough fallback; removing it drops a
  ~2.3 GB optional model dependency.

## [1.0.2] — 2026-06-07

### Fixed

- Packaging: the launchd plist, systemd unit, and capture/recall hooks now ship
  inside the wheel (under `iai_mcp/_deploy/`) and are resolved via
  `importlib.resources`. `iai-mcp daemon install` and hook setup no longer fail
  on a clean `pip install`.
- Python/CLI path resolution: the MCP server config and capture hooks now use the
  running interpreter (`sys.executable`) and resolve the `iai-mcp` CLI via `PATH`,
  so installs under pyenv and non-default layouts work.

## [1.0.1] — 2026-06-06

### Fixed

- Daemon RSS crash-loop: `find_record_by_tag` no longer materializes the full
  records table on every capture-dedup probe; it now uses a targeted SQL query.
  (Fixes high-RSS kill/respawn cycles on large stores.)

## [1.0.0] — 2026-06-04

First stable release. The architecture has settled and the public surface — the
MCP tool set and the on-disk store — is committed-to from here on. SemVer-major
bump from `0.4.2`.

### Added

- **Hippo storage engine** — a single encrypted local store holding records, the
  vector index, the graph, and the event ledger together, built on SQLite +
  `hnswlib` + AES-256-GCM. Replaces the previous embedded vector database.
- **Native Rust engine** (`iai_mcp_native`) — the text embedder and the graph
  kernels (centrality, clustering, connectivity) run as a compiled Rust
  extension. Built automatically during `pip install` via `setuptools-rust`;
  `iai-mcp build-native` rebuilds it in place.
- **MOSAIC community detection** — original MIT-licensed, pure-Python + Numba
  algorithm written for the memory-graph workload (small graph, heterogeneous
  edge weights, re-clustered every sleep cycle) with a calibrated quality floor,
  a hyper-fragmentation guard, and per-community lineage across consolidation.
- **Lilli HD substrate** — hyperdimensional memory representations (BSC / FHRR /
  sparse VSA) backing the episodic / semantic / procedural tiers, with
  structural recall by the shape of a memory at zero LLM cost.
- **Queryable cross-session episodic recall** — turns are captured verbatim and a
  relevant slice is surfaced at the start of each new session; recent turns are
  also queryable directly through the `iai` CLI and the `episodes_recent` tool.
- **`iai` user CLI** — `iai recall` / `capture` / `ask` / `status`, driven from
  any shell, separate from the operator-side `iai-mcp` CLI. Falls back to an
  offline scan when the daemon is down.
- **Subscription-billed consolidation** — the nightly LLM step runs through your
  existing Claude subscription via `claude -p`; no API key, capped at ≤1% of the
  daily quota.
- **Export / backup / restore CLI** — full data portability of the store, crypto
  key, and config.
- **Write-ahead log for destructive sleep operations** — consolidation and
  pruning steps are journaled and resume across a crash.
- **Typed exception hierarchy** — narrowed error handling across the daemon and
  pipeline.
- **`iai-mcp doctor`** — 23 health checks across the daemon, the store, the
  native engine, and the subscription credential path.

### Changed

- **Storage** moved from the previous embedded vector database to Hippo.
- **Embedder** is now the native Rust embedder (English-only, 384-dimensional),
  built locally — no large Python ML runtime is installed.
- **Graph algorithms** run through the native Rust engine plus a pure-numpy
  rich-club helper instead of a third-party graph library at runtime.
- **Install** is pip-native: `pip install` compiles the native engine through
  `setuptools-rust`. There is no shell install script.
- **Graph centrality** is computed unweighted.
- **Record schema** carries hyperdimensional tier fields; migration from an
  older store is idempotent.

### Removed

- The previous embedded vector database from the runtime path — it now installs
  only via the one-time `migration` extra to import a legacy store.
- The PyTorch-based embedding stack (`sentence-transformers`, `torch`) — replaced
  by the native Rust embedder.
- The third-party hyperdimensional-computing dependency — replaced by the in-tree
  Lilli HD substrate.
- The third-party graph library from the runtime path — it remains a test-only
  oracle in the `dev` extra.
- Language auto-detection — the store is English-only by design.
- `pydantic` and `structlog` — unused; replaced by the standard library.
- The API-key SDK path — the daemon never calls a paid token API; consolidation
  is subscription-only.

### Fixed

- Recall hot-path latency and daemon responsiveness under load (state I/O moved
  off the event loop).
- A range of store, migration, and consolidation stability issues surfaced while
  hardening the new storage and native-engine paths.

### Security

- All records encrypted at rest with AES-256-GCM; the key is local
  (`~/.iai-mcp/.key`, mode 0600). No telemetry, no cloud dependency, and no API
  key stored or required by the daemon.

### Migration

Existing installs with data in the legacy store must import it once before the
first `1.0.0` start:

```
pip install ".[migration]"
python scripts/migrate_lance_to_hippo.py
```

The script backs up the old data before any writes and verifies byte-for-byte
before removing it.

## [0.4.2] — 2026-05-14

### Added

- **Update-check SessionStart hook** (`iai_mcp/_deploy/hooks/iai-mcp-update-check.sh`): on new session startup, compares the installed version against the latest GitHub release. Prints one line when an update is available; silent otherwise. Result cached for 6 hours; fetch runs in a detached background subshell so session startup is never blocked.
- `capture-hooks install` now registers the update-check hook alongside capture and recall hooks. `capture-hooks uninstall` and `capture-hooks status` handle it symmetrically.

## [0.4.1] — 2026-05-14

### Fixed

- **GIL contention between REM cycles and MCP requests**: `_tick_body` now breaks the REM loop when `mcp_socket` reports active connections or recent activity (within the 30 s interrupt window). Previously, the SLEEP-state `interrupt_check` in `lifecycle_tick` covered only the new-lifecycle path; the legacy `_tick_body` REM loop could hold the GIL through consecutive cycles, blocking `memory_recall` responses.
- **`INTERRUPT_RECENT_ACTIVITY_WINDOW_SEC` promoted to module scope** so both `_tick_body` and `lifecycle_tick` reference the same constant. Previously duplicated as a local inside `main()`.

### Added

- **Session-capture hook**: `IAI_MCP_SESSION_CAPTURE_CLI` environment variable for developer-override of the CLI binary path. CLI lookup now uses a bash array instead of a backslash-continuation for-loop (mirrors the session-recall hook change in 0.4.0).
- 2 new regression tests covering the MCP-yield branch (active vs. idle socket scenarios).

## [0.4.0] — 2026-05-13

### Added

- **Memory bank** — denormalized read-side caches under `~/.iai-mcp/.memory-bank/`. Two tiers:
  - `processed/salience-top-N.jsonl`: daemon writes the top-1000 records by graph-centrality salience once per REM-loop completion. Plaintext JSONL with base64-encoded embeddings.
  - `recent/window-YYYY-MM-DD.jsonl`: each drained capture is mirrored as an AES-256-GCM encrypted JSONL line. AAD is bound to the window-file's date string so a cold reader can decrypt without knowing any record id. Retention sweep (default 30 days) runs at the end of every drain pass.
- **New CLI command `iai-mcp bank-recall`** — substring fallback over the bank tiers without booting the daemon or loading the embedder. Returns a `memory_recall`-shaped JSON response so the wrapper's socket-dead fallback path is wire-compatible.
- **FSM drift detection** (`fsm_reconcile.py`): daemon startup compares the canonical `lifecycle_state.json` and legacy `.daemon-state.json`; a mismatch emits a `fsm_drift_detected` warning event. Detect-only — no auto-correction.
- **Backup archiver** (`archive_backups.py`): daemon startup moves any leftover `lifecycle_state.json.HIBERNATION-stuck*.bak` recovery artifacts into `~/.iai-mcp/archive/` with mtime-stamped names. Idempotent and fail-safe.
- **Session-recall hook**: `IAI_MCP_SESSION_RECALL_CLI` environment variable for developer-override of the CLI binary path. CLI lookup now uses a bash array instead of a backslash-continuation for-loop.
- 18 new regression tests across 5 test files covering bank writers, bank-recall CLI, retry policy, FSM reconcile, and backup archiver.

### Changed

- **Deferred-capture retry policy**: failed `.jsonl` files are now retried up to 3 times with exponential backoff (60 s, 120 s, 240 s). After the third failure the file transitions to `.permanent-failed-<ts>.jsonl` and a `permanent_capture_failure` event is emitted at severity `critical`. Terminal files are never reprocessed. Previously, failed files were renamed once and skipped forever.
- **Session-recall hook**: removed the 24-hour staleness cap on the precache file. The daemon-written cache is now served whenever it exists and reads non-empty, regardless of age. Log marker changed from `cache-hit fresh` to `cache-hit age=`.

## [0.3.2] — 2026-05-13

### Security

- Precache file (`~/.iai-mcp/.session-start-payload.cached.md`) now created with mode 0600 instead of process umask default (was 0644 world-readable).

## [0.3.1] — 2026-05-13

### Added

- **Session-start precache**: the daemon writes the recall payload to a cache file (`~/.iai-mcp/.session-start-payload.cached.md`) once per REM-loop completion. The SessionStart hook reads this file when fresh (mtime < 24 h), avoiding a JSON-RPC call into core that would block on the exclusive store lock during DREAMING.
- 4 new regression tests covering the precache writer, cache-hit, cache-miss-absent, and cache-miss-stale paths.

### Changed

- `assemble_session_start` refactored into an emit-free `_compose_session_start_payload` helper plus a thin wrapper that adds the `session_started` event. Public API and return type unchanged.

## [0.3.0] — 2026-05-12

### Added

- **Per-turn ambient capture** via a new `UserPromptSubmit` hook (`iai_mcp/_deploy/hooks/iai-mcp-turn-capture.sh`). Each prompt and the preceding assistant turn(s) are appended to a per-session `.live.jsonl` buffer as pure file IO (~5 ms, no daemon RPC, no embedder). The Stop hook atomically renames the buffer at session end; the daemon drains it through the full pipeline on the next idle edge.
- **Session-start recall injection** via a new `SessionStart` hook (`iai_mcp/_deploy/hooks/iai-mcp-session-recall.sh`). On session open the hook calls `iai-mcp session-start` and pipes the assembled memory prefix (L0 identity, L1 critical facts, L2 communities, global rich-club) to stdout, capped at 10 000 chars. Claude Code injects it as `additionalContext`. Fail-safe: empty store or unreachable daemon exits 0 with empty stdout.
- **New CLI command `iai-mcp session-start`** exposes the payload formatter for manual or debug use. Connects to the daemon socket with a 5 s connect / 30 s read timeout.
- **New CLI command `iai-mcp capture-turn-deferred`** exposes the per-turn writer for manual or debug use.
- **3-hook installer**: `iai-mcp capture-hooks install` now wires `UserPromptSubmit`, `Stop`, and `SessionStart` hooks into `~/.claude/settings.json`. Uninstall and status report all three.
- **Daemon DROWSY drain**: the daemon now drains the deferred-captures buffer on the `WAKE → DROWSY` lifecycle edge (5-min idle) in addition to the existing post-REM drain. Buffers no longer sit indefinitely when a quiet window doesn't fire.
- **Auto-provision `.crypto.key`**: `iai-mcp daemon install` and `scripts/install.sh` auto-generate `~/.iai-mcp/.crypto.key` on fresh installs. Idempotent; the `IAI_MCP_CRYPTO_PASSPHRASE` fallback is preserved.
- **Drain cap**: each drain pass is capped at 5 000 events. Remainder is written to `*.partial.jsonl` for the next pass.
- README: headless/VPS deployment section, AVX2 requirement, troubleshooting table.

### Changed

- **Capture hooks section** in README rewritten for the 3-hook model.

## [0.2.0] — 2026-05-12

### Added

- **Opt-in int8 embedding quantization** via the `IAI_MCP_EMBED_QUANTIZE=int8` environment variable. The default `fp32` path is unchanged. Round-trip cosine similarity ≥ 0.99 on `bge-small-en-v1.5` in tests. New `Embedder.embed_quantized()` surface returns a `QuantizedVector` with per-vector `scale` and `zero_point` calibration.
- **Derived temporal validity**: `memory_recall` hits and anti-hits now carry `valid_from` and `valid_to` fields derived at recall time from the contradiction-edge graph. `valid_from` defaults to the record's `created_at`; `valid_to` is set only when a newer record contradicts it. Both default to `None` on paths that don't enrich (back-compat preserved).
- **MCP tool annotations and outputSchema** on every tool. Each tool now declares `readOnlyHint`, `destructiveHint`, `idempotentHint`, `openWorldHint`, and `title` annotations plus a structured `outputSchema`. Lifts Glama TDQS from C to B.
- **`BENCHMARKS.md`** — public methodology document covering the eight project benchmarks (M-01 token budget, M-02 latency, M-03 RSS, M-04 verbatim, M-05 trajectory, M-06 multilingual, M-07 session cost, M-08 LongMemEval-S).
- **Bench harness reliability**: `bench/longmemeval_blind.py` now supports `--resume` and `--fresh` flags, auto-cleans errored checkpoints by default, requires an explicit `IAI_MCP_STORE_PASSPHRASE` for the encrypted store, and classifies errored rows separately from genuine misses in the summary.
- **Codex CLI** as an optional `capture-hooks` target for ambient Stop-hook capture. New: `iai-mcp capture-hooks install --target codex|claude|all` and `iai-mcp capture-hooks status --target all`.
- README documents Claude Code and Codex setup paths for capture hooks and MCP wiring.

### Changed

- **Behavior — stale downweight on recall.** Records contradicted by a newer record are now downweighted (not hidden) in both `hits` and `anti_hits`. Score is multiplied by `STALE_DOWNWEIGHT_FACTOR`, and the `reason` field carries a ` · stale` suffix. Top-K ranking may shift compared to v0.1.0 — fresh lower-cosine records can outrank stale higher-cosine ones. Audit trail preserved (records are not removed).
- **API contract — deterministic `overnight_digest`.** The `overnight_digest` block in `memory_recall` responses is now deterministic: same inputs produce the same shape and field set. When no REM cycle has run, the digest is a zeroed default instead of a partial dict. Same top-level keys returned over both stdio and socket transports.
- **API contract — `camouflaging_status` outputSchema fields renamed** to match the actual Python response. `formality_trend` → `trajectory_slope`, `anomaly_score` → `current_mean`, plus new `sample_count: integer`. Permissive JSON Schema consumers were already tolerant; strict-validation consumers must update.

### Known fragile surfaces

- `IAI_MCP_EMBED_QUANTIZE` accepts only `int8` (lowercase) or unset. Any other value — including `INT8`, `int4`, or typos — causes the daemon to fail loud at startup with a `ValueError`. This is intentional; no silent fallback to `fp32`.
- New `valid_from` and `valid_to` keys in `hits[]` and `anti_hits[]` are additive (default `None`). Strict JSON Schema consumers that validate with `additionalProperties: false` will reject the response shape until they widen their schema.
- The `_knobs_applied` field is present in the `memory_recall` response but is not yet declared in the tool's `outputSchema`. Known debt; will be addressed in a follow-up release.

### Acknowledgements

- Reddit user [u/BeginningReflection4](https://www.reddit.com/user/BeginningReflection4) — feedback and testing that shaped this release.

## [0.1.0] — 2026-05-11

Initial public release. Local memory daemon for MCP-over-stdio hosts. Verbatim recall, ambient capture, sleep-cycle consolidation, encrypted-at-rest LanceDB store, configurable operating profile.

[1.0.0]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v1.0.0
[0.4.2]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.4.2
[0.4.1]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.4.1
[0.4.0]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.4.0
[0.3.2]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.3.2
[0.3.1]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.3.1
[0.3.0]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.3.0
[0.2.0]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.2.0
[0.1.0]: https://github.com/CodeAbra/iai-personal-memory-engine/releases/tag/v0.1.0
