# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.3] - 2026-08-18

### Fixed
- A stored embedding or a query cue carrying a non-finite value (`NaN` or
  `±inf`) no longer poisons exact-cosine recall. The value is coerced to zero
  at the single normalization point, so a corrupt row degrades to cosine 0
  instead of flattening the whole ranking; the coercion is counted and
  surfaced by a new `doctor` check rather than failing silently, and the read
  path never raises. Thanks [@Acroaticum](https://github.com/Acroaticum).
- `iai-mcp capture-transcript` reported records it never persisted — it built
  a store, wrote through the buffer, and exited without closing it, so records
  below the autoflush threshold were dropped while the command still printed
  them as inserted. The command now closes the store on every path, so the
  reported count matches what reaches disk. Thanks
  [@chrismartinsoares](https://github.com/chrismartinsoares).
- Running the test suite on Windows could read and write the operator's real
  store. The hermetic fixture patched only `HOME`, which Windows ignores in
  favour of `USERPROFILE`/`HOMEDRIVE`+`HOMEPATH`, so the sandbox was a no-op
  there. The fixture now redirects all of them, keeping the suite off the real
  store on every platform. Thanks
  [@chrismartinsoares](https://github.com/chrismartinsoares).
- `scripts/bootstrap.sh` probed for a supported interpreter and reported it,
  then handed off to `scripts/install.sh`, which built the venv with a bare
  `python3`. On a machine where `python3` resolves to 3.13 or newer the
  editable install then failed with `requires a different Python`. The
  installer now selects the interpreter itself (honouring a new `IAI_PYTHON`
  override), bootstrap passes the one it already validated down, and an
  existing `.venv` on an unsupported Python is rejected with a message naming
  the fix instead of failing later in the build. Thanks
  [@anubissbe](https://github.com/anubissbe).

## [3.0.2] - 2026-08-15

### Added
- A hand-set behavioural-profile knob now persists and
  is pinned the instant it is set, rather than waiting for a later write to
  flush the whole profile. A pinned knob is recorded in the durable blob at
  set time and survives a process restart even if no other write happens
  first.
- The task-support knob now auto-tunes from genuine
  same-session follow-through: if adjacent suggestions go consistently
  unused it switches to blank recall, and if that later stops fitting, a
  bounded nightly probe (at most once every two weeks) briefly re-shows
  suggestions and switches back only when the user actually uses them
  again. An unused probe changes nothing.
- The monotropism-depth knob now auto-tunes per
  topic, keyed by the topic's own human-readable name, from how
  concentrated recall cues are on one topic versus a spread of topics
  night over night; a genuinely dominant topic's rank gain grows
  (bounded, capped well below the level a manual override would need to
  reach), and a topic that stops recurring fades back out. The knob's
  proactive same-topic duplicate check now resolves the same way, so a
  hand-set depth on a named topic reaches both the rank gain and the
  duplicate check consistently.
- A hand-set monotropism-depth above the manual
  threshold now also reorders which hits are served first, promoting
  memories from that named topic (including exact-match hits) to the
  front of the response. This reorder is manual-only; ordinary
  night-over-night auto-tuning never reaches it.

### Removed
- Dropped the masking-detection tool and its behavioural
  knob. The `camouflaging_status` MCP tool is gone, leaving 14 tools on the MCP
  surface, and the `camouflaging_relaxation` knob is gone from the sealed
  behavioural profile registry, which now holds 10 entries. The recall path no
  longer applies a formality-relaxation adjustment based on that knob. Any host
  still calling `camouflaging_status` or reading/writing the removed knob must
  update to stop referencing them; historical audit-log rows recorded while the
  feature was active remain queryable. Thanks
  [@tom-shields-stitch](https://github.com/tom-shields-stitch), whose analysis
  showed the knob sitting at seed defaults on a real 33k-record store and whose
  dead-path reports drove the surrounding cleanup.

### Fixed
- A reader could briefly see fewer stored memories than were actually
  committed while writes were in flight — in the worst case a batch of recent
  memories vanished from one read and reappeared on the next. A read-only
  snapshot captured the write-ahead log in a single unguarded pass; racing a
  live writer could tear that pass mid-file, silently truncating the view to an
  older prefix that was then served as current. Snapshot capture now verifies it
  consumed the log to its true committed tail, a refresh can never regress a
  reader onto an older view, and a capture that cannot be verified fails loud and
  falls back to the strongly-consistent path instead of serving a phantom
  snapshot.
- Defense in depth for the same class: every cached row-count formation point now
  re-checks its freshness fence across the count read and recounts instead of
  caching when the fence moved, so a stale count can never be paired with a
  current generation.
- A stored memory whose internal vector-index label was
  ever left empty could not be recovered on the storage engine: the boot-time
  repair meant to backfill it silently did nothing (a column-to-column update
  wrote empty instead of copying the row key), so the store refused to reopen
  instead of self-healing. The update now copies the value correctly, so such a
  store recovers on the next start.
- A cached filtered row count could report more rows than
  actually match after a soft-delete, because the cache was not refreshed when a
  non-indexed column changed. The boot decision about whether to rebuild the
  vector index, the wake sequencing, and the corpus count all read that count, so
  a stale over-count could trigger needless work or mask a real mismatch; the
  count is now refreshed on every write that can change it.
- Setting a behavioural-profile knob to the value it
  already held recorded no pin, leaving it open to being moved by the
  nightly self-tuner even though the owner had explicitly locked it in.
  Any explicit set now pins the knob regardless of whether the value
  changed. Separately, a failed durable write during a knob set (disk full,
  encryption error) was silently swallowed and reported back as full
  success; the response now distinguishes a persisted set from one that
  only updated the live session, and a failed write raises a warning event.
- The per-turn memory refresh re-injected the entire
  session-start brief (identity, critical facts, topic communities, key
  memories) on every store advance instead of just what changed, so a busy
  session could re-pay thousands of tokens per turn for content the model
  already had. It now renders a bounded delta of only the records newer than
  the caller's own watermark, cross-session records carry an origin label, a
  burst larger than the render window still advances the watermark and marks
  the truncation, and a server-side debounce caps how often a session can
  fire the refresh without ever dropping a new record — it simply
  accumulates until the next allowed render.
- A retrieved memory was never marked as reviewed, so
  the nightly forgetting sweep and cluster-replay step both treated it as
  never used. Recall now stamps `last_reviewed` on every returned record as
  part of the write it already issues, independent of the reconsolidation
  dry-run flag and of that config loading successfully. Using a memory now
  protects it from the forgetting sweep's rolling window and lets nightly
  replay reach it.
- The monotropism-depth topic names did not survive a
  daemon restart: the map is persisted, but the in-process cache that the
  rank gain, the response reorder, and the duplicate check all read from
  stayed empty until a nightly cycle happened to run in that same process,
  so a restart silently disabled the whole feature until the next sleep.
  Boot hydration now loads the persisted topic names the same way it loads
  the rest of the behavioural profile, so the feature keeps working
  immediately after a restart.
- A topic's monotropism-depth learning could reset for
  no reason: when a topic's display name needed a disambiguating second
  word (two topics sharing the same top word), that second word was
  recomputed fresh every night, so it could rotate even though the topic
  itself never changed, starting that topic's accumulated depth over from
  zero under the new name. The disambiguating word is now sticky the same
  way the topic's main name already was, so a stable topic keeps its
  learned depth.
- Removed records could still be returned by recall. The two maintenance
  operations `iai-mcp blob-quarantine --apply` and `iai-mcp idem-dedup --apply`
  soft-deleted a record but did not mark it not-live in the same write, so recall
  (which serves only live rows) kept returning it, and the row also stayed
  resident in the warm caches. Both operations now clear the live flag together
  with the deletion timestamp and evict the removed ids from the exact-match and
  recency caches. A one-shot boot-time self-heal repairs a store already drifted
  by the old behavior, and a source-scan check keeps any future deletion path
  from reintroducing the split.
- Encryption key recovery resurrected deleted records. `iai-mcp crypto
  recover-prior-key` rebuilt every row through the fresh-insert serializer, which
  dropped the deletion timestamp, the live flag, and the pending-embedding flag —
  so recovery brought soft-deleted records back as findable and silently marked
  pending rows ready. Recovery now stages each row byte-for-byte from storage
  (preserving deletion and pending state) and re-keys only the encrypted columns
  under the new key. The recovery staging table is built from the canonical
  schema so the vector-index key and the record-id uniqueness constraint survive
  the swap. Stores already recovered under the old behavior cannot be repaired
  (the deletion timestamp was erased); this fix prevents it going forward.
- A duplicate identifier written straight to the memory
  table (bypassing the normal upsert path) silently appended a second row
  instead of being rejected, on the primary storage engine only — the
  fallback engine already refused it. Both engines now refuse a duplicate
  identifier the same way, and a routine batch write that hits a genuine
  duplicate no longer keeps retrying the same batch forever; the offending
  entry is dropped once its content is confirmed already stored, and
  anything else is surfaced loudly rather than assumed safe. Encryption key
  recovery also now refuses outright, rather than silently duplicating
  content, if it ever finds a source that already holds two entries under
  the same identifier.

## [3.0.1] — 2026-08-12

Six defects found by running the engine at production scale on a live box rather
than by testing it. Four of them ended with a dead daemon, and every one of them
reported green on every health surface we had. Upgrade with
`pip install -U iai-pme` and restart the daemon; nothing to migrate.

### Fixed

- **The MCP wrapper was terminating a healthy daemon.** `ensureDaemonAlive()`
  runs on every wrapper start — every session, every sub-agent — and probed the
  socket once with a one-second timeout. A daemon grinding a consolidation step
  at full CPU misses that without being dead, and the miss ran
  `launchctl kickstart -k`, which kills the running instance first. So opening a
  session during consolidation killed the daemon and lost the step. The kill flag
  is gone (a plain kickstart leaves a live service alone), one miss is no longer
  evidence — three probes, three seconds each — and before any restart the
  wrapper verifies the recorded pid actually identifies as this store's daemon.
- **Two more ways to kill the wrong daemon, both in the CLI.** `stop` signalled
  whatever pid the lifecycle lock named, so a stale record could fell a recycled
  pid or another store's daemon. `start` ran `launchctl bootout`
  unconditionally, which SIGTERMs a live instance — and `start` is what the
  unattended doctor repair invokes. Both now require the pid to identify as this
  store's daemon first.
- **The daemon killed itself on large stores.** Every node embedding in the
  memory graph was held as a list of boxed floats: about 15.6 KB per
  384-dimension vector against 1.5 KB as a contiguous float32 buffer, paid once
  per record and again in the community-detection child, the record cache and
  the candidate pool. At production shape (31k records, 114k edges) that was
  754 MB, now 144 MB, and the memory watchdog stopped firing. Stored vectors are
  unchanged on disk, so no ranking, centroid or community result moves.
- **Nightly consolidation could never start on an agent workload.** Sleep entry
  required 30 minutes of session silence, which an always-open assistant session
  never provides, and a live session scanner outranked the starvation backstop —
  which itself could not arm, because it measured from a timestamp only a
  completed cycle writes. On one live store nothing had consolidated in ten days
  while every check reported healthy. Entry now keys on operating-system input
  idle inside the window instead of session silence, with symmetric exit when
  you come back, and the entry reason is recorded so a quiet night can be told
  apart from a gated one.
- **The nightly insight never fired on macOS Keychain logins**, which is the
  default for anyone who signed in through the desktop app. Three stacked
  failures: the CLI's bare mode cannot see Keychain credentials, the service
  environment carries no user identity for the OAuth refresh to find the
  Keychain item, and the CLI's own error text sits behind a large usage block
  where a truncated read never reached it. A zero exit code carrying an error
  flag also used to pass as success, which could have stored the error text as
  the night's insight.
- **The crisis reclustering step could not finish.** Its updates fell back to a
  whole-table scan per statement, so a fragmented partition meant thousands of
  scans and hours of work that any foreground recall reset to zero. Now one
  bulk transaction per chunk: 31k rows in about two seconds.

### Added

- **A nightly self-heal for degraded embeddings.** Two bounded, resumable scans
  during sleep: one finds records whose stored vector is all zeros, the other
  re-embeds a rotating sample and flags rows that have drifted from what their
  own text embeds to. Flagged rows are re-embedded from their intact text, and
  the encryption and verbatim boundaries are never touched. It refuses to act on
  a systemic signal: if a large fraction of the sample looks wrong, that reads as
  "the embedder changed", so it reports and waits for you rather than rewriting
  good vectors. Runs only while asleep, never on the read path.
- **The doctor alarms when the nightly insight stops being produced**, including
  the case where one mint succeeded earlier the same day and a later attempt
  failed. It reads history through the daemon socket while the daemon holds the
  store, and falls back to a read-only open when the daemon is down. The check
  list is now 30 rows.

### Changed

- The storage engine serves an `UPDATE` through a secondary index for
  `WHERE id IN (...)`, `WHERE col IN (...)` and `WHERE col = <literal>`, not only
  a single-id match. Those shapes previously scanned the whole table per
  statement. Results are identical, only faster.
- Several doctor checks now read through the daemon socket instead of blindly
  waiting on a lock the daemon already holds, with a direct read-only fallback
  when the daemon is down.

## [3.0.0] — 2026-08-08

**Upgrading takes three steps.** The capture hooks and the engine must agree on
the new spool and pack formats, so reinstall them in lockstep:

```bash
pip install -U iai-pme
iai-mcp capture-hooks install
iai-mcp daemon restart
```

**If the daemon then refuses to start** with `refusing to mix vector
generations`, the store is telling the truth: its vectors were produced by a
different embedder than the one your environment now configures. Clear any
`IAI_MCP_EMBED_MODEL_ID` / `IAI_MCP_EMBED_TEXT_PREFIX` override from the service
environment (`iai-mcp daemon install --yes` re-renders the unit from a clean
template), or run `iai-mcp migrate --reembed-from-text` to move the store to the
new model. Working around the guard is the one wrong answer: the state it blocks
is a semantic lane that returns nothing while keyword matching hides the loss.

### Added

- **Memory can speak thirteen more languages, when you ask it to.** `iai lang
  add|remove|status` opts a store into `cs de es fr hi id it ja pt ru th vi zh`.
  English stays the default and nothing changes until you run the verb. The
  selection lives in the store's own config, never in the service environment,
  so a shell export cannot silently repoint an existing store at a foreign
  vector space. The embedder is `multilingual-e5-small` at a pinned revision,
  used raw: on a pre-registered 318-probe battery the raw model beat both a
  distilled student and the MiniLM control, so the pack ships zero training.
  Non-Latin scripts (ja, th, hi, zh) currently get no lexical contribution —
  Unicode-aware keyword matching is queued.
- **Reflection can ride a second subscription.** `iai reflect provider …`
  routes the nightly synthesis through the Codex or Gemini CLI instead of
  Claude. Child environments are allow-listed, the argv is sandboxed and
  non-interactive, and the prompt goes in on stdin.
- **Named things are recallable by name.** Capture extracts entity anchors from
  every stored turn — backticked identifiers, at-prefixed handles, dotted and
  dashed names, CamelCase, mid-sentence proper nouns including Cyrillic — and
  writes them as `entity:` tags. `iai-mcp entity-backfill` anchors a corpus
  captured before this shipped; `--refresh` recomputes the whole corpus.
- **`iai-mcp blob-quarantine`** tombstones machine-notification blobs captured
  before the noise filter existed. They drown real records on any cue that
  shares their vocabulary. Dry-run by default; applying snapshots the store
  first, journals every id, and spares pinned records.
- **Captured turns record their mechanics, not only their words.** Assistant
  turns carry a `[tools: …]` trailer, so memory remembers what was used and not
  just what was said.
- **The upload surface accepts source files.** Alongside prose and Office
  containers, `iai upload` and the dashboard now take 30-odd code and config
  suffixes, so a repository or a config tree ingests as readily as a document.

### Changed

- **The deferred capture spool is encrypted.** Every line is sealed with
  AES-256-GCM under the store's existing key file and a constant AAD, so the
  in-place renames the drain relies on cannot orphan a line. The inline hooks
  stay dependency-free and stage at `0600`; the daemon re-encrypts those lines
  in place on its first pass, line for line, fenced against concurrent appends.
  Readers accept both formats, a tampered line is skipped rather than fatal, and
  a missing key defers the file to a later pass instead of walking it toward
  permanent failure. Nothing is ever evicted — the verbatim guarantee holds —
  but a soft size cap (`IAI_MCP_SPOOL_SOFT_CAP_MB`, default 100) now warns
  loudly through `doctor` and `status`, because a growing spool means a daemon
  that has stopped draining.
- **Key rotation is recoverable from every partial state.** Retained
  generations are tracked in a sidecar, prior-key recovery spans multiple
  generations, and redaction re-keys before it redacts, so an interrupted
  rotation no longer leaves a store that opens under no key at all.
- **The socket binds before the embedder builds.** A cold native model build
  used to run to completion first, leaving a live process that no client could
  reach and every surface reporting the daemon as down. Liveness no longer
  waits on a model: `status` answers immediately and reports a warming
  identity, while `recall` waits on a single-flight build.
- **An embedder refusal says so.** A refused embedder selection — foreign
  dimension, misconfigured model, identity mismatch — now travels as a typed
  code across every serving surface instead of degrading into a zero cue vector
  or surfacing as "daemon unreachable".
- **The nightly cycle learns quiet from the machine, not from session counts.**
  The quiet window is derived from OS input idle, with a persisted 48-hour
  starvation backstop and a boot-time reset of a poisoned window. Parallel
  agent activity can no longer invert the window onto the working day and
  starve consolidation.
- **Ambient injection is per session.** The next-turn anticipation pack is
  published per session with a strict ownership model, so one conversation's
  refresh can no longer overwrite another's, and the session-start recent-work
  feed labels every line with its origin.
- **Text files declare their encoding.** Every state and spool write now names
  UTF-8 explicitly, across 161 call sites, so a Windows default codec cannot
  mangle stored content.

### Fixed

- The store records which embedder produced its vectors and refuses to open
  under a different one. Two 384-dimension models pass a dimension check and
  still write vectors that read as noise against each other.
- `memory_recall` honors a caller-supplied `cue_embedding` on the warm path, as
  the tool schema always promised. One validated vector drives candidate
  selection, the exact-cosine authority and the final rank.
- The re-embed migration covers every tier rather than episodic alone, rebuilds
  the recall index, and clears its checkpoint even on a zero-write resume.
- The Antigravity hook installer picks its script set per platform. As
  published in 2.9.0 it demanded the POSIX `.sh` set on Windows and could never
  succeed there.
- `detect_communities` reaches its own fallback. The numba-tainted import sat
  above the guard that was written to catch it, so an unusable numba escaped
  the function and surfaced on every tool call. Thanks
  [@ZhdanDesign](https://github.com/ZhdanDesign).
- The session-recall hook resolves the CLI the way the capture hook already
  did. A standard `pip install` puts the binary where the recall hook never
  looked, so capture worked and recall silently did not. Thanks
  [@ZhdanDesign](https://github.com/ZhdanDesign).
- Windows portability across the daemon, the locks, the state files and the
  migrations: `fcntl` degrades instead of crashing, `os.fchmod` is guarded, and
  PowerShell capture and recall hooks ship for Antigravity along with a Task
  Scheduler scanner for its GUI, which fires no hooks of its own. Thanks
  [@owlze](https://github.com/owlze).
- Documentation accuracy, found by reading the code rather than the prose: the
  encryption key is `~/.iai-mcp/.crypto.key` and two of three mentions named a
  file that does not exist; the doctor table promised 27 checks and listed 26;
  the upload list named 12 of 45 accepted suffixes; and one sentence claimed
  plaintext never reaches disk while the deferred spool wrote exactly that. The
  claim returns in this release as truth, because the spool is now encrypted.

### Security

- Deferred captures are encrypted at rest. Before this release they were
  written as owner-only plaintext and, on a machine where the daemon had
  stopped, accumulated without a ceiling.
- A guard test scans every tracked text file for cp1251 damage, for byte-order
  marks outside PowerShell scripts, and for debug prints in shipped code.

## [2.8.1] — 2026-08-02

### Fixed

- Recall no longer serves confident noise on topics the store barely
  covers, and its `reason` string cannot lie. When the candidate head's
  cosine spread collapses (nothing to distinguish), the degree term is
  dampened proportionally instead of silently deciding the ranking, and
  the response carries a `flat_cosine` hint — hints are now actually
  serialized over the recall RPC (they were declared in the tool schema
  but never sent). The aaak overlap matches cues against content fields
  only — entity anchors and doc names — never machine tokens or
  bookkeeping tag values, so a cue word like "user" cannot bias every
  record of one role; entity anchors stay dormant until the capture path
  writes them. `reason` is assembled during scoring with weighted terms
  and the full multiplier trail (trigram, fts, lex fusion, tier,
  temporal, profile gain, stability, valence, corrector anchoring), so
  the printed arithmetic reconciles with the served score.
- Served recall ordering is now deterministic. Equal-scored hits used to
  inherit candidate scan order (Python's sort is stable), so the same cue
  could return a different ordering across calls; the supersede-cap window
  had the same tie sensitivity, which could shift served scores. Every
  serve path — pipeline, bank fallback, and the daemon's authority merge —
  now sorts through one shared helper that breaks ties on `record_id`, and
  a guard test keeps score-only sorts out of those files. Reported by
  [@danielhertz1999-bit](https://github.com/danielhertz1999-bit).
- FHRR bundle golden parity now states its platform tolerance instead of
  encoding one machine's libm rounding: components whose mean angle sits
  on a quantisation boundary (byte inputs make that exact) may differ by
  exactly one step, components with a near-zero resultant are skipped
  (atan2 is undefined there), everything else stays byte-exact — so
  formula and rounding-mode regressions still fail. Goldens unchanged.
  Measured and reported by
  [@danielhertz1999-bit](https://github.com/danielhertz1999-bit).
- A read-only reader holding a stale snapshot now raises the typed fence
  error the caller expects, rather than the older integrity shape. The
  crash-recovery test covering it had been red since the fence was typed.

### Changed

- The Rust workspace is `rustfmt` clean, so formatting can be enforced in
  CI from here on.
- Two test-suite races removed: the multiprocess WAL test no longer
  depends on process interleaving, and detection-arena isolation is
  measured with a fresh process per arm.

## [2.8.0] — 2026-07-31

**Upgrading to this release is manual, once.** `iai-mcp self-update` ships *in*
2.8.0, so it is not yet on the machine of anyone running an earlier version.
Existing users take this one step by hand:

```bash
pip install -U iai-pme
iai-mcp daemon restart
```

From 2.8.0 onward `iai-mcp self-update` does both in one command and verifies the
restart by asking the running engine its version. Nothing ever installs itself:
the engine checks for a release once a day and tells you, in a doctor row and one
line at session start. `IAI_MCP_VERSION_CHECK=0` turns the check off.

If you run iai-mcp as a Claude Code plugin, note that the plugin files refresh on
their own while the Python package does not — upgrade the package too, or the
hooks and the engine drift apart.

### Added

- The MCP surface now tells the model when to reach for memory. Server
  `instructions` leads with routing prose — call `memory_recall` before a
  repository search when the question is about a decision, a preference,
  a past discussion, or rationale; keep file search for the current state
  of the code (ordering, never substitution) — with the machine config
  appended after a `config:` marker (the field is no longer a bare JSON
  string). `memory_recall` and `memory_search` descriptions carry the
  same routing standalone, so a host that ignores `instructions` still
  sees it; recalled claims about code are framed as historical until
  confirmed against the tree.
- Ambient memory reaches four more hosts. `iai-mcp capture-hooks install
  --target cursor|antigravity|hermes|openclaw` (and `all`) wires each
  host the way it natively allows, while every host shares the same four
  core hook scripts — per-host differences live in thin wrapper scripts
  that translate the payload in and the envelope out, and degrade to
  empty output on any failure like every other hook. Cursor gets session-start
  recall plus full ambient capture (its `beforeSubmitPrompt` cannot
  inject, so there is no per-turn slice there). Antigravity (the CLI)
  gets per-invocation recall — durable on the first call, ephemeral
  after — and Stop-time capture that reads the lossless
  `transcript_full.jsonl`, never the truncated short transcript. Hermes
  (>= 0.5.0) gets first-turn/per-turn recall through its `context`
  envelope and end-of-session capture read from its message store — it
  keeps no transcript file. OpenClaw has no shell hooks at all, so its
  target registers the bundled MCP wrapper: memory tools on request,
  ambient honestly marked as unavailable. The transcript parser now
  understands Cursor and Antigravity line formats alongside Claude Code
  and Codex. Installers refuse to rewrite configs they cannot merge
  safely — an unmergeable Hermes `hooks:` block is printed for manual
  merge, never overwritten.
- Wheel installs learn about new releases and can upgrade in one move.
  A notify-only version check (daily TTL, 3s timeout, silent offline,
  `IAI_MCP_VERSION_CHECK=0` disables it entirely) is refreshed by the
  daemon's tick and surfaced in two places: a new `(+) update available`
  doctor row and a one-line notice in the session-start payload — the
  session-start path reads only the cache, never the network. New
  `iai-mcp self-update` closes the gap `pip install -U` leaves: it
  upgrades the wheel AND restarts the daemon, so recall is never served
  by an old engine under a new version number. It refuses editable
  (source) checkouts, confirms before changing anything (`--yes` skips,
  `--check` only reports), and on a pip failure leaves the running
  daemon untouched. Nothing updates unattended — no timer ships.
- `iai-mcp doctor` heals what it diagnoses. The repair planner grew from 4
  to 11 actions: a parked engine (persisted HIBERNATION/SLEEP with no live
  daemon — check (j) now calls this a failure instead of printing it as
  information) gets a signal-first wake; a corrupt daemon-state file or
  vector index is quarantine-renamed aside so the daemon regenerates it
  (never deleted); stale wrapper heartbeats are swept; a sleep-cycle
  quarantine stuck past 12h is cleared; permanent-failed captures drain
  back into the store; collapsed timestamps and an oversized store map to
  their existing repair commands behind the usual confirmation prompt.
  New unattended mode `doctor --auto` runs ONLY the safe subset — no
  prompts, no process kills, no store mutations — damped to once per 6
  hours, and the MCP wrapper invokes it automatically when the daemon is
  still unreachable ten seconds after a wake attempt: the memory now heals
  itself at session start instead of waiting for someone to run a command.
- `iai-mcp daemon restart` — stop (waits for the old pid to die) then
  start, matching the restart control the brain view already had.

### Changed

- Curiosity questions come from real disagreements now. The miner
  re-ranks each deferred cue against the current store before deciding
  anything: a topic that gained records after the uncertainty snapshot
  is being actively worked — no question, and any pending question on
  that cue is resolved automatically (the conversation moved on). A
  question mints only when the topic's top candidates are joined by a
  live `contradicts` edge, and its text names both sides verbatim
  ("Two memories disagree — which is current: …"). Dense knowledge —
  many equally-strong memories with no contradiction — earns a silent
  telemetry log, never a question. Pending questions also surface in
  the next-turn pack when the current turn enters their topic (cosine
  against the question's cue), once per repeat window — asked in
  context, never as a cold list.
- The daemon's boot graph preload uses the delta-only rebuild path,
  falling back to the full rebuild internally whenever the cached
  payload cannot support a safe delta.

### Removed

- The per-insert temporal hash and its module: nothing consumed the
  hash since the time-cell readers were removed — temporal recall works
  from timestamps. Insert no longer pays the computation.
- The user model's `recent_projects` field and its aggregation: the
  event kind it counted was never written anywhere, so the list was
  empty for every user since the field shipped.

- Dead machinery superseded by live paths, deleted rather than left
  half-wired: GABA edge-weight annealing (nightly decay + pruning is the
  homeostasis mechanism; the annealer was computed and logged but never
  applied), time-cell neighbor search and sequence reconstruction
  (temporal recall works from timestamps; the per-insert temporal hash
  stays), the temporal-next linker, the degraded semantic recall variant
  (`recall_semantic_warm` is the live path), the guarded-insert wrapper
  (shield checks run inside the s5 identity-write and capture paths),
  the subagent session serializer, the batch-results bank writer, the
  synthetic M-metric placeholders, and the unused surprise/arousal
  probes (`record_surprise`, `basta_check`).

### Fixed

- Brain view chrome no longer collides with itself. The canvas hint is
  pointer-transparent and width-bounded — it used to silently swallow
  clicks on the legend tier rows at common desktop window widths — and it
  dismisses once a file is dropped. The memory panel got its side padding
  back (a duplicate CSS declaration left text flush against the rounded
  edge). Long filenames in the "studying …" pill are ellipsized so the
  drop-zone stops growing into the text above it, the stats row yields to
  the search field on very narrow windows, and the server-down banner no
  longer sits on the menu buttons' hit area.
- HIBERNATION is no longer a one-way door on macOS (issue #90). The state
  machine only left HIBERNATION on a wake signal, and on macOS the MCP
  wrapper returned right after a successful `launchctl kickstart` without
  ever writing `wake.signal` — so a daemon that persisted HIBERNATION
  overnight booted, found no signal, and exited after one tick, silently
  forever (no sleep cycle, `daemon DOWN`, capture still queueing). Three
  layers, each sufficient alone: the wrapper now writes `wake.signal`
  unconditionally and BEFORE the kickstart, so the booting daemon always
  finds it; a boot that restores HIBERNATION with a live wrapper session
  attached (fresh heartbeat from a live pid) wakes immediately instead of
  waiting to die; and the lifecycle tick checks for live demand — a fresh
  external socket request since boot, or a live wrapper heartbeat — before
  shutting down, waking the engine instead (`REQUEST_ARRIVED` is now
  actually dispatched, not just accepted by the transition table). With no
  live session the hibernation exit behaves exactly as before, so the CPU
  economy of hibernation is unchanged.

- The M3 trajectory metric (session-start token budget) reports real
  numbers: it now reads the `session_started` events the serve path has
  always written, instead of an event kind nothing ever emitted — M3 was
  0.0 for every session since the metric shipped.
- Curiosity generates questions again. Since the recall hot path was
  slimmed down (May 16), ambiguous recalls only buffered
  `deferred_curiosity_input` raw material and the promised background
  processor never existed — `curiosity_pending` has answered from an
  empty queue ever since. A new nightly CURIOSITY_MINE sleep step now
  replays that raw material through the entropy tiering and mints
  pending questions (watermarked, so re-runs never double-mint; capped
  per session and per run; already-asked cues are not re-asked). The
  recall path now records hit scores and the turn in the deferred
  payload — entropy is computable at night without reconstructing the
  ranker. Pending questions expire after 7 days instead of accumulating
  forever, and the KNOB_TUNE monotropism nudge no longer fires on a
  zero curiosity signal (a quiet generator is absence of evidence, not
  low curiosity). Opt out with `IAI_MCP_CURIOSITY_MINE_OFF=true`.
  The uncertainty measure is normalized to [0, 1] by log2 of the
  candidate count, so the tier thresholds mean the same thing for a
  two-way and a ten-way recall — raw Shannon entropy grows with
  candidate count and would rate every multi-hit recall as maximally
  uncertain.
- Associative recall no longer starves under recency pressure. The
  pending-recency freshness markers had unconditional right to evict
  ranked hits from the token budget, so on any active day (dozens of
  fresh turns) recall served only exact matches plus markers — the
  graph lane (community gate → spread → rank) was fully crowded out,
  and the curiosity payloads it feeds carried only zero scores. Markers
  now claim at most 25% of the recall budget when ranked hits exist;
  past that share a marker is dropped, never a ranked hit. The deferred
  curiosity payload also excludes markers outright — they are recency
  signal, not ranker output.
- Overnight insights carry provenance. The nightly insight was the one
  path where model-generated text entered the store with no evidence
  trail: it minted into the semantic tier with zero source links, so a
  hallucinated "insight" would have surfaced in recall indistinguishable
  from grounded knowledge. The insight record now gets
  `consolidated_from` edges to the verified records its prompt was built
  from (pattern evidence + the surprise event's sources), and when no
  verifiable sources exist the mint is skipped before the model is even
  called. Every other generated-content path already required evidence
  by construction; this closes the last one.

## [2.7.3] — 2026-07-29

### Added

- **A Claude Code plugin.** Two lines wire the MCP server and ambient capture
  together, with no config file to edit:

  ```
  /plugin marketplace add CodeAbra/iai-personal-memory-engine
  /plugin install iai-memory@iai-pme
  ```

  The plugin carries the wiring; `pip install iai-pme` carries the engine. Its
  launcher asks the installed package where the MCP server lives, so it works
  from a wheel and from a source checkout alike, and says plainly what to
  install if the engine is missing.

### Changed

- **`pip install iai-pme` now delivers the MCP server too.** The wheel ships the
  wrapper as a single self-contained file (its dependencies are bundled in), so a
  wheel install can be pointed at directly — no `npm install` step, no source
  checkout. Editable installs keep using the compiled tree as before.

### Fixed

- Performance tripwires retry once before failing. They assert wall-clock bounds,
  so a busy machine could report a regression that a second attempt disproves; a
  real regression still fails both attempts.

### Removed

- Nineteen duplicate test modules that shipped under two names at once. Each pair
  was the same file — one named after the plan that produced it, one named after
  what it tests — so the suite ran every assertion twice and an update to one copy
  left the other stale. A new guard fails the suite if a tracked source file name
  carries a planning code again.

## [2.7.2] — 2026-07-28

### Added

- **Codex CLI gets ambient memory.** `iai-mcp capture-hooks install --target
  codex` wires the same four hook scripts into `~/.codex/hooks.json` — with
  uninstall and status parity, idempotent, and any hooks you already had are
  preserved — and the transcript reader now understands Codex rollout files. So
  capture, per-turn capture and both recall hooks work on Codex unchanged. A
  missing transcript degrades to a skip; the host is never blocked.
- **A cue that names a date finds that day.** "on July 4", "in March 2026" or an
  ISO date now earns records created on the matching day or month a bounded rank
  boost (`IAI_MCP_TEMPORAL_BOOST`, default 1.15). Cues without a date rank
  exactly as before, and ambiguous English month words ("may I ask", "they
  march") never trigger it. The stored vectors stay pure text: encoding dates
  into embeddings measurably drags unrelated same-day records together, so the
  date lives in the rank term instead.
- **Injected memories now say how much to trust them.** Every hint carries a
  relative age next to its date (`Jul 14 (13d ago)`), a fact that has already
  replaced earlier beliefs is marked `↻N`, and each surface states the contract:
  hints are advisory, and the older or more-revised a memory is, the more it
  deserves re-verification. Session-start lines carry the same markers.
- **`iai teach` reads office and book formats** — `.docx`, `.pptx`, `.xlsx`,
  `.rtf`, `.epub`, plus `.tex` and `.bib` as plain text — with no new
  dependencies. The OOXML containers are parsed with the standard library alone,
  guarded against decompression bombs (per-member size cap read through the
  stream) and against XML entity attacks (any DTD-carrying payload is refused;
  legitimate office XML never declares one).
- **Sleep mines entity links.** The consolidation pass now draws `entity_shared`
  edges between memories that name the same identifiers, using the lexical
  postings it already maintains — so recall spreads along "same thing mentioned"
  paths, not similarity alone.
- New blind retrieval harnesses in `bench/` — LoCoMo, ConvoMem, and a portable
  longitudinal export plus its reference scorer — so the longitudinal claims can
  be reproduced against public datasets.

### Fixed

- **Wheel installs report their real version again.** The two
  `importlib.metadata` lookups still keyed on the old distribution name now
  resolve `iai-pme` first. From the published wheel `iai --version` printed
  `0+unknown` and the Cowork plugin `0.0.0`; source checkouts masked it by
  reading `pyproject.toml` first.
- **A look-alike clique can no longer outrank the real answer.** Cold-path
  ranking degree counts earned edges only — hebbian, contradicts, schema,
  temporal — and excludes the similarity links inferred at insert. Measured as a
  1.2 pp R@5 loss on 500 natural questions, fully recovered (pipeline back to
  parity with raw retrieve, 0.962 / 0.978). Warm daemons were never affected;
  fresh processes and daemon-down CLI recall were.
- **A superseded fact never outranks the fact that corrected it**, and the cap
  respects the serving window.
- The search dropdown in the brain dashboard now has a solid backing, so hits no
  longer blend into the memory graph behind them.

### Changed

- The warm lexical lane fires only for identifier-grade cues (snake_case,
  camelCase, letter+digit names, long ALL-CAPS, dotted or slashed paths, quoted
  phrases) and joins ranking as scored candidates rather than spread seeds.
  Measured on 500 natural-language questions, literal-token evidence is
  anti-correlated with the right answer for paraphrase-style prose at any token
  rarity, while identifier cues are exactly where the index is trustworthy — so
  the trigger now matches the lane's competence and prose recall is untouched by
  warmth. `IAI_MCP_LEX_FUSION_W` tunes the bonus.

## [2.7.1] — 2026-07-28

### Fixed

- The release workflow now names the environment its publisher expects, so a
  published release reaches PyPI. Nothing else changed: 2.7.0 is identical
  code that never left the build.

## [2.7.0] — 2026-07-28

### Added

- **Install in one command.** `scripts/bootstrap.sh` takes a machine from nothing
  to a working install: it checks prerequisites, clones, builds, registers the
  background engine and the capture hooks, adds the MCP server to Claude Code and
  runs `doctor`. Re-run it to update. `--dry-run` prints every step without
  changing anything; `--preflight-only` checks prerequisites and exits.

  ```bash
  curl -fsSL https://raw.githubusercontent.com/CodeAbra/iai-personal-memory-engine/main/scripts/bootstrap.sh | bash
  ```

- **Binary wheels build on every supported platform** — CPython 3.11 and 3.12 for
  Linux (manylinux_2_28), macOS arm64 and Windows x86_64 — and a release now
  publishes them to PyPI through OIDC trusted publishing, with no tokens or
  repository secrets involved.

### Changed

- The distribution package is named `iai-pme` (IAI Personal Memory Engine):
  `pip install iai-pme`. The import name (`iai_mcp`), the operator CLI
  (`iai-mcp`) and the user CLI (`iai`) are unchanged. No release was ever
  published under the old name, so nothing migrates.
- Linux wheels require glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, RHEL 8+).
  Intel macOS has no wheel: NumPy-stack projects stopped publishing x86_64 macOS
  wheels, so that platform installs from source with a Rust toolchain.
- The README leads with the memory in motion, states the token economy a memory
  pack replaces an agent search at roughly 88% less, compares iai-pme against the
  memory layers it is mistaken for, and documents the dashboard, the MCP tools and
  the per-turn recall hook that were already shipping undocumented.

### Fixed

- The Compatibility section no longer claims ambient capture on Codex. The memory
  tools work on any MCP-over-stdio host; the capture and recall hooks are wired
  for Claude Code only.

## [2.6.1] — 2026-07-24

### Fixed

- **Idle detection survives invalid UTF-8 in system command output.** All four
  subprocess readers in the idle detector (`pmset -g log`, `ioreg`, `busctl`,
  `pmset -g`) decoded stdout strictly, so a single non-UTF-8 byte — observed in
  a multi-megabyte power log — raised `UnicodeDecodeError` past the handler
  chain and cost the lifecycle tick its sleep-eligibility signal. They now
  decode leniently (`errors="replace"`), which keeps the check working: the
  sleep-marker scan and timestamp parse already tolerate replacement characters.
  Reported with a full root-cause trace by
  [@res-pstepan](https://github.com/res-pstepan) (#86), whose report also
  surfaced that all four readers shared the pattern, not just the one that
  failed.
- **The LongMemEval harness smoke no longer turns red on a clean checkout.** It
  armed itself whenever the Hugging Face caches and `huggingface_hub` happened to
  be present, then read a bench baseline JSON that is not tracked — so installing
  an unrelated package could flip the suite from green to a failure that looks
  like a bench regression. The module now sits behind an explicit `--bench`
  opt-in (same convention as `--perf` and `--live`), and the baseline-drift check
  additionally skips with a clear reason when the baseline artifact has not been
  generated. Its harness output also moves out of `tests/fixtures/` into a temp
  directory, so a `--bench` run no longer leaves untracked files behind in a
  clean checkout.

## [2.6.0] — 2026-07-23

### Changed

- **Recall ranks curated knowledge above raw chatter.** Knowledge-grade sources
  get a soft multiplier at final rank (default 1.05, `IAI_MCP_TIER_BOOST`, 1.0
  disables), and it stands down for verbatim cues. Teach chunks (`doc:*`) boost
  unconditionally — they are literal curated content — while semantic summaries
  boost only when the `literal_preservation` profile knob is relaxed past its
  strong default, because that knob is precisely the raw-versus-summary
  preference. A default install still keeps raw originals on top.
- **A query worded differently from the stored text can surface its target.**
  The in-tree BM25 lexical index now fuses into recall: when the cue carries a
  genuinely rare token (IDF-gated, `IAI_MCP_LEX_MIN_IDF`), warm BM25 hits join
  the seed set and earn a bounded rank bonus. `IAI_MCP_LEX_FUSION_OFF=true`
  kills the lane. The recall path never pays the index rebuild — the index is
  built by scoped search or the nightly warm-up and kept current by a cheap
  per-insert feed, and a cold index contributes nothing.
- **Exact-similarity authority no longer wipes a hit's graph rank.** When a
  record surfaces through both lanes the richer pipeline score survives the
  merge, so authority keeps guaranteeing inclusion without flattening the final
  order back to pure cosine.

### Fixed

- **A freshly created store no longer inherits a dead store's unflushed rows.**
  Write buffers key on the object id and Python reuses freed addresses, so a new
  store could silently take on another store's pending content. Buffers are now
  purged at store construction.
- **A knowledge summary's consolidation edges survive a crash at mint.** They
  flush to the edges table immediately instead of waiting in the write buffer
  for a threshold flush.

Both ranking levers came out of a recall-quality measurement on a real 18k-record
store by [@Marsu6996](https://github.com/Marsu6996), who also prototyped the
boost factor and the rare-token gate.

## [2.5.1] — 2026-07-22

### Fixed

- **The daemon starts on Windows.** `serve()` inspected `asyncio.start_unix_server`
  before the Windows branch returned, and that attribute does not exist on
  Windows, so the daemon died before reaching its own Windows transport and the
  install looked unreachable. The inspection now happens only on the POSIX path.
- **`doctor` runs on Windows.** The store-lock check imported `fcntl` directly —
  absent on Windows — instead of going through the same portable lock helper its
  sibling checks already use.
- **Deferred captures with non-ASCII text no longer fail to replay on Windows.**
  Capture files are written as UTF-8 but were read back in the system's default
  encoding, which is not UTF-8 on Windows, so a single accented or non-Latin
  character could stall the retry path. They are now read as UTF-8.

All three were found and verified against a live Windows install by
[@LC1207](https://github.com/LC1207).

## [2.5.0] — 2026-07-21

### Changed

- **The storage stack is now ours end to end.** Nearest-neighbour search and the
  record schema moved onto in-tree engines, and the `hnswlib` and `pyarrow`
  dependencies are gone. Recall is exact rather than approximate, the index
  grows on demand instead of hitting a fixed capacity wall, and a stale index is
  rebuilt from the store on first boot. Nothing is required of you — an existing
  store upgrades in place.
- **The runtime no longer needs `sqlite3`.** The daemon runs entirely on the
  in-tree engine; the standard-library driver remains only as an explicit
  fallback. Installs are smaller and the native extension links no system
  libraries.
- **Working memory survives parallel sessions.** Two conversations open at once
  shared one working task, and the per-turn injection could hand one
  conversation the other's context. Attention still holds a single focal task,
  but suspended tasks are now parked one per session: a turn from another
  session parks the current task — writing its finished results through to
  storage first — and restores that session's own. Each session reads only its
  own snapshot, so one conversation's work can no longer surface in another.
  **On upgrade, refresh the hooks together with the daemon** — run
  `iai-mcp capture-hooks install`. A hook left from an older version reads a
  file the new daemon no longer writes, and working memory silently stops being
  injected until it is refreshed.
- **Look-ahead context follows the session it belongs to** rather than whichever
  conversation last held attention.
- **"Turned into knowledge" now measures real coverage** — the share of moments
  actually condensed into a summary, traced through the links between them. It
  used to divide summaries by moments, a ratio pinned at a few percent no matter
  how well consolidation worked. A newer page against an older daemon falls back
  to the old number rather than breaking.
- **The brain view rests wordless.** Captions appear as you zoom in; hovering or
  selecting still names a memory at any zoom. The "center the brain" button
  moved clear of the bottom edge.
- **The brain view's economy card speaks plainly:** memory packs served, free
  tokens injected, fallback searches, tokens saved.

### Added

- **`iai-mcp idem-dedup`** removes exact-duplicate records left by earlier
  builds. It reports by default and only rewrites with `--apply`, which takes a
  store snapshot first.
- **`iai-mcp edge-backfill`** re-links knowledge summaries to the moments they
  came from, on stores where those links were lost. A summary that still holds
  at least one link claims the rest of its group; summaries with none are
  reported and left alone rather than guessed at — they recover on their own as
  their group is summarised again. Reports by default, snapshots the store on
  `--apply`, safe to re-run. Optional maintenance for long-lived stores; a fresh
  install never needs it. Stop the daemon first.
- **`CLAUDE_BIN`** points the daemon at your `claude` executable when the
  service manager's environment cannot find it.
- **Kill-switches for the consolidation work below:**
  `IAI_MCP_WAKE_COACTIVATION=0` stops recall from recording co-activations,
  `IAI_MCP_REM_DISABLED=1` skips the nightly insight, and
  `IAI_MCP_REM_MIN_INTERVAL_SEC` sets how often it may run.

### Fixed

- **Memory learns again.** Consolidation built its clusters from links that were
  never created while awake, so every night found nothing to summarise and no
  knowledge was ever written — silently, since the run reported success.
  Clustering now reads the links the system actually forms, recall records which
  memories come up together, oversized groups are split rather than merged into
  one useless summary, and a group already covered by an existing summary is left
  alone until it grows. On a mature store this is the difference between a
  permanently empty knowledge layer and one that fills every night.
- **A summary is no longer swallowed by the memories it quotes.** Because a
  summary repeats its own sources, the near-duplicate check treated it as a copy
  of them and merged it away. The check now compares only within the same kind of
  memory, and links left pointing at the merged-away copies clean themselves up.
- **The nightly insight runs again.** The pass that writes the overnight digest
  had lost its only caller and had been silent since early June. It is wired back
  in, spaced to once a night, and can be forced or disabled.
- **The daemon can find `claude` under a service manager.** Started by launchd or
  systemd, the daemon inherits a minimal environment and could not locate the
  executable, so every nightly insight failed with a file-not-found error while
  the same command worked from a terminal. It now looks at `CLAUDE_BIN`, then the
  path, then the usual install locations. If your overnight digests were never
  arriving, this is why.
- **Duplicate records stop accumulating.** A record's tag index could be left
  unbuilt, hiding it from the check that prevents re-inserting the same content.
  The index now repairs itself each night, and the new `idem-dedup` command
  clears what earlier builds already stored.
- **The brain view shows connections again.** As a store matured, three separate
  limits combined to leave the graph a field of unconnected dots: self-links
  filled the scan window, the window ended before recent memories, and the sample
  rarely held both ends of any link. Self-links are excluded, the window is far
  wider, and each sampled memory brings its strongest partners along.
- **Direct captures reach the fallback bank.** Only transcript-driven captures
  were mirrored into the recent window, so recall with the daemon stopped saw
  nothing newer than the last batch. Every successful capture is mirrored now.
- **Time-scoped recall returns a valid response.** `memory_temporal_recall`
  emitted an empty scope field on calls without a time bound, which failed
  schema validation. The field is omitted when it does not apply.
- **Pattern detection stops inventing patterns.** Per-record fingerprints, unique
  by construction, were mined as if they were recurring patterns; most stored
  patterns were this noise. They are excluded, and `schema-cleanup` prunes what
  was already stored.
- **A correction can no longer collapse into what it corrects.** If a correction
  was near-identical to the record it contradicts, the duplicate check merged the
  two and wired the record to contradict itself. This is now refused with a
  message asking for a clearer rephrase.
- **`doctor` distinguishes repaired history from damage** and reports honest
  counts for what consolidation created versus merged.
- **A rare storage error now records its own evidence.** Deleting links can
  intermittently fail on one uncommon page layout — loudly, without data loss,
  and the operation retries on the next cycle. When it happens the store image is
  copied to a bounded quarantine folder so the cause can be found. If you see an
  integrity error mentioning interior page overflow, that folder is what to send.
- **`iai status` no longer pins a core.** The status probe was dispatching the
  full topology computation on every call, so a monitoring loop could keep the
  daemon busy indefinitely. Status now answers from a light path, topology is
  cached with a single-flight guard, and the graph pass is sampled above a size
  threshold.
- **Startup no longer replays the whole capture backlog at once.** A large queue
  of deferred captures was drained in one uninterrupted pass at boot, which
  burned CPU for as long as it took. The pass is now bounded (500 records or 10
  seconds by default, tunable via `IAI_MCP_CAPTURE_DRAIN_MAX_RECORDS` and
  `IAI_MCP_CAPTURE_DRAIN_BUDGET_SEC`), reports what remains, and resumes on each
  idle window until the backlog clears.
- **Memory keeps its community structure at scale.** Above a few thousand
  records the clustering pass could collapse into a single flat group, losing the
  structure recall depends on. Partitioning now holds up at full corpus size.
- **Storage corruption is reported as corruption.** A damaged store surfaced as
  an opaque runtime error; disk damage, transient I/O faults, and read-only
  rejections are now distinguishable, so a caller cannot mistake one for another.
- **Deletes over large pages no longer fail.** Interior tree rebalancing measures
  page fullness in bytes, so a delete cascade across byte-full pages completes.
- **Windows can start the daemon again.** `iai-mcp daemon install` registers a
  per-user Task Scheduler task, with install, start, and uninstall parity with
  launchd and systemd.
- **Release wheels build on Linux and macOS.** The native extension no longer
  pulls OpenSSL into the dependency tree, and the macOS deployment target is
  pinned so the built wheel passes its own repair step.

## [2.4.1] — 2026-07-19

### Fixed

- **Re-embedding to a custom provider no longer breaks the store.**
  `iai-mcp migrate --reembed-to-configured-provider` rebuilt the records table
  from a schema that dropped the primary-key constraint, so records captured
  afterward stored a null internal label and the daemon crash-looped on the next
  restart with memory reported unreachable. The migration now preserves the
  table's canonical constraints; a null label left by an affected earlier run is
  repaired automatically on startup; and a genuinely null label now fails with a
  clear, actionable error instead of an opaque one. If you hit this on 2.4.0 or
  2.3.1, upgrading and restarting is enough — no manual repair needed.
- **`doctor` no longer reports a false failure while the daemon is winding down.**
  Check (e) rejected the `TRANSITIONING` state the daemon legitimately writes on
  the way to drowsy, showing a spurious failed check on a healthy install. It now
  accepts that state.

## [2.4.0] — 2026-07-17

### Added

- **Claude Cowork support.** `iai-mcp cowork install` wires the ambient memory
  hooks into Claude Cowork through a local plugin marketplace, so capture and
  recall work there the same way they do in the other supported clients.

### Changed

- **Recall stays fast while the store is being written.** The engine now keeps
  its read indexes and row counts on the writer side and publishes them at
  commit, and reads the corpus count off the write lock, so recall no longer
  slows down under a write load or right after a burst of captures.
- **Consolidation gets out of your way.** Sleep now yields to your reads rather
  than to background socket chatter, so a deferred cycle actually resumes
  instead of starving; it waits out a short cooldown after a clean cycle instead
  of looping while you work; edge decay runs in chunks so it finishes on a live
  daemon; and the orphan-edge sweep is about 54× faster on a real backlog.
- **Lower first-recall latency after a restart.** Boot warm-up publishes the read
  models once up front, so the first queries after a restart no longer each pay
  for a full table scan.

### Fixed

- **The daemon no longer gets throttled by macOS.** It now runs as an
  interactive service instead of a background one, so macOS stops starving it of
  CPU and I/O while other apps are busy — the fix that brought live recall from
  seconds back down to well under one.
- **Big integers compare exactly.** Past 2^53 the engine promoted integers to
  doubles, where adjacent values round together, so an UPDATE or DELETE whose
  WHERE named one giant integer could silently touch a different row.
  Comparisons and sort keys now keep integers exact.
- **Deletes over large pages no longer fail.** Interior B-tree rebalancing now
  measures page fullness in bytes, so a delete cascade across byte-full interior
  pages completes instead of erroring with a page overflow.
- **Recall degrades gracefully on a fragmented index.** A nearest-neighbor query
  now returns fewer results on a fragmented index instead of coming back empty.
- **Concurrent daemon-state writes are safe.** State is written atomically, so
  one writer's keys are no longer erased by another writer's whole-dictionary
  save.
- **Reinforcement no longer disturbs recall.** Writing a reinforcement no longer
  invalidates the recall read path.
- **Async writes shut down cleanly.** Turning async writes off, or tearing the
  store down, no longer leaks the writer thread.

## [2.3.1] — 2026-07-14

### Fixed

- **Backups now include your memories.** The backup command was archiving legacy
  file names and missed the actual database, so a restore could recover an empty
  store. It now captures the whole store directory crash-consistently (with its
  write-ahead log), skips rebuildable indexes, and warns loudly if no database is
  found.
- **Prebuilt wheels build on Linux and macOS.** Release wheels now build the
  bundled MCP wrapper on the host (the Linux build container has no Node) and
  install the x86_64 Rust target, so the macOS and Linux wheels build alongside
  the Windows one.

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
