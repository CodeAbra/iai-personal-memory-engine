# Opinion — Contradiction-longitudinal falsifiability bench (design only)

**Author:** Cursor agent (Composer), 2026-05-02.  
**Scope:** Design opinion only; no code changes, no runs, no MCP/daemon/user data touched.

---

## How Phase 10.x changes the v3 framing (short)

The v3 note in `CONTEXT_PEER_REVIEW_2026-04-30_v3.md` still fits **intent**: separate slices for **historical verbatim** vs **post-flip “current truth”** retrieval, and a gate that **B improves without A collapsing**.

What changed is **where consolidation “lives”** in the product: the five-step `sleep_pipeline` (including **DREAM_DECAY**) is now the **real** consolidation path, CLI-gated (`iai-mcp maintenance sleep-cycle` per `sleep_pipeline.py` docstring), with explicit **C5 / MEM-01** (“never mutate `literal_surface`”) and lifecycle authority (`lifecycle.py` / `.lifecycle.lock`). RSS-driven restart is gone; **HIBERNATION = process exit** — less noise for “how many synthetic nights,” more need to be explicit about **which** consolidation entrypoint the bench calls.

The stub in `bench/contradiction_longitudinal.py` already names “optional sleep/consolidation tick simulation”; the opinion below assumes the **real** harness will call the **same** consolidation surface you want to claim in the README (almost certainly `SleepPipeline.run` / maintenance CLI path), not a reimplemented toy decay.

---

## Open questions — picks + rationale

### 1. Integration depth — (a) daemon / (b) in-process / (c) hybrid

**Pick: (b) in-process for v1**, with a **documented path to (c)**.

**Why:** The falsifiability claim is about **store + graph + retrieval + consolidation math** under a controlled corpus, not about launchd, socket RPC, or `.lifecycle.lock` interop. In-process `MemoryStore` + explicit inserts + `memory_contradict` (or store-level equivalent) + **`sleep_pipeline` (or the single public wrapper the CLI uses)** keeps the bench **reproducible on the Mac Studio** with only `IAI_MCP_STORE` pointed at a throwaway directory.

**(a)** pays **lifecycle lock + daemon singleton + idle/heartbeat** semantics you are not trying to measure for Metric A/B unless you add a whole second project (“bench daemon lifecycle”). Worth it later for **one** “production smoke” row in CI or nightly, not for v1 disclosure.

**(c)** is the right **phase 2**: same corpus driver, swap a `Backend=inproc|daemon` flag once daemon integration is stable.

---

### 2. Sleep cycles between contradict and probe — 0 / 1 / N

**Pick: report **three** slices, not one magic number:** **N=0 (control)**, **N=1 (minimal signal)**, **N>1 (stress, optional)**.

**Why:** **N=0** isolates “contradiction edge + retrieval geometry” without decay. **N=1** is the smallest story that matches the product (“one maintenance sleep-cycle after flip”). **N>1** is where **DREAM_DECAY** can intentionally move scores — either a **bug finder** or a **second chart** (“robustness vs decay”), not the same headline as N=1.

For **README v1**, I would lead with **N=0 and N=1** tables; put **N=3–5** (or adaptive until marginal Δ) in an appendix or “stress mode” so you do not conflate “falsifies cosine” with “survived aggressive pruning.”

---

### 3. Corpus scale for v1 — what ships in the README

**Pick: “minimum-viable” as the headline row**, **smoke (4 rows) as CI default**, **honest-disclosure as an optional flag / separate run**.

**Why:** Four rows **cannot** stabilize rank/MRR noise; they exist to prove the harness wires. README should quote something with **enough probes per condition** (rough order of magnitude: **10² sessions / O(10²) probes** split pre/post flip) so readers do not overfit to a toy. **10³ / 5×10²** is “honest disclosure” when you have time on the Studio. **5×10³+** is stress / regression farm, not what you ask strangers to replicate on first read.

Concretely: **ship numbers from minimum-viable in v1 README**; state smoke as **CI gate only** (“schema + non-zero metrics”), not as the claim.

---

### 4. Scoring formula for Metric B (“beats cosine”)

**Pick: primary = paired comparison on a **predefined candidate pool** or on **full-table rank**, but summarize as **ΔMRR** (or ΔRR@10) with **Wilcoxon signed-rank** (or bootstrap CI) across probes**; secondary = **RR@1** only as a sanity column.**

**Why:** Rank-1 alone is **high variance**; the original v3 intent was “current winning fact **above** flat cosine,” which is naturally **ordinal** across many probes. **Paired** matters: same cue, same store snapshot policy, two scorers (pipeline vs cosine). If you fix a **candidate cap K** (e.g. top-200 by cosine) and only rerank inside it, say so — otherwise you are measuring **different search spaces**.

I did **not** originally mean “win on every single cue”; I meant **systematic upward shift** post-flip for the gold id (or session id) with a statistical gate, plus a **floor** on catastrophic failures (max allowed rank regression).

---

### 5. Threshold for Metric A (“no collapse”)

**Pick: tiered — smoke / CI: **A_after == A_baseline** (100%) on a tiny frozen set**; scaled runs: **A_after ≥ 0.98 × A_baseline** on hit@k for verbatim probes, with **zero tolerance** for “wrong fact replaced superseded wording” on explicitly labeled verbatim probes.**

**Why:** Verbatim memory is the north-star; **0.95** is defensible for **noisy** automatic string match metrics, but for a **synthetic** bench where you control probes and gold strings, **0.98–1.0** is more honest. If the metric is strict **exact record id** hit, use **1.0** for v1 synthetic. If the metric is “surface appears in top-k text,” allow a sliver of slack for tokenization.

State **k** (e.g. 5 or 10) next to the threshold — “no collapse” without k is ambiguous.

---

### 6. Output format

**Pick: **Markdown report** (human-first, README-quote-friendly) **+** one **JSON** blob (machine ingest) **+** optional **CSV** per-probe for spreadsheets.**

**Why:** Public README wants **one paragraph + a small table**; reviewers want **raw rows**. JUnit is great for CI dashboards but awkward to quote in prose — optional later if you wire Jenkins/GitHub Actions.

---

## Subtleties — opinion

### DREAM_DECAY vs contradiction edges

**Not a bug to “expose” in the primary gate** unless the product promise includes “contradiction edge survives N sleep cycles unchanged.” **Design it in explicitly:** either **hold N=1** for the headline metric, or **pre-register** that **N>1** may erode Hebbian weight and report **edge weight + rank trajectories** as diagnostic columns. Otherwise a failure is ambiguous: retrieval regression vs intended decay.

### C5 / MEM-01 vs Metric A

**C5 makes literal corruption an unlikely failure mode for the pipeline under test** — good. **Metric A** then tests **retrieval + indexing + graph visibility** of **superseded** content, not “did the pipeline strip strings.” So: **fused at the system level** (if A collapses, something in recall/index/graph broke), but **I would still keep dedicated unit tests** that grep/assert **no assignment to `literal_surface`** in sleep_pipeline paths — the bench should not be the only guard.

### Lifecycle lock (option a)

**Skip for v1 in-process.** If you later add daemon mode, **either** use the documented socket/wake path **or** a bench-only `shadow_run` / dedicated lifecycle profile so you are not fighting the user’s real daemon. **Not worth** lifecycle.lock complexity for the first falsifiability story.

### Mac Studio + isolated `IAI_MCP_STORE`

**Fine**, with two cautions: (1) **embedder cold load + Lance path** dominate wall time — document **machine + env** next to numbers; (2) ensure the harness **never** defaults `IAI_MCP_STORE` to `~/.iai-mcp` (bench README should show **explicit export**). No concern beyond normal “don’t touch production brain.”

---

## Questions you did not list — worth adding

1. **Candidate pool definition for “cosine”** — full table vs ANN vs same prefilter as pipeline stage-1; mismatch invalidates paired tests.
2. **Embedding cache / warm run** — first probe after insert can pay cold embedder cost; define **warm-up** passes excluded from timing or metrics.
3. **Session_id / provenance** — if recall uses session-scoped boosts, **fixture must pin** session ids the same way production does.
4. **Contradiction representation** — tool path vs raw edge insert: bench should use the **supported public API** you want customers to trust.
5. **Multi-seed** — one JSONL seed vs 3–5 RNG corpus seeds for variance bars on ΔMRR.

---

## Effort guess

| Mode | Effort (one engineer familiar with repo) |
| --- | --- |
| In-process driver + fixtures + MD/JSON report + N∈{0,1} | **~1–1.5 days** |
| + Wilcoxon/bootstrap + N>1 stress slice + honest-disclosure corpus | **+1–2 days** |
| + Daemon-spawn / lifecycle-faithful path (option a/c) | **+2–3 days** (flaky surface, lock/socket) |

---

## Summary line for merge with the other agent’s view

**Ship v1 as in-process, paired ΔMRR (or RR@k) vs cosine on post-flip probes, with N∈{0,1} slices and a strict verbatim slice for A; report Markdown+JSON; treat DREAM_DECAY as a controlled axis, not a surprise failure mode.**
