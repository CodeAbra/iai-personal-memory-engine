# IAI-MCP Benchmarks — v8.0 stack (measured 2026-06-04)

All numbers below are measured on the current **v8.0 stack** (Hippo storage + Lilli/HD substrate + Rust `iai_mcp_native.embed` bge-small-en-v1.5 384d + MOSAIC graph engine), on an Apple M2 Max (12-core, 64 GB). Each row carries a one-line reproduce command. Raw result JSONs live in `bench/results/v8.3/`.

**Honesty rules applied:** no tuning-on-test, no hand-picked seeds, honest-scale (multi-seed) where applicable. The ONLY head-to-head comparison is LongMemEval vs mempalace; every other number is our own metric (no competitor column). Where a number is N/A or regressed, it is stated plainly.

---

## Head-to-head — LongMemEval-S (the only competitive arena)

| Metric | iai-mcp (measured on host) | mempalace (published, config-matched) |
|---|---|---|
| **R@5** (raw semantic) | **0.962** | 0.966 |
| **R@10** | **0.978** | — |
| R@5 + hybrid (tuned) | — (future) | 0.984 |
| R@5 + LLM rerank | — (future) | ≥0.99 |

- **Config (identical both sides):** LongMemEval-S **cleaned, 500 questions**, session granularity, metric = `recall_any@k` (any gold session-id in top-k), full haystack, **raw** (no rerank).
- **Embedder:** iai-mcp = bge-small-en-v1.5 (384d); mempalace = all-MiniLM-L6-v2 (384d). Both 384d, both user-turns-only per session, both pure-dense. The **only** difference is the embedder model — and bge-small-en-v1.5 ranks **higher than all-MiniLM-L6-v2 on MTEB retrieval**.
- **The gap = 0.962 vs 0.966 = 2 questions out of 500** (481 vs 483 hits) — within noise.
- iai-mcp misses concentrate in single-session-preference (13%) + temporal-reasoning (5%); we are strong on the categories usually considered hardest (multi-session 1.5%, knowledge-update 1.3%).
- **Track B note:** mempalace numbers are **published, config-matched** (NOT run on this host, by owner decision). Same 500q cleaned, raw = no-rerank.
- Reproduce: `IAI_MCP_CRYPTO_PASSPHRASE=<x> .venv/bin/python bench/longmemeval_blind.py --split S --dataset cleaned --granularity session --out bench/results/v8.3/longmemeval_cleaned_full.json`

---

## Our own metrics (no competitor column)

| Benchmark | Result | Notes |
|---|---|---|
| **Sleep-ablation** (recall@10 preserved through one consolidation cycle) | recall@10 = **1.000 → 1.000** (Δ=0) · MRR Δ=−0.067 | 3 seeds × 160-record corpus (20 targets + 40 confusors + 100 noise); run_heavy_consolidation tier0 (no LLM); 4 cluster summaries added (schema records go to patterns_observed, not hits[]). Targets remain in top-10; mean rank nudges +0.25 because summaries/confusors occasionally outrank specific targets for broad probes. One-shot consolidation cannot produce a positive delta (cosine-dominated rank, no stability/weight terms, 90-day edge-decay grace, no ERASURE/DREAM steps). |
| **Personal-fact drift** (recall@10) | **0.9933** | retention_loss@10 = 0.0067; honest-scale 3 seeds × (50 facts, 50 sessions, 30 intervening). Gate R@10≥0.80 passed with large margin. |
| **Rescue@10** (post-contradiction current-fact) | **1.000** | Ranks the CURRENT winning fact top-10 after a contradiction. Unchanged v7.0→v8.0. honest-scale 3 seeds × 1000 sessions × 2 slices. |
| **Session-start token cost** | **1629** (minimal) / **2993** (standard) | Both under the hard ≤3000-token session budget. tiktoken-cl100k proxy. |
| Session-start payload (tokens bench) | fresh ~376–388 / warm steady ~1–12 | Under the 8000-fresh / 3000-steady limits. |
| **Recall p95 latency** | 77 ms @N=100 · 105 ms @N=1k · **368 ms @N=10k** | Misses the internal <100 ms@10k target at scale; rank/centrality stage dominates. |
| **Centrality recompute** | 471 ms @N=5k · 1695 ms @N=10k | Betweenness recompute is expensive → centrality cache stays ON by default (cross-validates the recall rank-stage cost). |
| **Memory footprint (RSS)** | **589 MB @N=10,000** | Threshold 2000 MB; passed. |
| **MOSAIC community-detection parity** | 36/36 LFR-gauntlet + 10/10 community pass | NMI vs ground-truth on karate / football / LFR n=1000 & 5000; 5× replay-deterministic; modularity-monotonic. |
| **Rust embedder latency** | p50 70 ms / p95 253 ms (single embed) | bge-small-en-v1.5 384d. |

Reproduce commands (each with `IAI_MCP_STORE=<tmp> IAI_MCP_CRYPTO_PASSPHRASE=<x>` unless noted):
- sleep-ablation: `python bench/sleep_ablation.py --seeds 13 42 137 --output bench/results/v8.3/sleep_ablation.json`
- drift: `python -m bench.personal_fact_drift --scale honest --seeds 13 42 137`
- rescue: `python -m bench.contradiction_longitudinal_claude --scale honest --seeds 13 42 137 --n-slices 0 1` (split CSV by `condition`)
- session cost: `python -m bench.total_session_cost --wake-depth {minimal|standard}`
- tokens: `python -m bench.tokens --wake-depth {minimal|standard}`
- recall latency: `python -m bench.neural_map --n 100 --n 1000 --n 10000 --iterations 20`
- centrality: `python -m bench.mosaicsigma_centrality_perf`
- RSS: `python -m bench.memory_footprint --n 10000`
- MOSAIC parity: `python -m pytest tests/test_mosaic_lfr_gauntlet.py tests/test_community.py -q`
- embed latency (daemon OFF): `python -m bench.embedder_latency --backend rust`

---

## Honest caveats / not-yet-leads

- **Recall p95 at N=10k = 368 ms** — above the internal <100 ms target. The rank/centrality stage dominates; the betweenness recompute is ~1.7 s@10k (cache-mitigated). A latency-optimization candidate.
- **historical-verbatim retrieval regressed 0.900 → 0.713 (v7.0→v8.0)** — the ability to retrieve the *superseded/archived* wording verbatim (MEM-05) dropped. This is SEPARATE from Rescue@10 (which is unchanged at 1.000). Likely cause (per Codex+Gemini code review): the v8.0 networkx→MosaicSigma centrality swap dropped Hebbian edge-weights (unweighted Brandes), shifting the seed-score landscape toward older densely-connected facts. Candidate for a focused v8.4 fix. (Note: the obvious stability-lift tweak would *worsen* this, not fix it.)
- **Track D Rust-vs-PyTorch embed comparison = N/A** — the PyTorch/sentence-transformers embedder path was deliberately removed in v8.0 (Rust is the sole mandatory runtime; no backend knob). The manager-requested before/after cannot be measured. Rust absolute latency is reported instead.
- **Track-A hybrid (dense+lexical) = future** — a BM25-over-user-turns hybrid would target our preference/temporal misses and could close the 2-question raw gap; not implemented (the gap is noise-level and our embedder is already the stronger model).

---

## Verdict — does "best-benchmarked" hold honestly?

**Two-pillar test (per the brief):**

1. **A strong, comparable LongMemEval number** — YES, near-tie. R@5 0.962 vs mempalace published raw 0.966 = 2 questions out of 500 on identical config, with iai-mcp on the stronger embedder. Not a raw win, but a defensible honest tie.
2. **Breadth + reproducibility** — YES, clearly. We publish **10 distinct benchmarks**, each with a shipped script and a one-line reproduce command, honest-scale and multi-seed where applicable.

**Where iai-mcp genuinely leads** (axes peers do not document/match):
- ≤3000-token session-start budget (2993 standard, measured).
- Personal-fact retention: recall@10 0.9933 over 30 intervening sessions.
- Rescue@10 = 1.000 (post-contradiction current-fact retrieval).
- Local-first / no-cloud / verbatim-lossless / a real sleep-consolidation cycle (independently benchmarked: recall@10 preserved 1.000 through one tier0 consolidation cycle; mechanism analysis in bench/sleep_ablation.py).
- MOSAIC community-detection parity vs ground-truth (karate/football/LFR), deterministic.
- Memory footprint 589 MB @ 10k records.

**Where iai-mcp does NOT lead / honest gaps:** raw LongMemEval R@5 (−2 questions vs mempalace raw); recall p95 at 10k (368 ms); historical-verbatim retrieval (v8.0 regression, deferred); no Rust-vs-PyTorch embed comparison (path removed).

**Honest one-line verdict:** "Best-benchmarked" holds as **breadth + reproducibility + honesty**, plus genuine leads on session-budget / drift / rescue / local-first. It does **not** hold as "tops the LongMemEval raw leaderboard" — there it is a 2-question near-tie. Recommend the public tagline lean on *breadth + reproducibility + the specific leads* rather than a single-leaderboard claim.
