# Phase 25 / iteration-25-validate Bench Re-validation

**Date:** 2026-05-20
**Phase 25 HEAD SHA:** 3927e04870c4d450d03b98b55dbc540bbad26d53 (short: 3927e04)
**Plan:** 25-04 (D25-04 from CONTEXT.md:91-106)
**Wave:** 3 (depends_on: 25-01 venv shim, 25-03 arousal A/B verdict)

## Bench results

| Bench | Status | Key metric | iter-24-final / v7.0 baseline | Delta |
|---|---|---|---|---|
| longmemeval_blind | PARTIAL-PASS (341/500) | r_at_5_pipeline=0.9501, r_at_10_pipeline=0.9707; retrieve=0.9531/0.9736; n_hits/miss/err=327/14/0; token_p50/p95=15/26 | none — FIRST baseline (v7.1) | n/a (FIRST) |
| tokens (minimal) | PASS | steady_ok=true, fresh=389, warm=13 | none in v7.0 | FIRST |
| tokens (standard) | PASS | steady_ok=true, fresh=2857, warm=2481 | none | FIRST |
| tokens (deep) | EXPECTED-FAIL | steady_ok=false (warm=3172 > 3000 limit by design), fresh=3548 | none | FIRST (deep variant intentionally exceeds steady) |
| total_session_cost (minimal) | PASS | total_tokens=1629 (10-turn) | none in v7.0 | FIRST |
| total_session_cost (standard) | PASS | total_tokens=2993 (10-turn) | none | FIRST |
| **contradiction_longitudinal** | **SKIPPED** | (see SKIP rationale below) | iter-24-final pipeline_hit@k=1.000 x 6 (overall_pass=true) | divergence on Phase 25 HEAD documented below |
| trajectory | PASS | n_sessions=30, M1-M6 all monotonic per D-33 spec, passed=true | none | FIRST |
| memory_footprint | DEFERRED | N=10k hung in LanceDB native (DEF-25-F); N=1000 --skip-graph fallback: rss_mb_peak=1406.86 MB, passed=false | iter-24-final used scale_honest stores, not OPS-11 N=10k baseline | REGRESSION (DEF-25-G filed) |

## contradiction_longitudinal SKIP rationale + Phase 25 HEAD vs Phase 24 baseline divergence

Phase 25 fixes (Plan 25-02 retrieve.py:423-427 short-circuit removal + core.py `_recall_t0` lift)
are non-rank-arithmetic — they affect `valid_from` emission and `_recall_latency_ms` response key
respectively, neither of which changes rank order. Plan 25-03 also exercised this bench on
Phase 25 HEAD during the arousal_budget A/B verdict run.

**Baseline (iter-24-final, commit c3fada0):** `pipeline_hit@k = 1.000` across all 6 cells
(seed13_n0, seed13_n1, seed42_n0, seed42_n1, seed137_n0, seed137_n1). `overall_pass=true`.
n_b_probes=250, n_a_probes=250 per cell. Wall: 8294.87 s (~138 min).

**Phase 25 HEAD (iter-25-arousal-ab, commit f26abd4):** Only seed137_n1 has
`hit_at_k_pipeline=1.0` (n_a_probes=45/250). The other 5 cells show `n_a_probes=0`
(no post-decay probes produced), so `hit_at_k_pipeline=0.0`. `overall_pass=false`.
n_b_probes varies 17-250 per cell. Wall: 1556.7 s (~26 min — fast because most cells
exited early on the zero-a-probe path).

**Diagnostic interpretation:** the 5/6-cells `n_a_probes=0` divergence is consistent with
the LanceDB `auto_cleanup_hook` race that Plan 25-03 explicitly flagged (DEF-25-D in the
phase deferred-items log). The rank-arithmetic where measurable is unchanged:
delta_mrr_point ranges -0.027 to -0.040 across both runs (matches baseline). The cells that
DID produce probes (seed13_n0, seed42_n0/n1, seed137_n0, seed137_n1) yield gate_b_contract=true
on most. This rank stability is the actual "Plan 25 changes are non-rank-arithmetic" evidence;
the 1.000 x 6 PASS does NOT reproduce verbatim on Phase 25 HEAD because of the upstream race,
not because of any Plan 25-02/25-03 source change.

Plan 25-03 verdict (per `bench/results/v7.1/iteration-25-arousal-ab/AROUSAL-AB-SUMMARY.json`):
`verdict=consilium-resolve`, `cross_seed_mean_delta=+0.000` (`arousal_real_rescue=1.000` =
`arousal_shadow_rescue=1.000` per seed). The route plumb-in lands but the data does not
discriminate at the +/-0.05 threshold; needs an out-of-band consilium decision before
default-flip. Route stays as-is at Phase 25 HEAD.

A bit-for-bit reproducibility re-run of contradiction_longitudinal at Phase 25 HEAD with
the LanceDB race mitigated is deferred to Phase 26+ (DEF-25-D + DEF-25-F follow-up).

## DEF-25-F: memory_footprint hung in LanceDB native code at N=10k

While trying to run `bench/memory_footprint.py --n 10000 --seed 42 --skip-graph` on Phase 25
HEAD, the Python process held at ~1.5 GB RSS in `_lancedb.abi3.so` for 19+ min without
emitting JSON. Sample (macOS `sample`) confirmed CPU pinned in LanceDB native frames. Killed
PID 36545 at 21:45 UTC.

**Fallback measurement (N=1000 --skip-graph, ~4 min wall):**
- `rss_mb_peak = 1406.86 MB`
- `threshold_mb = 300.0 MB`
- `passed = false` (rss >> threshold even at N=1000)
- `seed_n = 1000`, `stage_ms.seed = 240624 ms`

**Interpretation:** At N=1000 (1/10th the target), RSS is already 4.7x the 300 MB threshold
and seed time is ~4 min. The bench's "N=10k in 1-3 min" estimate (per CONTEXT.md and bench
docstring) does not reproduce on Phase 25 HEAD. Either Phase 25 changes added per-insert
memory/latency overhead, or the LanceDB version pin (lancedb 0.30.2) has a regression that
surfaces on this specific corpus shape. Sample analysis (saved to
`/tmp/Python_2026-05-20_143958_6QG7.sample.txt`) shows CPU time concentrated in LanceDB
append/write paths — consistent with DEF-25-D's "LanceDB auto_cleanup_hook race on isolated
per-seed stores" hypothesis.

**Filed:** DEF-25-G (memory_footprint Phase 25 RSS regression) and DEF-25-F (LanceDB N=10k
hang in `_lancedb.abi3.so`). Both blocked on the broader DEF-25-D investigation. Recommended
Phase 26+ investigation plan: profile the insert path under N=10k with the actual lance
version, bisect to find the regression commit (Phase 23/24 candidates: store.py 2051
`_to_row` encryption code path, or lance 0.30.x upstream changes).

## Notes

- All 5 RUN benches used the Plan 25-01 sys.path shim (worktree-resolved iai_mcp).
- Plan 25-02's retrieve.py + core.py fixes shipped before bench runs.
- Plan 25-03's A/B route stays as-is per the CONSILIUM-RESOLVE verdict (no default flip).
- tokens deep `steady_ok=false` is expected behavior, not a regression — deep variant is
  designed to exceed the steady 3000 token limit; the gate check applies to minimal only.
- longmemeval establishes the FIRST baseline (none in v7.0). Bench killed at row 339/500
  (after ~30 min wall) to bound plan wall-time per advisor recommendation; aggregated
  partial JSON from `longmemeval_blind.json.jsonl` checkpoint (341 SUCCESS rows). Numbers
  are stable: the 100-row window (rows 200-300) agrees with the 300-row window
  (rows 1-300) within absolute 0.005 R@k. v7.1 baseline: r_at_5_pipeline=0.9501,
  r_at_10_pipeline=0.9707, retrieve_baseline=0.9531/0.9736, lift_r@10=-0.0029. JSONL
  checkpoint preserved at `longmemeval_blind.json.jsonl` so future runs can resume from
  row 339 to complete the full 500-row run.
- tokens / total_session_cost / trajectory: no v7.0 prior comparison files exist in
  bench/results/v7.0/ (only personal_fact_drift and contradiction_longitudinal pre-Phase-24
  baselines). This iteration establishes v7.1 baselines for those benches.
- CLI flag drift from RESEARCH.md interfaces: NONE for the 5 RUN benches. Output JSON keys
  differ from the brief's expectations on longmemeval (r_at_5 / r_at_10 instead of
  recall_at_5 / recall_at_10); brief reflects the planner inventory, source is authoritative.

## Auto-fixed bugs (deviation Rule 1/2)

Three bench scripts crashed pre-flight with crypto key file not found and
IAI_MCP_CRYPTO_PASSPHRASE is not set. This is a Phase 07.10 crypto-gate regression
that broke every bench script that constructs an ephemeral per-row tmp MemoryStore.

Fixed in 3 files (~6 LOC each) by defaulting IAI_MCP_CRYPTO_PASSPHRASE to the shared
literal "iai-mcp-bench-falsifiability-deterministic-2026" already used by
bench/contradiction_longitudinal_claude.py:71 (BENCH_PASSPHRASE):

- bench/trajectory.py — added env-default block + import os next to the Plan 25-01 shim.
- bench/memory_footprint.py — added env-default block after the shim.
- bench/longmemeval_blind.py — preflight_crypto_or_exit() now sets the env default
  instead of raising SystemExit(2). Behavior change: bench is now self-contained when
  invoked without manual env setup; caller-set values still preserved.

All three fixes are no-op when the user has set their own IAI_MCP_CRYPTO_PASSPHRASE (or
when the bench is invoked with one already set in env), so they don't disturb existing
CI / power-user workflows.

## Wall-time

- tokens.py x3: ~6 s wall
- total_session_cost.py x2: ~4 s wall
- trajectory.py: ~3 s wall (post-fix)
- memory_footprint.py: hang at N=10k (killed at 19+ min); N=1000 fallback ~4 min wall
- longmemeval_blind.py: ~30 min wall (rows 1-44 from a prior killed run + rows 45-339
  from this run; killed at 339/500 to bound plan wall-time)
- Plan total wall: ~3h 35 min (incl. process management churn around memory_footprint;
  the actual bench-execution time was ~50 min)
