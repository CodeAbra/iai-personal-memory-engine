# iai-mcp Benchmarks

Measured on an Apple M2 Max (12-core, 64 GB). Each row carries a one-line
reproduce command so you can verify the numbers on your own hardware.

**Honesty rules:** no tuning on the test set, no hand-picked seeds, honest-scale
(multi-seed) where applicable. The only head-to-head comparison is LongMemEval
against a published baseline; every other number is our own metric with no
competitor column. Where a number is N/A it is stated plainly.

## Retrieval correctness

| Metric | iai-mcp | What it measures |
|---|---|---|
| **Rescue@10** | **1.000** | Ranks the current winning fact in the top 10 after a contradiction supersedes an earlier one. Honest-scale: 3 seeds × 1000 sessions × 2 slices. |
| **Historical-verbatim retrieval** | **1.000** | Retrieves the exact superseded/archived wording verbatim, separate from Rescue@10. |

Both are correctness metrics driven by the ranking and contradiction logic, not
by hardware.

Reproduce:

```bash
IAI_MCP_CRYPTO_PASSPHRASE=<x> python bench/contradiction_longitudinal.py --seeds 13 42 137
```

## LongMemEval-S (competitive arena)

LongMemEval-S, cleaned, 500 questions, session granularity, metric
`recall_any@k` (any gold session-id in the top-k), full haystack, raw (no
rerank). Config is identical on both sides; the only difference is the
embedder, and retrieval on this task is embedder-dominated.

| Metric | iai-mcp | Baseline (published, config-matched) |
|---|---|---|
| **R@5** (raw semantic) | **0.962** | 0.966 |

- **Embedders:** iai-mcp uses `bge-small-en-v1.5` (384d); the baseline uses
  `all-MiniLM-L6-v2` (384d). Both are 384d, user-turns-only per session, pure
  dense. `bge-small-en-v1.5` ranks higher than `all-MiniLM-L6-v2` on MTEB
  retrieval.
- The baseline numbers are published and config-matched (not re-run on this
  host).

Reproduce:

```bash
export HF_TOKEN='<your-huggingface-token>'
IAI_MCP_CRYPTO_PASSPHRASE=<x> python bench/longmemeval_blind.py \
  --split S --dataset cleaned --granularity session --rows 500
```

## Latency

Recall and embed latency depend on your CPU and store size, so reproduce them
on your own hardware rather than trusting a single reported figure:

```bash
python bench/full_recall_latency_probe.py    # recall p50/p95 across store sizes
python bench/embedder_latency.py             # single-embed p50/p95
```

The `bge-small-en-v1.5` embedder is 384d. Recall latency is dominated by the
rank/centrality stage at large store sizes.

## Honest gaps

- Raw LongMemEval R@5 is a near-tie with the published baseline, not a win
  (−2 questions out of 500 on identical config, on the stronger embedder).
- Recall latency grows with store size; the rank/centrality stage dominates at
  large N.
