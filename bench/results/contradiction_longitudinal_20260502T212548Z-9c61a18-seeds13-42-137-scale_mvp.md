# Contradiction-longitudinal falsifiability bench — FAIL

**Run ID:** 20260502T212548Z-9c61a18-seeds13-42-137-scale_mvp
**Duration:** 257.1s

## Environment

| Field | Value |
|---|---|
| `cpu_brand` | Apple M2 Max |
| `cpu_cores_physical` | 12 |
| `ram_gb` | 64.0 |
| `os` | Darwin |
| `os_version` | 25.3.0 |
| `python_version` | 3.12.13 |
| `iai_mcp_git_sha` | 9c61a18 |
| `iai_mcp_git_dirty` | True |
| `lance_version` | unknown |
| `lancedb_version` | 0.30.2 |
| `pyarrow_version` | 23.0.1 |
| `sentence_transformers_version` | 5.4.1 |
| `embedder_model` | bge-small-en-v1.5 |
| `seed_list` | [13, 42, 137] |
| `iai_mcp_store` | /private/tmp/iai-mcp-bench-claude/store |
| `wall_clock_start_utc` | 2026-05-02T21:25:48.462008+00:00 |
| `scale` | mvp |
| `n_sessions` | 200 |
| `n_probes_pre` | 50 |
| `n_probes_post` | 50 |
| `n_slices` | [0, 1] |
| `k_hits` | 10 |
| `a_threshold` | 0.98 |
| `candidate_pool_size` | 200 |
| `bootstrap_resamples` | 10000 |
| `wall_clock_duration_seconds` | 257.1 |

## Cross-seed (B robustness)

| N slice | ΔMRR mean | stdev | min | max | robust? |
|---|---|---|---|---|---|
| n_0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NO |
| n_1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NO |

## Per-cell detail

| seed | N | A hit@k (pipe / cos) | A floor viols | B ΔMRR (CI) | B max-regression | gate A | gate B |
|---|---|---|---|---|---|---|---|
| 13 | 0 | 1.000 / 0.700 | 37 | 0.0000 (0.0000, 0.0000) | 0 | FAIL | FAIL |
| 13 | 1 | 1.000 / 0.700 | 37 | 0.0000 (0.0000, 0.0000) | 0 | FAIL | FAIL |
| 42 | 0 | 1.000 / 0.640 | 34 | 0.0000 (0.0000, 0.0000) | 0 | FAIL | FAIL |
| 42 | 1 | 1.000 / 0.640 | 34 | 0.0000 (0.0000, 0.0000) | 0 | FAIL | FAIL |
| 137 | 0 | 1.000 / 0.820 | 36 | 0.0000 (0.0000, 0.0000) | 0 | FAIL | FAIL |
| 137 | 1 | 1.000 / 0.820 | 36 | 0.0000 (0.0000, 0.0000) | 0 | FAIL | FAIL |

**Cross-seed robust gate:** FAIL
**Overall verdict:** FAIL
