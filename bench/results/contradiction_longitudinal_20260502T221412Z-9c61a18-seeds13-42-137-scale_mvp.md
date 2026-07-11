# Contradiction-longitudinal falsifiability bench — PASS

**Run ID:** 20260502T221412Z-9c61a18-seeds13-42-137-scale_mvp
**Duration:** 253.6s

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
| `wall_clock_start_utc` | 2026-05-02T22:14:12.323255+00:00 |
| `scale` | mvp |
| `n_sessions` | 200 |
| `n_probes_pre` | 50 |
| `n_probes_post` | 50 |
| `n_slices` | [0, 1] |
| `k_hits` | 10 |
| `a_threshold` | 0.98 |
| `candidate_pool_size` | 200 |
| `bootstrap_resamples` | 10000 |
| `floor_mode` | relaxed |
| `wall_clock_duration_seconds` | 253.63 |

## Cross-seed (B robustness)

| N slice | ΔMRR mean | stdev | min | max | robust? |
|---|---|---|---|---|---|
| n_0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NO |
| n_1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | NO |

## Per-cell detail

| seed | N | A hit@k (pipe / cos) | A floor | B-class ΔMRR (CI) | B-contract hint% / anti-hits% | gate A | gate B-class | gate B-contract |
|---|---|---|---|---|---|---|---|---|
| 13 | 0 | 1.000 / 0.700 | 0 | 0.0000 (0.0000, 0.0000) | 1.000 / 0.800 | PASS | FAIL | PASS |
| 13 | 1 | 1.000 / 0.700 | 0 | 0.0000 (0.0000, 0.0000) | 1.000 / 0.800 | PASS | FAIL | PASS |
| 42 | 0 | 1.000 / 0.640 | 0 | 0.0000 (0.0000, 0.0000) | 1.000 / 1.000 | PASS | FAIL | PASS |
| 42 | 1 | 1.000 / 0.640 | 0 | 0.0000 (0.0000, 0.0000) | 1.000 / 1.000 | PASS | FAIL | PASS |
| 137 | 0 | 1.000 / 0.820 | 0 | 0.0000 (0.0000, 0.0000) | 1.000 / 1.000 | PASS | FAIL | PASS |
| 137 | 1 | 1.000 / 0.820 | 0 | 0.0000 (0.0000, 0.0000) | 1.000 / 1.000 | PASS | FAIL | PASS |

**Cross-seed robust gate (B-classical only):** FAIL (expected: B-class is not the architectural promise)
**Overall verdict (uses gate_a + gate_b_contract):** PASS

## Notes on metric design

- **Metric A (verbatim preserved)** tests REQUIREMENTS.md MEM-05 — the system's promise that contradiction = reconsolidation, never overwrite. Pipeline beating cosine here = real architectural advantage.
- **Metric B-classical (rank current above cosine)** tests an expectation that does NOT appear in any design doc. Per REQUIREMENTS.md MCP-01 + 02-CONTEXT.md, the system uses dual-route + inhibitory edges + hints, not rerank. Expect ΔMRR ≈ 0; this is a feature, not a bug.
- **Metric B-contract (s4_contradiction hint OR anti_hits ≥80%)** tests what the system actually promises (REQUIREMENTS.md MEM-08, MCP-01 dual-route). Cosine cannot do either; pipeline either signals contradictions or it doesn't.
