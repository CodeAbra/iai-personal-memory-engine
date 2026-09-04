# Contradiction-longitudinal falsifiability bench — FAIL

**Run ID:** 20260519T101140Z-c31fb99-seeds13-42-137-scale_honest
**Duration:** 8438.3s

## Environment

| Field | Value |
|---|---|
| `cpu_brand` | Apple M2 Max |
| `cpu_cores_physical` | 12 |
| `ram_gb` | 64.0 |
| `os` | Darwin |
| `os_version` | 25.3.0 |
| `python_version` | 3.12.13 |
| `iai_mcp_git_sha` | c31fb99 |
| `iai_mcp_git_dirty` | True |
| `lance_version` | unknown |
| `lancedb_version` | 0.30.2 |
| `pyarrow_version` | 24.0.0 |
| `sentence_transformers_version` | 5.5.0 |
| `embedder_model` | bge-small-en-v1.5 |
| `seed_list` | [13, 42, 137] |
| `iai_mcp_store` | /private/tmp/iai-mcp-bench-claude/store |
| `wall_clock_start_utc` | 2026-05-19T10:11:40.614753+00:00 |
| `scale` | honest |
| `n_sessions` | 1000 |
| `n_probes_pre` | 250 |
| `n_probes_post` | 250 |
| `n_slices` | [0, 1] |
| `k_hits` | 10 |
| `a_threshold` | 0.98 |
| `candidate_pool_size` | 200 |
| `bootstrap_resamples` | 10000 |
| `floor_mode` | relaxed |
| `wall_clock_duration_seconds` | 8438.31 |

## Cross-seed (B robustness)

| N slice | ΔMRR mean | stdev | min | max | robust? |
|---|---|---|---|---|---|
| n_0 | 0.0167 | 0.0031 | 0.0140 | 0.0200 | YES |
| n_1 | 0.0167 | 0.0031 | 0.0140 | 0.0200 | YES |

## Per-cell detail

| seed | N | A hit@k (pipe / cos) | A floor | B-class ΔMRR (CI) | B-contract hint% / anti-hits% | gate A | gate B-class | gate B-contract |
|---|---|---|---|---|---|---|---|---|
| 13 | 0 | 0.888 / 1.000 | 28 | 0.0200 (-0.0080, 0.0480) | 1.000 / 0.000 | FAIL | FAIL | PASS |
| 13 | 1 | 0.888 / 1.000 | 28 | 0.0200 (-0.0080, 0.0480) | 1.000 / 0.000 | FAIL | FAIL | PASS |
| 42 | 0 | 0.896 / 1.000 | 26 | 0.0140 (-0.0100, 0.0400) | 1.000 / 0.000 | FAIL | FAIL | PASS |
| 42 | 1 | 0.896 / 1.000 | 26 | 0.0140 (-0.0100, 0.0400) | 1.000 / 0.000 | FAIL | FAIL | PASS |
| 137 | 0 | 0.916 / 1.000 | 21 | 0.0160 (-0.0080, 0.0420) | 1.000 / 0.000 | FAIL | FAIL | PASS |
| 137 | 1 | 0.916 / 1.000 | 21 | 0.0160 (-0.0080, 0.0420) | 1.000 / 0.000 | FAIL | FAIL | PASS |

**Cross-seed robust gate (B-classical only):** PASS
**Overall verdict (uses gate_a + gate_b_contract):** FAIL

## Notes on metric design

- **Metric A (verbatim preserved)** tests REQUIREMENTS.md MEM-05 — the system's promise that contradiction = reconsolidation, never overwrite. Pipeline beating cosine here = real architectural advantage.
- **Metric B-classical (rank current above cosine)** tests an expectation that does NOT appear in any design doc. Per REQUIREMENTS.md MCP-01 + 02-CONTEXT.md, the system uses dual-route + inhibitory edges + hints, not rerank. Expect ΔMRR ≈ 0; this is a feature, not a bug.
- **Metric B-contract (s4_contradiction hint OR anti_hits ≥80%)** tests what the system actually promises (REQUIREMENTS.md MEM-08, MCP-01 dual-route). Cosine cannot do either; pipeline either signals contradictions or it doesn't.
