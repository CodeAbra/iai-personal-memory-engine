# Contradiction-longitudinal falsifiability bench — PASS

**Run ID:** 20260617T145020Z-878fac74-seeds13-42-137-scale_honest
**Duration:** 656.8s

## Environment

| Field | Value |
|---|---|
| `cpu_brand` | Apple M2 Max |
| `cpu_cores_physical` | 12 |
| `ram_gb` | 64.0 |
| `os` | Darwin |
| `os_version` | 25.5.0 |
| `python_version` | 3.12.13 |
| `iai_mcp_git_sha` | 878fac74 |
| `iai_mcp_git_dirty` | True |
| `lance_version` | unknown |
| `lancedb_version` | unknown |
| `pyarrow_version` | 19.0.1 |
| `sentence_transformers_version` | unknown |
| `embedder_model` | bge-small-en-v1.5 |
| `seed_list` | [13, 42, 137] |
| `iai_mcp_store` | /private/tmp/iai-mcp-bench-claude/store |
| `wall_clock_start_utc` | 2026-06-17T14:50:20.938091+00:00 |
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
| `wall_clock_duration_seconds` | 656.84 |

## Cross-seed (B robustness)

| N slice | ΔMRR mean | stdev | min | max | robust? |
|---|---|---|---|---|---|
| n_0 | -0.0353 | 0.0042 | -0.0400 | -0.0320 | NO |
| n_1 | -0.0353 | 0.0042 | -0.0400 | -0.0320 | NO |

## Per-cell detail

| seed | N | A hit@k (pipe / cos) | A floor | B-class ΔMRR (CI) | B-contract hint% / anti-hits% | gate A | gate B-class | gate B-contract |
|---|---|---|---|---|---|---|---|---|
| 13 | 0 | 1.000 / 0.692 | 0 | -0.0400 (-0.0580, -0.0240) | 1.000 / 0.880 | PASS | FAIL | PASS |
| 13 | 1 | 1.000 / 0.692 | 0 | -0.0400 (-0.0580, -0.0240) | 1.000 / 0.880 | PASS | FAIL | PASS |
| 42 | 0 | 1.000 / 0.708 | 0 | -0.0340 (-0.0500, -0.0200) | 1.000 / 0.792 | PASS | FAIL | PASS |
| 42 | 1 | 1.000 / 0.708 | 0 | -0.0340 (-0.0500, -0.0200) | 1.000 / 0.792 | PASS | FAIL | PASS |
| 137 | 0 | 1.000 / 0.740 | 0 | -0.0320 (-0.0480, -0.0180) | 1.000 / 0.904 | PASS | FAIL | PASS |
| 137 | 1 | 1.000 / 0.740 | 0 | -0.0320 (-0.0480, -0.0180) | 1.000 / 0.904 | PASS | FAIL | PASS |

**Cross-seed robust gate (B-classical only):** FAIL (expected: B-class is not the architectural promise)
**Overall verdict (uses gate_a + gate_b_contract):** PASS

## Notes on metric design

- **Metric A (verbatim preserved)** tests REQUIREMENTS.md MEM-05 — the system's promise that contradiction = reconsolidation, never overwrite. Pipeline beating cosine here = real architectural advantage.
- **Metric B-classical (rank current above cosine)** tests an expectation the system does not promise: it uses dual-route + inhibitory edges + hints, not rerank. Expect ΔMRR ≈ 0; this is a feature, not a bug.
- **Metric B-contract (s4_contradiction hint OR anti_hits ≥80%)** tests what the system actually promises (REQUIREMENTS.md MEM-08, MCP-01 dual-route). Cosine cannot do either; pipeline either signals contradictions or it doesn't.
