# LongMemEval-S Aggregate Report

- Source: `bench/lme500/output/lme500-v1.json`
- n = 500, errors = 0
- 95% CI via bootstrap percentile method (10000 resamples, seed=42)

## Overall

| Prong | R@5 | R@5 95% CI | R@10 | R@10 95% CI |
|---|---|---|---|---|
| X (retrieve_recall — flat-cosine baseline) | 0.956 | [0.938, 0.974] | 0.976 | [0.962, 0.988] |
| Y (pipeline_recall — full graph-native pipeline) | 0.944 | [0.924, 0.964] | 0.946 | [0.926, 0.964] |
| **Architecture lift Y − X** | **-0.012** | — | **-0.030** | — |

## Per question type

| Type | n | X R@5 | Y R@5 | Lift R@5 | X R@10 | Y R@10 | Lift R@10 |
|---|---|---|---|---|---|---|---|
| `knowledge-update` | 78 | 0.974 | 0.987 | +0.013 | 0.987 | 1.000 | +0.013 |
| `multi-session` | 133 | 0.970 | 0.955 | -0.015 | 0.985 | 0.955 | -0.030 |
| `single-session-assistant` | 56 | 1.000 | 1.000 | +0.000 | 1.000 | 1.000 | +0.000 |
| `single-session-preference` | 30 | 0.900 | 0.833 | -0.067 | 0.933 | 0.833 | -0.100 |
| `single-session-user` | 70 | 0.957 | 0.943 | -0.014 | 0.986 | 0.943 | -0.043 |
| `temporal-reasoning` | 133 | 0.925 | 0.910 | -0.015 | 0.955 | 0.910 | -0.045 |

⚠️ = n < 30, low statistical power for that bin.

## Notes

- Errors (graph-build failures, malformed rows, etc.) are counted as miss for **both** prongs (R@k = 0).
- Mean is the unweighted row average; CI is bootstrap percentile.
- Architecture lift = mean(Y) − mean(X). The CI of the lift itself is not computed here (would require paired bootstrap on the (Y_i, X_i) tuples — TODO if needed).
