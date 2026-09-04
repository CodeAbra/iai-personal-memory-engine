# A/B: ANN library vs own exact vector index

One store copy (20.3k live records, 384-d, production data), identical cue
set (10 RU/EN cues), both sides rebuild the index from storage at boot, warm
discipline (3 warmup cues, then 50 reps x 10 cues timed), M2 Max, idle
machine, sequential runs within one hour.

| metric | ANN library | own exact index | verdict |
|---|---|---|---|
| boot incl. index build | 16.06 s | 1.51 s | 10.6x faster |
| query_similar p50 | 3.61 ms | 8.01 ms | +4.4 ms |
| query_similar p95 | 4.13 ms | 8.63 ms | +4.5 ms |
| top-10 accuracy vs exact truth | 100% on this cue set (graph-degradation tail risk remains) | exact by construction | guaranteed |

Reading: the flat fill wins boot by an order of magnitude (no graph
construction), which is what every daemon restart and nightly rebuild pays.
The exact sweep costs ~4 ms more per query than graph traversal at this
corpus size — ~0.8% of the ~500-700 ms warm recall pipeline, invisible
end-to-end. The ANN's approximation happened to be accurate on this cue set;
the exact index makes that a guarantee instead of a property to monitor, and
retires the fragmented-graph query failures and delete-slot hazards the
recall path carried defensive code for.
