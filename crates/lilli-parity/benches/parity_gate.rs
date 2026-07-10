//! Parity-gate bench skeleton.
//!
//! The real measurement targets (the Rust store and recall path) do not exist
//! yet, so this body is a stub that references the committed threshold constants
//! from the library so they live in exactly one place. The live measurements and
//! their match-or-beat assertions land when the store is wired in.

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use lilli_parity::thresholds::{
    HDC_SPEEDUP_MIN, RECALL_P95_N100_MS, RECALL_P95_N10K_MS, RECALL_P95_N1K_MS, RSS_N10K_MB,
    SESSION_TOKEN_BUDGET,
};

fn parity_gate(c: &mut Criterion) {
    c.bench_function("threshold_reference", |b| {
        b.iter(|| {
            black_box(RECALL_P95_N100_MS);
            black_box(RECALL_P95_N1K_MS);
            black_box(RECALL_P95_N10K_MS);
            black_box(RSS_N10K_MB);
            black_box(HDC_SPEEDUP_MIN);
            black_box(SESSION_TOKEN_BUDGET);
        });
    });
}

criterion_group!(benches, parity_gate);
criterion_main!(benches);
