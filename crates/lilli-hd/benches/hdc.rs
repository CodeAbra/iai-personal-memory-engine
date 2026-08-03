//! Speedup gate for the hypervector kernels.
//!
//! Benches the pure-Rust inner kernels (projection-apply, BSC majority-vote
//! bundle, popcount Hamming) and divides the measured Python baseline
//! (`fixtures/golden/hdc/py_baseline.json`, produced by the baseline script)
//! into the measured Rust time to get a per-kernel speedup.
//!
//! The hard gate covers the production hot-path kernels only: the popcount
//! Hamming similarity and the SWAR majority-vote bundle, both of which must
//! clear `HDC_SPEEDUP_MIN`. The projection-apply is reported parity-only:
//! production projection runs on numpy (a single `emb @ P` that dispatches to
//! the platform BLAS sgemv, which is memory-bandwidth-bound and already at the
//! physical ceiling — a scalar Rust matmul cannot beat it). The Rust projection
//! kernel and its 100-embedding golden are kept to prove bit-identical
//! reproduction, but the projection speedup is NOT a gate. The Python
//! denominators are measured and committed, never invented.

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use criterion::{criterion_group, criterion_main, Criterion};

use lilli_hd::bsc::{bundle_impl, hamming_impl};
use lilli_hd::projection::project_impl;
use lilli_parity::thresholds::{meets_or_beats, HDC_SPEEDUP_MIN};

const D_4096: usize = 4096;

fn fixtures_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("IAI_MCP_HDC_FIXTURES") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/golden/hdc")
}

/// Read the measured Python ns/op for a named kernel from the committed baseline.
fn py_baseline_ns(field: &str) -> f64 {
    let path = fixtures_dir().join("py_baseline.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("read py_baseline.json ({}): {e}", path.display()));
    let json: serde_json::Value = serde_json::from_str(&text).expect("py_baseline.json parses");
    let ns = json
        .get(field)
        .and_then(|v| v.as_f64())
        .unwrap_or_else(|| panic!("py_baseline.json missing numeric field {field:?}"));
    assert!(
        ns > 0.0,
        "py_baseline.json field {field:?} must be positive"
    );
    ns
}

/// The first frozen 384-d embedding, used as the projection input.
fn first_embedding() -> Vec<f32> {
    let raw =
        std::fs::read(fixtures_dir().join("proj_100_in.bin")).expect("proj_100_in.bin readable");
    raw.chunks_exact(4)
        .take(384)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Eight distinct 4096-bit packed hypervectors to vote over (the mid-cap set the
/// Python baseline times).
fn bound_hvs_4096() -> Vec<Vec<u8>> {
    (0..8u8)
        .map(|seed| {
            let mut hv = vec![0u8; D_4096 / 8];
            for (i, byte) in hv.iter_mut().enumerate() {
                *byte = (i as u8)
                    .wrapping_mul(31)
                    .wrapping_add(seed.wrapping_mul(97));
            }
            hv
        })
        .collect()
}

/// Median ns/op over a manual timed loop. Criterion reports distribution; this
/// drives the hard assertion.
fn measure_ns<F: FnMut()>(iters: u64, mut f: F) -> f64 {
    // Warm up.
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    elapsed.as_secs_f64() / iters as f64 * 1e9
}

fn assert_speedup(name: &str, py_ns: f64, rust_ns: f64) -> f64 {
    let speedup = py_ns / rust_ns;
    println!(
        "[hdc speedup] {name}: python {py_ns:.1} ns/op / rust {rust_ns:.1} ns/op = {speedup:.1}x (gated >= {HDC_SPEEDUP_MIN:.0}x)"
    );
    assert!(
        meets_or_beats(speedup, HDC_SPEEDUP_MIN, false),
        "{name} speedup {speedup:.2}x is below the {HDC_SPEEDUP_MIN:.0}x floor \
         (python {py_ns:.1} ns/op, rust {rust_ns:.1} ns/op)"
    );
    speedup
}

/// Report a kernel's measured speedup WITHOUT gating it. Used for projection,
/// whose production path is numpy/BLAS (bandwidth-bound, at the physical
/// ceiling); the Rust kernel exists for bit-parity proof, not to win on speed.
fn report_only(name: &str, py_ns: f64, rust_ns: f64) -> f64 {
    let speedup = py_ns / rust_ns;
    println!(
        "[hdc speedup] {name}: python {py_ns:.1} ns/op / rust {rust_ns:.1} ns/op = {speedup:.2}x \
         (parity-only, numpy-backed in production — bandwidth-bound, NOT gated)"
    );
    speedup
}

fn hdc(c: &mut Criterion) {
    let emb = first_embedding();
    let bound = bound_hvs_4096();
    let (ha, hb) = (&bound[0], &bound[1]);

    // Criterion measurements (statistical, for the report).
    c.bench_function("project", |b| {
        b.iter(|| black_box(project_impl(black_box(&emb)).unwrap()))
    });
    c.bench_function("bsc_bundle_vote", |b| {
        b.iter(|| black_box(bundle_impl(black_box(&bound), D_4096).unwrap()))
    });
    c.bench_function("bsc_hamming", |b| {
        b.iter(|| black_box(hamming_impl(black_box(ha), black_box(hb))))
    });

    // Hard gate: measured Rust ns vs the measured Python baseline ns.
    let project_rust = measure_ns(20_000, || {
        black_box(project_impl(black_box(&emb)).unwrap());
    });
    let bundle_rust = measure_ns(200_000, || {
        black_box(bundle_impl(black_box(&bound), D_4096).unwrap());
    });
    let hamming_rust = measure_ns(2_000_000, || {
        black_box(hamming_impl(black_box(ha), black_box(hb)));
    });

    // Hot-path kernels are gated. The bundle denominator is the vote-only
    // Python time (`bsc_bundle_vote_ns`) — the exact majority vote the native
    // kernel replaces (role lookups and binds stay host-side, so timing the
    // full `bsc.bundle` would mismatch the kernel boundary).
    assert_speedup("hamming", py_baseline_ns("hamming_ns"), hamming_rust);
    assert_speedup(
        "bsc_bundle",
        py_baseline_ns("bsc_bundle_vote_ns"),
        bundle_rust,
    );

    // Projection is parity-only: production runs on numpy/BLAS. Reported, not gated.
    report_only("project", py_baseline_ns("from_embedding_ns"), project_rust);
}

criterion_group!(benches, hdc);
criterion_main!(benches);
