//! Insert-scaling gate: prove the insert cost is linear in the row count and
//! the per-insert page reads track tree height, not the number of rows.
//!
//! Each fixture inserts N records of the real store's shape (~1656 B: a 1536 B
//! float32 embedding plus a small ciphertext-sized tail) into a fresh tree, in
//! one write transaction (the production batched-commit path), in write-ahead
//! mode. It records:
//!   - wall time and throughput (rows/s),
//!   - average page reads per insert (the height proxy),
//!   - the whole-tree-walk count over the insert loop (must stay zero — a
//!     linear sweep per insert is the throughput-killing regression),
//! and asserts the committed gates: doubling the row count at most ~doubles the
//! time, throughput holds its floor, per-insert reads stay bounded by height,
//! and no whole-tree walk occurs on the insert path.
//!
//! `harness = false`: this is a one-shot scaling measurement (insert N once and
//! time it), not a micro-benchmark, so it runs as a plain measurement program
//! rather than under the statistical sampler.

use std::time::Instant;

use lillibrain::gates::{
    doubling_ratio_ok, reads_per_insert_ok, throughput_ok, DOUBLING_TIME_RATIO_MAX,
    INSERT_ROWS_PER_S_FLOOR, READS_PER_INSERT_MAX,
};
use lillibrain::Store;

/// The on-page record geometry mirrors the real store: a 1536 B float32
/// embedding plus a ciphertext-sized tail, ~1656 B total. Below the overflow
/// threshold, so every record lands inline on a leaf — the same page traffic
/// the production insert path sees.
const RECORD_LEN: usize = 1656;

/// One synthetic record. The content is irrelevant to the insert cost (the store
/// is blind to the bytes); only the length matters for page packing.
fn make_record(key: i64) -> Vec<u8> {
    let mut buf = vec![0u8; RECORD_LEN];
    buf[0..8].copy_from_slice(&key.to_be_bytes());
    // A varying tail so the bytes are not trivially compressible by the OS.
    for (i, b) in buf.iter_mut().enumerate().skip(8) {
        *b = (i as u8).wrapping_add(key as u8);
    }
    buf
}

/// Insert `n` records into a fresh write-ahead-mode store in one transaction.
/// Returns `(elapsed_seconds, reads_per_insert, full_scans)`.
fn measure_insert(n: i64) -> (f64, f64, u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("insert_gate.lilli");

    let store = Store::open(&path).expect("open");
    store.enable_wal_mode().expect("enable wal");
    let root = store.create_tree().expect("create tree");

    // Measure only the insert loop: counters reset after setup so the page
    // reads attributed below come purely from the inserts.
    store.reset_read_count();
    store.reset_full_scan_count();

    let start = Instant::now();
    store.begin_write().expect("begin");
    let tree = store.tree(root);
    for key in 0..n {
        let value = make_record(key);
        tree.insert(key, &value).expect("insert");
    }
    store.commit().expect("commit");
    let elapsed = start.elapsed().as_secs_f64();

    let reads = store.read_count();
    let full_scans = store.full_scan_count();
    let reads_per_insert = reads as f64 / n as f64;

    (elapsed, reads_per_insert, full_scans)
}

fn main() {
    // Fast drive-cache sync for the measurement run; the full-flush latency
    // would otherwise dominate and obscure the algorithmic shape.
    if std::env::var("LILLI_FSYNC_MODE").is_err() {
        std::env::set_var("LILLI_FSYNC_MODE", "fast");
    }

    let sizes: [i64; 4] = [5_000, 10_000, 20_000, 30_000];
    let mut elapsed = [0f64; 4];
    let mut reads_per_insert = [0f64; 4];
    let mut full_scans = [0u64; 4];

    println!();
    println!("insert scaling gate (record geometry {RECORD_LEN} B, write-ahead mode)");
    println!(
        "{:>8}  {:>10}  {:>12}  {:>16}  {:>11}",
        "rows", "secs", "rows/s", "reads/insert", "full_scans"
    );
    for (i, &n) in sizes.iter().enumerate() {
        let (secs, rpi, fs) = measure_insert(n);
        elapsed[i] = secs;
        reads_per_insert[i] = rpi;
        full_scans[i] = fs;
        let rps = n as f64 / secs;
        println!("{n:>8}  {secs:>10.4}  {rps:>12.1}  {rpi:>16.2}  {fs:>11}");
    }

    // --- Gate assertions ---------------------------------------------------

    // Doubling 10k -> 20k stays linear (constant overhead absorbed by the slack).
    let ratio_10_20 = elapsed[2] / elapsed[1];
    // 30k shows no super-linear blowup vs the 10k baseline (3x rows -> <= ~3.3x time).
    let ratio_10_30 = elapsed[3] / elapsed[1];

    println!();
    println!("t(20k)/t(10k) = {ratio_10_20:.3}  (gate <= {DOUBLING_TIME_RATIO_MAX})");
    println!(
        "t(30k)/t(10k) = {ratio_10_30:.3}  (gate <= {:.3}, linear 3x)",
        DOUBLING_TIME_RATIO_MAX * 1.5
    );

    let rps_30k = sizes[3] as f64 / elapsed[3];
    println!("throughput(30k) = {rps_30k:.1} rows/s  (gate >= {INSERT_ROWS_PER_S_FLOOR})");
    println!(
        "reads/insert(30k) = {:.2}  (gate <= {READS_PER_INSERT_MAX})",
        reads_per_insert[3]
    );
    println!("full_scans on the insert path = {} (gate == 0)", full_scans[3]);

    assert!(
        doubling_ratio_ok(ratio_10_20),
        "insert is not linear: t(20k)/t(10k) = {ratio_10_20:.3} exceeds {DOUBLING_TIME_RATIO_MAX}"
    );
    assert!(
        ratio_10_30 <= DOUBLING_TIME_RATIO_MAX * 1.5,
        "insert shows super-linear blowup at 30k: t(30k)/t(10k) = {ratio_10_30:.3}"
    );
    assert!(
        throughput_ok(rps_30k),
        "insert throughput below floor at 30k: {rps_30k:.1} rows/s"
    );
    for (i, &n) in sizes.iter().enumerate() {
        assert!(
            reads_per_insert_ok(reads_per_insert[i]),
            "per-insert page reads grow with rows at {n}: {:.2} reads/insert",
            reads_per_insert[i]
        );
        assert_eq!(
            full_scans[i], 0,
            "insert path issued a whole-tree walk at {n} rows (the linear-sweep regression)"
        );
    }

    // The per-insert read count must be independent of the row count: a tree of
    // ~3x the rows reads at most a couple more pages per insert (one extra level
    // at most), never proportionally more.
    let reads_growth = reads_per_insert[3] / reads_per_insert[0];
    println!(
        "reads/insert growth 5k -> 30k = {reads_growth:.3} (gate <= 2.0, height-bounded)"
    );
    assert!(
        reads_growth <= 2.0,
        "per-insert reads scale with the row count (5k -> 30k factor {reads_growth:.3}): \
         the insert cost is not height-bounded"
    );

    println!();
    println!("insert scaling gate: PASSED");
}
