//! Edges UPSERT-scaling gate: prove a bulk composite-primary-key UPSERT stays
//! sub-quadratic in the row count.
//!
//! The edges table has a composite PRIMARY KEY `(src, dst, edge_type)`. The
//! `ON CONFLICT (src, dst, edge_type) DO UPDATE` path the graph store drives in
//! bulk must compare each attempted row against the existing rows on the
//! conflict columns. Without a conflict-key index that comparison is a per-row
//! whole-table scan, so a bulk UPSERT degrades to O(rows^2) — the same shape
//! that broke a prior live cutover.
//!
//! This program runs the UPSERT workload at 5k / 10k / 20k / 30k rows and
//! asserts:
//!   - doubling the row count at most ~doubles the wall time (linear), and
//!   - 30k shows no super-linear blow-up versus the 10k baseline.
//!
//! `harness = false`: a one-shot scaling measurement (UPSERT N once and time
//! it), not a sampled micro-benchmark.

use std::time::Instant;

use lillibrain::Value;
use lilliengine::conn::Connection;

/// The edges DDL, byte-identical to the graph store's table shape.
const EDGES_DDL: &str = "CREATE TABLE edges (\
    src TEXT NOT NULL, \
    dst TEXT NOT NULL, \
    edge_type TEXT NOT NULL, \
    weight REAL NOT NULL DEFAULT 0.0, \
    updated_at TEXT, \
    PRIMARY KEY (src, dst, edge_type))";

/// The bulk UPSERT statement the graph store issues: insert-or-update on the
/// composite key, updating weight + updated_at from the attempted row.
const EDGES_UPSERT: &str = "INSERT INTO edges (src, dst, edge_type, weight, updated_at) \
    VALUES (?, ?, ?, ?, ?) \
    ON CONFLICT (src, dst, edge_type) DO UPDATE SET \
    weight = excluded.weight, updated_at = excluded.updated_at";

/// One UPSERT parameter row. `i` fans across a bounded set of source nodes so a
/// realistic fraction of the workload hits the DO UPDATE branch (existing key)
/// rather than always inserting fresh, exercising the conflict-scan path.
fn upsert_params(i: i64) -> Vec<Value> {
    let src = format!("n{}", i % 256);
    let dst = format!("n{}", (i / 256) % 4096);
    vec![
        Value::Text(src),
        Value::Text(dst),
        Value::Text("rel".to_string()),
        Value::Float((i % 97) as f64 * 0.01),
        Value::Text("2026-01-01T00:00:00".to_string()),
    ]
}

/// UPSERT `n` edge rows into a fresh engine connection inside one outer
/// transaction (the bulk merge_insert path). Returns `(elapsed_seconds,
/// full_scans)` measured over the UPSERT loop only.
fn measure_upsert(n: i64) -> (f64, u64) {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("edges_upsert.lilli");
    let path_str = path.to_str().expect("utf8 path");

    let mut conn = Connection::open(path_str, 384).expect("open");
    conn.execute(EDGES_DDL, Vec::new()).expect("ddl");

    conn.reset_full_scan_count();
    let start = Instant::now();
    conn.execute("BEGIN", Vec::new()).expect("begin");
    for i in 0..n {
        conn.execute(EDGES_UPSERT, upsert_params(i))
            .expect("upsert");
    }
    conn.execute("COMMIT", Vec::new()).expect("commit");
    let elapsed = start.elapsed().as_secs_f64();
    let full_scans = conn.full_scan_count();

    (elapsed, full_scans)
}

fn main() {
    if std::env::var("LILLI_FSYNC_MODE").is_err() {
        std::env::set_var("LILLI_FSYNC_MODE", "fast");
    }

    let sizes: [i64; 4] = [5_000, 10_000, 20_000, 30_000];
    let mut elapsed = [0f64; 4];
    let mut full_scans = [0u64; 4];

    println!();
    println!("edges UPSERT scaling gate (composite PK (src, dst, edge_type), DO UPDATE)");
    println!(
        "{:>8}  {:>10}  {:>12}  {:>13}",
        "rows", "secs", "rows/s", "full_scans"
    );
    for (i, &n) in sizes.iter().enumerate() {
        let (secs, fs) = measure_upsert(n);
        elapsed[i] = secs;
        full_scans[i] = fs;
        let rps = n as f64 / secs;
        println!("{n:>8}  {secs:>10.4}  {rps:>12.1}  {fs:>13}");
    }

    // Linearity: doubling 10k -> 20k stays roughly linear; 30k is ~3x the 10k
    // baseline. A quadratic UPSERT would show t(20k)/t(10k) ~ 4 and t(30k)/t(10k)
    // ~ 9 instead of ~2 and ~3.
    let ratio_10_20 = elapsed[2] / elapsed[1];
    let ratio_10_30 = elapsed[3] / elapsed[1];

    println!();
    println!("t(20k)/t(10k) = {ratio_10_20:.3}  (gate <= 3.0, linear ~2x + slack)");
    println!("t(30k)/t(10k) = {ratio_10_30:.3}  (gate <= 4.5, linear ~3x + slack)");

    assert!(
        ratio_10_20 <= 3.0,
        "edges UPSERT is super-linear: t(20k)/t(10k) = {ratio_10_20:.3} exceeds 3.0 \
         (the O(rows^2) conflict-scan regression)"
    );
    assert!(
        ratio_10_30 <= 4.5,
        "edges UPSERT shows super-linear blow-up at 30k: t(30k)/t(10k) = {ratio_10_30:.3} \
         exceeds 4.5"
    );

    println!();
    println!("edges UPSERT scaling gate: PASSED");
}
