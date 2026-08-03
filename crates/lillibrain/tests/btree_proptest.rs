//! Randomized property/fuzz harness for the B-tree: a long stream of
//! insert/delete/replace operations applied to both a `Store`/`Tree` and a
//! `BTreeMap` oracle. After each batch three independent assertions hold —
//! oracle equality (scan + point reads + max_key), structural integrity, and
//! page conservation (no leak, no double-free).
//!
//! Run densely so splits/merges/borrows fire on nearly every batch:
//!   `LILLI_LEAF_MAX_CELLS=4 LILLI_FSYNC_MODE=fast cargo test -p lillibrain --test btree_proptest`
//!
//! Two equivalent gates live here: a `proptest!` block that shrinks any
//! counterexample to a minimal op sequence, and a deterministic seeded-RNG loop
//! that always runs ≥5000 ops as the stable CI reproduction.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lillibrain::check::page_census;
use lillibrain::Store;

use proptest::prelude::*;

/// One operation against both the tree and the oracle.
#[derive(Debug, Clone)]
enum Op {
    Insert(i64, Vec<u8>),
    Delete(i64),
    Replace(i64, Vec<u8>),
}

fn temp_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

/// Force the dense split/merge cap for this test process.
fn force_dense_cap() {
    std::env::set_var("LILLI_LEAF_MAX_CELLS", "4");
    // Keep the per-commit sync cheap so a multi-thousand-op run is seconds, not
    // hours, of full drive-cache flushes.
    if std::env::var("LILLI_FSYNC_MODE").is_err() {
        std::env::set_var("LILLI_FSYNC_MODE", "fast");
    }
}

/// Apply `op` to the tree (inside its own transaction) and the oracle in lockstep.
fn apply(store: &Store, root: u32, oracle: &mut BTreeMap<i64, Vec<u8>>, op: &Op) {
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        match op {
            Op::Insert(k, v) | Op::Replace(k, v) => {
                t.insert(*k, v).unwrap();
                oracle.insert(*k, v.clone());
            }
            Op::Delete(k) => {
                t.delete(*k).unwrap();
                oracle.remove(k);
            }
        }
    }
    store.commit().unwrap();
}

/// The three per-batch assertions: oracle equality, structural integrity, and
/// page conservation. Returns `Err(reason)` so the proptest block can shrink.
fn assert_invariants(
    store: &Store,
    root: u32,
    oracle: &BTreeMap<i64, Vec<u8>>,
) -> std::result::Result<(), String> {
    // (a) ORACLE EQUALITY — full scan ascending + byte-identical values.
    let scan = store.tree(root).full_scan().map_err(|e| e.to_string())?;
    let want: Vec<(i64, Vec<u8>)> = oracle.iter().map(|(k, v)| (*k, v.clone())).collect();
    if scan != want {
        return Err(format!(
            "oracle mismatch: tree has {} keys, oracle has {} keys",
            scan.len(),
            want.len()
        ));
    }
    // Point reads for a sample of live keys + max_key parity.
    for (k, v) in oracle.iter().take(16) {
        let got = store.tree(root).get(*k).map_err(|e| e.to_string())?;
        if got.as_deref() != Some(v.as_slice()) {
            return Err(format!("get({k}) parity mismatch"));
        }
    }
    let want_max = oracle.keys().last().copied();
    let got_max = store.tree(root).max_key().map_err(|e| e.to_string())?;
    if got_max != want_max {
        return Err(format!(
            "max_key mismatch: got {got_max:?}, want {want_max:?}"
        ));
    }

    // (b) STRUCTURAL — check_integrity clean.
    let errs = store.check_integrity(root).map_err(|e| e.to_string())?;
    if !errs.is_empty() {
        return Err(format!("integrity errors: {errs:?}"));
    }

    // (c) PAGE CONSERVATION — every page accounted exactly once, no leak/double-free.
    let db_size = store.db_size().map_err(|e| e.to_string())?;
    let free_count = store.free_page_count().map_err(|e| e.to_string())?;
    let census = page_census(store.pager(), root).map_err(|e| e.to_string())?;

    // Reserved structural pages: header (1) plus the root-slot/meta reservations
    // (2, 3) when they are not the active tree root.
    let mut reserved: std::collections::HashSet<u32> = [1u32].into_iter().collect();
    for p in [2u32, 3u32] {
        if p != root {
            reserved.insert(p);
        }
    }

    // The five role sets must be pairwise disjoint and union to exactly 1..=db_size.
    let mut all: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut double = None;
    for set in [
        &reserved,
        &census.tree_nodes,
        &census.overflow_pages,
        &census.freelist,
    ] {
        for &p in set {
            if !all.insert(p) {
                double = Some(p);
            }
        }
    }
    if let Some(p) = double {
        return Err(format!(
            "page {p} appears in more than one role (double-free/use)"
        ));
    }
    // freelist header count must match the actual set (no phantom frees).
    if free_count as usize != census.freelist.len() {
        return Err(format!(
            "freelist count {} != actual freelist set {} (leak or double-free)",
            free_count,
            census.freelist.len()
        ));
    }
    // Conservation: reachable + overflow + free + reserved == db_size, no leak.
    if all.len() != db_size as usize {
        return Err(format!(
            "page conservation: accounted {} pages != db_size {} (leak or orphan)",
            all.len(),
            db_size
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Deterministic seeded loop — the always-runs gate (>=5000 ops).
// ---------------------------------------------------------------------------

/// A tiny xorshift64 RNG: deterministic, dependency-free, good enough to drive a
/// fuzz op stream with a fixed seed for stable reproduction.
struct XorShift64(u64);

impl XorShift64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// Build a value of mixed size: usually small, occasionally overflow-sized so the
/// spill chain is exercised.
fn gen_value(rng: &mut XorShift64) -> Vec<u8> {
    let roll = rng.below(100);
    let len = if roll < 5 {
        // Overflow-sized payload (> the single-page spill threshold).
        8200 + (rng.below(2000) as usize)
    } else {
        (rng.below(48) as usize) + 1
    };
    let fill = (rng.next_u64() & 0xFF) as u8;
    vec![fill; len]
}

#[test]
fn btree_proptest_seeded_loop_survives_5000_ops() {
    force_dense_cap();
    let (_d, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    let mut oracle: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
    let mut rng = XorShift64(0x1234_5678_9ABC_DEF0);

    const N_OPS: usize = 5200;
    const KEY_RANGE: u64 = 256;

    for i in 0..N_OPS {
        let key = rng.below(KEY_RANGE) as i64;
        let op = match rng.below(3) {
            0 => Op::Insert(key, gen_value(&mut rng)),
            1 => Op::Delete(key),
            _ => Op::Replace(key, gen_value(&mut rng)),
        };
        apply(&store, root, &mut oracle, &op);

        // Assert every 100 ops to keep the run fast while still catching the
        // first divergence early.
        if i % 100 == 0 {
            if let Err(reason) = assert_invariants(&store, root, &oracle) {
                panic!("seeded loop diverged at op {i} ({op:?}): {reason}");
            }
        }
    }
    // Final assertion after the full run.
    if let Err(reason) = assert_invariants(&store, root, &oracle) {
        panic!("seeded loop final assertion failed: {reason}");
    }
}

// ---------------------------------------------------------------------------
// proptest block — shrinks any counterexample to a minimal op sequence.
// ---------------------------------------------------------------------------

/// Strategy for one op over a small key range (so deletes/replaces hit live
/// keys) with mixed small / occasionally overflow-sized values.
fn op_strategy() -> impl Strategy<Value = Op> {
    let key = 0i64..256;
    // Small values dominate; a rare large value exercises the overflow chain.
    let value = prop_oneof![
        9 => proptest::collection::vec(any::<u8>(), 1..48),
        1 => proptest::collection::vec(any::<u8>(), 8200..9000),
    ];
    prop_oneof![
        (key.clone(), value.clone()).prop_map(|(k, v)| Op::Insert(k, v)),
        key.clone().prop_map(Op::Delete),
        (key, value).prop_map(|(k, v)| Op::Replace(k, v)),
    ]
}

proptest! {
    // 60 cases x up to ~120 ops each multiplies well past 5000 applied ops.
    #![proptest_config(ProptestConfig {
        cases: 60,
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    #[test]
    fn btree_proptest_oracle_equivalence(ops in proptest::collection::vec(op_strategy(), 1..120)) {
        force_dense_cap();
        let (_d, path) = temp_db();
        let store = Store::open(&path).unwrap();
        let root = store.create_tree().unwrap();
        let mut oracle: BTreeMap<i64, Vec<u8>> = BTreeMap::new();

        for op in &ops {
            apply(&store, root, &mut oracle, op);
            // Assert after every op so shrinking minimizes to the exact failing prefix.
            assert_invariants(&store, root, &oracle)
                .map_err(|reason| TestCaseError::fail(format!("after {op:?}: {reason}")))?;
        }
    }
}
