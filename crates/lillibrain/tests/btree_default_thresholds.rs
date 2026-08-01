//! Property/fuzz harness at DEFAULT rebalance thresholds.
//!
//! The dense-cap harness (`btree_proptest.rs`) injects a small cell ceiling,
//! which switches the delete rebalance onto its count-floor rules — so the
//! byte-driven rules that govern production trees were never under test. This
//! binary runs with the ceiling UNSET: wide varint keys and large payloads
//! build real-shaped deep trees, and bulk deletes drive the byte-driven
//! borrow / merge / redistribute paths that only exist at default thresholds.
//!
//! Own test binary on purpose: the cell-ceiling env var is process-wide, and
//! integration test FILES are separate processes.

use std::collections::BTreeMap;
use std::path::PathBuf;

use lillibrain::check::page_census;
use lillibrain::Store;

use proptest::prelude::*;

// Keys up here encode as 9-byte varints — the widest interior cell, the shape
// of a production id-indexed tree.
const WIDE_BASE: i64 = 0x0100_0000_0000_0000;
const WIDE_STEP: i64 = 1_000;

fn default_thresholds() {
    std::env::remove_var("LILLI_LEAF_MAX_CELLS");
    if std::env::var("LILLI_FSYNC_MODE").is_err() {
        std::env::set_var("LILLI_FSYNC_MODE", "fast");
    }
}

fn temp_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

#[derive(Debug, Clone)]
enum Op {
    Insert(i64, Vec<u8>),
    Delete(i64),
    Replace(i64, Vec<u8>),
}

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

fn assert_invariants(
    store: &Store,
    root: u32,
    oracle: &BTreeMap<i64, Vec<u8>>,
) -> std::result::Result<(), String> {
    let scan = store.tree(root).full_scan().map_err(|e| e.to_string())?;
    let want: Vec<(i64, Vec<u8>)> = oracle.iter().map(|(k, v)| (*k, v.clone())).collect();
    if scan != want {
        return Err(format!(
            "oracle mismatch: tree has {} keys, oracle has {} keys",
            scan.len(),
            want.len()
        ));
    }
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

    let errs = store.check_integrity(root).map_err(|e| e.to_string())?;
    if !errs.is_empty() {
        return Err(format!("integrity errors: {errs:?}"));
    }

    let db_size = store.db_size().map_err(|e| e.to_string())?;
    let free_count = store.free_page_count().map_err(|e| e.to_string())?;
    let census = page_census(store.pager(), root).map_err(|e| e.to_string())?;
    let mut reserved: std::collections::HashSet<u32> = [1u32].into_iter().collect();
    for p in [2u32, 3u32] {
        if p != root {
            reserved.insert(p);
        }
    }
    let mut all: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for set in [
        &reserved,
        &census.tree_nodes,
        &census.overflow_pages,
        &census.freelist,
    ] {
        for &p in set {
            if !all.insert(p) {
                return Err(format!("page {p} appears in more than one role"));
            }
        }
    }
    if free_count as usize != census.freelist.len() {
        return Err(format!(
            "freelist count {} != actual freelist set {}",
            free_count,
            census.freelist.len()
        ));
    }
    if all.len() != db_size as usize {
        return Err(format!(
            "page conservation: accounted {} != db_size {}",
            all.len(),
            db_size
        ));
    }
    Ok(())
}

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

/// 300–900 byte values: a handful of cells per leaf, so a few thousand keys
/// build a depth-3 tree whose interior pages are byte-full — the production
/// shape. A rare overflow-sized payload exercises spill chains among the
/// rebalances.
fn dense_value(rng: &mut XorShift64) -> Vec<u8> {
    let roll = rng.below(100);
    let len = if roll < 3 {
        8200 + (rng.below(1500) as usize)
    } else {
        300 + (rng.below(600) as usize)
    };
    let fill = (rng.next_u64() & 0xFF) as u8;
    vec![fill; len]
}

#[test]
fn default_thresholds_deep_tree_survives_bulk_delete_and_churn() {
    default_thresholds();
    assert!(
        lillibrain::btree::cursor::leaf_max_cells() >= 9999,
        "this binary must run at default thresholds"
    );

    let (_d, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    let mut oracle: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
    let mut rng = XorShift64(0x00DD_BA11_5EED_F00D);

    // Stage 1 — build the production shape: wide keys, dense values, depth 3.
    const N_KEYS: i64 = 2_600;
    for i in 0..N_KEYS {
        let key = WIDE_BASE + i * WIDE_STEP;
        apply(
            &store,
            root,
            &mut oracle,
            &Op::Insert(key, dense_value(&mut rng)),
        );
    }
    assert_invariants(&store, root, &oracle).unwrap_or_else(|r| panic!("after build: {r}"));

    // Stage 2 — bulk-delete a contiguous 60% band: cascaded leaf merges drive
    // interior underflow handling on byte-full interior pages (the exact path
    // that used to die with "interior page overflow while packing cells").
    let band_start = N_KEYS / 5;
    let band_end = band_start + (N_KEYS * 3 / 5);
    for i in band_start..band_end {
        let key = WIDE_BASE + i * WIDE_STEP;
        apply(&store, root, &mut oracle, &Op::Delete(key));
        if i % 250 == 0 {
            assert_invariants(&store, root, &oracle)
                .unwrap_or_else(|r| panic!("bulk delete at {i}: {r}"));
        }
    }
    assert_invariants(&store, root, &oracle).unwrap_or_else(|r| panic!("after band: {r}"));

    // Stage 3 — churn with MIXED varint widths: narrow keys interleave 1–2
    // byte cells into interiors that also hold 9-byte cells, making byte
    // footprints lumpy across siblings (the redistribute band lives here).
    const CHURN: usize = 4_000;
    for i in 0..CHURN {
        let key = if rng.below(2) == 0 {
            WIDE_BASE + (rng.below(N_KEYS as u64) as i64) * WIDE_STEP
        } else {
            rng.below(512) as i64
        };
        let op = match rng.below(3) {
            0 => Op::Insert(key, dense_value(&mut rng)),
            1 => Op::Delete(key),
            _ => Op::Replace(key, dense_value(&mut rng)),
        };
        apply(&store, root, &mut oracle, &op);
        if i % 200 == 0 {
            assert_invariants(&store, root, &oracle)
                .unwrap_or_else(|r| panic!("churn at {i} ({op:?}): {r}"));
        }
    }
    assert_invariants(&store, root, &oracle).unwrap_or_else(|r| panic!("after churn: {r}"));

    // Stage 4 — drain to empty: full cascade down to root shrink.
    let all_keys: Vec<i64> = oracle.keys().copied().collect();
    for (i, key) in all_keys.iter().enumerate() {
        apply(&store, root, &mut oracle, &Op::Delete(*key));
        if i % 400 == 0 {
            assert_invariants(&store, root, &oracle)
                .unwrap_or_else(|r| panic!("drain at {i}: {r}"));
        }
    }
    assert!(oracle.is_empty());
    assert_invariants(&store, root, &oracle).unwrap_or_else(|r| panic!("after drain: {r}"));
}

/// Shrinkable variant: mixed-width keys and dense values at default
/// thresholds; any divergence minimizes to the exact failing op prefix.
fn op_strategy() -> impl Strategy<Value = Op> {
    let key = prop_oneof![(0i64..64).prop_map(|k| WIDE_BASE + k * WIDE_STEP), 0i64..64,];
    let value = prop_oneof![
        9 => proptest::collection::vec(any::<u8>(), 300..900),
        1 => proptest::collection::vec(any::<u8>(), 8200..9000),
    ];
    prop_oneof![
        (key.clone(), value.clone()).prop_map(|(k, v)| Op::Insert(k, v)),
        key.clone().prop_map(Op::Delete),
        (key, value).prop_map(|(k, v)| Op::Replace(k, v)),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 40,
        max_shrink_iters: 4096,
        .. ProptestConfig::default()
    })]

    #[test]
    fn default_thresholds_oracle_equivalence(ops in proptest::collection::vec(op_strategy(), 1..120)) {
        default_thresholds();
        let (_d, path) = temp_db();
        let store = Store::open(&path).unwrap();
        let root = store.create_tree().unwrap();
        let mut oracle: BTreeMap<i64, Vec<u8>> = BTreeMap::new();

        for op in &ops {
            apply(&store, root, &mut oracle, op);
            assert_invariants(&store, root, &oracle)
                .map_err(|reason| TestCaseError::fail(format!("after {op:?}: {reason}")))?;
        }
    }
}
