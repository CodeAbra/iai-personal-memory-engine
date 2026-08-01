//! Production-shaped delete sweep at DEFAULT (byte-driven) thresholds.
//!
//! The live edges table hit "interior page overflow while packing cells"
//! during batched orphan-edge deletes. The existing byteful harness runs
//! uniform 9-byte varint keys; the live tree carries sequential rowids whose
//! varint width crosses 1→2→3 byte boundaries inside the tree, ~100-byte
//! payloads, and deletes arrive as sorted scattered batches. This test
//! reproduces exactly that shape: no injected cell ceiling, sequential keys
//! across varint boundaries, sweep-shaped batched deletes down to empty.

use std::path::PathBuf;

use lillibrain::store::Store;

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
}

/// ~100-byte payload like an edges row (two UUID strings + type + weight).
fn edge_payload(key: i64) -> Vec<u8> {
    let mut v =
        format!("{key:032x}-src-uuid-payload|{key:032x}-dst-uuid-payload|hebbian|1.0").into_bytes();
    v.truncate(100);
    v
}

#[test]
fn sweep_shaped_batched_deletes_never_overflow_interior() {
    default_thresholds();
    let (_dir, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    let tree = store.tree(root);

    // Sequential rowids crossing the 1->2->3 byte varint boundaries (127 and
    // 16383) deep inside the tree, exactly like an autoincrement rowid table.
    // 80k rows -> ~1300 leaves -> two interior levels, so interior borrow and
    // merge run against real byte-driven thresholds without the run becoming
    // I/O-bound (per-op journal writes dominate the wall time).
    const N: i64 = 80_000;
    for key in 0..N {
        tree.insert(key, &edge_payload(key)).unwrap();
    }

    // Sweep shape: sorted batches of ~200 keys scattered across the whole
    // keyspace, repeated until the tree is empty. Batch b deletes keys
    // congruent to a rotating residue so every batch touches many leaves and
    // interior separators, like an orphan sweep deleting by dead endpoint.
    let mut rng = XorShift64(0x1a1_5eed);
    let mut alive: Vec<i64> = (0..N).collect();
    while !alive.is_empty() {
        let batch_len = 200.min(alive.len());
        let mut batch: Vec<i64> = Vec::with_capacity(batch_len);
        for _ in 0..batch_len {
            let i = (rng.next_u64() as usize) % alive.len();
            batch.push(alive.swap_remove(i));
        }
        batch.sort_unstable();
        for key in batch {
            tree.delete(key).unwrap_or_else(|e| {
                panic!("delete({key}) failed with {} alive left: {e}", alive.len())
            });
        }
    }

    assert_eq!(tree.full_scan().unwrap().len(), 0);
}

#[test]
fn interleaved_insert_delete_churn_never_overflows_interior() {
    default_thresholds();
    let (_dir, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    let tree = store.tree(root);

    // Live shape: inserts (captures) interleave with sweep deletes, so the
    // tree repeatedly grows through split and shrinks through borrow/merge on
    // the same page neighborhoods.
    let mut rng = XorShift64(0xc0ffee);
    let mut alive: Vec<i64> = Vec::new();
    let mut next_key: i64 = 0;
    for round in 0..120 {
        for _ in 0..400 {
            tree.insert(next_key, &edge_payload(next_key)).unwrap();
            alive.push(next_key);
            next_key += 1;
        }
        let deletions = if round % 3 == 2 { 500 } else { 200 };
        let mut batch: Vec<i64> = Vec::new();
        for _ in 0..deletions.min(alive.len()) {
            let i = (rng.next_u64() as usize) % alive.len();
            batch.push(alive.swap_remove(i));
        }
        batch.sort_unstable();
        for key in batch {
            tree.delete(key)
                .unwrap_or_else(|e| panic!("round {round}: delete({key}): {e}"));
        }
    }
}
