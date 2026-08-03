//! The per-tree count cache must survive a rejected nested transaction: taking
//! the rollback snapshot only after the pager accepts the begin ensures a
//! rejected nested `begin_write` cannot overwrite the live transaction's
//! baseline with post-mutation counts, which a later rollback would restore.

use lillibrain::store::Store;
use tempfile::tempdir;

#[test]
fn rejected_nested_begin_does_not_corrupt_the_count_cache() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();

    store.tree(root).insert(1, b"a").unwrap();
    store.tree(root).insert(2, b"b").unwrap();
    // Warm the count cache at the true value.
    assert_eq!(store.tree(root).count_cells().unwrap(), 2);

    store.begin_write().unwrap();
    store.tree(root).insert(3, b"c").unwrap();
    // The cache now reflects the in-flight insert.
    assert_eq!(store.tree(root).count_cells().unwrap(), 3);

    // A nested begin must be rejected by the pager — and must not replace the
    // transaction's rollback baseline with these post-insert counts.
    assert!(
        store.begin_write().is_err(),
        "a nested begin_write must be rejected"
    );

    store.rollback().unwrap();
    // The rollback restores the pre-transaction baseline (2), not the mutated
    // count (3) a corrupted snapshot would have preserved.
    assert_eq!(
        store.tree(root).count_cells().unwrap(),
        2,
        "rollback must restore the true pre-transaction count"
    );
}

#[test]
fn count_cells_tracks_full_scan_across_churn() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();

    let scanned = |s: &Store| s.tree(root).full_scan().unwrap().len();
    let counted = |s: &Store| s.tree(root).count_cells().unwrap() as usize;

    store.begin_write().unwrap();
    for k in 0..100i64 {
        store
            .tree(root)
            .insert(k, format!("v{k}").as_bytes())
            .unwrap();
    }
    store.commit().unwrap();
    assert_eq!(counted(&store), scanned(&store), "after inserts");

    store.begin_write().unwrap();
    for k in (0..100i64).step_by(2) {
        store.tree(root).delete(k).unwrap();
    }
    store.commit().unwrap();
    assert_eq!(counted(&store), scanned(&store), "after deletes");
    assert_eq!(counted(&store), 50);

    // An aborted transaction leaves both the count and the scan at the committed
    // state — the incremental adjustments must not survive the rollback.
    let before = counted(&store);
    store.begin_write().unwrap();
    for k in 100..150i64 {
        store.tree(root).insert(k, b"x").unwrap();
    }
    store.rollback().unwrap();
    assert_eq!(counted(&store), before, "rollback must restore the count");
    assert_eq!(counted(&store), scanned(&store), "after rollback");
}
