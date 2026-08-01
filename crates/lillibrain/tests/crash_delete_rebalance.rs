//! Crash recovery through the byte-driven delete-rebalance paths.
//!
//! The transaction envelope's kill-point tests predate the byte-driven
//! interior rebalance (borrow / merge / redistribute / write-or-split), whose
//! delete cascades rewrite several interior pages in one transaction. These
//! cases crash exactly around such transactions at DEFAULT thresholds and
//! require reopen-time recovery to restore a consistent, fully usable tree.
//!
//! Own test binary: the cell-ceiling env var is process-wide and the
//! dense-cap harness sets it.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::path::{Path, PathBuf};

use lillibrain::check::page_census;
use lillibrain::consts::PAGE_SIZE;
use lillibrain::journal::{journal_path_for, Journal};
use lillibrain::pos_io::PosIo;
use lillibrain::Store;

const WIDE_BASE: i64 = 0x0100_0000_0000_0000;
const WIDE_STEP: i64 = 1_000;
const N_KEYS: i64 = 400;

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

fn value_for(k: i64) -> Vec<u8> {
    vec![(k % 251) as u8; 600]
}

/// Seed a committed, rebalance-prone tree: dense values at default thresholds
/// give a multi-level shape whose deletes cascade through interior pages.
fn seed(store: &Store, root: u32, oracle: &mut BTreeMap<i64, Vec<u8>>) {
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        for i in 0..N_KEYS {
            let k = WIDE_BASE + i * WIDE_STEP;
            let v = value_for(i);
            t.insert(k, &v).unwrap();
            oracle.insert(k, v);
        }
    }
    store.commit().unwrap();
}

fn assert_consistent(store: &Store, root: u32, oracle: &BTreeMap<i64, Vec<u8>>) {
    let scan = store.tree(root).full_scan().unwrap();
    let want: Vec<(i64, Vec<u8>)> = oracle.iter().map(|(k, v)| (*k, v.clone())).collect();
    assert_eq!(scan.len(), want.len(), "key count after recovery");
    assert_eq!(scan, want, "tree content after recovery");
    assert!(
        store.check_integrity(root).unwrap().is_empty(),
        "integrity after recovery"
    );

    let census = page_census(store.pager(), root).unwrap();
    let db_size = store.db_size().unwrap();
    let mut all: std::collections::HashSet<u32> = [1u32].into_iter().collect();
    for p in [2u32, 3u32] {
        if p != root {
            all.insert(p);
        }
    }
    for set in [&census.tree_nodes, &census.overflow_pages, &census.freelist] {
        for &p in set {
            assert!(
                all.insert(p),
                "page {p} in more than one role after recovery"
            );
        }
    }
    assert_eq!(
        all.len(),
        db_size as usize,
        "page conservation after recovery"
    );
}

fn churn(store: &Store, root: u32, oracle: &mut BTreeMap<i64, Vec<u8>>) {
    // Post-recovery usability: further deletes drive fresh rebalances over the
    // recovered pages; any latent inconsistency surfaces here.
    let keys: Vec<i64> = oracle.keys().copied().take(120).collect();
    for chunk in keys.chunks(30) {
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for k in chunk {
                t.delete(*k).unwrap();
                oracle.remove(k);
            }
        }
        store.commit().unwrap();
    }
    assert_consistent(store, root, oracle);
}

#[test]
fn crash_before_commit_mid_rebalance_delete_rolls_back() {
    default_thresholds();
    let (_d, path) = temp_db();
    let mut oracle: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = store.create_tree().unwrap();
        seed(&store, root, &mut oracle);

        // A rebalance-heavy delete band inside an open transaction, then a
        // simulated crash (store dropped, commit never reached).
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for i in 100..250 {
                t.delete(WIDE_BASE + i * WIDE_STEP).unwrap();
            }
        }
        drop(store);
    }

    // Reopen: the whole uncommitted cascade must be gone.
    let store = Store::open(&path).unwrap();
    assert_consistent(&store, root, &oracle);
    assert!(!journal_path_for(&path).exists());
    churn(&store, root, &mut oracle);
}

#[test]
fn crash_between_barriers_over_rebalanced_pages_restores_before_images() {
    default_thresholds();
    let (_d, path) = temp_db();
    let mut oracle: BTreeMap<i64, Vec<u8>> = BTreeMap::new();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = store.create_tree().unwrap();
        seed(&store, root, &mut oracle);

        // Commit a real rebalance so the tree pages on disk are the product of
        // the byte-driven delete paths.
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for i in 150..300 {
                let k = WIDE_BASE + i * WIDE_STEP;
                t.delete(k).unwrap();
                oracle.remove(&k);
            }
        }
        store.commit().unwrap();
    }

    // Hand-build the killed-mid-commit state: committed journal holding the
    // before-images of several live tree pages, with those pages scribbled on
    // disk (a crash after partial page writes, before the journal cleared).
    let victims: Vec<u32> = {
        let store = Store::open(&path).unwrap();
        let census = page_census(store.pager(), root).unwrap();
        let mut v: Vec<u32> = census.tree_nodes.iter().copied().collect();
        v.sort_unstable();
        v.into_iter().take(6).collect()
    };
    scribble_with_journal(&path, &victims);

    let store = Store::open(&path).unwrap();
    assert_consistent(&store, root, &oracle);
    assert!(!journal_path_for(&path).exists());
    churn(&store, root, &mut oracle);
}

fn scribble_with_journal(path: &Path, pages: &[u32]) {
    let mut j = Journal::create(path).unwrap();
    let f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .unwrap();
    for &p in pages {
        let off = (p as u64 - 1) * PAGE_SIZE as u64;
        let mut before = vec![0u8; PAGE_SIZE];
        f.read_exact_at(&mut before, off).unwrap();
        j.record_before_image(p, &before);
    }
    j.flush_pending().unwrap();
    j.sync().unwrap();
    j.write_page_count().unwrap();
    j.sync().unwrap();
    for &p in pages {
        let off = (p as u64 - 1) * PAGE_SIZE as u64;
        f.write_all_at(&vec![0xEE; PAGE_SIZE], off).unwrap();
    }
    f.sync_all().unwrap();
}
