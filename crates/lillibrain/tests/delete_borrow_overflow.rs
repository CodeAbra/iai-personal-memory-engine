//! Regression: a leaf right-borrow during delete must free the receiver leaf's
//! old overflow chains before the re-spilling rewrite, exactly as the left-borrow
//! path does. A retained spilled receiver cell whose original chain is not freed
//! leaks overflow pages — a slow corpus-size creep invisible to structural
//! integrity but caught by full page conservation.
//!
//! The state is hand-built at the page level so the exact trigger is forced:
//! an underflowing receiver leaf holding a single SPILLED cell, a right sibling
//! with surplus cells, and a borrow that fits. Driving it through random inserts
//! is unreliable (the split allocator rarely lands this precise layout — the very
//! reason the fuzzer misses the leak).

use std::collections::HashSet;
use std::path::PathBuf;

use lillibrain::btree::delete::rebalance_leaf;
use lillibrain::btree::overflow::write_overflow_chain;
use lillibrain::btree::page::{
    encode_inline_cell, encode_spilled_cell, set_sibling, write_interior_node, write_leaf_from_raw,
};
use lillibrain::check::page_census;
use lillibrain::consts::OVERFLOW_THRESHOLD;
use lillibrain::freelist::{alloc_page, freelist_pages};
use lillibrain::pager::Pager;

fn temp_path() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

/// Total pages accounted across every role: reachable tree nodes + overflow +
/// freelist + reserved. Equality with db_size means no leak and no orphan.
fn accounted_pages(pager: &Pager, root: u32, reserved: &[u32]) -> HashSet<u32> {
    let census = page_census(pager, root).unwrap();
    let mut all: HashSet<u32> = reserved.iter().copied().collect();
    for s in [&census.tree_nodes, &census.overflow_pages, &census.freelist] {
        all.extend(s.iter().copied());
    }
    all
}

#[test]
fn right_borrow_with_spilled_receiver_cell_does_not_leak_overflow_pages() {
    // The count-floor underflow path (receiver below leaf_min_cells) only fires
    // under the injected small cap; rebalance_leaf reads it live.
    std::env::set_var("LILLI_LEAF_MAX_CELLS", "4"); // leaf_min_cells == 2
    if std::env::var("LILLI_FSYNC_MODE").is_err() {
        std::env::set_var("LILLI_FSYNC_MODE", "fast");
    }

    let (_d, path) = temp_path();
    let pager = Pager::open(&path).unwrap();

    // A spilled payload for the receiver's surviving cell.
    let big: Vec<u8> = vec![0xA7u8; OVERFLOW_THRESHOLD + 800];
    let recv_chain = write_overflow_chain(&pager, &big).unwrap();

    // Allocate the page numbers for a 3-leaf tree under one interior root.
    let root = alloc_page(&pager).unwrap();
    let l0 = alloc_page(&pager).unwrap();
    let l1 = alloc_page(&pager).unwrap(); // the receiver
    let l2 = alloc_page(&pager).unwrap();

    // L0: two inline cells (a stable left sibling, not the borrow source).
    let l0_cells = vec![
        encode_inline_cell(0, b"a"),
        encode_inline_cell(1, b"b"),
    ];
    let mut l0_page = write_leaf_from_raw(&l0_cells, l1).unwrap();
    set_sibling(&mut l0_page, l1);
    pager.write_page(l0, &l0_page).unwrap();

    // L1 (receiver): exactly ONE cell, and it is SPILLED — underfull (1 < 2).
    let l1_cells = vec![encode_spilled_cell(10, big.len(), recv_chain)];
    let mut l1_page = write_leaf_from_raw(&l1_cells, l2).unwrap();
    set_sibling(&mut l1_page, l2);
    pager.write_page(l1, &l1_page).unwrap();

    // L2 (right sibling): three inline cells — surplus (3 > 2) so a borrow fires.
    let l2_cells = vec![
        encode_inline_cell(20, b"u"),
        encode_inline_cell(21, b"v"),
        encode_inline_cell(22, b"w"),
    ];
    let mut l2_page = write_leaf_from_raw(&l2_cells, 0).unwrap();
    set_sibling(&mut l2_page, 0);
    pager.write_page(l2, &l2_page).unwrap();

    // Root interior over [L0, L1, L2] with separators [10, 20].
    let root_page = write_interior_node(&[10, 20], &[l0, l1, l2]).unwrap();
    pager.write_page(root, &root_page).unwrap();

    let reserved = [1u32, 2, 3];
    let db_before = pager.db_size().unwrap();
    let accounted_before = accounted_pages(&pager, root, &reserved);
    assert_eq!(
        accounted_before.len(),
        db_before as usize,
        "hand-built tree is not conservative before the borrow"
    );
    let freelist_before = freelist_pages(&pager).unwrap();

    // Rebalance L1: child index 1 under the root, empty parent stack above root.
    let path_stack = vec![(root, 1usize)];
    rebalance_leaf(&pager, path_stack, l1).unwrap();

    // The borrow re-spilled the receiver's retained cell into a fresh chain. The
    // original chain rooted at `recv_chain` must have been freed (reachable via
    // the freelist), not orphaned. Full conservation is the verdict.
    let db_after = pager.db_size().unwrap();
    let accounted_after = accounted_pages(&pager, root, &reserved);
    assert_eq!(
        accounted_after.len(),
        db_after as usize,
        "overflow-page leak: a borrow orphaned the receiver's old overflow chain \
         ({} pages accounted != db_size {})",
        accounted_after.len(),
        db_after
    );

    // The old chain head must no longer be a live tree/overflow page — it was
    // either freed onto the freelist or recycled into the fresh chain.
    let freelist_after = freelist_pages(&pager).unwrap();
    assert!(
        freelist_after.len() >= freelist_before.len() || db_after > db_before,
        "the freed receiver chain did not return to the freelist"
    );
}
