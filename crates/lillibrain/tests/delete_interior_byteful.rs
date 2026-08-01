//! Regression: a leaf merge under a byte-full interior node must not cascade
//! into an interior merge that overflows one page.
//!
//! Interior cells carry variable-length varint keys, so two interior siblings
//! that look mergeable by count can exceed a page by bytes. The delete
//! rebalance must judge interior occupancy by bytes (like the leaf path) and,
//! when a merge cannot fit, redistribute instead of erroring with
//! "interior page overflow while packing cells".
//!
//! The tree is hand-built at the page level: a root over two interior
//! siblings, each packed to byte capacity with 9-byte-varint separators, each
//! child a small leaf. Driving this through random inserts is unreliable —
//! the failing shape needs full interior pages, which the fuzzer rarely holds
//! at the moment of a delete cascade.

use std::collections::HashSet;

use lillibrain::btree::cursor::leaf_max_cells;
use lillibrain::btree::delete::rebalance_leaf;
use lillibrain::btree::page::{
    encode_inline_cell, interior_fits, read_interior_node, set_sibling, write_interior_node,
    write_leaf_from_raw,
};
use lillibrain::check::page_census;
use lillibrain::freelist::alloc_page;
use lillibrain::pager::Pager;

// Keys in this range encode as 9-byte varints, the widest interior cell.
const KEY_BASE: i64 = 0x0100_0000_0000_0000;
const KEY_STEP: i64 = 1_000;

struct Subtree {
    page_no: u32,
    first_key: i64,
    leaves: Vec<u32>,
}

/// Build an interior node packed to byte capacity: small two-cell leaves, one
/// separator per leaf after the first, keys drawn from `counter`.
fn build_full_interior(pager: &Pager, counter: &mut i64, first_leaf_cells: usize) -> Subtree {
    let mut leaves: Vec<u32> = Vec::new();
    let mut leaf_first_keys: Vec<i64> = Vec::new();
    let mut seps: Vec<i64> = Vec::new();

    loop {
        let key = *counter;
        if !leaves.is_empty() {
            let mut probe = seps.clone();
            probe.push(key);
            if !interior_fits(&probe) {
                break;
            }
            seps.push(key);
        }
        let n_cells = if leaves.is_empty() {
            first_leaf_cells
        } else {
            2
        };
        let cells: Vec<Vec<u8>> = (0..n_cells)
            .map(|j| encode_inline_cell(key + j as i64, b"v"))
            .collect();
        let page = write_leaf_from_raw(&cells, 0).unwrap();
        let page_no = alloc_page(pager).unwrap();
        pager.write_page(page_no, &page).unwrap();
        leaves.push(page_no);
        leaf_first_keys.push(key);
        *counter += KEY_STEP;
    }

    let interior_page = write_interior_node(&seps, &leaves).unwrap();
    let page_no = alloc_page(pager).unwrap();
    pager.write_page(page_no, &interior_page).unwrap();
    Subtree {
        page_no,
        first_key: leaf_first_keys[0],
        leaves,
    }
}

fn chain_leaves(pager: &Pager, all_leaves: &[u32]) {
    for (i, &leaf) in all_leaves.iter().enumerate() {
        let mut page = pager.read_page(leaf).unwrap();
        let next = all_leaves.get(i + 1).copied().unwrap_or(0);
        set_sibling(&mut page, next);
        pager.write_page(leaf, &page).unwrap();
    }
}

#[test]
fn leaf_merge_under_byteful_interior_completes_without_overflow() {
    // The production shape needs DEFAULT thresholds (no injected small cap):
    // the count-based interior minimum then exceeds any real page's capacity,
    // which is exactly the miscalibration under test.
    std::env::remove_var("LILLI_LEAF_MAX_CELLS");
    assert!(leaf_max_cells() >= 9999, "test requires default thresholds");
    if std::env::var("LILLI_FSYNC_MODE").is_err() {
        std::env::set_var("LILLI_FSYNC_MODE", "fast");
    }

    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    let pager = Pager::open(&path).unwrap();

    let root = alloc_page(&pager).unwrap();
    let mut counter = KEY_BASE;
    // First leaf of the left subtree holds ONE cell so a single-key delete
    // empties it and forces a leaf merge with its right neighbour.
    let left = build_full_interior(&pager, &mut counter, 1);
    let right = build_full_interior(&pager, &mut counter, 2);

    let mut all_leaves = left.leaves.clone();
    all_leaves.extend_from_slice(&right.leaves);
    chain_leaves(&pager, &all_leaves);

    let root_page =
        write_interior_node(&[right.first_key], &[left.page_no, right.page_no]).unwrap();
    pager.write_page(root, &root_page).unwrap();

    let census_before = page_census(&pager, root).unwrap();
    let left_keys_before = read_interior_node(&pager.read_page(left.page_no).unwrap())
        .unwrap()
        .keys
        .len();
    assert!(left_keys_before > 200, "left interior must be byte-full");

    // Empty the first leaf and rebalance it: leaf merge -> separator removal
    // from the byte-full left interior. Before the byte-driven rules this
    // cascaded into an interior merge that could not fit one page and the
    // whole delete failed with an integrity error.
    let first_leaf = left.leaves[0];
    let empty = write_leaf_from_raw(&[], left.leaves[1]).unwrap();
    pager.write_page(first_leaf, &empty).unwrap();
    let path_stack = vec![(root, 0usize), (left.page_no, 0usize)];
    rebalance_leaf(&pager, path_stack, first_leaf).unwrap();

    // The tree must stay structurally sound and page-conservative.
    let census_after = page_census(&pager, root).unwrap();
    let accounted: HashSet<u32> = census_after
        .tree_nodes
        .iter()
        .chain(census_after.overflow_pages.iter())
        .chain(census_after.freelist.iter())
        .copied()
        .collect();
    let before_accounted: HashSet<u32> = census_before
        .tree_nodes
        .iter()
        .chain(census_before.overflow_pages.iter())
        .chain(census_before.freelist.iter())
        .copied()
        .collect();
    assert_eq!(
        accounted, before_accounted,
        "page conservation: the same page universe before and after (the merged \
         leaf moves to the freelist, nothing leaks, nothing appears)"
    );

    // Both interior siblings must still be readable and jointly hold one
    // separator less than before (the merged leaf's).
    let left_node = read_interior_node(&pager.read_page(left.page_no).unwrap()).unwrap();
    let right_node = read_interior_node(&pager.read_page(right.page_no).unwrap()).unwrap();
    assert_eq!(left_node.keys.len(), left_keys_before - 1);
    assert!(!right_node.keys.is_empty());
    assert_eq!(left_node.children.len(), left_node.keys.len() + 1);
    assert_eq!(right_node.children.len(), right_node.keys.len() + 1);
}
