//! Integration checks for the B-tree page format, overflow chains, cursor,
//! split, delete, and the Store/Tree public API.

use lillibrain::btree::overflow::{
    free_overflow_chain, read_overflow_chain, write_overflow_chain, OVERFLOW_USABLE,
};
use lillibrain::btree::page::{
    empty_leaf, encode_inline_cell, get_sibling, read_interior_node, read_leaf_cell_raw,
    read_leaf_header, read_leaf_keys, set_sibling, write_interior_node, write_leaf_from_raw,
    LEAF_PTR_ARRAY_START, USABLE_END,
};
use lillibrain::consts::{OVERFLOW_THRESHOLD, PAGE_SIZE};
use lillibrain::error::StoreError;
use lillibrain::pager::Pager;
use lillibrain::store::Store;

fn temp_path() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

// ---------------------------------------------------------------------------
// Page format
// ---------------------------------------------------------------------------

#[test]
fn page_leaf_header_and_sibling_roundtrip() {
    let mut page = empty_leaf();
    set_sibling(&mut page, 42);
    let hdr = read_leaf_header(&page);
    assert_eq!(hdr.num_cells, 0);
    assert_eq!(get_sibling(&page), 42);
}

#[test]
fn page_leaf_cells_pack_and_decode_in_key_order() {
    let cells: Vec<(i64, Vec<u8>)> = vec![
        (1, b"one".to_vec()),
        (5, b"five".to_vec()),
        (9, b"nine".to_vec()),
    ];
    let raw: Vec<Vec<u8>> = cells
        .iter()
        .map(|(k, v)| encode_inline_cell(*k, v))
        .collect();
    let page = write_leaf_from_raw(&raw, 7).unwrap();

    let hdr = read_leaf_header(&page);
    assert_eq!(hdr.num_cells, 3);
    assert_eq!(get_sibling(&page), 7);
    assert_eq!(read_leaf_keys(&page, OVERFLOW_THRESHOLD).unwrap(), vec![1, 5, 9]);

    for (i, (k, v)) in cells.iter().enumerate() {
        let raw_cell = read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD).unwrap();
        assert_eq!(raw_cell.key, *k);
        assert!(!raw_cell.spilled);
        let payload = &page[raw_cell.payload_start..raw_cell.payload_start + raw_cell.payload_len];
        assert_eq!(payload, v.as_slice());
    }
}

#[test]
fn page_interior_node_roundtrip_with_rightmost_child() {
    let keys = vec![10i64, 20, 30];
    let children = vec![2u32, 3, 4, 5];
    let page = write_interior_node(&keys, &children).unwrap();
    let node = read_interior_node(&page).unwrap();
    assert_eq!(node.keys, keys);
    assert_eq!(node.children, children);
}

// ---------------------------------------------------------------------------
// Overflow chains
// ---------------------------------------------------------------------------

#[test]
fn overflow_chain_roundtrips_large_value() {
    let (_d, path) = temp_path();
    let p = Pager::open(&path).unwrap();

    // A payload spanning several overflow pages.
    let payload: Vec<u8> = (0..(OVERFLOW_USABLE * 3 + 17))
        .map(|i| (i % 251) as u8)
        .collect();
    assert!(payload.len() > OVERFLOW_THRESHOLD);

    let first = write_overflow_chain(&p, &payload).unwrap();
    assert!(first >= 1);
    let recovered = read_overflow_chain(&p, first, payload.len()).unwrap();
    assert_eq!(recovered, payload);
}

#[test]
fn overflow_chain_uses_freelist_and_frees_back() {
    let (_d, path) = temp_path();
    let p = Pager::open(&path).unwrap();

    let payload: Vec<u8> = vec![0xAB; OVERFLOW_USABLE * 2 + 5];
    let first = write_overflow_chain(&p, &payload).unwrap();
    let grown = p.db_size().unwrap();
    free_overflow_chain(&p, first).unwrap();
    // After freeing, the freed pages are reused (file does not grow again).
    let again = write_overflow_chain(&p, &payload).unwrap();
    assert!(first >= 1 && again >= 1);
    assert_eq!(p.db_size().unwrap(), grown, "freed overflow pages must be reused");
    let recovered = read_overflow_chain(&p, again, payload.len()).unwrap();
    assert_eq!(recovered, payload);
}

#[test]
fn overflow_single_page_exact_boundary() {
    let (_d, path) = temp_path();
    let p = Pager::open(&path).unwrap();
    let payload: Vec<u8> = vec![1u8; OVERFLOW_USABLE];
    let first = write_overflow_chain(&p, &payload).unwrap();
    let recovered = read_overflow_chain(&p, first, payload.len()).unwrap();
    assert_eq!(recovered, payload);
}

// ---------------------------------------------------------------------------
// Cursor / Store: insert, get, scan, max_key
// ---------------------------------------------------------------------------

fn open_tree() -> (tempfile::TempDir, Store, u32) {
    let (dir, path) = temp_path();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    (dir, store, root)
}

#[test]
fn cursor_insert_get_roundtrip() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    t.insert(1, b"one").unwrap();
    t.insert(2, b"two").unwrap();
    t.insert(3, b"three").unwrap();
    assert_eq!(t.get(1).unwrap().as_deref(), Some(b"one".as_ref()));
    assert_eq!(t.get(2).unwrap().as_deref(), Some(b"two".as_ref()));
    assert_eq!(t.get(3).unwrap().as_deref(), Some(b"three".as_ref()));
    assert_eq!(t.get(99).unwrap(), None);
}

#[test]
fn cursor_insert_replaces_in_place() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    t.insert(5, b"first").unwrap();
    t.insert(5, b"second").unwrap();
    assert_eq!(t.get(5).unwrap().as_deref(), Some(b"second".as_ref()));
    assert_eq!(t.full_scan().unwrap().len(), 1);
}

#[test]
fn cursor_random_order_insert_reads_back_sorted() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);

    // A scripted pseudo-random permutation of 1..=N (deterministic LCG).
    let n: i64 = 400;
    let mut order: Vec<i64> = (1..=n).collect();
    let mut state: u64 = 0x1234_5678_9abc_def0;
    for i in (1..order.len()).rev() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        order.swap(i, j);
    }

    for &k in &order {
        let v = format!("val-{k}").into_bytes();
        t.insert(k, &v).unwrap();
    }

    let scanned = t.full_scan().unwrap();
    let keys: Vec<i64> = scanned.iter().map(|(k, _)| *k).collect();
    let want: Vec<i64> = (1..=n).collect();
    assert_eq!(keys, want, "full_scan must return keys in ascending order");
    for (k, v) in scanned {
        assert_eq!(v, format!("val-{k}").into_bytes());
    }
}

#[test]
fn cursor_max_key_is_correct_and_tree_height_cost() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);

    // Insert enough small cells (under the default byte cap a single leaf would
    // hold them all, so force a tall tree via the injected small ceiling in the
    // dedicated split test). Here, fill by bytes with sizeable values.
    let n: i64 = 600;
    for k in 1..=n {
        let v = vec![0xCDu8; 64];
        t.insert(k, &v).unwrap();
    }
    assert_eq!(t.max_key().unwrap(), Some(n));

    // max_key must read only (tree height) pages, far below the row count.
    store.pager().reset_read_count();
    let mk = t.max_key().unwrap();
    let reads = store.pager().read_count();
    assert_eq!(mk, Some(n));
    assert!(
        reads <= 8,
        "max_key read {reads} pages; expected O(tree height), not O(rows={n})"
    );
}

#[test]
fn cursor_insert_path_cost_is_independent_of_row_count() {
    // The insert path must descend in tree height, never scan the whole tree.
    // Compare the page reads of inserting one more key into a small tree vs a
    // large tree: the cost grows with height (logarithmic), not with row count.
    let (_d, store, root) = open_tree();
    let t = store.tree(root);

    for k in 1..=50 {
        t.insert(k, &vec![0u8; 64]).unwrap();
    }
    store.pager().reset_read_count();
    t.insert(51, &vec![0u8; 64]).unwrap();
    let small_reads = store.pager().read_count();

    for k in 52..=4000 {
        t.insert(k, &vec![0u8; 64]).unwrap();
    }
    store.pager().reset_read_count();
    t.insert(4001, &vec![0u8; 64]).unwrap();
    let large_reads = store.pager().read_count();

    // A whole-tree-scan insert would make large_reads grow ~linearly in the row
    // count. O(height) keeps it within a few pages of the small tree.
    assert!(
        large_reads <= small_reads + 10,
        "insert read {large_reads} pages on a 4000-key tree vs {small_reads} on a 50-key tree; \
         insert path is not O(tree height)"
    );
}

#[test]
fn cursor_range_scan_returns_inclusive_window() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    for k in 1..=100 {
        t.insert(k, format!("v{k}").as_bytes()).unwrap();
    }
    let got = t.range_scan(20, 30).unwrap();
    let keys: Vec<i64> = got.iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, (20..=30).collect::<Vec<_>>());
}

#[test]
fn cursor_persists_across_reopen() {
    let (_d, path) = temp_path();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = store.create_tree().unwrap();
        let t = store.tree(root);
        for k in 1..=300 {
            t.insert(k, format!("p{k}").as_bytes()).unwrap();
        }
    }
    let store = Store::open(&path).unwrap();
    let t = store.tree(root);
    assert_eq!(t.max_key().unwrap(), Some(300));
    assert_eq!(t.get(150).unwrap().as_deref(), Some(b"p150".as_ref()));
    assert_eq!(t.full_scan().unwrap().len(), 300);
}

#[test]
fn cursor_large_value_spills_and_roundtrips_through_tree() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    let big: Vec<u8> = (0..(OVERFLOW_THRESHOLD + 5000)).map(|i| (i % 251) as u8).collect();
    t.insert(7, &big).unwrap();
    t.insert(8, b"small").unwrap();
    assert_eq!(t.get(7).unwrap().unwrap(), big);
    assert_eq!(t.get(8).unwrap().as_deref(), Some(b"small".as_ref()));
    // Replace the spilled value with a small one; the old chain is freed.
    t.insert(7, b"now small").unwrap();
    assert_eq!(t.get(7).unwrap().as_deref(), Some(b"now small".as_ref()));
}

// ---------------------------------------------------------------------------
// Split (forced via the small cell ceiling) — see the LILLI_LEAF_MAX_CELLS gate
// ---------------------------------------------------------------------------

#[test]
fn split_keeps_key_order_and_fixed_root() {
    // This test runs under LILLI_LEAF_MAX_CELLS=4 (set by the verify command).
    if leaf_cap() >= 9999 {
        // Without the small cap, force splits by byte capacity instead.
        return split_by_bytes();
    }
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    for k in 1..=200 {
        t.insert(k, format!("k{k}").as_bytes()).unwrap();
    }
    // The root page number is fixed across all the splits.
    assert_eq!(t.root_page_no(), root);
    let keys: Vec<i64> = t.full_scan().unwrap().iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, (1..=200).collect::<Vec<_>>());
    assert_eq!(t.max_key().unwrap(), Some(200));
}

fn split_by_bytes() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    // Big values fill leaves by bytes, forcing copy-up splits without a small cap.
    for k in 1..=400 {
        let v = vec![(k % 256) as u8; 200];
        t.insert(k, &v).unwrap();
    }
    assert_eq!(t.root_page_no(), root, "root page number must be fixed for life");
    let keys: Vec<i64> = t.full_scan().unwrap().iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, (1..=400).collect::<Vec<_>>());
    assert_eq!(t.max_key().unwrap(), Some(400));
}

fn leaf_cap() -> usize {
    std::env::var("LILLI_LEAF_MAX_CELLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999)
}

// ---------------------------------------------------------------------------
// Delete (borrow / merge / root shrink)
// ---------------------------------------------------------------------------

#[test]
fn delete_is_idempotent_and_removes() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    for k in 1..=10 {
        t.insert(k, format!("d{k}").as_bytes()).unwrap();
    }
    t.delete(5).unwrap();
    assert_eq!(t.get(5).unwrap(), None);
    // Deleting an absent key is a no-op.
    t.delete(5).unwrap();
    t.delete(9999).unwrap();
    let keys: Vec<i64> = t.full_scan().unwrap().iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, vec![1, 2, 3, 4, 6, 7, 8, 9, 10]);
}

#[test]
fn delete_forces_rebalance_and_keeps_scan_complete() {
    // With LILLI_LEAF_MAX_CELLS=4 this drives borrow/merge/root-shrink; without
    // the cap big values drive byte-driven underflow.
    let (_d, store, root) = open_tree();
    let t = store.tree(root);

    let big_value = leaf_cap() >= 9999;
    let n: i64 = 300;
    for k in 1..=n {
        let v = if big_value {
            vec![(k % 256) as u8; 200]
        } else {
            format!("v{k}").into_bytes()
        };
        t.insert(k, &v).unwrap();
    }

    // Delete every other key to force underflow rebalances across the tree.
    let mut deleted = std::collections::HashSet::new();
    for k in (1..=n).step_by(2) {
        t.delete(k).unwrap();
        deleted.insert(k);
    }

    let scanned = t.full_scan().unwrap();
    let keys: Vec<i64> = scanned.iter().map(|(k, _)| *k).collect();
    // Ascending and complete: exactly the non-deleted keys remain.
    let want: Vec<i64> = (1..=n).filter(|k| !deleted.contains(k)).collect();
    assert_eq!(keys, want, "scan after deletes is incomplete or out of order");

    // Every survivor still resolves by get.
    for &k in &want {
        assert!(t.get(k).unwrap().is_some(), "lost key {k} after rebalance");
    }
    // Root page number unchanged even through root-shrink.
    assert_eq!(t.root_page_no(), root);
}

#[test]
fn delete_all_then_reinsert_collapses_and_regrows() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    for k in 1..=150 {
        t.insert(k, &vec![(k % 256) as u8; 150]).unwrap();
    }
    for k in 1..=150 {
        t.delete(k).unwrap();
    }
    assert_eq!(t.full_scan().unwrap().len(), 0);
    assert_eq!(t.max_key().unwrap(), None);
    assert_eq!(t.root_page_no(), root, "root collapses back but keeps its page number");

    // Regrow.
    for k in 1..=150 {
        t.insert(k, format!("r{k}").as_bytes()).unwrap();
    }
    let keys: Vec<i64> = t.full_scan().unwrap().iter().map(|(k, _)| *k).collect();
    assert_eq!(keys, (1..=150).collect::<Vec<_>>());
}

// ---------------------------------------------------------------------------
// Store: multiple trees in one file
// ---------------------------------------------------------------------------

#[test]
fn store_multiple_trees_are_independent() {
    let (_d, path) = temp_path();
    let store = Store::open(&path).unwrap();
    let a = store.create_tree().unwrap();
    let b = store.create_tree().unwrap();
    assert_ne!(a, b);

    let ta = store.tree(a);
    let tb = store.tree(b);
    for k in 1..=50 {
        ta.insert(k, b"a").unwrap();
    }
    for k in 100..=150 {
        tb.insert(k, b"b").unwrap();
    }
    assert_eq!(ta.max_key().unwrap(), Some(50));
    assert_eq!(tb.max_key().unwrap(), Some(150));
    assert_eq!(ta.get(120).unwrap(), None);
    assert_eq!(tb.get(120).unwrap().as_deref(), Some(b"b".as_ref()));
}

// ---------------------------------------------------------------------------
// Malformed-page decode: fail loud with a typed error, never panic
// ---------------------------------------------------------------------------

#[test]
fn leaf_cell_with_out_of_range_pointer_errors_not_panics() {
    // Build a valid one-cell leaf, then corrupt its cell pointer so it points
    // past the usable area — the kind of damage a writer bug could leave behind
    // a still-valid CRC. The decoder must return a typed Integrity error rather
    // than slice-index past the buffer and panic.
    let cells = vec![encode_inline_cell(1, b"hello")];
    let mut page = write_leaf_from_raw(&cells, 0).unwrap();

    let bogus = (USABLE_END + 16) as u16;
    page[LEAF_PTR_ARRAY_START..LEAF_PTR_ARRAY_START + 2].copy_from_slice(&bogus.to_be_bytes());

    let err = read_leaf_cell_raw(&page, 0, OVERFLOW_THRESHOLD).unwrap_err();
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity error on malformed cell pointer, got {err:?}"
    );
}

#[test]
fn interior_cell_with_out_of_range_pointer_errors_not_panics() {
    // A valid interior node, then a corrupted separator-cell pointer.
    let page_ok = write_interior_node(&[10, 20], &[2u32, 3, 4]).unwrap();
    let mut page = page_ok;

    // The interior pointer array begins at the interior header end; the first
    // pointer occupies the two bytes there. Point it past the usable area.
    let bogus = (USABLE_END + 8) as u16;
    let ptr_at = lillibrain::consts::INTERIOR_HEADER_SIZE;
    page[ptr_at..ptr_at + 2].copy_from_slice(&bogus.to_be_bytes());

    let err = read_interior_node(&page).unwrap_err();
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity error on malformed interior pointer, got {err:?}"
    );
}

#[test]
fn leaf_header_with_impossible_cell_count_errors_not_panics() {
    // A page whose header claims the maximum possible cell count. The pointer
    // array is filled with a single valid in-range pointer so the per-pointer
    // bounds check passes for every low index; the loop then marches the pointer
    // offset `ptr_array_start + idx * 2` past the end of the page, where an
    // unchecked u16 read would slice out of bounds and panic across the FFI
    // boundary. Reading the keys must surface a typed error first.
    let cells = vec![encode_inline_cell(1, b"hi")];
    let mut page = write_leaf_from_raw(&cells, 0).unwrap();

    // Claim the maximum cell count.
    page[3..5].copy_from_slice(&u16::MAX.to_be_bytes());
    // Recover the one valid cell pointer the packer wrote.
    let valid_ptr =
        u16::from_be_bytes(page[LEAF_PTR_ARRAY_START..LEAF_PTR_ARRAY_START + 2].try_into().unwrap());
    // Fill the whole pointer-array region with that valid pointer so every low
    // index reads a decodable cell and the loop survives toward the page end.
    let mut at = LEAF_PTR_ARRAY_START;
    while at + 2 <= PAGE_SIZE {
        page[at..at + 2].copy_from_slice(&valid_ptr.to_be_bytes());
        at += 2;
    }

    let err = read_leaf_keys(&page, OVERFLOW_THRESHOLD)
        .expect_err("an impossible cell count must be a typed error, not a panic");
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity error on impossible cell count, got {err:?}"
    );
}

#[test]
fn interior_header_with_impossible_cell_count_errors_not_panics() {
    // The interior reader loops over the claimed cell count too. Fill the pointer
    // array with a valid in-range pointer so the loop marches past the page end,
    // where an unchecked read would slice out of bounds; an impossible count must
    // surface a typed error first.
    let page_ok = write_interior_node(&[10], &[2u32, 3]).unwrap();
    let mut page = page_ok;
    page[3..5].copy_from_slice(&u16::MAX.to_be_bytes());
    let ptr_start = lillibrain::consts::INTERIOR_HEADER_SIZE;
    let valid_ptr = u16::from_be_bytes(page[ptr_start..ptr_start + 2].try_into().unwrap());
    let mut at = ptr_start;
    while at + 2 <= PAGE_SIZE {
        page[at..at + 2].copy_from_slice(&valid_ptr.to_be_bytes());
        at += 2;
    }

    let err = read_interior_node(&page)
        .expect_err("an impossible interior cell count must be a typed error, not a panic");
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity error on impossible interior cell count, got {err:?}"
    );
}

#[test]
fn leaf_cell_longer_than_usable_page_errors_not_panics() {
    // A single cell whose byte length exceeds the usable page area: packing it
    // subtracts its length from the write position, which would underflow usize
    // and panic across the FFI boundary. The packer must return a typed error.
    let oversized = vec![0xABu8; USABLE_END + 16];
    let err = write_leaf_from_raw(&[oversized], 0)
        .expect_err("an over-long leaf cell must be a typed error, not an underflow panic");
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity error on over-long leaf cell, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// Cells-visited counter
// ---------------------------------------------------------------------------

#[test]
fn cells_visited_counter_increments_once_per_visited_cell() {
    let (_d, store, root) = open_tree();
    let t = store.tree(root);

    let n: i64 = 250;
    for k in 1..=n {
        t.insert(k, format!("v{k}").as_bytes()).unwrap();
    }

    // A full streaming scan visits every cell exactly once.
    store.reset_cells_visited_count();
    let mut seen = 0u64;
    t.scan_cells_with::<StoreError, _>(|_k, _payload| {
        seen += 1;
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, n as u64, "the visitor must see every inserted cell");
    assert_eq!(
        store.cells_visited_count(),
        n as u64,
        "the cells-visited counter must increment once per visited cell across a full scan"
    );

    // An early-stopping visitor (the LIMIT shape) touches only the cells it
    // consumes before returning Err -- the counter must reflect the truncated
    // walk, not the whole corpus.
    let limit: u64 = 10;
    store.reset_cells_visited_count();
    let mut consumed = 0u64;
    let res = t.scan_cells_with::<StoreError, _>(|_k, _payload| {
        consumed += 1;
        if consumed >= limit {
            return Err(StoreError::Integrity {
                detail: "early-stop sentinel".to_string(),
            });
        }
        Ok(())
    });
    assert!(res.is_err(), "the early-stop sentinel must surface as Err");
    assert_eq!(
        store.cells_visited_count(),
        limit,
        "an early-stopped scan must visit exactly the cells it consumed (O(LIMIT)), \
         not the whole corpus (O(corpus))"
    );

    // Reset zeroes the counter.
    store.reset_cells_visited_count();
    assert_eq!(store.cells_visited_count(), 0);
}

#[test]
fn get_many_binary_seek_matches_per_key_gets() {
    // Multi-page tree with absent keys, duplicate requests, and page-boundary
    // targets: the walk's per-page binary seek must return exactly what
    // per-key gets return, in ascending order, absents skipped, one entry
    // per duplicated request.
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    // Payloads sized so leaves hold a handful of cells each -> many pages.
    for k in (0..3000).step_by(3) {
        let payload = vec![b'x'; 200 + (k % 7) as usize];
        t.insert(k as i64, &payload).unwrap();
    }

    let mut requests: Vec<i64> = Vec::new();
    for k in (0..3000).step_by(11) {
        requests.push(k as i64); // mix of present (k%3==0) and absent
    }
    requests.push(0);
    requests.push(0); // duplicate of the first key
    requests.push(2997); // last present key
    requests.push(9999); // beyond the tree
    requests.push(-5); // before the tree
    requests.sort_unstable();

    let got = t.get_many(&requests).unwrap();

    let mut expected: Vec<(i64, Vec<u8>)> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &k in &requests {
        if !seen.insert(k) {
            continue;
        }
        if let Some(p) = t.get(k).unwrap() {
            expected.push((k, p));
        }
    }
    assert_eq!(got.len(), expected.len());
    assert_eq!(got, expected);
}

#[test]
fn get_many_sparse_over_fat_rows_falls_back_and_stays_correct() {
    // Fat inline rows pack ~2 per leaf, so a few hundred rows span hundreds of
    // pages. A handful of sparse requests then exhaust the page budget
    // (~4 + 3*keys) long before the walk reaches them, forcing the per-key
    // descent fallback — which must still return every key, including one whose
    // value spills to an overflow chain.
    let (_d, store, root) = open_tree();
    let t = store.tree(root);
    for k in 0..400i64 {
        let payload = vec![b'y'; 4000 + (k % 5) as usize];
        t.insert(k, &payload).unwrap();
    }
    let big = vec![b'z'; OVERFLOW_THRESHOLD + 500];
    t.insert(1000, &big).unwrap();

    let requests = vec![0i64, 200, 399, 1000];
    let got = t.get_many(&requests).unwrap();

    let expected: Vec<(i64, Vec<u8>)> = requests
        .iter()
        .map(|&k| (k, t.get(k).unwrap().unwrap()))
        .collect();
    assert_eq!(got, expected, "fallback must return every key, overflow included");
    assert_eq!(
        got.last().unwrap().1.len(),
        OVERFLOW_THRESHOLD + 500,
        "the overflow payload must round-trip through get_many"
    );
}
