//! Structural integrity checker: clean on a well-formed tree, and flags an
//! injected key-order violation, an orphan page, and a freelist cycle.

use std::path::PathBuf;

use lillibrain::btree::page::{
    read_interior_node, write_interior_node, write_leaf_from_raw, INTERIOR_PTR_ARRAY_START,
};
use lillibrain::codec::{
    decode_record, decode_record_columns, encode_record, encode_varint, Value,
};
use lillibrain::consts::{HDR_FIRST_FREELIST_OFFSET, PAGE_SIZE};
use lillibrain::error::StoreError;
use lillibrain::Store;

fn temp_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

/// Build a tree large enough to be multi-level under a dense cell cap.
fn populated_store() -> (tempfile::TempDir, PathBuf, Store, u32) {
    let (dir, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        for k in 0..120i64 {
            t.insert(k, format!("val-{k}").as_bytes()).unwrap();
        }
    }
    store.commit().unwrap();
    (dir, path, store, root)
}

#[test]
fn integrity_well_formed_tree_is_clean() {
    let (_d, _p, store, root) = populated_store();
    let errs = store.check_integrity(root).unwrap();
    assert!(errs.is_empty(), "expected clean, got: {errs:?}");

    // After deletes the tree must still be clean.
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        for k in (0..120i64).step_by(3) {
            t.delete(k).unwrap();
        }
    }
    store.commit().unwrap();
    let errs = store.check_integrity(root).unwrap();
    assert!(
        errs.is_empty(),
        "expected clean after deletes, got: {errs:?}"
    );
}

#[test]
fn integrity_flags_injected_leaf_key_order_violation() {
    let (_d, _p, store, root) = populated_store();
    let pager = store.pager();

    // Find a leaf and write its cells in descending order to break group 3/10.
    // The root may be interior; descend to its leftmost leaf via the node.
    let mut page_no = root;
    loop {
        let page = pager.read_page(page_no).unwrap();
        if page[0] == lillibrain::consts::PAGE_LEAF_TABLE {
            break;
        }
        let node = read_interior_node(&page).unwrap();
        page_no = node.children[0];
    }

    // Rebuild this leaf with two out-of-order cells.
    let bad_cells = vec![
        // key 100 then key 1 — descending, violating strict ascending order.
        {
            let mut c = lillibrain::codec::encode_varint(100);
            c.extend_from_slice(&lillibrain::codec::encode_varint(1));
            c.push(0x41);
            c
        },
        {
            let mut c = lillibrain::codec::encode_varint(1);
            c.extend_from_slice(&lillibrain::codec::encode_varint(1));
            c.push(0x42);
            c
        },
    ];
    let bad_leaf = write_leaf_from_raw(&bad_cells, 0).unwrap();
    pager.write_page(page_no, &bad_leaf).unwrap();
    pager.flush().unwrap();

    let errs = store.check_integrity(root).unwrap();
    assert!(
        errs.iter()
            .any(|e| e.contains("not strictly in order") || e.contains("cross-page key ordering")),
        "expected a key-order violation, got: {errs:?}"
    );
}

#[test]
fn integrity_flags_orphan_page() {
    let (_d, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    store.begin_write().unwrap();
    store.tree(root).insert(1, b"x").unwrap();
    store.commit().unwrap();

    // Extend the file with a page that is neither in the tree nor on the
    // freelist — a true orphan.
    let pager = store.pager();
    store.begin_write().unwrap();
    let orphan = pager.extend_file().unwrap();
    pager.write_page(orphan, &vec![0u8; PAGE_SIZE]).unwrap();
    store.commit().unwrap();

    let errs = store.check_integrity(root).unwrap();
    assert!(
        errs.iter()
            .any(|e| e.contains(&format!("page {orphan}")) && e.contains("orphaned")),
        "expected an orphan-page violation for page {orphan}, got: {errs:?}"
    );
}

#[test]
fn integrity_flags_freelist_cycle() {
    let (_d, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = store.create_tree().unwrap();
    store.begin_write().unwrap();
    store.tree(root).insert(1, b"x").unwrap();
    store.commit().unwrap();

    let pager = store.pager();
    // Allocate a page and forge a self-referential trunk as the freelist head.
    store.begin_write().unwrap();
    let trunk = pager.extend_file().unwrap();
    let mut buf = vec![0u8; PAGE_SIZE];
    // next_trunk -> itself (cycle); leaf_count = 0.
    buf[0..4].copy_from_slice(&trunk.to_be_bytes());
    pager.write_page(trunk, &buf).unwrap();
    pager
        .update_header_u32(HDR_FIRST_FREELIST_OFFSET, trunk)
        .unwrap();
    store.commit().unwrap();

    let errs = store.check_integrity(root).unwrap();
    assert!(
        errs.iter()
            .any(|e| e.contains("freelist") && (e.contains("cycle") || e.contains("corruption"))),
        "expected a freelist-cycle violation, got: {errs:?}"
    );
}

#[test]
fn integrity_flags_child_out_of_range() {
    let (_d, _p, store, root) = populated_store();
    let pager = store.pager();

    // The root must be interior after 120 inserts under any reasonable cap; if it
    // is a leaf, force a child-range fault on a synthetic interior page instead.
    let page = pager.read_page(root).unwrap();
    if page[0] == lillibrain::consts::PAGE_INTERIOR_TABLE {
        // Corrupt the first child pointer to an out-of-range page number.
        let mut p = page.clone();
        let cell_ptr = u16::from_be_bytes(
            p[INTERIOR_PTR_ARRAY_START..INTERIOR_PTR_ARRAY_START + 2]
                .try_into()
                .unwrap(),
        ) as usize;
        let huge = u32::MAX;
        p[cell_ptr..cell_ptr + 4].copy_from_slice(&huge.to_be_bytes());
        pager.write_page(root, &p).unwrap();
        pager.flush().unwrap();
        let errs = store.check_integrity(root).unwrap();
        assert!(
            errs.iter().any(|e| e.contains("out of allocated range")),
            "expected an out-of-range child violation, got: {errs:?}"
        );
    } else {
        // Fallback: synthesize an interior root pointing at out-of-range children.
        let bad = write_interior_node(&[5], &[u32::MAX, u32::MAX - 1]).unwrap();
        pager.write_page(root, &bad).unwrap();
        pager.flush().unwrap();
        let errs = store.check_integrity(root).unwrap();
        assert!(
            errs.iter().any(|e| e.contains("out of allocated range")),
            "expected an out-of-range child violation, got: {errs:?}"
        );
    }
}

#[test]
fn integrity_flags_misrouting_separator() {
    let (_d, _p, store, root) = populated_store();
    let pager = store.pager();

    let page = pager.read_page(root).unwrap();
    if page[0] != lillibrain::consts::PAGE_INTERIOR_TABLE {
        return; // shallow tree — no separator to corrupt
    }
    let node = read_interior_node(&page).unwrap();

    // Drop the first separator below the leftmost child's key range: every key
    // in child[0] is now at or above its (exclusive) upper bound, a misroute the
    // global leaf-order check alone cannot see. Interior keys stay strictly
    // ordered (-1 < the next separator) and the leaves are untouched, so only the
    // subtree-bound check can catch this.
    let mut bad_keys = node.keys.clone();
    bad_keys[0] = -1;
    let bad = write_interior_node(&bad_keys, &node.children).unwrap();
    pager.write_page(root, &bad).unwrap();
    pager.flush().unwrap();

    let errs = store.check_integrity(root).unwrap();
    assert!(
        errs.iter().any(|e| e.contains("misrouting separator")),
        "expected a misrouting-separator violation, got: {errs:?}"
    );
}

// ---------------------------------------------------------------------------
// Codec hostile-input arithmetic: overflow yields a typed error, never a panic
// ---------------------------------------------------------------------------

/// Build a record header naming `serial_types` with no column bodies. The
/// projected decoder advances its body offset by each declared body size while
/// skipping unwanted columns, so a header alone is enough to exercise the
/// offset arithmetic on a buffer that carries no actual body bytes.
fn header_only_record(serial_types: &[i64]) -> Vec<u8> {
    let mut type_bytes = Vec::new();
    for st in serial_types {
        type_bytes.extend_from_slice(&encode_varint(*st));
    }
    // header_size counts itself; iterate to a fixed point on the varint width.
    let mut header_size_varint = encode_varint((encode_varint(0).len() + type_bytes.len()) as i64);
    for _ in 0..3 {
        let candidate = (header_size_varint.len() + type_bytes.len()) as i64;
        let next = encode_varint(candidate);
        if next.len() + type_bytes.len() == candidate as usize {
            header_size_varint = next;
            break;
        }
        header_size_varint = next;
    }
    let mut out = Vec::with_capacity(header_size_varint.len() + type_bytes.len());
    out.extend_from_slice(&header_size_varint);
    out.extend_from_slice(&type_bytes);
    out
}

#[test]
fn projected_decode_with_overflowing_body_offset_returns_integrity_error() {
    // Several skipped blob columns each declare a body size near half the usize
    // range. Summing them advances the body offset past usize without a single
    // body byte being read, so the unchecked `+=` would wrap and panic across
    // the FFI boundary. The decoder must surface a typed integrity error.
    let huge_blob = i64::MAX; // odd >= 13 => a blob body of (st-13)/2 bytes
    let bytes = header_only_record(&[huge_blob, huge_blob, huge_blob, huge_blob]);

    // No column is wanted: the skip path advances the offset for every column.
    let wanted = [false, false, false, false];
    let err = decode_record_columns(&bytes, 0, &wanted)
        .expect_err("an overflowing body offset must be a typed error, not a panic");
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity on body-offset overflow, got {err:?}"
    );
}

#[test]
fn full_decode_with_overflowing_body_offset_returns_integrity_error() {
    // The full decoder reads each body, so an out-of-range body slice is already
    // rejected; assert it stays a typed error (never a panic) for the same input.
    let huge_blob = i64::MAX;
    let bytes = header_only_record(&[huge_blob, huge_blob]);
    let err = decode_record(&bytes, 0)
        .expect_err("an out-of-range body must be a typed error, not a panic");
    assert!(
        matches!(err, StoreError::Integrity { .. }),
        "expected Integrity on out-of-range body, got {err:?}"
    );
}

#[test]
fn verbatim_blob_corpus_roundtrips_byte_identically_after_checked_arithmetic() {
    // The checked-arithmetic change must not touch the happy-path byte layout:
    // a NUL / emoji / CJK / RTL blob still round-trips byte-for-byte.
    let nul = b"head\x00\x00mid\x00tail".to_vec();
    let emoji = "fire 🔥 rocket 🚀 family 👨‍👩‍👧‍👦".as_bytes().to_vec();
    let cjk = "日本語と中文と한국어".as_bytes().to_vec();
    let rtl = "\u{202B}مرحبا\u{200F}שלום\u{202C}world".as_bytes().to_vec();

    let values = vec![
        Value::Blob(nul.clone()),
        Value::Text("plain text".to_string()),
        Value::Blob(emoji.clone()),
        Value::Blob(cjk.clone()),
        Value::Blob(rtl.clone()),
    ];
    let enc = encode_record(&values);
    let (dec, consumed) = decode_record(&enc, 0).unwrap();
    assert_eq!(consumed, enc.len());
    assert_eq!(dec[0], Value::Blob(nul));
    assert_eq!(dec[1], Value::Text("plain text".to_string()));
    assert_eq!(dec[2], Value::Blob(emoji));
    assert_eq!(dec[3], Value::Blob(cjk));
    assert_eq!(dec[4], Value::Blob(rtl));
}
