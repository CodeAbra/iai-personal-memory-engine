//! Structural integrity checker for the B-tree: a single-tree walk over a root
//! page that verifies the on-disk invariants and returns a list of violations
//! (empty when clean). Used by tests and as a maintenance diagnostic; it never
//! panics on a malformed page — a fault is surfaced as an error string.
//!
//! Invariant groups:
//!   1. Page-number bounds — no reference beyond `db_size`
//!   2. No cycle or double-referenced page in the tree walk
//!   3. Strictly increasing keys within every node
//!   4. Interior child count equals key count plus one
//!   5. All child page numbers within range
//!   6. All leaves at equal depth
//!   7. No orphaned pages (every allocated page is reachable, an overflow page,
//!      or on the freelist)
//!   8. No page on both the freelist and the live tree
//!   9. Freelist header count matches the freelist set size
//!  10. Cross-page leaf keys form one strictly ascending global sequence
//!  11. Leaf sibling-chain order matches the tree-walk leaf order
//!  12. Sibling links point within the allocated range
//!  13-15. Overflow pages are role-disjoint, cycle-free, and in range; the
//!      reassembled payload length matches the declared length
//!  16. No page appears in more than one role

use crate::btree::overflow::read_overflow_chain;
use crate::btree::page::{
    get_sibling, page_kind, read_interior_node, read_leaf_cell_raw, read_leaf_header, PageKind,
};
use crate::consts::OVERFLOW_THRESHOLD;
use crate::error::Result;
use crate::freelist::freelist_pages;
use crate::pager::Pager;

use std::collections::HashSet;

#[allow(clippy::too_many_arguments)]
fn walk_overflow_chain(
    pager: &Pager,
    first_page: u32,
    total_len: usize,
    overflow_pages: &mut HashSet<u32>,
    errors: &mut Vec<String>,
    db_size: u32,
) {
    let mut collected = 0usize;
    let mut current = first_page;
    let mut seen: HashSet<u32> = HashSet::new();

    while current != 0 {
        if seen.contains(&current) {
            errors.push(format!("overflow chain cycle at page {current}"));
            return;
        }
        if current < 1 || current > db_size {
            errors.push(format!(
                "overflow chain page {current} out of range (db_size={db_size})"
            ));
            return;
        }
        seen.insert(current);
        overflow_pages.insert(current);

        let page = match pager.read_page(current) {
            Ok(p) => p,
            Err(e) => {
                errors.push(format!("overflow page {current}: read error: {e}"));
                return;
            }
        };
        let next = u32::from_be_bytes(page[0..4].try_into().unwrap());
        let chunk_len = crate::btree::overflow::OVERFLOW_USABLE;
        collected += chunk_len;
        current = next;
    }

    if collected < total_len {
        errors.push(format!(
            "overflow chain starting at page {first_page}: collected {collected} bytes < declared {total_len}"
        ));
        return;
    }
    // An overlong chain carries at least one whole page more than the declared
    // payload needs: the surplus pages are reachable but never read back — leaked.
    if collected >= total_len + crate::btree::overflow::OVERFLOW_USABLE {
        errors.push(format!(
            "overflow chain starting at page {first_page}: {collected} bytes exceeds declared {total_len} by a full page (leaked overflow pages)"
        ));
    }

    match read_overflow_chain(pager, first_page, total_len) {
        Ok(reassembled) => {
            if reassembled.len() != total_len {
                errors.push(format!(
                    "overflow chain starting at page {first_page}: reassembled {} bytes != declared {total_len}",
                    reassembled.len()
                ));
            }
        }
        Err(e) => errors.push(format!(
            "overflow chain starting at page {first_page}: reassembly error: {e}"
        )),
    }
}

/// Recursively walk the subtree at `page_no`, returning its leaf depth (or
/// `None` when a fault prevented reaching leaves).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn walk_tree(
    pager: &Pager,
    page_no: u32,
    depth: usize,
    // Inclusive lower / exclusive upper key bound this subtree must fall within,
    // derived from the ancestor separators (routing uses bisect_right, so a child
    // holds keys in [keys[i-1], keys[i])). A key outside these bounds is a
    // separator that misroutes lookups even when leaves stay globally sorted.
    lower: Option<i64>,
    upper: Option<i64>,
    visited: &mut HashSet<u32>,
    leaf_order: &mut Vec<u32>,
    errors: &mut Vec<String>,
    db_size: u32,
    overflow_pages: &mut HashSet<u32>,
) -> Option<usize> {
    if page_no < 1 || page_no > db_size {
        errors.push(format!(
            "page {page_no}: page number out of allocated range (db_size={db_size})"
        ));
        return None;
    }
    if visited.contains(&page_no) {
        errors.push(format!(
            "page {page_no}: referenced more than once (cycle or double-reference)"
        ));
        return None;
    }
    visited.insert(page_no);

    let page = match pager.read_page(page_no) {
        Ok(p) => p,
        Err(e) => {
            errors.push(format!("page {page_no}: read error: {e}"));
            return None;
        }
    };

    let kind = match page_kind(&page) {
        Ok(k) => k,
        Err(e) => {
            errors.push(format!("page {page_no}: {e}"));
            return None;
        }
    };

    match kind {
        PageKind::Leaf => {
            let hdr = read_leaf_header(&page);
            let mut prev: Option<i64> = None;
            for i in 0..hdr.num_cells {
                let raw = match read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD) {
                    Ok(r) => r,
                    Err(e) => {
                        errors.push(format!("page {page_no}: cell {i} decode error: {e}"));
                        continue;
                    }
                };
                if let Some(p) = prev {
                    if raw.key <= p {
                        errors.push(format!(
                            "page {page_no}: leaf keys not strictly in order at position {i} (keys[{}]={p}, keys[{i}]={})",
                            i - 1,
                            raw.key
                        ));
                    }
                }
                prev = Some(raw.key);

                if lower.is_some_and(|lo| raw.key < lo) {
                    errors.push(format!(
                        "page {page_no}: leaf key {} below subtree lower bound {} (misrouting separator)",
                        raw.key,
                        lower.unwrap()
                    ));
                }
                if upper.is_some_and(|hi| raw.key >= hi) {
                    errors.push(format!(
                        "page {page_no}: leaf key {} at/above subtree upper bound {} (misrouting separator)",
                        raw.key,
                        upper.unwrap()
                    ));
                }

                if raw.spilled {
                    match crate::btree::page::read_spilled_pointer(&page, raw.payload_start) {
                        Ok(first_ovf) => walk_overflow_chain(
                            pager,
                            first_ovf,
                            raw.payload_len,
                            overflow_pages,
                            errors,
                            db_size,
                        ),
                        Err(e) => errors.push(format!(
                            "page {page_no}: cell {i} overflow pointer read error: {e}"
                        )),
                    }
                }
            }
            leaf_order.push(page_no);
            Some(depth)
        }
        PageKind::Interior => {
            let node = match read_interior_node(&page) {
                Ok(n) => n,
                Err(e) => {
                    errors.push(format!("page {page_no}: interior decode error: {e}"));
                    return None;
                }
            };
            for i in 1..node.keys.len() {
                if node.keys[i] <= node.keys[i - 1] {
                    errors.push(format!(
                        "page {page_no}: interior keys not strictly in order at position {i} (keys[{}]={}, keys[{i}]={})",
                        i - 1,
                        node.keys[i - 1],
                        node.keys[i]
                    ));
                }
            }
            if node.children.len() != node.keys.len() + 1 {
                errors.push(format!(
                    "page {page_no}: interior child count {} != len(keys)+1 ({})",
                    node.children.len(),
                    node.keys.len() + 1
                ));
            }
            for &child in &node.children {
                if child < 1 || child > db_size {
                    errors.push(format!(
                        "page {page_no}: child page {child} out of allocated range (db_size={db_size})"
                    ));
                }
            }

            let mut leaf_depths: Vec<usize> = Vec::new();
            for (i, &child) in node.children.iter().enumerate() {
                // child[i] holds keys in [keys[i-1], keys[i]); the outermost
                // children inherit the parent's own bounds.
                let child_lower = if i == 0 {
                    lower
                } else {
                    Some(node.keys[i - 1])
                };
                let child_upper = if i == node.keys.len() {
                    upper
                } else {
                    Some(node.keys[i])
                };
                if let Some(d) = walk_tree(
                    pager,
                    child,
                    depth + 1,
                    child_lower,
                    child_upper,
                    visited,
                    leaf_order,
                    errors,
                    db_size,
                    overflow_pages,
                ) {
                    leaf_depths.push(d);
                }
            }
            if leaf_depths.is_empty() {
                return None;
            }
            let first = leaf_depths[0];
            for &d in &leaf_depths[1..] {
                if d != first {
                    errors.push(format!(
                        "page {page_no}: subtrees have leaves at different depths ({first} vs {d})"
                    ));
                }
            }
            Some(first)
        }
    }
}

/// A page-role census: the distinct page numbers reachable as B-tree nodes,
/// reachable as overflow pages, and on the freelist. Used to assert page
/// conservation (no leak, no double-free) independently of the full checker.
pub struct PageCensus {
    /// Distinct B-tree node pages reachable from the root.
    pub tree_nodes: HashSet<u32>,
    /// Distinct overflow pages reachable from spilled leaf cells.
    pub overflow_pages: HashSet<u32>,
    /// Distinct pages on the freelist chain.
    pub freelist: HashSet<u32>,
}

/// Walk the tree and freelist, returning the page-role census for conservation
/// checks. Does not validate invariants — pair it with [`check_integrity`].
pub fn page_census(pager: &Pager, root_page_no: u32) -> Result<PageCensus> {
    let db_size = pager.db_size()?;
    let mut visited: HashSet<u32> = HashSet::new();
    let mut leaf_order: Vec<u32> = Vec::new();
    let mut errors: Vec<String> = Vec::new();
    let mut overflow_pages: HashSet<u32> = HashSet::new();
    walk_tree(
        pager,
        root_page_no,
        0,
        None,
        None,
        &mut visited,
        &mut leaf_order,
        &mut errors,
        db_size,
        &mut overflow_pages,
    );
    let freelist = freelist_pages(pager).unwrap_or_default();
    Ok(PageCensus {
        tree_nodes: visited,
        overflow_pages,
        freelist,
    })
}

/// Verify the structural invariants of the tree rooted at `root_page_no`.
/// Returns a list of violation descriptions (empty when clean).
///
/// The orphan-page check treats page 1 (the header) and any page in
/// `reserved_pages` as structurally valid residents even when they are not
/// reachable from the tree — the store reserves a small set of fixed pages
/// (e.g. a root-slot and a meta page) that are not part of a user tree.
pub fn check_integrity(pager: &Pager, root_page_no: u32) -> Result<Vec<String>> {
    check_integrity_with_reserved(pager, root_page_no, &[])
}

/// Like [`check_integrity`] but with an explicit set of reserved page numbers
/// that are exempt from the orphan-page check.
pub fn check_integrity_with_reserved(
    pager: &Pager,
    root_page_no: u32,
    reserved_pages: &[u32],
) -> Result<Vec<String>> {
    let reserved: HashSet<u32> = reserved_pages.iter().copied().collect();
    let mut errors: Vec<String> = Vec::new();
    let db_size = pager.db_size()?;

    // Trailing pages past the recorded db_size are invisible to the orphan scan
    // (which stops at db_size), so surface them explicitly as a leak signal.
    if let Ok(physical) = pager.physical_page_count() {
        if physical > db_size {
            errors.push(format!(
                "file has {physical} pages but header db_size={db_size}: {} trailing pages past the recorded extent",
                physical - db_size
            ));
        }
    }

    if db_size < 1 {
        errors.push(format!("db_size={db_size}: invalid (must be >= 1)"));
        return Ok(errors);
    }
    if root_page_no < 1 || root_page_no > db_size {
        errors.push(format!(
            "page {root_page_no}: root page number out of allocated range (db_size={db_size})"
        ));
        return Ok(errors);
    }

    let mut visited: HashSet<u32> = HashSet::new();
    let mut leaf_order: Vec<u32> = Vec::new();
    let mut overflow_pages: HashSet<u32> = HashSet::new();

    walk_tree(
        pager,
        root_page_no,
        0,
        None,
        None,
        &mut visited,
        &mut leaf_order,
        &mut errors,
        db_size,
        &mut overflow_pages,
    );

    // Freelist set (may itself surface corruption).
    let free_set = match freelist_pages(pager) {
        Ok(s) => s,
        Err(e) => {
            errors.push(format!("freelist corruption: {e}"));
            HashSet::new()
        }
    };

    // Group 8: freelist disjoint from live tree.
    let overlap: Vec<u32> = visited.intersection(&free_set).copied().collect();
    if !overlap.is_empty() {
        let mut sorted = overlap;
        sorted.sort_unstable();
        errors.push(format!(
            "pages on both the live tree and the freelist: {sorted:?}"
        ));
    }

    // Group 7: no orphaned pages. Page 1 is the header; the tree root may live
    // anywhere, so the orphan scan covers every page from 2 to db_size and
    // treats the root's own subtree (visited), overflow pages, and the freelist
    // as the reachable sets. Other trees' pages would show as orphans, so this
    // check is meaningful only for a single-tree store (the test scope).
    for page_no in 2..=db_size {
        if !visited.contains(&page_no)
            && !free_set.contains(&page_no)
            && !overflow_pages.contains(&page_no)
            && !reserved.contains(&page_no)
        {
            errors.push(format!(
                "page {page_no}: allocated but not reachable from the tree and not on the freelist (orphaned page)"
            ));
        }
    }

    // Group 9: freelist header count matches the actual set size.
    let header_count = pager.freelist_count()?;
    if header_count as usize != free_set.len() {
        errors.push(format!(
            "freelist count mismatch: header says {header_count}, actual freelist has {} pages",
            free_set.len()
        ));
    }

    // Group 10: cross-page leaf keys form one strictly ascending sequence.
    let mut all_keys: Vec<i64> = Vec::new();
    for &leaf_no in &leaf_order {
        let page = pager.read_page(leaf_no)?;
        let hdr = read_leaf_header(&page);
        for i in 0..hdr.num_cells {
            if let Ok(raw) = read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD) {
                all_keys.push(raw.key);
            }
        }
    }
    for i in 1..all_keys.len() {
        if all_keys[i] <= all_keys[i - 1] {
            errors.push(format!(
                "cross-page key ordering violation: global key sequence not strictly ascending at position {i} (key[{}]={}, key[{i}]={})",
                i - 1,
                all_keys[i - 1],
                all_keys[i]
            ));
            break;
        }
    }

    // Groups 11 & 12: sibling chain order and bounds.
    if let Some(&first_leaf) = leaf_order.first() {
        let mut chain: Vec<u32> = Vec::new();
        let mut seen: HashSet<u32> = HashSet::new();
        let mut current = first_leaf;
        while current != 0 {
            if current < 1 || current > db_size {
                errors.push(format!(
                    "leaf sibling link points to out-of-range page {current} (db_size={db_size})"
                ));
                break;
            }
            if seen.contains(&current) {
                errors.push(format!(
                    "leaf sibling chain cycle: page {current} visited twice"
                ));
                break;
            }
            seen.insert(current);
            chain.push(current);
            current = match pager.read_page(current) {
                Ok(p) => get_sibling(&p),
                Err(e) => {
                    errors.push(format!("leaf {current}: sibling read error: {e}"));
                    break;
                }
            };
        }
        if chain != leaf_order && !errors.iter().any(|e| e.contains("sibling")) {
            errors.push(format!(
                "leaf sibling chain order does not match tree-walk order: sibling={chain:?}, tree-walk={leaf_order:?}"
            ));
        }
    }

    // Groups 13 & 16: overflow pages role-disjoint from tree nodes and freelist.
    let tree_ovf: Vec<u32> = visited.intersection(&overflow_pages).copied().collect();
    if !tree_ovf.is_empty() {
        let mut s = tree_ovf;
        s.sort_unstable();
        errors.push(format!(
            "pages are both B-tree nodes and overflow pages: {s:?}"
        ));
    }
    let free_ovf: Vec<u32> = free_set.intersection(&overflow_pages).copied().collect();
    if !free_ovf.is_empty() {
        let mut s = free_ovf;
        s.sort_unstable();
        errors.push(format!(
            "pages are both on the freelist and overflow pages: {s:?}"
        ));
    }

    Ok(errors)
}
