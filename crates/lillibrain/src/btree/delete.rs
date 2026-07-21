//! B-tree deletion: leaf removal, borrow/merge rebalance, interior underflow,
//! and root shrink.
//!
//! A non-root node below the minimum occupancy rebalances. The minimum is
//! byte-driven by default (a leaf under half a page is underfull) with a
//! count floor that fires only under a small injected cell ceiling so the
//! borrow/merge paths can be exercised without large payloads. The root is
//! exempt and may hold any number of keys, including zero — a zero-key interior
//! root collapses, decreasing tree height while keeping the root page fixed.

use crate::btree::cursor::{build_cell_for, leaf_max_cells, read_full_cell, small_cap_active, Cell};
use crate::btree::overflow::free_overflow_chain;
use crate::btree::page::{
    get_sibling, interior_fits, interior_used_bytes, on_page_cell_size, page_kind,
    raw_leaf_cell_bytes, read_interior_node, read_leaf_cell_raw, read_leaf_header,
    read_spilled_pointer, write_interior_node, write_leaf_from_raw, PageKind,
    LEAF_PTR_ARRAY_START, USABLE_END,
};
use crate::btree::split::interior_split_and_insert;
use crate::consts::OVERFLOW_THRESHOLD;
use crate::error::{Result, StoreError};
use crate::freelist::free_page;
use crate::pager::Pager;

/// Minimum cell count for a non-root leaf under a small injected ceiling.
pub fn leaf_min_cells() -> usize {
    leaf_max_cells() / 2
}

/// Minimum key count for a non-root interior node under a small injected ceiling.
fn interior_min_keys() -> usize {
    leaf_max_cells() / 2
}

/// Whether a non-root interior node is below minimum occupancy. Byte-driven
/// by default (under half a page is underfull); the count floor applies only
/// under a small injected cell ceiling, mirroring the leaf rules. Interior
/// cells carry variable-length varint keys, so a count threshold derived from
/// the injected ceiling is meaningless against real byte capacity.
fn interior_underflows_node(keys: &[i64]) -> bool {
    if keys.is_empty() {
        return true;
    }
    if small_cap_active() {
        return keys.len() < interior_min_keys();
    }
    interior_used_bytes(keys) < USABLE_END / 2
}

/// Write an interior node, splitting push-up style when it exceeds one page.
/// A delete-path parent rewrite can GROW the page (a replacement separator's
/// varint may be longer than the one it replaces), so every such rewrite must
/// be prepared to split.
fn write_interior_or_split(
    pager: &Pager,
    path_stack: Vec<(u32, usize)>,
    page_no: u32,
    keys: &[i64],
    children: &[u32],
) -> Result<()> {
    if keys.len() <= leaf_max_cells() && interior_fits(keys) {
        let page = write_interior_node(keys, children)?;
        pager.write_page(page_no, &page)
    } else {
        interior_split_and_insert(pager, path_stack, page_no, keys, children)
    }
}

/// Split index for redistributing an over-capacity merged interior node: the
/// key at the returned index is promoted to the parent; both remaining halves
/// must fit one page, with the larger half minimized. The pre-merge boundary
/// is always feasible, so this cannot fail for a node built by a merge of two
/// valid pages.
fn interior_byte_midpoint(keys: &[i64]) -> Result<usize> {
    let cap = leaf_max_cells();
    let mut best: Option<(usize, usize)> = None;
    for i in 1..keys.len().saturating_sub(1).max(1) {
        let left = &keys[..i];
        let right = &keys[i + 1..];
        if left.len() > cap || right.len() > cap {
            continue;
        }
        if !interior_fits(left) || !interior_fits(right) {
            continue;
        }
        let larger = interior_used_bytes(left).max(interior_used_bytes(right));
        if best.map_or(true, |(_, b)| larger < b) {
            best = Some((i, larger));
        }
    }
    best.map(|(i, _)| i).ok_or_else(|| StoreError::Integrity {
        detail: "interior redistribute: no split keeps both halves within one page".to_string(),
    })
}

/// On-page footprint of a leaf laid out from `(key, full_len)` cells.
fn leaf_footprint(cells: &[(i64, usize)]) -> usize {
    let ptr_area = LEAF_PTR_ARRAY_START + cells.len() * 2;
    let cell_bytes: usize = cells
        .iter()
        .map(|(k, len)| on_page_cell_size(*k, *len, OVERFLOW_THRESHOLD))
        .sum();
    ptr_area + cell_bytes
}

/// Used on-page byte footprint of an existing leaf page.
fn leaf_used_bytes(page: &[u8], num_cells: usize) -> Result<usize> {
    let mut used = LEAF_PTR_ARRAY_START + num_cells * 2;
    for i in 0..num_cells {
        let raw = read_leaf_cell_raw(page, i, OVERFLOW_THRESHOLD)?;
        used += on_page_cell_size(raw.key, raw.payload_len, OVERFLOW_THRESHOLD);
    }
    Ok(used)
}

/// Whether a leaf is below the half-page fill threshold (or empty).
pub fn leaf_underflows(page: &[u8], num_cells: usize) -> Result<bool> {
    if num_cells == 0 {
        return Ok(true);
    }
    Ok(leaf_used_bytes(page, num_cells)? < USABLE_END / 2)
}

/// Whether a leaf built from `cells` would sit below the half-page fill
/// threshold — the byteful mirror of [`leaf_underflows`] for an in-memory cell
/// list, so a borrow donor can be tested without re-reading its page.
fn leaf_cells_underflow(cells: &[Cell]) -> bool {
    if cells.is_empty() {
        return true;
    }
    let footprint: Vec<(i64, usize)> = cells.iter().map(|(k, v)| (*k, v.len())).collect();
    leaf_footprint(&footprint) < USABLE_END / 2
}

/// Reassemble every `(key, value)` cell on a leaf page with full payloads.
fn read_all_full(pager: &Pager, page_no: u32) -> Result<(Vec<Cell>, u32)> {
    let page = pager.read_page(page_no)?;
    let hdr = read_leaf_header(&page);
    let mut cells = Vec::with_capacity(hdr.num_cells);
    for i in 0..hdr.num_cells {
        cells.push(read_full_cell(pager, &page, i)?);
    }
    Ok((cells, get_sibling(&page)))
}

/// Free the overflow chains of every spilled cell on a leaf page.
fn free_spilled_chains(pager: &Pager, page: &[u8], num_cells: usize) -> Result<()> {
    for i in 0..num_cells {
        let raw = read_leaf_cell_raw(page, i, OVERFLOW_THRESHOLD)?;
        if raw.spilled {
            let first = read_spilled_pointer(page, raw.payload_start)?;
            free_overflow_chain(pager, first)?;
        }
    }
    Ok(())
}

/// Write `(key, value)` cells to a page, re-spilling oversize payloads.
fn write_leaf_cells(pager: &Pager, page_no: u32, cells: &[Cell], sibling: u32) -> Result<()> {
    let mut raw_cells: Vec<Vec<u8>> = Vec::with_capacity(cells.len());
    for (k, v) in cells {
        raw_cells.push(build_cell_for(pager, *k, v)?);
    }
    let page = write_leaf_from_raw(&raw_cells, sibling)?;
    pager.write_page(page_no, &page)
}

// ---------------------------------------------------------------------------
// Leaf cell removal
// ---------------------------------------------------------------------------

/// Remove the cell for `key` from a leaf. Returns whether a cell was removed.
///
/// The removed cell's overflow chain (if any) is freed; the remaining cells are
/// copied verbatim so their overflow pointers are preserved without re-alloc.
pub fn remove_cell_from_leaf(pager: &Pager, leaf_page_no: u32, key: i64) -> Result<bool> {
    let page = pager.read_page(leaf_page_no)?;
    let hdr = read_leaf_header(&page);
    let keys = crate::btree::page::read_leaf_keys(&page, OVERFLOW_THRESHOLD)?;
    let idx = crate::btree::cursor::bisect_left(&keys, key);
    if idx >= keys.len() || keys[idx] != key {
        return Ok(false);
    }

    let removed = read_leaf_cell_raw(&page, idx, OVERFLOW_THRESHOLD)?;
    if removed.spilled {
        let first = read_spilled_pointer(&page, removed.payload_start)?;
        free_overflow_chain(pager, first)?;
    }

    let sibling = get_sibling(&page);
    let mut raw_cells: Vec<Vec<u8>> = Vec::with_capacity(hdr.num_cells - 1);
    for i in 0..hdr.num_cells {
        if i == idx {
            continue;
        }
        raw_cells.push(raw_leaf_cell_bytes(&page, i, OVERFLOW_THRESHOLD)?);
    }
    let new_page = write_leaf_from_raw(&raw_cells, sibling)?;
    pager.write_page(leaf_page_no, &new_page)?;
    Ok(true)
}

// ---------------------------------------------------------------------------
// Parent helpers
// ---------------------------------------------------------------------------

fn update_parent_separator(
    pager: &Pager,
    parent_stack: Vec<(u32, usize)>,
    parent_page_no: u32,
    key_idx: usize,
    new_key: i64,
) -> Result<()> {
    let page = pager.read_page(parent_page_no)?;
    let mut node = read_interior_node(&page)?;
    node.keys[key_idx] = new_key;
    write_interior_or_split(pager, parent_stack, parent_page_no, &node.keys, &node.children)
}

fn remove_from_parent(
    pager: &Pager,
    path_stack: Vec<(u32, usize)>,
    parent_page_no: u32,
    key_idx: usize,
    right_child_idx: usize,
) -> Result<()> {
    let page = pager.read_page(parent_page_no)?;
    let node = read_interior_node(&page)?;

    let mut new_keys = node.keys[..key_idx].to_vec();
    new_keys.extend_from_slice(&node.keys[key_idx + 1..]);
    let mut new_children = node.children[..right_child_idx].to_vec();
    new_children.extend_from_slice(&node.children[right_child_idx + 1..]);

    let new_page = write_interior_node(&new_keys, &new_children)?;
    pager.write_page(parent_page_no, &new_page)?;

    if path_stack.is_empty() {
        return Ok(());
    }
    if interior_underflows_node(&new_keys) {
        check_interior_underflow(pager, path_stack, parent_page_no)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Leaf borrow / merge
// ---------------------------------------------------------------------------

fn borrow_fits_receiver(receiver: &[Cell], borrowed: &(i64, Vec<u8>)) -> bool {
    let mut probe: Vec<(i64, usize)> = receiver.iter().map(|(k, v)| (*k, v.len())).collect();
    probe.push((borrowed.0, borrowed.1.len()));
    leaf_footprint(&probe) <= USABLE_END
}

fn borrow_from_right_leaf(
    pager: &Pager,
    parent_stack: Vec<(u32, usize)>,
    leaf_page_no: u32,
    right_sibling_no: u32,
    parent_page_no: u32,
    child_idx: usize,
) -> Result<()> {
    let (left_cells, left_sibling) = read_all_full(pager, leaf_page_no)?;
    let (right_cells, right_sibling) = read_all_full(pager, right_sibling_no)?;
    // Free BOTH the receiver's and the donor's old spilled chains before the
    // re-spilling rewrite: write_leaf_cells re-spills oversize payloads into
    // fresh chains, so a retained receiver cell that was itself spilled would
    // orphan its original chain unless freed here.
    let left_page = pager.read_page(leaf_page_no)?;
    free_spilled_chains(pager, &left_page, left_cells.len())?;
    let right_page = pager.read_page(right_sibling_no)?;
    free_spilled_chains(pager, &right_page, right_cells.len())?;

    let borrowed = right_cells[0].clone();
    let mut new_left = left_cells;
    new_left.push(borrowed);
    let new_right: Vec<Cell> = right_cells[1..].to_vec();
    let new_separator = new_right[0].0;

    write_leaf_cells(pager, leaf_page_no, &new_left, left_sibling)?;
    write_leaf_cells(pager, right_sibling_no, &new_right, right_sibling)?;
    update_parent_separator(pager, parent_stack, parent_page_no, child_idx, new_separator)
}

fn borrow_from_left_leaf(
    pager: &Pager,
    parent_stack: Vec<(u32, usize)>,
    leaf_page_no: u32,
    left_sibling_no: u32,
    parent_page_no: u32,
    child_idx: usize,
) -> Result<()> {
    let (left_cells, left_sibling) = read_all_full(pager, left_sibling_no)?;
    let (right_cells, right_sibling) = read_all_full(pager, leaf_page_no)?;
    let left_page = pager.read_page(left_sibling_no)?;
    free_spilled_chains(pager, &left_page, left_cells.len())?;
    let right_page = pager.read_page(leaf_page_no)?;
    free_spilled_chains(pager, &right_page, right_cells.len())?;

    let borrowed = left_cells[left_cells.len() - 1].clone();
    let new_left: Vec<Cell> = left_cells[..left_cells.len() - 1].to_vec();
    let mut new_right = vec![borrowed.clone()];
    new_right.extend(right_cells);
    let new_separator = borrowed.0;

    write_leaf_cells(pager, left_sibling_no, &new_left, left_sibling)?;
    write_leaf_cells(pager, leaf_page_no, &new_right, right_sibling)?;
    update_parent_separator(pager, parent_stack, parent_page_no, child_idx - 1, new_separator)
}

/// Split combined cells into (left, right) so both halves fit one page, choosing
/// the boundary that minimizes the larger half.
fn byte_midpoint_split(cells: &[Cell]) -> Result<(Vec<Cell>, Vec<Cell>)> {
    let n = cells.len();
    if n < 2 {
        return Err(StoreError::Integrity {
            detail: "redistribute needs at least two cells".to_string(),
        });
    }
    let footprint = |slice: &[Cell]| -> usize {
        let pairs: Vec<(i64, usize)> = slice.iter().map(|(k, v)| (*k, v.len())).collect();
        leaf_footprint(&pairs)
    };
    let mut best_idx: Option<usize> = None;
    let mut best_max: usize = usize::MAX;
    for split_idx in 1..n {
        let fl = footprint(&cells[..split_idx]);
        let fr = footprint(&cells[split_idx..]);
        if fl > USABLE_END || fr > USABLE_END {
            continue;
        }
        let larger = fl.max(fr);
        if larger < best_max {
            best_max = larger;
            best_idx = Some(split_idx);
        }
    }
    let idx = best_idx.ok_or_else(|| StoreError::Integrity {
        detail: "redistribute: no split keeps both halves within one page".to_string(),
    })?;
    Ok((cells[..idx].to_vec(), cells[idx..].to_vec()))
}

fn merge_leaves(
    pager: &Pager,
    path_stack: Vec<(u32, usize)>,
    left_page_no: u32,
    right_page_no: u32,
    parent_page_no: u32,
    separator_idx: usize,
) -> Result<()> {
    let (left_cells, _left_sibling) = read_all_full(pager, left_page_no)?;
    let (right_cells, right_sibling) = read_all_full(pager, right_page_no)?;

    let left_page = pager.read_page(left_page_no)?;
    free_spilled_chains(pager, &left_page, left_cells.len())?;
    let right_page = pager.read_page(right_page_no)?;
    free_spilled_chains(pager, &right_page, right_cells.len())?;

    let mut merged = left_cells;
    merged.extend(right_cells);

    let pairs: Vec<(i64, usize)> = merged.iter().map(|(k, v)| (*k, v.len())).collect();
    if leaf_footprint(&pairs) <= USABLE_END {
        // True merge.
        write_leaf_cells(pager, left_page_no, &merged, right_sibling)?;
        free_page(pager, right_page_no)?;
        remove_from_parent(pager, path_stack, parent_page_no, separator_idx, separator_idx + 1)?;
        return Ok(());
    }

    // Redistribute.
    let (new_left, new_right) = byte_midpoint_split(&merged)?;
    write_leaf_cells(pager, left_page_no, &new_left, right_page_no)?;
    write_leaf_cells(pager, right_page_no, &new_right, right_sibling)?;
    update_parent_separator(pager, path_stack, parent_page_no, separator_idx, new_right[0].0)
}

/// Fix an underflowing non-root leaf via borrow or merge.
pub fn rebalance_leaf(pager: &Pager, path_stack: Vec<(u32, usize)>, leaf_page_no: u32) -> Result<()> {
    if path_stack.is_empty() {
        return Ok(());
    }
    let (parent_page_no, child_idx) = *path_stack.last().unwrap();
    let parent_stack = path_stack[..path_stack.len() - 1].to_vec();

    let parent_page = pager.read_page(parent_page_no)?;
    let parent = read_interior_node(&parent_page)?;
    if parent.children[child_idx] != leaf_page_no {
        return Err(StoreError::Integrity {
            detail: "rebalance_leaf: parent child index mismatch".to_string(),
        });
    }

    let right_sibling = parent.children.get(child_idx + 1).copied();
    let left_sibling = if child_idx >= 1 {
        Some(parent.children[child_idx - 1])
    } else {
        None
    };

    let (recv_cells, _recv_sibling) = read_all_full(pager, leaf_page_no)?;

    // Borrow from right. A donor qualifies when it stays at or above the
    // half-page fill threshold after giving up its first cell (byte-driven by
    // default, mirroring the interior rule); the count floor applies only under
    // an injected small ceiling. The `> 1` / `.max(1)` floors also keep the
    // donor at two-plus cells so the borrow's first-cell access never indexes an
    // emptied list.
    if let Some(right_no) = right_sibling {
        let (right_cells, _) = read_all_full(pager, right_no)?;
        let donor_ok = if small_cap_active() {
            right_cells.len() > leaf_min_cells().max(1)
        } else {
            right_cells.len() > 1 && !leaf_cells_underflow(&right_cells[1..])
        };
        if donor_ok && borrow_fits_receiver(&recv_cells, &right_cells[0]) {
            return borrow_from_right_leaf(
                pager, parent_stack, leaf_page_no, right_no, parent_page_no, child_idx,
            );
        }
    }
    // Borrow from left.
    if let Some(left_no) = left_sibling {
        let (left_cells, _) = read_all_full(pager, left_no)?;
        let donor_ok = if small_cap_active() {
            left_cells.len() > leaf_min_cells().max(1)
        } else {
            left_cells.len() > 1 && !leaf_cells_underflow(&left_cells[..left_cells.len() - 1])
        };
        if donor_ok {
            let borrowed = left_cells[left_cells.len() - 1].clone();
            if borrow_fits_receiver(&recv_cells, &borrowed) {
                return borrow_from_left_leaf(
                    pager, parent_stack, leaf_page_no, left_no, parent_page_no, child_idx,
                );
            }
        }
    }
    // Merge: prefer the right sibling.
    if let Some(right_no) = right_sibling {
        merge_leaves(pager, parent_stack, leaf_page_no, right_no, parent_page_no, child_idx)
    } else if let Some(left_no) = left_sibling {
        merge_leaves(pager, parent_stack, left_no, leaf_page_no, parent_page_no, child_idx - 1)
    } else {
        // A non-root leaf reached via a parent always has a sibling: a well-formed
        // non-root interior holds two-plus children. No sibling here means the
        // tree is structurally corrupt — fail loud rather than silently leaving an
        // underfull leaf linked.
        Err(StoreError::Integrity {
            detail: "rebalance_leaf: underflowing non-root leaf has no sibling".to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// Interior borrow / merge
// ---------------------------------------------------------------------------

fn borrow_from_right_interior(
    pager: &Pager,
    parent_stack: Vec<(u32, usize)>,
    interior_page_no: u32,
    right_sibling_no: u32,
    parent_page_no: u32,
    child_idx: usize,
) -> Result<()> {
    let left = read_interior_node(&pager.read_page(interior_page_no)?)?;
    let right = read_interior_node(&pager.read_page(right_sibling_no)?)?;
    let parent = read_interior_node(&pager.read_page(parent_page_no)?)?;

    let pulled = parent.keys[child_idx];
    let pushed_up = right.keys[0];
    let adopted_child = right.children[0];

    let mut new_left_keys = left.keys.clone();
    new_left_keys.push(pulled);
    let mut new_left_children = left.children.clone();
    new_left_children.push(adopted_child);

    let new_right_keys = right.keys[1..].to_vec();
    let new_right_children = right.children[1..].to_vec();

    let mut new_parent_keys = parent.keys.clone();
    new_parent_keys[child_idx] = pushed_up;

    pager.write_page(interior_page_no, &write_interior_node(&new_left_keys, &new_left_children)?)?;
    pager.write_page(right_sibling_no, &write_interior_node(&new_right_keys, &new_right_children)?)?;
    write_interior_or_split(pager, parent_stack, parent_page_no, &new_parent_keys, &parent.children)
}

fn borrow_from_left_interior(
    pager: &Pager,
    parent_stack: Vec<(u32, usize)>,
    interior_page_no: u32,
    left_sibling_no: u32,
    parent_page_no: u32,
    child_idx: usize,
) -> Result<()> {
    let right = read_interior_node(&pager.read_page(interior_page_no)?)?;
    let left = read_interior_node(&pager.read_page(left_sibling_no)?)?;
    let parent = read_interior_node(&pager.read_page(parent_page_no)?)?;

    let pulled = parent.keys[child_idx - 1];
    let pushed_up = left.keys[left.keys.len() - 1];
    let donated_child = left.children[left.children.len() - 1];

    let mut new_right_keys = vec![pulled];
    new_right_keys.extend_from_slice(&right.keys);
    let mut new_right_children = vec![donated_child];
    new_right_children.extend_from_slice(&right.children);

    let new_left_keys = left.keys[..left.keys.len() - 1].to_vec();
    let new_left_children = left.children[..left.children.len() - 1].to_vec();

    let mut new_parent_keys = parent.keys.clone();
    new_parent_keys[child_idx - 1] = pushed_up;

    pager.write_page(interior_page_no, &write_interior_node(&new_right_keys, &new_right_children)?)?;
    pager.write_page(left_sibling_no, &write_interior_node(&new_left_keys, &new_left_children)?)?;
    write_interior_or_split(pager, parent_stack, parent_page_no, &new_parent_keys, &parent.children)
}

fn merge_interior_nodes(
    pager: &Pager,
    path_stack: Vec<(u32, usize)>,
    left_page_no: u32,
    right_page_no: u32,
    parent_page_no: u32,
    separator_idx: usize,
) -> Result<()> {
    let left = read_interior_node(&pager.read_page(left_page_no)?)?;
    let right = read_interior_node(&pager.read_page(right_page_no)?)?;
    let parent = read_interior_node(&pager.read_page(parent_page_no)?)?;

    let pulled = parent.keys[separator_idx];
    let mut merged_keys = left.keys.clone();
    merged_keys.push(pulled);
    merged_keys.extend_from_slice(&right.keys);
    let mut merged_children = left.children.clone();
    merged_children.extend_from_slice(&right.children);

    if merged_keys.len() <= leaf_max_cells() && interior_fits(&merged_keys) {
        // True merge.
        pager.write_page(left_page_no, &write_interior_node(&merged_keys, &merged_children)?)?;
        free_page(pager, right_page_no)?;

        let mut new_parent_keys = parent.keys[..separator_idx].to_vec();
        new_parent_keys.extend_from_slice(&parent.keys[separator_idx + 1..]);
        let mut new_parent_children = parent.children[..separator_idx + 1].to_vec();
        new_parent_children.extend_from_slice(&parent.children[separator_idx + 2..]);
        pager.write_page(
            parent_page_no,
            &write_interior_node(&new_parent_keys, &new_parent_children)?,
        )?;

        if path_stack.is_empty() {
            return Ok(());
        }
        if interior_underflows_node(&new_parent_keys) {
            check_interior_underflow(pager, path_stack, parent_page_no)?;
        }
        return Ok(());
    }

    // Redistribute: the combined cells exceed one page (interior cells carry
    // variable-length varint keys, so count-balanced siblings need not be
    // byte-mergeable). Re-split around a promoted separator; the parent keeps
    // a replaced separator instead of losing one.
    let split_idx = interior_byte_midpoint(&merged_keys)?;
    let promoted = merged_keys[split_idx];
    pager.write_page(
        left_page_no,
        &write_interior_node(&merged_keys[..split_idx], &merged_children[..=split_idx])?,
    )?;
    pager.write_page(
        right_page_no,
        &write_interior_node(&merged_keys[split_idx + 1..], &merged_children[split_idx + 1..])?,
    )?;
    let mut new_parent_keys = parent.keys.clone();
    new_parent_keys[separator_idx] = promoted;
    write_interior_or_split(pager, path_stack, parent_page_no, &new_parent_keys, &parent.children)
}

fn check_interior_underflow(
    pager: &Pager,
    path_stack: Vec<(u32, usize)>,
    interior_page_no: u32,
) -> Result<()> {
    if path_stack.is_empty() {
        return Ok(());
    }
    let (parent_page_no, child_idx) = *path_stack.last().unwrap();
    let parent_stack = path_stack[..path_stack.len() - 1].to_vec();

    let parent = read_interior_node(&pager.read_page(parent_page_no)?)?;
    if parent.children[child_idx] != interior_page_no {
        return Err(StoreError::Integrity {
            detail: "check_interior_underflow: parent child index mismatch".to_string(),
        });
    }

    let right_sibling = parent.children.get(child_idx + 1).copied();
    let left_sibling = if child_idx >= 1 {
        Some(parent.children[child_idx - 1])
    } else {
        None
    };

    let node = read_interior_node(&pager.read_page(interior_page_no)?)?;

    // A donor qualifies when it stays healthy after donating one cell; the
    // receiver must fit the pulled separator. Count rules apply only under an
    // injected small ceiling.
    if let Some(right_no) = right_sibling {
        let right = read_interior_node(&pager.read_page(right_no)?)?;
        let donor_ok = if small_cap_active() {
            right.keys.len() > interior_min_keys()
        } else {
            right.keys.len() > 1 && !interior_underflows_node(&right.keys[1..])
        };
        let mut probe = node.keys.clone();
        probe.push(parent.keys[child_idx]);
        if donor_ok && interior_fits(&probe) {
            return borrow_from_right_interior(
                pager, parent_stack, interior_page_no, right_no, parent_page_no, child_idx,
            );
        }
    }
    if let Some(left_no) = left_sibling {
        let left = read_interior_node(&pager.read_page(left_no)?)?;
        let donor_ok = if small_cap_active() {
            left.keys.len() > interior_min_keys()
        } else {
            left.keys.len() > 1 && !interior_underflows_node(&left.keys[..left.keys.len() - 1])
        };
        let mut probe = vec![parent.keys[child_idx - 1]];
        probe.extend_from_slice(&node.keys);
        if donor_ok && interior_fits(&probe) {
            return borrow_from_left_interior(
                pager, parent_stack, interior_page_no, left_no, parent_page_no, child_idx,
            );
        }
    }
    if let Some(right_no) = right_sibling {
        merge_interior_nodes(pager, parent_stack, interior_page_no, right_no, parent_page_no, child_idx)
    } else if let Some(left_no) = left_sibling {
        merge_interior_nodes(pager, parent_stack, left_no, interior_page_no, parent_page_no, child_idx - 1)
    } else {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Root shrink
// ---------------------------------------------------------------------------

/// Collapse a zero-key interior root: its sole child's content moves into the
/// fixed root page and the child page is freed. The root page number is fixed.
pub fn shrink_root_if_needed(pager: &Pager, root_page_no: u32) -> Result<()> {
    loop {
        let root_data = pager.read_page(root_page_no)?;
        match page_kind(&root_data)? {
            PageKind::Leaf => return Ok(()),
            PageKind::Interior => {
                let node = read_interior_node(&root_data)?;
                if !node.keys.is_empty() {
                    return Ok(());
                }
                if node.children.len() != 1 {
                    return Err(StoreError::Integrity {
                        detail: format!(
                            "shrink_root: zero keys but {} children",
                            node.children.len()
                        ),
                    });
                }
                let only_child = node.children[0];
                let child_data = pager.read_page(only_child)?;
                pager.write_page(root_page_no, &child_data)?;
                free_page(pager, only_child)?;
            }
        }
    }
}
