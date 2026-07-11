//! B-tree split: copy-up leaf split and push-up interior split.
//!
//! Two separate functions enforce distinct separator semantics:
//!
//!   Leaf split (copy-up): the separator is the first key of the right group and
//!   stays there; a copy of it is promoted to the parent.
//!
//!   Interior split (push-up): the boundary key is extracted and sent to the
//!   parent, absent from both resulting children; key count is conserved.
//!
//! Keeping them as two functions is deliberate — a single function with a node
//! kind flag makes the copy-up/push-up distinction easy to misconfigure.
//!
//! Root discipline: the root page number is fixed for the life of the tree. When
//! the root overflows, its content moves to a freshly allocated child page and
//! the fixed root page is rewritten as an interior node over the new children.

use crate::btree::cursor::{
    build_cell_for, leaf_max_cells, read_full_cell, read_sibling, write_sibling, Cell,
};
use crate::btree::page::{
    interior_fits, on_page_cell_size, read_interior_node, read_leaf_header, write_interior_node,
    write_leaf_from_raw, LEAF_PTR_ARRAY_START, USABLE_END,
};
use crate::consts::OVERFLOW_THRESHOLD;
use crate::error::{Result, StoreError};
use crate::freelist::alloc_page;
use crate::pager::Pager;

/// On-page footprint of a leaf laid out from `(key, full_payload_len)` cells.
fn leaf_footprint(cells: &[(i64, usize)]) -> usize {
    let ptr_area = LEAF_PTR_ARRAY_START + cells.len() * 2;
    let cell_bytes: usize = cells
        .iter()
        .map(|(k, len)| on_page_cell_size(*k, *len, OVERFLOW_THRESHOLD))
        .sum();
    ptr_area + cell_bytes
}

/// Greedily pack sorted cells into N>=1 key-ordered groups each fitting a page.
fn balanced_nway_leaf_split(cells: &[Cell]) -> Result<Vec<Vec<Cell>>> {
    for (k, v) in cells {
        if leaf_footprint(&[(*k, v.len())]) > USABLE_END {
            return Err(StoreError::Integrity {
                detail: "leaf split: a single cell exceeds one page".to_string(),
            });
        }
    }
    let mut groups: Vec<Vec<Cell>> = Vec::new();
    let mut current: Vec<Cell> = Vec::new();
    for cell in cells {
        let mut probe: Vec<(i64, usize)> = current.iter().map(|(k, v)| (*k, v.len())).collect();
        probe.push((cell.0, cell.1.len()));
        if !current.is_empty() && leaf_footprint(&probe) > USABLE_END {
            groups.push(std::mem::take(&mut current));
            current.push(cell.clone());
        } else {
            current.push(cell.clone());
        }
    }
    if !current.is_empty() {
        groups.push(current);
    }
    Ok(groups)
}

/// Write a group of `(key, value)` cells to a leaf page, spilling oversize
/// payloads to fresh overflow chains, preserving `sibling`.
fn write_leaf_group(pager: &Pager, page_no: u32, group: &[Cell], sibling: u32) -> Result<()> {
    let mut raw_cells: Vec<Vec<u8>> = Vec::with_capacity(group.len());
    for (k, v) in group {
        raw_cells.push(build_cell_for(pager, *k, v)?);
    }
    let page = write_leaf_from_raw(&raw_cells, sibling)?;
    pager.write_page(page_no, &page)
}

/// Split a full leaf and insert `(new_key, new_value)`, copy-up style.
pub fn leaf_split_and_insert(
    pager: &Pager,
    mut path_stack: Vec<(u32, usize)>,
    leaf_page_no: u32,
    new_key: i64,
    new_value: &[u8],
) -> Result<()> {
    let page = pager.read_page(leaf_page_no)?;
    let hdr = read_leaf_header(&page);

    // Reassemble all existing cells (full payloads) and free their old chains so
    // re-spilling does not orphan pages.
    let mut old_cells: Vec<Cell> = Vec::with_capacity(hdr.num_cells);
    for i in 0..hdr.num_cells {
        old_cells.push(read_full_cell(pager, &page, i)?);
    }
    free_spilled_chains(pager, &page, hdr.num_cells)?;

    // Merge the new cell into sorted position.
    let keys: Vec<i64> = old_cells.iter().map(|(k, _)| *k).collect();
    let idx = crate::btree::cursor::bisect_left(&keys, new_key);
    let mut all_cells = old_cells;
    all_cells.insert(idx, (new_key, new_value.to_vec()));

    let groups = balanced_nway_leaf_split(&all_cells)?;
    let old_right_sibling = read_sibling(pager, leaf_page_no)?;

    // A one-group partition fits every cell on one page: rewrite in place.
    if groups.len() == 1 {
        write_leaf_group(pager, leaf_page_no, &groups[0], old_right_sibling)?;
        return Ok(());
    }

    // Group 0 stays on the original page; allocate one new page per right group.
    let mut group_page_nos = vec![leaf_page_no];
    for _ in &groups[1..] {
        group_page_nos.push(alloc_page(pager)?);
    }

    for (page_no, group) in group_page_nos.iter().zip(&groups) {
        write_leaf_group(pager, *page_no, group, 0)?;
    }
    // Rethread the sibling chain: group0 -> new1 -> ... -> old_right_sibling.
    for i in 0..group_page_nos.len() {
        let next = if i + 1 < group_page_nos.len() {
            group_page_nos[i + 1]
        } else {
            old_right_sibling
        };
        write_sibling(pager, group_page_nos[i], next)?;
    }

    // Copy-up separators: first key of each right group.
    let separators: Vec<i64> = (1..groups.len()).map(|i| groups[i][0].0).collect();

    if path_stack.is_empty() {
        root_from_children(pager, &group_page_nos, &separators)
    } else {
        let (parent_page_no, _) = path_stack.pop().unwrap();
        insert_run_into_interior(
            pager,
            path_stack,
            parent_page_no,
            leaf_page_no,
            &group_page_nos[1..],
            &separators,
        )
    }
}

/// Free the overflow chains of every spilled cell on a leaf page.
fn free_spilled_chains(pager: &Pager, page: &[u8], num_cells: usize) -> Result<()> {
    use crate::btree::overflow::free_overflow_chain;
    use crate::btree::page::{read_leaf_cell_raw, read_spilled_pointer};
    for i in 0..num_cells {
        let raw = read_leaf_cell_raw(page, i, OVERFLOW_THRESHOLD)?;
        if raw.spilled {
            let first = read_spilled_pointer(page, raw.payload_start)?;
            free_overflow_chain(pager, first)?;
        }
    }
    Ok(())
}

/// Splice a run of `(separator, new_child)` pairs into an interior node, after
/// the old child pointer, splitting push-up style when it overflows.
fn insert_run_into_interior(
    pager: &Pager,
    path_stack: Vec<(u32, usize)>,
    parent_page_no: u32,
    old_child_page_no: u32,
    new_child_page_nos: &[u32],
    separators: &[i64],
) -> Result<()> {
    let page = pager.read_page(parent_page_no)?;
    let node = read_interior_node(&page)?;

    let j = node
        .children
        .iter()
        .position(|&c| c == old_child_page_no)
        .ok_or_else(|| StoreError::Integrity {
            detail: "split: old child not found in parent".to_string(),
        })?;

    let mut new_keys = node.keys[..j].to_vec();
    new_keys.extend_from_slice(separators);
    new_keys.extend_from_slice(&node.keys[j..]);

    let mut new_children = node.children[..=j].to_vec();
    new_children.extend_from_slice(new_child_page_nos);
    new_children.extend_from_slice(&node.children[j + 1..]);

    if new_keys.len() <= leaf_max_cells() && interior_fits(&new_keys) {
        let new_page = write_interior_node(&new_keys, &new_children)?;
        pager.write_page(parent_page_no, &new_page)?;
        Ok(())
    } else {
        interior_split_and_insert(pager, path_stack, parent_page_no, &new_keys, &new_children)
    }
}

/// Partition an over-capacity interior node into N>=1 page-fitting nodes,
/// extracting N-1 boundary keys for the parent (push-up).
#[allow(clippy::type_complexity)]
fn balanced_nway_interior_split(
    all_keys: &[i64],
    all_children: &[u32],
) -> (Vec<(Vec<i64>, Vec<u32>)>, Vec<i64>) {
    let mut nodes: Vec<(Vec<i64>, Vec<u32>)> = Vec::new();
    let mut push_up_keys: Vec<i64> = Vec::new();

    let mut cur_children: Vec<u32> = vec![all_children[0]];
    let mut cur_keys: Vec<i64> = Vec::new();

    for (i, key) in all_keys.iter().enumerate() {
        let next_child = all_children[i + 1];
        let mut probe = cur_keys.clone();
        probe.push(*key);
        if !cur_keys.is_empty() && !interior_fits(&probe) {
            nodes.push((std::mem::take(&mut cur_keys), std::mem::take(&mut cur_children)));
            push_up_keys.push(*key);
            cur_children = vec![next_child];
        } else {
            cur_keys.push(*key);
            cur_children.push(next_child);
        }
    }
    nodes.push((cur_keys, cur_children));
    (nodes, push_up_keys)
}

/// N-way split a full interior node, pushing N-1 boundary keys to the parent.
pub fn interior_split_and_insert(
    pager: &Pager,
    mut path_stack: Vec<(u32, usize)>,
    interior_page_no: u32,
    all_keys: &[i64],
    all_children: &[u32],
) -> Result<()> {
    let (nodes, push_up_keys) = balanced_nway_interior_split(all_keys, all_children);

    if nodes.len() == 1 {
        let new_page = write_interior_node(all_keys, all_children)?;
        pager.write_page(interior_page_no, &new_page)?;
        return Ok(());
    }

    let mut node_page_nos = vec![interior_page_no];
    for _ in &nodes[1..] {
        node_page_nos.push(alloc_page(pager)?);
    }
    for (page_no, (keys, children)) in node_page_nos.iter().zip(&nodes) {
        let new_page = write_interior_node(keys, children)?;
        pager.write_page(*page_no, &new_page)?;
    }

    if path_stack.is_empty() {
        root_from_children(pager, &node_page_nos, &push_up_keys)
    } else {
        let (parent_page_no, _) = path_stack.pop().unwrap();
        insert_run_into_interior(
            pager,
            path_stack,
            parent_page_no,
            interior_page_no,
            &node_page_nos[1..],
            &push_up_keys,
        )
    }
}

/// Reinitialize the fixed root page as an interior node over N children.
///
/// Node 0's current content (still on the root page) is copied to a freshly
/// allocated child page so the root page can be overwritten; the rest are the
/// new right children written by the caller. The root page number never changes.
fn root_from_children(pager: &Pager, node_page_nos: &[u32], separators: &[i64]) -> Result<()> {
    let root_page_no = node_page_nos[0];
    let new_node0 = alloc_page(pager)?;
    let root_data = pager.read_page(root_page_no)?;
    pager.write_page(new_node0, &root_data)?;

    let mut children = vec![new_node0];
    children.extend_from_slice(&node_page_nos[1..]);

    let new_root = write_interior_node(separators, &children)?;
    pager.write_page(root_page_no, &new_root)
}
