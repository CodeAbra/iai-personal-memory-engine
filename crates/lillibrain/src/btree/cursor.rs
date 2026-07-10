//! B-tree cursor: descent, insert with split dispatch, delete, scans, and the
//! tree-height max-key.
//!
//! The cursor reads and writes pages through the pager only. Every mutation is
//! persisted by a later pager flush. The leaf cell pointer array and the leaf
//! sibling chain are the two structures it walks: descent and point lookup walk
//! interior nodes by key, while ordered iteration walks the leaf sibling chain.

use crate::btree::overflow::{free_overflow_chain, read_overflow_chain, write_overflow_chain};
use crate::btree::page::{
    empty_leaf, encode_inline_cell, encode_spilled_cell, get_sibling, on_page_cell_size,
    page_kind, raw_leaf_cell_bytes, read_interior_header, read_interior_node, read_leaf_cell_raw,
    read_leaf_header, read_leaf_keys, read_spilled_pointer, set_sibling, write_leaf_from_raw,
    LeafCellRaw, PageKind, LEAF_NUM_CELLS_OFFSET, LEAF_PTR_ARRAY_START, LEAF_SIBLING_OFFSET,
    USABLE_END,
};
use crate::consts::OVERFLOW_THRESHOLD;
use crate::error::{Result, StoreError};
use crate::pager::Pager;

/// Cell-count ceiling that fires a split before byte capacity is reached. The
/// default is effectively unreachable (byte capacity governs splits); a smaller
/// override exercises the split/merge/borrow paths without large payloads.
pub fn leaf_max_cells() -> usize {
    std::env::var("LILLI_LEAF_MAX_CELLS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9999)
}

const SMALL_CAP_SENTINEL: usize = 9999;

/// A materialized `(key, value)` leaf cell with its full payload.
pub type Cell = (i64, Vec<u8>);

/// A cursor bound to one B-tree rooted at a fixed page number.
pub struct Cursor<'p> {
    pager: &'p Pager,
    root_page_no: u32,
    /// Path stack from the last descent: `(page_no, child_index)` root-to-parent.
    path_stack: Vec<(u32, usize)>,
}

impl<'p> Cursor<'p> {
    /// Bind a cursor to the tree rooted at `root_page_no`.
    pub fn new(pager: &'p Pager, root_page_no: u32) -> Cursor<'p> {
        Cursor {
            pager,
            root_page_no,
            path_stack: Vec::new(),
        }
    }

    // -----------------------------------------------------------------------
    // Descent
    // -----------------------------------------------------------------------

    /// Descend from the root to the leaf that should hold `key`, recording the
    /// `(page_no, child_index)` path. Returns `(leaf_page_no, insert_index)`.
    fn move_to(&mut self, key: i64) -> Result<(u32, usize)> {
        self.path_stack.clear();
        let mut page_no = self.root_page_no;
        loop {
            let page = self.pager.read_page(page_no)?;
            match page_kind(&page)? {
                PageKind::Leaf => {
                    let keys = read_leaf_keys(&page, OVERFLOW_THRESHOLD)?;
                    let idx = bisect_left(&keys, key);
                    return Ok((page_no, idx));
                }
                PageKind::Interior => {
                    let node = read_interior_node(&page)?;
                    let child_idx = bisect_right(&node.keys, key);
                    self.path_stack.push((page_no, child_idx));
                    page_no = node.children[child_idx];
                }
            }
        }
    }

    fn leftmost_leaf(&self) -> Result<u32> {
        let mut page_no = self.root_page_no;
        loop {
            let page = self.pager.read_page(page_no)?;
            match page_kind(&page)? {
                PageKind::Leaf => return Ok(page_no),
                PageKind::Interior => {
                    let node = read_interior_node(&page)?;
                    page_no = node.children[0];
                }
            }
        }
    }

    /// The largest key in the tree, or `None` when empty.
    ///
    /// Descends only the rightmost-child chain from the header to the rightmost
    /// leaf, then reads that leaf's last cell key. Cost is the tree height: one
    /// header read per level plus a single pointer and key decode at the leaf —
    /// no cell materialization and no sibling-chain walk. This is what keeps the
    /// insert path independent of the row count.
    pub fn max_key(&self) -> Result<Option<i64>> {
        let mut page_no = self.root_page_no;
        loop {
            let page = self.pager.read_page(page_no)?;
            match page_kind(&page)? {
                PageKind::Leaf => {
                    let hdr = read_leaf_header(&page);
                    if hdr.num_cells == 0 {
                        return Ok(None);
                    }
                    let raw = read_leaf_cell_raw(&page, hdr.num_cells - 1, OVERFLOW_THRESHOLD)?;
                    return Ok(Some(raw.key));
                }
                PageKind::Interior => {
                    page_no = read_interior_header(&page).rightmost_child;
                }
            }
        }
    }

    /// The path stack recorded by the last [`Cursor::move_to`].
    pub fn path_stack(&self) -> Vec<(u32, usize)> {
        self.path_stack.clone()
    }

    // -----------------------------------------------------------------------
    // Point lookup
    // -----------------------------------------------------------------------

    /// Return the value bytes for `key`, reassembling an overflow chain when the
    /// payload spilled. `None` if the key is absent.
    pub fn get(&mut self, key: i64) -> Result<Option<Vec<u8>>> {
        let (leaf_page_no, idx) = self.move_to(key)?;
        let page = self.pager.read_page(leaf_page_no)?;
        let hdr = read_leaf_header(&page);
        if idx >= hdr.num_cells {
            return Ok(None);
        }
        let raw = read_leaf_cell_raw(&page, idx, OVERFLOW_THRESHOLD)?;
        if raw.key != key {
            return Ok(None);
        }
        Ok(Some(self.payload_of(&page, &raw)?))
    }

    /// Materialize a leaf cell's payload, following the overflow chain if spilled.
    fn payload_of(&self, page: &[u8], raw: &LeafCellRaw) -> Result<Vec<u8>> {
        if raw.spilled {
            let first_ovf = read_spilled_pointer(page, raw.payload_start)?;
            read_overflow_chain(self.pager, first_ovf, raw.payload_len)
        } else {
            Ok(page[raw.payload_start..raw.payload_start + raw.payload_len].to_vec())
        }
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Insert or replace `(key, value)`.
    ///
    /// When the leaf has byte room (or the key already exists and is replaced in
    /// place) the cell is written in key order. A full leaf splits with copy-up
    /// and propagates the separator upward. Payloads past the threshold spill to
    /// an overflow chain.
    ///
    /// Returns `true` when the key was ADDED (a fresh cell) and `false` when an
    /// existing cell was replaced in place — the signal the store's incremental
    /// per-tree cell-count cache adjusts by.
    pub fn insert(&mut self, key: i64, value: &[u8]) -> Result<bool> {
        let (leaf_page_no, insert_idx) = self.move_to(key)?;
        let page = self.pager.read_page(leaf_page_no)?;
        let keys = read_leaf_keys(&page, OVERFLOW_THRESHOLD)?;

        // Replace path: the key already exists at insert_idx.
        if insert_idx < keys.len() && keys[insert_idx] == key {
            let old = read_leaf_cell_raw(&page, insert_idx, OVERFLOW_THRESHOLD)?;
            if old.spilled {
                let first_ovf = read_spilled_pointer(&page, old.payload_start)?;
                free_overflow_chain(self.pager, first_ovf)?;
            }
            if self.leaf_has_room_for(&page, keys.len(), Some(insert_idx), key, value)? {
                let sibling = get_sibling(&page);
                let new_page = self.rebuild_leaf(&page, keys.len(), Some(insert_idx), insert_idx, key, value, sibling)?;
                self.pager.write_page(leaf_page_no, &new_page)?;
                return Ok(false);
            }
            // Replacement too large for the page: slim out the old cell, then split.
            let sibling = get_sibling(&page);
            let slimmed = self.rebuild_leaf_excluding(&page, keys.len(), insert_idx, sibling)?;
            self.pager.write_page(leaf_page_no, &slimmed)?;
            let path = self.path_stack.clone();
            crate::btree::split::leaf_split_and_insert(
                self.pager, path, leaf_page_no, key, value,
            )?;
            return Ok(false);
        }

        // Fresh insert: fits if byte room AND below the cell ceiling.
        let fits = self.leaf_has_room_for(&page, keys.len(), None, key, value)?
            && keys.len() < leaf_max_cells();
        if fits {
            let sibling = get_sibling(&page);
            let new_page = self.rebuild_leaf(&page, keys.len(), None, insert_idx, key, value, sibling)?;
            self.pager.write_page(leaf_page_no, &new_page)?;
            return Ok(true);
        }

        // Full leaf — split.
        let path = self.path_stack.clone();
        crate::btree::split::leaf_split_and_insert(self.pager, path, leaf_page_no, key, value)?;
        Ok(true)
    }

    /// Whether a new cell fits the leaf alongside the retained existing cells.
    fn leaf_has_room_for(
        &self,
        page: &[u8],
        num_cells: usize,
        exclude_idx: Option<usize>,
        new_key: i64,
        new_value: &[u8],
    ) -> Result<bool> {
        let mut existing = 0usize;
        let mut kept = 0usize;
        for i in 0..num_cells {
            if Some(i) == exclude_idx {
                continue;
            }
            let raw = read_leaf_cell_raw(page, i, OVERFLOW_THRESHOLD)?;
            existing += on_page_cell_size(raw.key, raw.payload_len, OVERFLOW_THRESHOLD);
            kept += 1;
        }
        let new_cell = on_page_cell_size(new_key, new_value.len(), OVERFLOW_THRESHOLD);
        let n_total = kept + 1;
        let ptr_area = LEAF_PTR_ARRAY_START + n_total * 2;
        Ok(ptr_area + existing + new_cell <= USABLE_END)
    }

    /// Build the on-page bytes for a new cell, spilling to an overflow chain when
    /// the payload exceeds the threshold.
    fn build_cell(&self, key: i64, value: &[u8]) -> Result<Vec<u8>> {
        if value.len() > OVERFLOW_THRESHOLD {
            let first = write_overflow_chain(self.pager, value)?;
            Ok(encode_spilled_cell(key, value.len(), first))
        } else {
            Ok(encode_inline_cell(key, value))
        }
    }

    /// Rebuild a leaf, inserting or replacing exactly one cell. Unchanged cells
    /// are copied verbatim so overflow pointers are preserved without re-alloc.
    #[allow(clippy::too_many_arguments)]
    fn rebuild_leaf(
        &self,
        page: &[u8],
        num_cells: usize,
        replace_idx: Option<usize>,
        insert_idx: usize,
        key: i64,
        value: &[u8],
        sibling: u32,
    ) -> Result<Vec<u8>> {
        let new_cell = self.build_cell(key, value)?;
        let mut raw_cells: Vec<Vec<u8>> = Vec::with_capacity(num_cells + 1);
        match replace_idx {
            Some(r) => {
                for i in 0..num_cells {
                    if i == r {
                        raw_cells.push(new_cell.clone());
                    } else {
                        raw_cells.push(raw_leaf_cell_bytes(page, i, OVERFLOW_THRESHOLD)?);
                    }
                }
            }
            None => {
                for i in 0..num_cells {
                    raw_cells.push(raw_leaf_cell_bytes(page, i, OVERFLOW_THRESHOLD)?);
                }
                raw_cells.insert(insert_idx, new_cell);
            }
        }
        write_leaf_from_raw(&raw_cells, sibling)
    }

    /// Rebuild a leaf with the cell at `exclude_idx` removed (verbatim copy of
    /// the rest).
    fn rebuild_leaf_excluding(
        &self,
        page: &[u8],
        num_cells: usize,
        exclude_idx: usize,
        sibling: u32,
    ) -> Result<Vec<u8>> {
        let mut raw_cells: Vec<Vec<u8>> = Vec::with_capacity(num_cells);
        for i in 0..num_cells {
            if i == exclude_idx {
                continue;
            }
            raw_cells.push(raw_leaf_cell_bytes(page, i, OVERFLOW_THRESHOLD)?);
        }
        write_leaf_from_raw(&raw_cells, sibling)
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Remove `key`. Idempotent when the key is absent. After removal an
    /// underflowing non-root leaf rebalances by borrow or merge, and a zero-key
    /// interior root collapses.
    ///
    /// Returns `true` when a cell was actually removed and `false` on the
    /// absent-key no-op — the signal the store's incremental per-tree
    /// cell-count cache adjusts by.
    pub fn delete(&mut self, key: i64) -> Result<bool> {
        let (leaf_page_no, _idx) = self.move_to(key)?;
        let removed = crate::btree::delete::remove_cell_from_leaf(self.pager, leaf_page_no, key)?;
        if !removed {
            return Ok(false);
        }

        let is_root_leaf = leaf_page_no == self.root_page_no;
        if !is_root_leaf {
            let page = self.pager.read_page(leaf_page_no)?;
            let num_cells = read_leaf_header(&page).num_cells;
            let mut underflow = crate::btree::delete::leaf_underflows(&page, num_cells)?;
            if !underflow && leaf_max_cells() < SMALL_CAP_SENTINEL {
                underflow = num_cells < crate::btree::delete::leaf_min_cells();
            }
            if underflow {
                let path = self.path_stack.clone();
                crate::btree::delete::rebalance_leaf(self.pager, path, leaf_page_no)?;
            }
        }
        crate::btree::delete::shrink_root_if_needed(self.pager, self.root_page_no)?;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // Scans
    // -----------------------------------------------------------------------

    /// Every `(key, value)` pair in ascending key order.
    pub fn full_scan(&self) -> Result<Vec<(i64, Vec<u8>)>> {
        self.pager.note_full_scan();
        let mut out = Vec::new();
        let mut page_no = self.leftmost_leaf()?;
        while page_no != 0 {
            let page = self.pager.read_page(page_no)?;
            let hdr = read_leaf_header(&page);
            for i in 0..hdr.num_cells {
                let raw = read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD)?;
                let payload = self.payload_of(&page, &raw)?;
                out.push((raw.key, payload));
            }
            page_no = get_sibling(&page);
        }
        Ok(out)
    }

    /// Fetch many keys in ONE ordered leaf-chain walk.
    ///
    /// `keys` MUST be sorted ascending (duplicates are tolerated and returned
    /// once). Descends once to the leaf containing the first key, then walks
    /// the sibling chain with a two-pointer merge against the requested keys,
    /// stopping as soon as the request list is exhausted — O(pages spanned +
    /// tree height) instead of one O(height) descent PER key, which is the
    /// difference between a bulk candidate fetch and thousands of independent
    /// root-to-leaf walks. Absent keys are skipped (not an error). Uses the
    /// scan-resistant page read so a bulk fetch larger than the page cache
    /// causes no evictions.
    pub fn get_many(&mut self, keys: &[i64]) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut out: Vec<(i64, Vec<u8>)> = Vec::with_capacity(keys.len());
        if keys.is_empty() {
            return Ok(out);
        }
        // Adaptive strategy: the ordered walk wins when the requested keys
        // are DENSE relative to the leaf pages they span (many rows per page,
        // e.g. a hub adjacency fetch on a thin-row table); per-key descents
        // win when the keys are SPARSE across fat-row pages (one wide row per
        // leaf, where a chain walk would read every page between the first
        // and last key). Neither density is knowable up front, so the walk
        // carries a page budget equal to the descent-equivalent cost
        // (~3 pages per key); if the budget exhausts before the request list
        // does, the remaining keys fall back to per-key descents — worst case
        // a small constant factor over the better strategy, never the
        // pathological whole-chain read.
        let page_budget = 4usize.saturating_add(keys.len().saturating_mul(3));
        let mut pages_visited = 0usize;
        let (mut page_no, _) = self.move_to(keys[0])?;
        let mut ki = 0usize;
        while page_no != 0 && ki < keys.len() && pages_visited < page_budget {
            let page = self.pager.read_page_scan(page_no)?;
            pages_visited += 1;
            let hdr = read_leaf_header(&page);
            for i in 0..hdr.num_cells {
                if ki >= keys.len() {
                    break;
                }
                let raw = read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD)?;
                while ki < keys.len() && keys[ki] < raw.key {
                    ki += 1; // requested key absent from the tree: skip it
                }
                if ki < keys.len() && keys[ki] == raw.key {
                    let payload = self.payload_of(&page, &raw)?;
                    out.push((raw.key, payload));
                    // Advance past this key and any duplicates of it.
                    while ki < keys.len() && keys[ki] == raw.key {
                        ki += 1;
                    }
                }
            }
            page_no = get_sibling(&page);
        }
        // Budget exhausted with keys outstanding: finish them by per-key
        // descents (the sparse-key regime where the walk was the wrong tool).
        while ki < keys.len() {
            let key = keys[ki];
            if let Some(payload) = self.get(key)? {
                out.push((key, payload));
            }
            while ki < keys.len() && keys[ki] == key {
                ki += 1;
            }
        }
        Ok(out)
    }

    /// The total number of cells in the tree, counting leaf-page headers only.
    ///
    /// Walks the leftmost-leaf descent and the sibling chain reading only the
    /// leaf header's `num_cells` field per page — no cell payloads are read and
    /// no overflow chains are followed. Does NOT call `note_full_scan()`, so the
    /// full-scan counter stays at zero across a bare COUNT(*).
    pub fn count_cells(&self) -> Result<u64> {
        let first_leaf = self.leftmost_leaf()?;
        if first_leaf == 0 {
            return Ok(0);
        }
        // The pager walks the leaf sibling chain reading only each header's
        // num_cells (u16) and sibling pointer (u32). A resident page is read in
        // place (no clone); a non-resident page is read transiently and NOT
        // cached, so counting a corpus larger than the page cache makes zero
        // evictions — the walk is O(corpus), never O(corpus * cache).
        self.pager
            .count_leaf_cells(first_leaf, LEAF_NUM_CELLS_OFFSET, LEAF_SIBLING_OFFSET)
    }

    /// Stream every `(key, payload)` pair in ascending key order, applying
    /// `visitor` to each cell. Unlike [`full_scan`], this method does NOT call
    /// `note_full_scan()` and does NOT pre-materialise the entire table into a
    /// `Vec` before the first call returns — the visitor receives each cell as
    /// its leaf page is visited.
    ///
    /// The visitor's error type `E` must be convertible from [`StoreError`] so
    /// pager / overflow errors can propagate through the closure. A visitor that
    /// wants early exit returns an `Err`; the caller distinguishes its own
    /// sentinel from genuine store errors.
    ///
    /// Used by the column-selective scan paths (filtered COUNT, projected SELECT)
    /// so they avoid incrementing the full-scan counter while still reading each
    /// cell's payload bytes for column-selective decode.
    pub fn scan_cells_with<E, F>(&self, mut visitor: F) -> std::result::Result<(), E>
    where
        E: From<StoreError>,
        F: FnMut(i64, Vec<u8>) -> std::result::Result<(), E>,
    {
        let mut page_no = self.leftmost_leaf().map_err(E::from)?;
        while page_no != 0 {
            // Scan-resistant read: a leaf already resident is served from cache;
            // a miss is loaded transiently and NOT cached, so a full scan of a
            // corpus larger than the page cache makes zero evictions (no thrash).
            let page = self.pager.read_page_scan(page_no).map_err(E::from)?;
            let hdr = read_leaf_header(&page);
            for i in 0..hdr.num_cells {
                let raw = read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD).map_err(E::from)?;
                let payload = self.payload_of(&page, &raw).map_err(E::from)?;
                // One cell touched: the direct observable of streaming-scan cost.
                // A query that stops early (LIMIT satisfied) touches O(LIMIT)
                // cells; a linear column-selective sweep touches O(corpus).
                self.pager.note_cells_visited(1);
                visitor(raw.key, payload)?;
            }
            page_no = get_sibling(&page);
        }
        Ok(())
    }

    /// Every `(key, value)` pair with `lo <= key <= hi`, ascending.
    pub fn range_scan(&mut self, lo: i64, hi: i64) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut out = Vec::new();
        if lo > hi {
            return Ok(out);
        }
        let (mut page_no, _start_idx) = self.move_to(lo)?;
        while page_no != 0 {
            let page = self.pager.read_page(page_no)?;
            let hdr = read_leaf_header(&page);
            for i in 0..hdr.num_cells {
                let raw = read_leaf_cell_raw(&page, i, OVERFLOW_THRESHOLD)?;
                if raw.key < lo {
                    continue;
                }
                if raw.key > hi {
                    return Ok(out);
                }
                let payload = self.payload_of(&page, &raw)?;
                out.push((raw.key, payload));
            }
            page_no = get_sibling(&page);
        }
        Ok(out)
    }
}

/// Initialize a fresh empty leaf page as a tree root in place.
pub fn init_root_leaf(pager: &Pager, root_page_no: u32) -> Result<()> {
    pager.write_page(root_page_no, &empty_leaf())
}

/// Set the sibling link of a leaf page through the pager.
pub fn write_sibling(pager: &Pager, page_no: u32, sibling: u32) -> Result<()> {
    let mut page = pager.read_page(page_no)?;
    set_sibling(&mut page, sibling);
    pager.write_page(page_no, &page)
}

/// Read the sibling link of a leaf page through the pager.
pub fn read_sibling(pager: &Pager, page_no: u32) -> Result<u32> {
    Ok(get_sibling(&pager.read_page(page_no)?))
}

// ---------------------------------------------------------------------------
// Sorted-slice search helpers (no allocation, integer keys)
// ---------------------------------------------------------------------------

/// Leftmost index where `key` could be inserted to keep `keys` sorted.
pub fn bisect_left(keys: &[i64], key: i64) -> usize {
    keys.partition_point(|&k| k < key)
}

/// Rightmost index where `key` could be inserted to keep `keys` sorted.
pub fn bisect_right(keys: &[i64], key: i64) -> usize {
    keys.partition_point(|&k| k <= key)
}

/// Read the full payload of a spilled or inline leaf cell. Exposed for the split
/// and delete code, which must reassemble cells before redistributing them.
pub fn read_full_cell(pager: &Pager, page: &[u8], idx: usize) -> Result<(i64, Vec<u8>)> {
    let raw = read_leaf_cell_raw(page, idx, OVERFLOW_THRESHOLD)?;
    if raw.spilled {
        let first_ovf = read_spilled_pointer(page, raw.payload_start)?;
        let payload = read_overflow_chain(pager, first_ovf, raw.payload_len)?;
        Ok((raw.key, payload))
    } else {
        Ok((
            raw.key,
            page[raw.payload_start..raw.payload_start + raw.payload_len].to_vec(),
        ))
    }
}

/// Build a leaf cell's on-page bytes, spilling to an overflow chain when oversize.
pub fn build_cell_for(pager: &Pager, key: i64, value: &[u8]) -> Result<Vec<u8>> {
    if value.len() > OVERFLOW_THRESHOLD {
        let first = write_overflow_chain(pager, value)?;
        Ok(encode_spilled_cell(key, value.len(), first))
    } else {
        Ok(encode_inline_cell(key, value))
    }
}

/// Guard against a malformed read: surface a typed error rather than panicking.
pub fn ensure(cond: bool, detail: &str) -> Result<()> {
    if cond {
        Ok(())
    } else {
        Err(StoreError::Integrity {
            detail: detail.to_string(),
        })
    }
}
