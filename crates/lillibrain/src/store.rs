//! The record store: a single paged file holding one or more B-trees, each a
//! key->value index keyed by `i64` over opaque byte values.
//!
//! `Store` owns the pager; `Tree` is a handle to one B-tree rooted at a fixed
//! page. Mutations persist via a pager flush so they survive a reopen. The
//! name->root catalog lives above this seam; the store exposes "create a tree"
//! and "open a tree by root page".

use std::collections::HashMap;
use std::path::Path;
use std::sync::Mutex;

use crate::btree::cursor::{init_root_leaf, Cursor};
use crate::error::Result;
use crate::pager::Pager;

/// A paged record store.
pub struct Store {
    pager: Pager,
    /// Per-tree (keyed by root page) cell-count cache, maintained
    /// incrementally by [`Tree::insert`] / [`Tree::delete`] (which report
    /// whether a key was actually added / removed) and populated lazily by
    /// the first [`Tree::count_cells`] walk. Keeps a bare row count O(1) warm
    /// regardless of write traffic — the leaf-chain walk is otherwise
    /// O(pages) per count, a real per-read cost at production scale.
    cell_counts: Mutex<HashMap<u32, u64>>,
    /// Snapshot of `cell_counts` taken at [`Store::begin_write`], restored on
    /// [`Store::rollback`] so an aborted transaction's incremental
    /// adjustments never survive it. Dropped on commit.
    cell_counts_txn_snapshot: Mutex<Option<HashMap<u32, u64>>>,
}

impl Store {
    fn from_pager(pager: Pager) -> Store {
        Store {
            pager,
            cell_counts: Mutex::new(HashMap::new()),
            cell_counts_txn_snapshot: Mutex::new(None),
        }
    }

    /// Open or create the store file at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Store> {
        Ok(Store::from_pager(Pager::open(path)?))
    }

    /// Open an existing store file for lock-free read-only access.
    ///
    /// Takes no advisory lock and never writes, so a separate reader process can
    /// read committed pages while a writer process holds the store's exclusive
    /// lock. The caller must read a checkpointed store (no pending WAL).
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Store> {
        Ok(Store::from_pager(Pager::open_read_only(path)?))
    }

    /// Re-arm a read-only store's committed snapshot in place (see
    /// [`Pager::refresh_ro_snapshot`]). Cell counts are dropped when the
    /// snapshot moved — they were exact for the previous snapshot only and
    /// repopulate lazily on the next count walk.
    pub fn refresh_read_only(&mut self) -> Result<bool> {
        let changed = self.pager.refresh_ro_snapshot()?;
        if changed {
            // A poisoned lock must still clear: skipping would serve counts
            // from the previous snapshot as if they were exact.
            self.cell_counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clear();
        }
        Ok(changed)
    }

    /// Adjust the cached cell count for `root` by `delta`, when a cached
    /// entry exists (lazy population is owned by `count_cells`).
    fn adjust_cell_count(&self, root: u32, delta: i64) {
        if let Ok(mut counts) = self.cell_counts.lock() {
            if let Some(c) = counts.get_mut(&root) {
                *c = c.saturating_add_signed(delta);
            }
        }
    }

    /// Create a new empty B-tree and return its root page number.
    ///
    /// The root is a fresh leaf page allocated by extending the file; its page
    /// number is fixed for the life of the tree.
    pub fn create_tree(&self) -> Result<u32> {
        let root = self.pager.extend_file()?;
        init_root_leaf(&self.pager, root)?;
        self.pager.flush()?;
        Ok(root)
    }

    /// A handle to the tree rooted at `root_page_no`.
    pub fn tree(&self, root_page_no: u32) -> Tree<'_> {
        Tree {
            store: self,
            root_page_no,
        }
    }

    /// Set the page-cache budget (`None` is unbounded).
    pub fn set_cache_pages(&self, n: Option<usize>) {
        self.pager.set_cache_pages(n);
    }

    /// Flush every dirty page durably.
    pub fn flush(&self) -> Result<()> {
        self.pager.flush()
    }

    /// Release the advisory single-writer lock held on the store file, so a
    /// fresh open on the same path in the same process succeeds without waiting
    /// for this handle to be dropped.
    pub fn unlock(&self) -> Result<()> {
        self.pager.unlock()
    }

    /// True while the store file handle is still held — `false` once `unlock()`
    /// has released the file. A released store can serve no further I/O.
    pub fn is_open(&self) -> bool {
        self.pager.is_open()
    }

    /// Switch the store to write-ahead mode (the production write path: one
    /// fsync per commit). Idempotent; the mode persists across a reopen.
    pub fn enable_wal_mode(&self) -> Result<()> {
        self.pager.enable_wal_mode()
    }

    /// True when the store runs in write-ahead mode.
    pub fn is_wal_mode(&self) -> bool {
        self.pager.is_wal_mode()
    }

    /// Begin a write transaction. Tree mutations issued before [`Store::commit`]
    /// land atomically: a crash before commit leaves the prior consistent state.
    pub fn begin_write(&self) -> Result<()> {
        // Snapshot the count cache so a rollback can restore it alongside the
        // pager state (the incremental adjustments inside the transaction
        // must not survive an abort).
        if let (Ok(counts), Ok(mut snap)) = (
            self.cell_counts.lock(),
            self.cell_counts_txn_snapshot.lock(),
        ) {
            *snap = Some(counts.clone());
        }
        self.pager.begin_write()
    }

    /// Commit the open transaction (WAL append + one fsync, or the journal
    /// three-fsync barrier).
    pub fn commit(&self) -> Result<()> {
        let result = self.pager.commit();
        if result.is_ok() {
            if let Ok(mut snap) = self.cell_counts_txn_snapshot.lock() {
                *snap = None;
            }
        }
        result
    }

    /// Abort the open transaction, restoring the pre-transaction state.
    pub fn rollback(&self) -> Result<()> {
        // Restore the count cache to its pre-transaction snapshot; when no
        // snapshot exists (a rollback without a begin), drop the cache
        // entirely — a stale count must never survive an ambiguous abort.
        if let (Ok(mut counts), Ok(mut snap)) = (
            self.cell_counts.lock(),
            self.cell_counts_txn_snapshot.lock(),
        ) {
            match snap.take() {
                Some(restored) => *counts = restored,
                None => counts.clear(),
            }
        }
        self.pager.rollback()
    }

    /// Merge committed WAL frames into the main file (WAL mode only).
    pub fn checkpoint(&self) -> Result<()> {
        self.pager.checkpoint()
    }

    /// The total page count in the store file.
    pub fn db_size(&self) -> Result<u32> {
        self.pager.db_size()
    }

    /// The number of pages currently on the freelist.
    pub fn free_page_count(&self) -> Result<u32> {
        self.pager.freelist_count()
    }

    /// Verify the structural invariants of the tree rooted at `root_page_no`.
    /// Returns an empty list when clean.
    ///
    /// Pages 2 and 3 are structural reservations from file creation (a root slot
    /// and a meta page); when they are not the active tree root they are exempt
    /// from the orphan-page check.
    pub fn check_integrity(&self, root_page_no: u32) -> Result<Vec<String>> {
        let reserved: Vec<u32> = [2u32, 3u32]
            .into_iter()
            .filter(|&p| p != root_page_no)
            .collect();
        crate::check::check_integrity_with_reserved(&self.pager, root_page_no, &reserved)
    }

    /// The underlying pager (for structural checks and tests).
    pub fn pager(&self) -> &Pager {
        &self.pager
    }

    /// The store file path this store was opened against.
    pub fn path(&self) -> &Path {
        self.pager.path()
    }

    /// The number of page reads since the last reset. Surfaced so a cost
    /// assertion can show per-insert reads track tree height, not row count.
    pub fn read_count(&self) -> u64 {
        self.pager.read_count()
    }

    /// Reset the page-read counter to zero.
    pub fn reset_read_count(&self) {
        self.pager.reset_read_count();
    }

    /// The number of whole-tree ascending walks since the last reset. Stays at
    /// zero across the insert path — the linear-sweep regression would lift it.
    pub fn full_scan_count(&self) -> u64 {
        self.pager.full_scan_count()
    }

    /// Reset the full-scan counter to zero.
    pub fn reset_full_scan_count(&self) {
        self.pager.reset_full_scan_count();
    }

    /// The number of leaf cells visited by streaming scans
    /// ([`Tree::scan_cells_with`]) since the last reset. A query served by an
    /// index (early-stopping at LIMIT) touches O(LIMIT) cells; a linear
    /// column-selective sweep touches O(corpus).
    pub fn cells_visited_count(&self) -> u64 {
        self.pager.cells_visited_count()
    }

    /// Reset the cells-visited counter to zero.
    pub fn reset_cells_visited_count(&self) {
        self.pager.reset_cells_visited_count();
    }
}

/// A handle to one B-tree within a [`Store`].
pub struct Tree<'s> {
    store: &'s Store,
    root_page_no: u32,
}

impl Tree<'_> {
    /// The fixed root page number of this tree.
    pub fn root_page_no(&self) -> u32 {
        self.root_page_no
    }

    fn cursor(&self) -> Cursor<'_> {
        Cursor::new(&self.store.pager, self.root_page_no)
    }

    /// Insert or replace `(key, value)`. The insert path descends in tree height
    /// and never scans the whole tree.
    pub fn insert(&self, key: i64, value: &[u8]) -> Result<()> {
        let mut cursor = self.cursor();
        let added = cursor.insert(key, value)?;
        if added {
            self.store.adjust_cell_count(self.root_page_no, 1);
        }
        self.store.pager.flush()
    }

    /// The value bytes for `key`, or `None` when absent.
    pub fn get(&self, key: i64) -> Result<Option<Vec<u8>>> {
        let mut cursor = self.cursor();
        cursor.get(key)
    }

    /// Fetch many keys (sorted ascending, duplicates tolerated) in one
    /// ordered leaf-chain walk — O(pages spanned + height) instead of one
    /// O(height) descent per key. Absent keys are skipped.
    pub fn get_many(&self, keys: &[i64]) -> Result<Vec<(i64, Vec<u8>)>> {
        let mut cursor = self.cursor();
        cursor.get_many(keys)
    }

    /// Remove `key`. Idempotent when the key is absent.
    pub fn delete(&self, key: i64) -> Result<()> {
        let mut cursor = self.cursor();
        let removed = cursor.delete(key)?;
        if removed {
            self.store.adjust_cell_count(self.root_page_no, -1);
        }
        self.store.pager.flush()
    }

    /// The largest key in the tree, or `None` when empty. Cost is tree height.
    pub fn max_key(&self) -> Result<Option<i64>> {
        self.cursor().max_key()
    }

    /// Every `(key, value)` pair in ascending key order.
    pub fn full_scan(&self) -> Result<Vec<(i64, Vec<u8>)>> {
        self.cursor().full_scan()
    }

    /// The total number of rows in the tree, reading leaf-page headers only.
    ///
    /// Served O(1) from the store's per-tree count cache when warm; the first
    /// call per tree walks the leaf sibling chain summing `num_cells` per page
    /// (no cell payloads) and populates the cache, which insert/delete then
    /// maintain incrementally. Does NOT increment the full-scan counter, so
    /// a bare `COUNT(*)` issued through this method stays scan-free. The count
    /// equals `full_scan().len()` for every tree shape including trees with
    /// overflow cells (the header `num_cells` counts the cell on the leaf page,
    /// not the overflow continuation pages).
    pub fn count_cells(&self) -> Result<u64> {
        if let Ok(counts) = self.store.cell_counts.lock() {
            if let Some(&c) = counts.get(&self.root_page_no) {
                return Ok(c);
            }
        }
        let walked = self.cursor().count_cells()?;
        if let Ok(mut counts) = self.store.cell_counts.lock() {
            counts.insert(self.root_page_no, walked);
        }
        Ok(walked)
    }

    /// Stream every `(key, payload)` pair in ascending key order, calling
    /// `visitor` for each cell without pre-materialising the whole table.
    ///
    /// Does NOT call `note_full_scan()`, so the full-scan counter stays at zero
    /// across column-selective scans (filtered COUNT, projected SELECT). Every
    /// payload is byte-identical to what `full_scan()` returns. The visitor's
    /// error type `E` must be convertible from [`StoreError`] so pager errors
    /// propagate; a visitor that returns `Err` stops the scan early.
    pub fn scan_cells_with<E, F>(&self, visitor: F) -> std::result::Result<(), E>
    where
        E: From<crate::error::StoreError>,
        F: FnMut(i64, Vec<u8>) -> std::result::Result<(), E>,
    {
        self.cursor().scan_cells_with(visitor)
    }

    /// Every `(key, value)` pair with `lo <= key <= hi`, ascending.
    pub fn range_scan(&self, lo: i64, hi: i64) -> Result<Vec<(i64, Vec<u8>)>> {
        self.cursor().range_scan(lo, hi)
    }
}
