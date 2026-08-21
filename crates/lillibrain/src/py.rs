//! Thin record-level Python surface over the store.
//!
//! Exposes the paged record store to a host runtime for storage-level
//! differential testing: open a store on a path, create a tree, and drive
//! insert / get / delete / max_key / scans / integrity checks against it.
//!
//! Values cross the boundary as opaque byte strings — already-sealed from the
//! caller's view. The store is blind to the meaning of the bytes it holds, and
//! this surface takes no secret-material parameter of any kind.

use std::sync::{Mutex, MutexGuard};

use pyo3::import_exception;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyList, PyModule};

use crate::store::Store;

import_exception!(iai_mcp.errors, DatabaseError);
import_exception!(iai_mcp.errors, OperationalError);

/// Map a storage-engine error onto a Python exception by fault class.
///
/// A transient I/O fault maps to `OSError` and a transient snapshot fence to
/// `iai_mcp.errors.OperationalError` (both recoverable); every corruption class
/// (checksum mismatch, freelist corruption, integrity violation, page out of
/// bounds) maps to the bare `iai_mcp.errors.DatabaseError` so a caller can
/// branch on disk damage instead of seeing an indistinguishable `RuntimeError`.
fn to_pyerr(e: crate::error::StoreError) -> PyErr {
    use crate::error::StoreError as S;
    let msg = e.to_string();
    match e {
        S::Io(_) => pyo3::exceptions::PyOSError::new_err(msg),
        S::SnapshotFence { .. } => OperationalError::new_err(msg),
        S::CrcMismatch { .. }
        | S::FreelistCorruption { .. }
        | S::Integrity { .. }
        | S::PageOutOfBounds { .. } => DatabaseError::new_err(msg),
    }
}

/// Lock a mutex, failing closed if a prior operation panicked while holding it.
///
/// A panic across the boundary poisons the guarded state. Recovering the guard
/// would resume on a store left in an indeterminate mid-write condition; instead
/// every later call returns a permanent error directing the caller to reopen the
/// store. The guarded state is never handed back from a poisoned lock.
fn lock_fail_closed<T>(m: &Mutex<T>) -> PyResult<MutexGuard<'_, T>> {
    m.lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err(
            "store poisoned: a prior operation panicked; reopen the store",
        )
    })
}

/// A handle to a paged record store.
///
/// Wraps the native store behind a mutex so a host runtime can hold the handle
/// across threads. Tree access is by root page number, mirroring the native
/// `Store::create_tree` / `Store::tree` seam.
#[pyclass(name = "Store", module = "iai_mcp_native.store")]
pub struct PyStore {
    inner: Mutex<Store>,
}

impl PyStore {
    /// Lock the inner store, failing closed if a prior call panicked while
    /// holding it. A poisoned mutex means a previous operation unwound across the
    /// boundary mid-write; rather than resume on indeterminate state, every later
    /// call returns a permanent error telling the caller to reopen the store.
    fn locked(&self) -> PyResult<MutexGuard<'_, Store>> {
        lock_fail_closed(&self.inner)
    }
}

#[pymethods]
impl PyStore {
    /// Open or create the store at `path`.
    #[staticmethod]
    fn open(path: &str) -> PyResult<PyStore> {
        let store = Store::open(path).map_err(to_pyerr)?;
        Ok(PyStore {
            inner: Mutex::new(store),
        })
    }

    /// Create a new empty tree and return its fixed root page number.
    fn create_tree(&self) -> PyResult<u32> {
        let store = self.locked()?;
        store.create_tree().map_err(to_pyerr)
    }

    /// Switch the store to write-ahead mode (the production write path).
    fn enable_wal_mode(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.enable_wal_mode().map_err(to_pyerr)
    }

    /// True when the store runs in write-ahead mode.
    fn is_wal_mode(&self) -> PyResult<bool> {
        let store = self.locked()?;
        Ok(store.is_wal_mode())
    }

    /// Begin a write transaction.
    fn begin_write(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.begin_write().map_err(to_pyerr)
    }

    /// Commit the open transaction durably.
    fn commit(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.commit().map_err(to_pyerr)
    }

    /// Abort the open transaction, restoring the pre-transaction state.
    fn rollback(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.rollback().map_err(to_pyerr)
    }

    /// Merge committed write-ahead frames into the main file.
    fn checkpoint(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.checkpoint().map_err(to_pyerr)
    }

    /// Set the page-cache budget in pages (`None` is unbounded).
    #[pyo3(signature = (pages=None))]
    fn set_cache_pages(&self, pages: Option<usize>) -> PyResult<()> {
        let store = self.locked()?;
        store.set_cache_pages(pages);
        Ok(())
    }

    /// Insert or replace `(key, value)` in the tree rooted at `tree_root`.
    ///
    /// `value` is stored verbatim. It crosses as opaque bytes; the store never
    /// inspects, normalizes, or transforms it.
    fn insert(&self, tree_root: u32, key: i64, value: &[u8]) -> PyResult<()> {
        let store = self.locked()?;
        store.tree(tree_root).insert(key, value).map_err(to_pyerr)
    }

    /// The value bytes for `key`, or `None` when absent.
    fn get<'py>(
        &self,
        py: Python<'py>,
        tree_root: u32,
        key: i64,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let store = self.locked()?;
        let value = store.tree(tree_root).get(key).map_err(to_pyerr)?;
        Ok(value.map(|bytes| PyBytes::new(py, &bytes)))
    }

    /// Remove `key`. Idempotent when the key is absent.
    fn delete(&self, tree_root: u32, key: i64) -> PyResult<()> {
        let store = self.locked()?;
        store.tree(tree_root).delete(key).map_err(to_pyerr)
    }

    /// The largest key in the tree, or `None` when empty. Cost is tree height.
    fn max_key(&self, tree_root: u32) -> PyResult<Option<i64>> {
        let store = self.locked()?;
        store.tree(tree_root).max_key().map_err(to_pyerr)
    }

    /// Every `(key, value)` pair in ascending key order.
    fn full_scan<'py>(&self, py: Python<'py>, tree_root: u32) -> PyResult<Bound<'py, PyList>> {
        let store = self.locked()?;
        let pairs = store.tree(tree_root).full_scan().map_err(to_pyerr)?;
        pairs_to_pylist(py, pairs)
    }

    /// Every `(key, value)` pair with `lo <= key <= hi`, ascending.
    fn range_scan<'py>(
        &self,
        py: Python<'py>,
        tree_root: u32,
        lo: i64,
        hi: i64,
    ) -> PyResult<Bound<'py, PyList>> {
        let store = self.locked()?;
        let pairs = store.tree(tree_root).range_scan(lo, hi).map_err(to_pyerr)?;
        pairs_to_pylist(py, pairs)
    }

    /// Structural invariants of the tree rooted at `tree_root`. Empty when clean.
    fn check_integrity<'py>(
        &self,
        py: Python<'py>,
        tree_root: u32,
    ) -> PyResult<Bound<'py, PyList>> {
        let store = self.locked()?;
        let problems = store.check_integrity(tree_root).map_err(to_pyerr)?;
        PyList::new(py, problems)
    }

    /// The total page count in the store file.
    fn db_size(&self) -> PyResult<u32> {
        let store = self.locked()?;
        store.db_size().map_err(to_pyerr)
    }

    /// The number of pages currently on the freelist.
    fn free_page_count(&self) -> PyResult<u32> {
        let store = self.locked()?;
        store.free_page_count().map_err(to_pyerr)
    }

    /// The number of page reads since the last reset. Lets a cost assertion show
    /// per-insert reads track tree height, not row count.
    fn read_count(&self) -> PyResult<u64> {
        let store = self.locked()?;
        Ok(store.read_count())
    }

    /// Reset the page-read counter to zero.
    fn reset_read_count(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.reset_read_count();
        Ok(())
    }

    /// The number of whole-tree ascending walks since the last reset. Stays at
    /// zero across the insert path.
    fn full_scan_count(&self) -> PyResult<u64> {
        let store = self.locked()?;
        Ok(store.full_scan_count())
    }

    /// Reset the full-scan counter to zero.
    fn reset_full_scan_count(&self) -> PyResult<()> {
        let store = self.locked()?;
        store.reset_full_scan_count();
        Ok(())
    }
}

/// Marshal `(i64, bytes)` pairs into a Python list of `(int, bytes)` tuples.
fn pairs_to_pylist<'py>(
    py: Python<'py>,
    pairs: Vec<(i64, Vec<u8>)>,
) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<(i64, Bound<'py, PyBytes>)> = pairs
        .into_iter()
        .map(|(key, value)| (key, PyBytes::new(py, &value)))
        .collect();
    PyList::new(py, items)
}

/// Register the record-store surface on a host module.
///
/// Mirrors the other native cores: the wrapper crate calls this from its
/// module body to mount the store as a sub-module.
pub fn register(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStore>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn temp_store() -> (tempfile::TempDir, Store) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.lilli");
        let store = Store::open(&path).unwrap();
        (dir, store)
    }

    #[test]
    fn unpoisoned_lock_hands_back_the_guard() {
        let (_d, store) = temp_store();
        let m = Mutex::new(store);
        // A healthy mutex yields the guard so a normal operation proceeds.
        assert!(lock_fail_closed(&m).is_ok());
    }

    #[test]
    fn poisoned_lock_fails_closed_instead_of_recovering() {
        let (_d, store) = temp_store();
        let m = Arc::new(Mutex::new(store));

        // Poison the mutex by panicking while a thread holds the guard.
        let m2 = Arc::clone(&m);
        let _ = std::thread::spawn(move || {
            let _guard = m2.lock().unwrap();
            panic!("simulated mid-operation panic across the boundary");
        })
        .join();

        // The next lock must fail closed (a permanent reopen error), never recover
        // the guard and resume on the indeterminate store state.
        let result = lock_fail_closed(&m);
        assert!(
            result.is_err(),
            "a poisoned store mutex must fail closed, not recover the guard"
        );
    }
}
