//! Transaction atomicity and open-time crash recovery for both write modes.
//!
//! Covers the begin_write/commit/rollback envelope, write-ahead-mode routing,
//! and three kill-point scenarios: a crash before commit, between the journal
//! barriers, and mid-WAL-frame. After each simulated crash a fresh store reopens
//! the same path and must be consistent — committed transactions durable,
//! uncommitted gone — with a clean integrity check.

use std::fs::OpenOptions;
use lillibrain::pos_io::PosIo;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use lillibrain::consts::{HDR_WRITE_VERSION_OFFSET, PAGE_SIZE, WRITE_VERSION_WAL};
use lillibrain::journal::journal_path_for;
use lillibrain::wal::{wal_path_for, WalWriter};
use lillibrain::{Store, StoreError};

fn temp_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

fn new_tree(store: &Store) -> u32 {
    store.create_tree().unwrap()
}

// ---------------------------------------------------------------------------
// Transaction API — journal mode
// ---------------------------------------------------------------------------

#[test]
fn crash_recovery_rollback_restores_pre_transaction_state() {
    let (_d, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = new_tree(&store);

    // Seed committed baseline.
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        t.insert(1, b"one").unwrap();
        t.insert(2, b"two").unwrap();
    }
    store.commit().unwrap();
    let baseline = store.tree(root).full_scan().unwrap();

    // Mutate inside a transaction, then roll back.
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        t.insert(3, b"three").unwrap();
        t.insert(4, b"four").unwrap();
        t.delete(1).unwrap();
    }
    store.rollback().unwrap();

    let after = store.tree(root).full_scan().unwrap();
    assert_eq!(after, baseline, "rollback must restore the pre-transaction state");
    assert!(store.check_integrity(root).unwrap().is_empty());
}

#[test]
fn crash_recovery_journal_commit_survives_reopen() {
    let (_d, path) = temp_db();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = new_tree(&store);
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for k in 0..50i64 {
                t.insert(k, format!("v{k}").as_bytes()).unwrap();
            }
        }
        store.commit().unwrap();
    }
    // Reopen: the committed state is durable.
    let store = Store::open(&path).unwrap();
    let scan = store.tree(root).full_scan().unwrap();
    assert_eq!(scan.len(), 50);
    assert_eq!(scan[7], (7, b"v7".to_vec()));
    assert!(store.check_integrity(root).unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Write-ahead mode routing
// ---------------------------------------------------------------------------

#[test]
fn crash_recovery_enable_wal_flips_header_and_routes_commit() {
    let (_d, path) = temp_db();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = new_tree(&store);
        store.enable_wal_mode().unwrap();
        assert!(store.is_wal_mode());

        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for k in 0..30i64 {
                t.insert(k, format!("w{k}").as_bytes()).unwrap();
            }
        }
        store.commit().unwrap();

        // Reads inside the same session see committed-but-unmerged frames.
        let scan = store.tree(root).full_scan().unwrap();
        assert_eq!(scan.len(), 30);
    }
    // The on-disk header records WAL mode.
    let f = OpenOptions::new().read(true).open(&path).unwrap();
    let mut vbyte = [0u8; 1];
    f.read_exact_at(&mut vbyte, HDR_WRITE_VERSION_OFFSET as u64)
        .unwrap();
    assert_eq!(vbyte[0], WRITE_VERSION_WAL);

    // Reopen: open-time WAL recovery replays the sidecar; data is present.
    let store = Store::open(&path).unwrap();
    let scan = store.tree(root).full_scan().unwrap();
    assert_eq!(scan.len(), 30);
    assert_eq!(scan[5], (5, b"w5".to_vec()));
    assert!(store.check_integrity(root).unwrap().is_empty());
}

#[test]
fn crash_recovery_wal_rollback_drops_uncommitted() {
    let (_d, path) = temp_db();
    let store = Store::open(&path).unwrap();
    let root = new_tree(&store);
    store.enable_wal_mode().unwrap();

    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        t.insert(1, b"keep").unwrap();
    }
    store.commit().unwrap();

    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        t.insert(2, b"drop").unwrap();
        t.insert(3, b"drop").unwrap();
    }
    store.rollback().unwrap();

    let scan = store.tree(root).full_scan().unwrap();
    assert_eq!(scan, vec![(1, b"keep".to_vec())]);
    assert!(store.check_integrity(root).unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// Kill-point (a): crash before commit — journal mode
// ---------------------------------------------------------------------------

#[test]
fn crash_recovery_killpoint_before_commit_journal() {
    let (_d, path) = temp_db();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = new_tree(&store);
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            t.insert(1, b"committed").unwrap();
        }
        store.commit().unwrap();

        // Begin a second transaction and dirty pages, then DROP the store without
        // committing — the journal sidecar is left with page_count=0.
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            t.insert(2, b"never").unwrap();
            t.insert(3, b"never").unwrap();
        }
        // Simulate a crash: leak the open transaction by dropping the store.
        drop(store);
    }
    // An uncommitted journal may be on disk; reopen must discard it.
    let store = Store::open(&path).unwrap();
    let scan = store.tree(root).full_scan().unwrap();
    assert_eq!(scan, vec![(1, b"committed".to_vec())], "uncommitted txn gone");
    assert!(store.check_integrity(root).unwrap().is_empty());
    assert!(!journal_path_for(&path).exists());
}

// ---------------------------------------------------------------------------
// Kill-point (b): crash between the journal barriers
// ---------------------------------------------------------------------------

#[test]
fn crash_recovery_killpoint_committed_journal_replays() {
    // Hand-build a committed journal (before-images + non-zero page count) plus a
    // main file whose pages were overwritten with "after" images, simulating a
    // crash AFTER the DB pages were partly written but BEFORE the journal was
    // discarded. Open-time recovery must restore the before-images.
    let (_d, path) = temp_db();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = new_tree(&store);
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for k in 0..20i64 {
                t.insert(k, format!("b{k}").as_bytes()).unwrap();
            }
        }
        store.commit().unwrap();
    }
    let pre = {
        let store = Store::open(&path).unwrap();
        store.tree(root).full_scan().unwrap()
    };

    // Use the journal module to write a committed journal of the root page's
    // before-image, then corrupt the live root page on disk. Recovery restores it.
    use lillibrain::journal::Journal;
    let root_offset = (root as u64 - 1) * PAGE_SIZE as u64;
    let before_image = {
        let f = OpenOptions::new().read(true).open(&path).unwrap();
        let mut buf = vec![0u8; PAGE_SIZE];
        f.read_exact_at(&mut buf, root_offset).unwrap();
        buf
    };
    {
        let mut j = Journal::create(&path).unwrap();
        j.record_before_image(root, &before_image);
        j.flush_pending().unwrap();
        j.sync().unwrap();
        j.write_page_count().unwrap();
        j.sync().unwrap();
        // Now scribble an "after" image over the live root page (a partial,
        // crash-interrupted write).
        let f = OpenOptions::new().read(true).write(true).open(&path).unwrap();
        f.write_all_at(&vec![0xEE; PAGE_SIZE], root_offset).unwrap();
        f.sync_all().unwrap();
    }

    // Reopen routes through journal recovery and restores the before-image.
    let store = Store::open(&path).unwrap();
    let after = store.tree(root).full_scan().unwrap();
    assert_eq!(after, pre, "committed journal before-image restored on reopen");
    assert!(store.check_integrity(root).unwrap().is_empty());
    assert!(!journal_path_for(&path).exists());
}

// ---------------------------------------------------------------------------
// Kill-point (c): crash mid-WAL-frame
// ---------------------------------------------------------------------------

#[test]
fn crash_recovery_killpoint_torn_wal_frame() {
    let (_d, path) = temp_db();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = new_tree(&store);
        store.enable_wal_mode().unwrap();
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for k in 0..10i64 {
                t.insert(k, format!("c{k}").as_bytes()).unwrap();
            }
        }
        store.commit().unwrap();
        drop(store);
    }

    // Append a torn frame to the sidecar (a crash mid-append): a valid-looking
    // frame header truncated short of a full page.
    let wal_path = wal_path_for(&path);
    {
        let f = OpenOptions::new().read(true).write(true).open(&wal_path).unwrap();
        let len = f.metadata().unwrap().len();
        // Write a partial trailing frame (just a few bytes) to torn the tail.
        f.write_all_at(&[0u8; 16], len).unwrap();
        f.sync_all().unwrap();
    }

    // Reopen: WAL recovery replays the committed run, stops at the torn tail.
    let store = Store::open(&path).unwrap();
    let scan = store.tree(root).full_scan().unwrap();
    assert_eq!(scan.len(), 10, "committed frames replayed; torn tail ignored");
    assert!(store.check_integrity(root).unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// WAL idempotency: recovering twice == recovering once
// ---------------------------------------------------------------------------

#[test]
fn crash_recovery_wal_recovery_is_idempotent() {
    let (_d, path) = temp_db();
    let root;
    {
        let store = Store::open(&path).unwrap();
        root = new_tree(&store);
        store.enable_wal_mode().unwrap();
        store.begin_write().unwrap();
        {
            let t = store.tree(root);
            for k in 0..15i64 {
                t.insert(k, format!("i{k}").as_bytes()).unwrap();
            }
        }
        store.commit().unwrap();
        drop(store);
    }
    // First reopen runs recovery and removes the sidecar.
    let first = {
        let store = Store::open(&path).unwrap();
        store.tree(root).full_scan().unwrap()
    };
    // Re-create the same sidecar bytes by recovering once more would be a no-op
    // (sidecar already gone) — the second reopen yields identical state.
    let second = {
        let store = Store::open(&path).unwrap();
        store.tree(root).full_scan().unwrap()
    };
    assert_eq!(first, second, "replay twice == replay once");
    assert_eq!(first.len(), 15);
}

// A direct double-recovery over the same sidecar bytes must be byte-identical.
#[test]
fn crash_recovery_wal_double_replay_same_bytes() {
    let (_d, path) = temp_db();
    // Build a main file sized for pages 1..=5 and a sidecar with a committed run.
    let dbf = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&path)
        .unwrap();
    dbf.set_len((5 * PAGE_SIZE) as u64).unwrap();
    drop(dbf);

    // Build the sidecar via the writer, capture its bytes, then replay twice from
    // a fresh copy each time and compare the main-file result.
    let wal_path = wal_path_for(&path);
    {
        let mut w = WalWriter::open(&path).unwrap();
        w.commit_transaction(&[(4, vec![0xA1; PAGE_SIZE]), (5, vec![0xB2; PAGE_SIZE])], 5)
            .unwrap();
    }
    let sidecar_bytes = std::fs::read(&wal_path).unwrap();

    use lillibrain::wal::recover_wal_on_open;
    // First replay.
    recover_wal_on_open(&path, &wal_path, PAGE_SIZE).unwrap();
    let after_once = std::fs::read(&path).unwrap();

    // Re-materialize the same sidecar and replay again.
    std::fs::write(&wal_path, &sidecar_bytes).unwrap();
    recover_wal_on_open(&path, &wal_path, PAGE_SIZE).unwrap();
    let after_twice = std::fs::read(&path).unwrap();

    assert_eq!(after_once, after_twice, "double replay writes identical bytes");
}

// ---------------------------------------------------------------------------
// Lock-free read-only snapshot vs. a concurrent writer checkpoint
//
// The lock-free read-only reader's WAL overlay is a fixed open-time snapshot,
// and main-file pages it falls back to can be rewritten by a writer that
// checkpoints concurrently. Each rewritten page is individually checksum-valid,
// so the per-page checksum gate alone cannot tell that a read mixed pages from
// two checkpoint generations. The snapshot fence captured at open detects that a
// checkpoint advanced the generation and turns the window into a typed integrity
// error — the reader sees either a fully consistent snapshot or that error, never
// a silently inconsistent cross-generation B-tree.
// ---------------------------------------------------------------------------

/// Seed a write-ahead store with a multi-page tree and checkpoint it so the
/// committed state lives entirely in the main file (an empty sidecar overlay).
fn seed_checkpointed_tree(path: &std::path::Path, keys: std::ops::Range<i64>) -> u32 {
    let store = Store::open(path).unwrap();
    let root = new_tree(&store);
    store.enable_wal_mode().unwrap();
    store.begin_write().unwrap();
    {
        let t = store.tree(root);
        for k in keys {
            // Large values force the tree to span many pages, so a torn
            // cross-generation read would corrupt a scan, not just one cell.
            t.insert(k, vec![(k & 0xFF) as u8; 512].as_slice()).unwrap();
        }
    }
    store.commit().unwrap();
    store.checkpoint().unwrap();
    root
}

#[test]
fn ro_reader_happy_path_read_is_unchanged_without_a_concurrent_checkpoint() {
    let (_d, path) = temp_db();
    let root = seed_checkpointed_tree(&path, 0..200);

    // A writer keeps the store open (holding the lock) but performs no checkpoint.
    let writer = Store::open(&path).unwrap();

    // The read-only reader sees the full, consistent committed snapshot — the
    // fence never trips because no checkpoint advanced the generation.
    let reader = Store::open_read_only(&path).unwrap();
    let scan = reader.tree(root).full_scan().unwrap();
    assert_eq!(scan.len(), 200, "happy-path read returns the whole tree");
    assert_eq!(scan[7].0, 7);
    assert_eq!(scan[7].1, vec![7u8; 512]);

    drop(writer);
}

#[test]
fn ro_reader_under_concurrent_checkpoint_is_consistent_or_typed_error() {
    let (_d, path) = temp_db();
    let root = seed_checkpointed_tree(&path, 0..400);

    // The writer process keeps the store open and will checkpoint repeatedly
    // while the reader scans — rewriting main-file pages under it.
    let writer = Store::open(&path).unwrap();

    // Open the lock-free reader against the checkpointed store: its overlay is
    // empty, so every page read falls back to the main file (the windowed path).
    let reader = Store::open_read_only(&path).unwrap();

    let stop = Arc::new(AtomicBool::new(false));
    let writer_stop = stop.clone();
    let churn = std::thread::spawn(move || {
        // Repeatedly mutate and checkpoint, advancing the checkpoint generation
        // and rewriting main-file pages while the reader is live.
        let mut next = 1_000i64;
        while !writer_stop.load(Ordering::Relaxed) {
            writer.begin_write().unwrap();
            {
                let t = writer.tree(root);
                for _ in 0..32 {
                    t.insert(next, vec![(next & 0xFF) as u8; 512].as_slice())
                        .unwrap();
                    next += 1;
                }
            }
            writer.commit().unwrap();
            writer.checkpoint().unwrap();
        }
        writer
    });

    // Drive the reader through many scans while the writer churns. Each scan
    // must be either fully consistent (every value matches its key) or a typed
    // integrity error — never a silently inconsistent set of rows.
    let mut saw_typed_error = false;
    let mut saw_consistent = false;
    for _ in 0..400 {
        match reader.tree(root).full_scan() {
            Ok(rows) => {
                // A consistent snapshot: every row's value is the byte-pattern
                // its key encodes; a cross-generation mix would surface a value
                // that does not match its key, or a checksum error.
                for (k, v) in &rows {
                    assert_eq!(
                        v.as_slice(),
                        vec![(*k & 0xFF) as u8; 512].as_slice(),
                        "a returned row must be internally consistent, never a torn cross-generation read"
                    );
                }
                saw_consistent = true;
            }
            Err(StoreError::Integrity { .. }) | Err(StoreError::CrcMismatch { .. }) => {
                // The fence (or the per-page checksum) reported the window.
                saw_typed_error = true;
            }
            Err(other) => panic!("unexpected error kind: {other:?}"),
        }
    }

    stop.store(true, Ordering::Relaxed);
    let writer = churn.join().unwrap();

    // Either outcome is acceptable; the invariant is that the reader never
    // returned a silently inconsistent B-tree (the assert inside the Ok arm
    // would have fired). At least one scan must have happened.
    assert!(
        saw_consistent || saw_typed_error,
        "the reader produced neither a consistent snapshot nor a typed error"
    );

    drop(writer);
}

#[test]
fn ro_reader_returns_typed_error_when_checkpoint_advances_after_open() {
    // A deterministic demonstration of the detection: open the reader on a
    // checkpointed store (empty overlay), then advance the checkpoint generation
    // by committing and checkpointing through a writer. A main-file read of a page
    // the reader has not yet cached must report the typed integrity error rather
    // than serve a page from a generation it never snapshotted.
    let (_d, path) = temp_db();
    let root = seed_checkpointed_tree(&path, 0..400);

    let writer = Store::open(&path).unwrap();
    let reader = Store::open_read_only(&path).unwrap();

    // The reader's db_size captured at open (from cached page 1), so we can pick a
    // page that exists in its snapshot but is not yet cached. Page `root` is the
    // tree root; an interior/leaf page just past it is in-bounds and uncached.
    let snapshot_pages = reader.db_size().unwrap();
    assert!(snapshot_pages > root, "the seeded tree must span several pages");
    // A page within the snapshot extent that the reader has not read yet (only
    // page 1 has been touched, by open). Use the root page: it is in-bounds and
    // not yet cached, so reading it falls back to the main file.
    let uncached_page = root;
    assert!(!reader.pager().is_cached(uncached_page), "the chosen page must be uncached");

    // The writer advances the checkpoint generation, rewriting main-file pages.
    for batch in 0..4 {
        writer.begin_write().unwrap();
        {
            let t = writer.tree(root);
            for i in 0..64i64 {
                let k = 10_000 + batch * 64 + i;
                t.insert(k, vec![(k & 0xFF) as u8; 512].as_slice()).unwrap();
            }
        }
        writer.commit().unwrap();
        writer.checkpoint().unwrap();
    }

    // A fresh reader opened AFTER the generation advanced sees the new snapshot
    // consistently (its fence matches the current generation) — the fix does not
    // make a correctly-opened reader fail.
    let fresh = Store::open_read_only(&path).unwrap();
    let fresh_scan = fresh.tree(root).full_scan().unwrap();
    assert!(
        fresh_scan.len() >= 400,
        "a freshly opened reader sees a consistent, larger snapshot"
    );

    // The STALE reader, whose fence was captured before the checkpoints, must
    // report a typed error when it reads an uncached main-file page that the
    // checkpoint rewrote — never a torn cross-generation read.
    let err = reader.pager().read_page(uncached_page).unwrap_err();
    match err {
        // SnapshotFence is the specific variant the pager raises for this case;
        // Integrity is kept because the fence detail is the same contract and a
        // caller matching the broader variant is still correct.
        StoreError::SnapshotFence { detail } | StoreError::Integrity { detail } => {
            assert!(
                detail.contains("read-only snapshot invalidated"),
                "expected the snapshot-fence integrity error, got: {detail}"
            );
        }
        StoreError::CrcMismatch { .. } => {
            // Acceptable: the per-page gate caught a torn page first.
        }
        other => panic!("expected a typed integrity/checksum error, got {other:?}"),
    }

    drop(writer);
}
