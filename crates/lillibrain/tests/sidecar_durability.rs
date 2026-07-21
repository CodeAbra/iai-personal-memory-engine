//! Sidecar durability contract: journal/WAL creation surfaces a typed error
//! when the containing directory is unusable, and the journal-mode commit
//! point surfaces unlink failures instead of reporting a false success.

use std::path::PathBuf;

use lillibrain::consts::PAGE_SIZE;
use lillibrain::journal::{journal_path_for, Journal};
use lillibrain::pager::Pager;
use lillibrain::wal::WalWriter;
use lillibrain::StoreError;

fn temp_db() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("brain.lilli");
    (dir, path)
}

/// A path whose parent directory has been removed.
fn orphan_db_path() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let sub = dir.path().join("gone");
    std::fs::create_dir(&sub).unwrap();
    std::fs::remove_dir(&sub).unwrap();
    (dir, sub.join("brain.lilli"))
}

#[test]
fn journal_create_errors_when_parent_dir_missing() {
    let (_d, db) = orphan_db_path();
    match Journal::create(&db) {
        Err(StoreError::Io(_)) => {}
        Err(other) => panic!("expected StoreError::Io, got {other}"),
        Ok(_) => panic!("journal creation must fail without a parent directory"),
    }
}

#[test]
fn wal_open_errors_when_parent_dir_missing() {
    let (_d, db) = orphan_db_path();
    match WalWriter::open(&db) {
        Err(StoreError::Io(_)) => {}
        Err(other) => panic!("expected StoreError::Io, got {other}"),
        Ok(_) => panic!("wal open must fail without a parent directory"),
    }
}

#[test]
fn journal_create_and_wal_open_succeed_in_valid_dir() {
    let (_d, db) = temp_db();
    let j = Journal::create(&db).unwrap();
    assert!(j.path().exists());
    let w = WalWriter::open(&db).unwrap();
    assert!(w.path().exists());
}

#[test]
fn commit_tolerates_journal_already_removed() {
    let (_d, db) = temp_db();
    let p = Pager::open(&db).unwrap();
    p.begin_write().unwrap();
    let pno = p.extend_file().unwrap();
    p.write_page(pno, &vec![0x5Au8; PAGE_SIZE]).unwrap();
    std::fs::remove_file(journal_path_for(&db)).unwrap();
    p.commit().unwrap();
}

#[test]
fn commit_fails_when_journal_unlink_fails() {
    let (_d, db) = temp_db();
    let p = Pager::open(&db).unwrap();
    p.begin_write().unwrap();
    let pno = p.extend_file().unwrap();
    p.write_page(pno, &vec![0xA5u8; PAGE_SIZE]).unwrap();

    // Replace the journal path with a directory: unlinking a directory fails
    // on every supported platform, forcing the commit-point unlink error.
    let jp = journal_path_for(&db);
    std::fs::remove_file(&jp).unwrap();
    std::fs::create_dir(&jp).unwrap();

    let err = p.commit().unwrap_err();
    assert!(matches!(err, StoreError::Io(_)));
    std::fs::remove_dir(&jp).unwrap();
}
