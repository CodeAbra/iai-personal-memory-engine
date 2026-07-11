//! Rollback journal: before-images of every page a transaction dirties, plus the
//! three-fsync commit barrier.
//!
//! A `.journal` sidecar holds the magic, a page count (zero until a transaction
//! commits), and the before-image of each dirtied page. The header page count is
//! written zero by [`Journal::create`] so a crash before the commit barrier
//! leaves an uncommitted journal that open-time recovery discards. The real
//! count is stamped only after the before-images are durable, which is what arms
//! recovery to replay.
//!
//! Journal layout:
//! ```text
//!   [0:12]   magic
//!   [12:16]  page_count (big-endian u32; 0 = uncommitted sentinel)
//!   [16:]    page_count records of: u32 page_no + PAGE_SIZE page bytes
//! ```

use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::consts::PAGE_SIZE;
use crate::error::Result;

/// Journal magic: exactly 12 bytes.
pub const JOURNAL_MAGIC: [u8; 12] = *b"LILLI-JRNL\x00\x00";
/// Journal header size: magic (12) + page_count (4).
pub const JOURNAL_HEADER_SIZE: usize = 16;

/// The `.journal` sidecar path for a database file.
pub fn journal_path_for(db_path: &Path) -> PathBuf {
    let mut name = db_path.file_name().unwrap_or_default().to_os_string();
    name.push(".journal");
    db_path.with_file_name(name)
}

/// A rollback journal sidecar for one in-progress write transaction.
pub struct Journal {
    path: PathBuf,
    file: File,
    /// Page numbers whose before-image is already captured (first snapshot only).
    journaled: std::collections::HashSet<u32>,
    /// Buffered before-images in capture order, flushed to disk at commit.
    pending: Vec<(u32, Vec<u8>)>,
}

impl Journal {
    /// Create (truncating) the journal sidecar with the uncommitted sentinel header.
    pub fn create(db_path: &Path) -> Result<Journal> {
        let path = journal_path_for(db_path);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, perms);
        }

        let mut header = Vec::with_capacity(JOURNAL_HEADER_SIZE);
        header.extend_from_slice(&JOURNAL_MAGIC);
        header.extend_from_slice(&0u32.to_be_bytes());
        file.write_all_at(&header, 0)?;

        Ok(Journal {
            path,
            file,
            journaled: std::collections::HashSet::new(),
            pending: Vec::new(),
        })
    }

    /// The journal sidecar path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Buffer the before-image of `page_no`. Only the first snapshot per
    /// transaction is kept — a later snapshot would record an already-mutated
    /// page and recover to the wrong intermediate state.
    pub fn record_before_image(&mut self, page_no: u32, data: &[u8]) {
        if self.journaled.contains(&page_no) {
            return;
        }
        self.journaled.insert(page_no);
        self.pending.push((page_no, data.to_vec()));
    }

    /// Number of before-images captured so far.
    pub fn page_count(&self) -> u32 {
        self.pending.len() as u32
    }

    /// The before-images captured, in capture order.
    pub fn before_images(&self) -> &[(u32, Vec<u8>)] {
        &self.pending
    }

    /// Write all buffered before-images contiguously after the header in one pass.
    pub fn flush_pending(&mut self) -> Result<()> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let record_size = 4 + PAGE_SIZE;
        let mut buf = vec![0u8; self.pending.len() * record_size];
        let mut off = 0;
        for (page_no, data) in &self.pending {
            buf[off..off + 4].copy_from_slice(&page_no.to_be_bytes());
            off += 4;
            buf[off..off + PAGE_SIZE].copy_from_slice(data);
            off += PAGE_SIZE;
        }
        self.file.write_all_at(&buf, JOURNAL_HEADER_SIZE as u64)?;
        Ok(())
    }

    /// Durably sync the journal file.
    pub fn sync(&self) -> Result<()> {
        crate::io::durable_fsync(&self.file)
    }

    /// Stamp the real page count into the header (arming recovery).
    pub fn write_page_count(&self) -> Result<()> {
        let count = self.page_count();
        self.file
            .write_all_at(&count.to_be_bytes(), JOURNAL_MAGIC.len() as u64)?;
        Ok(())
    }
}

/// Replay a committed `.journal` into the main file, or discard an uncommitted
/// one. Routes on the header magic and page count: a count of zero (or a missing
/// magic) means the transaction never reached its barrier, so the journal is
/// discarded; a non-zero count replays each before-image into the main file,
/// then removes the journal — the commit point of recovery.
pub fn recover_journal_on_open(db_path: &Path, journal_path: &Path) -> Result<()> {
    if !journal_path.exists() {
        return Ok(());
    }
    let data = match std::fs::read(journal_path) {
        Ok(d) => d,
        Err(_) => return Ok(()),
    };
    if data.len() < JOURNAL_HEADER_SIZE {
        let _ = std::fs::remove_file(journal_path);
        return Ok(());
    }
    if data[..JOURNAL_MAGIC.len()] != JOURNAL_MAGIC {
        let _ = std::fs::remove_file(journal_path);
        return Ok(());
    }
    let page_count = u32::from_be_bytes(data[12..16].try_into().unwrap());
    if page_count == 0 {
        // Uncommitted transaction — discard.
        let _ = std::fs::remove_file(journal_path);
        return Ok(());
    }

    let db_file = OpenOptions::new().read(true).write(true).open(db_path)?;
    let mut pos = JOURNAL_HEADER_SIZE;
    let record_size = 4 + PAGE_SIZE;
    for _ in 0..page_count {
        if pos + record_size > data.len() {
            break; // truncated — stop replaying
        }
        let page_no = u32::from_be_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;
        let page_data = &data[pos..pos + PAGE_SIZE];
        pos += PAGE_SIZE;
        let offset = (page_no as u64 - 1) * PAGE_SIZE as u64;
        db_file.write_all_at(page_data, offset)?;
    }
    crate::io::durable_fsync(&db_file)?;

    // Removing the journal is the commit point of recovery.
    let _ = std::fs::remove_file(journal_path);
    if let Some(parent) = journal_path.parent() {
        let _ = crate::io::fsync_directory(parent);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.lilli");
        File::create(&path).unwrap();
        (dir, path)
    }

    #[test]
    fn journal_uncommitted_is_discarded_on_recovery() {
        let (_d, db) = temp_db();
        let mut j = Journal::create(&db).unwrap();
        j.record_before_image(2, &vec![0u8; PAGE_SIZE]);
        j.flush_pending().unwrap();
        // page_count header still zero — recovery must discard.
        let jp = j.path().to_path_buf();
        drop(j);
        recover_journal_on_open(&db, &jp).unwrap();
        assert!(!jp.exists());
    }

    #[test]
    fn journal_committed_replays_before_images() {
        let (_d, db) = temp_db();
        let dbf = OpenOptions::new().read(true).write(true).open(&db).unwrap();
        dbf.set_len((3 * PAGE_SIZE) as u64).unwrap();
        // Put a known "after" image on disk that recovery must overwrite.
        dbf.write_all_at(&vec![0xFF; PAGE_SIZE], PAGE_SIZE as u64)
            .unwrap();
        drop(dbf);

        let jp = {
            let mut j = Journal::create(&db).unwrap();
            j.record_before_image(2, &vec![0x42; PAGE_SIZE]);
            j.flush_pending().unwrap();
            j.sync().unwrap();
            j.write_page_count().unwrap();
            j.sync().unwrap();
            j.path().to_path_buf()
        };

        recover_journal_on_open(&db, &jp).unwrap();
        let dbf = File::open(&db).unwrap();
        let mut p2 = vec![0u8; PAGE_SIZE];
        dbf.read_exact_at(&mut p2, PAGE_SIZE as u64).unwrap();
        assert_eq!(p2, vec![0x42; PAGE_SIZE], "before-image restored");
        assert!(!jp.exists());
    }
}
