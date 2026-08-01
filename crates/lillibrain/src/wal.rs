//! Write-ahead log: a sidecar that holds page frames so the main file is never
//! overwritten until a checkpoint copies committed frames back.
//!
//! Frame layout (24-byte header + PAGE_SIZE bytes of page data):
//!
//! ```text
//!   Offset  Size  Field
//!        0     4  Page number (1-indexed)
//!        4     4  db_size_after: 0 for non-commit frames; page count on a commit frame
//!        8     4  Salt-1 (copied from the log header)
//!       12     4  Salt-2 (copied from the log header)
//!       16     4  Checksum-1: cumulative two-sum through this frame
//!       20     4  Checksum-2: second half of the cumulative checksum
//! ```
//!
//! The checksum chains the 32-byte header through every frame in order. A torn
//! or forged frame breaks the chain; recovery stops at the first mismatch. The
//! salt pair guards against replaying a stale frame from a prior cycle.

use crate::pos_io::PosIo;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use crate::consts::{
    PAGE_SIZE, WAL_FORMAT_VERSION, WAL_FRAME_HEADER_SIZE, WAL_FRAME_SIZE, WAL_HEADER_SIZE,
    WAL_MAGIC_BE,
};
use crate::error::Result;
use crate::io::{durable_fsync, fsync_parent_directory};

/// The two-sum checksum over `data`, chained from a prior `(s0, s1)` pair.
///
/// Input length must be a multiple of eight bytes; each big-endian u32 pair
/// folds into the running pair. Chaining across frames passes the previous
/// pair so a single torn frame breaks the whole tail.
pub fn wal_checksum(data: &[u8], mut s0: u32, mut s1: u32) -> (u32, u32) {
    for chunk in data.chunks_exact(8) {
        let w0 = u32::from_be_bytes(chunk[0..4].try_into().unwrap());
        let w1 = u32::from_be_bytes(chunk[4..8].try_into().unwrap());
        s0 = s0.wrapping_add(w0).wrapping_add(s1);
        s1 = s1.wrapping_add(w1).wrapping_add(s0);
    }
    (s0, s1)
}

/// A salt source that mixes full-resolution time, a process-global monotonic
/// counter, the process id, and a stack address (ASLR) so two salt draws never
/// collide — even within one coarse clock tick — without a crate dependency.
///
/// The salt is a secondary guard behind frame truncation and the checksum chain,
/// but a repeated salt weakens stale-frame rejection, so it must not lean on the
/// low bits of a single sub-second read.
fn fresh_salt() -> u32 {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seq = COUNTER.fetch_add(1, Ordering::Relaxed);

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let pid = u64::from(std::process::id());
    let stack_entropy = {
        let local = 0u8;
        (&local as *const u8) as u64
    };

    // SplitMix64 finalizer over the combined entropy so every input bit spreads
    // across the full width rather than clustering in the low bits.
    let mut x = nanos
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(seq.wrapping_mul(0xD1B5_4A32_9E37_79B9))
        .wrapping_add(pid.rotate_left(17))
        .wrapping_add(stack_entropy);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^= x >> 31;
    (x as u32) ^ ((x >> 32) as u32)
}

/// Owns the write-ahead sidecar file and the in-process page-to-frame index.
///
/// The sidecar is named by appending `-wal` to the database file path. The
/// caller holds the single-writer lock; this type is not internally
/// synchronized.
pub struct WalWriter {
    path: PathBuf,
    file: File,
    page_size: usize,
    salt1: u32,
    salt2: u32,
    checkpoint_seq: u32,
    running_s0: u32,
    running_s1: u32,
    next_frame_offset: u64,
    /// page_no -> 0-based committed frame index within the current cycle.
    index: std::collections::HashMap<u32, u64>,
    committed_frame_count: u64,
    committed_s0: u32,
    committed_s1: u32,
    committed_next_frame_offset: u64,
}

impl WalWriter {
    /// Open (or create) the `-wal` sidecar for `db_path` and write its header.
    pub fn open(db_path: impl AsRef<Path>) -> Result<WalWriter> {
        let db_path = db_path.as_ref().to_path_buf();
        let path = wal_path_for(&db_path);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        // A fresh-file path: recovery unlinks the sidecar before a writer is
        // built, so truncation only removes stale bytes if that guarantee ever
        // breaks — correctness does not rest on salt uniqueness alone.
        file.set_len(0)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, perms);
        }

        let mut w = WalWriter {
            path,
            file,
            page_size: PAGE_SIZE,
            salt1: fresh_salt(),
            salt2: fresh_salt().wrapping_add(0x5BD1_E995),
            checkpoint_seq: 0,
            running_s0: 0,
            running_s1: 0,
            next_frame_offset: WAL_HEADER_SIZE as u64,
            index: std::collections::HashMap::new(),
            committed_frame_count: 0,
            committed_s0: 0,
            committed_s1: 0,
            committed_next_frame_offset: WAL_HEADER_SIZE as u64,
        };
        w.write_header()?;
        // The sidecar's directory entry must be durable before any committed
        // frame relies on the file being findable after power loss.
        fsync_parent_directory(&w.path)?;
        Ok(w)
    }

    /// The sidecar path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The count of committed frames in the current cycle.
    pub fn committed_frame_count(&self) -> u64 {
        self.committed_frame_count
    }

    /// Write the 32-byte log header and seed the running and committed checksum
    /// pairs from its checksum.
    fn write_header(&mut self) -> Result<()> {
        let mut body = Vec::with_capacity(24);
        body.extend_from_slice(&WAL_MAGIC_BE.to_be_bytes());
        body.extend_from_slice(&WAL_FORMAT_VERSION.to_be_bytes());
        body.extend_from_slice(&(self.page_size as u32).to_be_bytes());
        body.extend_from_slice(&self.checkpoint_seq.to_be_bytes());
        body.extend_from_slice(&self.salt1.to_be_bytes());
        body.extend_from_slice(&self.salt2.to_be_bytes());
        let (s0, s1) = wal_checksum(&body, 0, 0);
        let mut header = body;
        header.extend_from_slice(&s0.to_be_bytes());
        header.extend_from_slice(&s1.to_be_bytes());
        self.file.write_all_at(&header, 0)?;

        self.running_s0 = s0;
        self.running_s1 = s1;
        self.committed_s0 = s0;
        self.committed_s1 = s1;
        self.committed_next_frame_offset = WAL_HEADER_SIZE as u64;
        Ok(())
    }

    /// Append one frame at the write-head and advance it.
    fn append_frame(&mut self, page_no: u32, page_data: &[u8], db_size_after: u32) -> Result<()> {
        debug_assert_eq!(page_data.len(), self.page_size);
        let mut raw_header = Vec::with_capacity(WAL_FRAME_HEADER_SIZE);
        raw_header.extend_from_slice(&page_no.to_be_bytes());
        raw_header.extend_from_slice(&db_size_after.to_be_bytes());
        raw_header.extend_from_slice(&self.salt1.to_be_bytes());
        raw_header.extend_from_slice(&self.salt2.to_be_bytes());

        // Checksum covers the first 8 bytes of the frame header plus page data.
        let mut chk_input = Vec::with_capacity(8 + self.page_size);
        chk_input.extend_from_slice(&raw_header[..8]);
        chk_input.extend_from_slice(page_data);
        let (new_s0, new_s1) = wal_checksum(&chk_input, self.running_s0, self.running_s1);

        raw_header.extend_from_slice(&new_s0.to_be_bytes());
        raw_header.extend_from_slice(&new_s1.to_be_bytes());

        let mut frame = Vec::with_capacity(WAL_FRAME_SIZE);
        frame.extend_from_slice(&raw_header);
        frame.extend_from_slice(page_data);
        self.file.write_all_at(&frame, self.next_frame_offset)?;

        self.next_frame_offset += WAL_FRAME_SIZE as u64;
        self.running_s0 = new_s0;
        self.running_s1 = new_s1;
        Ok(())
    }

    /// Append all dirty pages as frames and issue exactly one durable sync.
    ///
    /// The last frame carries the commit marker (`db_size_after > 0`); every
    /// preceding frame in the transaction carries zero. On success the
    /// in-process index and the committed snapshot advance to the post-commit
    /// state so a later [`WalWriter::rollback`] rewinds exactly here.
    pub fn commit_transaction(
        &mut self,
        dirty: &[(u32, Vec<u8>)],
        db_size_after: u32,
    ) -> Result<()> {
        if dirty.is_empty() {
            return Ok(());
        }
        let start_frame_idx = self.committed_frame_count;
        let last = dirty.len() - 1;
        for (i, (page_no, page_data)) in dirty.iter().enumerate() {
            let commit_size = if i == last { db_size_after } else { 0 };
            self.append_frame(*page_no, page_data, commit_size)?;
        }

        // One sync for the whole transaction — the only sync on the hot path.
        durable_fsync(&self.file)?;

        for (idx, (page_no, _)) in dirty.iter().enumerate() {
            self.index.insert(*page_no, start_frame_idx + idx as u64);
        }
        self.committed_frame_count += dirty.len() as u64;

        self.committed_s0 = self.running_s0;
        self.committed_s1 = self.running_s1;
        self.committed_next_frame_offset = self.next_frame_offset;
        Ok(())
    }

    /// Discard in-progress (un-fsynced) frames by rewinding to the committed
    /// snapshot. The rolled-back slots are harmlessly overwritten by the next
    /// commit; recovery ignores them because they carry no commit marker.
    pub fn rollback(&mut self) {
        self.running_s0 = self.committed_s0;
        self.running_s1 = self.committed_s1;
        self.next_frame_offset = self.committed_next_frame_offset;
    }

    /// Return the page bytes for `page_no` from the newest committed frame, or
    /// `None` when no frame holds it (the caller falls back to the main file).
    pub fn read_page(&self, page_no: u32) -> Result<Option<Vec<u8>>> {
        let Some(frame_idx) = self.index.get(&page_no) else {
            return Ok(None);
        };
        let frame_offset = WAL_HEADER_SIZE as u64 + frame_idx * WAL_FRAME_SIZE as u64;
        let mut raw = vec![0u8; WAL_FRAME_SIZE];
        self.file.read_exact_at(&mut raw, frame_offset)?;
        Ok(Some(raw[WAL_FRAME_HEADER_SIZE..].to_vec()))
    }

    /// Merge every committed frame into the main file and reset the sidecar.
    ///
    /// The two-sync ordering is the durability barrier: the sidecar is durable
    /// before any main-file page is overwritten, then the main file is durable
    /// before the sidecar is reset in place. The sidecar fd stays open — never
    /// closed and reopened, which would reset the kernel error sequence and
    /// could hide a prior failed writeback.
    pub fn checkpoint(&mut self, db_file: &File) -> Result<()> {
        if self.index.is_empty() {
            return Ok(());
        }

        // Step 1: the sidecar must be durable before the main file is touched.
        durable_fsync(&self.file)?;

        // Step 2: copy the newest frame of each indexed page into the main file.
        for (&page_no, &frame_idx) in self.index.iter() {
            let frame_offset = WAL_HEADER_SIZE as u64 + frame_idx * WAL_FRAME_SIZE as u64;
            let mut raw = vec![0u8; WAL_FRAME_SIZE];
            self.file.read_exact_at(&mut raw, frame_offset)?;
            let page_data = &raw[WAL_FRAME_HEADER_SIZE..];
            let db_offset = (page_no as u64 - 1) * self.page_size as u64;
            db_file.write_all_at(page_data, db_offset)?;
        }

        // Step 3: make all copied pages durable in the main file.
        durable_fsync(db_file)?;

        // Step 4: reset the sidecar in place — keep the fd open.
        self.checkpoint_seq += 1;
        self.salt2 = fresh_salt().wrapping_add(self.checkpoint_seq);
        self.file.set_len(WAL_HEADER_SIZE as u64)?;
        self.write_header()?;

        self.index.clear();
        self.committed_frame_count = 0;
        self.next_frame_offset = WAL_HEADER_SIZE as u64;
        Ok(())
    }
}

/// The `-wal` sidecar path for a database file.
pub fn wal_path_for(db_path: &Path) -> PathBuf {
    let mut name = db_path.file_name().unwrap_or_default().to_os_string();
    name.push("-wal");
    db_path.with_file_name(name)
}

/// A snapshot of the sidecar's checkpoint generation, observable lock-free from
/// a separate reader process.
///
/// A checkpoint copies committed frames into the main file, then bumps the
/// generation: it increments the sequence counter, redraws the secondary salt,
/// and truncates the sidecar back to its header. A reader that captured this
/// snapshot at open can therefore tell — without taking any lock and without
/// trusting any individual page's checksum — whether a checkpoint has rewritten
/// main-file pages since it opened. A `None` snapshot means no sidecar was
/// present (or it was unreadable) at observation time, which is itself a
/// generation transition relative to a `Some` snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointGeneration {
    /// Monotonic checkpoint counter from the sidecar header.
    pub seq: u32,
    /// Primary salt from the sidecar header.
    pub salt1: u32,
    /// Secondary salt; redrawn on every checkpoint.
    pub salt2: u32,
}

/// Read the sidecar's checkpoint generation, or `None` when no readable sidecar
/// is present. Reads only the 32-byte header and takes no lock, so it is safe to
/// call repeatedly from a lock-free read-only reader. A header that fails the
/// magic check yields `None` rather than a partially-parsed generation.
pub fn read_checkpoint_generation(wal_path: &Path) -> Option<CheckpointGeneration> {
    let wal_file = File::open(wal_path).ok()?;
    let mut raw_header = vec![0u8; WAL_HEADER_SIZE];
    wal_file.read_exact_at(&mut raw_header, 0).ok()?;
    let magic = u32::from_be_bytes(raw_header[0..4].try_into().unwrap());
    if magic != WAL_MAGIC_BE {
        return None;
    }
    let seq = u32::from_be_bytes(raw_header[12..16].try_into().unwrap());
    let salt1 = u32::from_be_bytes(raw_header[16..20].try_into().unwrap());
    let salt2 = u32::from_be_bytes(raw_header[20..24].try_into().unwrap());
    Some(CheckpointGeneration { seq, salt1, salt2 })
}

/// Scan the write-ahead sidecar and replay committed frames into the main file.
///
/// Called at open time before any reads are served. Validates the header magic,
/// page geometry, and checksum; walks frames, checking the salt pair before the
/// checksum so a stale frame from a prior cycle is rejected even when its own
/// checksum is internally valid; replays only complete committed runs; stops at
/// the first mismatch; then unlinks the sidecar.
///
/// Idempotent: re-running over the same sidecar writes identical bytes to the
/// same offsets, then unlinks. A crash between the replay and the unlink leaves
/// the next open in an identical starting state.
/// Build a page-no → newest-committed-page overlay from the WAL sidecar.
///
/// Walks the committed frame chain (identical verification to recovery: header
/// check, per-frame salt + checksum chain, in-progress tail dropped) and returns
/// each page's newest committed body. Used by a lock-free read-only open so a
/// reader sees data a writer has logged but not yet checkpointed into the main
/// file, WITHOUT writing the main file or touching the sidecar. An empty map
/// means there is nothing un-checkpointed to overlay.
pub fn read_committed_wal_frames(
    wal_path: &Path,
    page_size: usize,
) -> std::collections::HashMap<u32, Vec<u8>> {
    let mut newest: std::collections::HashMap<u32, Vec<u8>> = std::collections::HashMap::new();
    if !wal_path.exists() {
        return newest;
    }
    let wal_file = match File::open(wal_path) {
        Ok(f) => f,
        Err(_) => return newest,
    };

    let mut raw_header = vec![0u8; WAL_HEADER_SIZE];
    if wal_file.read_exact_at(&mut raw_header, 0).is_err() {
        return newest;
    }
    let magic = u32::from_be_bytes(raw_header[0..4].try_into().unwrap());
    let hdr_pgsz = u32::from_be_bytes(raw_header[8..12].try_into().unwrap()) as usize;
    let hdr_salt1 = u32::from_be_bytes(raw_header[16..20].try_into().unwrap());
    let hdr_salt2 = u32::from_be_bytes(raw_header[20..24].try_into().unwrap());
    let hdr_s0 = u32::from_be_bytes(raw_header[24..28].try_into().unwrap());
    let hdr_s1 = u32::from_be_bytes(raw_header[28..32].try_into().unwrap());

    if magic != WAL_MAGIC_BE || hdr_pgsz != page_size {
        return newest;
    }
    let (c0, c1) = wal_checksum(&raw_header[..24], 0, 0);
    if (c0, c1) != (hdr_s0, hdr_s1) {
        return newest;
    }

    let mut running_s0 = hdr_s0;
    let mut running_s1 = hdr_s1;
    let mut pending: Vec<(u32, Vec<u8>)> = Vec::new();
    let mut read_offset = WAL_HEADER_SIZE as u64;

    loop {
        let mut fh = vec![0u8; WAL_FRAME_HEADER_SIZE];
        if wal_file.read_exact_at(&mut fh, read_offset).is_err() {
            break;
        }
        let mut page_data = vec![0u8; page_size];
        if wal_file
            .read_exact_at(&mut page_data, read_offset + WAL_FRAME_HEADER_SIZE as u64)
            .is_err()
        {
            break;
        }
        let page_no = u32::from_be_bytes(fh[0..4].try_into().unwrap());
        let db_size_after = u32::from_be_bytes(fh[4..8].try_into().unwrap());
        let f_salt1 = u32::from_be_bytes(fh[8..12].try_into().unwrap());
        let f_salt2 = u32::from_be_bytes(fh[12..16].try_into().unwrap());
        let f_s0 = u32::from_be_bytes(fh[16..20].try_into().unwrap());
        let f_s1 = u32::from_be_bytes(fh[20..24].try_into().unwrap());

        if (f_salt1, f_salt2) != (hdr_salt1, hdr_salt2) {
            break;
        }
        let mut chk_input = Vec::with_capacity(8 + page_size);
        chk_input.extend_from_slice(&fh[..8]);
        chk_input.extend_from_slice(&page_data);
        let (new_s0, new_s1) = wal_checksum(&chk_input, running_s0, running_s1);
        if (new_s0, new_s1) != (f_s0, f_s1) {
            break;
        }
        running_s0 = new_s0;
        running_s1 = new_s1;
        pending.push((page_no, page_data));
        read_offset += WAL_FRAME_SIZE as u64;

        if db_size_after > 0 {
            // Commit barrier: fold the pending frames into the overlay (newest
            // body per page wins because later frames overwrite earlier ones).
            for (pno, body) in pending.drain(..) {
                newest.insert(pno, body);
            }
        }
    }
    newest
}

pub fn recover_wal_on_open(db_path: &Path, wal_path: &Path, page_size: usize) -> Result<()> {
    if !wal_path.exists() {
        return Ok(());
    }

    let wal_file = match File::open(wal_path) {
        Ok(f) => f,
        Err(_) => return Ok(()),
    };

    let mut committed: Vec<(u32, Vec<u8>)> = Vec::new();
    {
        // Step 1: read and verify the 32-byte header.
        let mut raw_header = vec![0u8; WAL_HEADER_SIZE];
        if wal_file.read_exact_at(&mut raw_header, 0).is_err() {
            let _ = std::fs::remove_file(wal_path);
            return Ok(());
        }
        let magic = u32::from_be_bytes(raw_header[0..4].try_into().unwrap());
        let hdr_pgsz = u32::from_be_bytes(raw_header[8..12].try_into().unwrap()) as usize;
        let hdr_salt1 = u32::from_be_bytes(raw_header[16..20].try_into().unwrap());
        let hdr_salt2 = u32::from_be_bytes(raw_header[20..24].try_into().unwrap());
        let hdr_s0 = u32::from_be_bytes(raw_header[24..28].try_into().unwrap());
        let hdr_s1 = u32::from_be_bytes(raw_header[28..32].try_into().unwrap());

        if magic != WAL_MAGIC_BE || hdr_pgsz != page_size {
            let _ = std::fs::remove_file(wal_path);
            return Ok(());
        }
        let (c0, c1) = wal_checksum(&raw_header[..24], 0, 0);
        if (c0, c1) != (hdr_s0, hdr_s1) {
            let _ = std::fs::remove_file(wal_path);
            return Ok(());
        }

        // Step 2: seed the running pair from the header checksum.
        let mut running_s0 = hdr_s0;
        let mut running_s1 = hdr_s1;

        let mut pending: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut read_offset = WAL_HEADER_SIZE as u64;

        // Step 3: walk frames sequentially.
        loop {
            let mut fh = vec![0u8; WAL_FRAME_HEADER_SIZE];
            if wal_file.read_exact_at(&mut fh, read_offset).is_err() {
                break; // torn or absent frame header — stop
            }
            let mut page_data = vec![0u8; page_size];
            if wal_file
                .read_exact_at(&mut page_data, read_offset + WAL_FRAME_HEADER_SIZE as u64)
                .is_err()
            {
                break; // torn page data — stop
            }

            let page_no = u32::from_be_bytes(fh[0..4].try_into().unwrap());
            let db_size_after = u32::from_be_bytes(fh[4..8].try_into().unwrap());
            let f_salt1 = u32::from_be_bytes(fh[8..12].try_into().unwrap());
            let f_salt2 = u32::from_be_bytes(fh[12..16].try_into().unwrap());
            let f_s0 = u32::from_be_bytes(fh[16..20].try_into().unwrap());
            let f_s1 = u32::from_be_bytes(fh[20..24].try_into().unwrap());

            // Salt check precedes the checksum check: a stale frame from a prior
            // cycle is rejected even when its checksum is valid under old salts.
            if (f_salt1, f_salt2) != (hdr_salt1, hdr_salt2) {
                break;
            }

            let mut chk_input = Vec::with_capacity(8 + page_size);
            chk_input.extend_from_slice(&fh[..8]);
            chk_input.extend_from_slice(&page_data);
            let (new_s0, new_s1) = wal_checksum(&chk_input, running_s0, running_s1);
            if (new_s0, new_s1) != (f_s0, f_s1) {
                break; // corrupt or torn frame — stop
            }

            running_s0 = new_s0;
            running_s1 = new_s1;
            pending.push((page_no, page_data));
            read_offset += WAL_FRAME_SIZE as u64;

            if db_size_after > 0 {
                committed.append(&mut pending);
                pending = Vec::new();
            }
        }
    }

    // Step 4: write committed pages to the main file. A later page wins, so a
    // newest-frame map is applied; iteration order does not matter because the
    // vec already records the newest occurrence last for each page.
    if !committed.is_empty() {
        let mut newest: std::collections::HashMap<u32, Vec<u8>> = std::collections::HashMap::new();
        for (page_no, page_data) in committed {
            newest.insert(page_no, page_data);
        }
        let db_file = OpenOptions::new().read(true).write(true).open(db_path)?;
        for (page_no, page_data) in &newest {
            let offset = (*page_no as u64 - 1) * page_size as u64;
            db_file.write_all_at(page_data, offset)?;
        }
        durable_fsync(&db_file)?;
    }

    // Step 5: remove the sidecar — the commit point of recovery. A surviving
    // sidecar would replay again, so the unlink must succeed and be durable.
    match std::fs::remove_file(wal_path) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(e.into()),
    }
    fsync_parent_directory(wal_path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.lilli");
        // Create an empty main file the recovery path can open.
        File::create(&path).unwrap();
        (dir, path)
    }

    fn page_of(byte: u8) -> Vec<u8> {
        vec![byte; PAGE_SIZE]
    }

    #[test]
    fn wal_checksum_matches_reference_on_fixed_input() {
        // 16 bytes: two big-endian u32 pairs.
        let data: [u8; 16] = [
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00,
            0x00, 0x04,
        ];
        // Hand-computed reference of the two-sum recurrence:
        //   start (0,0)
        //   pair (1,2): s0 = 0+1+0 = 1; s1 = 0+2+1 = 3
        //   pair (3,4): s0 = 1+3+3 = 7; s1 = 3+4+7 = 14
        let (s0, s1) = wal_checksum(&data, 0, 0);
        assert_eq!((s0, s1), (7, 14));
    }

    #[test]
    fn wal_append_two_frames_plus_commit_marker_one_header() {
        let (_d, db) = temp_db();
        let mut w = WalWriter::open(&db).unwrap();
        // Two data frames and a commit marker in one transaction.
        w.commit_transaction(&[(4, page_of(0xAA)), (5, page_of(0xBB))], 5)
            .unwrap();
        // The sidecar header carries the magic + format version + salt pair.
        let f = File::open(w.path()).unwrap();
        let mut hdr = vec![0u8; WAL_HEADER_SIZE];
        f.read_exact_at(&mut hdr, 0).unwrap();
        assert_eq!(
            u32::from_be_bytes(hdr[0..4].try_into().unwrap()),
            WAL_MAGIC_BE
        );
        assert_eq!(
            u32::from_be_bytes(hdr[4..8].try_into().unwrap()),
            WAL_FORMAT_VERSION
        );
        // Two committed frames recorded in the index.
        assert_eq!(w.committed_frame_count(), 2);
        assert_eq!(w.read_page(4).unwrap(), Some(page_of(0xAA)));
        assert_eq!(w.read_page(5).unwrap(), Some(page_of(0xBB)));
        assert_eq!(w.read_page(9).unwrap(), None);
    }

    #[test]
    fn wal_recovery_replays_committed_run_and_is_idempotent() {
        let (_d, db) = temp_db();
        // Size the main file so pages 1..=5 exist.
        let dbf = OpenOptions::new().read(true).write(true).open(&db).unwrap();
        dbf.set_len((5 * PAGE_SIZE) as u64).unwrap();
        drop(dbf);

        let wal_path = {
            let mut w = WalWriter::open(&db).unwrap();
            w.commit_transaction(&[(4, page_of(0x11)), (5, page_of(0x22))], 5)
                .unwrap();
            w.path().to_path_buf()
        };

        recover_wal_on_open(&db, &wal_path, PAGE_SIZE).unwrap();
        let dbf = File::open(&db).unwrap();
        let mut p4 = vec![0u8; PAGE_SIZE];
        let mut p5 = vec![0u8; PAGE_SIZE];
        dbf.read_exact_at(&mut p4, 3 * PAGE_SIZE as u64).unwrap();
        dbf.read_exact_at(&mut p5, 4 * PAGE_SIZE as u64).unwrap();
        assert_eq!(p4, page_of(0x11));
        assert_eq!(p5, page_of(0x22));
        // Sidecar is gone after replay.
        assert!(!wal_path.exists());

        // Idempotent: a second recovery over an absent sidecar is a no-op and
        // re-applying the same frames (had they survived) writes identical bytes.
        recover_wal_on_open(&db, &wal_path, PAGE_SIZE).unwrap();
        let dbf = File::open(&db).unwrap();
        let mut p4b = vec![0u8; PAGE_SIZE];
        dbf.read_exact_at(&mut p4b, 3 * PAGE_SIZE as u64).unwrap();
        assert_eq!(p4b, page_of(0x11));
    }

    #[test]
    fn wal_torn_final_frame_is_not_replayed() {
        let (_d, db) = temp_db();
        let dbf = OpenOptions::new().read(true).write(true).open(&db).unwrap();
        dbf.set_len((6 * PAGE_SIZE) as u64).unwrap();
        drop(dbf);

        let wal_path = {
            let mut w = WalWriter::open(&db).unwrap();
            // First committed transaction.
            w.commit_transaction(&[(4, page_of(0x33))], 6).unwrap();
            // Second transaction: write the frame bytes but truncate the tail so
            // the final frame is torn (simulating a crash mid-append).
            w.append_frame(5, &page_of(0x44), 6).unwrap();
            let p = w.path().to_path_buf();
            // Truncate the last frame's page data so the page read comes up short.
            let f = OpenOptions::new().read(true).write(true).open(&p).unwrap();
            let full = f.metadata().unwrap().len();
            f.set_len(full - 100).unwrap();
            p
        };

        recover_wal_on_open(&db, &wal_path, PAGE_SIZE).unwrap();
        let dbf = File::open(&db).unwrap();
        let mut p4 = vec![0u8; PAGE_SIZE];
        let mut p5 = vec![0u8; PAGE_SIZE];
        dbf.read_exact_at(&mut p4, 3 * PAGE_SIZE as u64).unwrap();
        dbf.read_exact_at(&mut p5, 4 * PAGE_SIZE as u64).unwrap();
        // The first committed frame applied; the torn frame did not.
        assert_eq!(p4, page_of(0x33));
        assert_ne!(p5, page_of(0x44));
    }

    #[test]
    fn wal_stale_frame_from_prior_cycle_is_rejected() {
        let (_d, db) = temp_db();
        let dbf = OpenOptions::new().read(true).write(true).open(&db).unwrap();
        dbf.set_len((6 * PAGE_SIZE) as u64).unwrap();
        drop(dbf);

        let wal_path = {
            let mut w = WalWriter::open(&db).unwrap();
            w.commit_transaction(&[(4, page_of(0x55))], 6).unwrap();
            let p = w.path().to_path_buf();
            // Forge a frame appended after the committed run carrying a different
            // salt pair (as a stale cycle frame would). It must be rejected.
            let f = OpenOptions::new().read(true).write(true).open(&p).unwrap();
            let off = f.metadata().unwrap().len();
            let mut fh = vec![0u8; WAL_FRAME_HEADER_SIZE];
            fh[0..4].copy_from_slice(&5u32.to_be_bytes());
            fh[4..8].copy_from_slice(&6u32.to_be_bytes()); // commit marker
            fh[8..12].copy_from_slice(&0xDEAD_BEEFu32.to_be_bytes()); // wrong salt1
            fh[12..16].copy_from_slice(&0xFEED_FACEu32.to_be_bytes()); // wrong salt2
            f.write_all_at(&fh, off).unwrap();
            f.write_all_at(&page_of(0x66), off + WAL_FRAME_HEADER_SIZE as u64)
                .unwrap();
            p
        };

        recover_wal_on_open(&db, &wal_path, PAGE_SIZE).unwrap();
        let dbf = File::open(&db).unwrap();
        let mut p5 = vec![0u8; PAGE_SIZE];
        dbf.read_exact_at(&mut p5, 4 * PAGE_SIZE as u64).unwrap();
        assert_ne!(p5, page_of(0x66), "stale-salt frame must not be replayed");
    }

    #[test]
    fn wal_checkpoint_merges_in_place_and_keeps_fd_open() {
        let (_d, db) = temp_db();
        let dbf = OpenOptions::new().read(true).write(true).open(&db).unwrap();
        dbf.set_len((5 * PAGE_SIZE) as u64).unwrap();
        drop(dbf);

        let mut w = WalWriter::open(&db).unwrap();
        w.commit_transaction(&[(4, page_of(0x77))], 5).unwrap();

        let dbf = OpenOptions::new().read(true).write(true).open(&db).unwrap();
        w.checkpoint(&dbf).unwrap();

        // Main file carries the merged page.
        let mut p4 = vec![0u8; PAGE_SIZE];
        dbf.read_exact_at(&mut p4, 3 * PAGE_SIZE as u64).unwrap();
        assert_eq!(p4, page_of(0x77));

        // Sidecar reset to header size; the index is empty; the writer keeps
        // serving new commits without a reopen.
        let sidecar_len = std::fs::metadata(w.path()).unwrap().len();
        assert_eq!(sidecar_len, WAL_HEADER_SIZE as u64);
        assert_eq!(w.committed_frame_count(), 0);
        assert_eq!(w.read_page(4).unwrap(), None);

        // The same writer can commit again post-checkpoint.
        w.commit_transaction(&[(5, page_of(0x88))], 5).unwrap();
        assert_eq!(w.read_page(5).unwrap(), Some(page_of(0x88)));
    }
}
