//! Paged file with a bounded LRU page cache, per-page checksums, and
//! dirty/pinned tracking.
//!
//! The pager owns the store file and mediates all page access. A page read from
//! disk is verified against its trailing per-page checksum and rejected on
//! mismatch. Writes land in an in-memory cache and are flushed durably on
//! [`Pager::flush`]. The cache is bounded by an optional page budget; only
//! clean, unpinned pages are evicted, so a dirty or pinned page is never
//! dropped regardless of pressure.
//!
//! Page roles follow the reference layout: page 1 is the 100-byte database
//! header, the B+tree root is reserved at page 2, and the meta page is reserved
//! at page 3. Higher layers build on these reservations.

use crate::pos_io::PosIo;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

use fs4::FileExt as Fs4FileExt;
use parking_lot::Mutex;

use crate::consts::{
    DB_MAGIC, HDR_DB_SIZE_OFFSET, HDR_FIRST_FREELIST_OFFSET, HDR_FREELIST_COUNT_OFFSET,
    HDR_MAGIC_OFFSET, HDR_PAGE_SIZE_OFFSET, HDR_READ_VERSION_OFFSET, HDR_WRITE_VERSION_OFFSET,
    PAGE_CRC_COVERED, PAGE_CRC_OFFSET, PAGE_SIZE, WRITE_VERSION_JOURNAL, WRITE_VERSION_WAL,
};
use crate::error::{Result, StoreError};
use crate::io::{durable_fsync, fsync_parent_directory};
use crate::journal::{journal_path_for, recover_journal_on_open, Journal};
use crate::wal::{
    read_checkpoint_generation, read_committed_wal_frames_to_end, recover_wal_on_open,
    wal_path_for, CheckpointGeneration, WalWriter,
};

/// The header page (page 1) is always pinned.
const HEADER_PAGE: u32 = 1;

/// Cross-language contract: the Python RO pool (iai_mcp/hippo/_ro_pool.py,
/// `_RO_SNAPSHOT_FENCE_MARKER`) discriminates the snapshot fence SOLELY by
/// this message prefix — there is no typed fence error across the PyO3
/// boundary. Changing this string without updating that marker silently
/// disables every fence retry on the recall read path.
pub const RO_SNAPSHOT_FENCE_PREFIX: &str = "read-only snapshot invalidated";

/// Compute the per-page checksum over the covered prefix of a page buffer.
fn page_crc(page: &[u8]) -> u32 {
    crc32fast::hash(&page[..PAGE_CRC_COVERED])
}

/// Write the per-page checksum into the trailing checksum slot of `page`.
fn stamp_crc(page: &mut [u8]) {
    let crc = page_crc(page);
    page[PAGE_CRC_OFFSET..PAGE_CRC_OFFSET + 4].copy_from_slice(&crc.to_be_bytes());
}

/// Read the stored per-page checksum from the trailing slot of `page`.
fn stored_crc(page: &[u8]) -> u32 {
    u32::from_be_bytes(
        page[PAGE_CRC_OFFSET..PAGE_CRC_OFFSET + 4]
            .try_into()
            .unwrap(),
    )
}

/// A single cached page and its recency token.
struct CacheEntry {
    buf: Vec<u8>,
    /// Monotonic recency token; the largest token is the most recently used.
    used: u64,
}

struct Inner {
    cache: HashMap<u32, CacheEntry>,
    dirty: std::collections::HashSet<u32>,
    pinned: std::collections::HashSet<u32>,
    /// Maximum cached pages; `None` is unbounded.
    max_cached_pages: Option<usize>,
    /// Monotonic clock for LRU recency.
    clock: u64,
}

/// Write-transaction state: the rollback journal, the write-ahead writer, and
/// the per-transaction bookkeeping the commit/rollback barriers need.
struct TxState {
    /// True between [`Pager::begin_write`] and a commit/rollback.
    open: bool,
    /// `db_size` captured at `begin_write`, used to truncate on rollback so a
    /// transaction that extended the file leaves no half-written tail pages.
    pre_txn_db_size: u32,
    /// Rollback-journal sidecar (present only in journal mode while a txn is open).
    journal: Option<Journal>,
    /// Write-ahead writer (present once WAL mode is enabled, across transactions).
    wal: Option<WalWriter>,
    /// True when this pager runs in write-ahead mode.
    wal_mode: bool,
}

/// A paged file with a checksummed, bounded-LRU page cache.
pub struct Pager {
    path: PathBuf,
    /// The main store file. Wrapped in `Mutex<Option<…>>` so the `unlock()`
    /// call — which takes `&self` — can close the fd deterministically at
    /// connection-close time rather than waiting for the Rust object to be
    /// dropped (which is non-deterministic under Python's GC).
    file: Mutex<Option<File>>,
    page_size: usize,
    inner: Mutex<Inner>,
    /// Write-transaction state (journal/WAL + per-txn bookkeeping).
    tx: Mutex<TxState>,
    /// B+tree root page number (reserved at page 2).
    pub root_page_no: u32,
    /// Count of [`Pager::read_page`] calls; surfaced for cost assertions.
    read_counter: std::sync::atomic::AtomicU64,
    /// Count of whole-tree ascending walks; surfaced so a cost assertion can
    /// prove the insert path never falls back to a linear sweep of the tree.
    full_scan_counter: std::sync::atomic::AtomicU64,
    /// Count of individual leaf cells visited by streaming scans
    /// ([`Tree::scan_cells_with`]); surfaced so a cost assertion can prove a
    /// query touches O(LIMIT) cells (index-served) rather than O(corpus) cells
    /// (a linear column-selective sweep). Distinct from `full_scan_counter`,
    /// which counts whole-tree materialising walks; the streaming path is
    /// deliberately not counted there, so this per-cell counter is the only
    /// direct observable of streaming-scan cost.
    cells_visited_counter: std::sync::atomic::AtomicU64,
    /// True for a lock-free read-only open. A read-only pager never takes the
    /// store's exclusive advisory lock and never writes, so a separate reader
    /// process can read committed pages while a writer holds the store open.
    read_only: bool,
    /// Read-only WAL overlay: page-no → newest committed body for pages a writer
    /// has logged but not yet checkpointed into the main file. Populated at a
    /// read-only open and consulted before the main file, so the reader sees a
    /// consistent committed snapshot without writing the main file. Empty for a
    /// read-write open (the live WAL writer index serves those reads instead).
    wal_overlay: HashMap<u32, Vec<u8>>,
    /// Read-only snapshot fence: the main-file length and the sidecar checkpoint
    /// generation captured at a read-only open. The overlay is a fixed open-time
    /// snapshot, so a writer that checkpoints concurrently rewrites main-file
    /// pages and advances the generation — mixing overlay pages from one
    /// generation with main-file pages from another would yield a cross-generation
    /// read where each page is individually checksum-valid yet the set is
    /// inconsistent. Re-validating this fence on every main-file fallback read
    /// turns that window into a reported integrity error rather than a silently
    /// torn B-tree, without taking any writer-blocking lock. `None` for a
    /// read-write open.
    ro_snapshot_fence: Option<RoSnapshotFence>,
    /// The committed-tail byte offset of the adopted overlay's parse, for the
    /// read-only refresh's never-regress check: within one checkpoint
    /// generation the sidecar only grows, so an accepted capture whose tail
    /// sits BEHIND this offset would regress the view onto an older snapshot
    /// and must be rejected. Reset when a checkpoint crosses. 0 for a
    /// read-write open.
    wal_committed_end: u64,
}

/// The open-time fence a lock-free read-only pager re-validates on every
/// main-file fallback read to detect a concurrent checkpoint.
struct RoSnapshotFence {
    /// Main-file length in bytes at open.
    main_len: u64,
    /// Sidecar checkpoint generation at open (`None` when no sidecar was present).
    generation: Option<CheckpointGeneration>,
    /// The sidecar path, re-read cheaply (32-byte header) on each validation.
    wal_path: PathBuf,
}

impl Pager {
    /// Open or create the store file at `path`.
    ///
    /// A fresh file is formatted with the database magic, page size, an initial
    /// `db_size` of 3 (header, root, meta reservations), rollback-journal mode,
    /// and an empty freelist. An existing file is opened and its header is left
    /// intact; crash recovery is a no-op at this layer.
    pub fn open(path: impl AsRef<Path>) -> Result<Pager> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        // Advisory single-writer lock on the store file.
        Fs4FileExt::try_lock(&file).map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::WouldBlock,
                format!("store is locked: {e}"),
            )
        })?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            let _ = std::fs::set_permissions(&path, perms);
        }

        let existing_size = file.metadata()?.len();

        let pager = Pager {
            path,
            file: Mutex::new(Some(file)),
            page_size: PAGE_SIZE,
            inner: Mutex::new(Inner {
                cache: HashMap::new(),
                dirty: std::collections::HashSet::new(),
                pinned: {
                    let mut s = std::collections::HashSet::new();
                    s.insert(HEADER_PAGE);
                    s
                },
                max_cached_pages: None,
                clock: 0,
            }),
            tx: Mutex::new(TxState {
                open: false,
                pre_txn_db_size: 0,
                journal: None,
                wal: None,
                wal_mode: false,
            }),
            root_page_no: 2,
            read_counter: std::sync::atomic::AtomicU64::new(0),
            full_scan_counter: std::sync::atomic::AtomicU64::new(0),
            cells_visited_counter: std::sync::atomic::AtomicU64::new(0),
            read_only: false,
            wal_overlay: HashMap::new(),
            ro_snapshot_fence: None,
            wal_committed_end: 0,
        };

        if existing_size == 0 {
            pager.init_fresh_file()?;
        } else {
            pager.recover_if_needed()?;
            // Load page 1 so header fields are accessible.
            let _ = pager.read_page(1)?;
            // If the on-disk header records write-ahead mode, attach a writer so
            // commits route through the log after a reopen.
            pager.attach_wal_if_header_says_so()?;
        }
        Ok(pager)
    }

    /// Open an existing store file for lock-free read-only access.
    ///
    /// Takes no advisory lock and never writes, so a separate reader process
    /// (the ANN-index rebuild worker, a daemon-down recall reader) can read
    /// committed pages while a writer process holds the store's exclusive lock.
    /// It performs no crash recovery — the writer owns recovery — so the caller
    /// must read a checkpointed store (the maintenance pass checkpoints the WAL
    /// into the main file before spawning the reader). Per-page checksums still
    /// verify every page read, so a page torn by a concurrent commit surfaces as
    /// an integrity error rather than silent corruption. Fails if the file does
    /// not exist or is empty.
    /// Capture a read-only view's `(WAL overlay, main-file length, checkpoint
    /// generation, committed-tail offset)` that is ONE committed snapshot at
    /// least as new as the sidecar's tail at capture time.
    ///
    /// A single unguarded parse racing a live writer can observe a torn WAL —
    /// a mid-file checksum break or a mid-reset header — and silently truncate
    /// the overlay to a stale prefix (or to empty), which a reader would then
    /// serve as a snapshot with committed rows absent. Acceptance: the
    /// checkpoint generation must not move across the parse, and the walk must
    /// end within one frame of the sidecar length measured AFTER the walk —
    /// i.e. every complete frame present when the walk finished was consumed.
    /// A live appender only ever leaves the sub-frame in-flight tail
    /// unconsumed, so the loop converges; a torn mid-file break retries. A
    /// short walk that repeats byte-identically across two attempts with a
    /// still sidecar is at-rest content beyond the last commit barrier
    /// (abandoned rollback bytes), not a race, and is accepted. Exhaustion is
    /// a hard error — the caller reopens or fails loud, never serves the torn
    /// read.
    fn capture_ro_view(
        wal_path: &Path,
        page_size: usize,
        main_len: &dyn Fn() -> Result<u64>,
    ) -> Result<(HashMap<u32, Vec<u8>>, u64, Option<CheckpointGeneration>, u64)> {
        const ATTEMPTS: u32 = 8;
        let mut prev_probe: Option<(u64, Option<CheckpointGeneration>, u64)> = None;
        for attempt in 0..ATTEMPTS {
            let gen_before = read_checkpoint_generation(wal_path);
            let (overlay, end) = read_committed_wal_frames_to_end(wal_path, page_size);
            let main_len_after = main_len()?;
            let wal_len_after = std::fs::metadata(wal_path).map(|m| m.len()).unwrap_or(0);
            let gen_after = read_checkpoint_generation(wal_path);
            if gen_after == gen_before {
                let complete = if wal_len_after <= crate::consts::WAL_HEADER_SIZE as u64 {
                    true
                } else {
                    end >= crate::consts::WAL_HEADER_SIZE as u64
                        && wal_len_after.saturating_sub(end)
                            < crate::consts::WAL_FRAME_SIZE as u64
                };
                if complete {
                    return Ok((overlay, main_len_after, gen_after, end));
                }
                let probe = (wal_len_after, gen_after, end);
                if prev_probe.as_ref() == Some(&probe) {
                    return Ok((overlay, main_len_after, gen_after, end));
                }
                prev_probe = Some(probe);
            } else {
                prev_probe = None;
            }
            std::thread::sleep(std::time::Duration::from_micros(
                150 * (attempt as u64 + 1),
            ));
        }
        Err(StoreError::Integrity {
            detail: "read-only snapshot capture: WAL kept tearing across bounded retries"
                .to_string(),
        })
    }

    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Pager> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .read(true)
            .write(false)
            .create(false)
            .open(&path)?;

        let existing_size = file.metadata()?.len();
        if existing_size == 0 {
            return Err(StoreError::Integrity {
                detail: "open_read_only: store file is empty".to_string(),
            });
        }

        // Overlay the committed-but-unmerged WAL frames so a reader sees data a
        // writer logged but has not yet checkpointed into the main file. No
        // main-file write, no recovery, no WAL writer. Captured stable-or-fail:
        // adopting a torn parse would open a snapshot with committed rows
        // absent.
        let wal_path = wal_path_for(&path);
        let main_len = |f: &File| -> Result<u64> { Ok(f.metadata()?.len()) };
        let (wal_overlay, fence_main_len, fence_generation, committed_end) =
            Self::capture_ro_view(&wal_path, PAGE_SIZE, &|| main_len(&file))?;

        // The snapshot fence: the main-file length and the sidecar checkpoint
        // generation captured in the SAME stable window as the overlay. Each
        // main-file fallback read re-validates against this so a concurrent
        // checkpoint that rewrites main-file pages is detected (and reported)
        // rather than silently mixed with the open-time overlay into a
        // cross-generation read.
        let ro_snapshot_fence = Some(RoSnapshotFence {
            main_len: fence_main_len,
            generation: fence_generation,
            wal_path,
        });

        let pager = Pager {
            path,
            file: Mutex::new(Some(file)),
            page_size: PAGE_SIZE,
            inner: Mutex::new(Inner {
                cache: HashMap::new(),
                dirty: std::collections::HashSet::new(),
                pinned: {
                    let mut s = std::collections::HashSet::new();
                    s.insert(HEADER_PAGE);
                    s
                },
                max_cached_pages: None,
                clock: 0,
            }),
            tx: Mutex::new(TxState {
                open: false,
                pre_txn_db_size: 0,
                journal: None,
                wal: None,
                wal_mode: false,
            }),
            root_page_no: 2,
            read_counter: std::sync::atomic::AtomicU64::new(0),
            full_scan_counter: std::sync::atomic::AtomicU64::new(0),
            cells_visited_counter: std::sync::atomic::AtomicU64::new(0),
            read_only: true,
            wal_overlay,
            ro_snapshot_fence,
            wal_committed_end: committed_end,
        };

        // Load page 1 so header fields are accessible; no recovery, no WAL writer.
        let _ = pager.read_page(1)?;
        Ok(pager)
    }

    /// Execute `op` with a reference to the main store file, returning
    /// `StoreError::Io` when the file has already been closed by `unlock()`.
    ///
    /// All pager I/O routes through this helper so post-close operations
    /// surface a typed error rather than a panic.
    fn with_file<T, Op>(&self, mut op: Op) -> Result<T>
    where
        Op: FnMut(&File) -> Result<T>,
    {
        let guard = self.file.lock();
        match guard.as_ref() {
            Some(file) => op(file),
            None => Err(StoreError::Io(std::io::Error::new(
                std::io::ErrorKind::NotConnected,
                "pager file already closed",
            ))),
        }
    }

    /// Re-validate the read-only snapshot fence before serving a main-file page.
    ///
    /// The overlay is a fixed open-time snapshot; only main-file pages can shift
    /// generation under a concurrent checkpoint. If the main-file length grew or
    /// the sidecar checkpoint generation advanced since open, a checkpoint has
    /// rewritten main-file pages and a main-file read would mix generations — so
    /// this returns a typed integrity error and the caller propagates it as a
    /// reported failure, never a torn read. Takes no lock: it reads the current
    /// main-file length and the sidecar's 32-byte header and compares them to the
    /// open-time capture. The happy path (no concurrent checkpoint) leaves the
    /// fence unchanged and returns `Ok(())`.
    fn validate_ro_snapshot_fence(&self) -> Result<()> {
        let Some(fence) = self.ro_snapshot_fence.as_ref() else {
            return Ok(());
        };
        let current_len = self.with_file(|f| Ok(f.metadata()?.len()))?;
        if current_len != fence.main_len {
            return Err(StoreError::SnapshotFence {
                detail: format!(
                    "{RO_SNAPSHOT_FENCE_PREFIX}: main file size changed under a concurrent checkpoint"
                ),
            });
        }
        let current_generation = read_checkpoint_generation(&fence.wal_path);
        if current_generation != fence.generation {
            return Err(StoreError::SnapshotFence {
                detail: format!(
                    "{RO_SNAPSHOT_FENCE_PREFIX}: checkpoint generation advanced under a concurrent checkpoint"
                ),
            });
        }
        Ok(())
    }

    /// Re-validate the read-only snapshot fence regardless of page cache state.
    ///
    /// `validate_ro_snapshot_fence` alone only fires on a genuine page-cache
    /// miss inside `load_and_verify_page` — a page already resident in the
    /// pager's cache (e.g. from an earlier read on this same connection) is
    /// served straight from memory and never re-validated, so a concurrent
    /// checkpoint that rewrites that exact page goes undetected for the rest
    /// of this pager's lifetime. Callers that must catch every invalidation
    /// (a statement-dispatch entry point, not a per-page cache path) call this
    /// unconditionally before serving a read-only statement. A no-op for a
    /// read-write pager (`ro_snapshot_fence` is `None`).
    pub fn revalidate_ro_snapshot_fence(&self) -> Result<()> {
        self.validate_ro_snapshot_fence()
    }

    /// Re-arm a read-only pager's committed snapshot to the writer's newest
    /// committed state, in place — the freshness alternative to a close +
    /// reopen. Returns `Ok(true)` when the visible snapshot moved, `Ok(false)`
    /// when the store is byte-identical to the current snapshot.
    ///
    /// Cache-correctness invariant: after this returns, no cached page may
    /// differ from the newly-armed committed image. A checkpoint (sidecar
    /// generation or main-file length moved) rewrites main-file pages
    /// wholesale, so the whole cache is dropped; otherwise only pages whose
    /// WAL-overlay image changed are evicted, and every other resident page is
    /// still valid — main-file pages cannot move without a checkpoint.
    pub fn refresh_ro_snapshot(&mut self) -> Result<bool> {
        if !self.read_only {
            return Err(StoreError::Integrity {
                detail: "refresh_ro_snapshot: pager is not read-only".to_string(),
            });
        }
        let Some(fence) = self.ro_snapshot_fence.as_ref() else {
            return Err(StoreError::Integrity {
                detail: "refresh_ro_snapshot: read-only pager has no snapshot fence".to_string(),
            });
        };
        let wal_path = fence.wal_path.clone();
        let fence_main_len = fence.main_len;
        let fence_generation = fence.generation;
        let (new_overlay, current_len, current_generation, committed_end) =
            Self::capture_ro_view(&wal_path, self.page_size, &|| {
                self.with_file(|f| Ok(f.metadata()?.len()))
            })?;
        let checkpoint_crossed =
            current_len != fence_main_len || current_generation != fence_generation;
        // Within one checkpoint generation the sidecar only grows; a capture
        // whose committed tail sits behind the adopted view would regress this
        // reader onto an older snapshot — reject it loud (the borrower reopens
        // rather than serving committed rows as absent).
        if !checkpoint_crossed && committed_end < self.wal_committed_end {
            return Err(StoreError::Integrity {
                detail: format!(
                    "refresh_ro_snapshot: captured tail {committed_end} behind adopted tail {}",
                    self.wal_committed_end
                ),
            });
        }
        // A journal-mode writer commits straight into main-file pages with no
        // sidecar generation to fence on, so an unchanged length and an empty
        // overlay prove nothing — drop the cache wholesale and re-arm.
        let unfenceable = current_generation.is_none()
            && fence_generation.is_none()
            && new_overlay.is_empty()
            && self.wal_overlay.is_empty();
        if !unfenceable && !checkpoint_crossed && new_overlay == self.wal_overlay {
            return Ok(false);
        }
        {
            let mut inner = self.inner.lock();
            if checkpoint_crossed || unfenceable {
                inner.cache.clear();
            } else {
                for page_no in new_overlay.keys() {
                    if self.wal_overlay.get(page_no) != new_overlay.get(page_no) {
                        inner.cache.remove(page_no);
                    }
                }
            }
        }
        self.wal_overlay = new_overlay;
        self.wal_committed_end = committed_end;
        self.ro_snapshot_fence = Some(RoSnapshotFence {
            main_len: current_len,
            generation: current_generation,
            wal_path,
        });
        Ok(true)
    }

    /// True when this pager was opened lock-free for read-only access.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// True while the main store file handle is still held — `false` once
    /// `unlock()` has released the file and lock. A handle whose file has been
    /// released cannot serve any further I/O.
    pub fn is_open(&self) -> bool {
        self.file.lock().is_some()
    }

    /// Format page 1 (header), page 2 (root), and page 3 (meta) on a new file.
    fn init_fresh_file(&self) -> Result<()> {
        let mut header = vec![0u8; self.page_size];
        header[HDR_MAGIC_OFFSET..HDR_MAGIC_OFFSET + DB_MAGIC.len()].copy_from_slice(&DB_MAGIC);
        header[HDR_PAGE_SIZE_OFFSET..HDR_PAGE_SIZE_OFFSET + 2]
            .copy_from_slice(&(self.page_size as u16).to_be_bytes());
        header[HDR_WRITE_VERSION_OFFSET] = WRITE_VERSION_JOURNAL;
        header[HDR_READ_VERSION_OFFSET] = WRITE_VERSION_JOURNAL;
        // db_size = 3: page 1 header, page 2 root reservation, page 3 meta reservation.
        header[HDR_DB_SIZE_OFFSET..HDR_DB_SIZE_OFFSET + 4].copy_from_slice(&3u32.to_be_bytes());
        // first_freelist_trunk = 0, freelist_count = 0 already zero.

        let root = vec![0u8; self.page_size];
        let meta = vec![0u8; self.page_size];

        self.write_page(1, &header)?;
        self.write_page(2, &root)?;
        self.write_page(3, &meta)?;
        self.flush()?;
        Ok(())
    }

    /// Replay any crash-interrupted transaction on open, routing on the header
    /// write-version byte: write-ahead mode replays the `-wal` sidecar;
    /// rollback-journal mode replays a committed `.journal`; neither present is a
    /// no-op. Runs before any reads are served and is idempotent.
    fn recover_if_needed(&self) -> Result<()> {
        // Peek the write-version byte directly off disk to select the path.
        let mut vbyte = [0u8; 1];
        if self
            .with_file(|f| Ok(f.read_exact_at(&mut vbyte, HDR_WRITE_VERSION_OFFSET as u64)?))
            .is_err()
        {
            return Ok(());
        }
        if vbyte[0] == WRITE_VERSION_WAL {
            let wal_path = wal_path_for(&self.path);
            recover_wal_on_open(&self.path, &wal_path, self.page_size)?;
            return Ok(());
        }
        let journal_path = journal_path_for(&self.path);
        recover_journal_on_open(&self.path, &journal_path)
    }

    /// Attach a [`WalWriter`] when the on-disk header records write-ahead mode so
    /// commits route through the log after a reopen.
    fn attach_wal_if_header_says_so(&self) -> Result<()> {
        let mut vbyte = [0u8; 1];
        self.with_file(|f| Ok(f.read_exact_at(&mut vbyte, HDR_WRITE_VERSION_OFFSET as u64)?))?;
        if vbyte[0] == WRITE_VERSION_WAL {
            let mut tx = self.tx.lock();
            if tx.wal.is_none() {
                tx.wal = Some(WalWriter::open(&self.path)?);
                tx.wal_mode = true;
            }
        }
        Ok(())
    }

    /// Switch this pager to write-ahead mode. Constructs the log writer, stamps
    /// both version bytes in the header, and durably flushes page 1 so the mode
    /// survives a reopen. Idempotent.
    pub fn enable_wal_mode(&self) -> Result<()> {
        {
            let tx = self.tx.lock();
            if tx.wal_mode {
                return Ok(());
            }
        }
        let writer = WalWriter::open(&self.path)?;

        // Stamp both version bytes into the cached page-1 buffer and persist it
        // directly to the main file — a header-only durability write, not a data
        // transaction.
        let header_bytes = {
            let mut inner = self.inner.lock();
            let mut h = self.ensure_cached(&mut inner, 1)?;
            h[HDR_WRITE_VERSION_OFFSET] = WRITE_VERSION_WAL;
            h[HDR_READ_VERSION_OFFSET] = WRITE_VERSION_WAL;
            stamp_crc(&mut h);
            self.store_dirty(&mut inner, 1, h.clone());
            inner.dirty.remove(&1);
            h
        };
        self.with_file(|f| Ok(f.write_all_at(&header_bytes, 0)?))?;
        self.with_file(|f| durable_fsync(f))?;

        let mut tx = self.tx.lock();
        tx.wal = Some(writer);
        tx.wal_mode = true;
        Ok(())
    }

    /// True when this pager runs in write-ahead mode.
    pub fn is_wal_mode(&self) -> bool {
        self.tx.lock().wal_mode
    }

    // -----------------------------------------------------------------------
    // Transaction API
    // -----------------------------------------------------------------------

    /// Begin a write transaction. In journal mode this creates the `.journal`
    /// sidecar and arms before-image capture; in WAL mode it only marks the
    /// transaction open. The pre-transaction `db_size` is recorded so a rollback
    /// can truncate away pages the transaction appended.
    pub fn begin_write(&self) -> Result<()> {
        if self.read_only {
            return Err(StoreError::Integrity {
                detail: "begin_write: pager is read-only".to_string(),
            });
        }
        let pre_size = self.db_size()?;
        let mut tx = self.tx.lock();
        if tx.open {
            return Err(StoreError::Integrity {
                detail: "begin_write: a transaction is already open".to_string(),
            });
        }
        tx.open = true;
        tx.pre_txn_db_size = pre_size;
        if !tx.wal_mode {
            tx.journal = Some(Journal::create(&self.path)?);
        }
        Ok(())
    }

    /// Whether a write transaction is currently open.
    pub fn in_transaction(&self) -> bool {
        self.tx.lock().open
    }

    /// Capture the before-image of `page_no` (journal mode, open transaction
    /// only). Reads the current on-disk bytes for a page that existed before the
    /// transaction; a page beyond the pre-transaction extent has no before-image
    /// (rollback truncates it away instead). Idempotent per page.
    fn capture_before_image(&self, page_no: u32) -> Result<()> {
        let mut tx = self.tx.lock();
        if !tx.open || tx.wal_mode {
            return Ok(());
        }
        let pre_size = tx.pre_txn_db_size;
        if page_no > pre_size {
            return Ok(());
        }
        if let Some(journal) = tx.journal.as_mut() {
            // Read the page's current on-disk image (the state to restore).
            let mut buf = vec![0u8; self.page_size];
            let offset = (page_no as u64 - 1) * self.page_size as u64;
            self.with_file(|f| Ok(f.read_exact_at(&mut buf, offset)?))?;
            journal.record_before_image(page_no, &buf);
        }
        Ok(())
    }

    /// The on-disk byte offset of a 1-indexed page.
    fn offset_of(&self, page_no: u32) -> u64 {
        (page_no as u64 - 1) * self.page_size as u64
    }

    /// Read a named big-endian u32 header field from page 1.
    fn read_header_u32(&self, offset: usize) -> Result<u32> {
        let mut inner = self.inner.lock();
        let buf = self.ensure_cached(&mut inner, 1)?;
        Ok(u32::from_be_bytes(
            buf[offset..offset + 4].try_into().unwrap(),
        ))
    }

    /// Total page count from the header.
    pub fn db_size(&self) -> Result<u32> {
        self.read_header_u32(HDR_DB_SIZE_OFFSET)
    }

    /// Physical page count implied by the on-disk file length (full pages only),
    /// independent of the header's recorded `db_size`. A checker compares the two
    /// to surface trailing pages past the recorded extent.
    pub fn physical_page_count(&self) -> Result<u32> {
        let len = self.with_file(|f| Ok(f.metadata()?.len()))?;
        Ok((len / PAGE_SIZE as u64) as u32)
    }

    /// First freelist trunk page (0 = empty).
    pub fn first_freelist_trunk(&self) -> Result<u32> {
        self.read_header_u32(HDR_FIRST_FREELIST_OFFSET)
    }

    /// Total free page count from the header.
    pub fn freelist_count(&self) -> Result<u32> {
        self.read_header_u32(HDR_FREELIST_COUNT_OFFSET)
    }

    /// Write a named big-endian u32 header field into the page-1 cache buffer.
    pub fn update_header_u32(&self, offset: usize, value: u32) -> Result<()> {
        self.capture_before_image(1)?;
        let mut inner = self.inner.lock();
        let mut buf = self.ensure_cached(&mut inner, 1)?;
        buf[offset..offset + 4].copy_from_slice(&value.to_be_bytes());
        self.store_dirty(&mut inner, 1, buf);
        Ok(())
    }

    /// Set the cache page budget (`None` is unbounded) and evict to fit.
    pub fn set_cache_pages(&self, n: Option<usize>) {
        let mut inner = self.inner.lock();
        inner.max_cached_pages = n;
        self.evict_if_over_budget(&mut inner);
    }

    /// Pin a page so it is never evicted.
    pub fn pin_page(&self, page_no: u32) {
        self.inner.lock().pinned.insert(page_no);
    }

    /// Unpin a page (the header page stays pinned regardless).
    pub fn unpin_page(&self, page_no: u32) {
        if page_no != HEADER_PAGE {
            self.inner.lock().pinned.remove(&page_no);
        }
    }

    /// Number of pages currently resident in the cache.
    pub fn cached_len(&self) -> usize {
        self.inner.lock().cache.len()
    }

    /// True when `page_no` is currently resident in the cache.
    pub fn is_cached(&self, page_no: u32) -> bool {
        self.inner.lock().cache.contains_key(&page_no)
    }

    /// Read `page_no`, serving from cache or reading and verifying from disk.
    ///
    /// A page read from disk is verified against its trailing checksum; a
    /// mismatch is reported as [`StoreError::CrcMismatch`].
    pub fn read_page(&self, page_no: u32) -> Result<Vec<u8>> {
        self.read_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut inner = self.inner.lock();
        let header = self.ensure_cached(&mut inner, 1)?;
        let db_size = u32::from_be_bytes(
            header[HDR_DB_SIZE_OFFSET..HDR_DB_SIZE_OFFSET + 4]
                .try_into()
                .unwrap(),
        );
        if page_no == 0 || page_no > db_size {
            return Err(StoreError::PageOutOfBounds { page_no, db_size });
        }
        self.ensure_cached(&mut inner, page_no)
    }

    /// Load a page's bytes from the WAL overlay / live WAL / main file and verify
    /// its trailing checksum. Does NOT touch the page cache — the caller decides
    /// whether to cache the result (`ensure_present`) or use it transiently (a
    /// scan that must not pollute or thrash the LRU cache). The integrity check is
    /// identical to the cached path: a mismatch is a typed `CrcMismatch`.
    fn load_and_verify_page(&self, page_no: u32) -> Result<Vec<u8>> {
        let buf = if self.read_only {
            match self.wal_overlay.get(&page_no) {
                Some(b) => b.clone(),
                None => {
                    // A main-file fallback can mix generations under a concurrent
                    // checkpoint. Re-validate the open-time snapshot fence first so
                    // a detected advance becomes a typed error, not a torn read.
                    self.validate_ro_snapshot_fence()?;
                    let mut b = vec![0u8; self.page_size];
                    let offset = self.offset_of(page_no);
                    self.with_file(|f| Ok(f.read_exact_at(&mut b, offset)?))?;
                    b
                }
            }
        } else {
            match self.wal_read_page(page_no)? {
                Some(b) => b,
                None => {
                    let mut b = vec![0u8; self.page_size];
                    let offset = self.offset_of(page_no);
                    self.with_file(|f| Ok(f.read_exact_at(&mut b, offset)?))?;
                    b
                }
            }
        };
        let actual = page_crc(&buf);
        let expected = stored_crc(&buf);
        if actual != expected {
            return Err(StoreError::CrcMismatch {
                page_no,
                expected,
                actual,
            });
        }
        Ok(buf)
    }

    /// Read `page_no` for a sequential SCAN: returns the page bytes from the
    /// cache if already resident, else loads + CRC-verifies them transiently
    /// WITHOUT inserting into the cache. A full-table scan (filtered COUNT,
    /// projected SELECT) on a corpus larger than the page cache therefore makes
    /// zero evictions — it neither pollutes nor thrashes the LRU. Without this,
    /// caching every scanned page on a corpus > cache forces an eviction (an
    /// O(cache) pass) per page, making the scan O(corpus * cache); the transient
    /// read keeps it strictly O(corpus). Integrity is identical to `read_page`:
    /// a transiently read page is CRC-verified before use.
    ///
    /// Correctness note: a resident page may carry uncommitted-but-cached writes
    /// the on-disk/WAL image does not yet reflect, so resident pages are always
    /// preferred; only a genuine miss falls through to the transient load, which
    /// reads the same committed image `read_page` would have cached.
    pub fn read_page_scan(&self, page_no: u32) -> Result<Vec<u8>> {
        {
            let mut inner = self.inner.lock();
            self.ensure_present(&mut inner, 1)?;
            let db_size = u32::from_be_bytes(
                inner.cache.get(&1).expect("header resident").buf
                    [HDR_DB_SIZE_OFFSET..HDR_DB_SIZE_OFFSET + 4]
                    .try_into()
                    .unwrap(),
            );
            if page_no == 0 || page_no > db_size {
                return Err(StoreError::PageOutOfBounds { page_no, db_size });
            }
            if let Some(entry) = inner.cache.get(&page_no) {
                return Ok(entry.buf.clone());
            }
        }
        // Miss: load transiently outside the cache lock; do NOT insert.
        self.load_and_verify_page(page_no)
    }

    /// Return a copy of `page_no` from the cache, loading and verifying from
    /// disk on a miss.
    fn ensure_cached(&self, inner: &mut Inner, page_no: u32) -> Result<Vec<u8>> {
        self.ensure_present(inner, page_no)?;
        // Resident and recency already bumped by `ensure_present`; hand the caller
        // its own owned copy.
        Ok(inner
            .cache
            .get(&page_no)
            .expect("page resident after ensure_present")
            .buf
            .clone())
    }

    /// Make `page_no` resident in the cache (loading and CRC-verifying from disk
    /// on a miss) and bump its recency — WITHOUT cloning the page out. Callers
    /// that only need to read a few header bytes (e.g. the leaf-chain count walk)
    /// use this plus an in-place peek to avoid the full per-page buffer clone that
    /// dominates a wide-table scan. The page integrity guarantee is identical to
    /// [`Pager::read_page`]: a miss reads the page and verifies its trailing
    /// checksum before it is cached; a hit reads from an already-verified buffer.
    fn ensure_present(&self, inner: &mut Inner, page_no: u32) -> Result<()> {
        if inner.cache.contains_key(&page_no) {
            inner.clock += 1;
            let clock = inner.clock;
            if let Some(entry) = inner.cache.get_mut(&page_no) {
                entry.used = clock;
            }
            return Ok(());
        }
        // Miss: a read-only open consults its committed-WAL overlay first, so a
        // reader sees data a writer logged but has not yet checkpointed. A
        // read-write open consults the live WAL writer index instead. Either
        // way, fall back to the main file.
        let buf = self.load_and_verify_page(page_no)?;
        inner.clock += 1;
        let used = inner.clock;
        // Insert by move — no extra clone of the freshly read buffer.
        inner.cache.insert(page_no, CacheEntry { buf, used });
        // Protect the just-inserted page: a caller that called `ensure_present`
        // is about to read it (clone it out, or read its header in place), so it
        // must survive this eviction pass even when the budget is tight and every
        // other slot is dirty/pinned.
        self.evict_if_over_budget_protecting(inner, Some(page_no));
        Ok(())
    }

    /// Count leaf cells across the sibling chain reading ONLY each leaf page's
    /// header — `num_cells` (u16) and the sibling pointer (u32) — never cloning a
    /// full page buffer. Each page is CRC-verified exactly as a normal read.
    ///
    /// SCAN-RESISTANT: a page already resident is read in place (no clone); a page
    /// NOT resident is loaded transiently (read + CRC-verified) and read WITHOUT
    /// being inserted into the cache. A full leaf-chain walk on a corpus larger
    /// than the cache therefore neither pollutes nor thrashes the LRU — it makes
    /// zero evictions. This is the dominant cost on a wide table: inserting every
    /// scanned page on a corpus > cache forces an eviction per page, and the
    /// eviction pass is O(cache), making the scan O(corpus * cache). The transient
    /// read keeps the walk strictly O(corpus).
    ///
    /// `first_leaf` is the leftmost-leaf page number; `num_cells_off` /
    /// `sibling_off` are the byte offsets of the 16-bit cell count and the 32-bit
    /// sibling pointer within a leaf page header. Returns the total cell count.
    pub fn count_leaf_cells(
        &self,
        first_leaf: u32,
        num_cells_off: usize,
        sibling_off: usize,
    ) -> Result<u64> {
        let mut inner = self.inner.lock();
        // Validate the requested chain stays within the live file, mirroring the
        // bound check `read_page` performs against the header's db_size.
        self.ensure_present(&mut inner, 1)?;
        let db_size = {
            let header = &inner.cache.get(&1).expect("header resident").buf;
            u32::from_be_bytes(
                header[HDR_DB_SIZE_OFFSET..HDR_DB_SIZE_OFFSET + 4]
                    .try_into()
                    .unwrap(),
            )
        };

        // Read the two header fields (num_cells: u16, sibling: u32) from a page,
        // resident or transient. `f` borrows the page bytes; this lets the
        // resident branch avoid any copy and the transient branch drop the buffer
        // immediately after.
        #[inline]
        fn header_fields(buf: &[u8], num_cells_off: usize, sibling_off: usize) -> (u64, u32) {
            let num_cells =
                u16::from_be_bytes(buf[num_cells_off..num_cells_off + 2].try_into().unwrap())
                    as u64;
            let sibling = u32::from_be_bytes(buf[sibling_off..sibling_off + 4].try_into().unwrap());
            (num_cells, sibling)
        }

        let mut total: u64 = 0;
        let mut page_no = first_leaf;
        while page_no != 0 {
            if page_no > db_size {
                return Err(StoreError::PageOutOfBounds { page_no, db_size });
            }
            let (num_cells, sibling) = if let Some(entry) = inner.cache.get(&page_no) {
                // Resident: read the header in place, no copy.
                header_fields(&entry.buf, num_cells_off, sibling_off)
            } else {
                // Not resident: load + CRC-verify transiently, do NOT cache. The
                // buffer is dropped at the end of this block.
                let buf = self.load_and_verify_page(page_no)?;
                header_fields(&buf, num_cells_off, sibling_off)
            };
            total += num_cells;
            page_no = sibling;
        }
        Ok(total)
    }

    /// Write a page into the cache and mark it dirty. The page is persisted on
    /// the next [`Pager::flush`]. `data` must be exactly one page in length;
    /// the trailing checksum slot is (re)stamped before caching.
    pub fn write_page(&self, page_no: u32, data: &[u8]) -> Result<()> {
        if data.len() != self.page_size {
            return Err(StoreError::Integrity {
                detail: format!(
                    "write_page: data length {} != page_size {}",
                    data.len(),
                    self.page_size
                ),
            });
        }
        self.capture_before_image(page_no)?;
        let buf = data.to_vec();
        let mut inner = self.inner.lock();
        // store_dirty stamps the trailing checksum slot.
        self.store_dirty(&mut inner, page_no, buf);
        Ok(())
    }

    /// Insert a dirty page into the cache, marking it dirty and recency-fresh.
    /// The trailing checksum slot is (re)stamped so the cached buffer always
    /// carries a valid checksum for the bytes it holds.
    fn store_dirty(&self, inner: &mut Inner, page_no: u32, mut buf: Vec<u8>) {
        stamp_crc(&mut buf);
        inner.clock += 1;
        let used = inner.clock;
        inner.cache.insert(page_no, CacheEntry { buf, used });
        inner.dirty.insert(page_no);
        self.evict_if_over_budget(inner);
    }

    /// Return `page_no` from the newest committed WAL frame, or `None` when the
    /// log holds no frame for it (or the pager is not in WAL mode).
    fn wal_read_page(&self, page_no: u32) -> Result<Option<Vec<u8>>> {
        let tx = self.tx.lock();
        match tx.wal.as_ref() {
            Some(wal) => wal.read_page(page_no),
            None => Ok(None),
        }
    }

    /// Allocate a fresh zero-filled page at the end of the file.
    ///
    /// Returns the new 1-indexed page number. A fresh buffer is always
    /// allocated so no stale bytes leak; the page is marked dirty so it is not
    /// evicted before the caller writes it, and `db_size` is incremented.
    pub fn extend_file(&self) -> Result<u32> {
        // Page 1 (the header) is mutated by the db_size bump below; capture its
        // before-image once per transaction so a rollback restores it.
        self.capture_before_image(1)?;
        let current = self.db_size()?;
        let new_page_no = current + 1;
        let buf = vec![0u8; self.page_size];
        {
            let mut inner = self.inner.lock();
            // store_dirty stamps the trailing checksum so a flush-before-write
            // (e.g. a freelist trunk recycle) still verifies on read.
            self.store_dirty(&mut inner, new_page_no, buf);
            let mut h = self.ensure_cached(&mut inner, 1)?;
            h[HDR_DB_SIZE_OFFSET..HDR_DB_SIZE_OFFSET + 4]
                .copy_from_slice(&new_page_no.to_be_bytes());
            self.store_dirty(&mut inner, 1, h);
        }
        Ok(new_page_no)
    }

    /// Evict least-recently-used clean, unpinned pages until the budget holds.
    /// Dirty and pinned pages are never evicted (correctness over budget).
    fn evict_if_over_budget(&self, inner: &mut Inner) {
        self.evict_if_over_budget_protecting(inner, None)
    }

    /// As [`Pager::evict_if_over_budget`], but `protected` (when set) is never
    /// evicted even if it is the LRU clean unpinned page. Used by the read path
    /// that has just made a page resident and is about to read its bytes in
    /// place: without this exemption, a tight budget whose other slots are all
    /// dirty/pinned would evict the very page just inserted, leaving the caller
    /// with nothing to read.
    fn evict_if_over_budget_protecting(&self, inner: &mut Inner, protected: Option<u32>) {
        let Some(max) = inner.max_cached_pages else {
            return;
        };
        if inner.cache.len() <= max {
            return;
        }
        // Collect evictable candidates (clean, unpinned, not the protected page)
        // sorted by recency.
        let mut candidates: Vec<(u64, u32)> = inner
            .cache
            .iter()
            .filter(|(pno, _)| {
                !inner.dirty.contains(*pno)
                    && !inner.pinned.contains(*pno)
                    && protected != Some(**pno)
            })
            .map(|(pno, e)| (e.used, *pno))
            .collect();
        candidates.sort_unstable_by_key(|(used, _)| *used);
        let mut over = inner.cache.len().saturating_sub(max);
        for (_, pno) in candidates {
            if over == 0 {
                break;
            }
            inner.cache.remove(&pno);
            over -= 1;
        }
    }

    /// Persist every dirty page durably, then clear the dirty set.
    ///
    /// When a transaction is open this is a no-op: durability is deferred to
    /// [`Pager::commit`], which routes through the WAL or the journal barrier.
    /// Outside a transaction it runs an implicit one-shot transaction so direct
    /// callers (and the tree mutations) still persist atomically.
    pub fn flush(&self) -> Result<()> {
        if self.in_transaction() {
            return Ok(());
        }
        // Implicit transaction: begin, capture nothing extra (writes already
        // landed in the cache), commit through the active mode.
        self.begin_write()?;
        self.commit()
    }

    /// Release the advisory single-writer lock held on the store file.
    ///
    /// Dropping the `Pager` releases the lock implicitly when the `File` closes,
    /// but a host that keeps the connection object alive (a reference cycle or a
    /// deferred GC) would otherwise hold the lock and block a fresh open on the
    /// same path in the same process. An explicit close calls this so the next
    /// open on that path succeeds without waiting for the object to be dropped.
    pub fn unlock(&self) -> Result<()> {
        // Drop the WAL writer first (consistent lock order: tx outer, file inner).
        // This closes the WAL sidecar fd immediately at connection-close time rather
        // than deferring it to Python GC. Read-only pagers have no WAL writer.
        if !self.read_only {
            self.tx.lock().wal = None;
        }
        // Release the advisory lock (read-only pagers never took one), then close
        // the main file fd so the OS handle is freed at connection-close time.
        let mut file_guard = self.file.lock();
        if !self.read_only {
            if let Some(ref f) = *file_guard {
                Fs4FileExt::unlock(f)?;
            }
        }
        *file_guard = None;
        Ok(())
    }

    /// Snapshot the current dirty set as `(page_no, bytes)` pairs.
    fn collect_dirty(&self) -> Vec<(u32, Vec<u8>)> {
        let inner = self.inner.lock();
        inner
            .dirty
            .iter()
            .filter_map(|pno| inner.cache.get(pno).map(|e| (*pno, e.buf.clone())))
            .collect()
    }

    /// Commit the open transaction. In WAL mode every dirty page is appended as
    /// a frame and the log issues one durable sync. In journal mode the
    /// three-fsync barrier persists before-images, stamps the page count, writes
    /// the dirty pages to the main file, then discards the journal.
    pub fn commit(&self) -> Result<()> {
        let dirty = self.collect_dirty();
        let db_size = self.db_size()?;

        let mut tx = self.tx.lock();
        if !tx.open {
            return Err(StoreError::Integrity {
                detail: "commit: no transaction is open".to_string(),
            });
        }

        if tx.wal_mode {
            if !dirty.is_empty() {
                let wal = tx.wal.as_mut().ok_or_else(|| StoreError::Integrity {
                    detail: "commit: WAL mode without a writer".to_string(),
                })?;
                if let Err(e) = wal.commit_transaction(&dirty, db_size) {
                    // A failed commit must not leave the running frame index
                    // advanced past the last durable frame: rewind to the
                    // committed baseline so a subsequent retry or rollback starts
                    // consistent and can never append duplicate frames past a
                    // torn gap.
                    wal.rollback();
                    return Err(e);
                }
                if wal.committed_frame_count() >= crate::consts::WAL_AUTOCHECKPOINT_FRAMES as u64 {
                    self.with_file(|f| wal.checkpoint(f))?;
                }
            }
            tx.open = false;
            drop(tx);
            self.inner.lock().dirty.clear();
            return Ok(());
        }

        // Rollback-journal mode: the three-fsync barrier.
        if let Some(journal) = tx.journal.as_mut() {
            // BARRIER 1: before-images durable.
            journal.flush_pending()?;
            journal.sync()?;
            // BARRIER 2: page count durable (arms recovery).
            journal.write_page_count()?;
            journal.sync()?;
        }
        // Write dirty pages to the main file, then BARRIER 3.
        for (pno, buf) in &dirty {
            let offset = self.offset_of(*pno);
            self.with_file(|f| Ok(f.write_all_at(buf, offset)?))?;
        }
        self.with_file(|f| durable_fsync(f))?;

        // Discard the journal — the commit point. The unlink and its directory
        // entry must be durable before commit reports success: a surviving
        // committed journal would roll this transaction back on the next open.
        let journal_path = journal_path_for(&self.path);
        tx.journal = None;
        match std::fs::remove_file(&journal_path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e.into()),
        }
        fsync_parent_directory(&journal_path)?;
        tx.open = false;
        drop(tx);
        self.inner.lock().dirty.clear();
        Ok(())
    }

    /// Abort the open transaction, restoring the store to its pre-transaction
    /// state. In journal mode the captured before-images are restored into the
    /// cache and the file is truncated to its pre-transaction page count; in WAL
    /// mode the in-progress (un-synced) frames are rewound and the dirty set is
    /// dropped.
    pub fn rollback(&self) -> Result<()> {
        let mut tx = self.tx.lock();
        if !tx.open {
            return Ok(());
        }
        let pre_size = tx.pre_txn_db_size;

        if tx.wal_mode {
            if let Some(wal) = tx.wal.as_mut() {
                wal.rollback();
            }
            tx.open = false;
            drop(tx);
            // Drop every dirty page from the cache so the next read reloads the
            // last committed state (from the WAL index or the main file).
            let mut inner = self.inner.lock();
            let dirty: Vec<u32> = inner.dirty.iter().copied().collect();
            for pno in dirty {
                inner.cache.remove(&pno);
            }
            inner.dirty.clear();
            return Ok(());
        }

        // Journal mode: restore before-images into the cache and discard the rest.
        let before: Vec<(u32, Vec<u8>)> = tx
            .journal
            .as_ref()
            .map(|j| j.before_images().to_vec())
            .unwrap_or_default();
        let journal_path = journal_path_for(&self.path);
        tx.journal = None;
        tx.open = false;
        drop(tx);

        let mut inner = self.inner.lock();
        // Drop all dirty pages from the cache.
        let dirty: Vec<u32> = inner.dirty.iter().copied().collect();
        for pno in dirty {
            inner.cache.remove(&pno);
        }
        inner.dirty.clear();
        // Restore each before-image as a clean cached page.
        for (pno, buf) in before {
            inner.clock += 1;
            let used = inner.clock;
            inner.cache.insert(pno, CacheEntry { buf, used });
        }
        drop(inner);
        // Truncate away any pages the transaction appended past the prior extent.
        let cur_len = self.with_file(|f| Ok(f.metadata()?.len()))?;
        let want_len = pre_size as u64 * self.page_size as u64;
        if cur_len > want_len {
            self.with_file(|f| Ok(f.set_len(want_len)?))?;
        }
        let _ = std::fs::remove_file(&journal_path);
        Ok(())
    }

    /// Merge committed WAL frames into the main file (WAL mode only; a no-op
    /// otherwise). Resets the sidecar in place and keeps its fd open.
    pub fn checkpoint(&self) -> Result<()> {
        let mut tx = self.tx.lock();
        if let Some(wal) = tx.wal.as_mut() {
            self.with_file(|f| wal.checkpoint(f))?;
        }
        Ok(())
    }

    /// The store file path.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The page size in bytes.
    pub fn page_size(&self) -> usize {
        self.page_size
    }

    /// The number of [`Pager::read_page`] calls so far.
    pub fn read_count(&self) -> u64 {
        self.read_counter.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the read counter to zero.
    pub fn reset_read_count(&self) {
        self.read_counter
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record one whole-tree ascending walk. Called by the cursor's full scan.
    pub fn note_full_scan(&self) {
        self.full_scan_counter
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }

    /// The number of whole-tree ascending walks so far.
    pub fn full_scan_count(&self) -> u64 {
        self.full_scan_counter
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the full-scan counter to zero.
    pub fn reset_full_scan_count(&self) {
        self.full_scan_counter
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }

    /// Record `n` leaf cells visited by a streaming scan. Called by the
    /// cursor's `scan_cells_with` once per visited cell.
    pub fn note_cells_visited(&self, n: u64) {
        self.cells_visited_counter
            .fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// The number of leaf cells visited by streaming scans so far.
    pub fn cells_visited_count(&self) -> u64 {
        self.cells_visited_counter
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Reset the cells-visited counter to zero.
    pub fn reset_cells_visited_count(&self) {
        self.cells_visited_counter
            .store(0, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_db() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("brain.lilli");
        (dir, path)
    }

    #[test]
    fn ro_fence_prefix_is_the_python_pool_contract() {
        // iai_mcp/hippo/_ro_pool.py pins _RO_SNAPSHOT_FENCE_MARKER to this
        // exact string; a rename must update both sides in lockstep.
        assert_eq!(RO_SNAPSHOT_FENCE_PREFIX, "read-only snapshot invalidated");
    }

    #[test]
    fn pager_open_writes_header_and_reopens() {
        let (_d, path) = temp_db();
        {
            let p = Pager::open(&path).unwrap();
            assert_eq!(p.db_size().unwrap(), 3);
            let hdr = p.read_page(1).unwrap();
            assert_eq!(&hdr[..DB_MAGIC.len()], &DB_MAGIC);
        }
        // Reopen: header round-trips.
        let p = Pager::open(&path).unwrap();
        assert_eq!(p.db_size().unwrap(), 3);
        let hdr = p.read_page(1).unwrap();
        let ps = u16::from_be_bytes(
            hdr[HDR_PAGE_SIZE_OFFSET..HDR_PAGE_SIZE_OFFSET + 2]
                .try_into()
                .unwrap(),
        );
        assert_eq!(ps as usize, PAGE_SIZE);
    }

    #[test]
    fn pager_write_read_roundtrip_through_cache_and_after_flush() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        let pno = p.extend_file().unwrap();
        let mut data = vec![0u8; PAGE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 251) as u8;
        }
        // Preserve the checksum slot expectation: write_page stamps it, so the
        // returned read equals the stamped buffer.
        p.write_page(pno, &data).unwrap();
        let got = p.read_page(pno).unwrap();
        let mut expected = data.clone();
        stamp_crc(&mut expected);
        assert_eq!(got, expected);

        // Flush, drop the cache by reopening, read back from disk.
        p.flush().unwrap();
        drop(p);
        let p2 = Pager::open(&path).unwrap();
        let got2 = p2.read_page(pno).unwrap();
        assert_eq!(got2, expected);
    }

    #[test]
    fn pager_crc_detects_flipped_byte_on_disk() {
        let (_d, path) = temp_db();
        let pno;
        {
            let p = Pager::open(&path).unwrap();
            pno = p.extend_file().unwrap();
            let data = vec![7u8; PAGE_SIZE];
            p.write_page(pno, &data).unwrap();
            p.flush().unwrap();
        }
        // Corrupt one byte of the persisted page directly on disk.
        let offset = (pno as u64 - 1) * PAGE_SIZE as u64;
        let f = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut one = [0u8; 1];
        f.read_exact_at(&mut one, offset + 10).unwrap();
        one[0] ^= 0xFF;
        f.write_all_at(&one, offset + 10).unwrap();
        f.sync_all().unwrap();
        drop(f);

        let p2 = Pager::open(&path).unwrap();
        let err = p2.read_page(pno).unwrap_err();
        match err {
            StoreError::CrcMismatch { page_no, .. } => assert_eq!(page_no, pno),
            other => panic!("expected CrcMismatch, got {other:?}"),
        }
    }

    #[test]
    fn pager_dirty_and_pinned_pages_survive_eviction() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        p.set_cache_pages(Some(2));

        // Create five distinct pages and write to them.
        let mut pages = vec![];
        for _ in 0..5 {
            let pno = p.extend_file().unwrap();
            p.write_page(pno, &vec![1u8; PAGE_SIZE]).unwrap();
            pages.push(pno);
        }
        // Mark the first as pinned; flush so the rest become clean & evictable.
        let pinned = pages[0];
        p.pin_page(pinned);

        // Keep one page dirty: rewrite it after the flush.
        p.flush().unwrap();
        let dirty = pages[1];
        p.write_page(dirty, &vec![2u8; PAGE_SIZE]).unwrap();

        // Touch the remaining clean pages to drive eviction.
        for &pno in &pages[2..] {
            let _ = p.read_page(pno).unwrap();
        }

        // Pinned and dirty pages must still be resident.
        assert!(p.is_cached(pinned), "pinned page must survive eviction");
        assert!(p.is_cached(dirty), "dirty page must survive eviction");
    }

    #[test]
    fn pager_extend_file_returns_fresh_zero_buffer() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        let before = p.db_size().unwrap();
        let pno = p.extend_file().unwrap();
        assert_eq!(pno, before + 1);
        assert_eq!(p.db_size().unwrap(), before + 1);
        let buf = p.read_page(pno).unwrap();
        // Everything except the trailing checksum slot is zero.
        assert!(buf[..PAGE_CRC_OFFSET].iter().all(|&b| b == 0));
    }

    #[test]
    fn pager_bounded_cache_stays_bounded_under_write_loop() {
        let (_d, path) = temp_db();
        let p = Pager::open(&path).unwrap();
        p.set_cache_pages(Some(4));
        // Write many pages, flushing so they are clean and evictable.
        for _ in 0..50 {
            let pno = p.extend_file().unwrap();
            p.write_page(pno, &vec![9u8; PAGE_SIZE]).unwrap();
            p.flush().unwrap();
            // Read it back so it is clean-resident, then keep going.
            let _ = p.read_page(pno).unwrap();
        }
        // Header (pinned) plus at most the budget of clean pages. The cache must
        // not have grown to the full corpus.
        assert!(
            p.cached_len() <= 4 + 2,
            "cache grew unbounded: {}",
            p.cached_len()
        );
    }

    #[test]
    fn pager_open_read_only_reads_committed_pages_without_a_lock() {
        let (_d, path) = temp_db();
        // Writer formats the file and commits a page, then keeps the store open
        // (holding the exclusive lock), exactly as a live writer process would.
        let writer = Pager::open(&path).unwrap();
        let pno = writer.extend_file().unwrap();
        let mut data = vec![0u8; PAGE_SIZE];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i % 97) as u8;
        }
        writer.write_page(pno, &data).unwrap();
        writer.flush().unwrap();

        // A read-only open succeeds while the writer still holds the file —
        // it takes no lock, so it never blocks on the writer's exclusive lock.
        let reader = Pager::open_read_only(&path).unwrap();
        assert!(reader.is_read_only());
        let mut expected = data.clone();
        stamp_crc(&mut expected);
        assert_eq!(reader.read_page(pno).unwrap(), expected);

        // The writer's exclusive lock is untouched: a fresh exclusive open on the
        // same path still fails while the writer holds it.
        let blocked = Pager::open(&path);
        assert!(blocked.is_err(), "writer's exclusive lock must still hold");
    }

    #[test]
    fn pager_read_only_rejects_writes() {
        let (_d, path) = temp_db();
        {
            let _ = Pager::open(&path).unwrap(); // format + release lock on drop
        }
        let reader = Pager::open_read_only(&path).unwrap();
        // A read-only pager refuses to begin a write transaction.
        assert!(reader.begin_write().is_err());
        // Its unlock is a no-op (it never locked) — must not error.
        reader.unlock().unwrap();
    }

    #[test]
    fn pager_open_read_only_missing_file_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("does-not-exist.lilli");
        assert!(Pager::open_read_only(&path).is_err());
    }

    #[test]
    fn ro_fence_fires_even_when_the_page_is_already_cache_resident() {
        // Regression: `validate_ro_snapshot_fence` alone only runs inside
        // `load_and_verify_page`'s cache-MISS branch. A page already resident
        // in the reader's own cache (from an earlier read on the same open
        // pager) bypassed the fence entirely on every later read of that same
        // page, silently serving the pre-checkpoint bytes forever instead of
        // detecting the invalidation. `revalidate_ro_snapshot_fence` must
        // catch it unconditionally, independent of cache residency.
        let (_d, path) = temp_db();
        let writer = Pager::open(&path).unwrap();
        writer.enable_wal_mode().unwrap();

        let pno = writer.extend_file().unwrap();
        writer.begin_write().unwrap();
        writer.write_page(pno, &vec![1u8; PAGE_SIZE]).unwrap();
        writer.commit().unwrap();
        writer.checkpoint().unwrap(); // merge into main file, reset WAL sidecar

        let reader = Pager::open_read_only(&path).unwrap();
        // First read: genuine cache miss, populates the reader's page cache.
        let first = reader.read_page(pno).unwrap();
        assert_eq!(first[0], 1u8);
        // A second read of the SAME page before any checkpoint is a cache
        // HIT and must not spuriously fail the fence.
        reader.revalidate_ro_snapshot_fence().unwrap();

        // A concurrent writer rewrites the SAME page in place, then
        // checkpoints — the exact reinforce/boost-then-checkpoint shape.
        writer.begin_write().unwrap();
        writer.write_page(pno, &vec![2u8; PAGE_SIZE]).unwrap();
        writer.commit().unwrap();
        writer.checkpoint().unwrap();

        // The unconditional revalidation entry point must detect the
        // invalidation regardless of whether `pno` happens to be cached.
        let result = reader.revalidate_ro_snapshot_fence();
        assert!(
            result.is_err(),
            "fence must fire after a concurrent checkpoint even though the \
             reader already has this page cached from its first read"
        );

        // And the cache-hit read path itself must still reflect this: a
        // caller that skips the explicit revalidation and just re-reads the
        // page directly must not silently receive the stale cached copy.
        // (This mirrors the statement-dispatch entry point in lilliengine,
        // which calls revalidate_ro_snapshot_fence() before every read-only
        // statement rather than relying on a per-page fence.)
        let cached_reread = reader.read_page(pno).unwrap();
        assert_eq!(
            cached_reread[0], 1u8,
            "the pager's own page cache is unconditionally trusted once \
             resident -- proving the fence gap lives in the CALLER needing \
             its own unconditional check, not in read_page itself"
        );
    }
}
