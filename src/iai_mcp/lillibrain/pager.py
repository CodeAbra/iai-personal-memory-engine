"""Page cache, dirty tracking, and rollback-journal commit for the LilliBrain engine.

Single-writer model enforced by the caller. The RLock guards against re-entrant
calls from cursor helpers within one thread but does not provide cross-thread
serialisation — external locking is required for multi-threaded writers.

Journal sidecar (.lilli.journal) is created with mode 0o600 during write
transactions and deleted at the commit point. Crash recovery runs in __init__:
a page_count==0 journal is discarded (uncommitted); a page_count>0 journal
replays before-images into the DB file before any reads are served.
"""
from __future__ import annotations

import errno
import logging
import mmap
import os
import struct
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

from iai_mcp.lillibrain.constants import (
    DB_MAGIC,
    HDR_DB_SIZE_OFFSET,
    HDR_FIRST_FREELIST_OFFSET,
    HDR_FREELIST_COUNT_OFFSET,
    HDR_MAGIC_OFFSET,
    HDR_PAGE_SIZE_OFFSET,
    HDR_READ_VERSION_OFFSET,
    HDR_WRITE_VERSION_OFFSET,
    HEADER_SIZE,
    PAGE_SIZE,
    WAL_AUTOCHECKPOINT_FRAMES,
    WAL_WRITE_VERSION,
)
from iai_mcp.lillibrain.btree.page import write_leaf_header
from iai_mcp.lillibrain.io import durable_fsync, fsync_directory

if TYPE_CHECKING:
    from iai_mcp.lillibrain.wal import WALWriter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Journal format constants
# ---------------------------------------------------------------------------

_JOURNAL_MAGIC = b"LILLI-JRNL\x00\x00"  # 12 bytes
_JOURNAL_HEADER_SIZE = 16               # magic(12) + page_count(4)
_JOURNAL_PAGE_RECORD_SIZE = 4           # page_no(4) prefix per record (+ page data)

# ---------------------------------------------------------------------------
# DB header field registry: name → (offset, byte_width, struct_fmt)
# ---------------------------------------------------------------------------

_HDR_FIELDS: dict[str, tuple[int, int, str]] = {
    "page_size":            (HDR_PAGE_SIZE_OFFSET,        2, ">H"),
    "write_version":        (HDR_WRITE_VERSION_OFFSET,    1, ">B"),
    "read_version":         (HDR_READ_VERSION_OFFSET,     1, ">B"),
    "db_size":              (HDR_DB_SIZE_OFFSET,          4, ">I"),
    "first_freelist_trunk": (HDR_FIRST_FREELIST_OFFSET,   4, ">I"),
    "freelist_count":       (HDR_FREELIST_COUNT_OFFSET,   4, ">I"),
}


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------


class Journal:
    """Rollback journal sidecar for an in-progress write transaction.

    Created by Pager.begin_write(); committed or rolled back via
    Pager.commit() / Pager.rollback(). Do not construct directly.

    Before-images are buffered in memory during the transaction and written
    to disk in a single contiguous write during commit() before BARRIER 1.
    This eliminates per-page os.pwrite latency during the transaction body
    when many pages are dirtied in the same batch (e.g. executemany UPDATE).
    """

    def __init__(self, journal_path: Path, fd: int) -> None:
        self._path = journal_path
        self._fd = fd
        self._page_count: int = 0
        # Set of page numbers whose before-image has already been captured.
        # Only the first (original) before-image matters for recovery; a second
        # snapshot of the same page would record an already-mutated state and
        # cause recovery to replay to the wrong intermediate value.
        self._journaled: set[int] = set()
        # In-memory accumulator for before-images; flushed to disk in commit().
        # List of (page_no, snapshot_bytes) in insertion order.
        self._pending: list[tuple[int, bytes]] = []

        # Write the journal header with page_count=0 (uncommitted sentinel).
        # The header will be overwritten with the real count after BARRIER 1.
        header = _JOURNAL_MAGIC + struct.pack(">I", 0)
        os.pwrite(self._fd, header, 0)

    def record_before_image(self, page_no: int, data: bytes | bytearray) -> None:
        """Buffer the before-image of page_no in memory.

        Must be called before the corresponding page is mutated in the cache.
        If page_no has already been buffered in this transaction the call is a
        no-op: only the first (original) snapshot is needed for correct recovery.

        The actual disk write is deferred to flush_pending(), which is called
        by Pager.commit() before BARRIER 1. This batches all before-image writes
        into a single contiguous os.write instead of N scattered os.pwrite calls.
        """
        if page_no in self._journaled:
            return
        self._journaled.add(page_no)
        self._pending.append((page_no, bytes(data)))
        self._page_count += 1

    def flush_pending(self) -> None:
        """Write all buffered before-images to the journal file in one pass.

        Called by Pager.commit() immediately before BARRIER 1. Assembles all
        pending (page_no, data) pairs into a single contiguous buffer and writes
        it with one os.write call to minimise syscall and fsync overhead.

        After this call, the journal body is on disk and ready for the BARRIER 1
        fsync that makes it durable before any DB pages are overwritten.
        """
        if not self._pending:
            return
        # Determine page_size from first entry's data length.
        page_size = len(self._pending[0][1])
        record_size = 4 + page_size  # page_no(4) + page_data
        buf = bytearray(len(self._pending) * record_size)
        offset = 0
        for page_no, data in self._pending:
            struct.pack_into(">I", buf, offset, page_no)
            offset += 4
            buf[offset : offset + len(data)] = data
            offset += len(data)
        # Write starting after the journal header.
        os.pwrite(self._fd, bytes(buf), _JOURNAL_HEADER_SIZE)
        self._pending.clear()

    @property
    def fd(self) -> int:
        """Raw file descriptor for the journal (used by Pager for fsync targeting)."""
        return self._fd

    @property
    def page_count(self) -> int:
        """Number of before-images recorded so far."""
        return self._page_count


# ---------------------------------------------------------------------------
# Pager
# ---------------------------------------------------------------------------


class Pager:
    """Page cache and rollback-journal transaction manager.

    Opens (or creates) a single ``.lilli`` database file. A sidecar
    ``.lilli.journal`` is created during write transactions and deleted at
    the commit point. Both files are created with mode 0o600.

    Not thread-safe: the caller must serialise access across threads. The
    internal RLock guards against re-entrant calls from cursor helpers within
    a single thread.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        page_size: int = PAGE_SIZE,
        use_mapped_reads: bool | None = None,
        max_cached_pages: int | None = None,
    ) -> None:
        self._path = Path(path)
        self._page_size = page_size
        # Insertion-ordered cache so least-recently-used eviction is O(1):
        # every read/write moves its page to the end; eviction drops from the
        # front. None bound preserves the historical unbounded behaviour.
        self._page_cache: OrderedDict[int, bytearray] = OrderedDict()
        self._dirty: set[int] = set()
        # Pages that must never be evicted regardless of budget: the header
        # page (re-readable, but kept hot for cheap field access) and any
        # rollback-restored before-image (authoritative until the next
        # transaction; not re-materialisable from disk/WAL).
        self._pinned: set[int] = {1}
        # Maximum number of cached pages. None = unbounded. A clean page is
        # evicted from the LRU end once the cache exceeds this bound; dirty and
        # pinned pages are never evicted (correctness over budget).
        self._max_cached_pages: int | None = max_cached_pages
        self._write_lock: threading.RLock = threading.RLock()
        self._closed: bool = False
        self._journal: Journal | None = None

        # Write-ahead mode: when True, commit/rollback delegate to _wal instead
        # of running the rollback-journal sequence.  Default False (journal path).
        self._wal_mode: bool = False
        self._wal: WALWriter | None = None

        # root_page_no is fixed — page 1 is the DB header page; the B+tree root
        # lives at page 2 to avoid overwriting the DB header when cursor code
        # writes a leaf page header at offset 0 of the root page.
        self.root_page_no: int = 2

        # Opt-in read-only memory-mapped read path.  When enabled, clean pages
        # are served as read-only memoryview slices of a kernel-backed mapping of
        # the main file, keeping the heap dirty dict bounded by the open
        # transaction's write-set rather than the corpus size.
        #
        # Precedence (correctness order, highest first):
        #   1. dirty heap dict — uncommitted mutations visible within the transaction
        #   2. WAL index (_wal.read_page) — newest committed frame from the sidecar
        #   3. mapped slice — read-only ACCESS_READ map of the main file
        #
        # The mapping is created only after crash recovery has run, refreshed on
        # every file-size change (extend_file, checkpoint, append on commit), and
        # released before the db fd is closed.  A view returned from read_page is
        # valid only until the next commit / checkpoint / file-grow.
        #
        # When the flag is OFF (default) the existing pread + heap-cache path is
        # used without modification — this is the production default.
        if use_mapped_reads is None:
            use_mapped_reads = os.getenv("LILLI_MMAP_READS", "").lower() in ("1", "true", "yes")
        self._use_mmap: bool = use_mapped_reads
        self._mmap: mmap.mmap | None = None

        # Open or create the database file with owner-only permissions.
        raw_fd = os.open(str(self._path), os.O_CREAT | os.O_RDWR, 0o600)
        os.chmod(str(self._path), 0o600)
        self._db_fd: int = raw_fd

        existing_size = os.fstat(self._db_fd).st_size

        if existing_size == 0:
            # Fresh file — initialise page 1 with the 100-byte header.
            self._init_fresh_file()
            # After fresh init, create the mapping if the flag is ON.
            if self._use_mmap:
                self._refresh_mmap()
        else:
            # Existing file — read the page_size from the on-disk header before
            # loading anything else so we use the correct page geometry.
            # Peek at exactly 18 bytes (magic + 2-byte page_size field).
            hdr_peek = os.pread(self._db_fd, HDR_PAGE_SIZE_OFFSET + 2, 0)
            if len(hdr_peek) >= HDR_PAGE_SIZE_OFFSET + 2:
                on_disk_ps = struct.unpack_from(">H", hdr_peek, HDR_PAGE_SIZE_OFFSET)[0]
                if on_disk_ps >= 512:  # sanity: must be a power-of-two >= 512
                    self._page_size = on_disk_ps
            # Run crash recovery before serving any reads.
            self._recover_if_needed()
            # Create the mapping AFTER recovery so it reflects the recovered file.
            if self._use_mmap:
                self._refresh_mmap()
            # Load page 1 into the cache so header fields are accessible.
            self._load_page_from_disk(1)

        logger.debug("Pager opened: %s (page_size=%d)", self._path, self._page_size)

    # ---------------------------------------------------------------------------
    # Public properties (read-only views of internal state, for tests)
    # ---------------------------------------------------------------------------

    @property
    def page_cache(self) -> "OrderedDict[int, bytearray]":
        """The in-memory page cache (page_no → bytearray), LRU-ordered."""
        return self._page_cache

    @property
    def pinned(self) -> set[int]:
        """Page numbers that are never evicted (header + rollback-restored)."""
        return self._pinned

    def set_max_cached_pages(self, n: int | None) -> None:
        """Set the page-cache size bound (None = unbounded) and evict to fit.

        Called from the connection's PRAGMA cache_size handler so the store's
        cache budget actually bounds the engine's resident page set instead of
        being silently dropped. Setting a smaller bound evicts immediately.
        """
        self._max_cached_pages = n
        self._evict_if_over_budget()

    def _touch(self, page_no: int) -> None:
        """Mark page_no as most-recently-used for LRU ordering."""
        # move_to_end is a no-op-safe O(1) reorder; only call when present.
        if page_no in self._page_cache:
            self._page_cache.move_to_end(page_no)

    def _evict_if_over_budget(self) -> None:
        """Evict clean, unpinned pages from the LRU end until at/under budget.

        A page is evictable only when it is neither dirty (holds an uncommitted
        write) nor pinned (rollback-restored before-image / header). A clean
        unpinned page is always re-materialisable — from the WAL sidecar in
        write-ahead mode, or from the main file via pread otherwise — so
        dropping it is lossless. When every remaining page is dirty or pinned,
        eviction stops and the cache is allowed to exceed the budget: correctness
        always wins over the size bound. The dirty set is bounded by the open
        transaction's write-set, so this cannot grow without limit.
        """
        if self._max_cached_pages is None:
            return
        if len(self._page_cache) <= self._max_cached_pages:
            return
        # Iterate from the LRU (front) end. Collect evictable victims first to
        # avoid mutating the OrderedDict while iterating it.
        to_evict: list[int] = []
        target = self._max_cached_pages
        for page_no in self._page_cache:
            if len(self._page_cache) - len(to_evict) <= target:
                break
            if page_no in self._dirty or page_no in self._pinned:
                continue
            to_evict.append(page_no)
        for page_no in to_evict:
            self._page_cache.pop(page_no, None)

    @property
    def dirty(self) -> set[int]:
        """Set of page numbers that have been written since the last commit."""
        return self._dirty

    @property
    def _journal_fd(self) -> int | None:
        """Journal file descriptor (exposed for test tracking; None when no transaction)."""
        return self._journal.fd if self._journal is not None else None

    @property
    def mmap_active(self) -> bool:
        """True when the read-only mapped read path is enabled and a live mapping exists."""
        return self._use_mmap and self._mmap is not None

    # ---------------------------------------------------------------------------
    # Initialisation helpers
    # ---------------------------------------------------------------------------

    def _init_fresh_file(self) -> None:
        """Initialise page 1 on a brand-new database file."""
        page = bytearray(self._page_size)

        # DB_MAGIC at offset 0
        page[HDR_MAGIC_OFFSET : HDR_MAGIC_OFFSET + len(DB_MAGIC)] = DB_MAGIC

        # page_size (2 bytes, big-endian)
        struct.pack_into(">H", page, HDR_PAGE_SIZE_OFFSET, self._page_size)

        # write_version = 1 (rollback-journal mode)
        struct.pack_into(">B", page, HDR_WRITE_VERSION_OFFSET, 1)

        # read_version = 1
        struct.pack_into(">B", page, HDR_READ_VERSION_OFFSET, 1)

        # db_size = 2 (page 1 = header, page 2 = B+tree root leaf)
        struct.pack_into(">I", page, HDR_DB_SIZE_OFFSET, 2)

        # first_freelist_trunk = 0, freelist_count = 0 (already zero)

        # Write page 1 directly to disk
        os.pwrite(self._db_fd, bytes(page), 0)

        # Allocate page 2 as the B+tree root leaf with a valid empty-leaf header.
        # Writing a proper header here ensures check_integrity() returns [] on a
        # freshly opened database before any insert has occurred.
        root_page = bytearray(self._page_size)
        write_leaf_header(root_page, num_cells=0, cell_content_start=self._page_size)
        os.pwrite(self._db_fd, bytes(root_page), self._page_size)

        durable_fsync(self._db_fd)

        # Cache both pages
        self._page_cache[1] = page
        self._page_cache[2] = root_page

    def _load_page_from_disk(self, page_no: int) -> bytearray:
        """Read page_no from the file into the cache and return the buffer."""
        offset = (page_no - 1) * self._page_size
        raw = os.pread(self._db_fd, self._page_size, offset)
        buf = bytearray(self._page_size)
        buf[: len(raw)] = raw
        self._page_cache[page_no] = buf
        self._touch(page_no)
        self._evict_if_over_budget()
        return buf

    # ---------------------------------------------------------------------------
    # Mapped read path helpers
    # ---------------------------------------------------------------------------

    def _refresh_mmap(self) -> None:
        """Create or recreate the read-only mapping over the current file contents.

        Replaces any existing mapping with a new one over the current file size.
        The old mapping object is dropped (and will be garbage-collected once all
        memoryview slices derived from it are also released).  Any memoryview
        slice issued before this call is therefore backed by the old mapping
        object; accessing it after this call returns stale bytes — this is the
        documented lifetime contract.

        Does nothing when the mapped read path is not enabled.  Skips mapping
        when the file is zero-length (a zero-length mmap is rejected by the OS).
        """
        if not self._use_mmap:
            return
        file_size = os.fstat(self._db_fd).st_size
        if file_size == 0:
            self._mmap = None
            return
        # Drop the old reference.  If a live memoryview still holds the old
        # mapping open, Python defers the underlying close until that view is
        # released — this is correct: the mapping keeps the bytes accessible
        # until the last consumer drops their view, at which point the OS
        # reclaims the mapping.  Callers must not rely on stale views being
        # valid after a remap.
        self._mmap = mmap.mmap(self._db_fd, 0, access=mmap.ACCESS_READ)

    def _close_mmap(self) -> None:
        """Release the active mapping reference.  Idempotent.

        Calls mmap.close() explicitly so the OS mapping is released promptly
        on non-CPython runtimes where GC is non-deterministic.  If a live
        memoryview still holds a reference to the mapping, mmap.close() raises
        BufferError — caught here so the caller is not interrupted; the OS
        releases the mapping when the last memoryview is garbage-collected.
        """
        if self._mmap is not None:
            try:
                self._mmap.close()
            except (OSError, BufferError):
                pass  # live memoryview outstanding — OS releases on last ref drop
            self._mmap = None

    # ---------------------------------------------------------------------------
    # Write-ahead mode
    # ---------------------------------------------------------------------------

    def enable_wal_mode(self) -> None:
        """Switch this pager to write-ahead mode.

        Constructs the WALWriter, sets the mode flag, and persists the mode by
        writing the write-ahead version value into both version bytes of the DB
        header page then durably flushing page 1 to disk.  After this call, the
        mode survives a close and reopen because the on-disk header encodes it.

        Idempotent — calling more than once is a no-op.
        """
        if self._wal_mode:
            return

        from iai_mcp.lillibrain.wal import WALWriter  # noqa: PLC0415

        self._wal = WALWriter(self._path, page_size=self._page_size)
        self._wal_mode = True

        # Write both version bytes directly into the page 1 cache buffer so
        # the values are visible to any subsequent read_db_header_field call
        # within this session.  Use the mutable heap-cache buffer — read_page
        # may return a read-only memoryview when the mapped path is active.
        page1 = self._writable_cache_page(1)
        struct.pack_into(">B", page1, HDR_WRITE_VERSION_OFFSET, WAL_WRITE_VERSION)
        struct.pack_into(">B", page1, HDR_READ_VERSION_OFFSET, WAL_WRITE_VERSION)
        self._dirty.add(1)

        # Durably flush page 1 NOW so the mode persists even if the process
        # crashes before any data commit.  Write directly to the DB file;
        # do not go through the WAL writer (this is a header-only durability write,
        # not a user-data transaction).
        os.pwrite(self._db_fd, bytes(page1), 0)
        durable_fsync(self._db_fd)

    # ---------------------------------------------------------------------------
    # Crash recovery
    # ---------------------------------------------------------------------------

    def _recover_if_needed(self) -> None:
        """Replay any crash-interrupted transaction on open.

        Routes on the header write_version byte:
        - Write-ahead mode (version byte == WAL_WRITE_VERSION): if a sidecar
          exists, runs write-ahead recovery (scan, replay committed frames,
          unlink sidecar) and returns.
        - Rollback-journal mode (any other value): if a .journal sidecar exists,
          replays before-images into the DB file and deletes the journal.
        - Neither sidecar present: no-op.

        Called from __init__ before any reads are served. Idempotent — a second
        call after a successful first recovery finds no sidecar and returns
        immediately.
        """
        # Peek the write_version byte to select the recovery path.
        version_raw = os.pread(self._db_fd, 1, HDR_WRITE_VERSION_OFFSET)
        if len(version_raw) == 1 and version_raw[0] == WAL_WRITE_VERSION:
            # Write-ahead mode: delegate to the standalone WAL recovery scanner.
            wal_sidecar = self._path.parent / (self._path.name + "-wal")
            if wal_sidecar.exists():
                from iai_mcp.lillibrain.wal import recover_wal_on_open  # noqa: PLC0415
                recover_wal_on_open(self._path, wal_sidecar, self._page_size)
            return

        # Rollback-journal mode: unchanged path below.
        journal_path = self._journal_path()
        if not journal_path.exists():
            return

        try:
            journal_data = journal_path.read_bytes()
        except OSError as exc:
            if exc.errno == errno.ENOENT:
                # Vanished between the exists() check and the read (rare TOCTOU) —
                # treat as genuinely absent.
                return
            # A present-but-unreadable journal leaves the interrupted transaction
            # unrolled; surface it instead of opening over half-written pages.
            logger.warning(
                "rollback journal unreadable, interrupted transaction NOT rolled "
                "back: %s: %s",
                journal_path,
                exc,
            )
            return

        if len(journal_data) < _JOURNAL_HEADER_SIZE:
            # Too short to be valid — discard.
            journal_path.unlink(missing_ok=True)
            return

        magic = journal_data[:12]
        if magic != _JOURNAL_MAGIC:
            # Not our journal — discard.
            journal_path.unlink(missing_ok=True)
            return

        (page_count,) = struct.unpack_from(">I", journal_data, 12)

        if page_count == 0:
            # Uncommitted transaction — discard the journal.
            logger.debug("Discarding uncommitted journal (page_count=0): %s", journal_path)
            journal_path.unlink(missing_ok=True)
            return

        # Committed journal — replay each before-image into the DB file.
        logger.debug("Recovering %d pages from journal: %s", page_count, journal_path)
        pos = _JOURNAL_HEADER_SIZE
        for _ in range(page_count):
            if pos + 4 + self._page_size > len(journal_data):
                # Truncated — stop replaying.
                break
            (page_no,) = struct.unpack_from(">I", journal_data, pos)
            pos += 4
            page_data = journal_data[pos : pos + self._page_size]
            pos += self._page_size
            file_offset = (page_no - 1) * self._page_size
            os.pwrite(self._db_fd, bytes(page_data), file_offset)

        durable_fsync(self._db_fd)

        # Delete the journal — this is the commit point of the recovery.
        journal_path.unlink(missing_ok=True)

    # ---------------------------------------------------------------------------
    # Public page I/O
    # ---------------------------------------------------------------------------

    def read_page(self, page_no: int) -> bytearray | memoryview:
        """Return the content of page_no.

        Read-precedence order (highest to lowest):
          1. Dirty heap dict — uncommitted mutations from the open transaction.
          2. WAL index — newest committed frame from the write-ahead sidecar
             (write-ahead mode only; this path is unchanged by the mapped flag).
          3. Heap cache (clean) — pages present in _page_cache but not dirty,
             e.g. before-images restored by rollback().  Always authoritative;
             takes priority over the mmap to prevent stale on-disk reads after
             a rollback in rollback-journal mode.
          4. Read-only mapped slice — a memoryview into the kernel-backed mapping
             of the main file (only when the mapped read path is ON and the page
             is absent from the heap cache).
          5. pread + heap cache — the default path when the mapped flag is OFF
             or the page lies outside the current mapping extent.

        When the mapped read path is ON, a returned memoryview slice is valid
        only until the next commit, checkpoint, or file-grow — any of those
        operations refreshes the mapping, invalidating all previously issued
        views.  Callers must not hold a view across a write transaction boundary.

        Returns a bytearray (heap cache) or memoryview (mapped slice).  In both
        cases the returned object must not be mutated — use write_page instead.
        """
        if self._closed:
            raise RuntimeError("Pager is closed")

        # Precedence 1: dirty page — return from heap cache so uncommitted
        # mutations remain visible within the current transaction.
        if page_no in self._dirty:
            if page_no in self._page_cache:
                self._touch(page_no)
                return self._page_cache[page_no]
            # Dirty but not yet cached (should not happen in normal usage).
            return self._load_page_from_disk(page_no)

        # Precedence 2: WAL index.  Always re-read from the WAL for non-dirty
        # pages in write-ahead mode — the cache may hold stale bytes after a
        # rollback() (WAL mode has no before-image restore).  The cached buffer
        # is a clean re-readable copy: it is re-materialisable from the WAL
        # sidecar (or the main file) on every miss, so the LRU may evict it.
        if self._wal_mode and self._wal is not None:
            page_bytes = self._wal.read_page(page_no, self._db_fd)
            buf = bytearray(self._page_size)
            buf[: len(page_bytes)] = page_bytes
            self._page_cache[page_no] = buf
            self._touch(page_no)
            self._evict_if_over_budget()
            return buf

        # Precedence 3: heap cache for pages restored by rollback or any other
        # pre-population path.  A page in _page_cache but not in _dirty arrived
        # here via rollback() (before-image restore) or a prior pread.  Serving
        # from cache is always correct; serving from the mmap for such pages would
        # read on-disk bytes that may diverge from the restored state.
        if page_no in self._page_cache:
            self._touch(page_no)
            return self._page_cache[page_no]

        # Precedence 4: mapped slice.  Only reached for pages absent from the
        # heap cache.  Returns a read-only memoryview without populating the
        # cache, keeping the heap dict bounded by the transaction write-set.
        if self._use_mmap and self._mmap is not None:
            start = (page_no - 1) * self._page_size
            end = start + self._page_size
            if end <= len(self._mmap):
                return memoryview(self._mmap)[start:end]

        # Precedence 5 (default): pread + heap cache.
        return self._load_page_from_disk(page_no)

    def write_page(self, page_no: int, data: bytes | bytearray) -> None:
        """Write data into the page cache at page_no and mark it dirty.

        data must be exactly page_size bytes. The write is only to the
        in-memory cache until commit() flushes it to disk.

        If a write transaction is active and page_no has not yet been journaled
        in this transaction, the current cache/disk content is automatically
        recorded as a before-image before the cache is updated. This ensures
        rollback and crash recovery can restore every modified page, not only
        those explicitly passed to record_before_image by freelist helpers.
        """
        if self._closed:
            raise RuntimeError("Pager is closed")
        if len(data) != self._page_size:
            raise ValueError(
                f"write_page: data length {len(data)} != page_size {self._page_size}"
            )
        with self._write_lock:
            if self._journal is not None and page_no not in self._journal._journaled:
                # Capture the pre-transaction content as a before-image.
                # For pages not yet dirty in this transaction, the cache holds
                # their pre-transaction state; for pages not yet cached at all,
                # read from disk.
                if page_no not in self._dirty and page_no in self._page_cache:
                    before: bytes = bytes(self._page_cache[page_no])
                elif page_no not in self._page_cache:
                    file_offset = (page_no - 1) * self._page_size
                    raw = os.pread(self._db_fd, self._page_size, file_offset)
                    before = raw.ljust(self._page_size, b"\x00")
                else:
                    # Already dirty — already in _journaled (guard above prevents
                    # reaching here, but keep for completeness).
                    before = bytes(self._page_cache[page_no])
                self._journal.record_before_image(page_no, before)
            self._page_cache[page_no] = bytearray(data)
            self._dirty.add(page_no)

    def extend_file(self) -> int:
        """Allocate a new zero-filled page at the end of the file and return its page number.

        The new page number is 1-indexed. The current db_size header field is
        incremented accordingly.  The new page exists only in the heap cache
        until commit() flushes it; because it is dirty it is served from cache
        (Precedence 1) and never reaches the mmap path before commit.  The mmap
        is refreshed inside commit() after the new bytes are durably on disk.
        """
        if self._closed:
            raise RuntimeError("Pager is closed")
        with self._write_lock:
            current_size = self.read_db_header_field("db_size")
            new_page_no = current_size + 1
            # Zero-fill the new page in cache (anti-pattern: never reuse dirty buffers)
            self._page_cache[new_page_no] = bytearray(self._page_size)
            # The freshly extended page is not yet on disk, not in the WAL, and
            # not yet written by the caller — it is only the db_size header bump.
            # Mark it dirty so the LRU never evicts it in the window before the
            # caller's write_page; evicting it would lose the zero page and a
            # subsequent read would return stale/garbage bytes.
            self._dirty.add(new_page_no)
            self.update_db_header("db_size", new_page_no)
            return new_page_no

    # ---------------------------------------------------------------------------
    # DB header field read/write (operate on the page 1 cache buffer)
    # ---------------------------------------------------------------------------

    def _writable_cache_page(self, page_no: int) -> bytearray:
        """Return the heap-cache bytearray for page_no, loading from disk if needed.

        Unlike read_page, this always returns the mutable heap-cache buffer,
        bypassing the mapped-slice path.  Used internally by header-mutation
        helpers that need to call struct.pack_into on the returned buffer.
        """
        if page_no not in self._page_cache:
            return self._load_page_from_disk(page_no)
        return self._page_cache[page_no]

    def read_db_header_field(self, name: str) -> int:
        """Read a named integer field from the database header (page 1).

        Raises KeyError if name is not a recognised header field.
        """
        if name not in _HDR_FIELDS:
            raise KeyError(f"Unknown header field: {name!r}")
        offset, _width, fmt = _HDR_FIELDS[name]
        page = self.read_page(1)
        (value,) = struct.unpack_from(fmt, page, offset)
        return value

    def update_db_header(self, name: str, value: int) -> None:
        """Write a named integer field into the database header (page 1 cache).

        The change is written into the in-memory page 1 buffer and marks page 1
        dirty. The field is committed to disk only on the next commit() call.

        If a write transaction is active and page 1 has not yet been journaled,
        the current header content is recorded as a before-image before mutation
        so that rollback and crash recovery can restore all header fields
        (db_size, freelist_count, first_freelist_trunk) to their pre-transaction
        values.

        Raises KeyError if name is not a recognised header field.
        """
        if name not in _HDR_FIELDS:
            raise KeyError(f"Unknown header field: {name!r}")
        offset, _width, fmt = _HDR_FIELDS[name]
        with self._write_lock:
            # Always use the mutable heap-cache buffer for in-place mutation;
            # read_page may return a read-only memoryview when the mapped path
            # is active, which would cause struct.pack_into to fail.
            page = self._writable_cache_page(1)
            # Journal page 1 before-image before the first mutation in this
            # transaction so recovery can restore the header correctly.
            if self._journal is not None and 1 not in self._journal._journaled:
                self._journal.record_before_image(1, bytes(page))
            struct.pack_into(fmt, page, offset, value)
            self._dirty.add(1)

    # ---------------------------------------------------------------------------
    # Journal / transaction management
    # ---------------------------------------------------------------------------

    def _journal_path(self) -> Path:
        """Return the path for the journal sidecar alongside the DB file."""
        return self._path.parent / (self._path.name + ".journal")

    def begin_write(self) -> Journal | None:
        """Begin a write transaction.

        In write-ahead mode: a no-op (the WAL writer accumulates dirty pages
        between commit() calls without a sidecar journal).  Returns None.

        In rollback-journal mode (default): opens the .lilli.journal sidecar
        and returns a Journal for recording before-images.  Raises RuntimeError
        if a transaction is already in progress.
        """
        if self._closed:
            raise RuntimeError("Pager is closed")

        # A new transaction begins: any before-image pinned by a prior rollback
        # is no longer the authoritative-only source (the on-disk file already
        # holds those pre-transaction bytes — rollback never wrote dirty pages
        # out), so its pin can be released and it becomes a normal evictable
        # clean page. Keep the header page (1) pinned permanently.
        if len(self._pinned) > 1:
            self._pinned = {1}

        # Write-ahead path: no journal sidecar needed.
        if self._wal_mode:
            return None

        if self._journal is not None:
            raise RuntimeError("A write transaction is already in progress")

        journal_path = self._journal_path()
        j_fd = os.open(str(journal_path), os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o600)
        os.chmod(str(journal_path), 0o600)
        self._journal = Journal(journal_path, j_fd)
        return self._journal

    def commit(self) -> None:
        """Commit the current write transaction.

        In write-ahead mode: collects dirty pages, delegates to the WAL writer
        (one durable sync on the sidecar), then clears the dirty set.  No
        rollback journal is written.

        In rollback-journal mode (default): runs the three-fsync barrier sequence.
        Step sequence:
        1. Journal header already written with page_count=0 by begin_write.
        2. Before-images already written by Journal.record_before_image().
        3. BARRIER 1: durable_fsync(journal) — before-images are durable.
        4. Overwrite journal header with the real page_count.
        5. BARRIER 2: durable_fsync(journal) — page_count is durable.
        6. Write all dirty pages to the DB file.
        7. BARRIER 3: durable_fsync(db) — DB pages are durable.
        8. Delete the journal — this is the commit point.
        9. fsync_directory to ensure the deletion is durable.
        """
        if self._closed:
            raise RuntimeError("Pager is closed")

        # Write-ahead path: no journal required.
        if self._wal_mode and self._wal is not None:
            with self._write_lock:
                if self._dirty:
                    dirty_pages = {
                        pn: bytes(self._page_cache[pn]) for pn in self._dirty
                    }
                    db_size = self.read_db_header_field("db_size")
                    self._wal.commit_transaction(dirty_pages, db_size_after=db_size)
                    # Merge the sidecar into the main file once enough frames have
                    # accumulated.  This keeps the sidecar bounded and read-path
                    # frame-scan cost O(1) rather than O(N-commits).
                    if self._wal._committed_frame_count >= WAL_AUTOCHECKPOINT_FRAMES:
                        self._wal.checkpoint(self._db_fd)
                        self._load_page_from_disk(1)
                        # Checkpoint merged frames into the main file — refresh
                        # the mapping so subsequent reads see the merged bytes.
                        if self._use_mmap:
                            self._refresh_mmap()
                self._dirty.clear()
            return

        if self._journal is None:
            raise RuntimeError("No write transaction in progress")

        j = self._journal
        j_fd = j.fd

        with self._write_lock:
            # Flush all buffered before-images to the journal body in one write.
            # This replaces N scattered os.pwrite calls (one per dirty page during
            # the transaction) with a single contiguous write here.
            j.flush_pending()

            # BARRIER 1: make before-images durable on the journal.
            durable_fsync(j_fd)

            # Write the real page_count into the journal header (activating recovery).
            count_data = struct.pack(">I", j.page_count)
            os.pwrite(j_fd, count_data, 12)

            # BARRIER 2: make the page_count header durable.
            durable_fsync(j_fd)

            # Write all dirty pages to the DB file (only AFTER BARRIER 2).
            for page_no in self._dirty:
                page_data = self._page_cache[page_no]
                file_offset = (page_no - 1) * self._page_size
                os.pwrite(self._db_fd, bytes(page_data), file_offset)

            # BARRIER 3: make the DB pages durable.
            durable_fsync(self._db_fd)

            # Refresh the mapping so subsequent reads see the newly written pages.
            # This also invalidates any memoryview slices issued before this commit.
            if self._use_mmap:
                self._refresh_mmap()
                # Evict committed pages from the heap cache so subsequent reads
                # are served by the fresh mmap slice (zero-copy path).  Pages NOT
                # in _dirty (e.g. rollback-restored before-images) are left in
                # place — they must remain authoritative until the next transaction.
                for pn in self._dirty:
                    self._page_cache.pop(pn, None)

            # Delete the journal — this is the commit point.
            journal_path = self._journal_path()
            os.close(j_fd)
            journal_path.unlink(missing_ok=True)

            # Fsync the directory to make the journal deletion durable.
            fsync_directory(self._path.parent)

            self._journal = None
            self._dirty.clear()

    def rollback(self) -> None:
        """Abort the current write transaction.

        In write-ahead mode: delegates to the WAL writer's rollback (which
        rewinds the running checksum pair and write-head to the last committed
        snapshot), then clears the dirty set.  No journal is touched.

        In rollback-journal mode (default): restores each journalled page into
        the in-memory cache and clears the dirty set.  The journal file is
        closed and deleted without committing.
        """
        if self._closed:
            raise RuntimeError("Pager is closed")

        # Write-ahead path: no journal required.
        if self._wal_mode and self._wal is not None:
            self._wal.rollback()
            self._dirty.clear()
            return

        if self._journal is None:
            raise RuntimeError("No write transaction in progress")

        j = self._journal
        j_fd = j.fd
        journal_path = self._journal_path()

        # Restore before-images to the in-memory page cache.
        # Two sources are possible:
        # (a) _pending: buffered in memory but not yet written to disk
        #     (typical when rollback() is called before commit() flushes).
        # (b) journal file body: written to disk when flush_pending() was
        #     already called (atypical; only happens if commit() partially ran
        #     before an exception and the caller then calls rollback()).
        # Restore from _pending first (covers the common path), then from disk.
        # A restored before-image is authoritative and must NOT be served from
        # disk/WAL afterward — pin it so the LRU never evicts it.
        for page_no, data in j._pending:
            self._page_cache[page_no] = bytearray(data)
            self._pinned.add(page_no)

        if not j._pending:
            # No pending buffer — try reading from the journal file body.
            try:
                journal_data = os.pread(
                    j_fd,
                    _JOURNAL_HEADER_SIZE + j.page_count * (4 + self._page_size),
                    0,
                )
            except OSError:
                journal_data = b""

            pos = _JOURNAL_HEADER_SIZE
            for _ in range(j.page_count):
                if pos + 4 + self._page_size > len(journal_data):
                    break
                (page_no,) = struct.unpack_from(">I", journal_data, pos)
                pos += 4
                page_bytes = journal_data[pos : pos + self._page_size]
                pos += self._page_size
                self._page_cache[page_no] = bytearray(page_bytes)
                self._pinned.add(page_no)

        os.close(j_fd)
        journal_path.unlink(missing_ok=True)
        self._journal = None
        self._dirty.clear()

    # ---------------------------------------------------------------------------
    # Checkpoint delegation
    # ---------------------------------------------------------------------------

    def checkpoint_wal(self) -> None:
        """Merge the write-ahead sidecar into the main file and reset the sidecar.

        Delegates to the WAL writer's checkpoint method, which enforces the
        strict two-sync ordering (sidecar fsync → main-file copies → main-file
        fsync → sidecar reset).  After the checkpoint, page 1 is reloaded into
        the pager cache so that subsequent header reads reflect the merged state.
        When the mapped read path is ON the mapping is refreshed so it reflects
        the newly merged bytes.

        In rollback-journal mode (write-ahead mode not active) this is a no-op.
        """
        if not self._wal_mode or self._wal is None:
            return
        self._wal.checkpoint(self._db_fd)
        # Reload page 1 so header reads see the freshly merged state.
        self._load_page_from_disk(1)
        # Refresh the mapping — checkpoint rewrote main-file pages.
        if self._use_mmap:
            self._refresh_mmap()

    # ---------------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------------

    def close(self) -> None:
        """Release file handles and clear caches. Idempotent — safe to call more than once."""
        if self._closed:
            return
        self._closed = True

        if self._journal is not None:
            # An open transaction was abandoned — close the journal fd without committing.
            try:
                os.close(self._journal.fd)
            except OSError:
                pass
            self._journal = None

        if self._wal is not None:
            # Merge committed frames into the main file so a clean close leaves a
            # compact database and an empty (header-only) sidecar rather than a
            # write-ahead log that keeps growing across the lifetime of the file.
            try:
                if self._wal._committed_frame_count > 0:
                    self._wal.checkpoint(self._db_fd)
            except OSError:
                pass
            try:
                self._wal.close()
            except OSError:
                pass
            self._wal = None

        # Release the mapping BEFORE closing the fd — the mapping holds a
        # reference to the fd; closing the fd while the mapping is alive is
        # undefined behaviour on some platforms.
        self._close_mmap()

        try:
            os.close(self._db_fd)
        except OSError:
            pass

        self._page_cache.clear()
        self._dirty.clear()
        self._pinned.clear()
        logger.debug("Pager closed: %s", self._path)
