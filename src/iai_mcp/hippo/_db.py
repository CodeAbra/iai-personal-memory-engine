"""Database lifecycle class and transaction helpers for the Hippo storage backend."""

from __future__ import annotations

import contextlib
import errno
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import hnswlib
import numpy as np
import pyarrow as pa

from iai_mcp._filelock import (
    LOCK_EX,
    LOCK_NB,
    LOCK_SH,
    LOCK_UN,
    SHARED_IS_EXCLUSIVE,
    flock,
)
from iai_mcp.crypto import (
    decrypt_field,
    encrypt_field,
    is_encrypted,
)

# AccessMode is defined in __init__.py (DEFINE-IN-INIT).  It must be available
# at class-body execution time because HippoDB.__init__ uses it as a default
# argument value.  Since __init__.py defines AccessMode before triggering this
# sub-module import, the partially-initialised package module already carries it.
from iai_mcp.hippo import AccessMode  # noqa: E402

_log = logging.getLogger(__name__)


# Bounds for the deferred-embed pass driven by the wake sequence. The pass
# embeds pending rows (written cheaply by the in-daemon drain) in windows of
# REEMBED_BATCH_SIZE, releasing idle allocator pages between windows, and yields
# once the resident set exceeds REEMBED_RSS_SOFT_CAP — leaving the remaining
# rows pending for the next cycle. This keeps a large deferred backlog from
# climbing the resident set through one long synchronous embed run. Both are
# operator-overridable; a non-positive / malformed value disables that bound.
REEMBED_BATCH_SIZE_DEFAULT = 128
REEMBED_RSS_SOFT_CAP_DEFAULT_BYTES = 2_684_354_560


def _reembed_batch_size() -> int:
    raw = os.environ.get("IAI_MCP_REEMBED_BATCH_SIZE")
    if raw is None:
        return REEMBED_BATCH_SIZE_DEFAULT
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return REEMBED_BATCH_SIZE_DEFAULT
    return val if val > 0 else 0


def _reembed_rss_soft_cap() -> int:
    raw = os.environ.get("IAI_MCP_REEMBED_RSS_SOFT_CAP_BYTES")
    if raw is None:
        return REEMBED_RSS_SOFT_CAP_DEFAULT_BYTES
    try:
        val = int(raw)
    except (TypeError, ValueError):
        return REEMBED_RSS_SOFT_CAP_DEFAULT_BYTES
    return val if val > 0 else 0


_txn_owners: dict[int, int] = {}
_txn_owners_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager  # type: ignore[misc]
def _txn(conn: "sqlite3.Connection"):
    from iai_mcp.hippo import HippoIntegrityError
    if conn.in_transaction:
        with _txn_owners_lock:
            owner = _txn_owners.get(id(conn))
        if owner is None:
            yield
            return
        if owner == threading.get_ident():
            yield
            return
        raise HippoIntegrityError(
            f"Shared connection transaction owned by thread {owner} "
            f"observed by thread {threading.get_ident()} — a transactional "
            f"mutator site is missing _conn_lock serialization."
        )
    conn_id = id(conn)
    with _txn_owners_lock:
        _txn_owners[conn_id] = threading.get_ident()
    try:
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        conn.execute("COMMIT")
    finally:
        with _txn_owners_lock:
            _txn_owners.pop(conn_id, None)


_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(name: str) -> str:
    if not _TABLE_NAME_RE.match(name):
        raise ValueError(
            f"Invalid table name {name!r}: must match [A-Za-z_][A-Za-z0-9_]*"
        )
    return name


class HippoDB:

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        crypto_key_provider: Callable[[], bytes] | None = None,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
        _lock_timeout_override: float | None = None,
    ) -> None:
        from iai_mcp.hippo import _resolve_root, EMBED_DIM as _EMBED_DIM
        self._crypto_key_provider: Callable[[], bytes] | None = crypto_key_provider
        self._access_mode: AccessMode = access_mode
        self._read_only: bool = read_only

        root = _resolve_root(path)
        self._store_root: Path = root
        self._hippo_dir: Path = root / "hippo"
        self._hippo_dir.mkdir(parents=True, exist_ok=True)

        self._lock_path: Path = self._hippo_dir / ".lock"
        self._lock_key: str = str(self._lock_path.resolve())

        if access_mode is AccessMode.EXCLUSIVE:
            self._acquire_exclusive_lock()
        else:
            self._acquire_shared_lock(
                lock_timeout_override=_lock_timeout_override,
            )

        db_path = self._hippo_dir / "brain.sqlite3"
        _cached_stmts_raw = os.environ.get("IAI_MCP_SQLITE_CACHED_STATEMENTS", "128")
        try:
            _cached_stmts = max(0, int(_cached_stmts_raw))
        except ValueError:
            _cached_stmts = 128
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(db_path),
            check_same_thread=False,
            isolation_level=None,
            cached_statements=_cached_stmts,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=2000")
        # Cap the SQLite page cache so a long-running daemon process cannot
        # let it drift unbounded across cycles. Negative values are KiB.
        # tracemalloc cannot see C-level allocations, so this cap is the
        # only explicit bound on the page cache footprint in process RSS.
        _cache_kib = os.environ.get("IAI_MCP_SQLITE_CACHE_SIZE_KIB", "65536")
        try:
            _cache_kib_int = int(_cache_kib)
        except ValueError:
            _cache_kib_int = 65536
        self._conn.execute(f"PRAGMA cache_size=-{abs(_cache_kib_int)}")
        if read_only:
            self._conn.execute("PRAGMA query_only=ON")

        _env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
        self._embed_dim: int = (
            int(_env_dim) if _env_dim and _env_dim.isdigit() else _EMBED_DIM
        )
        self._closed: bool = False
        self._hnsw_path: Path = self._hippo_dir / "records.hnsw"
        self._hnsw_tmp_path: Path = self._hippo_dir / "records.hnsw.tmp"
        self._hnsw_lock: threading.RLock = threading.RLock()
        self._conn_lock: threading.RLock = threading.RLock()
        if not read_only:
            self._ensure_tables()

        if not read_only:
            meta_dim = self._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = 'embed_dim'"
            ).fetchone()
            if meta_dim is not None:
                self._embed_dim = int(meta_dim[0])
        self._label_map: dict[str, int] = {}
        self._write_counter: int = 0

        if read_only:
            self._hnsw: hnswlib.Index | None = None  # type: ignore[assignment]
            self._hnsw_standby: hnswlib.Index | None = None
            try:
                self._repopulate_label_map_from_sqlite()
            except Exception:  # noqa: BLE001
                pass
        else:
            self._repopulate_label_map_from_sqlite()
            self._initialize_hnsw_index()


    def _acquire_exclusive_lock(self) -> None:
        from iai_mcp.hippo import (
            HippoLockHeldError,
            _PROCESS_LOCKS,
            _PROCESS_LOCKS_SHARED,
            _PROCESS_LOCKS_GUARD,
        )
        with _PROCESS_LOCKS_GUARD:
            if self._lock_key in _PROCESS_LOCKS_SHARED:
                raise HippoLockHeldError(
                    self._lock_path,
                    "same-process-holds-SHARED",
                )
            held = _PROCESS_LOCKS.get(self._lock_key)
            if held is not None:
                base_fd, refcount = held
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS[self._lock_key] = (base_fd, refcount + 1)
            else:
                base_fd = os.open(
                    str(self._lock_path),
                    os.O_CREAT | os.O_RDWR,
                    0o600,
                )
                os.chmod(str(self._lock_path), 0o600)
                try:
                    flock(base_fd, LOCK_EX | LOCK_NB)
                except OSError as exc:
                    os.close(base_fd)
                    if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                        raise HippoLockHeldError(self._lock_path, "unknown") from exc
                    raise
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS[self._lock_key] = (base_fd, 1)

    def _acquire_shared_lock(
        self,
        lock_timeout_override: float | None = None,
    ) -> None:
        from iai_mcp.hippo import (
            HippoLockHeldError,
            ConsolidationPendingError,
            _PROCESS_LOCKS,
            _PROCESS_LOCKS_SHARED,
            _PROCESS_LOCKS_GUARD,
            _SHARED_LOCK_TIMEOUT_S,
            _SHARED_RETRY_SLEEP_S,
            _SHARED_MAX_RETRIES,
        )
        _intent_path = self._hippo_dir / ".consolidation-pending"

        with _PROCESS_LOCKS_GUARD:
            if self._lock_key in _PROCESS_LOCKS:
                raise HippoLockHeldError(
                    self._lock_path,
                    "same-process-holds-EXCLUSIVE",
                )
            held_sh = _PROCESS_LOCKS_SHARED.get(self._lock_key)
            if held_sh is not None:
                base_fd, refcount = held_sh
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, refcount + 1)
                return

            base_fd = os.open(
                str(self._lock_path),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            os.chmod(str(self._lock_path), 0o600)

        _timeout = (
            lock_timeout_override
            if lock_timeout_override is not None
            else _SHARED_LOCK_TIMEOUT_S
        )
        deadline = time.monotonic() + _timeout
        acquired = False
        for _ in range(_SHARED_MAX_RETRIES + 1):
            if _intent_path.exists():
                if time.monotonic() >= deadline:
                    break
                time.sleep(_SHARED_RETRY_SLEEP_S)
                continue

            try:
                flock(base_fd, LOCK_SH | LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_SHARED_RETRY_SLEEP_S)
                    continue
                os.close(base_fd)
                raise

            if _intent_path.exists():
                flock(base_fd, LOCK_UN)
                if time.monotonic() >= deadline:
                    break
                time.sleep(_SHARED_RETRY_SLEEP_S)
                continue

            acquired = True
            break

        if not acquired:
            os.close(base_fd)
            raise ConsolidationPendingError(self._lock_path)

        with _PROCESS_LOCKS_GUARD:
            held_sh = _PROCESS_LOCKS_SHARED.get(self._lock_key)
            if held_sh is not None:
                flock(base_fd, LOCK_UN)
                os.close(base_fd)
                base_fd2, refcount2 = held_sh
                self._lock_fd = os.dup(base_fd2)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd2, refcount2 + 1)
            else:
                self._lock_fd = os.dup(base_fd)
                _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, 1)


    def downgrade_to_shared(self) -> None:
        from iai_mcp.hippo import (
            _PROCESS_LOCKS,
            _PROCESS_LOCKS_SHARED,
            _PROCESS_LOCKS_GUARD,
        )
        _intent_path = self._hippo_dir / ".consolidation-pending"

        with _PROCESS_LOCKS_GUARD:
            if self._access_mode is not AccessMode.EXCLUSIVE:
                return
            held = _PROCESS_LOCKS.get(self._lock_key)
            if held is None:
                return
            base_fd, refcount = held
            # Downgrade EXCLUSIVE -> SHARED in place. On POSIX fcntl.flock
            # converts the held lock atomically and never blocks (SHARED is
            # weaker than the EXCLUSIVE this fd already owns). On Windows msvcrt
            # has no shared locks and no atomic conversion: LOCK_SH is serviced
            # as the *same* exclusive byte-range lock this fd already holds, so a
            # *blocking* flock(LOCK_SH) polls LK_NBLCK forever against our own
            # range -- wedging the daemon while _PROCESS_LOCKS_GUARD is held,
            # which the liveness watchdog then force-kills. Use the non-blocking
            # form and treat "already held" (EWOULDBLOCK) as success: we are
            # relabelling a lock this fd already owns, so no OS re-acquisition is
            # needed on either platform.
            try:
                flock(base_fd, LOCK_SH | LOCK_NB)
            except OSError as exc:
                if exc.errno not in (errno.EAGAIN, errno.EWOULDBLOCK):
                    return
            del _PROCESS_LOCKS[self._lock_key]
            _PROCESS_LOCKS_SHARED[self._lock_key] = (base_fd, refcount)
        self._access_mode = AccessMode.SHARED

        try:
            _intent_path.unlink()
        except FileNotFoundError:
            pass

    def escalate_to_exclusive(self, intent_budget_ms: int = 4000) -> None:
        from iai_mcp.hippo import (
            HippoLockHeldError,
            _PROCESS_LOCKS,
            _PROCESS_LOCKS_SHARED,
            _PROCESS_LOCKS_GUARD,
        )
        _intent_path = self._hippo_dir / ".consolidation-pending"

        try:
            fd = os.open(str(_intent_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
        except FileExistsError:
            pass

        if self._access_mode is AccessMode.EXCLUSIVE:
            return

        with _PROCESS_LOCKS_GUARD:
            held = _PROCESS_LOCKS_SHARED.get(self._lock_key)
        if held is None:
            base_fd = os.open(str(self._lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        else:
            base_fd, _ = held

        deadline = time.monotonic() + intent_budget_ms / 1000.0
        acquired = False
        # On Windows the shim has no shared locks: a "shared" hold is already an
        # exclusive msvcrt byte-range lock that no other process can co-hold, so
        # if this fd already owns it (held is not None) we effectively already
        # have exclusive access. A real flock(LOCK_EX) here would self-conflict
        # against our own range and spuriously time out into HippoLockHeldError,
        # so relabel without touching the OS lock. On POSIX shared locks are
        # genuinely shared (another reader may hold one), so escalation must
        # still contend for LOCK_EX below.
        if SHARED_IS_EXCLUSIVE and held is not None:
            acquired = True
        while not acquired and time.monotonic() < deadline:
            try:
                flock(base_fd, LOCK_EX | LOCK_NB)
                acquired = True
                break
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    time.sleep(0.040)
                    continue
                raise

        if not acquired:
            if held is None:
                os.close(base_fd)
            raise HippoLockHeldError(self._lock_path, "escalate_timeout")

        with _PROCESS_LOCKS_GUARD:
            if held is not None:
                _, refcount = held
                del _PROCESS_LOCKS_SHARED[self._lock_key]
            else:
                refcount = 1
                self._lock_fd = os.dup(base_fd)
            _PROCESS_LOCKS[self._lock_key] = (base_fd, refcount)
        self._access_mode = AccessMode.EXCLUSIVE


    def _initialize_hnsw_index(self) -> None:
        from iai_mcp.hippo import (
            HNSW_INITIAL_CAPACITY,
            HNSW_EF_CONSTRUCTION,
            HNSW_M,
            HNSW_EF,
            RECALL_INDEX_EF,
            HippoIntegrityError,
        )
        _sqlite_count_row = self._conn.execute(
            "SELECT COUNT(*) FROM records"
            " WHERE tombstoned_at IS NULL"
            " AND COALESCE(embedding_pending, 0) = 0"
        ).fetchone()
        if _sqlite_count_row is None:
            raise HippoIntegrityError(
                "_initialize_hnsw_index: SELECT COUNT(*) returned no row — "
                "connection may be in an error state"
            )
        sqlite_count = _sqlite_count_row[0]
        cap = max(HNSW_INITIAL_CAPACITY, sqlite_count * 2)

        loaded = False

        for candidate in (self._hnsw_tmp_path, self._hnsw_path):
            if candidate.exists():
                try:
                    idx = hnswlib.Index(space="cosine", dim=self._embed_dim)
                    idx.load_index(str(candidate), max_elements=cap, allow_replace_deleted=True)
                    idx.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
                    idx.set_num_threads(1)
                    self._hnsw: hnswlib.Index = idx
                    loaded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    _log.warning("Failed to load hnswlib index from %s: %s", candidate, exc)

        if not loaded:
            self._hnsw = hnswlib.Index(space="cosine", dim=self._embed_dim)
            self._hnsw.init_index(
                max_elements=cap,
                ef_construction=HNSW_EF_CONSTRUCTION,
                M=HNSW_M,
                allow_replace_deleted=True,
            )
            self._hnsw.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
            self._hnsw.set_num_threads(1)
            if sqlite_count > 0:
                _log.info(
                    "No valid hnswlib file found; rebuilding from %d SQLite records",
                    sqlite_count,
                )
                self._rebuild_index_from_sqlite()
                self._allocate_standby_index(cap)
                return

        hnsw_count = self._hnsw.get_current_count()
        if hnsw_count != sqlite_count:
            _log.info(
                "Boot integrity check: hnsw count=%d != sqlite count=%d — rebuilding",
                hnsw_count,
                sqlite_count,
            )
            self._rebuild_index_from_sqlite()

        # Second standby buffer, reused across rebuilds so steady-state
        # consolidation cycles never allocate a fresh C++ index.  Allocated
        # after any boot-integrity rebuild so the rebuild above takes the
        # fresh-alloc fallback path (the standby is intentionally absent then).
        self._allocate_standby_index(cap)

    def _allocate_standby_index(self, cap: int) -> None:
        from iai_mcp.hippo import (
            HNSW_EF_CONSTRUCTION,
            HNSW_M,
            HNSW_EF,
            RECALL_INDEX_EF,
        )
        standby = hnswlib.Index(space="cosine", dim=self._embed_dim)
        standby.init_index(
            max_elements=cap,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
            allow_replace_deleted=True,
        )
        standby.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
        standby.set_num_threads(1)
        self._hnsw_standby: hnswlib.Index | None = standby

    def _repopulate_label_map_from_sqlite(self) -> None:
        _lock = getattr(self, "_conn_lock", None)
        if _lock is not None:
            with _lock:
                rows = self._conn.execute(
                    "SELECT id, vec_label FROM records"
                    " WHERE tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0"
                ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT id, vec_label FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchall()
        self._label_map.clear()
        for row in rows:
            self._label_map[row["id"]] = int(row["vec_label"])

    def _rebuild_index_from_sqlite(self) -> dict:
        from iai_mcp.hippo import (
            HNSW_INITIAL_CAPACITY,
            HNSW_EF_CONSTRUCTION,
            HNSW_M,
            HNSW_EF,
            RECALL_INDEX_EF,
        )
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT vec_label, embedding FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
                " ORDER BY vec_label"
            ).fetchall()

        n = len(rows)
        cap = max(HNSW_INITIAL_CAPACITY, n * 2)

        vecs = None
        labels = None
        if n > 0:
            vecs = np.stack([
                np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
            ])
            labels = np.array([int(row["vec_label"]) for row in rows], dtype=np.int64)

        if getattr(self, "_hnsw_standby", None) is None:
            # Boot path: the standby is not allocated yet (it is created at the
            # end of _initialize_hnsw_index, after any boot-integrity rebuild).
            # Build a fresh active index directly.
            self._hnsw = hnswlib.Index(space="cosine", dim=self._embed_dim)
            self._hnsw.init_index(
                max_elements=cap,
                ef_construction=HNSW_EF_CONSTRUCTION,
                M=HNSW_M,
                allow_replace_deleted=True,
            )
            self._hnsw.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
            self._hnsw.set_num_threads(1)
            if n > 0:
                self._hnsw.add_items(vecs, labels)
            self._save_index_atomic()
            self._repopulate_label_map_from_sqlite()
            return {"action": "rebuild", "rebuilt_count": n}

        # Steady-state reuse path: build into the standby buffer lock-free so
        # readers never observe a torn index, then commit with a single atomic
        # buffer swap under the recall lock.  Reusing the standby avoids
        # allocating a fresh C++ index every consolidation cycle.
        standby = self._hnsw_standby
        if standby.get_current_count() > 0:
            for label in list(standby.get_ids_list()):
                standby.mark_deleted(label)
        if n > 0:
            if n > standby.get_max_elements():
                standby.resize_index(n * 2)
            standby.add_items(vecs, labels, replace_deleted=True)

        # Publish the freshly-built buffer together with its label map under the
        # recall lock so a concurrent knn_query reader always sees a consistent
        # (buffer, label_map) pair — never a half-deleted or half-refilled index.
        with self._hnsw_lock:
            self._hnsw, self._hnsw_standby = self._hnsw_standby, self._hnsw
            self._repopulate_label_map_from_sqlite()
            self._save_index_atomic()

        return {"action": "rebuild", "rebuilt_count": n}

    def _save_index_atomic(self) -> None:
        try:
            self._hnsw.save_index(str(self._hnsw_tmp_path))
            os.replace(self._hnsw_tmp_path, self._hnsw_path)
        except OSError as exc:
            _log.warning("hnswlib index save failed: %s", exc)

    def _maybe_resize(self) -> None:
        from iai_mcp.hippo import HNSW_RESIZE_HEADROOM
        current = self._hnsw.get_current_count()
        max_el = self._hnsw.get_max_elements()
        if max_el > 0 and current > HNSW_RESIZE_HEADROOM * max_el:
            self._hnsw.resize_index(max_el * 2)


    def insert_pending_row(
        self,
        *,
        record_id: str,
        tier: str,
        literal_surface: str,
        tags_json: str,
        provenance_json: str,
        created_at: str,
        updated_at: str,
    ) -> None:
        import struct as _struct
        zero_blob = _struct.pack(f"<{self._embed_dim}f", *([0.0] * self._embed_dim))
        # Encrypt the confidential columns at rest with the same key provider and
        # uuid-derived associated data the normal insert path uses, so a row in
        # the deferred-embed window is indistinguishable from a fully-embedded
        # row on disk. When no key provider is configured these are no-ops and
        # the values are stored as plaintext, unchanged.
        literal_surface = self._encrypt_for_uuid(record_id, literal_surface)
        provenance_json = self._encrypt_for_uuid(record_id, provenance_json)
        with self._conn_lock:
            self._conn.execute(
                "INSERT INTO records"
                " (id, tier, literal_surface, aaak_index, embedding, embedding_pending,"
                "  provenance_json, created_at, updated_at, tags_json,"
                "  community_id, detail_level, centrality, stability, difficulty,"
                "  pinned, never_decay, never_merge, s5_trust_score,"
                "  schema_version, language,"
                "  hv_tier, structure_hv_payload)"
                " VALUES (?, ?, ?, '', ?, 1, ?, ?, ?, ?, '', 1, 0.0, 0.0, 0.0,"
                "  0, 0, 0, 0.5, 1, 'en', 'bsc', x'')",
                (
                    record_id,
                    tier,
                    literal_surface,
                    zero_blob,
                    provenance_json,
                    created_at,
                    updated_at,
                    tags_json,
                ),
            )
            self._conn.commit()

    def has_pending_rows(self) -> bool:
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1 LIMIT 1"
            ).fetchone()
        return row is not None

    def reembed_pending_rows(
        self,
        embedder: Any,
        *,
        batch_size: int = 0,
        rss_soft_cap_bytes: int = 0,
    ) -> int:
        """Embed pending rows, optionally bounded by batch size and resident set.

        The deferred-embed pass of the two-phase capture drain. With the defaults
        (both bounds 0) it embeds every pending row in one pass — the historical
        behaviour direct-write recovery and the wake sequence rely on.

        When ``batch_size`` is positive the rows are embedded in windows of that
        size, each window committed before the next is read, with an allocator
        relief between windows so the per-window embedder/columnar transient is
        handed back to the OS instead of accumulating. When ``rss_soft_cap_bytes``
        is positive the pass stops cleanly before reading the next window once the
        resident set exceeds the cap — the remaining rows stay pending for the
        next cycle, exactly like the drain's own soft cap. This keeps a large
        deferred backlog from climbing the resident set through one long run.
        """
        import struct as _struct

        with self._conn_lock:
            ids = [
                r["id"]
                for r in self._conn.execute(
                    "SELECT id FROM records"
                    " WHERE COALESCE(embedding_pending, 0) = 1"
                    " AND tombstoned_at IS NULL"
                    " ORDER BY rowid"
                ).fetchall()
            ]
        if not ids:
            return 0

        window = batch_size if batch_size and batch_size > 0 else len(ids)
        soft_cap = rss_soft_cap_bytes if rss_soft_cap_bytes and rss_soft_cap_bytes > 0 else 0
        bounded = bool(batch_size and batch_size > 0)

        count = 0
        for start in range(0, len(ids), window):
            # Soft resident-memory ceiling — checked before reading the next
            # window so the pass yields with the remaining rows still pending
            # (deferred, not lost). A 0 reading (psutil unavailable) is treated as
            # unknown and never trips the cap.
            if soft_cap > 0 and start > 0:
                rss_now = self._reembed_rss_bytes()
                if rss_now > soft_cap:
                    _log.info(
                        "reembed_pending_rows: rss soft cap hit (rss=%d > cap=%d),"
                        " %d rows left pending",
                        rss_now,
                        soft_cap,
                        len(ids) - start,
                    )
                    break

            window_ids = ids[start:start + window]
            window_count = 0
            for rid in window_ids:
                with self._conn_lock:
                    row = self._conn.execute(
                        "SELECT literal_surface FROM records WHERE id = ?",
                        (rid,),
                    ).fetchone()
                if row is None:
                    continue
                surface_raw = row["literal_surface"] or ""
                # Decrypt the at-rest text before embedding, mirroring the
                # record-read decrypt path. On an unencrypted store (no key
                # provider) this returns the value unchanged; on an encrypted
                # store it recovers the plaintext so the vector is of the message,
                # never of the ciphertext. A row whose text is empty or
                # undecryptable is skipped, never embedded — no vector is
                # fabricated.
                try:
                    surface = self._decrypt_record_field(
                        rid, "literal_surface", surface_raw
                    )
                except Exception as exc:  # noqa: BLE001 -- decrypt fail-safe, skip
                    _log.warning(
                        "reembed_pending_rows: decrypt failed for id=%s: %s", rid, exc
                    )
                    continue
                if not (surface or "").strip():
                    continue
                try:
                    vec = list(embedder.embed(surface))
                except Exception as exc:  # noqa: BLE001
                    _log.warning(
                        "reembed_pending_rows: embed failed for id=%s: %s", rid, exc
                    )
                    continue
                blob = _struct.pack(f"<{len(vec)}f", *vec)
                with self._conn_lock:
                    self._conn.execute(
                        "UPDATE records SET embedding = ?, embedding_pending = 0"
                        " WHERE id = ?",
                        (blob, rid),
                    )
                count += 1
                window_count += 1

            if window_count > 0:
                with self._conn_lock:
                    self._conn.commit()

            # Hand the per-window embedder/columnar transient back to the OS
            # between windows so the bounded pass does not build a warm plateau.
            if bounded and window_count > 0:
                self._reembed_window_relief()

        return count

    @staticmethod
    def _reembed_rss_bytes() -> int:
        try:
            import psutil

            return int(psutil.Process().memory_info().rss)
        except Exception:  # noqa: BLE001 -- psutil flakiness must not stop the pass
            return 0

    @staticmethod
    def _reembed_window_relief() -> None:
        try:
            from iai_mcp.lilli.cycle.sleep_pipeline._memory_relief import (
                _step_memory_relief,
            )

            _step_memory_relief(label="reembed_pending")
        except Exception as exc:  # noqa: BLE001 -- relief is advisory, never fatal
            _log.debug("reembed_pending_rows: window relief failed: %s", exc)

    def ingest_pending_embeddings(self) -> int:
        import json as _json
        import struct as _struct

        sidecar_dir = self._store_root / ".pending-embeddings"
        if not sidecar_dir.exists():
            return 0

        ingested = 0
        for npy_path in sorted(sidecar_dir.glob("*.npy")):
            uuid_str = npy_path.stem
            json_path = sidecar_dir / f"{uuid_str}.json"
            if not json_path.exists():
                _log.debug("ingest_pending_embeddings: skipping partial sidecar %s (no .json)", npy_path)
                continue
            try:
                vec_bytes = npy_path.read_bytes()
                n_floats = len(vec_bytes) // 4
                if n_floats == 0 or len(vec_bytes) % 4 != 0:
                    _log.warning("ingest_pending_embeddings: malformed .npy %s, skipping", npy_path)
                    continue
                vec = list(_struct.unpack(f"<{n_floats}f", vec_bytes))
                meta = _json.loads(json_path.read_text(encoding="utf-8"))
                vec_label = int(meta["vec_label"])
            except Exception as exc:  # noqa: BLE001
                _log.warning("ingest_pending_embeddings: failed to load %s: %s", npy_path, exc)
                continue

            import numpy as _np
            with self._hnsw_lock:
                self._maybe_resize()
                self._hnsw.add_items(
                    _np.array([vec], dtype=_np.float32),
                    _np.array([vec_label], dtype=_np.int64),
                )
                self._label_map[uuid_str] = vec_label
                self._save_index_atomic()

            try:
                npy_path.unlink()
                json_path.unlink()
            except OSError as exc:
                _log.warning("ingest_pending_embeddings: cleanup failed for %s: %s", npy_path, exc)

            ingested += 1
        return ingested

    def pending_embeddings_wake_sequence(self, embedder: Any | None = None) -> dict:
        has_pending = self.has_pending_rows()
        sidecar_dir = self._store_root / ".pending-embeddings"
        has_sidecars = sidecar_dir.exists() and any(sidecar_dir.glob("*.npy"))
        with self._conn_lock:
            non_pending_row = self._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        non_pending_count = non_pending_row[0] if non_pending_row else 0
        index_count = len(self._label_map)
        has_mismatch = (index_count != non_pending_count)

        if not has_pending and not has_sidecars and not has_mismatch:
            return {"action": "skip", "reason": "clean"}

        reembed_count = 0
        if has_pending and embedder is not None:
            # Bounded deferred-embed pass: windowed embed + per-window allocator
            # relief + resident-set soft cap, so a large pending backlog (the
            # two-phase drain writes all genuinely-new turns as pending rows) is
            # embedded in resident-memory-bounded steps. Rows beyond the cap stay
            # pending for the next wake.
            reembed_count = self.reembed_pending_rows(
                embedder,
                batch_size=_reembed_batch_size(),
                rss_soft_cap_bytes=_reembed_rss_soft_cap(),
            )

        ingest_count = 0
        if has_sidecars:
            ingest_count = self.ingest_pending_embeddings()

        rebuild_result = self._rebuild_index_from_sqlite()

        # A pending->ready transition (a row flipping embedding_pending 1->0, or a
        # sidecar embedding landing) adds an active, index-findable row whose +1
        # corpus delta can stay within the warm graph's drift tolerance and not
        # flip its staleness-window cache key. Without invalidation the next warm
        # build reuses the stale node set and the re-embedded row is absent from
        # community gating and centrality until an unrelated over-tolerance
        # rebuild. Invalidate at the data-operation boundary so every caller of
        # the wake sequence — not just the daemon — forces the row into the next
        # warm graph. The unlink is key-free; the AES fence is untouched.
        if reembed_count > 0 or ingest_count > 0:
            try:
                from iai_mcp import runtime_graph_cache as _rgc
                _rgc.invalidate_at_root(self._store_root)
            except Exception:  # noqa: BLE001 -- invalidation must never break wake
                pass

        return {
            "action": "wake_sequence",
            "reembed_count": reembed_count,
            "ingest_count": ingest_count,
            "rebuild": rebuild_result,
        }


    def _encrypt_for_uuid(self, uuid_str: str, value: str) -> str:
        if self._crypto_key_provider is None:
            return value
        if value is None:
            return value
        if is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        return encrypt_field(value, key, associated_data=ad)

    def _decrypt_record_field(self, uuid_str: str, column: str, value: str) -> str:
        from iai_mcp.hippo import HippoDecryptError
        if self._crypto_key_provider is None:
            return value
        if value is None or not is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        try:
            return decrypt_field(value, key, associated_data=ad)
        except Exception as exc:
            try:
                self._emit_record_decrypt_failed(
                    uuid_str=uuid_str,
                    column=column,
                    error=f"{type(exc).__name__}: {exc}"[:200],
                )
            except Exception:
                pass
            raise HippoDecryptError(
                f"records.{column} decrypt failed for id={uuid_str}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    def _decrypt_event_field(self, uuid_str: str, column: str, value: str) -> str:
        if self._crypto_key_provider is None:
            return value
        if value is None or not is_encrypted(value):
            return value
        key = self._crypto_key_provider()
        ad = uuid_str.lower().encode("ascii")
        try:
            return decrypt_field(value, key, associated_data=ad)
        except Exception:
            return "{}" if column.endswith("_json") else ""

    def _emit_record_decrypt_failed(
        self,
        *,
        uuid_str: str,
        column: str,
        error: str,
    ) -> None:
        import json as _json
        from uuid import uuid4

        event_id = str(uuid4())
        ts = datetime.now(timezone.utc).isoformat()
        payload = _json.dumps({
            "record_id": uuid_str,
            "column": column,
            "error": error,
        })
        try:
            self._conn.execute(
                "INSERT INTO events (id, kind, severity, domain, ts, "
                "data_json, session_id, source_ids_json) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    "record_decrypt_failed",
                    "error",
                    "storage",
                    ts,
                    payload,
                    None,
                    None,
                ),
            )
        except Exception:
            pass


    def _ensure_tables(self) -> None:
        from iai_mcp.hippo import (
            _DDL_RECORDS,
            _DDL_RECORDS_INDEXES,
            _DDL_EDGES,
            _DDL_EDGES_INDEXES,
            _DDL_EVENTS,
            _DDL_EVENTS_INDEXES,
            _DDL_BUDGET_LEDGER,
            _DDL_BUDGET_LEDGER_INDEXES,
            _DDL_RATELIMIT_LEDGER,
            _DDL_HIPPO_META,
        )
        conn = self._conn
        conn.execute("BEGIN")
        try:
            conn.execute(_DDL_RECORDS)
            self._reconcile_columns(
                "records",
                [
                    ("wing", "TEXT"),
                    ("room", "TEXT"),
                    ("drawer", "TEXT"),
                    ("valence", "REAL DEFAULT 0.0"),
                    ("hv_tier", "TEXT NOT NULL DEFAULT 'bsc'"),
                    ("structure_hv_payload", "BLOB NOT NULL DEFAULT x''"),
                    ("embedding_pending", "INTEGER NOT NULL DEFAULT 0"),
                ],
            )
            for idx in _DDL_RECORDS_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_EDGES)
            for idx in _DDL_EDGES_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_EVENTS)
            for idx in _DDL_EVENTS_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_BUDGET_LEDGER)
            for idx in _DDL_BUDGET_LEDGER_INDEXES:
                conn.execute(idx)

            conn.execute(_DDL_RATELIMIT_LEDGER)

            conn.execute(_DDL_HIPPO_META)
            conn.execute(
                "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
                ("schema_version", "1"),
            )
            conn.execute(
                "INSERT OR IGNORE INTO _hippo_meta (key, value) VALUES (?, ?)",
                ("embed_dim", str(self._embed_dim)),
            )
        except Exception:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def _reconcile_columns(
        self, table_name: str, expected: list[tuple[str, str]]
    ) -> None:
        plain_to_pa = {
            "TEXT": pa.string(),
            "INTEGER": pa.int64(),
            "REAL": pa.float64(),
            "BLOB": pa.binary(),
        }
        allowed_with_default = {
            "REAL DEFAULT 0.0",
            "INTEGER DEFAULT 0",
            "INTEGER DEFAULT 1",
            "TEXT NOT NULL DEFAULT 'bsc'",
            "BLOB NOT NULL DEFAULT x''",
            "INTEGER NOT NULL DEFAULT 0",
        }
        safe_table = _validate_table_name(table_name)
        pragma_stmt = "PRAGMA table_info(" + safe_table + ")"
        _lock = getattr(self, "_conn_lock", None)
        if _lock is not None:
            with _lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        existing = {row["name"] for row in _pragma_rows}

        tbl = self.open_table(safe_table)
        missing_plain: list[pa.Field] = []
        missing_with_default: list[tuple[str, str]] = []
        for col_name, sqlite_type in expected:
            if col_name in existing:
                continue
            if sqlite_type in plain_to_pa:
                missing_plain.append(pa.field(col_name, plain_to_pa[sqlite_type]))
            elif sqlite_type in allowed_with_default:
                missing_with_default.append((col_name, sqlite_type))
            else:
                raise RuntimeError(
                    f"_reconcile_columns rejected non-canonical type "
                    f"declaration {sqlite_type!r} for column {col_name!r}"
                )

        failing: list[str] = []
        if missing_plain:
            try:
                tbl.add_columns(missing_plain)
            except Exception:  # noqa: BLE001 -- aggregate names, raise once below
                failing.extend(f.name for f in missing_plain)

        for col_name, sqlite_type in missing_with_default:
            if col_name in failing:
                continue
            safe_col = _validate_table_name(col_name)
            alter_stmt = (
                "ALTER TABLE " + safe_table + " ADD COLUMN "
                + safe_col + " " + sqlite_type
            )
            try:
                self._conn.execute(alter_stmt)
            except Exception:  # noqa: BLE001 -- aggregate names, raise once below
                failing.append(col_name)

        if failing:
            raise RuntimeError(
                f"schema reconciliation failed for table {safe_table!r}: "
                f"could not add columns {failing!r}"
            )


    def table_names(self) -> list[str]:
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' "
                "ORDER BY name"
            ).fetchall()
        return [row["name"] for row in rows]

    def list_tables(self) -> "HippoTableList":
        from iai_mcp.hippo import HippoTableList
        return HippoTableList(self.table_names())


    def open_table(self, name: str) -> "HippoTable":
        from iai_mcp.hippo import HippoTable
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def create_table(
        self,
        name: str,
        schema: pa.Schema | None = None,
        data: Any = None,
    ) -> "HippoTable":
        from iai_mcp.hippo import HippoTable, _pa_type_to_sqlite
        _validate_table_name(name)
        if name not in self.table_names():
            if schema is not None:
                cols = []
                for f in schema:
                    sqlite_type = _pa_type_to_sqlite(f.type)
                    col_name = _validate_table_name(f.name)
                    cols.append(f"{col_name} {sqlite_type}")
                ddl = f"CREATE TABLE IF NOT EXISTS {name} ({', '.join(cols)})"
                self._conn.execute("BEGIN")
                try:
                    self._conn.execute(ddl)
                except Exception:
                    self._conn.execute("ROLLBACK")
                    raise
                self._conn.execute("COMMIT")
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def drop_table(self, name: str) -> None:
        _validate_table_name(name)
        self._conn.execute(f"DROP TABLE IF EXISTS {name}")


    def close(self) -> None:
        from iai_mcp.hippo import (
            _PROCESS_LOCKS,
            _PROCESS_LOCKS_SHARED,
            _PROCESS_LOCKS_GUARD,
        )
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "_hnsw"):
            try:
                with self._hnsw_lock:
                    self._save_index_atomic()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._conn.commit()
        except Exception:  # noqa: BLE001
            pass
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass
        if self._lock_fd is not None:
            lock_key = getattr(self, "_lock_key", None)
            access_mode = getattr(self, "_access_mode", AccessMode.EXCLUSIVE)
            registry = (
                _PROCESS_LOCKS_SHARED
                if access_mode is AccessMode.SHARED
                else _PROCESS_LOCKS
            )
            with _PROCESS_LOCKS_GUARD:
                held = registry.get(lock_key) if lock_key else None
                if held is not None:
                    base_fd, refcount = held
                    if refcount <= 1:
                        try:
                            flock(base_fd, LOCK_UN)
                        except Exception:  # noqa: BLE001
                            pass
                        try:
                            os.close(base_fd)
                        except Exception:  # noqa: BLE001
                            pass
                        del registry[lock_key]
                    else:
                        registry[lock_key] = (base_fd, refcount - 1)
                try:
                    os.close(self._lock_fd)
                except Exception:  # noqa: BLE001
                    pass
                self._lock_fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "HippoDB":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
