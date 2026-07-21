"""Database lifecycle class and transaction helpers for the Hippo storage backend."""

from __future__ import annotations

import contextlib
import errno
from iai_mcp import _flock as fcntl
import logging
import os
import re
import threading
import time
import weakref
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from iai_mcp.hippo import _vecindex
import numpy as np
from iai_mcp import _sqlite_stdlib
from iai_mcp import errors
from iai_mcp.hippo import _schema

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


# Every RO pool open on a store path, so a committed write through ANY writer
# connection for that path invalidates ALL in-process pooled readers — not just
# the pool owned by the HippoDB instance that issued the write.
_RO_POOLS_BY_PATH: dict[str, "weakref.WeakSet"] = {}
_RO_POOLS_LOCK = threading.Lock()


def _register_ro_pool(db_path: str, pool: Any) -> None:
    with _RO_POOLS_LOCK:
        _RO_POOLS_BY_PATH.setdefault(db_path, weakref.WeakSet()).add(pool)


def _mark_path_pools_stale(db_path: str) -> None:
    with _RO_POOLS_LOCK:
        pools = list(_RO_POOLS_BY_PATH.get(db_path, ()))
    for pool in pools:
        try:
            pool.mark_stale()
        except Exception as exc:  # noqa: BLE001 -- no-raise: staleness marking must never fail a write
            _log.debug("mark_stale failed for pool on %s: %s", db_path, exc)


class _MutationSignallingConn:
    """Writer-connection proxy that reports mutations to the RO reader pools.

    A pooled read-only reader is a snapshot-at-open; it never live-tracks the
    writer. Without a staleness bump on EVERY mutating statement (not only
    corpus-count-changing ones), a warm reader keeps serving pre-write field
    values (e.g. an updated provenance_json) indefinitely.
    """

    _MUTATING_VERBS = frozenset(
        {
            "INSERT",
            "UPDATE",
            "DELETE",
            "REPLACE",
            "CREATE",
            "ALTER",
            "DROP",
            "VACUUM",
            "COMMIT",
            "END",
        }
    )

    def __init__(self, wrapped: Any, on_mutation: Callable[[], None]) -> None:
        object.__setattr__(self, "_wrapped", wrapped)
        object.__setattr__(self, "_on_mutation", on_mutation)

    def _signal_if_mutating(self, sql: Any) -> None:
        try:
            text = str(sql)
            verb = text.lstrip().split(None, 1)[0].rstrip(";").upper()
        except (AttributeError, IndexError):
            return
        mutating = verb in self._MUTATING_VERBS
        if not mutating and verb == "WITH":
            # A CTE-prefixed write (WITH ... UPDATE/DELETE/INSERT) does not
            # start with a mutating verb; a missed signal here means warm RO
            # slots serve pre-write values until an unrelated fence trips.
            # Word-scan the statement — over-signalling a CTE read costs one
            # cheap slot refresh; under-signalling a CTE write costs
            # correctness.
            upper = text.upper()
            mutating = any(
                re.search(rf"\b{v}\b", upper)
                for v in ("INSERT", "UPDATE", "DELETE", "REPLACE")
            )
        if mutating:
            try:
                self._on_mutation()
            except Exception as exc:  # noqa: BLE001 -- no-raise into the write path
                _log.debug("RO-pool mutation signal failed: %s", exc)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        result = self._wrapped.execute(*args, **kwargs)
        if args:
            self._signal_if_mutating(args[0])
        return result

    def executemany(self, *args: Any, **kwargs: Any) -> Any:
        result = self._wrapped.executemany(*args, **kwargs)
        if args:
            self._signal_if_mutating(args[0])
        return result

    def commit(self) -> Any:
        result = self._wrapped.commit()
        try:
            self._on_mutation()
        except Exception as exc:  # noqa: BLE001 -- no-raise into the write path
            _log.debug("RO-pool mutation signal failed: %s", exc)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_wrapped"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_wrapped"), name, value)

    @property
    def __class__(self) -> type:  # type: ignore[override]
        # isinstance-transparent: callers type-checking the connection must
        # see the engine class, not the signalling seam.
        return type(object.__getattribute__(self, "_wrapped"))


# Bounds for the deferred-embed pass driven by the wake sequence. The pass
# embeds pending rows (written cheaply by the in-daemon drain) in windows of
# REEMBED_BATCH_SIZE, releasing idle allocator pages between windows, and yields
# once the resident set exceeds REEMBED_RSS_SOFT_CAP — leaving the remaining
# rows pending for the next cycle. This keeps a large deferred backlog from
# climbing the resident set through one long synchronous embed run. Both are
# operator-overridable; a non-positive / malformed value disables that bound.
REEMBED_BATCH_SIZE_DEFAULT = 128
REEMBED_RSS_SOFT_CAP_DEFAULT_BYTES = 2_684_354_560
REEMBED_RSS_GROWTH_BUDGET_BYTES = 805_306_368
# Absolute backstop over the growth-relative cap: the growth budget floats
# with the pass's starting baseline, so a baseline creeping cycle-over-cycle
# (allocator fragmentation) would otherwise never bound absolute RSS. Half of
# physical memory is far above any legitimate daemon baseline; at that level
# starving the backlog is the correct trade against an OS memory kill.
REEMBED_RSS_HARD_CEILING_FRACTION = 0.5


def _reembed_rss_hard_ceiling() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total * REEMBED_RSS_HARD_CEILING_FRACTION)
    except Exception:  # noqa: BLE001 -- unknown total memory: no ceiling
        return 0


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


def _ro_pool_off() -> bool:
    """Kill-switch: force ro_conn() onto the writer-lock fallback path."""
    return os.environ.get("IAI_MCP_RO_POOL_OFF", "").strip() in ("1", "true", "TRUE", "yes")


_txn_owners: dict[int, int] = {}
_txn_owners_lock: threading.Lock = threading.Lock()


@contextlib.contextmanager  # type: ignore[misc]
def _txn(conn: Any):
    from iai_mcp.hippo import HippoIntegrityError
    # Owner keys use the UNWRAPPED connection: two HippoDBs sharing one raw
    # lilli connection wrap it in distinct signalling proxies, and keying on
    # the proxy id would give each its own owner slot — the double-writer
    # tripwire would silently yield instead of raising.
    conn_id = id(getattr(conn, "_wrapped", conn))
    if conn.in_transaction:
        with _txn_owners_lock:
            owner = _txn_owners.get(conn_id)
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
    with _txn_owners_lock:
        _txn_owners[conn_id] = threading.get_ident()
    try:
        conn.execute("BEGIN")
        try:
            yield
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except (errors.Error, RuntimeError):
                # errors.Error covers driver rollback failures (both the
                # engine and the translated stdlib driver). RuntimeError
                # covers the lilli engine's pager.rollback() when the
                # transaction state is already cleared — suppress both so
                # the original exception propagates unmasked.
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


def _resolve_effective_driver(db_path: str) -> str:
    """Resolve the storage driver for a backing file, on-disk format first.

    The ``LILLI_STORAGE_DRIVER`` environment variable expresses the *intended*
    backend, but it is only visible to processes that inherit it (the daemon,
    via its service manager). A short-lived client process — e.g. the ``iai
    recall`` direct-store fallback — runs in a user shell where the variable is
    unset and thus resolves to the ``"stdlib"`` default. Selecting the driver
    from that env alone would open a native-engine store with the stdlib
    ``sqlite3`` driver, which raises ``file is not a database`` and silently
    yields zero results.

    The on-disk file format is the authority. When a non-empty backing file
    already exists, its full 16-byte header unambiguously identifies the
    writer:

    - a native-engine store begins with the engine magic (``DB_MAGIC``);
    - a stdlib ``sqlite3`` store begins with the full SQLite header magic
      (``b"SQLite format 3\\x00"``, also 16 bytes).

    Detection wins over the env for an existing file: whichever process opens
    the store reads it with the driver that wrote it, regardless of the ambient
    variable. The env is honoured only when there is no file to sniff yet (a
    fresh store), so a first open still creates the intended format.
    """
    from iai_mcp.lillibrain.constants import DB_MAGIC  # noqa: PLC0415

    _SQLITE_MAGIC = b"SQLite format 3\x00"

    env_driver = os.environ.get("LILLI_STORAGE_DRIVER", "stdlib").lower()

    try:
        with open(db_path, "rb") as fh:
            header = fh.read(len(DB_MAGIC))
    except OSError:
        header = b""

    if not header:
        # No file yet (fresh store) or unreadable — honour the env intent.
        return env_driver

    if header[:len(DB_MAGIC)] == DB_MAGIC:
        detected = "lilli"
    elif header[:len(_SQLITE_MAGIC)] == _SQLITE_MAGIC:
        detected = "stdlib"
    else:
        # Unknown/corrupt header: defer to the env so the writer that produced
        # it (or the intended driver) reports its own error, unchanged.
        return env_driver

    if detected != env_driver:
        _log.info(
            "storage driver resolved from on-disk format: file=%s detected=%s "
            "env=%s — using detected (on-disk format is authoritative)",
            db_path,
            detected,
            env_driver,
        )
    return detected


def _open_storage_connection(
    db_path: str,
    *,
    embed_dim: int,
    cached_statements: int,
    driver: str | None = None,
):
    """Open the storage connection for the backing database file.

    The backend is selected by ``driver`` when provided, else resolved from the
    on-disk file format (falling back to the ``LILLI_STORAGE_DRIVER`` env for a
    fresh store) via :func:`_resolve_effective_driver`:

    - ``"stdlib"`` / any non-``"lilli"`` value → the stdlib ``sqlite3`` driver
      (the default; the connection kwargs are byte-identical to the historical
      direct ``sqlite3.connect`` call).
    - ``"lilli"`` → the in-tree pager/B+tree/WAL engine. The engine connection
      is registered under ``db_path`` so raw-access helpers can reach it without
      opening a second pager on the same file.

    The connection is the only seam through which the encrypted fields flow;
    encryption is applied above this layer (in the table/crypto code), so no
    crypto key or provider is ever threaded through here — the engine stores
    ciphertext BLOBs and the plaintext embedding only. ``cached_statements`` is
    honoured by the stdlib driver and accepted-and-ignored by the engine.
    """
    if driver is None:
        driver = _resolve_effective_driver(db_path)
    if driver == "lilli":
        from iai_mcp.lillibrain.connection import (  # noqa: PLC0415
            get_lilli_engine_conn,
            register_lilli_conn,
        )
        from iai_mcp_native import engine  # noqa: PLC0415

        # The engine holds an exclusive advisory file lock on the store for the
        # life of the connection, so a single file admits exactly one open pager
        # in-process (the single-writer durability invariant). A second open at
        # the same path therefore reuses the already-registered live connection
        # rather than opening a second pager — which would fail the lock. The
        # in-process serialization that makes this safe lives one layer up in
        # HippoDB (_PROCESS_LOCKS refcount + _conn_lock); the engine connection
        # is shared, never duplicated. The borrower does not own the connection
        # and must not close it (the owner — the first opener — closes it).
        existing = get_lilli_engine_conn(db_path)
        if existing is not None:
            return existing, False

        # First opener: the Rust engine opens the store, replays the persisted
        # DDL + root mappings, and presents a sqlite3-shaped Connection. It is
        # registered under db_path so the daemon-down raw path resolves it and
        # calls conn.raw_conn(read_only=) on it for direct access.
        # A just-closed pager releases its advisory lock a beat after close()
        # returns; retry briefly so open-after-close does not fail on that
        # lag. A lock held by a LIVE owner (the daemon) still fails after the
        # bound — callers handle that honestly.
        _deadline = time.monotonic() + 2.0
        while True:
            try:
                conn = engine.Connection.open(db_path, embed_dim)
                break
            except errors.OperationalError as exc:
                if (
                    "store is locked" not in str(exc)
                    or time.monotonic() >= _deadline
                ):
                    raise
                time.sleep(0.05)
        register_lilli_conn(db_path, conn)
        return conn, True
    conn = _sqlite_stdlib.connect(
        db_path,
        check_same_thread=False,
        isolation_level=None,
        cached_statements=cached_statements,
    )
    return conn, True


class HippoDB:

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        crypto_key_provider: Callable[[], bytes] | None = None,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
        persist_index: bool = True,
        _lock_timeout_override: float | None = None,
    ) -> None:
        from iai_mcp.hippo import _resolve_root, EMBED_DIM as _EMBED_DIM
        self._crypto_key_provider: Callable[[], bytes] | None = crypto_key_provider
        self._access_mode: AccessMode = access_mode
        self._read_only: bool = read_only
        # Disk hnsw is derived data (SQLite is the source of truth); only the
        # index-owning handle may persist it, or a stale in-RAM index from a
        # concurrent SHARED client would clobber the owner's on-disk file.
        self._persist_index: bool = persist_index

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
        self._db_path: str = str(db_path)
        _cached_stmts_raw = os.environ.get("IAI_MCP_SQLITE_CACHED_STATEMENTS", "128")
        try:
            _cached_stmts = max(0, int(_cached_stmts_raw))
        except ValueError:
            _cached_stmts = 128

        # Resolved before the connection so the storage driver receives the
        # embedding dimension (the engine sizes its vector column from it).
        # The meta-table override below may refine this for an existing store.
        _env_dim = os.environ.get("IAI_MCP_EMBED_DIM")
        self._embed_dim: int = (
            int(_env_dim) if _env_dim and _env_dim.isdigit() else _EMBED_DIM
        )

        # Record the driver chosen at open-time so close() uses the same branch
        # regardless of later env mutations (e.g. between open and close in tests).
        # Resolved from the on-disk file format first (env only for a fresh
        # store) so a process without LILLI_STORAGE_DRIVER — e.g. the user-shell
        # `iai recall` direct-store fallback — still opens a native-engine store
        # with the engine, never mis-opening it as stdlib sqlite3.
        self._storage_driver: str = _resolve_effective_driver(self._db_path)
        self._conn, self._owns_conn = _open_storage_connection(
            self._db_path,
            embed_dim=self._embed_dim,
            cached_statements=_cached_stmts,
            driver=self._storage_driver,
        )
        if self._storage_driver == "lilli":
            # The registry keeps the RAW engine connection; only this
            # instance's SQL surface goes through the signalling proxy.
            self._conn = _MutationSignallingConn(
                self._conn,
                lambda _p=self._db_path: _mark_path_pools_stale(_p),
            )
        else:
            # Engine rows are name-keyed natively; only the stdlib driver
            # needs the Row factory (assigning it on the lilli branch would
            # lazily import sqlite3 into an engine-only process).
            self._conn.row_factory = _sqlite_stdlib.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=2000")
        # Cap the SQLite page cache so a long-running daemon process cannot
        # let it drift unbounded across cycles. Negative values are KiB.
        # tracemalloc cannot see C-level allocations, so this cap is the
        # only explicit bound on the page cache footprint in process RSS.
        _DEFAULT_CACHE_KIB = 65536  # 64 MiB
        _MIN_CACHE_KIB = 4096  # 4 MiB — below this the page cache thrashes
        _cache_kib = os.environ.get("IAI_MCP_SQLITE_CACHE_SIZE_KIB", str(_DEFAULT_CACHE_KIB))
        try:
            _cache_kib_int = int(_cache_kib)
        except ValueError:
            _cache_kib_int = _DEFAULT_CACHE_KIB
        # Non-positive → use default; sub-floor positive → clamp to floor.
        # Never abs()-rewrite a negative env value; treat <=0 as "use default".
        if _cache_kib_int <= 0:
            _cache_kib_int = _DEFAULT_CACHE_KIB
        elif _cache_kib_int < _MIN_CACHE_KIB:
            _cache_kib_int = _MIN_CACHE_KIB
        self._conn.execute(f"PRAGMA cache_size=-{_cache_kib_int}")
        if read_only:
            self._conn.execute("PRAGMA query_only=ON")

        self._closed: bool = False
        self._hnsw_path: Path = self._hippo_dir / "records.hnsw"
        self._hnsw_tmp_path: Path = self._hippo_dir / "records.hnsw.tmp"
        self._hnsw_lock: threading.RLock = threading.RLock()
        self._conn_lock: threading.RLock = threading.RLock()
        # Single-writer invariant for ANN rebuilds: every rebuild entry point
        # (wake sequence, mid-run reconcile, sleep-pipeline reconcile,
        # migrations, child-worker swap in maintenance.py) MUST hold this lock
        # across build->swap. Two unserialized builders mutating one vector-index
        # buffer is a C++ data race (GIL released inside add_items); two
        # unserialized swaps can publish a stale buffer as active. Lock order:
        # _rebuild_lock OUTER, then _hnsw_lock, then _conn_lock.
        self._rebuild_lock: threading.RLock = threading.RLock()
        # Journal of active-index mutations committed while a rebuild is in
        # flight. Non-None only between snapshot and swap (set/cleared under
        # _hnsw_lock by the rebuilder); writers holding _hnsw_lock append
        # ("add", vec, label, rid) / ("del", None, label, rid) so the
        # swapped-in buffer AND the off-lock-loaded label map replay them —
        # without it a record add()-committed during the build window
        # silently vanishes from ANN until the next rebuild.
        self._ann_journal: "list[tuple[str, Any, int, str]] | None" = None
        # Schema (including the records vec_label index) is persisted into engine
        # meta ONLY by a read-write open — _ensure_tables runs the CREATE INDEX
        # passthrough that meta.record_ddl stores. A read-only open reconstructs
        # the catalog purely from previously-persisted DDL via meta.replay, so it
        # sees the vec_label index only if some earlier read-write open already
        # recorded it. A store upgraded on an older binary whose FIRST post-upgrade
        # open is read-only therefore full-scans its vec_label lookups until a
        # read-write open heals the meta — correct (scan-parity), just ~8s slow.
        # In the normal daemon lifecycle a read-write boot precedes any read-only
        # recall, so this window is transient. See _db._debug_log_readonly_index_gap
        # for the operator-visible DEBUG signal on the read-only path.
        if not read_only:
            self._ensure_tables()

        if not read_only:
            with self._conn_lock:
                meta_dim = self._conn.execute(
                    "SELECT value FROM _hippo_meta WHERE key = 'embed_dim'"
                ).fetchone()
            if meta_dim is not None:
                self._embed_dim = int(meta_dim[0])
        self._label_map: dict[str, int] = {}
        self._write_counter: int = 0
        # Observable counter for the reuse-path collision fallback (fresh
        # C++ allocation on a rebuild). The zero-per-cycle-allocation invariant
        # holds only while this stays near zero; consumers read it via
        # rebuild results / doctor surfaces rather than grepping warnings.
        self._reuse_collision_count: int = 0
        # Recency buffer integration callbacks — registered by MemoryStore eagerly
        # at construction time (before any lifecycle tick can fire a reembed or
        # pending-row write that must be visible to the buffer).
        # Both slots are None until a MemoryStore registers them; HippoDB never
        # calls them if None, and fires them inside try/except so a callback
        # failure never disrupts a write or the reembed pass.
        self._recency_reconcile: "Callable[[str], None] | None" = None
        self._recency_pending_feed: "Callable | None" = None
        # Corpus-count invalidation callback — registered by MemoryStore eagerly
        # at construction time (after _corpus_count_cache is built).  Fires with
        # the affected count-cache key names so the caller invalidates only the
        # counts that changed.  None until a MemoryStore registers it; HippoDB
        # fires it hook-isolated (try/except) and never calls it when None.
        self._corpus_count_invalidate: "Callable | None" = None
        # Exact-delta twin: fires with (key, delta) where the write's count
        # delta is known exactly, so the cached count shifts instead of being
        # dropped and recounted by a full filtered scan on the next read.
        self._corpus_count_adjust: "Callable | None" = None
        # Exact-cosine authority feed callback — registered by MemoryStore
        # eagerly at construction time.  Fires with (record_id: str, vec) at
        # the reembed pending->active flag flip so the resident matrix stays
        # in sync without a rebuild.  None until a MemoryStore registers it;
        # HippoDB fires it hook-isolated and never calls it when None.
        self._exact_index_feed: "Callable | None" = None

        # Pooled read-only reader substrate (lilli only, eager object / lazy
        # fill — construction here pays nothing until the first ro_conn()
        # borrow actually opens a slot). None on the stdlib driver: ro_conn()
        # falls back to the shared writer connection unconditionally.
        self._ro_pool: "Any | None" = None
        if self._storage_driver == "lilli":
            from iai_mcp.hippo._ro_pool import RoConnPool as _RoConnPool
            self._ro_pool = _RoConnPool(self._db_path)
            _register_ro_pool(self._db_path, self._ro_pool)

        if read_only:
            self._hnsw: _vecindex.Index | None = None  # type: ignore[assignment]
            self._hnsw_standby: _vecindex.Index | None = None
            try:
                self._repopulate_label_map_from_sqlite()
            except Exception:  # noqa: BLE001
                pass
        else:
            self._repopulate_label_map_from_sqlite()
            self._initialize_hnsw_index()

        # One-shot guard so the read-only-full-scan DEBUG signal fires at most
        # once per connection instead of on every recall (hot path stays clean).
        self._readonly_index_gap_logged: bool = False

    def _debug_log_readonly_index_gap(self, conn: object) -> None:
        """Emit a one-time DEBUG signal when a read-only open serves a
        ``vec_label IN`` recall by full scan.

        A store upgraded on an older binary whose first post-upgrade open is
        read-only lacks the records vec_label index in engine meta (it is
        persisted only by a read-write ``_ensure_tables``), so its recall
        full-scans — correct but ~8s slow. A read-write boot heals it. This
        signal lets an operator observe an un-upgraded store still on the slow
        path rather than silently paying the latency; it never runs a check on
        the hot path (only reads a counter the engine already maintains) and
        logs at most once per connection.
        """
        if self._readonly_index_gap_logged or not self._read_only:
            return
        counter = getattr(conn, "full_scan_count", None)
        if counter is None:  # stdlib driver has no engine scan counter
            return
        try:
            scans = counter()
        except Exception:  # noqa: BLE001
            return
        if scans > 0:
            self._readonly_index_gap_logged = True
            _log.debug(
                "read-only open served a vec_label lookup by full scan "
                "(records vec_label index absent from engine meta): a store "
                "upgraded on an older binary whose first post-upgrade open is "
                "read-only stays on the slow path until a read-write open "
                "persists the index. A read-write daemon boot heals it."
            )

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
                    fcntl.flock(base_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
            _consolidation_intent_is_stale,
        )
        _intent_path = self._hippo_dir / ".consolidation-pending"

        def _intent_blocks() -> bool:
            # Honor the consolidation intent ONLY when a live holder owns it. An
            # orphaned intent (PID dead / unreadable / past the time ceiling) is
            # left behind by a crashed daemon — a dead daemon must NEVER gate a
            # write, so self-heal by removing it and proceeding to acquire SH.
            if not _intent_path.exists():
                return False
            if _consolidation_intent_is_stale(_intent_path):
                try:
                    _intent_path.unlink()
                    _log.warning(
                        "hippo_shared_lock: removed orphaned consolidation "
                        "intent at %s (no live holder) — proceeding with "
                        "daemon-free acquisition",
                        _intent_path,
                    )
                except FileNotFoundError:
                    pass
                except OSError:
                    return True
                return False
            return True

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
            if _intent_blocks():
                if time.monotonic() >= deadline:
                    break
                time.sleep(_SHARED_RETRY_SLEEP_S)
                continue

            try:
                fcntl.flock(base_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(_SHARED_RETRY_SLEEP_S)
                    continue
                os.close(base_fd)
                raise

            if _intent_blocks():
                fcntl.flock(base_fd, fcntl.LOCK_UN)
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
                fcntl.flock(base_fd, fcntl.LOCK_UN)
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
            if fcntl.SHARED_IS_EXCLUSIVE:
                # No shared locks on this platform: a re-lock of the region
                # this process already owns raises (and a blocking one polls
                # forever against itself) — keep the held exclusive region and
                # change only the bookkeeping label.
                pass
            else:
                try:
                    fcntl.flock(base_fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                except OSError:
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

        # Stamp the intent with the holder PID + timestamp so a daemon-free
        # writer can tell a live in-progress consolidation (PID alive, holds the
        # EX flock) from an orphan left by a SIGKILL'd daemon (PID dead). The
        # write is atomic (temp file + rename) so a concurrent reader never sees
        # a half-written PID. A pre-existing intent from a crashed daemon is
        # reclaimed — the rename overwrites it with our own live PID.
        _intent_payload = f"{os.getpid()}\n{time.time()}\n".encode()
        _intent_tmp = self._hippo_dir / f".consolidation-pending.tmp.{os.getpid()}"
        try:
            fd = os.open(str(_intent_tmp), os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
            try:
                os.write(fd, _intent_payload)
            finally:
                os.close(fd)
            os.replace(str(_intent_tmp), str(_intent_path))
        except OSError:
            try:
                _intent_tmp.unlink()
            except OSError:
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
        if held is not None and fcntl.SHARED_IS_EXCLUSIVE:
            # The "shared" label on this platform already holds the exclusive
            # region — re-locking it would fail against ourselves. Relabel.
            acquired = True
        while not acquired and time.monotonic() < deadline:
            try:
                fcntl.flock(base_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
        with self._conn_lock:
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
                    idx = _vecindex.Index(space="cosine", dim=self._embed_dim)
                    idx.load_index(str(candidate), max_elements=cap, allow_replace_deleted=True)
                    idx.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
                    idx.set_num_threads(1)
                    self._hnsw: _vecindex.Index = idx
                    loaded = True
                    break
                except Exception as exc:  # noqa: BLE001
                    _log.warning("Failed to load vector index from %s: %s", candidate, exc)

        if not loaded:
            self._hnsw = _vecindex.Index(space="cosine", dim=self._embed_dim)
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
                    "No valid vector-index file found; rebuilding from %d SQLite records",
                    sqlite_count,
                )
                self._rebuild_index_from_sqlite()
                self._allocate_standby_index(cap)
                return

        active_label_count = len(self._label_map)
        hnsw_loaded_count = self._hnsw.get_current_count()
        vectors_match = self._loaded_hnsw_matches_sqlite_sample()
        if (
            active_label_count != sqlite_count
            or hnsw_loaded_count != sqlite_count
            or not vectors_match
        ):
            _log.info(
                "Boot integrity check: labels=%d hnsw=%d sqlite=%d "
                "vectors_match=%s — rebuilding",
                active_label_count,
                hnsw_loaded_count,
                sqlite_count,
                vectors_match,
            )
            self._rebuild_index_from_sqlite()

        # Second standby buffer, reused across rebuilds so steady-state
        # consolidation cycles never allocate a fresh C++ index.  Allocated
        # after any boot-integrity rebuild so the rebuild above takes the
        # fresh-alloc fallback path (the standby is intentionally absent then).
        self._allocate_standby_index(cap)

    def _loaded_hnsw_matches_sqlite_sample(self) -> bool:
        """Reject a stale on-disk vector index after a model or dimension
        migration: sample rows from both ends of the label range and require
        the indexed vector to equal the stored embedding (normalized)."""
        with self._conn_lock:
            rows = self._conn.execute(
                "SELECT vec_label, embedding FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
                " ORDER BY vec_label LIMIT 3"
            ).fetchall()
            rows += self._conn.execute(
                "SELECT vec_label, embedding FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
                " ORDER BY vec_label DESC LIMIT 3"
            ).fetchall()
        try:
            for row in rows:
                stored = np.frombuffer(row["embedding"], dtype=np.float32)
                indexed = self._hnsw.get_items([int(row["vec_label"])])[0]
                norm = float(np.linalg.norm(stored))
                expected = stored if norm == 0.0 else stored / norm
                if indexed.shape != expected.shape or not np.allclose(
                    indexed,
                    expected,
                    rtol=1e-4,
                    atol=1e-5,
                ):
                    return False
        except (RuntimeError, ValueError, IndexError):
            return False
        return True

    def _allocate_standby_index(self, cap: int) -> None:
        from iai_mcp.hippo import (
            HNSW_EF_CONSTRUCTION,
            HNSW_M,
            HNSW_EF,
            RECALL_INDEX_EF,
        )
        standby = _vecindex.Index(space="cosine", dim=self._embed_dim)
        standby.init_index(
            max_elements=cap,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
            allow_replace_deleted=True,
        )
        standby.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
        standby.set_num_threads(1)
        self._hnsw_standby: _vecindex.Index | None = standby

    def _load_label_map_from_sqlite(self) -> "dict[str, int]":
        """Fresh id->vec_label map for the live corpus. O(corpus) — callers
        on the rebuild path run this OFF _hnsw_lock and commit the result by
        an O(1) reference swap; holding the recall lock across this read
        stalls every concurrent knn_query for the scan's duration."""
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
        from iai_mcp.hippo import HippoIntegrityError  # noqa: PLC0415

        label_map: dict[str, int] = {}
        for row in rows:
            label = row["vec_label"]
            if label is None:
                # The boot heal backfills these; one appearing here means a
                # writer is still inserting without the rowid alias.
                raise HippoIntegrityError(
                    f"records row {row['id']!r} has NULL vec_label; the "
                    "records table lost its INTEGER PRIMARY KEY alias — "
                    "reopen the store to heal, and report the writer"
                )
            label_map[row["id"]] = int(label)
        return label_map

    def _repopulate_label_map_from_sqlite(self) -> None:
        fresh = self._load_label_map_from_sqlite()
        self._label_map.clear()
        self._label_map.update(fresh)

    def _rebuild_index_from_sqlite(self) -> dict:
        # All rebuild entry points serialize here — see _rebuild_lock's
        # declaration for the invariant. The journal is cleared in the finally
        # so an exception mid-build never leaves writers appending forever.
        with self._rebuild_lock:
            try:
                return self._rebuild_index_impl()
            finally:
                with self._hnsw_lock:
                    self._ann_journal = None

    def _rebuild_index_impl(self) -> dict:
        from iai_mcp.hippo import (
            HNSW_INITIAL_CAPACITY,
            HNSW_EF_CONSTRUCTION,
            HNSW_M,
            HNSW_EF,
            RECALL_INDEX_EF,
        )
        # Journal must open BEFORE the row snapshot: a write that lands after
        # the snapshot is replayed from the journal at swap time; one that
        # lands before is in the snapshot. A write in both is an idempotent
        # in-place label update.
        with self._hnsw_lock:
            self._ann_journal = []
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
            self._hnsw = _vecindex.Index(space="cosine", dim=self._embed_dim)
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
        self._refill_index_in_place(standby, vecs, labels)

        # Correctness net for the reuse fast-path: verify every expected label
        # is resolvable in the rebuilt standby before committing it; the check
        # is cheap relative to the build and runs lock-free on the
        # not-yet-published buffer. With the in-place refill this should never
        # fire — a hit means a new drop mechanism, count it and fall back to
        # the known-good fresh-allocation build.
        commit_buffer = standby
        if n > 0 and not self._standby_has_all_labels(standby, labels):
            _log.warning(
                "ANN reuse-path rebuild dropped live records (expected %d labels);"
                " falling back to a fresh-allocation rebuild",
                n,
            )
            self._reuse_collision_count += 1
            commit_buffer = self._build_fresh_index(cap, vecs, labels)

        # Everything O(corpus) happens OFF the recall lock: the fresh label
        # map is loaded here, and the not-yet-published buffer is saved to
        # disk here (it is private, so the save needs no lock). The disk file
        # can miss journal rows landing before the swap — the boot integrity
        # check (labels vs sqlite count) heals that with a rebuild, which is
        # the pre-existing cold-boot path, not a new failure mode.
        new_label_map = self._load_label_map_from_sqlite()
        self._save_index_atomic(index=commit_buffer)

        # Publish the freshly-built buffer together with its label map under the
        # recall lock so a concurrent knn_query reader always sees a consistent
        # (buffer, label_map) pair — never a half-deleted or half-refilled index.
        # The critical section is O(journal) + two reference swaps — a recall
        # landing mid-swap no longer stalls behind a corpus scan or a disk
        # write. Swap atomicity relies on the CPython GIL; a free-threaded
        # build needs explicit synchronization at this site.
        with self._hnsw_lock:
            journal = self._ann_journal
            self._ann_journal = None
            if journal:
                self._apply_ann_journal(commit_buffer, journal, new_label_map)
            if commit_buffer is standby:
                self._hnsw, self._hnsw_standby = self._hnsw_standby, self._hnsw
            else:
                self._hnsw, self._hnsw_standby = commit_buffer, self._hnsw
            self._label_map = new_label_map

        return {
            "action": "rebuild",
            "rebuilt_count": n,
            "reuse_collisions_total": self._reuse_collision_count,
        }

    @staticmethod
    def _refill_index_in_place(
        index: "_vecindex.Index", vecs: "np.ndarray | None", labels: "np.ndarray | None"
    ) -> None:
        """Bring a retired buffer to the snapshot state by updating in place:
        survivors are re-added under their existing labels (an existing label
        MUST update its slot, never allocate or re-home one — the invariant
        this refill relies on), stale labels are only marked deleted, and
        labels new to the buffer append. No slot a live label points at is
        ever reassigned mid-refill.
        """
        incoming: "dict[int, int]" = (
            {int(l): i for i, l in enumerate(labels)} if labels is not None else {}
        )
        in_lookup = {int(x) for x in index.get_ids_list()}
        for stale in in_lookup - set(incoming):
            try:
                index.mark_deleted(stale)
            except RuntimeError:
                # Already soft-deleted in a prior cycle — already the goal state.
                pass
        survivors = [l for l in incoming if l in in_lookup]
        for l in survivors:
            try:
                index.unmark_deleted(l)
            except RuntimeError:
                # Not deleted — the common case; nothing to restore.
                pass
        if survivors:
            index.add_items(
                vecs[[incoming[l] for l in survivors]],
                np.array(survivors, dtype=np.int64),
            )
        new = [l for l in incoming if l not in in_lookup]
        if new:
            needed = index.get_current_count() + len(new)
            if needed > index.get_max_elements():
                index.resize_index(max(needed * 2, 16))
            index.add_items(
                vecs[[incoming[l] for l in new]],
                np.array(new, dtype=np.int64),
                replace_deleted=True,
            )

    def _apply_ann_journal(
        self,
        buffer: "_vecindex.Index",
        entries: "list[tuple[str, Any, int, str]]",
        label_map: "dict[str, int] | None" = None,
    ) -> None:
        """Replay journaled active-index mutations into the about-to-publish
        buffer (and, when given, the about-to-publish label map — it was
        loaded off-lock BEFORE these writes landed). Held under _hnsw_lock by
        the caller so no new entries land mid-replay. Adds never use
        replace_deleted: a journaled label can already be live in the buffer
        (committed pre-snapshot AND journaled), and the replace path would
        re-home it, leaking its old slot as a duplicate ghost;
        unmark-then-add updates in place or plain-inserts."""
        for op, vec, label, rid in entries:
            if op == "add":
                try:
                    buffer.unmark_deleted(label)
                except RuntimeError:
                    pass
                try:
                    needed = buffer.get_current_count() + 1
                    if needed > buffer.get_max_elements():
                        buffer.resize_index(max(needed * 2, 16))
                    buffer.add_items(vec, np.array([label], dtype=np.int64))
                except RuntimeError as exc:
                    _log.warning(
                        "ANN journal replay: add of label %d failed: %s", label, exc
                    )
                    continue
                if label_map is not None and rid:
                    label_map[rid] = int(label)
            else:
                try:
                    buffer.mark_deleted(label)
                except RuntimeError:
                    # Not present in the snapshot-built buffer — already absent.
                    pass
                if label_map is not None and rid:
                    label_map.pop(rid, None)

    @staticmethod
    def _standby_has_all_labels(index: "_vecindex.Index", labels: np.ndarray) -> bool:
        """True iff every expected label is resolvable in the rebuilt index.

        get_items raises when any label is missing (the slot-reuse drop), so a
        single bulk lookup over the full label set is a complete check: it
        succeeds only when all labels are present, and the active element count
        must also equal the expected live count.
        """
        try:
            if index.get_current_count() < len(labels):
                return False
            # get_items raises RuntimeError("Label not found") for a dropped
            # label, so a successful bulk lookup proves all labels are present.
            index.get_items(labels.tolist())
        except RuntimeError:
            return False
        return True

    def _build_fresh_index(
        self, cap: int, vecs: np.ndarray, labels: np.ndarray
    ) -> "_vecindex.Index":
        """Allocate and fill a fresh ANN index — the boot-path build, reused as
        the reuse-path correctness fallback. No slot reuse, so no overlap drop."""
        from iai_mcp.hippo import (
            HNSW_EF_CONSTRUCTION,
            HNSW_M,
            HNSW_EF,
            RECALL_INDEX_EF,
        )
        fresh = _vecindex.Index(space="cosine", dim=self._embed_dim)
        fresh.init_index(
            max_elements=cap,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
            allow_replace_deleted=True,
        )
        fresh.set_ef(max(HNSW_EF, RECALL_INDEX_EF))
        fresh.set_num_threads(1)
        fresh.add_items(vecs, labels)
        return fresh

    def _save_index_atomic(self, index: "_vecindex.Index | None" = None) -> None:
        # Saving the ACTIVE index requires _hnsw_lock (concurrent add_items
        # during save_index is a C++ race); the rebuild path instead passes
        # its still-private commit buffer and saves lock-free. The temp file
        # is UNIQUE per save (mkstemp): the off-lock rebuild save and the
        # on-lock periodic savers would otherwise interleave on one shared
        # temp name and publish a torn index file.
        if not self._persist_index:
            _log.debug("hnsw index save skipped: non-persisting handle")
            return
        import tempfile as _tempfile

        target = index if index is not None else self._hnsw
        tmp_name: "str | None" = None
        try:
            fd, tmp_name = _tempfile.mkstemp(
                dir=str(self._hnsw_path.parent),
                prefix=self._hnsw_path.name + ".",
                suffix=".tmp",
            )
            os.close(fd)
            target.save_index(tmp_name)
            os.replace(tmp_name, self._hnsw_path)
            tmp_name = None
        except OSError as exc:
            _log.warning("vector index save failed: %s", exc)
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass

    def _maybe_resize(self) -> None:
        from iai_mcp.hippo import HNSW_RESIZE_HEADROOM
        current = self._hnsw.get_current_count()
        max_el = self._hnsw.get_max_elements()
        if max_el > 0 and current > HNSW_RESIZE_HEADROOM * max_el:
            self._hnsw.resize_index(max_el * 2)

    # ------------------------------------------------------------------
    # Recency buffer callback registration (set by MemoryStore at init time)
    # ------------------------------------------------------------------

    def register_recency_reconcile(self, cb: "Callable[[str], None]") -> None:
        """Register the callback fired after each successful reembed flag-flip.

        The callback receives the record id (str) and should flip the
        ``embedding_pending`` flag on the in-process recency buffer entry for
        that id.  Registered once and eagerly by MemoryStore at construction.
        """
        self._recency_reconcile = cb

    def register_recency_pending_feed(self, cb: "Callable") -> None:
        """Register the callback fired after each successful insert_pending_row.

        The callback receives the same plaintext keyword arguments that were
        passed to ``insert_pending_row`` (minus ``updated_at`` and ``tier``
        which the buffer does not use).  Fired before the method returns so
        the buffer is current for any reader that checks immediately after
        the insert.  Registered once and eagerly by MemoryStore at construction.
        """
        self._recency_pending_feed = cb

    def register_corpus_count_invalidate(self, cb: "Callable") -> None:
        """Register the callback fired after each count-changing write in HippoDB.

        The callback receives one or more cache-key names (``"active"``,
        ``"pending"``) as positional strings.  HippoDB fires it inside
        ``insert_pending_row`` (pending-only) and ``reembed_pending_rows``
        (both active and pending).

        Stored in a separate slot from the recency callbacks so the two concerns
        remain independent.  Registered once and eagerly by MemoryStore at
        construction.
        """
        self._corpus_count_invalidate = cb

    def register_corpus_count_adjust(self, cb: "Callable") -> None:
        """Register the exact-delta twin of the invalidation callback.

        The callback receives ``(key, delta)``. HippoDB fires it where the
        write's count delta is known exactly (a pending insert is +1 pending;
        a reembed flip is +1 active / -1 pending), so the cached value shifts
        in place instead of forcing the next reader into a full filtered
        COUNT scan. When absent, the fire sites fall back to invalidation.
        """
        self._corpus_count_adjust = cb

    def mark_ro_pool_stale(self) -> None:
        """Bump the RO pool's staleness generation. No-raise, no-op on stdlib.

        Called by the ``CorpusCountCache.on_invalidate`` hook (wired at
        ``MemoryStore.__init__``) and directly at the ``insert_pending_row`` /
        ``reembed_pending_rows`` fire sites so a bare ``HippoDB`` used without
        a ``MemoryStore`` (no cache exists there) still bounds RO staleness.
        Double-firing through both routes is harmless — the bump is an
        idempotent int increment. Takes only the pool's own trivial lock,
        never ``_conn_lock``/``_hnsw_lock`` — no new lock-ordering edge.
        """
        pool = getattr(self, "_ro_pool", None)
        if pool is None:
            return
        try:
            pool.mark_stale()
        except Exception as exc:  # noqa: BLE001 -- no-raise contract
            _log.debug("mark_ro_pool_stale failed: %s", exc)

    @contextlib.contextmanager
    def ro_conn(self):
        """Yield a read-only connection for the recall read path.

        On the lilli driver (pool healthy, kill-switch unset): yields a
        borrowed pooled lock-free RO connection — acquisition never touches
        ``_conn_lock``. On the stdlib driver, or when
        ``IAI_MCP_RO_POOL_OFF=1``, or on any borrow/open failure: falls back
        to today's exact behavior — enters ``_conn_lock`` and yields the
        shared writer ``_conn``. The fallback IS the current code path, not a
        new one, so every existing caller of this pattern stays correct even
        when the pool is disabled or fails.

        Callers write ``with self.db.ro_conn() as conn: conn.execute(...)``
        uniformly regardless of which branch actually served the call. Only a
        failure to ACQUIRE a pool slot triggers the writer-path fallback — an
        exception raised by the caller's own use of the yielded connection
        (e.g. a rejected write attempt) propagates unchanged, exactly as it
        would from the shared writer connection.
        """
        pool = getattr(self, "_ro_pool", None)
        borrowed = None
        if pool is not None and not _ro_pool_off():
            try:
                borrowed = pool.borrow()
            except Exception as exc:  # noqa: BLE001 -- fall back, never raise into recall
                _log.debug("ro_conn: pool borrow failed, falling back to writer conn: %s", exc)
                try:
                    pool.writer_fallback_count += 1
                except Exception:  # noqa: BLE001 -- counting must never break the fallback
                    pass
                borrowed = None
        if borrowed is not None:
            with borrowed as conn:
                yield conn
            return
        with self._conn_lock:
            yield self._conn

    def register_exact_index_feed(self, cb: "Callable") -> None:
        """Register the callback fired after each successful reembed flag-flip.

        The callback receives the record id (str) and the newly-embedded
        vector, so the resident exact-cosine matrix can upsert the row without
        a full rebuild.  Registered once and eagerly by MemoryStore at
        construction.
        """
        self._exact_index_feed = cb

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
        # Capture the plaintext surface before encryption so the recency buffer
        # feed callback receives the human-readable text (the buffer holds
        # already-decrypted plaintext in volatile RAM, never ciphertext).
        _plaintext_surface = literal_surface
        _plaintext_provenance = provenance_json
        # Encrypt the confidential columns at rest with the same key provider and
        # uuid-derived associated data the normal insert path uses, so a row in
        # the deferred-embed window is indistinguishable from a fully-embedded
        # row on disk. When no key provider is configured these are no-ops and
        # the values are stored as plaintext, unchanged.
        literal_surface = self._encrypt_for_uuid(record_id, literal_surface)
        provenance_json = self._encrypt_for_uuid(record_id, provenance_json)
        # Derive the role value from the tags JSON so the role column is set
        # at insert time. This mirrors the _derive_role helper in the store layer
        # so write-time and read-time derivation are always identical.
        import json as _json
        _pending_role: str | None = None
        try:
            _tags_list = _json.loads(tags_json) if tags_json else []
            for _t in (_tags_list or []):
                if isinstance(_t, str) and _t.startswith("role:"):
                    _pending_role = _t[len("role:"):]
                    break
        except Exception:  # noqa: BLE001 — tags parse failure must not abort insert
            pass

        from iai_mcp.hippo._table import _upsert_record_tags
        # The record row and its tag rows must land atomically: a crash or
        # exception between them would leave a committed pending record with no
        # record_tags, silently invisible to tag-gated recall until a rebuild
        # re-derives them. _txn makes both one transaction (autocommit mode
        # would otherwise commit the INSERT alone), matching _table.add().
        with self._conn_lock, _txn(self._conn):
            self._conn.execute(
                "INSERT INTO records"
                " (id, tier, literal_surface, aaak_index, embedding, embedding_pending,"
                "  provenance_json, created_at, updated_at, tags_json,"
                "  community_id, detail_level, centrality, stability, difficulty,"
                "  pinned, never_decay, never_merge, s5_trust_score,"
                "  schema_version, language,"
                "  hv_tier, structure_hv_payload, role, live)"
                " VALUES (?, ?, ?, '', ?, 1, ?, ?, ?, ?, '', 1, 0.0, 0.0, 0.0,"
                "  0, 0, 0, 0.5, 1, 'en', 'bsc', x'', ?, 1)",
                (
                    record_id,
                    tier,
                    literal_surface,
                    zero_blob,
                    provenance_json,
                    created_at,
                    updated_at,
                    tags_json,
                    _pending_role,
                ),
            )
            _upsert_record_tags(self._conn, record_id, tags_json)

        # Fire the recency buffer pending-feed callback (if registered) so the
        # buffer is current for any reader that checks immediately after this
        # insert — without requiring a boot rebuild.  Hook isolation: any
        # failure must not disrupt the insert.
        _rfeed = self._recency_pending_feed
        if _rfeed is not None:
            try:
                _rfeed(
                    record_id=record_id,
                    literal_surface=_plaintext_surface,
                    tags_json=tags_json,
                    provenance_json=_plaintext_provenance,
                    created_at=created_at,
                    tier=tier,
                )
            except Exception as _exc:  # noqa: BLE001 -- hook isolation
                _log.debug(
                    "recency_pending_feed callback failed id=%s: %s", record_id, _exc
                )

        # A new pending row increases the pending count by exactly one while
        # leaving the active count unchanged — shift the cached value in place
        # (fall back to invalidation when no adjust callback is registered).
        _cca = self._corpus_count_adjust
        if _cca is not None:
            try:
                _cca("pending", 1)
            except Exception as _exc:  # noqa: BLE001 -- hook isolation
                _log.debug(
                    "corpus_count_adjust callback failed (pending) id=%s: %s",
                    record_id,
                    _exc,
                )
        else:
            _cci = self._corpus_count_invalidate
            if _cci is not None:
                try:
                    _cci("pending")
                except Exception as _exc:  # noqa: BLE001 -- hook isolation
                    _log.debug(
                        "corpus_count_invalidate callback failed (pending) id=%s: %s",
                        record_id,
                        _exc,
                    )
        # Direct RO-pool staleness bump for the bare-HippoDB case (no
        # MemoryStore, so no CorpusCountCache.on_invalidate hook exists).
        # Idempotent with the cache-hook route when a MemoryStore IS present.
        self.mark_ro_pool_stale()

    def has_pending_rows(self) -> bool:
        # MUST match reembed_pending_rows' predicate exactly: a tombstoned row
        # keeps its pending flag forever, and a looser check here makes every
        # wake-sequence pass fire (and rebuild the ANN) for rows the embed
        # pass will never select.
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT 1 FROM records WHERE COALESCE(embedding_pending, 0) = 1"
                " AND tombstoned_at IS NULL LIMIT 1"
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
        # The cap must track the pass's own GROWTH, not an absolute number: a
        # daemon whose legitimate resident baseline already exceeds an absolute
        # cap would have every pass die on its first check and the backlog
        # would starve forever. The hard ceiling (fraction of physical memory)
        # backstops the float — a creeping baseline must not carry the "cap"
        # toward an OS memory kill.
        soft_cap = rss_soft_cap_bytes if rss_soft_cap_bytes and rss_soft_cap_bytes > 0 else 0
        if soft_cap > 0:
            rss_start = self._reembed_rss_bytes()
            if rss_start > 0:
                soft_cap = max(soft_cap, rss_start + REEMBED_RSS_GROWTH_BUDGET_BYTES)
            hard_ceiling = _reembed_rss_hard_ceiling()
            if hard_ceiling > 0:
                soft_cap = min(soft_cap, hard_ceiling)
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
            # ONE lock acquisition reads the whole window: a per-row
            # SELECT/UPDATE round-trip against a consolidating pipeline that
            # holds _conn_lock for long stretches makes the pass crawl at
            # lock-wait speed (observed ~1 row/40s on a live SLEEP daemon).
            placeholders = ", ".join("?" for _ in window_ids)
            with self._conn_lock:
                surface_rows = self._conn.execute(
                    "SELECT id, literal_surface FROM records"
                    f" WHERE id IN ({placeholders})",  # nosemgrep: sql-injection
                    tuple(window_ids),
                ).fetchall()
            surfaces = {r["id"]: (r["literal_surface"] or "") for r in surface_rows}

            # Decrypt + embed OUTSIDE the lock. Decrypt mirrors the record-read
            # path: plaintext store passes through, encrypted store recovers the
            # message so the vector is never of the ciphertext. A row whose text
            # is empty, undecryptable, or unembeddable is skipped — no vector is
            # fabricated.
            embedded: list[tuple[bytes, str]] = []
            vecs: dict[str, list[float]] = {}
            for rid in window_ids:
                if rid not in surfaces:
                    continue
                try:
                    surface = self._decrypt_record_field(
                        rid, "literal_surface", surfaces[rid]
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
                embedded.append((_struct.pack(f"<{len(vec)}f", *vec), rid))
                vecs[rid] = vec

            window_count = len(embedded)
            if window_count > 0:
                # ONE lock acquisition writes + commits the whole window.
                with self._conn_lock:
                    self._conn.executemany(
                        "UPDATE records SET embedding = ?, embedding_pending = 0"
                        " WHERE id = ?",
                        embedded,
                    )
                    self._conn.commit()
                count += window_count

            for _blob, rid in embedded:
                # Notify the recency buffer that this row's pending flag has
                # flipped 1→0 so the buffer can move the entry from the pending
                # branch to the role branch without dropping or duplicating it.
                _rr = self._recency_reconcile
                if _rr is not None:
                    try:
                        _rr(str(rid))
                    except Exception as _exc:  # noqa: BLE001 -- hook isolation
                        _log.debug(
                            "recency_reconcile callback failed id=%s: %s", rid, _exc
                        )
                # Feed the flip into the LIVE ANN buffer incrementally — the
                # wake sequence then needs no full index rebuild for a clean
                # pass. A failed add self-heals: the next wake's index/count
                # mismatch probe routes that pass through the full rebuild.
                try:
                    self._ann_add_flipped_row(str(rid), vecs[rid])
                except Exception as _exc:  # noqa: BLE001 -- hook isolation
                    _log.debug(
                        "ann incremental add failed id=%s: %s", rid, _exc
                    )
                # Feed the newly-embedded row into the resident exact-cosine
                # matrix so the flipped row is answerable without a rebuild.
                _xf = self._exact_index_feed
                if _xf is not None:
                    try:
                        _xf(str(rid), vecs[rid])
                    except Exception as _exc:  # noqa: BLE001 -- hook isolation
                        _log.debug(
                            "exact_index_feed callback failed id=%s: %s", rid, _exc
                        )
                # The flip moves exactly one row from pending to active: shift
                # BOTH cached counts in place (touching only one leaves the
                # other stale). Falls back to invalidation when no adjust
                # callback is registered.
                _cca = self._corpus_count_adjust
                if _cca is not None:
                    try:
                        _cca("active", 1)
                        _cca("pending", -1)
                    except Exception as _exc:  # noqa: BLE001 -- hook isolation
                        _log.debug(
                            "corpus_count_adjust callback failed (active,pending)"
                            " id=%s: %s",
                            rid,
                            _exc,
                        )
                else:
                    _cci = self._corpus_count_invalidate
                    if _cci is not None:
                        try:
                            _cci("active", "pending")
                        except Exception as _exc:  # noqa: BLE001 -- hook isolation
                            _log.debug(
                                "corpus_count_invalidate callback failed (active,pending)"
                                " id=%s: %s",
                                rid,
                                _exc,
                            )
            if window_count > 0:
                # Direct RO-pool staleness bump for the bare-HippoDB case (no
                # MemoryStore, so no CorpusCountCache.on_invalidate hook
                # exists). Idempotent with the cache-hook route when a
                # MemoryStore IS present.
                self.mark_ro_pool_stale()

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
                meta = _json.loads(json_path.read_text())
                vec_label = int(meta["vec_label"])
            except Exception as exc:  # noqa: BLE001
                _log.warning("ingest_pending_embeddings: failed to load %s: %s", npy_path, exc)
                continue

            import numpy as _np
            with self._hnsw_lock:
                self._maybe_resize()
                _ingest_vec = _np.array([vec], dtype=_np.float32)
                self._hnsw.add_items(
                    _ingest_vec,
                    _np.array([vec_label], dtype=_np.int64),
                )
                if self._ann_journal is not None:
                    self._ann_journal.append(("add", _ingest_vec, vec_label, uuid_str))
                self._label_map[uuid_str] = vec_label
                self._save_index_atomic()

            try:
                npy_path.unlink()
                json_path.unlink()
            except OSError as exc:
                _log.warning("ingest_pending_embeddings: cleanup failed for %s: %s", npy_path, exc)

            ingested += 1
        return ingested

    def _ann_add_flipped_row(self, rid: str, vec) -> None:
        """Add one pending→active flip to the live ANN buffer in place.

        Same discipline as the insert path: the label is the row's rowid, the
        add is journaled when a rebuild is between snapshot and swap, and the
        label map is updated under ``_hnsw_lock`` (``_conn_lock`` nested
        inside, matching the reuse-path lock order). Re-adding an existing
        label is hnsw's update-in-place, so a re-flip is safe.
        """
        import numpy as _np

        with self._hnsw_lock:
            with self._conn_lock:
                row = self._conn.execute(
                    "SELECT rowid FROM records WHERE id = ?", (rid,)
                ).fetchone()
            if row is None:
                return
            vec_label = int(row[0])
            emb_vec = _np.array(vec, dtype=_np.float32).reshape(1, -1)
            self._hnsw.add_items(emb_vec, _np.array([vec_label], dtype=_np.int64))
            if self._ann_journal is not None:
                self._ann_journal.append(("add", emb_vec, vec_label, rid))
            self._label_map[rid] = vec_label
            self._write_counter += 1
            self._maybe_resize()

    def pending_embeddings_wake_sequence(self, embedder: Any | None = None) -> dict:
        has_pending = self.has_pending_rows()
        sidecar_dir = self._store_root / ".pending-embeddings"
        has_sidecars = sidecar_dir.exists() and any(sidecar_dir.glob("*.npy"))
        from iai_mcp.hippo import HippoIntegrityError
        with self._conn_lock:
            non_pending_row = self._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        if non_pending_row is None:
            # A COUNT(*) must return a row; None means the shared cursor was
            # reset mid-fetch. Coercing to 0 would fake a mismatch and drive a
            # needless rebuild (or mask a real one) — fail loud instead.
            raise HippoIntegrityError(
                "pending_embeddings_wake_sequence: COUNT(*) returned no row"
            )
        non_pending_count = non_pending_row[0]
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

        # Flips land in the ANN incrementally (per-row add above), so a clean
        # pass needs NO full rebuild — an unconditional rebuild here re-added
        # the whole corpus for a handful of flipped rows on every ambient
        # drain. The full rebuild remains the self-heal for a pre-existing
        # index/corpus mismatch and for the sidecar-ingest path (which does
        # not add incrementally).
        if has_mismatch or ingest_count > 0:
            rebuild_result = self._rebuild_index_from_sqlite()
        else:
            rebuild_result = {"action": "skip", "reason": "incremental_adds"}

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
        # This runs from the off-lock decrypt loop, so the write MUST take
        # _conn_lock + _txn like every other mutator — a bare execute on the
        # shared source-of-truth connection would interleave with a concurrent
        # writer's open transaction and corrupt its cursor/commit. Telemetry
        # stays best-effort (swallowed), but never at the cost of the writer.
        try:
            with self._conn_lock, _txn(self._conn):
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
            _DDL_RECORD_TAGS,
            _DDL_RECORD_TAGS_INDEXES,
        )
        conn = self._conn
        with self._conn_lock, _txn(conn):
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
                    ("role", "TEXT"),
                    ("live", "INTEGER"),
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

            conn.execute(_DDL_RECORD_TAGS)
            for idx in _DDL_RECORD_TAGS_INDEXES:
                conn.execute(idx)

        self._heal_null_vec_labels()

    def _heal_null_vec_labels(self) -> None:
        """Backfill NULL ``vec_label`` from the rowid. A staging table built
        without the ``INTEGER PRIMARY KEY`` alias (a past reembed migration
        did exactly that) leaves every post-swap insert with a NULL
        vec_label; the first label-map load after restart then dies on
        ``int(None)`` and launchd respawns the daemon into a crash loop.
        ``lastrowid == rowid`` for those inserts, so the rowid IS the label
        the ANN index and the in-memory map were using."""
        with self._conn_lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM records WHERE vec_label IS NULL"
            ).fetchone()
        n_null = int(row[0]) if row is not None else 0
        if n_null == 0:
            return
        _log.warning(
            "healing %d records rows with NULL vec_label (backfill from rowid)",
            n_null,
        )
        with self._conn_lock, _txn(self._conn):
            self._conn.execute(
                "UPDATE records SET vec_label = rowid WHERE vec_label IS NULL"
            )

    def _reconcile_columns(
        self, table_name: str, expected: list[tuple[str, str]]
    ) -> None:
        plain_to_pa = {
            "TEXT": _schema.string(),
            "INTEGER": _schema.int64(),
            "REAL": _schema.float64(),
            "BLOB": _schema.binary(),
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
        missing_plain: list[_schema.Field] = []
        missing_with_default: list[tuple[str, str]] = []
        for col_name, sqlite_type in expected:
            if col_name in existing:
                continue
            if sqlite_type in plain_to_pa:
                missing_plain.append(_schema.field(col_name, plain_to_pa[sqlite_type]))
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
        schema: _schema.Schema | None = None,
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
                # Serialize DDL under _conn_lock + _txn: a bare BEGIN on the
                # shared connection while another thread holds an open txn would
                # error ("transaction within a transaction") or commit that
                # thread's work, and bypasses the double-writer tripwire.
                with self._conn_lock, _txn(self._conn):
                    self._conn.execute(ddl)
        return HippoTable(self._conn, name, embed_dim=self._embed_dim, db=self)

    def drop_table(self, name: str) -> None:
        _validate_table_name(name)
        with self._conn_lock, _txn(self._conn):
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
        _pool = getattr(self, "_ro_pool", None)
        if _pool is not None:
            try:
                _pool.close()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(self, "_hnsw"):
            try:
                with self._hnsw_lock:
                    self._save_index_atomic()
            except Exception:  # noqa: BLE001
                pass
        # A borrowed (non-owned) engine connection is shared with the first
        # opener and is closed by that owner, not here — committing or closing
        # it would tear the owner's live handle. Only the owner commits + closes.
        owns_conn = getattr(self, "_owns_conn", True)
        if owns_conn:
            try:
                self._conn.commit()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            if getattr(self, "_storage_driver", "stdlib") == "lilli":
                try:
                    from iai_mcp.lillibrain.connection import (  # noqa: PLC0415
                        deregister_lilli_conn,
                    )

                    _conn_for_dereg = getattr(self, "_conn", None)
                    # The registry stores the raw engine connection; unwrap the
                    # signalling proxy or the identity check never matches.
                    _conn_for_dereg = getattr(_conn_for_dereg, "_wrapped", _conn_for_dereg)
                    deregister_lilli_conn(self._db_path, _conn_for_dereg)
                except Exception:  # noqa: BLE001
                    pass
        elif getattr(self, "_storage_driver", "stdlib") == "lilli":
            # A borrowed (non-owned) lilli handle is a distinct shared holder of
            # the backing store. Close it deterministically rather than waiting for
            # garbage collection: the engine release is refcount-gated, so this
            # frees the file/lock only when this borrow is the last live holder —
            # while any other holder remains, it is a safe flush-only no-op on the
            # release side. This restores last-reference release at close() time
            # for the borrow case (no GC dependence) without risking a premature
            # release under a live holder.
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
                            fcntl.flock(base_fd, fcntl.LOCK_UN)
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
