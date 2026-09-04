from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable
from uuid import UUID

import logging

from iai_mcp.hippo import _REAL_IAI_ROOT, AccessMode, HippoDB, HippoIntegrityError

from iai_mcp.hippo import _schema

from iai_mcp.crypto import (
    CryptoKey,
    decrypt_field,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import (
    EPISTEMIC_STATUS_ENUM,
    HV_TIER_ENUM,
    SALIENCE_LEVEL_ENUM,
    SALIENCE_LEVEL_RANK,
    SCHEMA_VERSION_CURRENT,
    MemoryRecord,
    TIER_ENUM,
)

from iai_mcp.store import (
    RECORDS_TABLE, EDGES_TABLE, EDGE_TYPES, _STC_TIER_ORDER, GateAction, GatePayload, _resolve_embed_dim, _PendingTurn,
)
from iai_mcp.store._buffers import (
    _record_buffer, _record_last_flush_at,
    _edge_buffer, _edge_last_flush_at,
    flush_record_buffer, should_flush_record_buffer,
    flush_edge_buffer, should_flush_edge_buffer,
    reset_store_buffers,
)

logger = logging.getLogger(__name__)

# boost_edges' cheap per-pair branch vs its full edge_type-table scan
# branch. Every write site that chunks pairs into boost_edges must stay
# at or under this size, or it falls onto the full-table scan.
BOOST_EDGES_SMALL_BATCH = 4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _slow_ann_log_threshold_ms() -> float:
    try:
        return float(os.environ.get("IAI_MCP_SLOW_ANN_LOG_MS", "500"))
    except ValueError:
        return 500.0


def _recency_buffer_maxlen() -> int:
    """Return the recency buffer capacity from the environment (default 200).

    200 = 4 × the worst-case consumer over-fetch (50 hits × 4).
    Override with ``IAI_MCP_RECENCY_BUFFER_MAXLEN`` for testing.
    """
    import os as _os
    raw = _os.environ.get("IAI_MCP_RECENCY_BUFFER_MAXLEN", "")
    try:
        v = int(raw)
        return v if v > 0 else 200
    except (TypeError, ValueError):
        return 200


_UUID_HYPHEN_POSITIONS = (8, 13, 18, 23)
_UUID_HEX = frozenset("0123456789abcdefABCDEF")


def _is_canonical_uuid_str(s: str) -> bool:
    """Return True iff *s* is a canonical 36-char hyphenated UUID string.

    Equivalent to ``UUID(s)`` succeeding for the canonical form that
    ``_uuid_literal`` writes for every ``edges.src``/``edges.dst`` value, but
    without constructing a UUID object — used on the hot degree-build path
    where only the string key is needed. Both this predicate and the
    UUID-object path reject a malformed key identically.
    """
    if len(s) != 36:
        return False
    for i, ch in enumerate(s):
        if i in _UUID_HYPHEN_POSITIONS:
            if ch != "-":
                return False
        elif ch not in _UUID_HEX:
            return False
    return True


def _uuid_literal(value):
    # late import so the package attribute is re-fetched per call and monkeypatches stay visible
    from iai_mcp.store import _uuid_literal as _impl
    return _impl(value)


def _normalize_ts_for_compare(value) -> str:
    """Return a canonical UTC ISO string for SQL TEXT comparison."""
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if hasattr(value, "to_pydatetime"):
        dt = value.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    try:
        dt = datetime.fromisoformat(
            str(value).replace("Z", "+00:00").replace(" ", "T")
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"must be ISO-8601, got {value!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _derive_role(tags: list[str] | None) -> str | None:
    """Return the conversational role value from a tags list.

    Scans the list for the first tag with a ``role:`` prefix and returns
    the verbatim suffix (e.g. "user", "assistant", "tool"). Returns None
    when no role tag is present or the input is empty/None.

    This is the single source of truth for role derivation — used by
    _to_row, insert_pending_row, and the backfill migrator so that
    write-time and read-time derivation are always identical.
    """
    for t in (tags or []):
        if isinstance(t, str) and t.startswith("role:"):
            return t[len("role:"):]
    return None


def _derive_live(tombstoned_at: "str | None") -> int:
    """Return the denormalized live-flag value for a ``tombstoned_at`` cell.

    ``1`` when the record is live (``tombstoned_at`` is unset), ``0`` when it
    carries a tombstone timestamp. This is the single source of truth for
    live-flag derivation, kept in permanent agreement with the
    ``tombstoned_at IS NULL`` predicate it denormalizes — every write path
    that sets ``tombstoned_at`` MUST derive ``live`` from the same value in
    the same write.
    """
    return 0 if tombstoned_at else 1


def _parse_ts_field(val: Any) -> "datetime | None":
    """Shared by _from_row and _from_row_rank_view — must stay one source
    of truth so the two decode tiers can never disagree on a timestamp."""
    import pandas as pd
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(val, datetime):
        return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
    if hasattr(val, "to_pydatetime"):
        dt = val.to_pydatetime()
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


@dataclass
class RankCandidateView:
    """Cheap rank-only candidate decode — literal_surface is the only
    AES-decrypted field; every other field is a plaintext column. Mutable,
    but the recall scoring loop no longer writes community_id or
    profile_modulation_gain onto instances sitting in records_cache -- those
    per-call values are tracked in call-local dicts and read back at hit
    construction instead, so a records_cache value that IS cache-shared
    across calls (SimpleRecordView) is never field-mutated."""

    id: UUID
    embedding: "list[float] | None"
    literal_surface: str
    aaak_index: str = ""
    created_at: datetime = field(default_factory=_utc_now)
    stability: float = 0.0
    tier: str = "episodic"
    tags: list = field(default_factory=list)
    language: str = "en"
    community_id: "UUID | None" = None
    structure_hv: bytes = b""
    salience_level: str = "unflagged"
    provenance: list = field(default_factory=list)
    profile_modulation_gain: dict = field(default_factory=dict)
    valence: float = 0.0
    directive: bool = False


class MemoryStore:

    def __init__(
        self,
        path: Path | str | None = None,
        user_id: str = "default",
        read_consistency_interval: timedelta | None = None,
        *,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
        persist_index: bool = True,
    ) -> None:
        # late import so the package attribute is re-fetched per call and monkeypatches stay visible
        from iai_mcp.store import DEFAULT_STORAGE_PATH as _DSP
        env_path = os.environ.get("IAI_MCP_STORE")
        if path is not None:
            self.root = Path(path)
        elif env_path:
            self.root = Path(env_path)
        else:
            self.root = Path(_DSP)
        if os.environ.get("PYTEST_CURRENT_TEST") and self.root == _REAL_IAI_ROOT:
            raise RuntimeError(
                "hermeticity guard: store-root resolved to the real home store "
                "during a test run; tests must use a tmp path (autouse redirect "
                "fixture). This guard never fires in normal operation."
            )
        self.root.mkdir(parents=True, exist_ok=True)
        # id(self) may equal a collected store's id — inherited buffer rows
        # would poison this store's writes.
        from iai_mcp.store._buffers import reset_store_buffers
        reset_store_buffers(id(self))
        self._read_consistency_interval: timedelta | None = read_consistency_interval
        self._user_id: str = user_id
        self._crypto_key_wrapper: CryptoKey = CryptoKey(user_id=user_id, store_root=self.root)
        self._crypto_key: bytes | None = None
        import weakref
        _weak_key = weakref.WeakMethod(self._key)
        def _key_via_weakref() -> bytes:
            fn = _weak_key()
            if fn is None:
                raise RuntimeError("MemoryStore already collected")
            return fn()
        self.db: HippoDB = HippoDB(
            self.root,
            crypto_key_provider=_key_via_weakref,
            access_mode=access_mode,
            read_only=read_only,
            persist_index=persist_index,
        )
        self._embed_dim: int = _resolve_embed_dim()
        # Assumed healthy until _run_boot_migrations proves otherwise (or
        # fails outright before reaching the assignment) — find_record_by_tag
        # only falls back to the LIKE-scan when this is explicitly False.
        self._record_tags_backfill_ok: bool = True
        self._ensure_tables()
        self._run_boot_migrations()
        self._graph_sync_hook: Callable[[str, "MemoryRecord"], None] | None = None
        # In-process recency buffer: bounded, volatile, RAM-only.
        # Constructed eagerly alongside the other write-path organs so the
        # reconcile callback (registered below) is live before any lifecycle
        # tick can fire a reembed.
        from iai_mcp.store._recency_buffer import RecencyBuffer as _RecencyBuffer
        self._recency_buffer = _RecencyBuffer(maxlen=_recency_buffer_maxlen())
        # Register the reembed-reconcile callback eagerly in the constructor —
        # before any wake tick or pending_embeddings_wake_sequence can flip
        # embedding_pending 1→0.  Lazy registration (e.g. inside
        # warm_recency_buffer) would leave the callback None during the window
        # between store construction and the first warm call, causing silent
        # divergence when a reembed fires before warm.
        # Bind the callbacks through weak references so HippoDB never holds a
        # strong reference back to this store. A bound method (or a closure
        # capturing ``self``) stored on ``self.db`` would form a store<->db
        # reference cycle that only the cyclic GC can break — leaving the
        # underlying file lock held until a non-deterministic gc pass runs,
        # which surfaces as spurious HippoLockHeldError when a short-lived store
        # is dropped and the path is reopened. Mirrors the crypto-key provider
        # above, which uses the same WeakMethod discipline for the same reason.
        _weak_reconcile = weakref.ref(self._recency_buffer)

        def _recency_reconcile_via_weakref(rid) -> None:
            buf = _weak_reconcile()
            if buf is not None:
                buf.reconcile_reembed(str(rid))

        self.db.register_recency_reconcile(_recency_reconcile_via_weakref)
        # Register the pending-row feed callback eagerly so insert_pending_row
        # on the underlying HippoDB always feeds the buffer, even when called
        # via store.db.insert_pending_row() directly (e.g. from tests or the
        # daemon's own pending capture path).
        _weak_feed = weakref.WeakMethod(self._feed_recency_pending)

        def _feed_recency_pending_via_weakref(**kwargs) -> None:
            fn = _weak_feed()
            if fn is not None:
                fn(**kwargs)

        self.db.register_recency_pending_feed(_feed_recency_pending_via_weakref)
        # Single-flight warm lock: serializes the lazy warm-on-first-read so only
        # ONE thread rebuilds the buffer while concurrent cold readers wait, then
        # observe a fully-warm buffer.  Prevents the TOCTOU where two cold recalls
        # both warm and the second empties the buffer the first is about to read.
        self._recency_warm_lock = threading.Lock()
        self._write_queue = None  # type: ignore[assignment]
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self._async_conn = None
        self._provenance_queue = None  # type: ignore[assignment]
        self._reinforce_queue = None  # type: ignore[assignment]
        # Per-operation COUNT memo. None when inactive. When a `count_memo()`
        # scope is open it holds the corpus-count results for the duration of one
        # logical operation (e.g. a single runtime-graph build) so the same
        # full-corpus COUNT(*) is not re-scanned a dozen times within that
        # operation. The scope is opened around a self-contained read sequence and
        # torn down on exit, so a memoized count can never outlive the operation
        # that opened it or be served to an unrelated caller.
        self._count_memo: dict[str, int] | None = None
        # Cross-read corpus-count cache: holds the three corpus COUNT values
        # (active/pending/edges) between writes so a read-only recall burst pays
        # one live SQL COUNT per count on the first call, then serves O(1) cached
        # reads for the rest of the burst.  Invalidated on every count-changing
        # write (the write path wires invalidation).  Constructed eagerly — before
        # any lifecycle tick — so the invalidation callbacks the write path
        # registers always find a live cache instance.
        from iai_mcp.store._corpus_count_cache import CorpusCountCache as _CorpusCountCache
        self._corpus_count_cache = _CorpusCountCache()
        # Wire the cache's on_invalidate hook to the RO pool's staleness bump.
        # This single wire covers the whole union of committed corpus-changing
        # writes: buffered record/edge flush (_buffers.py invalidates the cache
        # directly), async-drain, delete, sleep tombstones/optimize/erasure,
        # dedupe, and the db-fired insert_pending_row/reembed_pending_rows
        # callbacks — every one of them ends at cache.invalidate()/clear().
        # getattr-defensive: a db without mark_ro_pool_stale (stdlib driver,
        # or an older/mocked db in a test) simply never gets a hook wired.
        _mark_stale = getattr(self.db, "mark_ro_pool_stale", None)
        if _mark_stale is not None:
            self._corpus_count_cache.on_invalidate = _mark_stale
        # Register the corpus-count invalidation callback eagerly so every
        # count-changing write path that fires through HippoDB (insert_pending_row,
        # reembed_pending_rows) invalidates the correct cache keys immediately.
        # Registered after the cache is constructed so the callback always finds
        # a live cache instance.
        _weak_invalidate = weakref.WeakMethod(self._invalidate_corpus_count)

        def _invalidate_corpus_count_via_weakref(*keys: str) -> None:
            fn = _weak_invalidate()
            if fn is not None:
                fn(*keys)

        self.db.register_corpus_count_invalidate(_invalidate_corpus_count_via_weakref)

        _weak_adjust = weakref.WeakMethod(self._adjust_corpus_count)

        def _adjust_corpus_count_via_weakref(key: str, delta: int) -> None:
            fn = _weak_adjust()
            if fn is not None:
                fn(key, delta)

        _register_adjust = getattr(self.db, "register_corpus_count_adjust", None)
        if _register_adjust is not None:
            _register_adjust(_adjust_corpus_count_via_weakref)

        # Resident exact-cosine authority: the object is constructed eagerly so
        # the write seams below always have somewhere to feed, but the matrix
        # itself stays cold until the first exact_top_k call (lazy build keeps
        # the boot budget untouched — no scan runs at store open).
        from iai_mcp.store._exact_index import ExactCosineIndex as _ExactCosineIndex
        self._exact_index = _ExactCosineIndex(self._embed_dim)
        # Build-failure cooldown: a monotonic deadline before which
        # exact_top_k skips the full-corpus SELECT + build retry entirely
        # (see exact_top_k / invalidate_exact_index).
        self._exact_index_build_cooldown_until = 0.0
        # Single-flight guard for the cold-matrix build: concurrent callers
        # (a warm-up probe racing a first recall) must never run two
        # whole-corpus builds of the same matrix side by side.
        self._exact_index_build_lock = threading.Lock()
        # Rate-limit gate for the cue-coercion telemetry emit (recall is a
        # hot path; an upstream NaN-cue source must not emit every call).
        self._last_cue_nonfinite_emit_at = 0.0
        # Register the reembed-flip feed callback eagerly, mirroring the
        # recency-buffer and count-cache registrations above: bound through a
        # weakref so HippoDB never holds a strong reference back to this store.
        _weak_exact_feed = weakref.WeakMethod(self._feed_exact_index)

        def _feed_exact_index_via_weakref(record_id: str, vec) -> None:
            fn = _weak_exact_feed()
            if fn is not None:
                fn(record_id, vec)

        self.db.register_exact_index_feed(_feed_exact_index_via_weakref)

    def _invalidate_corpus_count(self, *keys: str) -> None:
        """Forward a cache-key invalidation to the corpus-count cache.

        Every count-changing write path routes its invalidation through here
        so there is exactly one place that touches the cache from the write
        side.  Best-effort: a missing or broken cache never propagates an
        exception to the caller.
        """
        try:
            self._corpus_count_cache.invalidate(*keys)
        except Exception:  # noqa: BLE001
            pass

    def _adjust_corpus_count(self, key: str, delta: int) -> None:
        """Shift one cached corpus count by a known-exact write delta.

        The exact-delta twin of ``_invalidate_corpus_count``: hot write paths
        that know precisely how many rows they changed keep the cached value
        live instead of forcing the next reader into a full filtered COUNT.
        Best-effort, same as the invalidation funnel.
        """
        try:
            self._corpus_count_cache.adjust(key, delta)
        except Exception:  # noqa: BLE001
            pass

    @contextmanager
    def count_memo(self):
        """Memoize corpus COUNT(*) results for the duration of one operation.

        Within the scope, repeated calls to `active_records_count` (and the
        runtime-graph cache key's corpus counts) reuse the first result instead
        of re-scanning the table. A runtime-graph build probes the same counts
        ~a dozen times (cache-key derivation runs per cache read, and the drift
        gate plus the impl each ask again); on the lilli engine each scan reads
        every leaf page, so the redundancy dominates a warm boot. The scope makes
        those reads collapse to one per distinct count.

        The memo is strictly operation-scoped: it is cleared on exit and nested
        scopes reuse the outermost memo, so no count is ever cached across a write
        or served to an unrelated read path.
        """
        if self._count_memo is not None:
            # Already inside a scope — reuse the outer memo, do not reset it.
            yield
            return
        self._count_memo = {}
        try:
            yield
        finally:
            self._count_memo = None

    def close(self) -> None:
        if self.db is None:
            return

        from iai_mcp.events import _BUFFER_LOCK, flush_event_buffer

        with _BUFFER_LOCK:
            _log = logging.getLogger(__name__)
            try:
                flush_event_buffer(self)
            except Exception as exc:  # noqa: BLE001 -- drain MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "flush_event_buffer",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )
            try:
                flush_record_buffer(self)
            except Exception as exc:  # noqa: BLE001 -- drain MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "flush_record_buffer",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )
            try:
                flush_edge_buffer(self)
            except Exception as exc:  # noqa: BLE001 -- drain MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "flush_edge_buffer",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )

            try:
                from iai_mcp.retrieve import _tv_cache, _tv_cache_dirty
                # Weak-keyed on the store object (evicts with it); explicit
                # pop here just frees the maps at close instead of at GC.
                _tv_cache.pop(self, None)
                _tv_cache_dirty.pop(self, None)
            except (ImportError, TypeError):
                pass

            self._drain_async_writes_on_close()

            self.db.close()
            self.db = None

        # Outside _BUFFER_LOCK: reset_store_buffers dispatches to every
        # registered id(store)-keyed family. Lock position here cannot close
        # a write-during-close race either way: write_event's buffered path
        # (events.py) appends to _event_buffer without ever taking
        # _BUFFER_LOCK, so a concurrent buffered write on a store already
        # inside close() (db already None) is caller misuse, not something
        # this lock scope can prevent.
        reset_store_buffers(id(self))

    def _drain_async_writes_on_close(self) -> None:
        """Drain and tear down a live async write queue before closing the store.

        A store with async writes enabled owns a background event loop, thread,
        and coalesce task.  Closing without draining leaks all three into the
        process (a pending coalesce task plus a live daemon thread).  This stops
        the queue, stops the loop, joins the thread, and tears down the
        provenance queue too, then nulls the async handles so the store is left
        in a clean state.

        It is a strict no-op when no queue is live, so it is production-safe for
        the common case where async writes were never enabled.  The stop is
        inlined (rather than calling the async ``disable_async_writes``) because
        ``close()`` is synchronous and cannot await; it mirrors that teardown.
        Every step is bounded so a wedged loop cannot hang ``close()``, and a
        failure is logged-and-swallowed so a best-effort teardown never makes
        ``close()`` raise; the handles are nulled in a ``finally`` regardless.
        """
        _log = logging.getLogger(__name__)
        try:
            self.disable_provenance_queue()
        except Exception as exc:  # noqa: BLE001 -- teardown MUST NOT block close()
            _log.warning(
                "memorystore_close_drain_failed",
                extra={
                    "flush": "disable_provenance_queue",
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:120],
                },
            )
        try:
            self.disable_reinforce_queue()
        except Exception as exc:  # noqa: BLE001 -- teardown MUST NOT block close()
            _log.warning(
                "memorystore_close_drain_failed",
                extra={
                    "flush": "disable_reinforce_queue",
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:120],
                },
            )

        if self._write_queue is None:
            return

        bg_loop = self._async_loop
        queue = self._write_queue
        try:
            if bg_loop is not None:
                asyncio.run_coroutine_threadsafe(
                    queue.stop(), bg_loop
                ).result(timeout=5.0)
        except Exception as exc:  # noqa: BLE001 -- teardown MUST NOT block close()
            _log.warning(
                "memorystore_close_drain_failed",
                extra={
                    "flush": "async_write_queue_stop",
                    "err_type": type(exc).__name__,
                    "err": str(exc)[:120],
                },
            )
        finally:
            try:
                if bg_loop is not None:
                    bg_loop.call_soon_threadsafe(bg_loop.stop)
                if self._async_thread is not None:
                    self._async_thread.join(timeout=5.0)
            except Exception as exc:  # noqa: BLE001 -- teardown MUST NOT block close()
                _log.warning(
                    "memorystore_close_drain_failed",
                    extra={
                        "flush": "async_loop_thread_stop",
                        "err_type": type(exc).__name__,
                        "err": str(exc)[:120],
                    },
                )
            finally:
                self._write_queue = None
                self._async_loop = None
                self._async_thread = None
                self._async_conn = None

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


    def _ensure_tables(self) -> None:
        try:
            tbl = self.db.open_table(RECORDS_TABLE)
            arrow_schema = tbl.schema
            emb_field = arrow_schema.field("embedding")
            actual_dim = getattr(emb_field.type, "list_size", None)
            if actual_dim and int(actual_dim) > 0:
                self._embed_dim = int(actual_dim)
        except (OSError, KeyError, ValueError, AttributeError) as exc:
            logger.debug("records table schema introspection skipped: %s", exc)

    def _run_boot_migrations(self) -> None:
        """Run lightweight idempotent migrations that backfill derived columns.

        Called once per store open, after ``_ensure_tables``. Each migration is
        gated by a persisted ``_hippo_meta`` stamp written once the backfill has
        run to completion — so the overhead on a post-migration open is a single
        keyed meta lookup (O(1)), never a scan of ``records``. The first open of
        a store written before the derived column existed pays the one-time
        backfill, then sets the stamp.
        """
        from iai_mcp.migrate._role_column import migrate_role_column
        try:
            migrate_role_column(self)
        except Exception as exc:  # noqa: BLE001 — boot migration must not abort open
            logger.warning("boot migration migrate_role_column failed (non-fatal): %s", exc)

        from iai_mcp.migrate._live_flag_backfill import migrate_live_flag_backfill
        try:
            migrate_live_flag_backfill(self)
        except Exception as exc:  # noqa: BLE001 — boot migration must not abort open
            logger.warning(
                "boot migration migrate_live_flag_backfill failed (non-fatal): %s", exc
            )

        from iai_mcp.migrate._live_flag_backfill import reconcile_live_flag_drift
        try:
            reconcile_live_flag_drift(self)
        except Exception as exc:  # noqa: BLE001 — boot migration must not abort open
            logger.warning(
                "boot migration reconcile_live_flag_drift failed (non-fatal): %s", exc
            )

        from iai_mcp.migrate._record_tags_backfill import migrate_record_tags_backfill
        try:
            result = migrate_record_tags_backfill(self)
            self._record_tags_backfill_ok = bool(result.get("ok", False))
        except Exception as exc:  # noqa: BLE001 — boot migration must not abort open
            logger.warning(
                "boot migration migrate_record_tags_backfill failed (non-fatal): %s", exc
            )
            self._record_tags_backfill_ok = False

    def _table_names(self) -> list[str]:
        result = self.db.list_tables()
        if hasattr(result, "tables"):
            return list(result.tables)
        return list(result)

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def user_id(self) -> str:
        return self._user_id


    def _key(self) -> bytes:
        if self._crypto_key is None:
            self._crypto_key = self._crypto_key_wrapper.get_or_create()
        return self._crypto_key

    def _ad(self, record_id: UUID | str) -> bytes:
        return _uuid_literal(record_id).encode("ascii")

    def _encrypt_for_record(self, record_id: UUID, value: str) -> str:
        if is_encrypted(value):
            return value
        return encrypt_field(value, self._key(), associated_data=self._ad(record_id))

    def _decrypt_for_record(self, record_id: UUID, value: str) -> str:
        if not is_encrypted(value):
            return value
        try:
            return decrypt_field(
                value, self._key(), associated_data=self._ad(record_id)
            )
        except ValueError as exc:
            raise HippoIntegrityError(
                f"records decrypt failed for id={record_id}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 -- AEAD failures carry no message
            from cryptography.exceptions import InvalidTag

            if not isinstance(exc, InvalidTag):
                raise
            # A record under a different key generation (e.g. after a partial
            # rotation) must surface as a named, actionable error — never a
            # bare empty-message InvalidTag traceback.
            raise HippoIntegrityError(
                f"records decrypt failed for id={record_id}: InvalidTag "
                "(wrong key generation? run `iai-mcp crypto recover-prior-key` "
                "with the retained .crypto.key.pre-rotate file)"
            ) from exc

    def _decrypt_for_record_or_none(
        self, record_id: UUID, value: str
    ) -> str | None:
        """None when the ciphertext opens under no available key — redaction
        and recovery paths classify such rows; they must never die on them,
        whatever exception taxonomy _decrypt_for_record adopts."""
        try:
            return self._decrypt_for_record(record_id, value)
        except HippoIntegrityError:
            return None

    def register_graph_sync_hook(
        self, hook: Callable[[str, MemoryRecord], None] | None
    ) -> None:
        self._graph_sync_hook = hook

    def _fire_graph_sync_hook(self, op: str, record: MemoryRecord) -> None:
        hook = self._graph_sync_hook
        if hook is None:
            return
        try:
            hook(op, record)
        except Exception as exc:  # noqa: BLE001 -- hook isolation, daemon stability
            logger.warning("graph_sync_hook failed op=%s: %s", op, exc, exc_info=True)
            try:
                sys.stderr.write(
                    json.dumps({
                        "event": "graph_sync_failed",
                        "op": op,
                        "record_id": str(getattr(record, "id", "")),
                        "error": str(exc),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    })
                    + "\n"
                )
            except Exception:  # noqa: BLE001 -- logger-of-logger recursion guard
                pass

    # ------------------------------------------------------------------
    # Recency buffer — feed, pending-feed, boot-rebuild
    # ------------------------------------------------------------------

    def _feed_recency(self, record: "MemoryRecord") -> None:
        """Push a MemoryRecord into the recency buffer if it is buffer-relevant.

        Buffer-relevant means: role='user' OR embedding_pending=1.  All other
        records (assistant, no-role, non-pending) are skipped so the capacity
        budget is spent on the served set.

        Resolves ``source_rowid`` with a quick ``SELECT rowid FROM records
        WHERE id = ? LIMIT 1``; falls back to -1 on failure (the boot rebuild
        will re-key the entry from the true rowid).

        Failures are logged-and-swallowed so a buffer feed error never breaks
        a write or crashes the daemon.
        """
        try:
            from iai_mcp.store._recency_buffer import RecencyMarker as _RecencyMarker
            role = _derive_role(record.tags)
            ep = int(getattr(record, "embedding_pending", 0) or 0)
            if role != "user" and ep != 1:
                return
            session_id: str | None = None
            if record.provenance:
                try:
                    session_id = record.provenance[0].get("session_id")
                except Exception:  # noqa: BLE001
                    pass
            # Resolve the rowid of the just-written row by id.
            source_rowid = -1
            try:
                with self.db._conn_lock:
                    row = self.db._conn.execute(
                        "SELECT rowid FROM records WHERE id = ? LIMIT 1",
                        (str(record.id),),
                    ).fetchone()
                if row is not None:
                    source_rowid = int(row[0])
            except Exception as exc:  # noqa: BLE001
                logger.debug("recency_feed rowid lookup failed id=%s: %s", record.id, exc)
            from datetime import timezone as _tz_feed
            created_at = record.created_at
            if created_at is not None and created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=_tz_feed.utc)
            marker = _RecencyMarker(
                id=str(record.id),
                literal_surface=record.literal_surface,
                created_at=created_at,
                session_id=session_id,
                embedding_pending=ep,
                role=role,
                source_rowid=source_rowid,
                tier=getattr(record, "tier", "episodic") or "episodic",
            )
            self._recency_buffer.push(marker)
        except Exception as exc:  # noqa: BLE001 -- hook isolation
            logger.debug("recency_feed failed: %s", exc)

    def _feed_working(self, record: "MemoryRecord") -> None:
        """Feed a MemoryRecord into the working tier's per-turn update.

        Mirrors _feed_recency's isolation contract exactly: a failure here
        must never crash or block a write.
        """
        try:
            from iai_mcp import working_tier
            working_tier.update_from_record(record, store=self)
        except Exception as exc:  # noqa: BLE001 -- hook isolation
            logger.debug("working_feed failed: %s", exc)

    def _feed_lexical(self, record: "MemoryRecord", *, restamp: bool = False) -> None:
        """Keep an already-built lexical index current across writes —
        appending one document is cheap, so recall's warm lane never goes
        stale between full builds. A never-built index stays unbuilt.
        The feed happens where plaintext exists (the in-memory record);
        the generation restamp happens where the row lands (buffer flush,
        or here for direct writes with restamp=True)."""
        try:
            idx = getattr(self, "_lexical_idx", None)
            if idx is None or idx.generation is None:
                return
            idx.add_document(str(record.id), record.literal_surface or "")
            if restamp:
                try:
                    gen = self._corpus_count_cache.generation()
                except Exception:  # noqa: BLE001
                    gen = None
                idx.restamp(gen)
        except Exception as exc:  # noqa: BLE001 -- hook isolation
            logger.debug("lexical_feed failed: %s", exc)

    def _emit_store_watermark(self, record: "MemoryRecord") -> None:
        """Stamp the store-advance sidecar the per-turn hooks watch. It lives
        beside the DB file so readers need no knowledge of the store layout."""
        try:
            from iai_mcp.store_watermark import emit
            created = getattr(record, "created_at", None)
            sidecar_dir = getattr(self.db, "_hippo_dir", self.root / "hippo")
            emit(sidecar_dir, created.isoformat() if created is not None else "")
        except Exception as exc:  # noqa: BLE001 -- sidecar is advisory
            logger.debug("watermark emit failed: %s", exc)

    def _feed_recency_pending(
        self,
        *,
        record_id: str,
        literal_surface: str,
        tags_json: str,
        provenance_json: str,
        created_at: str,
        tier: str = "episodic",
    ) -> None:
        """Feed the buffer after an ``insert_pending_row`` write.

        ``literal_surface`` is the plaintext value as passed to
        ``insert_pending_row`` (before encryption at the HippoDB layer).  The
        buffer receives plaintext; it never decrypts or holds ciphertext.

        This is the dedicated pending-capture feed.  It is registered as a
        callback on the HippoDB so it fires for every caller of
        ``insert_pending_row``, including direct callers that bypass
        ``MemoryStore.insert_pending()``.

        Failures are logged-and-swallowed for hook isolation.
        """
        try:
            import json as _json
            from iai_mcp.store._recency_buffer import RecencyMarker as _RecencyMarker
            from datetime import timezone as _tz

            # Derive role from the tags JSON (mirrors insert_pending_row's own logic).
            role: str | None = None
            try:
                tags_list = _json.loads(tags_json) if tags_json else []
                for t in tags_list:
                    if isinstance(t, str) and t.startswith("role:"):
                        role = t[len("role:"):]
                        break
            except Exception:  # noqa: BLE001
                pass

            # Only role:user pending rows belong in the buffer.
            if role != "user":
                return

            # Parse created_at.
            ca = None
            try:
                ca = datetime.fromisoformat(
                    str(created_at).replace("Z", "+00:00").replace(" ", "T")
                )
                if ca.tzinfo is None:
                    ca = ca.replace(tzinfo=_tz.utc)
            except (TypeError, ValueError):
                pass

            # Parse session_id from provenance JSON.
            session_id: str | None = None
            try:
                prov = _json.loads(provenance_json) if provenance_json else []
                if prov:
                    session_id = prov[0].get("session_id")
            except Exception:  # noqa: BLE001
                pass

            # Resolve the rowid of the just-inserted pending row.
            source_rowid = -1
            try:
                with self.db._conn_lock:
                    row = self.db._conn.execute(
                        "SELECT rowid FROM records WHERE id = ? LIMIT 1",
                        (record_id,),
                    ).fetchone()
                if row is not None:
                    source_rowid = int(row[0])
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "recency_pending_feed rowid lookup failed id=%s: %s",
                    record_id, exc,
                )

            marker = _RecencyMarker(
                id=record_id,
                literal_surface=literal_surface,
                created_at=ca,
                session_id=session_id,
                embedding_pending=1,
                role=role,
                source_rowid=source_rowid,
                tier=tier or "episodic",
            )
            self._recency_buffer.push(marker)
        except Exception as exc:  # noqa: BLE001 -- hook isolation
            logger.debug("recency_pending_feed failed: %s", exc)

    def insert_pending(
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
        """Store-level wrapper for ``HippoDB.insert_pending_row``.

        Delegates to the underlying DB method then feeds the recency buffer.
        The buffer feed is also registered as a callback on HippoDB so all
        callers (including direct ``store.db.insert_pending_row(...)`` calls)
        receive the feed — this wrapper is kept for callers that prefer the
        store API surface.
        """
        self.db.insert_pending_row(
            record_id=record_id,
            tier=tier,
            literal_surface=literal_surface,
            tags_json=tags_json,
            provenance_json=provenance_json,
            created_at=created_at,
            updated_at=updated_at,
        )
        # The feed also fires via the HippoDB callback registered in __init__,
        # so no explicit feed call is needed here.  The callback is idempotent
        # (push-by-id updates in place), so a double-push is safe, but we
        # avoid it for efficiency.

    def warm_recency_buffer(self) -> None:
        """Rebuild the recency buffer from a bounded, embedding-free SQL query.

        Issues ``ORDER BY created_at DESC LIMIT maxlen`` against the live
        store connection, selects marker fields only (no embedding column),
        and fills the buffer with up to ``maxlen`` entries.

        Idempotent: clears then refills on every call.  Daemon-independent:
        reads from ``self.db._conn`` (the store's own connection), so it works
        on a directly-opened store without the daemon running.

        Re-keys every entry from the true SQL rowid, correcting any write-path
        entries that were fed with an unresolved (-1) rowid.

        The boot rebuild decodes NO embedding — it uses a projection that
        explicitly omits the embedding column.
        """
        from iai_mcp.store._recency_buffer import RecencyMarker as _RecencyMarker
        import json as _json
        from datetime import timezone as _tz

        maxlen = self._recency_buffer._maxlen

        # Embedding-free bounded recency query.
        # Includes rowid (the tie-break authority) and excludes the embedding
        # BLOB so the per-row decode is cheap (no frombuffer, no AES on a BLOB).
        # Push the EXACT eligibility predicate of the SQL authority into the
        # warm query BEFORE the LIMIT, so the top-maxlen rows are the top-maxlen
        # ELIGIBLE by created_at — not the top-maxlen of ALL rows (which would let
        # recent non-eligible turns, e.g. role:assistant, crowd out older eligible
        # role:user / pending markers the authority surfaces). The predicate is the
        # union of the authority's two branches:
        #   pending branch: embedding_pending = 1   (tier-agnostic)
        #   role branch:    role = 'user' AND tier = 'episodic'
        # Stays embedding-free (no embedding column) so the warm scan rides the
        # role / created_at index and never decodes a BLOB.
        _RECENCY_BOOT_SQL = (
            "SELECT"
            " rowid, id, literal_surface, provenance_json,"
            " created_at, embedding_pending, role, tier, tags_json"
            " FROM records"
            " WHERE tombstoned_at IS NULL"
            "   AND ("
            "        COALESCE(embedding_pending, 0) = 1"
            "        OR (role = 'user' AND tier = 'episodic')"
            "       )"
            " ORDER BY created_at DESC"
            " LIMIT ?"
        )

        with self.db._conn_lock:
            rows = self.db._conn.execute(_RECENCY_BOOT_SQL, (maxlen,)).fetchall()

        new_buffer_entries: list[_RecencyMarker] = []
        for raw in rows:
            row = dict(raw)
            row_id = row.get("id") or ""
            ep = int(row.get("embedding_pending") or 0)
            role_raw = row.get("role")
            role: str | None = str(role_raw) if role_raw is not None else None
            tier_raw = row.get("tier")
            tier_val = str(tier_raw) if tier_raw is not None else None
            # Mirror the SQL authority's eligibility exactly so the warm set is
            # byte-identical to _recent_pending_markers_sql:
            #   pending branch: embedding_pending = 1 (tier-agnostic, role-agnostic)
            #   role branch:    role = 'user' AND tier = 'episodic'
            # The role decision reads the role COLUMN (as the authority does); no
            # tags fallback here, because the authority's role predicate is the
            # column — a tags-only derivation would diverge from the authority.
            role_eligible = role == "user" and tier_val == "episodic"
            if ep != 1 and not role_eligible:
                continue

            # Decrypt literal_surface if encrypted.
            literal_raw = row.get("literal_surface") or ""
            try:
                from uuid import UUID as _UUID
                row_uuid = _UUID(row_id)
                if is_encrypted(literal_raw):
                    literal_raw = self._decrypt_for_record(row_uuid, literal_raw)
            except Exception as exc:  # noqa: BLE001
                logger.debug("warm_recency_buffer decrypt failed id=%s: %s", row_id, exc)

            # Parse session_id from provenance_json.
            session_id: str | None = None
            prov_raw = row.get("provenance_json") or "[]"
            try:
                from uuid import UUID as _UUID2
                if is_encrypted(prov_raw):
                    prov_raw = self._decrypt_for_record(_UUID2(row_id), prov_raw)
                prov = _json.loads(prov_raw)
                if prov:
                    session_id = prov[0].get("session_id")
            except Exception:  # noqa: BLE001
                pass

            # Parse created_at.
            ca = None
            ca_raw = row.get("created_at")
            try:
                ca = datetime.fromisoformat(
                    str(ca_raw).replace("Z", "+00:00").replace(" ", "T")
                )
                if ca.tzinfo is None:
                    ca = ca.replace(tzinfo=_tz.utc)
            except (TypeError, ValueError):
                pass

            source_rowid = int(row.get("rowid") or -1)

            new_buffer_entries.append(_RecencyMarker(
                id=row_id,
                literal_surface=literal_raw,
                created_at=ca,
                session_id=session_id,
                embedding_pending=ep,
                role=role,
                source_rowid=source_rowid,
                tier=tier_val or "episodic",
            ))

        # Merge (not replace) the SQL-warmed set under a single lock acquisition:
        # a cold-buffer warm must not wipe a pushed-but-unflushed marker the
        # write path already fed into the buffer (the SQL query above cannot
        # see it yet). merge_warm preserves every still-buffered sentinel
        # exempt from eviction; a same-id SQL row supersedes its sentinel.
        self._recency_buffer.merge_warm(new_buffer_entries)

    def insert(self, record: MemoryRecord) -> None:
        if record.tier not in TIER_ENUM:
            raise ValueError(f"invalid tier {record.tier!r}")
        if len(record.embedding) != self._embed_dim:
            raise ValueError(
                f"embedding must be {self._embed_dim}d, got {len(record.embedding)}"
            )
        if not record.structure_hv:
            try:
                from iai_mcp.tem import bind_structure
                record.structure_hv = bind_structure(record)
            except ImportError:
                pass

        from iai_mcp.daemon_config import _load_patsep_config
        from iai_mcp.events import TELEMETRY_EMBED_NONFINITE, write_event
        import numpy as _np_ins
        _emb_arr = _np_ins.asarray(record.embedding, dtype=_np_ins.float32)
        if not bool(_np_ins.all(_np_ins.isfinite(_emb_arr))):
            write_event(
                self,
                TELEMETRY_EMBED_NONFINITE,
                {
                    "record_id": str(record.id),
                    "nan_count": int(_np_ins.sum(_np_ins.isnan(_emb_arr))),
                    "inf_count": int(_np_ins.sum(_np_ins.isinf(_emb_arr))),
                },
                severity="error",
            )
            raise ValueError(
                f"embedding for record {record.id} contains non-finite values "
                f"(NaN or inf); a corrupt embedding must not be persisted"
            )
        _psep_cfg = _load_patsep_config()
        (
            _psep_action,
            _psep_payload,
            _psep_hits,
        ) = self._pattern_separation_gate_with_hits(record)
        _psep_near_dup_hit_id: str | None = None
        _psep_near_dup_cos: float | None = None
        _psep_edges_seeded = 0
        _psep_top_k_probed = len(_psep_hits)
        _psep_ann_prefilter_cos = getattr(record, "_psep_ann_prefilter_cos", None)
        _psep_exact_confirm_cos = getattr(record, "_psep_exact_confirm_cos", None)
        if _psep_action == GateAction.SKIP:
            _psep_near_dup_hit_id = str(_psep_payload)
            if _psep_hits:
                _psep_near_dup_cos = float(_psep_hits[0][1])
            if not _psep_cfg.dry_run:
                existing_id = (
                    _psep_payload if isinstance(_psep_payload, UUID)
                    else UUID(str(_psep_payload))
                )
                self.reinforce_record(existing_id)
                record.id = existing_id
                write_event(self, "pattern_separation_pass", {
                    "action": "skip",
                    "near_dup_hit_id": _psep_near_dup_hit_id,
                    "near_dup_cos": _psep_near_dup_cos,
                    "edges_seeded": 0,
                    "top_k_probed": _psep_top_k_probed,
                    "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                    "threshold_link": float(_psep_cfg.link_threshold),
                    "dry_run_mode": False,
                    "ann_prefilter_cos": _psep_ann_prefilter_cos,
                    "exact_confirm_cos": _psep_exact_confirm_cos,
                }, severity="info", buffered=True)
                return
            write_event(self, "pattern_separation_pass", {
                "action": "skip",
                "near_dup_hit_id": _psep_near_dup_hit_id,
                "near_dup_cos": _psep_near_dup_cos,
                "edges_seeded": 0,
                "top_k_probed": _psep_top_k_probed,
                "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                "threshold_link": float(_psep_cfg.link_threshold),
                "dry_run_mode": True,
                "ann_prefilter_cos": _psep_ann_prefilter_cos,
                "exact_confirm_cos": _psep_exact_confirm_cos,
            }, severity="info", buffered=True)

        if _psep_action == GateAction.INSERT:
            self._maybe_tag_schema_bypass(record)
            self._maybe_spatial_tag(record)

        if self._write_queue is not None and self._async_loop is not None:
            coro = self._write_queue.enqueue(record)
            submit = asyncio.run_coroutine_threadsafe(coro, self._async_loop)
            fut = submit.result()
            done_event = threading.Event()
            result_box: dict = {}

            def _watch(_f: asyncio.Future) -> None:
                if _f.cancelled():
                    result_box["exc"] = asyncio.CancelledError()
                elif _f.exception() is not None:
                    result_box["exc"] = _f.exception()
                else:
                    result_box["val"] = _f.result()
                done_event.set()

            self._async_loop.call_soon_threadsafe(fut.add_done_callback, _watch)
            done_event.wait()
            if "exc" in result_box:
                raise result_box["exc"]
            # Flush-time gate fold: mirror the sync path's identity rewrite so
            # callers (summary/contradict edge writers) bind to the surviving
            # record — otherwise their edges dangle on an id that never landed.
            _merged = result_box.get("val")
            if isinstance(_merged, UUID) and _merged != record.id:
                record.id = _merged
            if _psep_action == GateAction.INSERT and not _psep_cfg.dry_run:
                self.boost_edges(
                    [(record.id, record.id)],
                    delta=float(_psep_cfg.link_initial_weight),
                    edge_type="hebbian",
                )
                if _psep_payload:
                    edge_targets = _psep_payload
                    pairs = [
                        (record.id, target_uuid) for target_uuid, _cos in edge_targets
                    ]
                    self.boost_edges(
                        pairs,
                        delta=float(_psep_cfg.link_initial_weight),
                        edge_type="pattern_separation_seed",
                    )
                    _psep_edges_seeded = len(edge_targets)
            if not (_psep_action == GateAction.SKIP and _psep_cfg.dry_run):
                write_event(self, "pattern_separation_pass", {
                    "action": "insert",
                    "near_dup_hit_id": _psep_near_dup_hit_id,
                    "near_dup_cos": _psep_near_dup_cos,
                    "edges_seeded": _psep_edges_seeded,
                    "top_k_probed": _psep_top_k_probed,
                    "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                    "threshold_link": float(_psep_cfg.link_threshold),
                    "dry_run_mode": bool(_psep_cfg.dry_run),
                    "ann_prefilter_cos": _psep_ann_prefilter_cos,
                    "exact_confirm_cos": _psep_exact_confirm_cos,
                }, severity="info", buffered=True)
            return

        row = self._to_row(record)
        _record_buffer.setdefault(id(self), []).append(row)
        if should_flush_record_buffer(id(self)):
            flush_record_buffer(self)
        from iai_mcp.retrieve import invalidate_temporal_validity_cache
        invalidate_temporal_validity_cache(self)
        self._fire_graph_sync_hook("insert", record)
        self._feed_recency(record)
        self._feed_working(record)
        self._feed_lexical(record)
        self._emit_store_watermark(record)
        if _psep_action == GateAction.INSERT and not _psep_cfg.dry_run:
            self.boost_edges(
                [(record.id, record.id)],
                delta=float(_psep_cfg.link_initial_weight),
                edge_type="hebbian",
            )
            if _psep_payload:
                edge_targets = _psep_payload
                pairs = [
                    (record.id, target_uuid) for target_uuid, _cos in edge_targets
                ]
                self.boost_edges(
                    pairs,
                    delta=float(_psep_cfg.link_initial_weight),
                    edge_type="pattern_separation_seed",
                )
                _psep_edges_seeded = len(edge_targets)
        if not (_psep_action == GateAction.SKIP and _psep_cfg.dry_run):
            write_event(self, "pattern_separation_pass", {
                "action": "insert",
                "near_dup_hit_id": _psep_near_dup_hit_id,
                "near_dup_cos": _psep_near_dup_cos,
                "edges_seeded": _psep_edges_seeded,
                "top_k_probed": _psep_top_k_probed,
                "threshold_near_dup": float(_psep_cfg.near_dup_threshold),
                "threshold_link": float(_psep_cfg.link_threshold),
                "dry_run_mode": bool(_psep_cfg.dry_run),
                "ann_prefilter_cos": _psep_ann_prefilter_cos,
                "exact_confirm_cos": _psep_exact_confirm_cos,
            }, severity="info", buffered=True)


    async def enable_async_writes(
        self,
        coalesce_ms: int = 100,
        max_batch: int = 128,
        max_queue_size: int = 4096,
    ) -> None:
        if self._write_queue is not None:
            return

        from iai_mcp.write_queue import AsyncWriteQueue

        ready = threading.Event()
        loop_holder: dict = {}

        def _run() -> None:
            import concurrent.futures as _cf

            loop = asyncio.new_event_loop()
            # Own executor with a bounded, non-waiting shutdown: loop.close()
            # waits on the DEFAULT executor's workers, and a worker stuck on
            # a contended write lock keeps this thread alive past every join
            # timeout — the "async-writes thread leaked" flake. The queue is
            # stopped and flushed before the loop stops, so cancelling a
            # leftover pending future here never drops an accepted write.
            executor = _cf.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="iai-mcp-async-writes-io",
            )
            loop.set_default_executor(executor)
            loop_holder["loop"] = loop
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_forever()
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
                loop.close()

        thread = threading.Thread(
            target=_run, name="iai-mcp-async-writes", daemon=True,
        )
        thread.start()
        ready.wait()
        bg_loop: asyncio.AbstractEventLoop = loop_holder["loop"]

        sync_records_tbl = self.db.open_table(RECORDS_TABLE)

        to_row = self._to_row

        class _RecordTableAdapter:

            def __init__(self, real_tbl, to_row_fn) -> None:
                self._real = real_tbl
                self._to_row = to_row_fn

            async def add(self, records: list) -> None:
                rows = [self._to_row(r) for r in records]
                await asyncio.to_thread(self._real.add, rows)

        adapter = _RecordTableAdapter(sync_records_tbl, to_row)

        fire_hook = self._fire_graph_sync_hook
        feed = self._feed_recency
        feed_exact = self._feed_exact_index
        feed_working = self._feed_working

        def _on_flushed(batch: list) -> None:
            # The async write queue calls on_flushed AFTER adapter.add writes the
            # rows to SQL, so the active count in SQL has already increased by
            # exactly the batch's active rows — shift the cached count in place
            # before notifying the graph hook so any hook that immediately
            # reads the count sees the post-flush live value.
            try:
                n_active = sum(
                    1
                    for rec in batch
                    if getattr(rec, "tombstoned_at", None) is None
                    and not int(getattr(rec, "embedding_pending", 0) or 0)
                )
                if n_active:
                    self._adjust_corpus_count("active", n_active)
            except Exception:  # noqa: BLE001 -- adjustment must not crash flush
                pass
            for rec in batch:
                fire_hook("insert", rec)
                feed(rec)
                feed_working(rec)
                try:
                    feed_exact(str(rec.id), rec.embedding)
                except Exception as exc:  # noqa: BLE001 -- hook isolation
                    logger.debug(
                        "exact-index feed failed on flush for %s: %s",
                        getattr(rec, "id", None),
                        type(exc).__name__,
                    )

        queue = AsyncWriteQueue(
            adapter,
            coalesce_ms=coalesce_ms,
            max_batch=max_batch,
            max_queue_size=max_queue_size,
            on_flushed=_on_flushed,
        )
        asyncio.run_coroutine_threadsafe(queue.start(), bg_loop).result()

        self._async_loop = bg_loop
        self._async_thread = thread
        self._async_conn = None
        self._write_queue = queue

        self.enable_provenance_queue()
        self.enable_reinforce_queue()

    async def disable_async_writes(self) -> None:
        if self._write_queue is None:
            self.disable_provenance_queue()
            self.disable_reinforce_queue()
            return
        self.disable_provenance_queue()
        self.disable_reinforce_queue()
        bg_loop = self._async_loop
        queue = self._write_queue
        try:
            asyncio.run_coroutine_threadsafe(queue.stop(), bg_loop).result()
        finally:
            if bg_loop is not None:
                bg_loop.call_soon_threadsafe(bg_loop.stop)
            if self._async_thread is not None:
                self._async_thread.join(timeout=5.0)
            self._write_queue = None
            self._async_loop = None
            self._async_thread = None
            self._async_conn = None


    def enable_provenance_queue(self, *, coalesce_ms: int = 50) -> None:
        if self._provenance_queue is not None:
            return
        from iai_mcp.provenance_queue import ProvenanceWriteQueue

        q = ProvenanceWriteQueue(self, coalesce_ms=coalesce_ms)
        q.start()
        self._provenance_queue = q

    def disable_provenance_queue(self) -> None:
        q = self._provenance_queue
        if q is None:
            return
        try:
            q.flush(timeout=2.0)
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("provenance queue flush during teardown: %s", exc)
        try:
            q.stop()
        except (OSError, RuntimeError) as exc:
            logger.debug("provenance queue stop during teardown: %s", exc)
        self._provenance_queue = None

    def queue_provenance_batch(
        self,
        pairs: "list[tuple[UUID, dict]]",
        records_cache: "dict | None" = None,
    ) -> None:
        if not pairs:
            return
        q = self._provenance_queue
        if q is not None:
            q.enqueue(pairs)
            return
        self.append_provenance_batch(pairs, records_cache=records_cache)

    def enable_reinforce_queue(self, *, coalesce_ms: int = 50) -> None:
        if self._reinforce_queue is not None:
            return
        from iai_mcp.reinforce_queue import ReinforceWriteQueue
        q = ReinforceWriteQueue(self, coalesce_ms=coalesce_ms)
        q.start()
        self._reinforce_queue = q

    def disable_reinforce_queue(self) -> None:
        q = self._reinforce_queue
        if q is None:
            return
        try:
            q.flush(timeout=2.0)
        except (OSError, RuntimeError, TimeoutError) as exc:
            logger.debug("reinforce queue flush during teardown: %s", exc)
        try:
            q.stop()
        except (OSError, RuntimeError) as exc:
            logger.debug("reinforce queue stop during teardown: %s", exc)
        self._reinforce_queue = None

    def queue_coactivation(
        self, pairs: "list[tuple[UUID, UUID]]", delta: float,
    ) -> None:
        """Deferred pairwise hebbian potentiation for co-recalled records.
        Rides the reinforce queue so the edge write (which scans the hebbian
        edge set) never runs on the synchronous recall path; without the
        queue (tests, non-daemon) it degrades to a direct bounded write."""
        if not pairs:
            return
        q = self._reinforce_queue
        if q is not None and hasattr(q, "enqueue_pairs"):
            q.enqueue_pairs(list(pairs), delta)
            return
        try:
            self.boost_edges(list(pairs), delta=delta, edge_type="hebbian")
        except Exception as exc:  # noqa: BLE001 -- best-effort, never raise into caller
            logger.debug("queue_coactivation_sync_fallback_failed: %s", exc)

    def queue_profile_modulate(
        self, pairs: "list[tuple[UUID, UUID]]", deltas: "list[float]",
    ) -> None:
        """Deferred pairwise profile-modulation potentiation, one call per
        recall. Rides the reinforce queue exactly like queue_coactivation so
        the edge write never runs on the synchronous recall path; without the
        queue (tests, non-daemon) it degrades to a direct bounded write.

        Accepts per-pair deltas: boost_edges accumulates weight per canonical
        edge additively regardless of how many calls carry a given pair, so
        grouping pairs by delta value before enqueue is exact-equivalent to
        passing the full per-pair delta list through in one synchronous call.
        """
        if not pairs:
            return
        if len(deltas) != len(pairs):
            raise ValueError(
                f"deltas length {len(deltas)} != pairs length {len(pairs)}"
            )
        q = self._reinforce_queue
        if q is not None and hasattr(q, "enqueue_pairs"):
            by_delta: dict[float, list] = {}
            for (a, b), d in zip(pairs, deltas):
                by_delta.setdefault(float(d), []).append((a, b))
            for delta, group in by_delta.items():
                q.enqueue_pairs(group, delta, edge_type="profile_modulates")
            return
        # profile_modulates edges share one endpoint (PROFILE_SENTINEL_UUID)
        # -- keep this chunk size == BOOST_EDGES_SMALL_BATCH or this fallback
        # re-triggers the full-table scan on the recall hot path.
        for start in range(0, len(pairs), BOOST_EDGES_SMALL_BATCH):
            chunk_pairs = pairs[start:start + BOOST_EDGES_SMALL_BATCH]
            chunk_deltas = deltas[start:start + BOOST_EDGES_SMALL_BATCH]
            try:
                self.boost_edges(chunk_pairs, delta=chunk_deltas, edge_type="profile_modulates")
            except Exception as exc:  # noqa: BLE001 -- best-effort, never raise into caller
                logger.debug("queue_profile_modulate_sync_fallback_failed: %s", exc)

    def queue_reinforce(self, record_ids: "list[UUID]") -> None:
        """Enqueue record ids for deferred reinforcement, or reinforce synchronously.

        When the background reinforce queue is live (enabled), record ids are
        enqueued for deferred processing on the background thread — the call
        returns immediately without waiting for the writes.  When the queue is
        not live (e.g. in non-daemon or test contexts without async writes),
        reinforcement falls back to synchronous per-id calls so the writes
        still happen (the fallback path is never a no-op).
        """
        if not record_ids:
            return
        q = self._reinforce_queue
        if q is not None:
            q.enqueue(list(record_ids))
            return
        # Synchronous fallback: reinforce each id directly.
        for rid in record_ids:
            try:
                self.reinforce_record(rid, is_retrieval=True)
            except Exception as exc:  # noqa: BLE001 -- best-effort, never raise into caller
                logger.debug("queue_reinforce_sync_fallback_failed: rid=%s err=%s", rid, exc)

    def update(self, record: MemoryRecord) -> None:
        if len(record.embedding) != self._embed_dim:
            raise ValueError(
                f"embedding must be {self._embed_dim}d, got {len(record.embedding)}"
            )
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        if df.empty or str(record.id) not in set(df["id"].tolist()):
            return
        literal_ct = self._encrypt_for_record(record.id, record.literal_surface)
        tbl.update(
            where=f"id = '{_uuid_literal(record.id)}'",
            values={
                "literal_surface": literal_ct,
                "embedding": [float(x) for x in record.embedding],
                "centrality": float(record.centrality),
                "tier": record.tier,
                "pinned": bool(record.pinned),
                "updated_at": datetime.now(timezone.utc),
            },
        )
        self._fire_graph_sync_hook("update", record)
        self._feed_recency(record)
        self._feed_working(record)
        self._feed_lexical(record, restamp=True)

    def delete(self, record_id: UUID) -> None:
        tbl = self.db.open_table(RECORDS_TABLE)
        try:
            tbl.delete(where=f"id = '{_uuid_literal(record_id)}'")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("store delete normalised to no-op for %s: %s", record_id, exc)
            return

        # The row has been removed from SQL: the active count decreased.  Invalidate
        # so the next active_records_count() recomputes the live filtered count.
        # store.delete does not cascade to edges, so only the active key is affected.
        self._invalidate_corpus_count("active")
        # The deleted id must never linger in the resident exact-cosine matrix:
        # invalidate so the next exact_top_k rebuilds from the post-delete SQL
        # state.
        self.invalidate_exact_index()

        class _DeleteShim:
            def __init__(self, rid):
                self.id = rid
        self._fire_graph_sync_hook("delete", _DeleteShim(record_id))

    def get(self, record_id: UUID) -> MemoryRecord | None:
        tbl = self.db.open_table(RECORDS_TABLE)
        df = (
            tbl.search()
            .where(f"id = '{_uuid_literal(record_id)}'")
            .limit(1)
            .to_pandas()
        )
        if df.empty:
            return None
        return self._from_row(df.iloc[0].to_dict())

    def all_records(self) -> list[MemoryRecord]:
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        return [self._from_row(r.to_dict()) for _, r in df.iterrows()]

    def active_records_count(self) -> int:
        """COUNT of active records (not tombstoned, not pending an embedding).

        Read sequence (memo-then-cache-then-live):
        1. Operation-scoped ``count_memo()`` checked first: collapses repeats
           within a single build and is the correct boundary for within-build
           stale-count masking (a write inside an open memo is reflected on
           the NEXT memo scope, not immediately — this is intentional; the
           post-write recall opens a fresh memo, misses the invalidated cache,
           and recomputes live, so the effect is never lost permanently).
        2. Cross-read ``_corpus_count_cache`` checked second: on a cache hit
           the live SQL COUNT is skipped and the cached value is seeded into
           the open memo (if any) so subsequent same-scope reads are free.
        3. On any miss (cache miss or any cache-layer exception), the live
           FILTERED SQL COUNT is run under ``_conn_lock``.  The cache-layer
           ``try/except`` ensures that a ``KeyError``, ``AttributeError``, or
           missing-attribute error on the cache path degrades to the live
           filtered COUNT and is never propagated to the caller.  This keeps
           the unfiltered ``count_rows()`` fallback in
           ``runtime_graph_cache._cache_key`` unreachable via this method.

        Lock ordering: the cache lock is taken and released around the dict
        read only; the SQL COUNT runs on the read-only pool (or its internal
        ``_conn_lock`` fallback) with no cache lock held, so the cache lock is
        never held together with a connection lock.
        """
        memo = self._count_memo
        if memo is not None and "active" in memo:
            return memo["active"]
        # Cross-read cache: check then, on a hit, skip the SQL COUNT.
        try:
            cached = self._corpus_count_cache.get("active")
        except Exception:  # noqa: BLE001
            cached = None
        if cached is not None:
            if memo is not None:
                memo["active"] = cached
            return cached
        # Cache miss: snapshot the generation BEFORE the live COUNT so a
        # concurrent invalidate during the COUNT refuses the stale put.
        try:
            gen = self._corpus_count_cache.generation()
        except Exception:  # noqa: BLE001
            gen = None
        # Run the live FILTERED COUNT on the read-only pool: an O(corpus)
        # count on the lilli engine re-scans every leaf page, and taking the
        # shared writer's _conn_lock for it made every cache-miss recall wait
        # behind an active write batch. The RO snapshot is the last committed
        # state — exactly what a generation-fenced cache entry may hold; the
        # pool falls back to the _conn_lock path by itself when unhealthy.
        with self.db.ro_conn() as _ro:
            row = _ro.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        value = int(row[0]) if row else 0
        # Populate the cross-read cache (compare-and-set on generation) and the
        # open memo (best-effort).
        try:
            if gen is not None:
                self._corpus_count_cache.put_if_gen("active", value, gen)
        except Exception:  # noqa: BLE001
            pass
        if memo is not None:
            memo["active"] = value
        return value

    def pending_records_count(self) -> int:
        """COUNT of records still awaiting an embedding (embedding_pending = 1).

        The runtime-graph cache key folds this raw (unwindowed) count in so a
        single re-embed flip (pending −1) changes the key and forces a warm
        rebuild, ensuring the newly-embedded row is included.

        Read sequence is memo-then-cache-then-live (same as
        ``active_records_count``).  A cache-layer exception degrades to the
        live FILTERED COUNT; the unfiltered fallback in ``_cache_key`` is
        unreachable via this method.
        """
        memo = self._count_memo
        if memo is not None and "pending" in memo:
            return memo["pending"]
        try:
            cached = self._corpus_count_cache.get("pending")
        except Exception:  # noqa: BLE001
            cached = None
        if cached is not None:
            if memo is not None:
                memo["pending"] = cached
            return cached
        try:
            gen = self._corpus_count_cache.generation()
        except Exception:  # noqa: BLE001
            gen = None
        with self.db._conn_lock:
            row = self.db._conn.execute(
                "SELECT COUNT(*) FROM records WHERE embedding_pending = 1"
            ).fetchone()
        value = int(row[0]) if row else 0
        try:
            if gen is not None:
                self._corpus_count_cache.put_if_gen("pending", value, gen)
        except Exception:  # noqa: BLE001
            pass
        if memo is not None:
            memo["pending"] = value
        return value

    def edges_count(self) -> int:
        """COUNT of rows in the edges table.

        Folded into the runtime-graph cache key alongside the record counts.
        Read sequence is memo-then-cache-then-live (same as
        ``active_records_count``).  A cache-layer exception degrades to the
        live count; the unfiltered fallback in ``_cache_key`` is unreachable
        via this method.
        """
        memo = self._count_memo
        if memo is not None and "edges" in memo:
            return memo["edges"]
        try:
            cached = self._corpus_count_cache.get("edges")
        except Exception:  # noqa: BLE001
            cached = None
        if cached is not None:
            if memo is not None:
                memo["edges"] = cached
            return cached
        try:
            gen = self._corpus_count_cache.generation()
        except Exception:  # noqa: BLE001
            gen = None
        value = int(self.db.open_table("edges").count_rows())
        try:
            if gen is not None:
                self._corpus_count_cache.put_if_gen("edges", value, gen)
        except Exception:  # noqa: BLE001
            pass
        if memo is not None:
            memo["edges"] = value
        return value

    def find_record_by_tag(self, tag: str) -> UUID | None:
        """Resolve a record id by an exact tag match.

        Indexed equality lookup on ``record_tags(record_id, tag)`` — no scan
        of ``records.tags_json``. Falls back to the legacy LIKE-scan only when
        the one-time backfill (``migrate_record_tags_backfill``) failed for
        this store instance, so a store whose tag index could not be
        populated still resolves correctly rather than silently returning
        wrong answers. This is a per-store-instance flag set once at open —
        never a silent per-call fallback.

        Single-table queries only (both drivers' SQL surfaces support no
        JOIN and no column-qualified aliases across tables — confirmed on
        the lilli engine: ``JOIN`` and ``SELECT r.col`` both raise). Resolves
        via two single-table lookups: candidates from ``record_tags`` ordered
        by its own insertion order (rowid), then the first candidate that
        still exists in ``records`` wins. ``record_tags`` rows are written
        in the same relative order as their owning ``records`` row (each
        record's tags are upserted immediately after that record's insert, in
        the same transaction), so this reproduces the pre-index LIKE-scan's
        natural table-scan (insertion order) first-match semantics. The
        existence re-check also guards against an orphaned ``record_tags``
        row surviving a swallowed ``_delete_record_tags`` failure.
        """
        if not getattr(self, "_record_tags_backfill_ok", True):
            found = self._find_record_by_tag_like_scan(tag)
            if found is not None:
                return found
            return self._find_buffered_record_by_tag(tag)

        with self.db._conn_lock:
            candidates = self.db._conn.execute(
                "SELECT record_id FROM record_tags WHERE tag = ? ORDER BY rowid",
                (tag,),
            ).fetchall()
            for cand in candidates:
                raw_id = cand["record_id"]
                if raw_id is None:
                    continue
                exists = self.db._conn.execute(
                    "SELECT 1 FROM records WHERE id = ? LIMIT 1",
                    (str(raw_id),),
                ).fetchone()
                if exists is None:
                    continue
                try:
                    return UUID(str(raw_id))
                except (ValueError, AttributeError):
                    continue
        return self._find_buffered_record_by_tag(tag)

    def _find_buffered_record_by_tag(self, tag: str) -> UUID | None:
        """Tag lookup over rows still sitting in the in-memory insert buffer.
        Without this, an exact-key dedup check between an insert and its
        flush misses the just-inserted row and mints a duplicate."""
        from iai_mcp.store._buffers import _record_buffer

        for row in list(_record_buffer.get(id(self), [])):
            try:
                if tag in json.loads(row.get("tags_json") or "[]"):
                    return UUID(str(row.get("id")))
            except (TypeError, ValueError):
                continue
        return None

    def add_tags(self, record_id: UUID, tags: "list[str]") -> bool:
        """Union tags onto an existing record, keeping the indexed
        ``record_tags`` table in sync (the generic ``update()`` does not
        maintain it). Plaintext column; verbatim surface untouched. Returns
        True when the stored set changed."""
        new_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]
        if not new_tags:
            return False
        rec = self.get(record_id)
        if rec is None:
            return False
        current = list(rec.tags or [])
        missing = [t for t in new_tags if t not in current]
        if not missing:
            return False
        tags_json = json.dumps(current + missing)
        self.db.open_table(RECORDS_TABLE).update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={"tags_json": tags_json},
        )
        from iai_mcp.hippo._table import _upsert_record_tags
        with self.db._conn_lock:
            _upsert_record_tags(self.db._conn, str(record_id), tags_json)
        return True

    def set_aaak_index(self, record_id: UUID, aaak_index: str) -> bool:
        """Persist a regenerated AAAK index. Plaintext column; the generic
        ``update()`` does not carry it, and the current value is read as a
        bare column — a full ``get()`` here would decrypt the record just
        to compare plaintext. Returns True when the value changed."""
        with self.db._conn_lock:
            row = self.db._conn.execute(
                f"SELECT aaak_index FROM {RECORDS_TABLE}"
                " WHERE id = ? AND tombstoned_at IS NULL",
                (_uuid_literal(record_id),),
            ).fetchone()
        if row is None or str(row["aaak_index"] or "") == str(aaak_index):
            return False
        self.db.open_table(RECORDS_TABLE).update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={"aaak_index": str(aaak_index)},
        )
        return True

    def remove_tags(self, record_id: UUID, tags: "list[str]") -> bool:
        """Remove tags from an existing record, mirrored into ``record_tags``.
        Returns True when the stored set changed."""
        drop = {str(t).strip() for t in (tags or []) if str(t).strip()}
        if not drop:
            return False
        rec = self.get(record_id)
        if rec is None:
            return False
        current = list(rec.tags or [])
        kept = [t for t in current if t not in drop]
        if len(kept) == len(current):
            return False
        self.db.open_table(RECORDS_TABLE).update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={"tags_json": json.dumps(kept)},
        )
        with self.db._conn_lock:
            try:
                self.db._conn.executemany(
                    "DELETE FROM record_tags WHERE record_id = ? AND tag = ?",
                    [(str(record_id), t) for t in sorted(drop)],
                )
            except Exception:  # noqa: BLE001 -- tag-index maintenance must not abort
                logger.debug(
                    "record_tags delete failed for %s", record_id, exc_info=True,
                )
        return True

    def _find_record_by_tag_like_scan(self, tag: str) -> UUID | None:
        """The pre-index LIKE-scan implementation, kept as the fallback path
        for the backfill-failed edge case only (never a silent per-call
        fallback — gated on ``_record_tags_backfill_ok`` in
        ``find_record_by_tag``).
        """
        tag_json_literal = json.dumps(tag)
        sql = (
            "SELECT id, tags_json FROM records"
            " WHERE tags_json LIKE :pat"
        )
        params = {"pat": f"%{tag_json_literal}%"}
        with self.db._conn_lock:
            rows = self.db._conn.execute(sql, params).fetchall()
        if rows is None:
            raise HippoIntegrityError(
                "find_record_by_tag: fetchall() returned None — connection may be"
                " in an error state"
            )
        for row in rows:
            tags_raw = row["tags_json"] if row["tags_json"] else "[]"
            try:
                tags = json.loads(tags_raw)
            except (ValueError, TypeError):
                continue
            if tag in tags:
                raw_id = row["id"]
                if raw_id is None:
                    continue
                try:
                    return UUID(str(raw_id))
                except (ValueError, AttributeError):
                    continue
        return None

    def centrality_for_ids(self, ids: list[UUID]) -> dict[UUID, float]:
        if not ids:
            return {}
        target = frozenset(str(i) for i in ids)
        out: dict[UUID, float] = {}
        for row in self.iter_record_columns(["id", "centrality"]):
            raw_id = row.get("id")
            if raw_id is None:
                continue
            id_str = str(raw_id)
            if id_str not in target:
                continue
            try:
                centrality = float(row.get("centrality") or 0.0)
            except (TypeError, ValueError):
                centrality = 0.0
            try:
                out[UUID(id_str)] = centrality
            except (ValueError, AttributeError):
                continue
        return out

    _RECENT_USER_TURNS_COLUMNS = [
        "id",
        "tier",
        "literal_surface",
        "tags_json",
        "provenance_json",
        "created_at",
        "embedding_pending",
    ]

    # Bare-shape widening bound for the recent_user_turns candidate read:
    # the SQL below must stay a WHERE-less `ORDER BY created_at DESC LIMIT`
    # to ride OrderedColIndex's strict top-K fast path (any residual
    # conjunct forces a full scan). MAX_WIDEN_DOUBLINGS bounds the Python-
    # side widening loop so a sparse-and-old corpus degrades gracefully
    # (returns fewer than n candidates) instead of widening toward a
    # full-corpus scan.
    _RECENT_USER_TURNS_MAX_WIDEN_DOUBLINGS = 6

    def _recent_user_turns_candidate_rows(self, k: int) -> list:
        """Bare top-K read: no WHERE clause, so it rides the ordered-column
        index's strict fast path (a residual predicate would force a full
        scan). Filtering happens in Python on the fetched window only.
        """
        cols = ", ".join(self._RECENT_USER_TURNS_COLUMNS)
        sql = f"SELECT {cols} FROM records ORDER BY created_at DESC LIMIT ?"
        with self.db._conn_lock:
            rows = self.db._conn.execute(sql, (int(k),)).fetchall()
        return [self._from_row(dict(row)) for row in rows]

    def recent_user_turns(
        self,
        n: int = 10,
        session_id: str | None = None,
        pending_live_events: "list | None" = None,
    ) -> "list":
        from iai_mcp.capture import _idem_tag as _cap_idem_tag

        n_effective = max(1, int(n)) if n and n > 0 else 1
        k = max(4 * n_effective, 40)
        cands: list = []
        doublings = 0
        while True:
            fetched = self._recent_user_turns_candidate_rows(k)
            cands = [
                r for r in fetched
                if r.tier == "episodic"
                and (
                    "role:user" in (r.tags or [])
                    or r.embedding_pending
                )
            ]
            if session_id:
                cands = [
                    r for r in cands
                    if (r.provenance or [{}])[0].get("session_id") == session_id
                ]
            enough_matches = len(cands) >= n_effective
            window_exhausted = len(fetched) < k
            if enough_matches or window_exhausted:
                break
            if doublings >= self._RECENT_USER_TURNS_MAX_WIDEN_DOUBLINGS:
                break
            k *= 2
            doublings += 1

        if pending_live_events is not None:
            seen_pending_idem: set[str] = set()

            pending_wrappers = []
            for ev in pending_live_events:
                if ev.get("role") != "user":
                    continue
                ev_session = ev.get("session_id", "-")
                if session_id and ev_session != session_id:
                    continue
                src_uuid = ev.get("source_uuid")
                ts_iso = ev["ts_iso"]
                text = ev.get("text", "")
                idem = _cap_idem_tag(ev_session, "user", ts_iso, text, source_uuid=src_uuid)
                if self.find_record_by_tag(idem) is not None:
                    continue
                if idem in seen_pending_idem:
                    continue
                seen_pending_idem.add(idem)
                pending_wrappers.append(_PendingTurn(
                    text=text,
                    session_id=ev_session,
                    ts=ev["ts"],
                    idem_tag=idem,
                    source_uuid=src_uuid,
                ))

            cands = list(cands) + pending_wrappers  # type: ignore[arg-type]

        cands.sort(key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return cands[:n]

    def iter_records(
        self,
        *,
        columns: list[str] | None = None,
        batch_size: int = 1024,
        where: str | None = None,
    ):
        tbl = self.db.open_table(RECORDS_TABLE)
        query = tbl.search()
        if where is not None:
            query = query.where(where)
        if columns is not None:
            query = query.select(columns)
        reader = query.to_batches(batch_size=batch_size)
        for batch in reader:
            for row_dict in batch.to_pylist():
                yield self._from_row(row_dict)

    def iter_record_columns(
        self,
        columns: list[str],
        *,
        batch_size: int = 1024,
        where: str | None = None,
    ):
        if not columns:
            raise ValueError("iter_record_columns requires a non-empty columns list")
        db_path = getattr(self.db, "_db_path", None)
        if not db_path:
            # Detached / pre-construction store: no lock-free snapshot to page
            # over — fall back to the streaming reader.
            tbl = self.db.open_table(RECORDS_TABLE)
            query = tbl.search()
            if where is not None:
                query = query.where(where)
            query = query.select(columns)
            for batch in query.to_batches(batch_size=batch_size):
                yield from batch.to_pylist()
            return

        # Keyset pagination over the primary key: each page is a short-lived
        # query on its own read-only snapshot, resuming from the last id seen.
        # A concurrent WAL checkpoint fences the snapshot ("read-only snapshot
        # invalidated") — but because the cursor advances by id, a reopen
        # resumes exactly where it left off instead of re-scanning from the
        # start. Forward progress is therefore guaranteed at any corpus size:
        # the pipeline can no longer fence its own full-corpus scans to death.
        from iai_mcp import errors as _errors
        from iai_mcp.hippo._ro_pool import _is_snapshot_fence

        want = list(columns)
        select_cols = want if "id" in want else ["id", *want]
        col_sql = ", ".join(select_cols)
        where_sql = f"({where}) AND " if where else ""
        page_sql = (
            f"SELECT {col_sql} FROM {RECORDS_TABLE} WHERE {where_sql}id > ?"
            f" ORDER BY id LIMIT {int(batch_size)}"
        )

        strip_id = "id" not in want
        conn = self._open_keyset_snapshot(db_path)
        last_id = ""
        # Bound CONSECUTIVE fence reopens: a page that fences before returning
        # its first row, over and over (a checkpoint flapping faster than the
        # page drains), would otherwise spin forever with last_id never
        # advancing. A successful page read clears the streak, so genuine
        # progress is never capped.
        _MAX_CONSECUTIVE_FENCE = 8
        consecutive_fence = 0
        try:
            while True:
                try:
                    rows = conn.execute(page_sql, (last_id,)).fetchall()
                except _errors.OperationalError as exc:
                    if not _is_snapshot_fence(exc):
                        raise
                    consecutive_fence += 1
                    if consecutive_fence > _MAX_CONSECUTIVE_FENCE:
                        # No forward progress across the bound — surface it so a
                        # higher layer can fall back rather than livelock here.
                        raise
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    time.sleep(0.005 * consecutive_fence)  # brief backoff to let the checkpoint settle
                    conn = self._open_keyset_snapshot(db_path)  # resume from last_id — no re-scan
                    continue
                consecutive_fence = 0  # a page read succeeded — reset the streak
                if not rows:
                    break
                for row in rows:
                    d = dict(row)
                    last_id = d["id"]
                    if strip_id:
                        d.pop("id", None)
                    # The raw page carries the embedding column as its stored
                    # BLOB; without this decode a 384-d vector leaks downstream
                    # as 1536 per-byte ints and poisons every consumer matrix.
                    yield self._decode_raw_row(d)
        finally:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass

    def _open_keyset_snapshot(self, db_path: str):
        """Short-lived read-only snapshot for a keyset scan page (driver-aware,
        lock-free). A single seam so a fence retry always reopens the same way."""
        import contextlib as _cl

        from iai_mcp import _sqlite_stdlib
        from iai_mcp.hippo._raw_open import open_store_conn

        conn = open_store_conn(db_path, read_only=True)
        if conn is None:
            conn = _sqlite_stdlib.connect(
                db_path, check_same_thread=False, isolation_level=None,
            )
            conn.row_factory = _sqlite_stdlib.Row
        with _cl.suppress(Exception):
            conn.execute("PRAGMA busy_timeout=2000")
        with _cl.suppress(Exception):
            conn.execute("PRAGMA query_only=ON")
        return conn

    _EXACT_INDEX_BUILD_BACKOFF_SEC = 30.0
    _CUE_NONFINITE_EMIT_WINDOW_SEC = 60.0

    def _maybe_emit_cue_nonfinite(self, count: int, total: int) -> None:
        """Rate-limited, buffered ``TELEMETRY_EMBED_NONFINITE`` emit for a
        coerced query cue. The first coercion always emits (``total ==
        count``); subsequent ones are suppressed within the window, but
        ``total`` stays the exact running count regardless."""
        now = time.monotonic()
        if total > count and (now - self._last_cue_nonfinite_emit_at) < self._CUE_NONFINITE_EMIT_WINDOW_SEC:
            return
        self._last_cue_nonfinite_emit_at = now
        from iai_mcp.events import TELEMETRY_EMBED_NONFINITE, emit_best_effort
        emit_best_effort(
            self,
            TELEMETRY_EMBED_NONFINITE,
            {
                "source": "cue",
                "action": "coerced",
                "context": "cue",
                "count": count,
                "total": total,
            },
            severity="warning",
        )

    def _build_exact_index_sync(self) -> None:
        """Single-flight cold build of the resident exact-cosine matrix.

        Reads the active corpus (plaintext id+embedding through
        ``self.db.ro_conn()``, no AES decrypt, no record materialization,
        never touching ``_hnsw_lock``) and installs the normalized matrix.
        Serialized by ``_exact_index_build_lock`` so a warm-up probe racing a
        first recall never runs two whole-corpus builds side by side — the
        loser of the race re-checks warmth under the lock and no-ops.

        A build failure starts a cooldown window
        (``_EXACT_INDEX_BUILD_BACKOFF_SEC``): while it is active, callers skip
        the full-corpus SELECT + build retry entirely, so a persistently
        unbuildable state costs one scan, not one scan per recall.
        ``invalidate_exact_index()`` clears the cooldown immediately, so a
        fresh write always gets an immediate retry. No-raise.
        """
        with self._exact_index_build_lock:
            try:
                if self._exact_index.is_warm:
                    return
                _cooldown_until = getattr(self, "_exact_index_build_cooldown_until", 0.0)
                if time.monotonic() < _cooldown_until:
                    return
                _expected_gen = self._exact_index.generation
                # The whole-corpus embedding fetch holds its connection for
                # the full scan — run it on a DEDICATED short-lived snapshot
                # reader (cheap to open; the decoded index sidecar is served
                # from the process-wide cache) so this maintenance read never
                # occupies a pooled reader slot or the shared writer
                # connection that live recalls depend on.
                _sql = (
                    "SELECT id, embedding FROM records"
                    " WHERE tombstoned_at IS NULL"
                    " AND COALESCE(embedding_pending, 0) = 0"
                )
                rows = None
                _snap = None
                try:
                    from iai_mcp.hippo._raw_open import open_store_conn

                    _db_path = getattr(self.db, "_db_path", None)
                    if _db_path is not None:
                        _snap = open_store_conn(_db_path, read_only=True)
                except Exception:  # noqa: BLE001 -- snapshot open is an
                    # optimization; the pooled reader below stays correct.
                    _snap = None
                if _snap is not None:
                    try:
                        # Engine raw rows are name-keyed natively; no row
                        # factory needed (open_store_conn never returns a
                        # stdlib connection — it yields engine-or-None).
                        rows = _snap.execute(_sql).fetchall()
                    finally:
                        try:
                            _snap.close()
                        except Exception:  # noqa: BLE001 -- close is best-effort
                            pass
                if rows is None:
                    with self.db.ro_conn() as conn:
                        rows = conn.execute(_sql).fetchall()
                build_rows = [(row["id"], row["embedding"]) for row in rows]
                _before_row = self._exact_index.coerced_row_total
                try:
                    self._exact_index.build(build_rows, expected_generation=_expected_gen)
                except Exception:
                    self._exact_index_build_cooldown_until = (
                        time.monotonic() + self._EXACT_INDEX_BUILD_BACKOFF_SEC
                    )
                    logger.warning(
                        "exact-cosine matrix build failed; backing off for %.0fs",
                        self._EXACT_INDEX_BUILD_BACKOFF_SEC,
                        exc_info=True,
                    )
                else:
                    _after_row = self._exact_index.coerced_row_total
                    if _after_row > _before_row:
                        try:
                            from iai_mcp.events import (
                                TELEMETRY_EMBED_NONFINITE,
                                write_event,
                            )
                            write_event(
                                self,
                                TELEMETRY_EMBED_NONFINITE,
                                {
                                    "source": "row",
                                    "action": "coerced",
                                    "context": "build",
                                    "count": _after_row - _before_row,
                                    "total": _after_row,
                                },
                                severity="warning",
                            )
                        except Exception:  # noqa: BLE001 -- a telemetry emit
                            # must never turn a successful build into a
                            # reported failure.
                            pass
            except Exception:  # noqa: BLE001 -- no-raise contract
                logger.debug("exact-cosine matrix build failed", exc_info=True)

    def _schedule_exact_index_build(self) -> None:
        """Kick the cold-matrix build on a background thread, at most one at a
        time (the build lock is the single-flight gate; a second kick while a
        build is in flight starts a thread that immediately queues on the lock,
        re-checks warmth, and exits). No-raise."""
        try:
            threading.Thread(
                target=self._build_exact_index_sync,
                name="exact-index-build",
                daemon=True,
            ).start()
        except Exception:  # noqa: BLE001 -- scheduling failure degrades to the
            # next caller's kick; never breaks the caller.
            logger.debug("exact-index build scheduling failed", exc_info=True)

    def lexical_search(
        self, query: str, k: int = 10,
    ) -> "list[tuple[MemoryRecord, float]]":
        """Identifier-grade lexical lane over decrypted surfaces (in RAM only).

        Complements the semantic lane where embeddings are weakest: exact
        code identifiers and env names. Rebuilds on demand when the corpus
        generation moved; surfaces are write-once verbatim, so a generation
        keyed on corpus-changing writes is a sufficient freshness fence.
        The rebuild never runs on the recall critical path — recall reads
        the warm index only (``lexical_query_warm``); this building entry is
        the scoped-search surface (MCP memory_search / CLI / warm-up)."""
        from iai_mcp.store._lexical_index import LexicalIndex

        idx = getattr(self, "_lexical_idx", None)
        if idx is None:
            idx = LexicalIndex()
            self._lexical_idx = idx
        try:
            gen = self._corpus_count_cache.generation()
        except Exception:  # noqa: BLE001
            gen = None
        if idx.generation is None or idx.generation != gen:
            # Single-flight: the rebuild decrypts the whole corpus; a second
            # concurrent search waits here and finds the fresh generation.
            with idx.build_lock:
                if idx.generation is None or idx.generation != gen:
                    ids: list[UUID] = []
                    # Pending-embed rows ARE included: the lexical lane needs
                    # only the decrypted surface, never the vector, so a
                    # captured-but-not-yet-embedded row must be findable by
                    # exact identifier immediately.
                    for row in self.iter_record_columns(
                        ["id"],
                        batch_size=2048,
                        where="tombstoned_at IS NULL",
                    ):
                        try:
                            ids.append(UUID(str(row["id"])))
                        except (TypeError, ValueError):
                            continue
                    rows: list[tuple[str, str]] = []
                    for i in range(0, len(ids), 400):
                        batch = self.get_batch(ids[i : i + 400])
                        rows.extend(
                            (str(rid), rec.literal_surface or "")
                            for rid, rec in batch.items()
                        )
                    idx.build(rows, gen)
        pairs = idx.query(query, k=k)
        if not pairs:
            return []
        recs = self.get_batch(
            [u for u in (self._maybe_uuid(rid) for rid, _s in pairs) if u]
        )
        out: "list[tuple[MemoryRecord, float]]" = []
        for rid, score in pairs:
            rec = recs.get(self._maybe_uuid(rid))
            if rec is not None:
                out.append((rec, score))
        return out

    def lexical_query_warm(
        self, query: str, k: int = 10, *, min_idf: "float | None" = None,
    ) -> "list[tuple[str, float]]":
        """Recall-path lexical lane: answers ONLY from a warm,
        generation-current index — a cold or stale index yields empty with
        ZERO side effects (no rebuild, no thread, no store reads). The
        index is built by the scoped-search surface or the nightly warm-up
        and kept current by the per-insert feed.

        min_idf gates fusion on lexical signal: a cue whose in-corpus tokens
        are all ubiquitous returns empty rather than noise."""
        idx = getattr(self, "_lexical_idx", None)
        if idx is None or idx.generation is None:
            return []
        try:
            gen = self._corpus_count_cache.generation()
        except Exception:  # noqa: BLE001 -- no generation means no freshness fence
            gen = None
        if idx.generation != gen:
            return []
        if min_idf is not None and idx.max_idf(query) < min_idf:
            return []
        return idx.query(query, k=k)

    @staticmethod
    def _maybe_uuid(value: str) -> "UUID | None":
        try:
            return UUID(str(value))
        except (TypeError, ValueError):
            return None

    def exact_top_k(
        self, vec: list[float], k: int = 10, *, build_if_cold: bool = True,
    ) -> list[tuple[UUID, float]]:
        """Exact cosine top-k over the active corpus, from the resident matrix.

        Lazily builds the matrix on first call (single-flight, see
        ``_build_exact_index_sync``). With ``build_if_cold=False`` — the recall
        critical path — a cold matrix is never built synchronously: the build
        is kicked onto a background thread and THIS call degrades to an empty
        result immediately, so index maintenance never sits on the awake
        recall path. The exact authority is a backstop over the ANN candidates
        (its callers already treat an empty result as "authority contributed
        nothing"), and it warms within one background build regardless.

        No-raise contract: any internal failure — a cold rebuild that cannot
        complete, a corrupted index, an unexpected exception — degrades to an
        empty result, never to a raised exception. A broken exact-similarity
        authority must never make recall worse than baseline.

        The matrix is invalidated on every corpus-changing destructive write
        (delete, sleep-pipeline optimize/erasure, dedupe migrator) so it never
        serves a tombstoned or stale id; the recall-side authority path adds a
        live-id-set liveness filter on top of this for the same reason.
        """
        try:
            if not self._exact_index.is_warm:
                _cooldown_until = getattr(self, "_exact_index_build_cooldown_until", 0.0)
                if time.monotonic() < _cooldown_until:
                    return []
                if build_if_cold:
                    self._build_exact_index_sync()
                else:
                    self._schedule_exact_index_build()
                    return []
            _before_cue = self._exact_index.coerced_cue_total
            result = self._exact_index.top_k(vec, k)
            _after_cue = self._exact_index.coerced_cue_total
            if _after_cue > _before_cue:
                try:
                    self._maybe_emit_cue_nonfinite(_after_cue - _before_cue, _after_cue)
                except Exception:  # noqa: BLE001 -- a telemetry emit must
                    # never discard the ranking already computed above.
                    pass
            if result is None:
                return []
            return [(UUID(rid), score) for rid, score in result]
        except Exception:  # noqa: BLE001 -- no-raise contract, a broken authority
            # must never break recall.
            logger.debug("exact_top_k failed; returning empty result", exc_info=True)
            return []

    def _feed_exact_index(self, record_id: str, vec) -> None:
        """No-raise upsert forward into the resident exact-cosine matrix.

        A no-op when the matrix is cold (the index itself no-ops an upsert
        while cold); the next exact_top_k call rebuilds from live SQL instead.
        """
        try:
            _before_row = self._exact_index.coerced_row_total
            self._exact_index.upsert(record_id, vec)
            _after_row = self._exact_index.coerced_row_total
            if _after_row > _before_row:
                try:
                    from iai_mcp.events import TELEMETRY_EMBED_NONFINITE, emit_best_effort
                    emit_best_effort(
                        self,
                        TELEMETRY_EMBED_NONFINITE,
                        {
                            "source": "row",
                            "action": "coerced",
                            "context": "upsert",
                            "count": _after_row - _before_row,
                            "total": _after_row,
                        },
                        severity="warning",
                    )
                except Exception:  # noqa: BLE001 -- a telemetry emit must
                    # never turn a successful upsert into a reported failure.
                    pass
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "exact-index upsert failed for %s: %s",
                record_id,
                type(exc).__name__,
            )

    def invalidate_exact_index(self) -> None:
        """Mark the resident exact-cosine matrix cold; no-raise.

        Every corpus-changing write path that fires through HippoDB or the
        sleep pipeline routes its destructive-write invalidation through here
        so the matrix never serves a tombstoned or stale id: the next
        exact_top_k call rebuilds from a fresh active-corpus scan. Clears any
        active build-failure cooldown so a fresh write always gets an
        immediate retry rather than waiting out the backoff window.
        """
        try:
            self._exact_index.invalidate()
        except Exception:  # noqa: BLE001
            pass
        self._exact_index_build_cooldown_until = 0.0

    def query_similar(
        self,
        vec: list[float],
        k: int = 10,
        tier: str | None = None,
        *,
        n: int | None = None,
        over_fetch_factor: int = 3,
        decode: str = "full",
        substage_timings: "dict | None" = None,
    ) -> "list[tuple[MemoryRecord | RankCandidateView, float]]":
        # substage_timings is only wired on the fast-decode branch
        # (IAI_MCP_ANN_FAST_DECODE_OFF unset) — the fallback DataFrame path
        # is not sub-instrumented.
        # decode="rank" is the only lazy tier; any other value (including a
        # typo) falls back to the eager full decode — never a silent no-op.
        _lazy_decode = (
            decode == "rank"
            and os.environ.get("IAI_MCP_LAZY_DECODE_OFF") != "1"
        )
        _decode_row = self._from_row_rank_view if _lazy_decode else self._from_row
        _return_records_only = n is not None
        if n is not None:
            k = n
        if tier is not None and tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {tier!r}; must be one of {sorted(TIER_ENUM)}"
            )

        tbl = self.db.open_table(RECORDS_TABLE)
        # Emptiness guard through the corpus-count cache (O(1) warm), never a
        # raw count_rows: a COUNT on the lilli engine re-scans every leaf
        # page, and this guard sat on the per-recall hot path. Zero active
        # records short-circuits identically — the ANN post-filter would
        # return nothing for that corpus anyway.
        if self.active_records_count() == 0:
            return []
        q = tbl.search(list(vec)).distance_type("cosine")
        where_clause = "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        if tier is not None:
            where_clause = f"tier = '{tier}' AND " + where_clause
        q = q.where(where_clause)
        # Over-fetch then trim: the ANN tier runs knn_query(k) FIRST and
        # applies the tombstone/pending predicate as a post-filter (tombstoned
        # vectors stay in the hnsw index until the next sleep-cycle rebuild),
        # so a k-sized fetch can silently underfill when live neighbors are
        # outnumbered by tombstoned ones in the raw top-k. Mirrors
        # query_similar_temporal's existing over-fetch discipline.
        k_effective = max(1, int(k) * max(1, int(over_fetch_factor)))
        q = q.limit(k_effective)
        if _lazy_decode:
            # Same query object feeds both the fast row-dict arm and the
            # pandas fallback arm below — one gate covers both.
            q = q.select_rank_view()

        out: list[tuple[MemoryRecord, float]] | None = None
        _slow_t0 = time.perf_counter()
        _fetch_ms: float | None = None
        _decode_ms: float | None = None
        _rows_fetched = 0
        if os.environ.get("IAI_MCP_ANN_FAST_DECODE_OFF") != "1":
            # The seed query decodes fetched rows directly, skipping the
            # pandas DataFrame round-trip _ann_to_pandas otherwise builds:
            # the raw cursor rows are fetched once and fed straight into
            # _from_row. The DataFrame materialization stays available (and
            # exercised, byte-for-byte) behind the kill-switch below.
            try:
                row_dicts = (
                    q.to_row_dicts(substage_timings=substage_timings)
                    if substage_timings is not None
                    else q.to_row_dicts()
                )
                _fetch_ms = (time.perf_counter() - _slow_t0) * 1000.0
                _rows_fetched = len(row_dicts)
                if substage_timings is not None:
                    substage_timings["rows_fetched"] = float(_rows_fetched)
                _decode_t0 = time.perf_counter()
                fast_out: list[tuple[MemoryRecord, float]] = []
                # Decode-then-stop at k valid rows -- never the whole
                # over-fetched pool. The tombstoned_at/embedding_pending
                # re-check is redundant with the SQL WHERE clause but
                # load-bearing if that predicate is ever not native-side;
                # a skipped row is never counted toward k, so the scan
                # tops up into the remaining fetched rows automatically.
                for row in row_dicts:
                    if row.get("tombstoned_at") is not None:
                        continue
                    if row.get("embedding_pending") not in (None, 0, False):
                        continue
                    record = _decode_row(row)
                    distance = float(row.get("_distance", 1.0)) if "_distance" in row else 1.0
                    score = 1.0 - distance
                    fast_out.append((record, score))
                    if len(fast_out) >= k:
                        break
                _decode_ms = (time.perf_counter() - _decode_t0) * 1000.0
                if substage_timings is not None:
                    substage_timings["escalation_decode_ms"] = (
                        substage_timings.get("escalation_decode_ms", 0.0) + _decode_ms
                    )
                out = fast_out
            except Exception as exc:  # noqa: BLE001 — fail-safe: never break recall
                logger.debug(
                    "query_similar fast decode branch failed, falling back to "
                    "the DataFrame materialization path: %s", exc,
                )
                out = None

        if out is None:
            results = q.to_pandas()
            out = []
            for _, row in results.iterrows():
                record = _decode_row(row.to_dict())
                distance = float(row.get("_distance", 1.0)) if "_distance" in row else 1.0
                score = 1.0 - distance
                out.append((record, score))

        _total_ms = (time.perf_counter() - _slow_t0) * 1000.0
        if _total_ms >= _slow_ann_log_threshold_ms():
            logger.warning(
                "slow ann seed query: total=%.0fms fetch=%s decode=%s "
                "rows=%d k_effective=%d",
                _total_ms,
                f"{_fetch_ms:.0f}ms" if _fetch_ms is not None else "n/a",
                f"{_decode_ms:.0f}ms" if _decode_ms is not None else "n/a",
                _rows_fetched,
                k_effective,
            )

        out = out[:k]
        if substage_timings is not None:
            substage_timings["rows_served"] = float(len(out))
        if _return_records_only:
            return [r for r, _s in out]
        return out

    def query_similar_temporal(
        self,
        vec: list[float] | None = None,
        *,
        as_of: str | None = None,
        k: int = 10,
        tier: str | None = None,
        over_fetch_factor: int = 3,
    ) -> list[tuple[MemoryRecord, float]]:
        """Time-bounded similarity search; sibling of query_similar.

        Four cue/as_of combinations:
          - vec given, as_of None: delegates to query_similar(vec, k, tier).
          - vec given, as_of given: over-fetch ANN candidates by k * over_fetch_factor,
            AND-append created_at <= as_of AND tombstoned_at IS NULL, trim to k.
          - vec None, as_of given: skip ANN; direct SQL SELECT ORDER BY created_at DESC.
          - vec None, as_of None: returns []. (Range-less, cue-less recall is a no-op.)

        as_of MUST be a canonical UTC ISO string from _normalize_ts_for_compare; the
        method does NOT re-normalize. Caller (dispatch branch) owns normalization.
        """
        if tier is not None and tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {tier!r}; must be one of {sorted(TIER_ENUM)}"
            )

        if vec is not None and as_of is None:
            return self.query_similar(list(vec), k=k, tier=tier)

        if vec is None and as_of is None:
            return []

        if vec is None and as_of is not None:
            # datetime() wrapping makes the comparison format-agnostic: both
            # T-form and space-form created_at values are normalized by SQLite
            # before the inequality, so same-date timestamps compare by actual
            # time rather than by the separator byte (0x20 vs 0x54).
            # Tombstone filter is T-relative: a record tombstoned after the
            # as_of point existed at T and must appear in the view.
            sql_where = [
                "(tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime(?))",
                "COALESCE(embedding_pending, 0) = 0",
                "datetime(created_at) <= datetime(?)",
            ]
            params: list = [as_of, as_of]
            if tier is not None:
                sql_where.append("tier = ?")
                params.append(tier)
            sql = (
                "SELECT"
                " id, tier, literal_surface, aaak_index,"
                " community_id, centrality, detail_level, pinned,"
                " stability, difficulty, last_reviewed, never_decay, never_merge,"
                " provenance_json, created_at, updated_at, tags_json, language,"
                " s5_trust_score, profile_modulation_gain_json, schema_version,"
                " hv_tier, structure_hv_payload,"
                " COALESCE(embedding_pending, 0) AS embedding_pending"
                " FROM records"
                f" WHERE {' AND '.join(sql_where)}"
                " ORDER BY datetime(created_at) DESC"
                " LIMIT ?"
            )
            params.append(int(k))
            with self.db._conn_lock:
                cursor = self.db._conn.execute(sql, params)
                rows = cursor.fetchall()
            out: list[tuple[MemoryRecord, float]] = []
            for row in rows:
                record = self._from_row(dict(row))
                out.append((record, 0.0))
            return out

        tbl = self.db.open_table(RECORDS_TABLE)
        if tbl.count_rows() == 0:
            return []
        k_effective = max(1, int(k) * max(1, int(over_fetch_factor)))
        q = tbl.search(list(vec)).distance_type("cosine")
        # Defensive re-normalization: ensure as_of is a valid ISO string before
        # interpolation into SQL. _normalize_ts_for_compare raises ValueError on
        # any non-ISO input, so an injected quote string would fail here.
        as_of = _normalize_ts_for_compare(as_of)
        # datetime() wrapping makes the comparison format-agnostic (T-form vs
        # space-form). Tombstone filter is T-relative: a record tombstoned after
        # the as_of point existed at T and must appear in the view.
        where_clause = (
            f"datetime(created_at) <= datetime('{as_of}')"
            f" AND (tombstoned_at IS NULL OR datetime(tombstoned_at) > datetime('{as_of}'))"
            " AND COALESCE(embedding_pending, 0) = 0"
        )
        if tier is not None:
            where_clause = f"tier = '{tier}' AND " + where_clause
        q = q.where(where_clause)
        results = q.limit(k_effective).to_pandas()
        out_hybrid: list[tuple[MemoryRecord, float]] = []
        for _, row in results.iterrows():
            record = self._from_row(row.to_dict())
            distance = float(row.get("_distance", 1.0)) if "_distance" in row else 1.0
            score = 1.0 - distance
            out_hybrid.append((record, score))
        return out_hybrid[:k]

    def pattern_separation_gate(
        self,
        record: MemoryRecord,
    ) -> tuple["GateAction", "GatePayload"]:
        action, payload, _hits = self._pattern_separation_gate_with_hits(record)
        return (action, payload)

    def _exact_scan_full_corpus(
        self, record: MemoryRecord,
    ) -> list[tuple[UUID, float]]:
        """Exact cosine top-k over the whole active corpus for ``record``.

        k is the full corpus size — the exact authority is the lossless
        backstop over the ANN's approximate top-k window, so k-truncation
        here would silently reintroduce the same tail-miss the backstop
        exists to close. The matrix is already resident (float32, ~1.5KB
        per row at d=384), so an argsort at any prod corpus size is a
        sub-ms hold under the exact-index lock. Write-path only:
        ``build_if_cold=True`` must never reach recall.
        """
        corpus_size = len(self._exact_index)
        k = max(corpus_size, 10)
        return self.exact_top_k(list(record.embedding), k=k, build_if_cold=True)

    def _exact_confirm_near_dup(
        self,
        record: MemoryRecord,
        candidate_id: UUID,
        threshold: float,
    ) -> bool:
        """Compare the exact-cosine authority's own score for ``candidate_id``
        against ``threshold``. Returns True iff a full exact scan carries
        evidence for ``candidate_id`` at cosine >= ``threshold``.

        Public helper. NOT on the write hot path: ``_exact_near_dup_target``
        already carries every candidate's exact score from its outer scan
        and decides SKIP-vs-INSERT directly, so calling this helper inside
        that loop would repeat the same full-corpus matmul + argsort per
        candidate. Kept as a standalone comparator for callers outside
        that walk (external confirms, ad-hoc checks) and as an armed
        forbidden-symbol on the recall-purity guard.
        """
        for exact_id, exact_cos in self._exact_scan_full_corpus(record):
            if exact_id == candidate_id:
                return exact_cos >= threshold
        return False

    def _exact_near_dup_target(
        self,
        record: MemoryRecord,
        threshold: float,
    ) -> tuple[UUID, float] | None:
        """The exact-cosine authority's own best SKIP-eligible near-dup for
        ``record``, or None if no exact candidate clears ``threshold`` while
        respecting ``never_merge`` on both sides.

        Runs a SINGLE full-corpus exact scan; the outer walk already knows
        each candidate's exact cosine (``exact_cos`` from the descending
        exact ranking IS the authority's own score — no rescan needed to
        re-derive it). The ANN top-k prefilter only decides WHETHER to
        consult the exact authority at all; the actual SKIP target comes
        from the exact ranking itself, which may find the true near-dup
        elsewhere in the ranking when the ANN's own top-1 is
        ``never_merge``-locked or when the true match sits outside the
        ANN's approximate window.
        """
        # exact score is the truth; ANN score is the prefilter
        # A budget-accepted directive must land its own row -- folding it
        # into an existing neighbour would silently drop the standing-order
        # intent, same as a never_merge lock.
        if getattr(record, "never_merge", False) or getattr(record, "directive", False):
            return None
        for exact_id, exact_cos in self._exact_scan_full_corpus(record):
            if exact_cos < threshold:
                break
            neighbour = self.get(exact_id)
            if neighbour is None:
                continue
            if getattr(neighbour, "never_merge", False):
                continue
            return (exact_id, float(exact_cos))
        return None

    def _pattern_separation_gate_with_hits(
        self,
        record: MemoryRecord,
    ) -> tuple["GateAction", "GatePayload", list[tuple[MemoryRecord, float]]]:
        from iai_mcp.daemon_config import _load_patsep_config
        cfg = _load_patsep_config()

        # Reset transient telemetry so a repeat-gate on the same record
        # instance cannot leak the prior pass's cosines into this event.
        record._psep_ann_prefilter_cos = None
        record._psep_exact_confirm_cos = None

        hits = self.query_similar(list(record.embedding), k=cfg.top_k)
        # Dedup is a WITHIN-tier operation: a semantic summary quoting its
        # episodic members sits at cos ~0.96 to them, and a cross-tier fold
        # would collapse knowledge into an episode (or vice versa) —
        # different tiers are different kinds of memory, never duplicates.
        hits = [
            (rec, cos) for rec, cos in hits if rec.tier == record.tier
        ]

        _ortho_enabled = os.environ.get(
            "IAI_MCP_ORTHO_ENABLED", "",
        ).lower() in {"1", "true"}
        if _ortho_enabled and hits:
            try:
                from iai_mcp.pattern_separation import orthogonalize_for_routing
                import numpy as _np
                neighbor_vecs = [r.embedding for r, _ in hits]
                routing_vec, _ortho_result = orthogonalize_for_routing(
                    list(record.embedding), neighbor_vecs, strength=0.3,
                )
                _rv = _np.asarray(routing_vec, dtype=_np.float32)
                _rv_norm = float(_np.linalg.norm(_rv))
                if _rv_norm > 1e-8:
                    _rv = _rv / _rv_norm
                    _new_hits: list[tuple[MemoryRecord, float]] = []
                    for _rec, _ in hits:
                        _ev = _np.asarray(
                            _rec.embedding, dtype=_np.float32,
                        )
                        _en = float(_np.linalg.norm(_ev))
                        if _en > 1e-8:
                            _ev = _ev / _en
                            _new_hits.append(
                                (_rec, float(_np.dot(_rv, _ev))),
                            )
                        else:
                            _new_hits.append((_rec, 0.0))
                    hits = _new_hits
            except Exception as _exc:  # noqa: BLE001 -- routing MUST NOT crash gate
                logger.debug(
                    "pattern_separation orthogonalize skipped: %s",
                    str(_exc)[:120],
                )

        _record_tags = list(getattr(record, "tags", None) or [])
        _is_conv = (
            record.tier == "episodic"
            and ("role:user" in _record_tags or "role:assistant" in _record_tags)
        )
        if _is_conv:
            _idem_tag_val: str | None = next(
                (t for t in _record_tags if t.startswith("idem:")), None
            )
            if _idem_tag_val is not None:
                _existing_id = self.find_record_by_tag(_idem_tag_val)
                if _existing_id is not None:
                    return (GateAction.SKIP, _existing_id, hits)
        else:
            if hits:
                top_record, top_cos = hits[0]
                if top_cos >= cfg.near_dup_threshold:
                    # ANN prefilter (approximate) only decides whether to
                    # consult the exact authority at all; the SKIP target is
                    # always re-derived from the full exact-cosine scan, since
                    # the ANN's own top-1 candidate may not be the exact
                    # authority's true near-dup (or may itself be
                    # never_merge-locked).
                    target = self._exact_near_dup_target(
                        record, cfg.near_dup_threshold,
                    )
                    record._psep_ann_prefilter_cos = float(top_cos)
                    record._psep_exact_confirm_cos = (
                        target[1] if target is not None else None
                    )
                    if target is not None:
                        return (GateAction.SKIP, target[0], hits)

        edges: list[tuple[UUID, float]] = []
        for rec, cos in hits:
            if cfg.link_threshold <= cos < cfg.near_dup_threshold:
                edges.append((rec.id, float(cos)))
        return (GateAction.INSERT, edges, hits)

    def update_record(self, record: MemoryRecord) -> None:
        tbl = self.db.open_table(RECORDS_TABLE)
        predicate = f"id = '{_uuid_literal(record.id)}'"
        tbl.update(
            where=predicate,
            values={
                "stability": float(record.stability),
                "difficulty": float(record.difficulty),
                "last_reviewed": record.last_reviewed,
                "updated_at": datetime.now(timezone.utc),
            },
        )


    def append_provenance(self, record_id: UUID, entry: dict) -> None:
        tbl = self.db.open_table(RECORDS_TABLE)
        predicate = f"id = '{_uuid_literal(record_id)}'"
        df = tbl.search().where(predicate).limit(1).to_pandas()
        if df.empty:
            return
        raw = df.iloc[0].get("provenance_json") or "[]"
        if is_encrypted(raw):
            raw = self._decrypt_for_record(record_id, raw)
        try:
            existing = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            existing = []
        existing.append(entry)
        new_json_plain = json.dumps(existing)
        new_json_ct = self._encrypt_for_record(record_id, new_json_plain)
        tbl.update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={
                "provenance_json": new_json_ct,
                "updated_at": datetime.now(timezone.utc),
            },
        )

    def append_provenance_batch(
        self, pairs: "list[tuple[UUID, dict]]",
        records_cache: "dict | None" = None,
    ) -> None:
        if not pairs:
            return
        tbl = self.db.open_table(RECORDS_TABLE)

        from collections import defaultdict
        grouped: dict[str, list[dict]] = defaultdict(list)
        for rid, entry in pairs:
            grouped[str(rid)].append(entry)

        now = datetime.now(timezone.utc)
        update_ids: list[str] = []
        update_prov: list[str] = []

        if records_cache is not None:
            for rid_str, entries in grouped.items():
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                try:
                    rec = records_cache.get(UUID(rid_str))
                except (TypeError, ValueError):
                    rec = None
                if rec is None:
                    rec = records_cache.get(rid_str)
                if rec is None:
                    continue
                existing = list(rec.provenance or [])
                existing.extend(entries)
                new_plain = json.dumps(existing)
                new_ct = self._encrypt_for_record(UUID(rid_str), new_plain)
                update_ids.append(canonical)
                update_prov.append(new_ct)
        else:
            # Read only the provenance rows for the ids being appended, never the
            # whole corpus. The previous full-table materialization decoded every
            # record's embedding just to look up a handful of provenance strings —
            # the dominant cost of provenance bookkeeping on the recall path.
            target_ids = list(grouped.keys())
            ph = ", ".join("?" for _ in target_ids)
            sql = (  # nosemgrep: sql-injection
                f"SELECT id, provenance_json FROM records WHERE id IN ({ph})"  # noqa: S608
            )
            with self.db._conn_lock:
                rows = self.db._conn.execute(sql, target_ids).fetchall()
            prov_by_id: dict[str, str] = {}
            for raw in rows:
                rid_key = str(raw[0] if hasattr(raw, "__getitem__") else raw["id"])
                prov_by_id[rid_key] = (
                    raw[1] if hasattr(raw, "__getitem__") else raw["provenance_json"]
                )
            for rid_str, entries in grouped.items():
                if rid_str not in prov_by_id:
                    continue
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                raw_prov = prov_by_id[rid_str] or "[]"
                if is_encrypted(raw_prov):
                    try:
                        raw_prov = self._decrypt_for_record(UUID(rid_str), raw_prov)
                    except (ValueError, OSError, TypeError) as exc:
                        logger.warning("provenance decrypt failed for %s: %s", rid_str, exc)
                        raw_prov = "[]"
                try:
                    existing = json.loads(raw_prov)
                except (TypeError, ValueError):
                    existing = []
                existing.extend(entries)
                new_plain = json.dumps(existing)
                new_ct = self._encrypt_for_record(UUID(rid_str), new_plain)
                update_ids.append(canonical)
                update_prov.append(new_ct)

        if not update_ids:
            return

        from iai_mcp.hippo import _schema
        update_tbl = _schema.table({
            "id": update_ids,
            "provenance_json": update_prov,
            "updated_at": [now] * len(update_ids),
        })
        try:
            tbl.merge_insert("id").when_matched_update_all().execute(update_tbl)
        except Exception as exc:  # noqa: BLE001 -- fallback gate, must stay broad
            logger.warning("provenance merge_insert fallback triggered: %s", exc, exc_info=True)
            for rid_str, new_json in zip(update_ids, update_prov):
                try:
                    tbl.update(
                        where=f"id = '{rid_str}'",
                        values={
                            "provenance_json": new_json,
                            "updated_at": now,
                        },
                    )
                except Exception as exc_inner:  # noqa: BLE001 -- per-row fallback continue
                    logger.debug("provenance per-row update failed for %s: %s", rid_str, exc_inner)
                    continue


    _RECORD_COLS = (
        "id, tier, literal_surface, aaak_index, embedding, structure_hv,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending,"
        " role, epistemic_status, salience_level, valence, directive"
    )

    # Soft-tombstoned rows (tombstoned_at IS NOT NULL) are dead and must never
    # surface as recency markers. Both authority branches filter them so the
    # SQL path agrees with the embedding-free warm path (which already excludes
    # tombstoned rows) — preserving byte-identity after a dedupe soft-tombstone.
    _PENDING_READ_SQL = (
        f"SELECT {_RECORD_COLS} FROM records"  # noqa: S608
        " WHERE embedding_pending = 1 AND tombstoned_at IS NULL"
        " ORDER BY rowid DESC LIMIT ?"
    )

    _ROLE_USER_READ_SQL = (
        f"SELECT {_RECORD_COLS} FROM records"  # noqa: S608
        " WHERE tier='episodic' AND role='user' AND tombstoned_at IS NULL"
        " ORDER BY rowid DESC LIMIT ?"
    )

    @staticmethod
    def _decode_raw_row(row: "dict") -> "dict":
        import numpy as _np
        emb_raw = row.get("embedding")
        # The stored embedding BLOB arrives as bytes on the stdlib sqlite3 driver
        # and as a zero-copy memoryview slice on the engine driver's projected
        # scan. np.frombuffer reads all three; admitting memoryview here keeps the
        # decoded vector at its true float length instead of leaking the raw byte
        # count downstream (which would mis-read a 384-d vector as 1536 elements).
        # Use .nbytes for the length guard: len(memoryview) returns element count,
        # which equals byte count only when itemsize==1; .nbytes is always bytes.
        if isinstance(emb_raw, (bytes, bytearray, memoryview)):
            # .nbytes for the length guard: len(memoryview) returns element count,
            # which equals byte count only when itemsize==1; .nbytes is always bytes.
            n = emb_raw.nbytes if isinstance(emb_raw, memoryview) else len(emb_raw)
            if n:
                row = dict(row)
                row["embedding"] = _np.frombuffer(emb_raw, dtype=_np.float32).tolist()
        return row

    def incident_edges(
        self,
        ids: "list[UUID]",
        edge_types: "list[str] | None" = None,
        top_k: "int | None" = 5,
        neighbor_keys_as_str: bool = False,
    ) -> "dict[UUID, list[tuple[UUID, str, float]]] | dict[UUID, list[tuple[str, str, float]]]":
        """Return the edges incident to every id in *ids*.

        Parameters
        ----------
        ids:
            Candidate node ids to look up.
        edge_types:
            Optional whitelist of edge-type strings (e.g. ``["hebbian"]``).
            ``None`` returns all edge types.
        top_k:
            Per-node neighbour cap.  ``None`` = uncapped.  When set, the
            result is ordered by weight descending, ties broken
            deterministically by neighbour id then edge type, before the
            cap is applied.
        neighbor_keys_as_str:
            When ``True``, neighbour ids in the returned tuples are plain
            strings instead of ``UUID`` objects.  This avoids constructing a
            ``UUID`` object for each neighbour on callers that only need a
            dict key or an immediate ``str()`` value (e.g. the degree-map and
            contradicts builds on the spread path).  The outer dict keys are
            always ``UUID`` objects regardless of this flag.
        """
        if not ids:
            return {}

        str_ids = [str(i) for i in ids]
        id_set = set(str_ids)

        ph = ", ".join("?" for _ in str_ids)
        sql = (  # nosemgrep: sql-injection
            f"SELECT src, dst, edge_type, weight FROM edges"  # noqa: S608
            f" WHERE (src IN ({ph}) OR dst IN ({ph}))"
        )
        params: list = str_ids + str_ids

        if edge_types is not None:
            et_ph = ", ".join("?" for _ in edge_types)
            sql += f" AND edge_type IN ({et_ph})"
            params += list(edge_types)

        with self.db.ro_conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        result: dict = {i: [] for i in ids}
        id_to_uuid: dict[str, UUID] = {str(i): i for i in ids}

        for row in rows:
            src_s = str(row[0] if hasattr(row, "__getitem__") else row["src"])
            dst_s = str(row[1] if hasattr(row, "__getitem__") else row["dst"])
            et = str(row[2] if hasattr(row, "__getitem__") else row["edge_type"])
            wt = float(row[3] if hasattr(row, "__getitem__") else row["weight"])

            if src_s in id_set:
                qid = id_to_uuid[src_s]
                if neighbor_keys_as_str:
                    # String fast path: validate with the canonical-UUID
                    # predicate (same acceptance as the UUID-object path for
                    # canonical keys) WITHOUT constructing a UUID object, so
                    # the per-edge UUID reconstruction is skipped on the hot
                    # degree-build path.
                    if _is_canonical_uuid_str(dst_s):
                        result[qid].append((dst_s, et, wt))
                else:
                    try:
                        neighbour = UUID(dst_s)
                    except (ValueError, AttributeError):
                        continue
                    result[qid].append((neighbour, et, wt))

            if dst_s in id_set and dst_s != src_s:
                qid = id_to_uuid[dst_s]
                if neighbor_keys_as_str:
                    if _is_canonical_uuid_str(src_s):
                        result[qid].append((src_s, et, wt))
                else:
                    try:
                        neighbour = UUID(src_s)
                    except (ValueError, AttributeError):
                        continue
                    result[qid].append((neighbour, et, wt))

        if top_k is not None:
            for uid, edges in result.items():
                edges.sort(key=lambda t: (-t[2], str(t[0]), t[1]))
                result[uid] = edges[:top_k]

        return result

    # Per-statement id ceiling for the batched fetch below: bounds the SQL
    # placeholder count per round trip while still collapsing a whole recall's
    # candidate set into a handful of statements.
    _GET_BATCH_CHUNK = 400

    def get_batch(
        self, ids: "list[UUID]", *, decode: str = "full", conn=None,
    ) -> "dict[UUID, MemoryRecord | RankCandidateView]":
        if not ids:
            return {}

        _lazy_decode = (
            decode == "rank"
            and os.environ.get("IAI_MCP_LAZY_DECODE_OFF") != "1"
        )
        _decode_row = self._from_row_rank_view if _lazy_decode else self._from_row

        # Dedup the input so a repeated id is fetched once, then resolve the
        # whole set through chunked `id IN (...)` statements on ONE borrowed
        # reader connection. The engine serves `id IN` through the same
        # id-index probes as a per-id equality, so the result set is
        # identical to a per-id loop — but a recall's hop expansion (a
        # hundred-plus candidates) costs a handful of round trips instead of
        # one connection borrow + one statement PER id.
        seen_ids: set[str] = set()
        uniq: list[str] = []
        for i in ids:
            id_str = str(i)
            if id_str not in seen_ids:
                seen_ids.add(id_str)
                uniq.append(id_str)

        out: "dict[UUID, MemoryRecord | RankCandidateView]" = {}
        skipped = 0

        def _run_chunks(_conn) -> None:
            nonlocal skipped
            for start in range(0, len(uniq), self._GET_BATCH_CHUNK):
                chunk = uniq[start : start + self._GET_BATCH_CHUNK]
                ph = ", ".join("?" for _ in chunk)
                sql = (  # nosemgrep: sql-injection
                    f"SELECT {self._RECORD_COLS} FROM records"  # noqa: S608
                    f" WHERE id IN ({ph})"
                )
                raw_rows = _conn.execute(sql, chunk).fetchall()
                for raw in raw_rows:
                    # _decode_raw_row must run before either decode fn: it is
                    # the only place the embedding BLOB is turned into a
                    # float list — skipping it silently mis-reads the vector.
                    row_dict = self._decode_raw_row(dict(raw))
                    try:
                        rec = _decode_row(row_dict)
                        out[rec.id] = rec
                    except Exception as exc:  # noqa: BLE001 — skip corrupt rows, never crash
                        skipped += 1
                        logger.debug(
                            "skipping undeserializable record %s: %s",
                            row_dict.get("id"),
                            type(exc).__name__,
                        )
                        continue

        # conn=None (default, every caller but the single-borrow auth-liveness
        # site): own borrow, unchanged. conn=<borrowed connection>: reuse the
        # caller's borrow for every chunk instead of opening a second one —
        # chunking (self._GET_BATCH_CHUNK) is preserved either way so the
        # lilli engine's IN-list cap is respected on both routes.
        if conn is not None:
            _run_chunks(conn)
        else:
            with self.db.ro_conn() as _owned_conn:
                _run_chunks(_owned_conn)
        if skipped:
            logger.warning(
                "skipped %d undeserializable record(s) in get_batch", skipped
            )
        return out

    def _recency_marker_to_record(self, marker) -> "MemoryRecord":
        """Build a lightweight MemoryRecord from a RecencyMarker for the buffer read path.

        Only populates the fields the recall pipeline consumer reads:
        ``id``, ``literal_surface``, ``created_at``, ``provenance`` (for
        ``session_id``), ``embedding_pending``, ``role``.  All other fields
        carry neutral defaults.  This record must NOT be written back to the
        store or used outside the recency read path.
        """
        from uuid import UUID as _UUID
        prov: list[dict] = []
        if marker.session_id is not None:
            prov = [{"session_id": marker.session_id}]
        return MemoryRecord(
            id=_UUID(marker.id),
            tier="episodic",
            literal_surface=marker.literal_surface,
            aaak_index="",
            embedding=[],
            community_id=None,
            centrality=0.0,
            detail_level=1,
            pinned=False,
            stability=0.0,
            difficulty=0.0,
            last_reviewed=None,
            never_decay=False,
            never_merge=False,
            provenance=prov,
            created_at=marker.created_at or datetime.min.replace(tzinfo=timezone.utc),
            updated_at=marker.created_at or datetime.min.replace(tzinfo=timezone.utc),
            language="en",
            embedding_pending=marker.embedding_pending,
            role=marker.role,
        )

    def _recent_pending_markers_sql(self, n: int) -> "list[MemoryRecord]":
        """SQL union path — the lossless-authority fallback for recent_pending_markers.

        Returns the same ordered list that the warm buffer path returns, by
        reading directly from the store.  Called when the buffer is not warm or
        on any buffer error.  The result is always correct; it is slow at scale
        because it decodes every candidate row including its embedding BLOB.
        """
        seen: dict[UUID, "MemoryRecord"] = {}
        skipped = 0

        with self.db._conn_lock:
            rows_a = self.db._conn.execute(
                self._PENDING_READ_SQL, (n,)
            ).fetchall()

        for raw in rows_a:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
                if rec.id not in seen:
                    seen[rec.id] = rec
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.debug(
                    "skipping undeserializable record %s: %s",
                    row_dict.get("id"),
                    type(exc).__name__,
                )
                continue

        # Over-fetch by rowid before the created_at re-rank below: the final
        # ordering is by created_at, so a wider rowid window must be gathered to
        # surface the true most-recent-by-created_at turns when rowid order and
        # created_at order disagree (post-reembed / bulk-copy re-keying). The
        # indexed role='user' predicate makes this wide gather cheap.
        over_fetch = n * 4
        with self.db._conn_lock:
            rows_b = self.db._conn.execute(
                self._ROLE_USER_READ_SQL, (over_fetch,)
            ).fetchall()

        for raw in rows_b:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
            except Exception as exc:  # noqa: BLE001
                skipped += 1
                logger.debug(
                    "skipping undeserializable record %s: %s",
                    row_dict.get("id"),
                    type(exc).__name__,
                )
                continue
            if rec.id not in seen:
                seen[rec.id] = rec

        if skipped:
            logger.warning(
                "skipped %d undeserializable record(s) in _recent_pending_markers_sql",
                skipped,
            )
        candidates = list(seen.values())
        candidates.sort(
            key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return candidates[:n]

    def recent_pending_markers(self, n: int = 50) -> "list[MemoryRecord]":
        # Fast path: serve from the in-process recency buffer when it is warm.
        # The buffer holds already-decrypted, embedding-free markers in RAM,
        # so this path decodes nothing from the store and returns immediately.
        #
        # Lazy warm-on-first-read: if the buffer has not been warmed yet, warm
        # it once here so the first recall after a store open benefits from the
        # buffer without requiring an explicit warm call at every open site.
        buf = self._recency_buffer
        if not buf.is_warm:
            # Single-flight, double-checked: only one thread warms while others
            # wait on the lock; they then see is_warm True and skip re-warming.
            # warm_recency_buffer swaps the buffer atomically (replace_all), so a
            # waiter that proceeds to markers() never observes a torn buffer.
            with self._recency_warm_lock:
                if not buf.is_warm:
                    try:
                        self.warm_recency_buffer()
                    except Exception as exc:  # noqa: BLE001 -- warm failure falls to SQL
                        logger.debug("recency_buffer lazy warm failed, using SQL: %s", exc)
                        return self._recent_pending_markers_sql(n)

        # Buffer is warm (either was already, or just became warm above).
        # Serve from RAM: markers(n) returns [] for an empty corpus, which is correct.
        try:
            return [self._recency_marker_to_record(m) for m in buf.markers(n)]
        except Exception as exc:  # noqa: BLE001 -- hydration error → SQL fallback
            logger.debug("recency_buffer serve failed, falling back to SQL: %s", exc)
            return self._recent_pending_markers_sql(n)


    def boost_edges(
        self,
        pairs: list[tuple[UUID, UUID]],
        delta: float | Sequence[float] = 0.1,
        edge_type: str = "hebbian",
    ) -> dict[tuple[str, str], float]:
        if edge_type not in EDGE_TYPES:
            raise ValueError(
                f"invalid edge_type {edge_type!r}; must be one of {sorted(EDGE_TYPES)}"
            )

        if isinstance(delta, (int, float)):
            deltas = [float(delta)] * len(pairs)
        else:
            deltas = [float(d) for d in delta]
            if len(deltas) != len(pairs):
                raise ValueError(
                    f"deltas length {len(deltas)} != pairs length {len(pairs)}"
                )

        if not pairs:
            return {}

        coalesced: dict[tuple[str, str], float] = {}
        for (a, b), d in zip(pairs, deltas):
            key = (str(a), str(b))
            canonical = tuple(sorted(key))
            coalesced[canonical] = coalesced.get(canonical, 0.0) + d
        if not coalesced:
            return {}

        tbl = self.db.open_table(EDGES_TABLE)

        update_rows: list[dict] = []
        insert_rows: list[dict] = []
        new_weights: dict[tuple[str, str], float] = {}
        now = datetime.now(timezone.utc)

        # Edge reads here go through the WRITER connection, never the RO pool:
        # this is a read-modify-write — an RO snapshot can miss an edge this
        # same call graph just committed, and a write-path verb must never
        # queue behind (or wedge on) RO-slot opens.
        if len(coalesced) <= BOOST_EDGES_SMALL_BATCH:
            for (src_str, dst_str), accum_delta in coalesced.items():
                with self.db._conn_lock:
                    row = self.db._conn.execute(
                        f"SELECT weight FROM {EDGES_TABLE}"
                        " WHERE src = ? AND dst = ? AND edge_type = ? LIMIT 1",
                        (src_str, dst_str, edge_type),
                    ).fetchone()
                if row is not None:
                    cur = float(row["weight"])
                    nw = cur + accum_delta
                    update_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                else:
                    nw = accum_delta
                    insert_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                new_weights[(src_str, dst_str)] = nw
        else:
            with self.db._conn_lock:
                _edge_rows = self.db._conn.execute(
                    f"SELECT src, dst, weight FROM {EDGES_TABLE}"
                    " WHERE edge_type = ?",
                    (edge_type,),
                ).fetchall()
            # Build a single (src, dst) -> weight lookup over the rows of this
            # edge_type ONCE, then resolve each pair in O(1). The edges table's
            # PRIMARY KEY (src, dst, edge_type) makes the mapping unambiguous —
            # at most one row per key — so the dict is exact, not lossy. This
            # replaces a per-pair full-table boolean mask (O(pairs x edges)) with
            # O(pairs + edges): on a large boost it is the difference between
            # minutes and milliseconds.
            existing_weights: dict[tuple[str, str], float] = {
                (r["src"], r["dst"]): float(r["weight"]) for r in _edge_rows
            }
            for (src_str, dst_str), accum_delta in coalesced.items():
                cur = existing_weights.get((src_str, dst_str))
                if cur is not None:
                    nw = cur + accum_delta
                    update_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                else:
                    nw = accum_delta
                    insert_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                new_weights[(src_str, dst_str)] = nw

        if update_rows:
            try:
                upd_arrow = _schema.Table.from_pylist(
                    update_rows,
                    schema=_schema.schema(
                        [
                            ("src", _schema.string()),
                            ("dst", _schema.string()),
                            ("edge_type", _schema.string()),
                            ("weight", _schema.float32()),
                            ("updated_at", _schema.timestamp("us", tz="UTC")),
                        ]
                    ),
                )
                _WRITE_RETRYABLE_SIGNALS = (
                    "retryable commit conflict",
                    "too many concurrent writers",
                )
                _WRITE_MAX_RETRIES = 2
                for _w_attempt in range(_WRITE_MAX_RETRIES + 1):
                    try:
                        (
                            tbl.merge_insert(["src", "dst", "edge_type"])
                            .when_matched_update_all()
                            .execute(upd_arrow)
                        )
                        break
                    except (RuntimeError, OSError) as _w_exc:
                        _w_msg = str(_w_exc).lower()
                        if (
                            any(sig in _w_msg for sig in _WRITE_RETRYABLE_SIGNALS)
                            and _w_attempt < _WRITE_MAX_RETRIES
                        ):
                            time.sleep(0.050 + random.uniform(0, 0.050))
                            try:
                                tbl = self.db.open_table(EDGES_TABLE)
                            except (OSError, RuntimeError, ValueError):
                                pass
                        else:
                            raise
            except Exception as exc:  # noqa: BLE001 -- fallback gate, must stay broad
                logger.warning("edge merge_insert fallback triggered: %s", exc, exc_info=True)
                for r in update_rows:
                    tbl.update(
                        where=(
                            f"src = '{_uuid_literal(r['src'])}' "
                            f"AND dst = '{_uuid_literal(r['dst'])}' "
                            f"AND edge_type = '{edge_type}'"
                        ),
                        values={
                            "weight": r["weight"],
                            "updated_at": r["updated_at"],
                        },
                    )

        if insert_rows:
            buf = _edge_buffer.setdefault(id(self), [])
            buf.extend(insert_rows)
            if should_flush_edge_buffer(id(self)):
                flush_edge_buffer(self)

        return new_weights

    def reinforce_record(
        self,
        record_id: UUID,
        anchor_id: UUID | None = None,
        edge_type: str = "hebbian",
        delta: float = 0.1,
        *,
        is_retrieval: bool = False,
    ) -> dict[tuple[str, str], float]:
        if anchor_id is None:
            pair = (record_id, record_id)
        else:
            pair = (anchor_id, record_id)
        result = self.boost_edges([pair], delta=delta, edge_type=edge_type)
        if is_retrieval:
            # Must not depend on the reconsolidation dry-run flag or config load.
            now = _utc_now()
            values: dict[str, object] = {"last_reviewed": now}
            try:
                from iai_mcp.daemon_config import _load_reconsolidation_config
                cfg = _load_reconsolidation_config()
                if not cfg.dry_run:
                    values["labile_until"] = now + timedelta(
                        seconds=cfg.labile_window_sec
                    )
            except (ImportError, ValueError) as exc:
                logger.debug("reconsolidation config unavailable, skipped: %s", exc)
            from iai_mcp.errors import DatabaseError

            tbl = self.db.open_table(RECORDS_TABLE)
            try:
                tbl.update(
                    where=f"id = '{_uuid_literal(record_id)}'",
                    values=values,
                )
            except (RuntimeError, ValueError, OSError, KeyError, DatabaseError) as exc:
                msg = str(exc).lower()
                column_missing = (
                    "last_reviewed" in msg
                    or "labile_until" in msg
                    or "no such column" in msg
                    or ("column" in msg and "not found" in msg)
                )
                if not column_missing:
                    raise
                logger.debug("records column missing, skipped: %s", exc)
        return result

    def raise_salience_level_if_higher(self, record_id: UUID, level: str) -> bool:
        """Monotone raise, scoped to exactly one column -- never reads,
        compares, or writes pinned or never_merge. An invalid level is a
        no-op (never a lower-than-intended write from coerced garbage)."""
        if level not in SALIENCE_LEVEL_ENUM:
            return False
        flush_record_buffer(self)
        current = self.get(record_id)
        if current is None:
            return False
        if SALIENCE_LEVEL_RANK.get(level, 0) <= SALIENCE_LEVEL_RANK.get(
            current.salience_level, 0
        ):
            return False
        tbl = self.db.open_table(RECORDS_TABLE)
        tbl.update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={"salience_level": level},
        )
        # Mutate the already-fetched object (never a cache-shared instance --
        # `current` came fresh off `self.get()` above) and route through the
        # registered hook, mirroring upgrade_tier's identical pattern: this
        # bumps graph._pool_content_version and feeds the rank index as a
        # side effect of the normal write-time sync, no bespoke sync point.
        current.salience_level = level
        self._fire_graph_sync_hook("update", current)
        return True

    def raise_valence(self, record_id: UUID, new_value: float) -> bool:
        """Monotone raise, scoped to exactly one column. Clamped to [0.0, 1.0]
        at the write boundary (the only path that sets this column) so a
        poisoned caller value can never reach the rank multiplier unbounded."""
        if os.environ.get("IAI_MCP_VALENCE_WRITE_OFF") == "1":
            return False
        try:
            _raw = float(new_value)
        except (TypeError, ValueError):
            return False
        clamped = 0.0 if _raw != _raw else max(0.0, min(1.0, _raw))
        flush_record_buffer(self)
        current = self.get(record_id)
        if current is None:
            return False
        if clamped <= current.valence:
            return False
        tbl = self.db.open_table(RECORDS_TABLE)
        tbl.update(
            where=f"id = '{_uuid_literal(record_id)}'",
            values={"valence": clamped},
        )
        # Mutate the already-fetched object and route through the registered
        # hook, mirroring raise_salience_level_if_higher's identical pattern:
        # bumps graph._pool_content_version and feeds the rank index as a
        # side effect of the normal write-time sync, no bespoke sync point.
        current.valence = clamped
        self._fire_graph_sync_hook("update", current)
        return True

    def upgrade_tier(
        self,
        record_id: UUID,
        new_tier: str,
        *,
        trigger_event_type: str,
        dry_run: bool = False,
    ) -> bool:
        record = self.get(record_id)
        if record is None:
            return False
        current_tier = record.tier
        if new_tier not in _STC_TIER_ORDER:
            raise ValueError(
                f"upgrade_tier: invalid new_tier {new_tier!r}, "
                f"expected one of {set(_STC_TIER_ORDER.keys())}"
            )
        if _STC_TIER_ORDER[new_tier] <= _STC_TIER_ORDER[current_tier]:
            raise ValueError(
                f"upgrade_tier: refusing non-upgrade "
                f"{current_tier!r} -> {new_tier!r} (never downgrade)"
            )

        if not dry_run:
            tbl = self.db.open_table(RECORDS_TABLE)
            try:
                tbl.update(
                    where=f"id = '{_uuid_literal(record_id)}'",
                    values={
                        "tier": new_tier,
                        "updated_at": datetime.now(timezone.utc),
                    },
                )
            except (RuntimeError, ValueError, OSError, KeyError) as exc:
                msg = str(exc).lower()
                column_missing = (
                    "tier" in msg
                    and (
                        "no such column" in msg
                        or ("column" in msg and "not found" in msg)
                    )
                )
                if not column_missing:
                    raise
                logger.debug("tier column missing in upgrade_tier, skipped: %s", exc)
            else:
                record.tier = new_tier
                self._fire_graph_sync_hook("update", record)

        from iai_mcp.events import write_event
        write_event(
            self,
            "stc_upgrade_pass",
            {
                "record_id": str(record_id),
                "from_tier": current_tier,
                "to_tier": new_tier,
                "trigger_event_type": trigger_event_type,
                "dry_run_mode": dry_run,
            },
            severity="info",
            source_ids=[record_id],
        )
        return True

    def add_contradicts_edge(self, original: UUID, new_id: UUID) -> None:
        flush_record_buffer(self)
        row = {
            "src": str(original),
            "dst": str(new_id),
            "edge_type": "contradicts",
            "weight": 1.0,
            "updated_at": datetime.now(timezone.utc),
        }
        _edge_buffer.setdefault(id(self), []).append(row)
        flush_edge_buffer(self)


    def _to_row(self, r: MemoryRecord) -> dict:
        literal_ct = self._encrypt_for_record(r.id, r.literal_surface)
        provenance_plain = json.dumps(r.provenance)
        provenance_ct = self._encrypt_for_record(r.id, provenance_plain)
        gain_plain = json.dumps(r.profile_modulation_gain or {})
        gain_ct = self._encrypt_for_record(r.id, gain_plain)
        return {
            "id": str(r.id),
            "tier": r.tier,
            "literal_surface": literal_ct,
            "aaak_index": r.aaak_index,
            "embedding": [float(x) for x in r.embedding],
            "structure_hv": bytes(r.structure_hv or b""),
            "community_id": str(r.community_id) if r.community_id else "",
            "centrality": float(r.centrality),
            "detail_level": int(r.detail_level),
            "pinned": bool(r.pinned),
            "stability": float(r.stability),
            "difficulty": float(r.difficulty),
            "last_reviewed": r.last_reviewed,
            "never_decay": bool(r.never_decay),
            "never_merge": bool(r.never_merge),
            "provenance_json": provenance_ct,
            "created_at": r.created_at,
            "updated_at": r.updated_at,
            "tags_json": json.dumps(r.tags),
            "language": str(r.language),
            "s5_trust_score": float(r.s5_trust_score),
            "profile_modulation_gain_json": gain_ct,
            "schema_version": int(r.schema_version),
            "schema_bypass": bool(getattr(r, "_schema_bypass", False)),
            "labile_until": getattr(r, "_labile_until", None),
            "wing": getattr(r, "wing", None),
            "room": getattr(r, "room", None),
            "drawer": getattr(r, "drawer", None),
            "hv_tier": r.hv_tier,
            "structure_hv_payload": bytes(r.structure_hv_payload or b""),
            "role": _derive_role(r.tags),
            "epistemic_status": r.epistemic_status,
            "salience_level": r.salience_level,
            "valence": float(r.valence),
            # A freshly-inserted record is always live by construction — no
            # write path through _to_row ever carries a pre-set tombstone.
            "live": _derive_live(None),
            "directive": bool(r.directive),
        }

    def _maybe_tag_schema_bypass(self, record: MemoryRecord) -> None:
        max_cos: float = 0.0
        tagged: bool = False
        dry_run: bool = False
        try:
            from iai_mcp.daemon_config import _load_reconsolidation_config
            cfg = _load_reconsolidation_config()
            dry_run = bool(cfg.dry_run)
            centroids: dict[Any, list[float]] = {}
            try:
                from iai_mcp import runtime_graph_cache
                cached = runtime_graph_cache.try_load(self)
                if cached is not None:
                    assignment = cached[0]
                    centroids = (
                        getattr(assignment, "community_centroids", {}) or {}
                    )
            except (OSError, ValueError, ImportError, RuntimeError) as exc:
                logger.debug("schema-bypass centroid load skipped: %s", exc)
                centroids = {}
            if centroids:
                emb = record.embedding
                emb_dim = len(emb)
                import numpy as _np
                e_arr = _np.asarray(emb, dtype=_np.float32)
                e_norm = float(_np.linalg.norm(e_arr))
                if e_norm > 0.0:
                    for cent in centroids.values():
                        if cent is None or len(cent) != emb_dim:
                            continue
                        c_arr = _np.asarray(cent, dtype=_np.float32)
                        c_norm = float(_np.linalg.norm(c_arr))
                        if c_norm <= 0.0:
                            continue
                        sim = float(
                            _np.dot(e_arr, c_arr) / (e_norm * c_norm)
                        )
                        if sim > max_cos:
                            max_cos = sim
                if max_cos >= float(cfg.schema_bypass_cos_threshold):
                    tagged = True
            if tagged and not dry_run:
                record._schema_bypass = True
            else:
                pass
        except Exception as exc:  # noqa: BLE001 -- advisory tagger, must never abort insert
            logger.warning("schema-bypass tagging failed (advisory): %s", exc, exc_info=True)
            return
        try:
            from iai_mcp.events import write_event
            write_event(
                self,
                "schema_bypass_pass",
                {
                    "record_id": str(record.id),
                    "max_cos": float(max_cos),
                    "tagged": bool(tagged and not dry_run),
                    "dry_run_mode": bool(dry_run),
                },
                severity="info",
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.debug("schema_bypass_pass event emit failed: %s", exc)

    def _maybe_spatial_tag(self, record: MemoryRecord) -> None:
        try:
            from iai_mcp.daemon_config import _load_spatial_config
            config = _load_spatial_config()
        except Exception as exc:  # noqa: BLE001 -- advisory tagger, must never abort insert
            logger.warning("spatial config load failed (advisory): %s", exc, exc_info=True)
            return
        if not config.auto_tag:
            return

        existing_wing = getattr(record, "wing", None)
        existing_room = getattr(record, "room", None)
        existing_drawer = getattr(record, "drawer", None)
        if (
            existing_wing is not None
            or existing_room is not None
            or existing_drawer is not None
        ):
            return

        source_path: str | None = None
        try:
            prov = getattr(record, "provenance", None)
            if isinstance(prov, list):
                for entry in prov:
                    if isinstance(entry, dict) and "source_path" in entry:
                        candidate = entry.get("source_path")
                        if isinstance(candidate, str) and candidate.strip():
                            source_path = candidate
                            break
        except (TypeError, ValueError, AttributeError) as exc:
            logger.debug("spatial source_path extraction skipped: %s", exc)
            source_path = None

        wing: str | None = None
        room: str | None = None
        drawer: str | None = None
        try:
            from iai_mcp.spatial_tagger import SpatialTagger
            wing, room, drawer = SpatialTagger.tag(
                record,
                source_path,
                default_wing=config.default_wing,
            )
        except Exception as exc:  # noqa: BLE001 -- advisory tagger, must never abort insert
            logger.warning("spatial tagger inference failed (advisory): %s", exc, exc_info=True)
            wing, room, drawer = (None, None, None)

        if not config.dry_run:
            record.wing = wing
            record.room = room
            record.drawer = drawer

        try:
            from iai_mcp.events import write_event
            write_event(
                self,
                "spatial_tag_pass",
                {
                    "record_id": str(record.id),
                    "wing": wing,
                    "room": room,
                    "drawer": drawer,
                    "source_path": source_path,
                    "dry_run_mode": bool(config.dry_run),
                },
                severity="info",
            )
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            logger.debug("spatial_tag_pass event emit failed: %s", exc)

    def _from_row(self, row: dict) -> MemoryRecord:
        from uuid import UUID as _UUID

        _parse_ts = _parse_ts_field

        def _safe_int(val: Any, default: int) -> int:
            if val is None:
                return default
            try:
                fval = float(val)
                if fval != fval:
                    return default
                return int(fval)
            except (TypeError, ValueError):
                return default

        if "id" not in row:
            raise KeyError(
                "iter_records consumer must include 'id' in column projection"
            )

        structure_raw = row.get("structure_hv")
        if structure_raw is None:
            structure_hv = b""
        elif isinstance(structure_raw, (bytes, bytearray, memoryview)):
            structure_hv = bytes(structure_raw)
        else:
            structure_hv = b""

        from iai_mcp import events as _ev_mod
        _codec_event_kind = getattr(
            _ev_mod,
            "TELEMETRY_CODEC_MARKER_MISSING",
            "codec_marker_missing",
        )
        hv_tier_raw = row.get("hv_tier")
        structure_hv_payload_raw = row.get("structure_hv_payload")
        _codec_reason: str | None = None

        if hv_tier_raw is None:
            hv_tier = "bsc"
            structure_hv_payload = b""
        elif hv_tier_raw not in HV_TIER_ENUM:
            _codec_reason = f"hv_tier {hv_tier_raw!r} not in HV_TIER_ENUM; reset to bsc"
            hv_tier = "bsc"
            structure_hv_payload = b""
        elif structure_hv_payload_raw is not None and not isinstance(
            structure_hv_payload_raw, (bytes, bytearray, memoryview)
        ):
            _codec_reason = (
                f"structure_hv_payload expected bytes, "
                f"got {type(structure_hv_payload_raw).__name__}"
            )
            hv_tier = "bsc"
            structure_hv_payload = b""
        else:
            hv_tier = str(hv_tier_raw)
            structure_hv_payload = (
                bytes(structure_hv_payload_raw)
                if isinstance(structure_hv_payload_raw, (bytes, bytearray, memoryview))
                else b""
            )

        if _codec_reason is not None:
            try:
                _ev_mod.write_event(
                    self,
                    kind=_codec_event_kind,
                    data={
                        "record_id": row.get("id", ""),
                        "reason": _codec_reason,
                    },
                    severity="warning",
                )
            except Exception:  # noqa: BLE001 — telemetry must never crash _from_row
                pass

        _community_val = row.get("community_id")
        try:
            import math as _math
            if _community_val is not None and not isinstance(_community_val, str):
                if _math.isnan(float(_community_val)):
                    _community_val = None
        except (TypeError, ValueError):
            pass
        community_raw = (_community_val or "")
        community_id = _UUID(community_raw) if community_raw and isinstance(community_raw, str) else None

        lang_raw = row.get("language")
        raw_version = row.get("schema_version")
        try:
            version_int = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
        except (TypeError, ValueError):
            version_int = SCHEMA_VERSION_CURRENT
        schema_version = version_int

        is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
        if is_empty_language and schema_version == 1:
            language = "__LEGACY_EMPTY__"
        elif is_empty_language:
            language = "en"
        else:
            language = str(lang_raw)

        s5_raw = row.get("s5_trust_score")
        try:
            _s5 = float(s5_raw) if s5_raw is not None else 0.5
            s5_trust_score = _s5 if (_s5 == _s5 and 0.0 <= _s5 <= 1.0) else 0.5
        except (TypeError, ValueError):
            s5_trust_score = 0.5

        from uuid import UUID as _UUID2
        _row_uuid = _UUID2(row["id"])
        gain_raw = row.get("profile_modulation_gain_json") or "{}"
        if is_encrypted(gain_raw):
            gain_raw = self._decrypt_for_record(_row_uuid, gain_raw)
        try:
            profile_modulation_gain = json.loads(gain_raw) or {}
        except (TypeError, json.JSONDecodeError):
            profile_modulation_gain = {}

        last_reviewed = _parse_ts(row.get("last_reviewed"))

        row_uuid = _UUID(row["id"])
        literal_raw = row.get("literal_surface", "")
        if is_encrypted(literal_raw):
            literal_raw = self._decrypt_for_record(row_uuid, literal_raw)
        provenance_raw = row.get("provenance_json") or "[]"
        if is_encrypted(provenance_raw):
            provenance_raw = self._decrypt_for_record(row_uuid, provenance_raw)
        try:
            provenance_list = json.loads(provenance_raw) if provenance_raw else []
        except (TypeError, json.JSONDecodeError):
            provenance_list = []

        role_raw = row.get("role")
        role = str(role_raw) if role_raw is not None else None

        epistemic_status_raw = row.get("epistemic_status")
        if epistemic_status_raw is None or epistemic_status_raw not in EPISTEMIC_STATUS_ENUM:
            epistemic_status = "unknown"
        else:
            epistemic_status = str(epistemic_status_raw)

        salience_level_raw = row.get("salience_level")
        if salience_level_raw is None or salience_level_raw not in SALIENCE_LEVEL_ENUM:
            salience_level = "unflagged"
        else:
            salience_level = str(salience_level_raw)

        valence_raw = row.get("valence")
        try:
            _valence = float(valence_raw) if valence_raw is not None else 0.0
            valence = _valence if (_valence == _valence and 0.0 <= _valence <= 1.0) else 0.0
        except (TypeError, ValueError):
            valence = 0.0

        directive = bool(row.get("directive") or False)

        rec = MemoryRecord(
            id=row_uuid,
            tier=row.get("tier", "episodic"),
            literal_surface=literal_raw,
            aaak_index=row.get("aaak_index") or "",
            embedding=(
                list(row["embedding"])
                if row.get("embedding") is not None
                else []
            ),
            community_id=community_id,
            centrality=float(row.get("centrality", 0.0) or 0.0),
            detail_level=_safe_int(row.get("detail_level"), 1),
            pinned=bool(row.get("pinned", False) or False),
            stability=float(row.get("stability") or 0.0),
            difficulty=float(row.get("difficulty") or 0.0),
            last_reviewed=last_reviewed,
            never_decay=bool(row.get("never_decay", False) or False),
            never_merge=bool(row.get("never_merge", False) or False),
            provenance=provenance_list,
            created_at=_parse_ts(row.get("created_at")) or datetime.now(timezone.utc),
            updated_at=_parse_ts(row.get("updated_at")) or datetime.now(timezone.utc),
            tags=json.loads((row.get("tags_json") or "[]") if isinstance(row.get("tags_json"), str) else "[]"),
            language=language,
            s5_trust_score=s5_trust_score,
            profile_modulation_gain=profile_modulation_gain,
            schema_version=schema_version,
            structure_hv=structure_hv,
            hv_tier=hv_tier,
            structure_hv_payload=structure_hv_payload,
            embedding_pending=_safe_int(row.get("embedding_pending"), 0),
            role=role,
            epistemic_status=epistemic_status,
            salience_level=salience_level,
            valence=valence,
            directive=directive,
        )
        if language == "__LEGACY_EMPTY__":
            rec.language = ""
        return rec

    def _from_row_rank_view(self, row: dict) -> RankCandidateView:
        if "id" not in row:
            raise KeyError(
                "query_similar/get_batch rank-view decode requires 'id' in "
                "the column projection"
            )
        row_uuid = UUID(row["id"])

        structure_raw = row.get("structure_hv")
        if isinstance(structure_raw, (bytes, bytearray, memoryview)):
            structure_hv = bytes(structure_raw)
        else:
            structure_hv = b""

        _community_val = row.get("community_id")
        try:
            import math as _math
            if _community_val is not None and not isinstance(_community_val, str):
                if _math.isnan(float(_community_val)):
                    _community_val = None
        except (TypeError, ValueError):
            pass
        community_raw = (_community_val or "")
        community_id = (
            UUID(community_raw)
            if community_raw and isinstance(community_raw, str)
            else None
        )

        literal_raw = row.get("literal_surface", "")
        if is_encrypted(literal_raw):
            literal_raw = self._decrypt_for_record(row_uuid, literal_raw)

        tags_json_raw = row.get("tags_json")
        tags = json.loads(
            (tags_json_raw or "[]") if isinstance(tags_json_raw, str) else "[]"
        )

        salience_level_raw = row.get("salience_level")
        if salience_level_raw is None or salience_level_raw not in SALIENCE_LEVEL_ENUM:
            salience_level = "unflagged"
        else:
            salience_level = str(salience_level_raw)

        valence_raw = row.get("valence")
        try:
            _valence = float(valence_raw) if valence_raw is not None else 0.0
            valence = _valence if (_valence == _valence and 0.0 <= _valence <= 1.0) else 0.0
        except (TypeError, ValueError):
            valence = 0.0

        directive = bool(row.get("directive") or False)

        # Must stay identical to _from_row's legacy-schema-v1 language
        # handling -- row already carries schema_version at zero extra cost
        # (part of the shared _RECORD_COLS projection used by both decode
        # tiers), so there is no reason for this tier to diverge.
        lang_raw = row.get("language")
        raw_version = row.get("schema_version")
        try:
            schema_version = int(raw_version) if raw_version is not None else SCHEMA_VERSION_CURRENT
        except (TypeError, ValueError):
            schema_version = SCHEMA_VERSION_CURRENT
        is_empty_language = lang_raw is None or (isinstance(lang_raw, str) and lang_raw == "")
        if is_empty_language and schema_version == 1:
            language = "__LEGACY_EMPTY__"
        elif is_empty_language:
            language = "en"
        else:
            language = str(lang_raw)

        rv = RankCandidateView(
            id=row_uuid,
            embedding=(
                list(row["embedding"]) if row.get("embedding") is not None else []
            ),
            literal_surface=literal_raw,
            aaak_index=row.get("aaak_index") or "",
            created_at=_parse_ts_field(row.get("created_at")) or _utc_now(),
            stability=float(row.get("stability") or 0.0),
            tier=row.get("tier", "episodic"),
            tags=tags,
            language=language,
            community_id=community_id,
            structure_hv=structure_hv,
            salience_level=salience_level,
            valence=valence,
            directive=directive,
        )
        if language == "__LEGACY_EMPTY__":
            rv.language = ""
        return rv
