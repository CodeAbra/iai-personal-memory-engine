from __future__ import annotations

import asyncio
import base64
import functools
import json
import os
import random
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Callable
from uuid import UUID

import logging

from iai_mcp.hippo import _REAL_IAI_ROOT, AccessMode, HippoDB, HippoIntegrityError

import pyarrow as pa

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from iai_mcp.crypto import (
    CIPHERTEXT_PREFIX,
    NONCE_BYTES,
    CryptoKey,
    encrypt_field,
    is_encrypted,
)
from iai_mcp.types import (
    HV_TIER_ENUM,
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
)

logger = logging.getLogger(__name__)


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


class MemoryStore:

    def __init__(
        self,
        path: Path | str | None = None,
        user_id: str = "default",
        read_consistency_interval: timedelta | None = None,
        *,
        access_mode: AccessMode = AccessMode.EXCLUSIVE,
        read_only: bool = False,
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
        )
        self._embed_dim: int = _resolve_embed_dim()
        self._ensure_tables()
        self._graph_sync_hook: Callable[[str, "MemoryRecord"], None] | None = None
        self._write_queue = None  # type: ignore[assignment]
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._async_thread: threading.Thread | None = None
        self._async_conn = None
        self._provenance_queue = None  # type: ignore[assignment]

    def close(self) -> None:
        if self.db is None:
            return

        from iai_mcp.events import (
            _BUFFER_LOCK,
            _event_buffer,
            _last_flush_at,
            flush_event_buffer,
        )

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

            _event_buffer.pop(id(self), None)
            _last_flush_at.pop(id(self), None)
            _record_buffer.pop(id(self), None)
            _record_last_flush_at.pop(id(self), None)
            _edge_buffer.pop(id(self), None)
            _edge_last_flush_at.pop(id(self), None)
            try:
                from iai_mcp.retrieve import _tv_cache, _tv_cache_dirty
                _tv_cache.pop(id(self), None)
                _tv_cache_dirty.pop(id(self), None)
            except ImportError:
                pass

            self.db.close()
            self.db = None

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

    @functools.cached_property
    def _cached_aesgcm(self) -> AESGCM:
        # late import so the package attribute is re-fetched per call and monkeypatches stay visible
        from iai_mcp.store import AESGCM as _AESGCM
        return _AESGCM(self._key())

    def _invalidate_aesgcm_cache(self) -> None:
        self.__dict__.pop("_cached_aesgcm", None)

    def _decrypt_for_record(self, record_id: UUID, value: str) -> str:
        if not is_encrypted(value):
            return value
        if not value.startswith(CIPHERTEXT_PREFIX):
            raise ValueError("field is not iai:enc:v1:-prefixed ciphertext")
        payload_b64 = value[len(CIPHERTEXT_PREFIX):]
        payload = base64.b64decode(payload_b64)
        if len(payload) < NONCE_BYTES + 16:
            raise ValueError("ciphertext payload too short")
        nonce = payload[:NONCE_BYTES]
        ct_with_tag = payload[NONCE_BYTES:]
        associated_data = self._ad(record_id)
        plaintext_bytes = self._cached_aesgcm.decrypt(
            nonce, ct_with_tag, associated_data or None
        )
        return plaintext_bytes.decode("utf-8")


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

        try:
            from iai_mcp.time_cells import compute_temporal_hash
            if record.created_at and not getattr(record, "_temporal_hash", None):
                record._temporal_hash = compute_temporal_hash(
                    session_id=getattr(record, "session_id", "-") or "-",
                    timestamp=record.created_at,
                )
        except (ImportError, TypeError, ValueError):
            pass

        from iai_mcp.daemon_config import _load_patsep_config
        from iai_mcp.events import write_event
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
                }, severity="info", buffered=True)
            return

        row = self._to_row(record)
        _record_buffer.setdefault(id(self), []).append(row)
        if should_flush_record_buffer(id(self)):
            flush_record_buffer(self)
        from iai_mcp.retrieve import invalidate_temporal_validity_cache
        invalidate_temporal_validity_cache(self)
        self._fire_graph_sync_hook("insert", record)
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
            loop = asyncio.new_event_loop()
            loop_holder["loop"] = loop
            asyncio.set_event_loop(loop)
            ready.set()
            try:
                loop.run_forever()
            finally:
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

        def _on_flushed(batch: list) -> None:
            for rec in batch:
                fire_hook("insert", rec)

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

    async def disable_async_writes(self) -> None:
        if self._write_queue is None:
            self.disable_provenance_queue()
            return
        self.disable_provenance_queue()
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

    def delete(self, record_id: UUID) -> None:
        tbl = self.db.open_table(RECORDS_TABLE)
        try:
            tbl.delete(where=f"id = '{_uuid_literal(record_id)}'")
        except (OSError, ValueError, RuntimeError) as exc:
            logger.warning("store delete normalised to no-op for %s: %s", record_id, exc)
            return

        class _DeleteShim:
            def __init__(self, rid):
                self.id = rid
        self._fire_graph_sync_hook("delete", _DeleteShim(record_id))

    def get(self, record_id: UUID) -> MemoryRecord | None:
        rid = _uuid_literal(record_id)
        sql = (  # nosemgrep: sql-injection
            f"SELECT {self._RECORD_COLS} FROM records WHERE id = ? LIMIT 1"
        )
        with self.db._conn_lock:
            raw = self.db._conn.execute(sql, (rid,)).fetchone()
        if raw is None:
            return None
        return self._from_row(self._decode_raw_row(dict(raw)))

    def all_records(self) -> list[MemoryRecord]:
        tbl = self.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        return [self._from_row(r.to_dict()) for _, r in df.iterrows()]

    def active_records_count(self) -> int:
        with self.db._conn_lock:
            row = self.db._conn.execute(
                "SELECT COUNT(*) FROM records"
                " WHERE tombstoned_at IS NULL"
                " AND COALESCE(embedding_pending, 0) = 0"
            ).fetchone()
        return int(row[0]) if row else 0

    def find_record_by_tag(self, tag: str) -> UUID | None:
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
        str_ids = list(dict.fromkeys(str(i) for i in ids))
        placeholders = ", ".join("?" for _ in str_ids)
        out: dict[UUID, float] = {}
        with self.db._conn_lock:
            rows = self.db._conn.execute(
                f"SELECT id, centrality FROM records WHERE id IN ({placeholders})",
                str_ids,
            ).fetchall()
        for row in rows:
            raw_id = row["id"] if hasattr(row, "keys") else row[0]
            if raw_id is None:
                continue
            id_str = str(raw_id)
            try:
                raw_centrality = row["centrality"] if hasattr(row, "keys") else row[1]
                centrality = float(raw_centrality or 0.0)
            except (TypeError, ValueError):
                centrality = 0.0
            try:
                out[UUID(id_str)] = centrality
            except (ValueError, AttributeError):
                continue
        return out

    def recent_user_turns(
        self,
        n: int = 10,
        session_id: str | None = None,
        pending_live_events: "list | None" = None,
    ) -> "list":
        from iai_mcp.capture import _idem_tag as _cap_idem_tag

        records = self.all_records()
        cands = [
            r for r in records
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

        if pending_live_events is not None:
            store_idem_set: set[str] = set()
            for r in records:
                for tag in (r.tags or []):
                    if tag.startswith("idem:"):
                        store_idem_set.add(tag)

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
                if idem in store_idem_set:
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
        tbl = self.db.open_table(RECORDS_TABLE)
        query = tbl.search()
        if where is not None:
            query = query.where(where)
        query = query.select(columns)
        reader = query.to_batches(batch_size=batch_size)
        for batch in reader:
            for row_dict in batch.to_pylist():
                yield row_dict

    def query_similar(
        self,
        vec: list[float],
        k: int = 10,
        tier: str | None = None,
        *,
        n: int | None = None,
    ) -> list[tuple[MemoryRecord, float]]:
        _return_records_only = n is not None
        if n is not None:
            k = n
        if tier is not None and tier not in TIER_ENUM:
            raise ValueError(
                f"invalid tier {tier!r}; must be one of {sorted(TIER_ENUM)}"
            )

        tbl = self.db.open_table(RECORDS_TABLE)
        q = tbl.search(list(vec)).distance_type("cosine")
        where_clause = "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        if tier is not None:
            where_clause = f"tier = '{tier}' AND " + where_clause
        try:
            out = self._query_similar_fast(q, where_clause, k)
        except Exception as exc:  # noqa: BLE001 -- preserve legacy fallback
            logger.warning("query_similar_fast_failed_degraded: %s", exc)
            out = []
        if _return_records_only:
            return [r for r, _s in out]
        return out

    def _query_similar_fast(
        self,
        query,
        where_clause: str,
        k: int,
    ) -> list[tuple[MemoryRecord, float]]:
        db = query._ann_db
        ann_vector = query._ann_vector
        if db is None or ann_vector is None:
            return []
        with db._hnsw_lock:
            active_count = len(db._label_map)
            hnsw_count = db._hnsw.get_current_count()
            k_clamped = min(max(0, int(k)), active_count, hnsw_count)
            if k_clamped == 0:
                return []
            try:
                labels, distances = db._hnsw.knn_query(ann_vector, k=k_clamped)
            except RuntimeError as exc:
                logger.warning(
                    "hnswlib knn_query failed (k=%d, hnsw_count=%d, "
                    "active_count=%d); returning empty result: %s",
                    k_clamped, hnsw_count, active_count, exc,
                )
                return []

        flat_labels = [int(label) for label in labels[0].tolist()]
        if not flat_labels:
            return []
        flat_distances = [
            max(0.0, min(2.0, float(distance))) for distance in distances[0]
        ]
        dist_map = {
            int(label): float(distance)
            for label, distance in zip(flat_labels, flat_distances)
        }

        placeholders = ", ".join("?" for _ in flat_labels)
        sql = (  # nosemgrep: sql-injection
            f"SELECT {self._RECORD_COLS}, vec_label FROM records"
            f" WHERE vec_label IN ({placeholders}) AND ({where_clause})"
        )
        with self.db._conn_lock:
            raw_rows = self.db._conn.execute(sql, flat_labels).fetchall()

        out: list[tuple[MemoryRecord, float]] = []
        sort_distance_by_id: dict[UUID, float] = {}
        for raw in raw_rows:
            row_dict = self._decode_raw_row(dict(raw))
            label = int(row_dict.get("vec_label") or 0)
            try:
                record = self._from_row(row_dict)
            except Exception:  # noqa: BLE001 -- skip corrupt rows, never crash recall
                continue
            distance = dist_map.get(label, 1.0)
            sort_distance_by_id[record.id] = distance
            out.append((record, 1.0 - distance))
        out.sort(key=lambda item: sort_distance_by_id.get(item[0].id, 1.0))
        return out

    def pattern_separation_gate(
        self,
        record: MemoryRecord,
    ) -> tuple["GateAction", "GatePayload"]:
        action, payload, _hits = self._pattern_separation_gate_with_hits(record)
        return (action, payload)

    def _pattern_separation_gate_with_hits(
        self,
        record: MemoryRecord,
    ) -> tuple["GateAction", "GatePayload", list[tuple[MemoryRecord, float]]]:
        from iai_mcp.daemon_config import _load_patsep_config
        cfg = _load_patsep_config()

        hits = self.query_similar(list(record.embedding), k=cfg.top_k)

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
                    if not (
                        getattr(record, "never_merge", False)
                        or getattr(top_record, "never_merge", False)
                    ):
                        return (GateAction.SKIP, top_record.id, hits)

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
            df = tbl.to_pandas()
            if df.empty:
                return
            for rid_str, entries in grouped.items():
                idx_list = df.index[df["id"] == rid_str].tolist()
                if not idx_list:
                    continue
                try:
                    canonical = _uuid_literal(rid_str)
                except ValueError:
                    continue
                i = idx_list[0]
                raw_prov = df.at[i, "provenance_json"] or "[]"
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

        import pyarrow as pa
        update_tbl = pa.table({
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
        "id, tier, literal_surface, aaak_index, embedding,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending"
    )

    _PENDING_READ_SQL = (
        f"SELECT {_RECORD_COLS} FROM records"  # noqa: S608
        " WHERE embedding_pending = 1"
        " ORDER BY rowid DESC LIMIT ?"
    )

    _ROLE_USER_READ_SQL = (
        f"SELECT {_RECORD_COLS} FROM records"  # noqa: S608
        " WHERE tier='episodic' AND tags_json LIKE ?"
        " ORDER BY rowid DESC LIMIT ?"
    )

    @staticmethod
    def _decode_raw_row(row: "dict") -> "dict":
        import numpy as _np
        emb_raw = row.get("embedding")
        if isinstance(emb_raw, (bytes, bytearray)) and emb_raw:
            row = dict(row)
            row["embedding"] = _np.frombuffer(emb_raw, dtype=_np.float32).tolist()
        return row

    def incident_edges(
        self,
        ids: "list[UUID]",
        edge_types: "list[str] | None" = None,
        top_k: "int | None" = 5,
    ) -> "dict[UUID, list[tuple[UUID, str, float]]]":
        if not ids:
            return {}

        str_ids = [str(i) for i in ids]
        id_set = set(str_ids)

        if edge_types == []:
            return {i: [] for i in ids}

        if top_k is not None:
            result: dict[UUID, list[tuple[UUID, str, float]]] = {i: [] for i in ids}
            limit = max(0, int(top_k))
            if limit <= 0:
                return result

            edge_type_sql = ""
            edge_type_params: list[str] = []
            if edge_types is not None:
                et_ph = ", ".join("?" for _ in edge_types)
                edge_type_sql = f" AND edge_type IN ({et_ph})"
                edge_type_params = list(edge_types)

            with self.db._conn_lock:
                for uid, sid in zip(ids, str_ids):
                    rows = []
                    rows.extend(self.db._conn.execute(
                        "SELECT dst, edge_type, weight FROM edges"
                        f" WHERE src = ?{edge_type_sql}"
                        " ORDER BY weight DESC LIMIT ?",
                        [sid, *edge_type_params, limit],
                    ).fetchall())
                    rows.extend(self.db._conn.execute(
                        "SELECT src, edge_type, weight FROM edges"
                        f" WHERE dst = ? AND src != ?{edge_type_sql}"
                        " ORDER BY weight DESC LIMIT ?",
                        [sid, sid, *edge_type_params, limit],
                    ).fetchall())

                    for row in rows:
                        nbr_s = str(row[0] if hasattr(row, "__getitem__") else row["dst"])
                        et = str(row[1] if hasattr(row, "__getitem__") else row["edge_type"])
                        wt = float(row[2] if hasattr(row, "__getitem__") else row["weight"])
                        try:
                            neighbour = UUID(nbr_s)
                        except (ValueError, AttributeError):
                            continue
                        result[uid].append((neighbour, et, wt))

                    result[uid].sort(key=lambda t: t[2], reverse=True)
                    result[uid] = result[uid][:limit]

            return result

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

        with self.db._conn_lock:
            rows = self.db._conn.execute(sql, params).fetchall()

        result: dict[UUID, list[tuple[UUID, str, float]]] = {i: [] for i in ids}
        id_to_uuid: dict[str, UUID] = {str(i): i for i in ids}

        for row in rows:
            src_s = str(row[0] if hasattr(row, "__getitem__") else row["src"])
            dst_s = str(row[1] if hasattr(row, "__getitem__") else row["dst"])
            et = str(row[2] if hasattr(row, "__getitem__") else row["edge_type"])
            wt = float(row[3] if hasattr(row, "__getitem__") else row["weight"])

            if src_s in id_set:
                qid = id_to_uuid[src_s]
                try:
                    neighbour = UUID(dst_s)
                except (ValueError, AttributeError):
                    continue
                result[qid].append((neighbour, et, wt))

            if dst_s in id_set and dst_s != src_s:
                qid = id_to_uuid[dst_s]
                try:
                    neighbour = UUID(src_s)
                except (ValueError, AttributeError):
                    continue
                result[qid].append((neighbour, et, wt))

        if top_k is not None:
            for uid, edges in result.items():
                edges.sort(key=lambda t: t[2], reverse=True)
                result[uid] = edges[:top_k]

        return result

    def get_batch(self, ids: "list[UUID]") -> "dict[UUID, MemoryRecord]":
        if not ids:
            return {}

        str_ids = [str(i) for i in ids]
        ph = ", ".join("?" for _ in str_ids)
        sql = (  # nosemgrep: sql-injection
            f"SELECT {self._RECORD_COLS} FROM records"  # noqa: S608
            f" WHERE id IN ({ph})"
        )
        with self.db._conn_lock:
            raw_rows = self.db._conn.execute(sql, str_ids).fetchall()

        out: dict[UUID, MemoryRecord] = {}
        for raw in raw_rows:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
                out[rec.id] = rec
            except Exception:  # noqa: BLE001 — skip corrupt rows, never crash
                continue
        return out

    def recent_pending_markers(self, n: int = 50) -> "list[MemoryRecord]":
        seen: dict[UUID, "MemoryRecord"] = {}

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
            except Exception:  # noqa: BLE001
                continue

        over_fetch = n * 4
        with self.db._conn_lock:
            rows_b = self.db._conn.execute(
                self._ROLE_USER_READ_SQL, ('%"role:user"%', over_fetch)
            ).fetchall()

        for raw in rows_b:
            row_dict = self._decode_raw_row(dict(raw))
            try:
                rec = self._from_row(row_dict)
            except Exception:  # noqa: BLE001
                continue
            if "role:user" not in (rec.tags or []):
                continue
            if rec.id not in seen:
                seen[rec.id] = rec

        candidates = list(seen.values())
        candidates.sort(
            key=lambda r: r.created_at or datetime.min.replace(tzinfo=timezone.utc),
            reverse=True,
        )
        return candidates[:n]


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

        _SMALL_BATCH = 4
        update_rows: list[dict] = []
        insert_rows: list[dict] = []
        new_weights: dict[tuple[str, str], float] = {}
        now = datetime.now(timezone.utc)

        if len(coalesced) <= _SMALL_BATCH:
            for (src_str, dst_str), accum_delta in coalesced.items():
                predicate = (
                    f"src = '{_uuid_literal(src_str)}' "
                    f"AND dst = '{_uuid_literal(dst_str)}' "
                    f"AND edge_type = '{edge_type}'"
                )
                try:
                    df = (
                        tbl.search()
                        .where(predicate)
                        .select(["weight"])
                        .limit(1)
                        .to_pandas()
                    )
                except Exception as exc:  # noqa: BLE001 -- fallback gate
                    logger.warning(
                        "edge predicate-scan fallback for (%s,%s,%s): %s",
                        src_str, dst_str, edge_type, exc, exc_info=True,
                    )
                    df = None
                if df is not None and not df.empty:
                    cur = float(df["weight"].iloc[0])
                    nw = cur + accum_delta
                    update_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                elif df is not None:
                    nw = accum_delta
                    insert_rows.append({
                        "src": src_str, "dst": dst_str,
                        "edge_type": edge_type,
                        "weight": nw, "updated_at": now,
                    })
                else:
                    existing = tbl.to_pandas()
                    mask = (
                        (existing["src"] == src_str)
                        & (existing["dst"] == dst_str)
                        & (existing["edge_type"] == edge_type)
                    ) if len(existing) > 0 else None
                    if mask is not None and mask.any():
                        cur = float(existing.loc[mask, "weight"].iloc[0])
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
            existing = tbl.to_pandas()
            for (src_str, dst_str), accum_delta in coalesced.items():
                if len(existing) > 0:
                    mask = (
                        (existing["src"] == src_str)
                        & (existing["dst"] == dst_str)
                        & (existing["edge_type"] == edge_type)
                    )
                else:
                    mask = None
                if mask is not None and mask.any():
                    cur = float(existing.loc[mask, "weight"].iloc[0])
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
                upd_arrow = pa.Table.from_pylist(
                    update_rows,
                    schema=pa.schema(
                        [
                            ("src", pa.string()),
                            ("dst", pa.string()),
                            ("edge_type", pa.string()),
                            ("weight", pa.float32()),
                            ("updated_at", pa.timestamp("us", tz="UTC")),
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
            try:
                from iai_mcp.daemon_config import _load_reconsolidation_config
                cfg = _load_reconsolidation_config()
                if not cfg.dry_run:
                    labile_until = datetime.now(timezone.utc) + timedelta(
                        seconds=cfg.labile_window_sec
                    )
                    tbl = self.db.open_table(RECORDS_TABLE)
                    try:
                        tbl.update(
                            where=f"id = '{_uuid_literal(record_id)}'",
                            values={"labile_until": labile_until},
                        )
                    except (RuntimeError, ValueError, OSError, KeyError) as exc:
                        msg = str(exc).lower()
                        column_missing = (
                            "labile_until" in msg
                            or "no such column" in msg
                            or ("column" in msg and "not found" in msg)
                        )
                        if not column_missing:
                            raise
                        logger.debug("labile_until column missing, skipped: %s", exc)
            except ImportError:
                pass
        return result

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

        def _parse_ts(val: Any) -> datetime | None:
            if val is None:
                return None
            try:
                if val != val:
                    return None
            except (TypeError, ValueError):
                pass
            try:
                if str(val) in {"NaT", "nan", "NaN"}:
                    return None
            except (TypeError, ValueError):
                pass
            if isinstance(val, datetime):
                return val if val.tzinfo is not None else val.replace(tzinfo=timezone.utc)
            if hasattr(val, "to_pydatetime"):
                try:
                    dt = val.to_pydatetime()
                except (TypeError, ValueError, AttributeError):
                    return None
                if dt is None:
                    return None
                return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
            try:
                dt = datetime.fromisoformat(str(val).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                return None
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)

        if "id" not in row:
            raise KeyError(
                "iter_records consumer must include 'id' in column projection"
            )

        structure_raw = row.get("structure_hv")
        if structure_raw is None:
            structure_hv = b""
        elif isinstance(structure_raw, (bytes, bytearray)):
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
            structure_hv_payload_raw, (bytes, bytearray)
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
                if isinstance(structure_hv_payload_raw, (bytes, bytearray))
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
        )
        if language == "__LEGACY_EMPTY__":
            rec.language = ""
        return rec
