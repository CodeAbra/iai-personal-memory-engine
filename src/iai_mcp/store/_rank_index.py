"""Store-identity-keyed adapter over the Rust rank-feature index.

Builds `iai_mcp_native.rank.RankIndex` lazily from the store's own corpus on
first use. The handle is stored as an attribute of the `MemoryStore`
instance itself, never a module-level cache, so two stores in one process
never share state. Generation is owned by this handle (`self._generation`,
reconciled to `max(local, engine)` on every `snapshot()`) -- there is no
graph-identity lineage token; `feed()` marks the handle dirty and the next
`snapshot()` requests one generation past whatever was last published.

Bulk read accessors only: no per-candidate method exists on this surface.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from uuid import UUID

import numpy as np

from iai_mcp.types import SALIENCE_LEVEL_RANK

_HANDLE_ATTR = "_rank_index_handle"
_RESIDENT_CAP_ENV = "IAI_MCP_RESIDENT_CANDIDATE_CAP"
_RESIDENT_CAP_DEFAULT = 4096


class _ResidentFields:
    """Non-vector fields of one hydrated candidate, cached for the
    decrypt-skip residency path. No `embedding` (sourced fresh from
    `published_view()` per hit) and no `structure_hv` (~10 KB HDC payload,
    `directive`, or `provenance` -- none has a read site on this path)."""

    __slots__ = (
        "literal_surface", "aaak_index", "created_at", "stability", "tier",
        "tags", "language", "community_id", "centrality", "valence",
        "salience_level",
    )

    def __init__(self, view) -> None:
        self.literal_surface = getattr(view, "literal_surface", "") or ""
        self.aaak_index = getattr(view, "aaak_index", "") or ""
        self.created_at = getattr(view, "created_at", None)
        self.stability = float(getattr(view, "stability", 0.5) or 0.5)
        self.tier = getattr(view, "tier", None) or "episodic"
        self.tags = list(getattr(view, "tags", []) or [])
        self.language = getattr(view, "language", None) or "en"
        self.community_id = getattr(view, "community_id", None)
        self.centrality = float(getattr(view, "centrality", 0.0) or 0.0)
        self.valence = float(getattr(view, "valence", 0.0) or 0.0)
        self.salience_level = getattr(view, "salience_level", None) or "unflagged"


class _ResidentCandidates:
    """Bounded id-keyed LRU of `_ResidentFields` -- the decrypt-skip
    residency cache replacing the unbounded per-store builder graph.
    Capacity from `IAI_MCP_RESIDENT_CANDIDATE_CAP` (default 4096; 0
    disables the cache). Thread-safe for the daemon's `asyncio.to_thread`
    fan-out."""

    def __init__(self, capacity: "int | None" = None) -> None:
        if capacity is None:
            try:
                capacity = int(
                    os.environ.get(_RESIDENT_CAP_ENV, "") or _RESIDENT_CAP_DEFAULT
                )
            except ValueError:
                capacity = _RESIDENT_CAP_DEFAULT
        self._capacity = max(0, capacity)
        self._entries: "OrderedDict[UUID, _ResidentFields]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, rid: UUID) -> "_ResidentFields | None":
        with self._lock:
            fields = self._entries.get(rid)
            if fields is not None:
                self._entries.move_to_end(rid)
            return fields

    def put(self, view) -> None:
        if self._capacity <= 0:
            return
        rid = view.id
        with self._lock:
            self._entries[rid] = _ResidentFields(view)
            self._entries.move_to_end(rid)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)

    def evict(self, rid: UUID) -> None:
        with self._lock:
            self._entries.pop(rid, None)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def _bulk_pending_records(store) -> list:
    """`MemoryRecord`s for every non-tombstoned `embedding_pending=1` row --
    the same corpus slice today's `LexicalIndex` build includes but
    `retrieve.build_runtime_graph`'s active-only scan excludes at the SQL
    level. Fetched via the same `get_batch`-decrypt path `LexicalIndex`
    uses, so surfaces are never empty for a genuinely-captured pending row."""
    from uuid import UUID as _UUID

    ids: list = []
    try:
        for row in store.iter_record_columns(
            ["id"],
            batch_size=2048,
            where="tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 1",
        ):
            try:
                ids.append(_UUID(str(row["id"])))
            except (TypeError, ValueError):
                continue
    except Exception:  # noqa: BLE001 -- a failed scan degrades to no pending rows fed, never breaks the build
        return []
    if not ids:
        return []
    records: list = []
    for i in range(0, len(ids), 400):
        try:
            batch = store.get_batch(ids[i : i + 400])
        except Exception:  # noqa: BLE001 -- one bad chunk must not drop the rest
            continue
        records.extend(batch.values())
    return records


def _bulk_salience_ranks(store) -> dict[str, int]:
    """id-string -> SALIENCE_LEVEL_RANK read directly off the plaintext
    column -- the only correct source for the pre-existing corpus, written
    before the write-time hook payload carried this field at all."""
    ranks: dict[str, int] = {}
    try:
        with store.db.ro_conn() as conn:
            rows = conn.execute(
                "SELECT id, salience_level FROM records"
            ).fetchall()
    except Exception:  # noqa: BLE001 -- a failed backfill degrades every id to rank 0, never breaks the build
        return ranks
    for row in rows:
        rid, level = row[0], row[1]
        ranks[str(rid)] = SALIENCE_LEVEL_RANK.get(level, 0)
    return ranks


class _RankIndexHandle:
    """One handle per store, lazily built on first `snapshot()`/`feed()`."""

    def __init__(self, store) -> None:
        self._store = store
        self._index = None
        # Last-published generation and dirty flag -- the sole lineage
        # state for this handle. `feed()` sets dirty; `snapshot()`
        # reconciles to `max(self._generation, engine.generation())` and
        # requests one past that when dirty, then clears it.
        self._generation = 0
        self._dirty = False
        # Guards the build-once sequence in snapshot()/score()/
        # published_view() -- without it, two threads racing a first build
        # both build and the loser's build (plus any feed() ops landed on
        # it) is silently discarded.
        self._lock = threading.RLock()
        # Transitional per-generation export memo: caches the last bulk
        # (matrix/degree/postings) export keyed by (generation, tokens), so
        # a repeated ask at an unchanged generation+token-set returns the
        # cached artifacts instead of re-running the export. Reset on
        # every rebuild -- a fresh `_index` invalidates any prior export.
        self._export_memo: "tuple[tuple[int, tuple[str, ...]], tuple] | None" = None
        # Memo for published_view(), keyed on the Rust engine's own
        # generation() -- NOT on any per-call graph's _pool_content_version,
        # which resets per ephemeral MemoryGraph instance and would give
        # zero cross-call reuse. Reset on every rebuild alongside
        # _export_memo.
        self._pool_view_memo: "tuple[int, tuple[np.ndarray, dict[int, int]]] | None" = None
        # Memo for hit_view_source(), same generation-keyed scheme as
        # _pool_view_memo. Reset on every rebuild alongside it.
        self._hit_view_memo: "tuple[int, tuple[np.ndarray, dict[int, int], dict]] | None" = None
        self._resident = _ResidentCandidates()

    @property
    def resident(self) -> _ResidentCandidates:
        """The bounded decrypt-skip residency cache for this store."""
        return self._resident

    def _needs_build(self) -> bool:
        """A full Python-side rebuild is required only the first time this
        handle is used. Content growth thereafter is NEVER treated as
        staleness here -- it is drained through the Rust engine's own
        generation-tagged double-buffer inside snapshot(), never by
        re-running this rebuild."""
        return self._index is None

    def feed(
        self, op: str, record, edges: "list[tuple[int, float, str]] | None" = None,
    ) -> None:
        """Forward one incremental write. A no-op before the first build --
        the eventual first `snapshot()` reads current state fresh, so a
        pre-build write is never lost, only deferred. `edges` is the
        record's OWN adjacency (id, weight, edge_type) when the caller
        already has it at hand (e.g. a write-time hook that just wrote the
        edge) -- never fetched here, so a caller with no edge data on hand
        (the common hydrate-time upsert) simply omits it rather than paying
        a per-candidate incident-edges lookup on the recall path.

        The `_dirty` set and the underlying engine `feed()`/`delete()` call
        run under `self._lock`, the SAME lock `snapshot()` holds for its
        entire dirty-read/generation-request/dirty-clear sequence -- a
        `feed()` call therefore either completes in full before a
        concurrent `snapshot()` starts (so its write is queued before that
        snapshot's drain) or starts only after that snapshot has fully
        published, never interleaved. Without this, `self._dirty = True`
        landing between a `snapshot()`'s Rust `snapshot()` call and its own
        `self._dirty = False` would clear the dirty SIGNAL without that
        write ever having been the one the in-flight snapshot drained."""
        with self._lock:
            if self._index is None:
                return
            self._dirty = True
            if op == "delete":
                self._index.feed("delete", record.id.int)
                return
            vector = np.asarray(list(record.embedding or []), dtype=np.float32)
            self._index.feed(
                "upsert",
                record.id.int,
                vector=vector,
                edges=list(edges) if edges else None,
                surface=str(getattr(record, "literal_surface", "") or ""),
                aaak_index=str(getattr(record, "aaak_index", "") or ""),
                created_at=(
                    record.created_at.isoformat()
                    if getattr(record, "created_at", None) else ""
                ),
                stability=float(getattr(record, "stability", 0.5) or 0.5),
                tier=str(getattr(record, "tier", "") or ""),
                tags=list(getattr(record, "tags", []) or []),
                salience_level=SALIENCE_LEVEL_RANK.get(
                    getattr(record, "salience_level", "unflagged"), 0
                ),
                centrality=float(getattr(record, "centrality", 0.0) or 0.0),
                pending=bool(getattr(record, "embedding_pending", 0)),
                valence=float(getattr(record, "valence", 0.0) or 0.0),
                language=str(getattr(record, "language", "en") or "en"),
            )

    def overlay_len(self) -> int:
        """Count of ids the bounded delta overlay currently carries a
        verdict for. `0` before the first build -- nothing to fold."""
        if self._index is None:
            return 0
        return int(self._index.overlay_len())

    def fold(self) -> None:
        """The ~174ms wholesale CSR rebuild. Callers MUST keep this off the
        recall read path -- see `maybe_fold()`, the only wiring this
        adapter exposes to a real caller."""
        if self._index is None:
            return
        self._index.fold()

    def maybe_fold(self, threshold: "int | None" = None) -> bool:
        """Fold the overlay back into the committed CSR when it exceeds
        `threshold` (env-overridable via `IAI_MCP_RANK_OVERLAY_FOLD_THRESHOLD`,
        default 500). ONLY call this from a write-time site (the store's
        graph_sync_hook) or a maintenance/idle tick -- NEVER from the recall
        read path (`snapshot()`/`score()`), which would reintroduce the
        wholesale-rebuild cost the bounded delta overlay design moved off it.
        Returns whether a fold actually ran."""
        if self._index is None:
            return False
        if threshold is None:
            try:
                threshold = int(
                    os.environ.get("IAI_MCP_RANK_OVERLAY_FOLD_THRESHOLD", "") or 500
                )
            except ValueError:
                threshold = 500
        if self.overlay_len() < threshold:
            return False
        self.fold()
        return True

    def snapshot(self, tokens: "list[str] | None" = None):
        """Bulk zero-copy views at this handle's own generation. Builds on
        first use; otherwise reconciles `self._generation` to
        `max(self._generation, engine.generation())` first (a durable guard
        against any path that advances the engine generation outside this
        method -- `fold()` is proven not to, but the reconcile costs one int
        compare), then requests one generation past that when dirty, else a
        pure read at the reconciled generation. Defers to the Rust struct's
        own generation-tagged double-buffer to drain any queued `feed()` ops.

        The returned export tuple is memoized at this adapter layer, keyed
        by (generation, tokens): a repeated ask at an unchanged generation
        and token set returns the cached artifacts instead of re-running
        the matrix/degree/postings export. The Rust engine's own
        matching-generation pure-read `Arc` fast path is untouched either
        way -- this memo only decides whether that call happens at all.
        """
        with self._lock:
            if self._needs_build():
                self._build()
            base = max(self._generation, int(self._index.generation()))
            requested = base + 1 if self._dirty else base
            token_key = tuple(tokens or [])
            memo = self._export_memo
            if memo is not None and memo[0] == (requested, token_key):
                return memo[1]
            exported = self._index.snapshot(requested, list(token_key))
            # Key off the Rust engine's actually-published generation
            # (exported[0]), not the requested one -- a racing caller can
            # win a later generation than it asked for (see
            # DoubleBuffer::snapshot).
            self._generation = exported[0]
            self._dirty = False
            self._export_memo = ((exported[0], token_key), exported)
            return exported

    def published_view(self) -> "tuple[np.ndarray, dict[int, int]]":
        """Memoized on the engine's own `generation()` -- one memoized bulk
        export per engine generation, reused across every caller in that
        window."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._build()
        index = self._index
        generation = int(index.generation())
        memo = self._pool_view_memo
        if memo is not None and memo[0] == generation:
            return memo[1]
        ids, matrix = index.vectors()
        slot_map = {int(rid): i for i, rid in enumerate(ids)}
        view = (matrix, slot_map)
        self._pool_view_memo = (generation, view)
        return view

    def salience_levels(self) -> dict:
        """Bulk id(int)->rank map off the currently active buffer. Call
        after `snapshot()` to read what that snapshot published."""
        if self._index is None:
            return {}
        return self._index.salience_levels()

    def pending_flags(self) -> dict:
        """Bulk id(int)->pending-flag map off the currently active buffer,
        the same `buf.current()` `vectors()`/`published_view()` reads --
        a caller resolving a resident-cache hit against `published_view()`'s
        matrix must route an id whose flag is `True` to a miss rather than
        trust its slot: that row is `_build()`'s fabricated
        `np.zeros(dim)` fill for a genuinely `embedding_pending` record,
        never a real embedding."""
        if self._index is None:
            return {}
        return self._index.pending()

    def hit_view_source(self) -> "tuple[np.ndarray, dict[int, int], dict]":
        """Bulk (matrix, slot_map, pending) triple from a single
        `self._lock` acquisition covering both `vectors()` and `pending()`
        -- the same lock `snapshot()` holds for its buffer swap, so the
        pair is always read from one committed buffer, unlike calling
        `published_view()` and `pending_flags()` separately. Memoized on
        the engine's own `generation()`."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._build()
        with self._lock:
            index = self._index
            generation = int(index.generation())
            memo = self._hit_view_memo
            if memo is not None and memo[0] == generation:
                return memo[1]
            ids, matrix = index.vectors()
            pending = index.pending()
            slot_map = {int(rid): i for i, rid in enumerate(ids)}
            triple = (matrix, slot_map, pending)
            self._hit_view_memo = (generation, triple)
            return triple

    def adjacency_by_type(self) -> dict:
        """Bulk id(int) -> {edge_type: count} map off the currently active
        buffer. The ranking-degree edge-type exclusion is applied by the
        caller over this one bulk result -- never a second per-node call."""
        if self._index is None:
            return {}
        return self._index.adjacency_by_type()

    def score(  # noqa: PLR0913 -- mirrors the FFI params contract 1:1, no grouping shortens the boundary
        self,
        pool_ids: "list[UUID]",
        cosine: np.ndarray,
        cosine_top_indices: np.ndarray,
        spread_indices: np.ndarray,
        rich_indices: np.ndarray,
        lex_indices: np.ndarray,
        t11_flags: np.ndarray,
        t12_flags: np.ndarray,
        verbatim_filter: bool,
        cue: str,
        now: int,
        effective_w_degree: float,
        effective_w_cosine: float,
        excluded_edge_types: "frozenset[str] | set[str]",
        spread_provenance: "dict[UUID, tuple[UUID, int, bool]]",
        w_spread_act: float,
        spread_act_decay: float,
        community_id_by_member: "dict[UUID, UUID]",
        community_scores: "dict[UUID, float]",
        max_community_score: float,
        mode_bias: float,
        cos_spread_min: float,
        structural_weight: float,
        cue_structure_hv: "bytes | None",
        lex_lane_enabled: bool,
        min_idf: float,
        lex_fusion_w: float,
        k: int,
        k_margin: int,
    ):
        """One hybrid fused-score call. Builds only when never built
        (`self._index is None`); the handle-owned generation removes the
        graph-identity generation-regression hazard entirely -- there is no
        second lineage counter to desync against. `_build()` reads the
        corpus from the store, not from any graph payload.
        Draining a WARM index for a post-write delta stays the caller's
        responsibility: call `snapshot()` before `score()` this same recall.
        `pool_ids` and the four index arrays plus the two T11/T12 flag
        arrays cross exactly as the caller already built them (zero-copy
        numpy for the array params); only the bounded per-call
        dicts (spread provenance / community membership / community scores)
        are re-keyed to plain ints here, never a per-candidate structure."""
        if self._index is None:
            with self._lock:
                if self._index is None:
                    self._build()
        return self._index.score(
            [rid.int for rid in pool_ids],
            cosine,
            cosine_top_indices,
            spread_indices,
            rich_indices,
            lex_indices,
            t11_flags,
            t12_flags,
            verbatim_filter,
            cue,
            now,
            effective_w_degree,
            effective_w_cosine,
            set(excluded_edge_types),
            {
                sid.int: (seed.int, int(hop), bool(carries))
                for sid, (seed, hop, carries) in spread_provenance.items()
            },
            w_spread_act,
            spread_act_decay,
            {mid.int: gid.int for mid, gid in community_id_by_member.items()},
            {gid.int: float(score) for gid, score in community_scores.items()},
            max_community_score,
            mode_bias,
            cos_spread_min,
            structural_weight,
            cue_structure_hv,
            lex_lane_enabled,
            min_idf,
            lex_fusion_w,
            k,
            k_margin,
        )

    def _build(self) -> None:
        """Sources the WHOLE corpus (records + edges + surface text) directly
        from the store, mirroring `retrieve.build_runtime_graph`'s own
        whole-corpus scan. `literal_surface` plaintext is reused BY
        REFERENCE from a resident, freshness-checked full-corpus graph when
        one is available (no second decrypt); only a bundle-absent row pays
        its own `_decrypt_for_record` call. `LexicalIndex` tokenizes and
        discards the raw string, keeping only token postings, so this index
        still needs the plaintext at build time for the per-slot AAAK/
        lex-rank text lookups the Rust scorer performs."""
        dim = int(getattr(self._store, "embed_dim", 384))
        # Stamp from this handle's own last-published generation (0 on a
        # fresh handle), read BEFORE the bulk data scan: if a concurrent
        # write lands in between, the data can be fresher than the stamped
        # generation -- a later stale snapshot then simply replays an
        # already-applied upsert (upsert_record is insert-or-replace, so
        # idempotent). Stamping ahead of the data instead would let a
        # future matching-generation snapshot take the pure-read branch and
        # never drain a write that never actually made it into this build.
        generation = self._generation
        salience_ranks = _bulk_salience_ranks(self._store)

        # A resident, freshness-checked full-corpus graph already holds a
        # boot-decrypted `surface` per node (kept live by the write-time
        # graph_sync_hook) -- reusing that reference instead of decrypting
        # again is what removes this build's second whole-corpus decrypt
        # pass. Node-presence in this map already implies a successful
        # decrypt (a boot-time decrypt failure excludes the node entirely),
        # so no independent re-verification is needed for a resident id.
        _warm_node_payload: "dict[str, dict] | None" = None
        try:
            from iai_mcp import retrieve as _retrieve
            _warm_bundle = _retrieve._warm_graph_bundle_if_valid(self._store)
            if _warm_bundle is not None:
                _warm_node_payload = getattr(_warm_bundle[0], "_node_payload", None)
        except Exception:  # noqa: BLE001 -- a failed warm-bundle probe must fall back to per-row decrypt, never block the build
            _warm_node_payload = None

        ids: list[int] = []
        id_strs: set[str] = set()
        vector_rows: list[np.ndarray] = []
        surfaces: list[str] = []
        aaak_index: list[str] = []
        created_at: list[str] = []
        stability: list[float] = []
        tier: list[str] = []
        tags: list[list[str]] = []
        salience_level: list[int] = []
        centrality: list[float] = []
        pending: list[bool] = []
        valence: list[float] = []
        language: list[str] = []

        stream_cols = [
            "id", "embedding", "literal_surface", "tier", "centrality",
            "tags_json", "language", "aaak_index", "created_at", "stability",
            "valence",
        ]
        active_where = "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0"
        for row in self._store.iter_record_columns(
            stream_cols, batch_size=1024, where=active_where,
        ):
            try:
                node_id = UUID(str(row.get("id")))
            except (TypeError, ValueError):
                continue
            emb_raw = row.get("embedding")
            emb = (
                np.asarray(emb_raw, dtype=np.float32)
                if emb_raw is not None else np.zeros(dim, dtype=np.float32)
            )
            if emb.shape[0] != dim:
                # A dimension-mismatched resident row must never kill the
                # whole build -- skip it, keep the rest of the corpus
                # index-findable.
                continue
            warm_payload = (
                _warm_node_payload.get(str(node_id))
                if _warm_node_payload is not None else None
            )
            if warm_payload is not None and "surface" in warm_payload:
                literal_raw = warm_payload["surface"]
            else:
                literal_raw = row.get("literal_surface") or ""
                try:
                    from iai_mcp.crypto import is_encrypted
                    if is_encrypted(literal_raw):
                        literal_raw = self._store._decrypt_for_record(node_id, literal_raw)
                except Exception:  # noqa: BLE001 -- InvalidTag/OSError/ValueError/RuntimeError; one bad row must not kill the build
                    continue
            tags_raw = row.get("tags_json") or "[]"
            try:
                import json as _json
                tags_list = _json.loads(tags_raw) if isinstance(tags_raw, str) else list(tags_raw)
                if not isinstance(tags_list, list):
                    tags_list = []
            except (ValueError, TypeError):
                tags_list = []

            ids.append(node_id.int)
            id_strs.add(str(node_id))
            vector_rows.append(emb)
            surfaces.append(str(literal_raw))
            aaak_index.append(str(row.get("aaak_index") or ""))
            created_at.append(str(row.get("created_at") or ""))
            stability.append(float(row.get("stability") or 0.5))
            tier.append(str(row.get("tier") or "episodic"))
            tags.append(list(tags_list))
            salience_level.append(salience_ranks.get(str(node_id), 0))
            centrality.append(float(row.get("centrality") or 0.0))
            pending.append(False)
            valence.append(float(row.get("valence") or 0.0))
            language.append(str(row.get("language") or "en"))

        # Pending rows never reach the active-records scan above (excluded
        # by the same WHERE `build_runtime_graph` uses). Fed here as a
        # second source so the resident index's lexical membership matches
        # `LexicalIndex`'s (which includes them) while the `pending` flag
        # lets a cosine consumer exclude them exactly as the graph already
        # does.
        resident_ids = set(ids)
        for record in _bulk_pending_records(self._store):
            if record.id.int in resident_ids:
                continue
            emb = record.embedding
            row = (
                np.asarray(emb, dtype=np.float32)
                if emb else np.zeros(dim, dtype=np.float32)
            )
            if row.shape[0] != dim:
                continue
            ids.append(record.id.int)
            id_strs.add(str(record.id))
            vector_rows.append(row)
            surfaces.append(str(getattr(record, "literal_surface", "") or ""))
            aaak_index.append(str(getattr(record, "aaak_index", "") or ""))
            created_at.append(
                record.created_at.isoformat()
                if getattr(record, "created_at", None) else ""
            )
            stability.append(float(getattr(record, "stability", 0.5) or 0.5))
            tier.append(str(getattr(record, "tier", "") or ""))
            tags.append(list(getattr(record, "tags", []) or []))
            salience_level.append(salience_ranks.get(str(record.id), 0))
            centrality.append(float(getattr(record, "centrality", 0.0) or 0.0))
            pending.append(True)
            valence.append(float(getattr(record, "valence", 0.0) or 0.0))
            language.append(str(getattr(record, "language", "en") or "en"))

        # Whole-corpus edges, streamed the same way `build_runtime_graph`
        # streams them (batched row dicts over a read-only snapshot, never
        # one full-table materialization). Both endpoints must be active
        # residents (mirrors `_replay_runtime_edge`); each row is expanded
        # into BOTH directions (mirrors `MemoryGraph.add_edge`, which is
        # what every other adjacency consumer -- degree, BFS -- assumes),
        # since the edges table stores one canonicalized direction per pair.
        edge_acc: dict[int, list[tuple[int, float, str]]] = {}
        edges_query = (
            self._store.db.open_table("edges")
            .search()
            .select(["src", "dst", "weight", "edge_type"])
        )
        for batch in edges_query.to_batches(batch_size=2048):
            for erow in batch.to_pylist():
                src_s, dst_s = erow.get("src"), erow.get("dst")
                if src_s not in id_strs or dst_s not in id_strs:
                    continue
                try:
                    src_id, dst_id = UUID(src_s), UUID(dst_s)
                except (TypeError, ValueError):
                    continue
                weight = float(erow.get("weight") or 1.0)
                edge_type = str(erow.get("edge_type") or "hebbian")
                edge_acc.setdefault(src_id.int, []).append((dst_id.int, weight, edge_type))
                if src_id != dst_id:
                    edge_acc.setdefault(dst_id.int, []).append((src_id.int, weight, edge_type))
        edges: list[tuple[int, list[tuple[int, float, str]]]] = list(edge_acc.items())

        matrix = (
            np.stack(vector_rows).astype(np.float32)
            if vector_rows else np.zeros((0, dim), dtype=np.float32)
        )

        from iai_mcp_native import rank as _rank_native

        self._index = _rank_native.RankIndex(
            dim, generation, ids, matrix, edges, surfaces, aaak_index,
            created_at, stability, tier, tags, salience_level, centrality,
            pending, valence, language,
        )
        self._export_memo = None
        self._pool_view_memo = None
        self._hit_view_memo = None

    @property
    def generation(self) -> int:
        """Last-published generation. Read-only -- advances only through
        `snapshot()`."""
        return self._generation


def rank_index_for(store) -> _RankIndexHandle:
    """Store-identity-keyed handle: an attribute of `store` itself, never a
    module-level cache -- two stores in one process never share a handle."""
    handle = getattr(store, _HANDLE_ATTR, None)
    if handle is None:
        handle = _RankIndexHandle(store)
        setattr(store, _HANDLE_ATTR, handle)
    return handle
