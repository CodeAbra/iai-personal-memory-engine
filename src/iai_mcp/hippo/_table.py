"""Table-level CRUD, query, and merge-insert classes for the Hippo storage backend."""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from iai_mcp import _sqlite_stdlib
from iai_mcp import errors
from iai_mcp.hippo import _schema

_log = logging.getLogger(__name__)

#: Column identifiers are interpolated into generated SQL (add / update /
#: merge_insert). Values are always parameterized and table names validated, but
#: column names come from caller-supplied row dicts, so they are validated to the
#: same identifier grammar as a defense-in-depth guard against injection.
_COLUMN_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_columns(cols: "list[str]") -> "list[str]":
    for c in cols:
        if not _COLUMN_NAME_RE.match(c):
            raise ValueError(
                f"Invalid column name {c!r}: must match [A-Za-z_][A-Za-z0-9_]*"
            )
    return cols

#: Bound on snapshot reopen attempts when a concurrent WAL checkpoint fences a
#: streaming scan mid-drain. Mirrors the RO-pool's borrow()-level bound; a real
#: fence resolves on one reopen, the bound only stops a pathological flap.
_MAX_DRAIN_FENCE_ATTEMPTS = 8


class HippoTableList:

    def __init__(self, tables: list[str]) -> None:
        self.tables: list[str] = tables

    def __iter__(self) -> Iterator[str]:
        return iter(self.tables)

    def __repr__(self) -> str:  # pragma: no cover
        return f"HippoTableList(tables={self.tables!r})"


_PA_TO_SQLITE: dict[str, str] = {
    "int8": "INTEGER",
    "int16": "INTEGER",
    "int32": "INTEGER",
    "int64": "INTEGER",
    "uint8": "INTEGER",
    "uint16": "INTEGER",
    "uint32": "INTEGER",
    "uint64": "INTEGER",
    "float16": "REAL",
    "float32": "REAL",
    "float64": "REAL",
    "bool": "INTEGER",
    "string": "TEXT",
    "large_string": "TEXT",
    "binary": "BLOB",
    "large_binary": "BLOB",
}


def _pa_type_to_sqlite(t: _schema.DataType) -> str:
    type_str = str(t)
    if type_str in _PA_TO_SQLITE:
        return _PA_TO_SQLITE[type_str]
    if _schema.types.is_integer(t):
        return "INTEGER"
    if _schema.types.is_floating(t):
        return "REAL"
    if _schema.types.is_boolean(t):
        return "INTEGER"
    if _schema.types.is_list(t) or _schema.types.is_large_list(t):
        return "BLOB"
    if _schema.types.is_timestamp(t):
        return "TEXT"
    return "TEXT"


_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
    "detail_level",
})

_STRICT_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
})


def _sqlite_type_to_pa(col_name: str, type_str: str, embed_dim: int) -> _schema.DataType:
    t_upper = type_str.upper()
    if col_name == "embedding":
        return _schema.list_(_schema.float32(), embed_dim)
    if col_name in _STRICT_BOOL_COLUMNS:
        return _schema.bool_()
    if t_upper in ("TEXT",):
        return _schema.string()
    if t_upper in ("REAL",):
        return _schema.float32()
    if t_upper in ("INTEGER",):
        return _schema.int64()
    if t_upper in ("BLOB",):
        return _schema.binary()
    return _schema.string()


_DDL_RECORDS = """\
CREATE TABLE IF NOT EXISTS records (
    vec_label       INTEGER PRIMARY KEY AUTOINCREMENT,
    id              TEXT NOT NULL UNIQUE,
    tier            TEXT NOT NULL,
    literal_surface TEXT,
    aaak_index      TEXT,
    embedding       BLOB NOT NULL,
    structure_hv    BLOB,
    community_id    TEXT,
    centrality      REAL,
    detail_level    INTEGER,
    pinned          INTEGER,
    stability       REAL,
    difficulty      REAL,
    last_reviewed   TEXT,
    never_decay     INTEGER,
    never_merge     INTEGER,
    tombstoned_at   TEXT,
    schema_bypass   INTEGER,
    labile_until    TEXT,
    provenance_json TEXT,
    created_at      TEXT NOT NULL,
    updated_at      TEXT,
    tags_json       TEXT,
    language        TEXT,
    s5_trust_score  REAL,
    profile_modulation_gain_json TEXT,
    schema_version  INTEGER DEFAULT 1,
    wing            TEXT,
    room            TEXT,
    drawer          TEXT,
    valence         REAL DEFAULT 0.0,
    hv_tier              TEXT NOT NULL DEFAULT 'bsc',
    structure_hv_payload BLOB NOT NULL DEFAULT x'',
    embedding_pending    INTEGER NOT NULL DEFAULT 0,
    role                 TEXT,
    live                 INTEGER
)"""

_DDL_RECORDS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_records_id        ON records(id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tier      ON records(tier)",
    "CREATE INDEX IF NOT EXISTS idx_records_community ON records(community_id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tomb      ON records(tombstoned_at) WHERE tombstoned_at IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_records_pending   ON records(embedding_pending) WHERE embedding_pending=1",
    "CREATE INDEX IF NOT EXISTS idx_records_live      ON records(live)",
    "CREATE INDEX IF NOT EXISTS idx_records_created_at ON records(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_records_role      ON records(role)",
    "CREATE INDEX IF NOT EXISTS idx_records_vec_label ON records(vec_label)",
]

_DDL_EDGES = """\
CREATE TABLE IF NOT EXISTS edges (
    src         TEXT NOT NULL,
    dst         TEXT NOT NULL,
    edge_type   TEXT NOT NULL,
    weight      REAL NOT NULL DEFAULT 0.0,
    updated_at  TEXT,
    PRIMARY KEY (src, dst, edge_type)
)"""

_DDL_EDGES_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst)",
    "CREATE INDEX IF NOT EXISTS idx_edges_type ON edges(edge_type)",
]

_DDL_EVENTS = """\
CREATE TABLE IF NOT EXISTS events (
    id              TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    severity        TEXT,
    domain          TEXT,
    ts              TEXT NOT NULL,
    data_json       TEXT,
    session_id      TEXT,
    source_ids_json TEXT
)"""

_DDL_EVENTS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_events_kind    ON events(kind)",
    "CREATE INDEX IF NOT EXISTS idx_events_ts      ON events(ts)",
    "CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id)",
    # Composite index serving `WHERE kind = ? ORDER BY ts DESC LIMIT ?`
    # (query_events). On the stdlib driver this removes the ORDER BY
    # filesort. On the native engine, multi-column CREATE INDEX captures
    # only the leading column in its catalog, so this degrades to a
    # leading-column (kind) index there -- a no-harm add, not a filesort
    # removal on that driver.
    "CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts)",
]

_DDL_BUDGET_LEDGER = """\
CREATE TABLE IF NOT EXISTS budget_ledger (
    date        TEXT,
    usd_spent   REAL,
    kind        TEXT,
    ts          TEXT
)"""

_DDL_BUDGET_LEDGER_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_budget_date_kind ON budget_ledger(date, kind)",
]

_DDL_RATELIMIT_LEDGER = """\
CREATE TABLE IF NOT EXISTS ratelimit_ledger (
    ts          TEXT,
    status_code INTEGER,
    endpoint    TEXT
)"""

_DDL_HIPPO_META = """\
CREATE TABLE IF NOT EXISTS _hippo_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
)"""

_DDL_RECORD_TAGS = """\
CREATE TABLE IF NOT EXISTS record_tags (
    record_id TEXT NOT NULL,
    tag       TEXT NOT NULL,
    UNIQUE(record_id, tag)
)"""

_DDL_RECORD_TAGS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_record_tags_tag ON record_tags(tag)",
]


_TABLE_SQL: dict[str, dict[str, str]] = {
    "records": {
        "count":          "SELECT COUNT(*) FROM records",
        "select_all":     "SELECT * FROM records",
        "delete_prefix":  "DELETE FROM records WHERE ",
        "pragma":         "PRAGMA table_info(records)",
        "update_prefix":  "UPDATE records SET ",
        "insert_prefix":  "INSERT INTO records ",
        "alter_prefix":   "ALTER TABLE records ADD COLUMN ",
    },
    "edges": {
        "count":          "SELECT COUNT(*) FROM edges",
        "select_all":     "SELECT * FROM edges",
        "delete_prefix":  "DELETE FROM edges WHERE ",
        "pragma":         "PRAGMA table_info(edges)",
        "update_prefix":  "UPDATE edges SET ",
        "insert_prefix":  "INSERT INTO edges ",
        "alter_prefix":   "ALTER TABLE edges ADD COLUMN ",
    },
    "events": {
        "count":          "SELECT COUNT(*) FROM events",
        "select_all":     "SELECT * FROM events",
        "delete_prefix":  "DELETE FROM events WHERE ",
        "pragma":         "PRAGMA table_info(events)",
        "update_prefix":  "UPDATE events SET ",
        "insert_prefix":  "INSERT INTO events ",
        "alter_prefix":   "ALTER TABLE events ADD COLUMN ",
    },
    "budget_ledger": {
        "count":          "SELECT COUNT(*) FROM budget_ledger",
        "select_all":     "SELECT * FROM budget_ledger",
        "delete_prefix":  "DELETE FROM budget_ledger WHERE ",
        "pragma":         "PRAGMA table_info(budget_ledger)",
        "update_prefix":  "UPDATE budget_ledger SET ",
        "insert_prefix":  "INSERT INTO budget_ledger ",
        "alter_prefix":   "ALTER TABLE budget_ledger ADD COLUMN ",
    },
    "ratelimit_ledger": {
        "count":          "SELECT COUNT(*) FROM ratelimit_ledger",
        "select_all":     "SELECT * FROM ratelimit_ledger",
        "delete_prefix":  "DELETE FROM ratelimit_ledger WHERE ",
        "pragma":         "PRAGMA table_info(ratelimit_ledger)",
        "update_prefix":  "UPDATE ratelimit_ledger SET ",
        "insert_prefix":  "INSERT INTO ratelimit_ledger ",
        "alter_prefix":   "ALTER TABLE ratelimit_ledger ADD COLUMN ",
    },
    "_hippo_meta": {
        "count":          "SELECT COUNT(*) FROM _hippo_meta",
        "select_all":     "SELECT * FROM _hippo_meta",
        "delete_prefix":  "DELETE FROM _hippo_meta WHERE ",
        "pragma":         "PRAGMA table_info(_hippo_meta)",
        "update_prefix":  "UPDATE _hippo_meta SET ",
        "insert_prefix":  "INSERT INTO _hippo_meta ",
        "alter_prefix":   "ALTER TABLE _hippo_meta ADD COLUMN ",
    },
    "record_tags": {
        "count":          "SELECT COUNT(*) FROM record_tags",
        "select_all":     "SELECT * FROM record_tags",
        "delete_prefix":  "DELETE FROM record_tags WHERE ",
        "pragma":         "PRAGMA table_info(record_tags)",
        "update_prefix":  "UPDATE record_tags SET ",
        "insert_prefix":  "INSERT INTO record_tags ",
        "alter_prefix":   "ALTER TABLE record_tags ADD COLUMN ",
    },
}


def _encode_embedding(vec: list[float] | np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.array(vec, dtype=np.float32).tobytes()


#: Buffer types a stored BLOB cell may arrive as across drivers. The stdlib
#: sqlite3 driver returns BLOBs as ``bytes``; the projected-scan zero-copy path
#: of the engine driver returns them as ``memoryview`` slices into the mapped
#: page. ``np.frombuffer`` reads all three losslessly, so the embedding decode
#: must accept the whole set — otherwise a memoryview falls through the decode
#: gate and is consumed downstream as a raw byte sequence (a 384-float vector
#: mis-read as its 1536 raw bytes), corrupting the recall cosine math.
_BLOB_TYPES = (bytes, bytearray, memoryview)


def _decode_embedding(blob: "bytes | bytearray | memoryview | None") -> list[float] | None:
    if blob is None:
        return None
    if isinstance(blob, memoryview) and (blob.format not in ("B", "b", "c") or blob.itemsize != 1):
        # Non-byte-format or strided memoryview: cast to a flat byte view so any
        # future typed/strided view raises loudly (TypeError/ValueError from cast)
        # rather than silently mis-decoding. Zero cost on the real format-'B' path.
        blob = blob.cast("B")
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _encode_row_for_insert(row: dict) -> dict:
    out = dict(row)
    if "embedding" in out and out["embedding"] is not None:
        out["embedding"] = _encode_embedding(out["embedding"])
    return out


def _tags_from_json(tags_json: "str | None") -> list[str]:
    """Parse a ``tags_json`` cell into a list of tag strings.

    Malformed or empty input yields an empty list rather than raising, so a
    write-seam tag-index maintenance call never aborts the record write it is
    attached to.
    """
    if not tags_json:
        return []
    import json as _json
    try:
        parsed = _json.loads(tags_json)
    except (ValueError, TypeError):
        return []
    if not isinstance(parsed, list):
        return []
    return [t for t in parsed if isinstance(t, str)]


def _upsert_record_tags(
    conn: Any,
    record_id: str,
    tags_json: "str | None",
) -> None:
    """Upsert ``record_tags`` rows for one record's tag list.

    Must be called while the caller already holds ``_conn_lock`` (and, for the
    ``records`` insert path, ``_hnsw_lock``) and is inside the same logical
    write transaction as the record write it accompanies — this function does
    not acquire any lock itself and does not commit. Uses
    ``ON CONFLICT (record_id, tag) DO NOTHING`` (the record_tags table is
    all-key columns, so there is nothing to update on a conflict) — never a
    plain ``INSERT`` that would raise on a duplicate.

    Failures are swallowed: a tag-index maintenance fault must not abort the
    record write it accompanies. ``find_record_by_tag`` degrades to the
    LIKE-scan fallback if the index ever falls out of sync (differential
    tested), so a missed upsert here is a bounded correctness risk, not a
    crash.
    """
    tags = _tags_from_json(tags_json)
    if not tags:
        return
    try:
        conn.executemany(
            "INSERT INTO record_tags (record_id, tag) VALUES (?, ?)"
            " ON CONFLICT (record_id, tag) DO NOTHING",
            [(record_id, tag) for tag in tags],
        )
    except Exception:  # noqa: BLE001 -- tag-index maintenance must not abort the write
        _log.debug("record_tags upsert failed for record_id=%s", record_id, exc_info=True)


def _delete_record_tags(conn: Any, record_id: str) -> None:
    """Delete every ``record_tags`` row for ``record_id``.

    Called only from a HARD delete of the record itself (``HippoTable.delete``
    on the ``records`` table) — never from a soft tombstone, which does not
    remove the row from ``records`` and must not remove it from the tag index
    either (matches the pre-existing ``find_record_by_tag`` LIKE-scan, which
    never filtered on ``tombstoned_at``). Must be called while the caller
    already holds ``_conn_lock`` and is inside the same transaction as the
    record delete. Failures are swallowed for the same reason as
    ``_upsert_record_tags``.
    """
    try:
        conn.execute("DELETE FROM record_tags WHERE record_id = ?", (record_id,))
    except Exception:  # noqa: BLE001 -- tag-index maintenance must not abort the delete
        _log.debug("record_tags delete failed for record_id=%s", record_id, exc_info=True)


def _decode_df_embedding(df: pd.DataFrame) -> pd.DataFrame:
    if "embedding" in df.columns:
        df = df.copy()
        df["embedding"] = df["embedding"].apply(
            lambda b: _decode_embedding(b) if isinstance(b, _BLOB_TYPES) else b
        )
    return df


def _decode_raw_row_embedding(row: dict) -> dict:
    """Row-dict twin of ``_decode_df_embedding`` for a plain dict fetch (no
    DataFrame). ``_from_row`` does not decode the embedding BLOB itself
    (it does ``list(row["embedding"])``), so any row-dict fed to it must
    arrive with the embedding column already turned into a plain float list —
    otherwise a 384-d vector is mis-read as its 1536 raw bytes.
    """
    emb_raw = row.get("embedding")
    if isinstance(emb_raw, _BLOB_TYPES):
        decoded = _decode_embedding(emb_raw)
        if decoded is not None:
            row = dict(row)
            row["embedding"] = decoded
    return row


def _rows_to_df(
    cursor: Any,
    rows: "list",
) -> pd.DataFrame:
    """Build a DataFrame from a raw cursor fetch without pandas' SQL introspection.

    ``pd.read_sql_query`` re-issues type discovery and emits a DBAPI-compat
    warning on every call; on the recall hot path that per-call materialization
    is the dominant cost. Constructing the frame directly from the cursor
    description + fetched rows yields the same columns (in select order) and the
    same row values, while skipping the introspection round-trip. The embedding
    BLOB is decoded by ``_decode_df_embedding`` exactly as on the read_sql path.
    """
    if cursor.description is None:
        return pd.DataFrame()
    columns = [d[0] for d in cursor.description]
    if not rows:
        return pd.DataFrame(columns=columns)
    # Materialise each row as a positional value tuple. Both row factories
    # (stdlib ``sqlite3.Row`` and the engine's ``LilliBrainRow``) iterate as
    # values in column-declaration order, so ``tuple(r)`` is the column-aligned
    # value sequence. Going through tuples (rather than handing dict-subclass
    # rows to ``from_records``) keeps the column projection identical on both
    # drivers regardless of pandas' record-vs-mapping detection.
    df = pd.DataFrame.from_records([tuple(r) for r in rows], columns=columns)
    return _decode_df_embedding(df)


def _decrypt_df_columns(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    decrypt_fn: "Callable[[str, str, str], str]",
) -> pd.DataFrame:
    active_cols = [c for c in columns if c in df.columns]
    if not active_cols or df.empty or "id" not in df.columns:
        return df
    df = df.copy()
    for col in active_cols:
        decrypted: list = []
        for _, row in df.iterrows():
            val = row[col]
            uid = str(row["id"])
            if val is None or not isinstance(val, str):
                decrypted.append(val)
            else:
                decrypted.append(decrypt_fn(uid, col, val))
        df[col] = decrypted
    return df


def _normalize_to_row_list(data: Any) -> list[dict]:
    if data is None:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, pd.DataFrame):
        return data.to_dict(orient="records")
    # Duck-typed row carrier: our own Table/RecordBatch and any external
    # columnar table alike expose to_pylist() -> list[dict].
    to_pylist = getattr(data, "to_pylist", None)
    if callable(to_pylist):
        return to_pylist()
    return list(data)


class HippoTable:

    def __init__(
        self,
        conn: Any,
        name: str,
        *,
        embed_dim: int,
        db: "Any | None" = None,
        ann_index: Any = None,
    ) -> None:
        from iai_mcp.hippo import _validate_table_name
        self._name = _validate_table_name(name)
        self._conn = conn
        self._embed_dim = embed_dim
        self._db: "Any | None" = db
        self._ann_index = ann_index
        self._sql: dict[str, str] | None = _TABLE_SQL.get(self._name)

    def _stmt(self, key: str) -> str:
        if self._sql is not None:
            return self._sql[key]
        raise KeyError(f"No pre-built SQL for key {key!r} on dynamic table {self._name!r}")


    def count_rows(self, filter: str | None = None) -> int:  # noqa: A002
        from iai_mcp.hippo import HippoIntegrityError
        if self._sql is not None:
            base = self._sql["count"]
        else:
            base = "SELECT COUNT(*) FROM " + self._name
        stmt = (base + " WHERE " + filter) if filter else base
        if self._db is not None:
            with self._db.ro_conn() as conn:
                row = conn.execute(stmt).fetchone()
        else:
            row = self._conn.execute(stmt).fetchone()
        if row is None:
            raise HippoIntegrityError(
                f"count_rows({self._name!r}): SELECT COUNT(*) returned no row — "
                f"connection may be in an error state. "
                f"(filter={filter!r}, in_transaction={getattr(self._conn, 'in_transaction', '?')})",
            )
        return int(row[0])

    def to_pandas(self) -> pd.DataFrame:
        if self._sql is not None:
            stmt = self._sql["select_all"]
        else:
            stmt = "SELECT * FROM " + self._name
        if self._db is not None:
            with self._db.ro_conn() as conn:
                cur = conn.execute(stmt)
                fetched = cur.fetchall()
        else:
            cur = self._conn.execute(stmt)
            fetched = cur.fetchall()
        return _rows_to_df(cur, fetched)

    def _decrypt_df(self, df: pd.DataFrame) -> pd.DataFrame:
        from iai_mcp.hippo import _ENCRYPTED_RECORD_COLUMNS, _ENCRYPTED_EVENTS_COLUMNS
        if self._db is None or self._db._crypto_key_provider is None:
            return df
        if self._name == "records":
            return _decrypt_df_columns(
                df, _ENCRYPTED_RECORD_COLUMNS, self._db._decrypt_record_field
            )
        if self._name == "events":
            return _decrypt_df_columns(
                df, _ENCRYPTED_EVENTS_COLUMNS, self._db._decrypt_event_field
            )
        return df

    def _encrypt_rows(self, rows: list[dict]) -> list[dict]:
        from iai_mcp.hippo import _ENCRYPTED_RECORD_COLUMNS, _ENCRYPTED_EVENTS_COLUMNS
        if self._db is None or self._db._crypto_key_provider is None:
            return rows
        if self._name == "records":
            enc_cols = _ENCRYPTED_RECORD_COLUMNS
        elif self._name == "events":
            enc_cols = _ENCRYPTED_EVENTS_COLUMNS
        else:
            return rows
        result = []
        for row in rows:
            uid = str(row.get("id", ""))
            if not uid:
                result.append(row)
                continue
            new_row = dict(row)
            for col in enc_cols:
                val = new_row.get(col)
                if val is not None:
                    new_row[col] = self._db._encrypt_for_uuid(uid, val)
            result.append(new_row)
        return result

    def search(self, vector: Any = None, **kwargs: Any) -> "HippoQuery":
        if vector is None:
            return HippoQuery(
                self._conn,
                self._name,
                embed_dim=self._embed_dim,
                db=self._db,
            )

        if self._db is None:
            raise NotImplementedError("ANN search requires a HippoDB reference")

        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        return HippoQuery(
            self._conn,
            self._name,
            embed_dim=self._embed_dim,
            ann_vector=vec,
            ann_db=self._db,
            db=self._db,
        )

    def list_versions(self) -> list[dict]:
        return [{"version": 1, "ts": datetime.now(timezone.utc).isoformat()}]

    def optimize(
        self,
        cleanup_older_than: Any = None,
        delete_unverified: bool = False,
        **kwargs: Any,
    ) -> dict:
        return {"compaction": "noop_hippo"}


    def add(self, rows: Any) -> None:
        from iai_mcp.hippo import _txn, HNSW_SAVE_INTERVAL
        row_list = _normalize_to_row_list(rows)
        if not row_list:
            return
        row_list = self._encrypt_rows(row_list)
        encoded = [_encode_row_for_insert(r) for r in row_list]
        cols = _validate_columns(list(encoded[0].keys()))
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        if self._sql is not None:
            stmt = self._sql["insert_prefix"] + "(" + col_names + ") VALUES (" + placeholders + ")"
        else:
            stmt = "INSERT INTO " + self._name + " (" + col_names + ") VALUES (" + placeholders + ")"

        if self._name == "records" and self._db is not None:
            db = self._db
            with db._hnsw_lock:
                with db._conn_lock:
                    with _txn(self._conn):
                        for r, enc in zip(row_list, encoded):
                            cursor = self._conn.execute(stmt, tuple(enc.get(c) for c in cols))
                            vec_label = int(cursor.lastrowid)
                            emb_raw = r.get("embedding")
                            if emb_raw is not None:
                                emb_vec = np.array(emb_raw, dtype=np.float32).reshape(1, -1)
                                db._hnsw.add_items(emb_vec, np.array([vec_label], dtype=np.int64))
                                if db._ann_journal is not None:
                                    # A rebuild is between snapshot and swap:
                                    # journal this add so the swapped-in buffer
                                    # AND its label map replay it (else the
                                    # record vanishes from ANN until the next
                                    # rebuild).
                                    db._ann_journal.append(
                                        ("add", emb_vec, vec_label, str(r["id"]))
                                    )
                                db._label_map[str(r["id"])] = vec_label
                                db._write_counter += 1
                            db._maybe_resize()
                            _upsert_record_tags(self._conn, str(r["id"]), r.get("tags_json"))
                if db._write_counter > 0 and db._write_counter % HNSW_SAVE_INTERVAL == 0:
                    # Time-throttled: the save serializes the WHOLE index under
                    # _hnsw_lock, so a large drain hitting the count interval
                    # every 200 rows would stall concurrent recalls repeatedly.
                    # The disk file is a boot accelerator, not the source of
                    # truth — five-minute granularity loses nothing durable.
                    import time as _t
                    _last = getattr(db, "_last_periodic_save_mono", 0.0)
                    if (_t.monotonic() - _last) >= 300.0:
                        db._save_index_atomic()
                        db._last_periodic_save_mono = _t.monotonic()
        else:
            if self._db is not None:
                lock_ctx = self._db._conn_lock
            else:
                lock_ctx = contextlib.nullcontext()
            with lock_ctx:
                with _txn(self._conn):
                    self._conn.executemany(stmt, [tuple(r.get(c) for c in cols) for r in encoded])

    def update(self, where: str, values: dict[str, Any]) -> None:
        from iai_mcp.hippo import _txn, _ENCRYPTED_RECORD_COLUMNS, _ENCRYPTED_EVENTS_COLUMNS
        if not values:
            return

        enc_cols: tuple[str, ...] = ()
        if self._db is not None and self._db._crypto_key_provider is not None:
            if self._name == "records":
                enc_cols = _ENCRYPTED_RECORD_COLUMNS
            elif self._name == "events":
                enc_cols = _ENCRYPTED_EVENTS_COLUMNS

        encrypted_being_updated = [c for c in values if c in enc_cols]
        if encrypted_being_updated:
            match = re.search(r"""id\s*=\s*['"]([^'"]+)['"]""", where)
            if match is None:
                raise ValueError(
                    f"Encrypted column(s) {encrypted_being_updated!r} can only be "
                    f"updated with an id-keyed WHERE clause (e.g. \"id = '<uuid>'\") "
                    f"for AAD binding. Received WHERE: {where!r}"
                )
            uuid_str = match.group(1)
            encrypted_values = dict(values)
            for col in encrypted_being_updated:
                encrypted_values[col] = self._db._encrypt_for_uuid(
                    uuid_str, encrypted_values[col]
                )
            values = encrypted_values

        encoded_values: dict = {}
        for col, val in values.items():
            if col == "embedding" and isinstance(val, (list, np.ndarray)):
                encoded_values[col] = _encode_embedding(val)
            else:
                encoded_values[col] = val

        _validate_columns(list(encoded_values))
        set_clause = ", ".join(col + "=?" for col in encoded_values)
        if self._sql is not None:
            stmt = self._sql["update_prefix"] + set_clause + " WHERE " + where
        else:
            stmt = "UPDATE " + self._name + " SET " + set_clause + " WHERE " + where
        _lock_m1 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m1:
            with _txn(self._conn):
                self._conn.execute(stmt, list(encoded_values.values()))

    def update_many_by_id(self, rows: "list[tuple[str, dict[str, Any]]]") -> int:
        """Update many rows by id under ONE durable commit.

        Every statement is `... WHERE id = ?` because each row carries a
        distinct SET payload (heterogeneous per-row values), which no
        set-based UPDATE can express; the id-keyed statement rides the
        engine's id-index point lookup, and the single wrapping transaction
        pays one fsync for the whole batch instead of one per statement.
        Encryption semantics match
        update(): encrypted columns are AAD-bound to the row id. Column
        names are validated against the table's real column set BEFORE the
        transaction opens — the engine silently accepts an unknown SET
        column, and at bulk scale that would be a silently-empty batch
        reporting full success. `embedding` is refused: this seam performs
        no ANN index maintenance, and a bulk write through it would desync
        the vector index.

        Returns the sum of engine-reported matched-row counts; a driver
        without a usable rowcount degrades the sum to a per-statement
        upper bound.
        """
        from iai_mcp.hippo import _txn, _ENCRYPTED_RECORD_COLUMNS, _ENCRYPTED_EVENTS_COLUMNS
        if not rows:
            return 0

        enc_cols: tuple[str, ...] = ()
        if self._db is not None and self._db._crypto_key_provider is not None:
            if self._name == "records":
                enc_cols = _ENCRYPTED_RECORD_COLUMNS
            elif self._name == "events":
                enc_cols = _ENCRYPTED_EVENTS_COLUMNS

        known_cols = {f.name for f in self.schema}
        requested_cols: set[str] = set()
        for _rid, values in rows:
            requested_cols.update(values)
        unknown = sorted(requested_cols - known_cols)
        if unknown:
            raise ValueError(
                f"update_many_by_id: unknown column(s) {unknown!r} for table "
                f"{self._name!r}"
            )
        if "embedding" in requested_cols:
            raise ValueError(
                "update_many_by_id refuses 'embedding': the bulk seam does "
                "no ANN index maintenance"
            )

        prefix = (
            self._sql["update_prefix"]
            if self._sql is not None
            else "UPDATE " + self._name + " SET "
        )
        prepared: list[tuple[str, str, list[Any]]] = []
        for rid, values in rows:
            if not values:
                continue
            encoded: dict[str, Any] = {}
            for col, val in values.items():
                if col in enc_cols:
                    val = self._db._encrypt_for_uuid(str(rid), val)
                encoded[col] = val
            _validate_columns(list(encoded))
            set_clause = ", ".join(col + "=?" for col in encoded)
            prepared.append(
                (str(rid), prefix + set_clause + " WHERE id = ?", list(encoded.values()))
            )
        if not prepared:
            return 0

        matched = 0
        _lock_m3 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m3:
            with _txn(self._conn):
                for rid, stmt, vals in prepared:
                    cur = self._conn.execute(stmt, [*vals, rid])
                    rc = getattr(cur, "rowcount", -1)
                    if rc is None or rc < 0:
                        matched += 1
                    else:
                        matched += int(rc)
        return matched

    def delete(self, where: str) -> None:
        from iai_mcp.hippo import _txn
        if self._name == "records" and self._db is not None:
            db = self._db
            with db._hnsw_lock:
                sel_sql = "SELECT id, vec_label FROM records WHERE " + where
                del_sql = "DELETE FROM records WHERE " + where
                with db._conn_lock:
                    affected = self._conn.execute(sel_sql).fetchall()
                    with _txn(self._conn):
                        self._conn.execute(del_sql)
                        for row in affected:
                            _delete_record_tags(self._conn, str(row["id"]))
                for row in affected:
                    label = int(row["vec_label"])
                    try:
                        db._hnsw.mark_deleted(label)
                    except RuntimeError:
                        pass
                    if db._ann_journal is not None:
                        # Journal the delete so an in-flight rebuild's snapshot
                        # (which may still contain this row) does not resurrect
                        # it in the swapped-in buffer or label map.
                        db._ann_journal.append(("del", None, label, str(row["id"])))
                    db._label_map.pop(str(row["id"]), None)
            return

        if self._sql is not None:
            stmt = self._sql["delete_prefix"] + where
        else:
            stmt = "DELETE FROM " + self._name + " WHERE " + where
        _lock_m2 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m2:
            with _txn(self._conn):
                self._conn.execute(stmt)

    def merge_insert(self, key_cols: str | list[str]) -> "HippoMergeInsert":
        if isinstance(key_cols, str):
            key_cols = [key_cols]
        return HippoMergeInsert(self, list(key_cols))


    @property
    def schema(self) -> _schema.Schema:
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        fields: list[_schema.Field] = []
        for row in pragma_rows:
            col_name = row["name"]
            type_str = row["type"] if row["type"] else "TEXT"
            pa_type = _sqlite_type_to_pa(col_name, type_str, self._embed_dim)
            nullable = not bool(row["notnull"])
            fields.append(_schema.field(col_name, pa_type, nullable=nullable))
        return _schema.schema(fields)

    def add_columns(self, fields: list[_schema.Field]) -> None:
        from iai_mcp.hippo import _validate_table_name
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
            alter_prefix = self._sql["alter_prefix"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"
            alter_prefix = "ALTER TABLE " + self._name + " ADD COLUMN "
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        existing = {row["name"] for row in _pragma_rows}
        for f in fields:
            if f.name in existing:
                continue
            sqlite_type = _pa_type_to_sqlite(f.type)
            col_name = _validate_table_name(f.name)
            self._conn.execute(alter_prefix + col_name + " " + sqlite_type)
            existing.add(f.name)

    def drop_columns(self, column_names: list[str]) -> None:
        from iai_mcp.hippo import _validate_table_name
        if _sqlite_stdlib.is_stdlib_connection(self._conn):
            version = _sqlite_stdlib.sqlite_version()
            major, minor, _ = (int(x) for x in version.split("."))
            if (major, minor) < (3, 35):
                raise RuntimeError(
                    f"ALTER TABLE DROP COLUMN requires SQLite >= 3.35; "
                    f"installed: {version}"
                )
        if self._sql is not None:
            pragma_stmt = self._sql["pragma"]
        else:
            pragma_stmt = "PRAGMA table_info(" + self._name + ")"
        drop_prefix = "ALTER TABLE " + self._name + " DROP COLUMN "
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        else:
            _pragma_rows = self._conn.execute(pragma_stmt).fetchall()
        existing = {row["name"] for row in _pragma_rows}
        for col in column_names:
            if col not in existing:
                continue
            col_name = _validate_table_name(col)
            self._conn.execute(drop_prefix + col_name)
            existing.discard(col)


class HippoQuery:

    def __init__(
        self,
        conn: Any,
        table_name: str,
        *,
        embed_dim: int,
        ann_vector: "np.ndarray | None" = None,
        ann_db: "Any | None" = None,
        db: "Any | None" = None,
    ) -> None:
        from iai_mcp.hippo import _validate_table_name
        self._conn = conn
        self._table_name = _validate_table_name(table_name)
        self._embed_dim = embed_dim
        self._where_clauses: list[str] = []
        self._select_cols: list[str] | None = None
        self._limit_val: int | None = None
        self._ann_vector: "np.ndarray | None" = ann_vector
        self._ann_db: "Any | None" = ann_db
        self._db: "Any | None" = db if db is not None else ann_db

    def where(self, predicate: str) -> "HippoQuery":
        self._where_clauses.append(predicate)
        return self

    def select(self, columns: list[str]) -> "HippoQuery":
        self._select_cols = list(columns)
        return self

    def limit(self, n: int) -> "HippoQuery":
        self._limit_val = n
        return self

    def distance_type(self, metric: str) -> "HippoQuery":
        return self


    def _build_sql(self) -> str:
        col_clause = (
            ", ".join(self._select_cols) if self._select_cols else "*"
        )
        sql = f"SELECT {col_clause} FROM {self._table_name}"
        if self._where_clauses:
            sql += " WHERE " + " AND ".join(
                f"({clause})" for clause in self._where_clauses
            )
        if self._limit_val is not None:
            sql += f" LIMIT {self._limit_val}"
        return sql

    def to_pandas(self) -> pd.DataFrame:
        if self._ann_vector is not None and self._ann_db is not None:
            return self._ann_to_pandas()
        sql = self._build_sql()
        if self._db is not None:
            with self._db.ro_conn() as conn:
                cur = conn.execute(sql)
                fetched = cur.fetchall()
        else:
            cur = self._conn.execute(sql)
            fetched = cur.fetchall()
        return _rows_to_df(cur, fetched)

    def _ann_knn_fetch_core(
        self,
    ) -> "tuple[Any, list, dict[int, float]] | None":
        """Run the knn_query + vec_label row fetch shared by every ANN output
        shape (DataFrame or row-dict). Returns ``None`` on every "no
        neighbours" outcome (empty index, RuntimeError, empty label result),
        matching ``_ann_to_pandas``'s empty-DataFrame returns exactly so both
        output shapes degrade identically.
        """
        db = self._ann_db
        k = self._limit_val if self._limit_val is not None else 10

        with db._hnsw_lock:
            active_count = len(db._label_map)
            if active_count == 0:
                return None
            # Clamp k to the live index count, not just the label_map count —
            # the two can diverge briefly during boot/rebuild, and asking an
            # index for more neighbors than it holds is an error condition on
            # some backends.
            hnsw_count = db._hnsw.get_current_count()
            k_clamped = min(k, active_count, hnsw_count)
            if k_clamped == 0:
                return None
            # ef handling survives from the ANN era: the exact index accepts
            # and ignores set_ef, so this block is a no-op there, but the
            # restore-after-query pairing stays (same lock block = atomic) in
            # case a graph-ANN backend ever returns at larger corpus scale.
            try:
                _cur_ef = int(getattr(db._hnsw, "ef", 0))
            except (AttributeError, TypeError, ValueError):
                _cur_ef = 0
            _ef_was_set = False
            if k_clamped > _cur_ef:
                try:
                    db._hnsw.set_ef(k_clamped)
                    _ef_was_set = True
                except (AttributeError, RuntimeError):
                    pass
            try:
                labels = None
                distances = None
                _k_try = k_clamped
                _last_exc: "RuntimeError | None" = None
                # Degrade k instead of dying: on a delete-churned index the
                # graph can be too fragmented to reach k neighbors ("Cannot
                # return the results in a contiguous 2D array") even with
                # ef >= k. Half the requested neighbors beat an EMPTY lane —
                # an empty return silently drops ANN from recall (both the
                # ranking quality and the escalation paths) until the next
                # full rebuild.
                while _k_try >= 1:
                    try:
                        labels, distances = db._hnsw.knn_query(
                            self._ann_vector, k=_k_try
                        )
                        break
                    except RuntimeError as exc:
                        _last_exc = exc
                        _k_try //= 2
                if labels is None:
                    # Defensive: treat a query raise as "no neighbors found"
                    # so the insert path's pattern-separation gate can
                    # proceed instead of failing the whole capture.
                    _log.warning(
                        "vector-index knn_query failed at every k down from "
                        "%d (index_count=%d, active_count=%d); returning "
                        "empty result: %s",
                        k_clamped, hnsw_count, active_count, _last_exc,
                    )
                    return None
                if _k_try < k_clamped:
                    _log.warning(
                        "vector-index knn_query degraded k=%d -> %d: %s",
                        k_clamped, _k_try, _last_exc,
                    )
            finally:
                if _ef_was_set:
                    try:
                        from iai_mcp.hippo import RECALL_INDEX_EF
                        # Always restore once we raised ef. When the prior value
                        # was unreadable, fall back to a known-safe default rather
                        # than leaving the shared index at the elevated big-k ef,
                        # which would slow every later recall until the next rebuild.
                        db._hnsw.set_ef(_cur_ef if _cur_ef > 0 else RECALL_INDEX_EF)
                    except (AttributeError, RuntimeError):
                        pass

        flat_labels: list[int] = labels[0].tolist()
        # Clamp cosine distance to its mathematical range — the BLAS backend
        # can produce sub-epsilon negatives on Linux.
        flat_distances: list[float] = [
            max(0.0, min(2.0, float(d))) for d in distances[0]
        ]

        if not flat_labels:
            return None

        placeholders = ", ".join("?" for _ in flat_labels)
        sql = (  # nosemgrep: sql-injection
            f"SELECT * FROM {self._table_name} WHERE vec_label IN ({placeholders})"
        )
        if self._where_clauses:
            sql += " AND " + " AND ".join(f"({c})" for c in self._where_clauses)

        # This vec_label IN lookup runs through db.ro_conn() — a pooled
        # lock-free reader on the lilli driver (never touches _conn_lock), or
        # the shared self._conn under _conn_lock as a fallback (stdlib driver,
        # kill-switch, or any pool-acquisition failure) — NOT the
        # _open_snapshot_conn path (only to_batches uses that). When the
        # daemon is up self._conn is the read-write writer that already ran
        # _ensure_tables (index present); when the store is opened read-only
        # (daemon down / reader process) self._conn is the read-only engine
        # connection that gets the index via meta.replay.
        if db is not None:
            with db.ro_conn() as conn:
                cur = conn.execute(sql, flat_labels)
                fetched = cur.fetchall()
        else:
            cur = self._conn.execute(sql, flat_labels)
            fetched = cur.fetchall()
        # Best-effort operator signal: on a read-only open whose engine meta
        # lacks the vec_label index (un-upgraded store, no read-write boot yet),
        # this lookup full-scans — correct but slow. Log once, never disrupt.
        if db is not None:
            _emit_gap = getattr(db, "_debug_log_readonly_index_gap", None)
            if _emit_gap is not None:
                try:
                    _emit_gap(self._conn)
                except Exception:  # noqa: BLE001
                    pass

        dist_map: dict[int, float] = {
            int(lbl): float(d) for lbl, d in zip(flat_labels, flat_distances)
        }
        return cur, fetched, dist_map

    def _ann_to_pandas(self) -> pd.DataFrame:
        core = self._ann_knn_fetch_core()
        if core is None:
            return pd.DataFrame()
        cur, fetched, dist_map = core

        df = _rows_to_df(cur, fetched)
        if df.empty:
            return df

        df["_distance"] = df["vec_label"].apply(lambda lbl: dist_map.get(int(lbl), float("nan")))
        # kind="stable" (mergesort) preserves fetch order among tied distances
        # and matches the row-dict twin's stable Python sort exactly — the
        # default quicksort permutes tied blocks arbitrarily, which would make
        # the kill-switch / per-call fail-safe return a different record set
        # than the fast path for the identical cue whenever a tie straddles
        # the k boundary.
        df = df.sort_values("_distance", kind="stable").reset_index(drop=True)
        return df

    def to_row_dicts(self) -> "list[dict]":
        """Row-dict twin of ``_ann_to_pandas`` — same knn_query + vec_label
        fetch + distance mapping + ascending-distance sort, without building a
        pandas DataFrame. Each dict carries the embedding BLOB already decoded
        to a plain float list (mirroring ``_rows_to_df``'s
        ``_decode_df_embedding`` step) so it is a drop-in feed for
        ``MemoryStore._from_row``, which does not decode the BLOB itself.

        Only defined for the ANN branch (``self._ann_vector`` set) — the
        query_similar fast-decode caller is the sole consumer.
        """
        if self._ann_vector is None or self._ann_db is None:
            # Mirrors ``to_pandas``'s ANN-branch guard: a non-ANN query has no
            # hnsw index to knn_query against. Degrade to an empty result
            # instead of an AttributeError on ``None._hnsw_lock``.
            return []
        core = self._ann_knn_fetch_core()
        if core is None:
            return []
        cur, fetched, dist_map = core

        if cur.description is None or not fetched:
            return []
        columns = [d[0] for d in cur.description]
        rows: list[dict] = []
        for raw_row in fetched:
            values = tuple(raw_row)
            row = dict(zip(columns, values))
            row = _decode_raw_row_embedding(row)
            vec_label = row.get("vec_label")
            row["_distance"] = dist_map.get(int(vec_label), float("nan")) if vec_label is not None else float("nan")
            rows.append(row)

        # Python's list.sort() key comparisons treat NaN as neither less-than
        # nor greater-than anything, so a NaN key never moves during a stable
        # sort — it stays wherever it happened to land in fetch order. pandas'
        # sort_values (the DataFrame path this mirrors) explicitly places NaN
        # last (na_position="last" is its default). Sorting on
        # (is_nan, distance) reproduces that ordering exactly: every non-NaN
        # row sorts by its true distance first, and all NaN rows land after
        # them, in their original relative (stable) order.
        rows.sort(key=lambda r: (r["_distance"] != r["_distance"], r["_distance"]))
        return rows

    def _decrypt_query_df(self, df: pd.DataFrame) -> pd.DataFrame:
        from iai_mcp.hippo import _ENCRYPTED_RECORD_COLUMNS, _ENCRYPTED_EVENTS_COLUMNS
        if self._db is None or self._db._crypto_key_provider is None:
            return df
        if self._table_name == "records":
            return _decrypt_df_columns(
                df, _ENCRYPTED_RECORD_COLUMNS, self._db._decrypt_record_field
            )
        if self._table_name == "events":
            return _decrypt_df_columns(
                df, _ENCRYPTED_EVENTS_COLUMNS, self._db._decrypt_event_field
            )
        return df

    def to_batches(self, batch_size: int = 1000) -> Iterator[_schema.RecordBatch]:
        sql = self._build_sql()
        db = self._db
        db_path = getattr(db, "_db_path", None) if db is not None else None
        if db_path is not None:
            # Bounded-hold discipline: a full-corpus scan must NEVER hold the
            # shared ``_conn_lock`` across the consumer's per-batch work, or it
            # starves every concurrent writer / index rebuild for the whole scan.
            # Stream from a dedicated read-only snapshot connection on the same
            # WAL file instead — it observes the committed snapshot, holds NO
            # shared lock, and is closed when this generator is exhausted or
            # closed. This preserves lazy streaming (one fetch per yielded batch,
            # bounded memory) while restoring the bounded-hold invariant. The
            # snapshot sees ciphertext ``literal_surface`` BLOBs exactly as the
            # shared connection would; decryption stays above this layer.
            yield from self._drain_snapshot_with_fence_recovery(
                sql, batch_size, db_path
            )
        else:
            # Detached / pre-construction table (no HippoDB, no shared
            # connection to starve): drain the local cursor directly, unlocked.
            yield from self._drain_to_batches(sql, batch_size)

    def _drain_snapshot_with_fence_recovery(
        self, sql: str, batch_size: int, db_path: str
    ) -> Iterator[_schema.RecordBatch]:
        """Stream a snapshot scan, surviving checkpoint fences mid-drain.

        A concurrent WAL checkpoint invalidates an open read-only snapshot on
        the engine driver ("read-only snapshot invalidated"). Recovery reopens
        a fresh snapshot and re-executes the same statement; rows already
        yielded are skipped by their ``id`` so the consumer sees each row at
        most once. Recovery therefore requires ``id`` among the selected
        columns UNLESS nothing was yielded yet (a clean restart). A scan
        without ``id`` that fenced mid-drain re-raises — silently re-yielding
        rows would be worse than failing loud.
        """
        from iai_mcp.hippo._ro_pool import _is_snapshot_fence

        yielded_ids: set = set()
        rows_yielded = False
        dedup_possible = True
        attempts = 0
        while True:
            conn = self._open_snapshot_conn(db_path)
            try:
                for batch in self._drain_to_batches(sql, batch_size, conn=conn):
                    has_id = "id" in batch.schema.names
                    if has_id and yielded_ids:
                        keep = [
                            i for i, v in enumerate(batch.column("id").to_pylist())
                            if v not in yielded_ids
                        ]
                        if not keep:
                            continue
                        if len(keep) < batch.num_rows:
                            batch = batch.take(_schema.array(keep, type=_schema.int32()))
                    if has_id:
                        yielded_ids.update(batch.column("id").to_pylist())
                    else:
                        dedup_possible = False
                    rows_yielded = True
                    yield batch
                return
            except errors.OperationalError as exc:
                if not _is_snapshot_fence(exc):
                    raise
                attempts += 1
                if attempts > _MAX_DRAIN_FENCE_ATTEMPTS:
                    raise
                if rows_yielded and not dedup_possible:
                    raise
                _log.debug(
                    "snapshot scan fenced by concurrent checkpoint "
                    "(attempt %d/%d, %d rows already served) — reopening",
                    attempts, _MAX_DRAIN_FENCE_ATTEMPTS, len(yielded_ids),
                )
            finally:
                with contextlib.suppress(Exception):
                    conn.close()

    def _open_snapshot_conn(self, db_path: str) -> Any:
        """Open a short-lived read-only connection for a lock-free scan.

        Mirrors the audited daemon-down direct-access pattern: driver-aware
        open (engine raw connection on the lilli driver, else the stdlib
        driver), read-only, name-keyed row access. Holds no shared
        ``_conn_lock``.
        """
        from iai_mcp.hippo._raw_open import open_store_conn
        conn = open_store_conn(db_path, read_only=True)
        if conn is None:
            conn = _sqlite_stdlib.connect(
                db_path,
                check_same_thread=False,
                isolation_level=None,
            )
            conn.row_factory = _sqlite_stdlib.Row
        with contextlib.suppress(Exception):
            conn.execute("PRAGMA busy_timeout=2000")
        with contextlib.suppress(Exception):
            conn.execute("PRAGMA query_only=ON")
        return conn

    def _drain_to_batches(
        self, sql: str, batch_size: int, conn: Any = None
    ) -> Iterator[_schema.RecordBatch]:
        cursor = (conn if conn is not None else self._conn).execute(sql)
        try:
            column_names = [desc[0] for desc in cursor.description]
            while True:
                raw_rows = cursor.fetchmany(batch_size)
                if not raw_rows:
                    break
                data: dict[str, list] = {c: [] for c in column_names}
                for row in raw_rows:
                    for col in column_names:
                        data[col].append(row[col])
                if "embedding" in data:
                    data["embedding"] = [
                        _decode_embedding(b)
                        if isinstance(b, _BLOB_TYPES)
                        else b
                        for b in data["embedding"]
                    ]
                yield _schema.record_batch(data)
        finally:
            # A fenced snapshot can refuse even close(); never let that mask
            # the in-flight error the recovery wrapper needs to see.
            with contextlib.suppress(Exception):
                cursor.close()


class HippoMergeInsert:

    def __init__(self, table: HippoTable, key_cols: list[str]) -> None:
        self._table = table
        self._key_cols = key_cols
        self._update_all: bool = False

    def when_matched_update_all(self) -> "HippoMergeInsert":
        self._update_all = True
        return self

    def execute(self, data: Any) -> None:
        from iai_mcp.hippo import _txn
        rows = _normalize_to_row_list(data)
        if not rows:
            return

        rows = self._table._encrypt_rows(rows)
        encoded = [_encode_row_for_insert(r) for r in rows]
        all_cols = _validate_columns(list(encoded[0].keys()))
        _validate_columns(self._key_cols)
        non_key = [c for c in all_cols if c not in self._key_cols]
        key_conflict = ", ".join(self._key_cols)
        conn = self._table._conn

        _db = self._table._db
        _conn_lock = (
            _db._conn_lock if _db is not None else contextlib.nullcontext()
        )

        with _conn_lock:
            try:
                actual_cols_count = len(
                    conn.execute(  # nosemgrep
                        f"SELECT * FROM {self._table._name} LIMIT 0"
                    ).description or []
                )
            except Exception:  # noqa: BLE001
                actual_cols_count = len(all_cols)

            is_partial = len(all_cols) < actual_cols_count

            if is_partial and non_key and self._update_all:
                update_clause = ", ".join(f"{c}=?" for c in non_key)
                where_clause = " AND ".join(f"{k}=?" for k in self._key_cols)
                sql = (  # nosemgrep: sql-injection
                    f"UPDATE {self._table._name} SET {update_clause} WHERE {where_clause}"
                )
                params = [
                    tuple(r.get(c) for c in non_key) + tuple(r.get(k) for k in self._key_cols)
                    for r in encoded
                ]
                with _txn(conn):
                    conn.executemany(sql, params)
                return

            placeholders = ", ".join("?" for _ in all_cols)
            col_names = ", ".join(all_cols)

            if non_key:
                update_clause = ", ".join(f"{c}=excluded.{c}" for c in non_key)
                sql = (  # nosemgrep: sql-injection
                    f"INSERT INTO {self._table._name} ({col_names}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT ({key_conflict}) DO UPDATE SET {update_clause}"
                )
            else:
                sql = (  # nosemgrep: sql-injection
                    f"INSERT INTO {self._table._name} ({col_names}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT ({key_conflict}) DO NOTHING"
                )

            with _txn(conn):
                conn.executemany(sql, [tuple(r.get(c) for c in all_cols) for r in encoded])
