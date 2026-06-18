"""Table-level CRUD, query, and merge-insert classes for the Hippo storage backend."""

from __future__ import annotations

import contextlib
import logging
import re
import sqlite3
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

_log = logging.getLogger(__name__)


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


def _pa_type_to_sqlite(t: pa.DataType) -> str:
    type_str = str(t)
    if type_str in _PA_TO_SQLITE:
        return _PA_TO_SQLITE[type_str]
    if pa.types.is_integer(t):
        return "INTEGER"
    if pa.types.is_floating(t):
        return "REAL"
    if pa.types.is_boolean(t):
        return "INTEGER"
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return "BLOB"
    if pa.types.is_timestamp(t):
        return "TEXT"
    return "TEXT"


_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
    "detail_level",
})

_STRICT_BOOL_COLUMNS: frozenset[str] = frozenset({
    "pinned", "never_decay", "never_merge", "schema_bypass",
})


def _sqlite_type_to_pa(col_name: str, type_str: str, embed_dim: int) -> pa.DataType:
    t_upper = type_str.upper()
    if col_name == "embedding":
        return pa.list_(pa.float32(), embed_dim)
    if col_name in _STRICT_BOOL_COLUMNS:
        return pa.bool_()
    if t_upper in ("TEXT",):
        return pa.string()
    if t_upper in ("REAL",):
        return pa.float32()
    if t_upper in ("INTEGER",):
        return pa.int64()
    if t_upper in ("BLOB",):
        return pa.binary()
    return pa.string()


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
    embedding_pending    INTEGER NOT NULL DEFAULT 0
)"""

_DDL_RECORDS_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_records_id        ON records(id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tier      ON records(tier)",
    "CREATE INDEX IF NOT EXISTS idx_records_community ON records(community_id)",
    "CREATE INDEX IF NOT EXISTS idx_records_tomb      ON records(tombstoned_at) WHERE tombstoned_at IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS idx_records_pending   ON records(embedding_pending) WHERE embedding_pending=1",
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
}


def _encode_embedding(vec: list[float] | np.ndarray | None) -> bytes | None:
    if vec is None:
        return None
    return np.array(vec, dtype=np.float32).tobytes()


def _decode_embedding(blob: bytes | None) -> list[float] | None:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _encode_row_for_insert(row: dict) -> dict:
    out = dict(row)
    if "embedding" in out and out["embedding"] is not None:
        out["embedding"] = _encode_embedding(out["embedding"])
    return out


def _decode_df_embedding(df: pd.DataFrame) -> pd.DataFrame:
    if "embedding" in df.columns:
        df = df.copy()
        df["embedding"] = df["embedding"].apply(
            lambda b: _decode_embedding(b) if isinstance(b, (bytes, bytearray)) else b
        )
    return df


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
    if isinstance(data, pa.Table):
        return data.to_pylist()
    return list(data)


class HippoTable:

    def __init__(
        self,
        conn: sqlite3.Connection,
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
        lock = self._db._conn_lock if self._db is not None else None
        if lock is not None:
            with lock:
                row = self._conn.execute(stmt).fetchone()
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
            with self._db._conn_lock:
                df = pd.read_sql_query(stmt, self._conn)
        else:
            df = pd.read_sql_query(stmt, self._conn)
        df = _decode_df_embedding(df)
        return df

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
        cols = list(encoded[0].keys())
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
                                db._label_map[str(r["id"])] = vec_label
                                db._write_counter += 1
                            db._maybe_resize()
                if db._write_counter > 0 and db._write_counter % HNSW_SAVE_INTERVAL == 0:
                    db._save_index_atomic()
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

        set_clause = ", ".join(col + "=?" for col in encoded_values)
        if self._sql is not None:
            stmt = self._sql["update_prefix"] + set_clause + " WHERE " + where
        else:
            stmt = "UPDATE " + self._name + " SET " + set_clause + " WHERE " + where
        _lock_m1 = self._db._conn_lock if self._db is not None else contextlib.nullcontext()
        with _lock_m1:
            with _txn(self._conn):
                self._conn.execute(stmt, list(encoded_values.values()))

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
                    label = int(row["vec_label"])
                    try:
                        db._hnsw.mark_deleted(label)
                    except RuntimeError:
                        pass
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
    def schema(self) -> pa.Schema:
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
        fields: list[pa.Field] = []
        for row in pragma_rows:
            col_name = row["name"]
            type_str = row["type"] if row["type"] else "TEXT"
            pa_type = _sqlite_type_to_pa(col_name, type_str, self._embed_dim)
            nullable = not bool(row["notnull"])
            fields.append(pa.field(col_name, pa_type, nullable=nullable))
        return pa.schema(fields)

    def add_columns(self, fields: list[pa.Field]) -> None:
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
        import sqlite3 as _sqlite3
        major, minor, _ = (int(x) for x in _sqlite3.sqlite_version.split("."))
        if (major, minor) < (3, 35):
            raise RuntimeError(
                f"ALTER TABLE DROP COLUMN requires SQLite >= 3.35; "
                f"installed: {_sqlite3.sqlite_version}"
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
        conn: sqlite3.Connection,
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
        _lock = self._db._conn_lock if self._db is not None else None
        if _lock is not None:
            with _lock:
                df = pd.read_sql_query(sql, self._conn)
        else:
            df = pd.read_sql_query(sql, self._conn)
        df = _decode_df_embedding(df)
        return df

    def _ann_to_pandas(self) -> pd.DataFrame:
        db = self._ann_db
        k = self._limit_val if self._limit_val is not None else 10

        with db._hnsw_lock:
            active_count = len(db._label_map)
            if active_count == 0:
                return pd.DataFrame()
            # Clamp k to the live hnswlib index count, not just the label_map
            # count. The two can diverge briefly during boot/rebuild, and
            # hnswlib raises "Probably ef or M is too small" when asked for
            # more neighbors than the index actually contains.
            hnsw_count = db._hnsw.get_current_count()
            k_clamped = min(k, active_count, hnsw_count)
            if k_clamped == 0:
                return pd.DataFrame()
            try:
                labels, distances = db._hnsw.knn_query(
                    self._ann_vector, k=k_clamped
                )
            except RuntimeError as exc:
                # Defensive: hnswlib still raises this on rare label-map/index
                # skew. Treat as "no neighbors found" so the insert path's
                # pattern-separation gate can proceed instead of failing the
                # whole capture.
                _log.warning(
                    "hnswlib knn_query failed (k=%d, hnsw_count=%d, "
                    "active_count=%d); returning empty result: %s",
                    k_clamped, hnsw_count, active_count, exc,
                )
                return pd.DataFrame()

        flat_labels: list[int] = labels[0].tolist()
        # Clamp cosine distance to its mathematical range — the BLAS backend
        # can produce sub-epsilon negatives on Linux.
        flat_distances: list[float] = [
            max(0.0, min(2.0, float(d))) for d in distances[0]
        ]

        if not flat_labels:
            return pd.DataFrame()

        placeholders = ", ".join("?" for _ in flat_labels)
        sql = (  # nosemgrep: sql-injection
            f"SELECT * FROM {self._table_name} WHERE vec_label IN ({placeholders})"
        )
        if self._where_clauses:
            sql += " AND " + " AND ".join(f"({c})" for c in self._where_clauses)

        _lock = db._conn_lock if db is not None else None
        if _lock is not None:
            with _lock:
                df = pd.read_sql_query(sql, self._conn, params=flat_labels)
        else:
            df = pd.read_sql_query(sql, self._conn, params=flat_labels)
        df = _decode_df_embedding(df)

        if df.empty:
            return df

        dist_map: dict[int, float] = {
            int(lbl): float(d) for lbl, d in zip(flat_labels, flat_distances)
        }
        df["_distance"] = df["vec_label"].apply(lambda lbl: dist_map.get(int(lbl), float("nan")))

        df = df.sort_values("_distance").reset_index(drop=True)
        return df

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

    def to_batches(self, batch_size: int = 1000) -> Iterator[pa.RecordBatch]:
        sql = self._build_sql()
        _lock = self._db._conn_lock if self._db is not None else None
        if _lock is not None:
            with _lock:
                batches = self._drain_to_batches(sql, batch_size)
        else:
            batches = self._drain_to_batches(sql, batch_size)
        yield from batches

    def _drain_to_batches(
        self, sql: str, batch_size: int
    ) -> list[pa.RecordBatch]:
        batches: list[pa.RecordBatch] = []
        cursor = self._conn.execute(sql)
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
                        if isinstance(b, (bytes, bytearray))
                        else b
                        for b in data["embedding"]
                    ]
                batches.append(pa.record_batch(data))
        finally:
            cursor.close()
        return batches


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
        all_cols = list(encoded[0].keys())
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
