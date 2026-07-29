"""Parent-side read-only export helpers used by the periodic graph refresh.

Opens a read-only connection to the store (driver-aware: engine raw connection
on the lilli driver, `file:...?mode=ro` URI on stdlib), wraps the export in a
read-only transaction context for stdlib snapshot consistency, and walks
records and edges via narrow-projection keyset pagination.

The edges SELECT JOINs against the records table on both src and dst with
the active-records predicate `tombstoned_at IS NULL AND
COALESCE(embedding_pending, 0) = 0` (identical to the active_records_count
predicate used elsewhere in the store), so any edge with a filtered
endpoint is dropped at the streaming SQL layer and no dangling endpoint
reaches the worker. The export issues only read-only SELECTs and takes
no write transaction.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator

from iai_mcp import _sqlite_stdlib
from iai_mcp import errors

from iai_mcp.hippo import open_store_conn


DEFAULT_RECORDS_CHUNK_SIZE: int = 2000
DEFAULT_EDGES_CHUNK_SIZE: int = 5000


def open_ro_connection(db_path: Path) -> Any:
    """Open a read-only connection to the store, driver-aware.

    On the lilli driver: returns the engine raw connection for the file
    (live registered connection when HippoDB is open, else a temporary
    reopen). On stdlib: opens a `file:...?mode=ro` URI connection so
    SQLite refuses DML at the file layer. Both paths apply `PRAGMA
    query_only=ON` (a no-op on the engine) and `PRAGMA busy_timeout=2000`
    for defense-in-depth. The caller is responsible for closing the
    connection in a `finally`.
    """
    _eng = open_store_conn(db_path, read_only=True)
    if _eng is not None:
        conn = _eng
    else:
        conn = _sqlite_stdlib.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = _sqlite_stdlib.Row
    conn.execute("PRAGMA query_only=ON")
    conn.execute("PRAGMA busy_timeout=2000")
    return conn


@contextlib.contextmanager
def read_transaction(conn: Any):
    """Wrap the export in a read-only transaction context.

    On stdlib: issues BEGIN DEFERRED / COMMIT so every paginated SELECT
    sees a single consistent WAL snapshot. On the lilli driver: the engine
    has no read-only snapshot transaction — every BEGIN* is a write
    transaction that would conflict with concurrent daemon writes. The
    export issues only SELECTs, so an implicit autocommit read is correct
    and the BEGIN/COMMIT are skipped on that path.

    Branches on the concrete connection object returned by
    `open_ro_connection` (already driver-resolved from the on-disk file
    format), not on the ambient env var — a lilli-format store opened with
    `LILLI_STORAGE_DRIVER` unset must still take the no-transaction branch,
    and vice versa for a genuine stdlib connection opened with the env set
    to "lilli".
    """
    if not _sqlite_stdlib.is_stdlib_connection(conn):
        yield
        return
    conn.execute("BEGIN DEFERRED")
    try:
        yield
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except errors.Error:
            pass
        raise
    else:
        try:
            conn.execute("COMMIT")
        except errors.Error:
            pass


def iter_records_chunks(
    conn: Any,
    chunk_size: int = DEFAULT_RECORDS_CHUNK_SIZE,
) -> Iterator[list[tuple[str, bytes]]]:
    """Yield lists of `(id_str, embedding_blob_bytes)` of length <= chunk_size.

    Keyset cursor on the integer `vec_label` primary key. The predicate
    `tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0` matches
    the active-records semantics applied elsewhere in the store, so the
    streamed records are exactly the records the system considers
    recall-visible.
    """
    cursor = -1
    sql = (
        "SELECT vec_label, id, embedding FROM records"
        " WHERE tombstoned_at IS NULL"
        " AND COALESCE(embedding_pending, 0) = 0"
        " AND vec_label > ?"
        " ORDER BY vec_label"
        " LIMIT ?"
    )
    while True:
        rows = conn.execute(sql, (cursor, chunk_size)).fetchall()
        if not rows:
            return
        chunk: list[tuple[str, bytes]] = []
        for row in rows:
            chunk.append((str(row["id"]), bytes(row["embedding"])))
            cursor = int(row["vec_label"])
        yield chunk
        if len(rows) < chunk_size:
            return


def _fetch_active_ids(conn: Any) -> set[str]:
    """Return the set of record ids that are active (not tombstoned, not pending).

    Fetches once using a narrow projection — no JOIN, no dotted columns —
    so the result is compatible with both the stdlib driver and the lilli
    engine (which supports no JOIN or table-qualified column references).
    The result set is bounded by the live record count (~30 k ids, a few MB).
    """
    sql = (
        "SELECT id FROM records"
        " WHERE tombstoned_at IS NULL"
        " AND COALESCE(embedding_pending, 0) = 0"
    )
    rows = conn.execute(sql).fetchall()
    return {str(row["id"]) for row in rows}


def iter_edges_chunks(
    conn: Any,
    chunk_size: int = DEFAULT_EDGES_CHUNK_SIZE,
) -> Iterator[list[tuple[str, str, float, str]]]:
    """Yield `(src_str, dst_str, weight_float, edge_type)` lists <= chunk_size.

    Edges whose src or dst is not in the active-records set (tombstoned or
    embedding-pending) are dropped at the Python layer.  Active ids are
    fetched once in a single narrow SELECT (no JOIN, no dotted-column
    references), making the query compatible with both the stdlib driver and
    the lilli engine.

    Pagination strategy is driver-aware, branching on the concrete
    connection object (already driver-resolved from the on-disk file format
    by `open_ro_connection`), not on the ambient env var:
    - stdlib connection: keyset pagination
      using compound tuple comparison `(src, dst, edge_type) > (?, ?, ?)`
      for snapshot-consistent streaming.
    - lilli engine (anything else): single full scan followed by
      Python-side chunking. The lilli engine is an in-process store with a
      bounded edge count comparable to the record count; a full scan avoids
      the tuple-comparison syntax that the engine's SQL parser does not
      support.

    The `edge_type` travels with each edge so the worker's ranking-degree
    map can count earned edges only, matching the cold recall path.
    """
    active_ids = _fetch_active_ids(conn)

    if not _sqlite_stdlib.is_stdlib_connection(conn):
        # Lilli path: single full scan, chunk in Python.
        # The engine does not support multi-column ORDER BY or compound-tuple
        # WHERE comparisons, so no sort is applied — order is unspecified but
        # consistent for a single full B+tree scan, which is sufficient for
        # the graph-rebuild consumer.
        sql = "SELECT src, dst, edge_type, weight FROM edges"
        all_rows = conn.execute(sql).fetchall()
        chunk: list[tuple[str, str, float, str]] = []
        for row in all_rows:
            row_src = str(row["src"])
            row_dst = str(row["dst"])
            if row_src in active_ids and row_dst in active_ids:
                chunk.append((
                    row_src, row_dst, float(row["weight"]),
                    str(row["edge_type"]),
                ))
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk
        return

    # Stdlib path: keyset pagination with compound tuple comparison.
    src = ""
    dst = ""
    edge_type = ""
    sql = (
        "SELECT src, dst, edge_type, weight"
        " FROM edges"
        " WHERE (src, dst, edge_type) > (?, ?, ?)"
        " ORDER BY src, dst, edge_type"
        " LIMIT ?"
    )
    while True:
        rows = conn.execute(sql, (src, dst, edge_type, chunk_size)).fetchall()
        if not rows:
            return
        chunk = []
        for row in rows:
            row_src = str(row["src"])
            row_dst = str(row["dst"])
            row_et = str(row["edge_type"])
            src = row_src
            dst = row_dst
            edge_type = row_et
            if row_src in active_ids and row_dst in active_ids:
                chunk.append((row_src, row_dst, float(row["weight"]), row_et))
        if chunk:
            yield chunk
        if len(rows) < chunk_size:
            return
