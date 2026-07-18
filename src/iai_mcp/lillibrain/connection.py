"""Multi-table bootstrap and sqlite3-shaped DB-API wrapper for the LilliBrain engine.

Contains two independent responsibilities:

1. Meta table (page-3 pinned B+tree root): stores DDL and table→page mappings so the
   catalog and root_map can be reconstructed on every re-open without a catalog bootstrap.

2. LilliBrainRow / LilliBrainCursor / LilliBrainConnection: a sqlite3-shaped DB-API
   wrapper that exposes execute/executemany/fetch*/description/commit/close/in_transaction
   so that HippoDB can use the LilliBrain engine as a drop-in storage driver without
   changing any code above the driver seam.

Meta table row kinds (identified by first column):

  kind = "ddl"   → (kind, seq_no, ddl_text)
      Records CREATE TABLE / ALTER TABLE / CREATE INDEX DDL in insertion order.
      Replayed through Catalog.apply_ddl() on re-open.

  kind = "root"  → (kind, table_name, root_page_no)
      Maps a table name to its extend_file-allocated B+tree root page number.
      Reconstructs root_map on re-open.

Rows are keyed by a monotonically increasing integer written via BtreeCursor
directly (the meta table is not exposed to the SQL executor or catalog).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from iai_mcp.lillibrain.btree.cursor import BtreeCursor
from iai_mcp.lillibrain.btree.page import write_leaf_header
from iai_mcp.lillibrain.record import decode_record, encode_record

if TYPE_CHECKING:
    from iai_mcp.lillibrain.pager import Pager
    from iai_mcp.lillibrain.sql.catalog import Catalog

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constant: the pinned meta table root page
# ---------------------------------------------------------------------------

#: B+tree root page for the bootstrap meta table.
#: Fixed at page 3 — one past Pager.root_page_no (2) — so it is
#: findable with no catalog and no prior allocation.
META_TABLE_PAGE: int = 3

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_PAGE_TYPE_LEAF = 0x0D      # PAGE_LEAF_TABLE
_PAGE_TYPE_INTERIOR = 0x05  # PAGE_INTERIOR_TABLE


def _page_is_initialised_btree(page: bytearray) -> bool:
    """Return True if the page is an initialised B+tree page (leaf or interior).

    After sufficient inserts the meta table root at page 3 is promoted from a
    leaf to an interior node during a root split. Checking only for the leaf
    type byte causes a spurious re-initialisation attempt on every re-open of
    a database that has grown past the single-leaf threshold, which then fails
    because extend_file() returns a page number > 3 on a non-fresh file.
    """
    return len(page) > 0 and page[0] in (_PAGE_TYPE_LEAF, _PAGE_TYPE_INTERIOR)


def _meta_cursor(pager: Pager) -> BtreeCursor:
    """Return a BtreeCursor rooted at META_TABLE_PAGE."""
    return BtreeCursor(pager, META_TABLE_PAGE)


def _meta_next_key(pager: Pager) -> int:
    """Return the next available integer key for a meta table row."""
    cursor = _meta_cursor(pager)
    rows = cursor.full_scan()
    if not rows:
        return 1
    return rows[-1][0] + 1


def _init_meta_page(pager: Pager) -> None:
    """Allocate and initialise page 3 as an empty leaf B+tree root.

    Called only on a fresh file (page 3 has not yet been allocated).
    Extends the file to include page 3 and writes the empty leaf header.
    """
    from iai_mcp.lillibrain.constants import PAGE_SIZE  # noqa: PLC0415

    # extend_file returns the new page number; we need to ensure it is 3.
    # On a fresh file db_size=2 (pages 1 and 2), so extend_file returns 3.
    pager.begin_write()
    new_page_no = pager.extend_file()
    if new_page_no != META_TABLE_PAGE:
        # This should never happen on a fresh file, but be defensive.
        pager.rollback()
        raise RuntimeError(
            f"Expected meta table at page {META_TABLE_PAGE}, "
            f"got page {new_page_no}. File may be corrupt."
        )
    page = bytearray(PAGE_SIZE)
    write_leaf_header(page, num_cells=0, cell_content_start=PAGE_SIZE)
    pager.write_page(META_TABLE_PAGE, page)
    pager.commit()
    logger.debug("Initialised meta table at page %d", META_TABLE_PAGE)


def _write_meta_ddl_row(pager: Pager, seq: int, ddl_text: str) -> None:
    """Append a DDL row to the meta table inside a write transaction."""
    key = _meta_next_key(pager)
    # kind="ddl", seq=seq (int), ddl_text (str)
    payload = encode_record(["ddl", seq, ddl_text])
    cursor = _meta_cursor(pager)
    pager.begin_write()
    cursor.insert(key, payload)
    pager.commit()


def _write_meta_root_row(pager: Pager, table_name: str, root_page_no: int) -> None:
    """Append a table→root_page mapping row to the meta table."""
    key = _meta_next_key(pager)
    # kind="root", table_name (str), root_page_no (int)
    payload = encode_record(["root", table_name, root_page_no])
    cursor = _meta_cursor(pager)
    pager.begin_write()
    cursor.insert(key, payload)
    pager.commit()


def _read_meta_rows(pager: Pager) -> list[list]:
    """Return all decoded rows from the meta table in key-ascending order."""
    cursor = _meta_cursor(pager)
    result: list[list] = []
    for _key, payload in cursor.full_scan():
        values, _ = decode_record(payload, 0)
        result.append(values)
    return result


def _read_hwm_from_meta(pager: Pager, table_name: str) -> int:
    """Return the persisted high-water mark for table_name, or 0 if absent.

    The high-water mark is the highest vec_label ever assigned to the table.
    A stored HWM of 0 means no row has been inserted yet; the first row gets 1.
    """
    rows = _read_meta_rows(pager)
    hwm = 0
    for row in rows:
        if row and len(row) >= 3 and row[0] == "hwm" and row[1] == table_name:
            hwm = int(row[2])
    return hwm


def _update_hwm_in_meta(
    pager: Pager,
    table_name: str,
    old_meta_key: int | None,
    new_hwm: int,
) -> int:
    """Write or overwrite the HWM row for table_name in the meta table.

    If old_meta_key is None, appends a new hwm row.
    If old_meta_key is provided, overwrites the existing row at that B+tree key.
    Returns the meta-table B+tree key that holds the hwm row.
    """
    cursor = _meta_cursor(pager)
    payload = encode_record(["hwm", table_name, new_hwm])
    pager.begin_write()
    if old_meta_key is None:
        key = _meta_next_key(pager)
        cursor.insert(key, payload)
    else:
        cursor.insert(old_meta_key, payload)
    pager.commit()
    return old_meta_key if old_meta_key is not None else _meta_next_key_after(pager)


def _meta_next_key_after(pager: Pager) -> int:
    """Return the last inserted key in the meta table (used only for return value)."""
    cursor = _meta_cursor(pager)
    rows = cursor.full_scan()
    return rows[-1][0] if rows else 1


def _find_hwm_meta_key(pager: Pager, table_name: str) -> int | None:
    """Return the B+tree key of the hwm row for table_name, or None if absent."""
    cursor = _meta_cursor(pager)
    for key, payload in cursor.full_scan():
        values, _ = decode_record(payload, 0)
        if values and len(values) >= 3 and values[0] == "hwm" and values[1] == table_name:
            return key
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def register_table(
    pager: Pager,
    catalog: Catalog,
    root_map: dict[str, int],
    ddl: str,
) -> int:
    """Register a new table: allocate its B+tree root and persist DDL + root mapping.

    Parses the table name from the CREATE TABLE DDL, allocates a fresh B+tree
    root page via pager.extend_file(), writes an empty leaf header into that
    page, records the DDL text and the table→root mapping in the meta table,
    and registers the table in the live catalog.

    Returns the allocated root page number.

    If the table is already present in root_map (idempotent re-registration),
    this is a no-op and returns the existing root page number.
    """
    table_name = _extract_table_name(ddl)
    if table_name in root_map:
        # If the alias is present but the catalog is missing (e.g. a stale root
        # row left by a RENAME that was not tombstoned), re-register the catalog
        # so the table becomes query-able rather than bricking with CatalogError.
        if table_name not in catalog._tables:
            from iai_mcp.lillibrain.sql.executor import run  # noqa: PLC0415
            run(ddl, [], pager, catalog, root_map)
        return root_map[table_name]

    from iai_mcp.lillibrain.constants import PAGE_SIZE  # noqa: PLC0415

    # Determine the next DDL sequence number from existing meta rows.
    existing = _read_meta_rows(pager)
    ddl_rows = [r for r in existing if r and r[0] == "ddl"]
    seq = len(ddl_rows)  # 0-based insertion order

    # Allocate a new B+tree root via extend_file.
    pager.begin_write()
    root_page_no = pager.extend_file()
    root_page = bytearray(PAGE_SIZE)
    write_leaf_header(root_page, num_cells=0, cell_content_start=PAGE_SIZE)
    pager.write_page(root_page_no, root_page)
    pager.commit()

    # Persist DDL text and root mapping to the meta table.
    _write_meta_ddl_row(pager, seq, ddl)
    _write_meta_root_row(pager, table_name, root_page_no)

    # Apply DDL to the live catalog.
    from iai_mcp.lillibrain.sql.executor import run  # noqa: PLC0415
    run(ddl, [], pager, catalog, root_map)

    # Register the root page in root_map.
    root_map[table_name] = root_page_no

    logger.debug(
        "Registered table %r at root page %d (seq=%d)", table_name, root_page_no, seq
    )
    return root_page_no


def _open_lilli_db(
    db_path: str | Path,
    *,
    embed_dim: int = 0,
) -> tuple[Pager, Catalog, dict[str, int]]:
    """Open (or create) a LilliBrain database file.

    On a fresh file, initialises the meta table at page 3 with an empty leaf
    header and returns an empty catalog + root_map.

    On an existing file, reads the meta table and replays all stored DDL rows
    through Catalog.apply_ddl() in insertion order, reconstructing root_map
    from the persisted table→root_page entries. This makes open idempotent
    across both first-create and re-open.

    Returns (pager, catalog, root_map). The caller is responsible for calling
    pager.close() when done.
    """
    from iai_mcp.lillibrain.pager import Pager  # noqa: PLC0415
    from iai_mcp.lillibrain.sql.catalog import Catalog  # noqa: PLC0415

    db_path = Path(db_path)
    pager = Pager(db_path)
    catalog = Catalog()
    root_map: dict[str, int] = {}

    # Check whether page 3 (meta table root) has been initialised.
    db_size = pager.read_db_header_field("db_size")

    if db_size < META_TABLE_PAGE:
        # Fresh file — page 3 does not exist yet. Allocate and initialise it.
        _init_meta_page(pager)
        logger.debug("_open_lilli_db: fresh file, initialised meta page at %d", META_TABLE_PAGE)
    else:
        # Existing file — check whether page 3 is a valid leaf.
        page3 = pager.read_page(META_TABLE_PAGE)
        if not _page_is_initialised_btree(page3):
            # Page 3 exists but is not an initialised leaf — treat as fresh.
            _init_meta_page(pager)
            logger.debug(
                "_open_lilli_db: page 3 existed but was uninitialised; re-initialised"
            )
        else:
            # Replay meta table: reconstruct catalog and root_map.
            _replay_meta(pager, catalog, root_map)
            logger.debug(
                "_open_lilli_db: replayed meta table, %d tables restored",
                len(root_map),
            )

    return pager, catalog, root_map


def _replay_meta(
    pager: Pager,
    catalog: Catalog,
    root_map: dict[str, int],
) -> None:
    """Read the meta table and replay DDL + root_map entries into live state.

    DDL rows are replayed through the catalog's apply_ddl() in seq-ascending
    order. Root-mapping rows populate root_map. Called only when the meta
    page is already present and initialised.
    """
    from iai_mcp.lillibrain.sql.ast_nodes import DDLStmt  # noqa: PLC0415
    from iai_mcp.lillibrain.sql.parser import parse_sql  # noqa: PLC0415

    rows = _read_meta_rows(pager)

    # Separate by kind
    ddl_rows: list[tuple[int, str]] = []  # (seq, ddl_text)
    root_rows: list[tuple[str, int]] = []  # (table_name, root_page_no)

    for row in rows:
        if not row:
            continue
        kind = row[0]
        if kind == "ddl" and len(row) >= 3:
            seq = int(row[1])
            ddl_text = str(row[2])
            ddl_rows.append((seq, ddl_text))
        elif kind == "root" and len(row) >= 3:
            table_name = str(row[1])
            root_page_no = int(row[2])
            root_rows.append((table_name, root_page_no))

    # Replay DDL in insertion order (ascending seq).
    ddl_rows.sort(key=lambda t: t[0])
    for _seq, ddl_text in ddl_rows:
        stmt = parse_sql(ddl_text)
        if isinstance(stmt, DDLStmt):
            catalog.apply_ddl(stmt)

    # Populate root_map, then drop any entry whose name is absent from the
    # catalog (stale root rows left by an untombstoned RENAME).  This closes
    # the re-open re-alias class for any future un-tombstoned root row.
    for table_name, root_page_no in root_rows:
        root_map[table_name] = root_page_no

    stale = [name for name in list(root_map) if name not in catalog._tables]
    for name in stale:
        del root_map[name]


def _extract_table_name(ddl: str) -> str:
    """Extract the table name from a CREATE TABLE DDL string.

    Handles: CREATE TABLE name (...) and CREATE TABLE IF NOT EXISTS name (...)
    """
    import re  # noqa: PLC0415

    match = re.search(
        r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?\[?(\w+)\]?",
        ddl,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot extract table name from DDL: {ddl!r}")
    return match.group(1)


def _extract_rename_names(ddl: str) -> tuple[str, str]:
    """Extract (old_name, new_name) from an ALTER TABLE ... RENAME TO ... string.

    Bracket-quoted names ([records]) are accepted; the brackets are stripped.
    """
    import re  # noqa: PLC0415

    match = re.search(
        r"ALTER\s+TABLE\s+\[?(\w+)\]?\s+RENAME\s+TO\s+\[?(\w+)\]?",
        ddl,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot extract rename names from DDL: {ddl!r}")
    return match.group(1), match.group(2)


def _extract_drop_name(ddl: str) -> str:
    """Extract the table name from a DROP TABLE [IF EXISTS] <name> string.

    Bracket-quoted names ([records]) are accepted; the brackets are stripped.
    """
    import re  # noqa: PLC0415

    match = re.search(
        r"DROP\s+TABLE\s+(?:IF\s+EXISTS\s+)?\[?(\w+)\]?",
        ddl,
        re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Cannot extract table name from DDL: {ddl!r}")
    return match.group(1)


def _find_root_meta_keys(pager: Pager, table_name: str) -> list[int]:
    """Return the B+tree key(s) of every ``root`` meta row for table_name.

    A table may have a single root row, but a re-created name can accumulate
    more than one; return all so the caller can tombstone every stale mapping.
    """
    cursor = _meta_cursor(pager)
    keys: list[int] = []
    for key, payload in cursor.full_scan():
        values, _ = decode_record(payload, 0)
        if values and len(values) >= 3 and values[0] == "root" and values[1] == table_name:
            keys.append(key)
    return keys


# ---------------------------------------------------------------------------
# DB-API wrapper — LilliBrainRow, LilliBrainCursor, LilliBrainConnection
# ---------------------------------------------------------------------------

import re as _re  # noqa: E402  (module-level; placed after top-of-file imports above)

from iai_mcp.lillibrain.sql.types import SqlNull  # noqa: E402


class LilliBrainRow(dict):
    """sqlite3.Row analog: a dict subclass supporting both name and positional access.

    Integer __getitem__ looks up the value by positional column index; string
    __getitem__ delegates to the dict superclass. SqlNull and Python None both
    normalise to None at construction time so hippo's None-guards behave
    correctly downstream.

    Iteration yields VALUES in column declaration order — matching sqlite3.Row
    behaviour so that callers using ``dict(zip(col_names, row))`` or
    ``tuple(row)`` receive values, not keys.
    """

    def __init__(self, data: dict, col_names: list[str]) -> None:
        # Normalise SqlNull → None before storing; keep Python None as-is.
        normalised = {
            k: (None if (v is SqlNull or v is None) else v)
            for k, v in data.items()
        }
        super().__init__(normalised)
        self._col_names: list[str] = col_names

    def __getitem__(self, key: str | int):  # type: ignore[override]
        if isinstance(key, int):
            col = self._col_names[key]
            return super().__getitem__(col)
        return super().__getitem__(key)

    def __iter__(self):  # type: ignore[override]
        """Yield values in column declaration order (sqlite3.Row compatible)."""
        for col in self._col_names:
            yield super().__getitem__(col)


class LilliBrainCursor:
    """sqlite3.Cursor analog: materialised result set with a position-offset fetch API.

    The rows list is captured at construction. Fetch methods advance a shared
    integer offset (_pos) rather than re-running any query — this is the contract
    the brute-force ANN streaming scan depends on.
    """

    def __init__(
        self,
        rows: list[dict],
        col_names: list[str],
        connection: "LilliBrainConnection | None" = None,
    ) -> None:
        self._col_names: list[str] = col_names
        self._rows: list[LilliBrainRow] = [
            LilliBrainRow(r, col_names) for r in rows
        ]
        self._pos: int = 0
        # Set on the cursor an INSERT returns: the AUTOINCREMENT value the
        # engine injected, mirroring sqlite3.Cursor.lastrowid. None for any
        # statement that injects no autoincrement key.
        self.lastrowid: int | None = None
        # Optional back-reference so a cursor obtained from conn.cursor() can
        # run statements via execute(). None for the result-set cursors that
        # execute()/_run() construct directly.
        self._connection: "LilliBrainConnection | None" = connection

    @property
    def description(self) -> tuple:
        """DB-API description: a tuple of 7-element tuples, name in slot 0."""
        return tuple(
            (name, None, None, None, None, None, None)
            for name in self._col_names
        )

    def execute(self, sql: str, params=()) -> "LilliBrainCursor":
        """Run a statement on the bound connection, then mirror its result here.

        Matches the DB-API pattern pandas drives: ``cur = conn.cursor();
        cur.execute(sql); cur.fetchall()`` reads against the SAME object. So
        this copies the result cursor's rows/columns/position/lastrowid onto
        self and returns self. Requires a bound connection (set by
        conn.cursor()); an unbound cursor raises rather than silently no-op.
        """
        if self._connection is None:
            raise RuntimeError(
                "LilliBrainCursor.execute requires a connection-bound cursor; "
                "obtain one via connection.cursor()"
            )
        result = self._connection.execute(sql, params)
        self._col_names = result._col_names
        self._rows = result._rows
        self._pos = result._pos
        self.lastrowid = result.lastrowid
        return self

    def executemany(self, sql: str, seq_of_params) -> "LilliBrainCursor":
        """Run sql once per parameter tuple on the bound connection, then mirror
        its result here. Mirrors execute()'s delegation pattern: requires a bound
        connection (set by conn.cursor()); an unbound cursor raises rather than
        silently no-op'ing. Returns self, matching the DB-API executemany contract.
        """
        if self._connection is None:
            raise RuntimeError(
                "LilliBrainCursor.executemany requires a connection-bound cursor; "
                "obtain one via connection.cursor()"
            )
        result = self._connection.executemany(sql, seq_of_params)
        self._col_names = result._col_names
        self._rows = result._rows
        self._pos = result._pos
        self.lastrowid = result.lastrowid
        return self

    def close(self) -> None:
        """Release the cursor. No engine resource is held, so this is a no-op."""

    def fetchone(self) -> LilliBrainRow | None:
        """Return the next row or None when exhausted."""
        if self._pos >= len(self._rows):
            return None
        row = self._rows[self._pos]
        self._pos += 1
        return row

    def fetchall(self) -> list[LilliBrainRow]:
        """Return all remaining rows and advance pos to the end."""
        remaining = self._rows[self._pos:]
        self._pos = len(self._rows)
        return remaining

    def fetchmany(self, size: int) -> list[LilliBrainRow]:
        """Return up to size rows and advance pos by that count."""
        batch = self._rows[self._pos: self._pos + size]
        self._pos += len(batch)
        return batch

    def __iter__(self):
        """Yield each remaining row, advancing the shared fetch offset.

        sqlite3.Cursor-compatible: ``for row in cursor`` consumes the same
        position offset the fetch* methods use, so iteration after a partial
        fetch resumes from where fetch left off and a subsequent fetchone()
        returns None once iteration is exhausted. No query is re-run — this
        walks the already-materialised row list.
        """
        while self._pos < len(self._rows):
            row = self._rows[self._pos]
            self._pos += 1
            yield row


# No-op pager transaction stubs — returned by _suppress_pager_txn when an
# outer transaction is already open, so the executor's internal begin/commit
# calls become harmless while DML accumulates in the outer transaction.

def _noop(*_args, **_kwargs):  # type: ignore[return]
    """Accept any arguments and do nothing."""


class LilliBrainConnection:
    """sqlite3.Connection analog wrapping Pager + Catalog + root_map.

    Reconciles two transaction models:
    - Outer model (hippo): explicit BEGIN / COMMIT / ROLLBACK SQL strings
      checked via the in_transaction property.
    - Inner model (executor): DML branches call pager.begin_write() + pager.commit()
      around every INSERT/UPDATE/DELETE.

    When an outer transaction is open (in_transaction is True) the executor's
    internal begin/commit calls are suppressed via temporary method replacement
    on the pager object so all DML accumulates in the one outer transaction.
    The pager raises RuntimeError on a nested begin, so suppression is required
    for correctness.

    row_factory is a class attribute that hippo assigns sqlite3.Row to
    unconditionally — it must exist but its value is ignored.
    """

    row_factory = None  # hippo assigns sqlite3.Row here; the value is ignored

    # PRAGMA prefixes that are accepted as no-ops (or near-no-ops). These are
    # connection-tuning hints with no meaning for the pager engine
    # (foreign-key / synchronous toggles).  PRAGMA cache_size and busy_timeout
    # carry a value that has a side effect (the page-cache bound and the
    # advisory timeout respectively) and are handled by dedicated branches
    # above this short-circuit; they are not in this tuple.
    _NOOP_PRAGMA_PREFIXES = (
        "PRAGMA SYNCHRONOUS",
        "PRAGMA FOREIGN_KEYS",
    )

    def __init__(
        self,
        pager: "Pager",  # type: ignore[name-defined]
        catalog: "Catalog",  # type: ignore[name-defined]
        root_map: dict[str, int],
    ) -> None:
        self._pager = pager
        self._catalog = catalog
        self._root_map = root_map
        self._in_transaction: bool = False
        # vec_label AUTOINCREMENT high-water marks: table -> current HWM.
        # Loaded lazily from the meta table on first insert per table.
        self._vec_label_hwm: dict[str, int] = {}
        # Meta-table B+tree key holding the HWM row (None until first write).
        self._hwm_meta_key: dict[str, int | None] = {}
        # Per-table id→row_key index for O(log N) UPDATE point lookups.
        # None means the cache for that table has not been built or was
        # invalidated. Built lazily on the first UPDATE WHERE id=? miss.
        self._id_row_cache: dict[str, dict[str, int] | None] = {}
        # Connection-level serialization mutex. RLock (not Lock): three same
        # thread re-entrant paths re-enter execute() while it is held —
        # cursor.execute -> connection.execute, raw_conn.execute ->
        # connection.execute, and the pandas read_sql path. Mirrors the
        # recursive SERIALIZED mutex a real sqlite3 connection provides under
        # check_same_thread=False; the engine advertises that flag and so must
        # serialize cross-thread statement execution itself.
        self._exec_lock = threading.RLock()
        # True after PRAGMA query_only=ON — write attempts raise OperationalError.
        self._query_only: bool = False

    # ------------------------------------------------------------------
    # in_transaction property
    # ------------------------------------------------------------------

    @property
    def in_transaction(self) -> bool:
        """True when an outer BEGIN has been issued and not yet committed/rolled back."""
        return self._in_transaction

    # ------------------------------------------------------------------
    # Transaction suppression helpers
    # ------------------------------------------------------------------

    def _suppress_pager_txn(self) -> tuple:
        """Replace pager begin_write/commit/rollback with no-ops; return originals."""
        orig_begin = self._pager.begin_write
        orig_commit = self._pager.commit
        orig_rollback = self._pager.rollback
        self._pager.begin_write = _noop  # type: ignore[method-assign]
        self._pager.commit = _noop  # type: ignore[method-assign]
        self._pager.rollback = _noop  # type: ignore[method-assign]
        return orig_begin, orig_commit, orig_rollback

    def _restore_pager_txn(self, orig: tuple) -> None:
        """Restore pager transaction methods from saved originals."""
        self._pager.begin_write, self._pager.commit, self._pager.rollback = orig

    # ------------------------------------------------------------------
    # Core execute / DML dispatch
    # ------------------------------------------------------------------

    def _build_id_row_cache(self, table: str) -> dict[str, int]:
        """Scan table once and return a fresh {id_str → row_key} map.

        Called lazily before the first UPDATE WHERE id=? on a table whose
        cache is absent or was invalidated. The full scan cost is paid once;
        subsequent point-lookups skip the scan entirely.
        """
        from iai_mcp.lillibrain.btree.cursor import BtreeCursor  # noqa: PLC0415
        from iai_mcp.lillibrain.record import decode_record  # noqa: PLC0415

        root_page = self._root_map.get(table)
        if root_page is None:
            return {}
        col_names = self._catalog.column_names(table)
        try:
            id_col_idx = col_names.index("id")
        except ValueError:
            return {}  # table has no 'id' column — index not applicable
        cursor = BtreeCursor(self._pager, root_page)
        index: dict[str, int] = {}
        for key, payload in cursor.full_scan():
            try:
                vals, _ = decode_record(payload, 0)
                if id_col_idx < len(vals):
                    id_val = vals[id_col_idx]
                    if isinstance(id_val, str):
                        index[id_val] = key
            except Exception:  # noqa: BLE001 — skip undecodable rows; index stays partial
                pass
        return index

    def _get_id_index_for_update(self, table: str) -> "dict[str, int] | None":
        """Return the live id→row_key index for table, building it if needed.

        Returns None when the table has no 'id' column or the index cannot
        be built (fallback to full scan in the executor).
        """
        cached = self._id_row_cache.get(table)
        if cached is None and table in self._id_row_cache:
            # Explicit None sentinel: table was invalidated, needs rebuild.
            new_cache = self._build_id_row_cache(table)
            self._id_row_cache[table] = new_cache
            return new_cache
        if cached is not None:
            return cached
        # Table not yet in cache dict at all: build it.
        new_cache = self._build_id_row_cache(table)
        if new_cache:
            self._id_row_cache[table] = new_cache
        return new_cache if new_cache else None

    def _invalidate_id_cache(self, table: str) -> None:
        """Mark the id→row_key cache for table as needing a rebuild.

        Called after any INSERT, UPDATE, or DELETE on the table so that
        subsequent point-lookups see the current state.
        """
        if table in self._id_row_cache:
            self._id_row_cache[table] = None  # type: ignore[assignment]

    def _run(
        self, sql: str, params: list, named_params: "dict | None" = None
    ) -> LilliBrainCursor:
        """Dispatch a non-transaction-control SQL string to the executor.

        When an outer transaction is open, the executor's internal pager
        begin_write/commit calls are suppressed so DML accumulates in the
        outer transaction.

        For SELECT and UPDATE statements with ``WHERE id = 'literal'``, an
        in-memory id→row_key index is passed to the executor so it can skip
        the O(N) full scan and use an O(log N) B+tree point lookup instead.

        named_params carries the ``{name → value}`` map for a ``:name`` query
        through to the executor unflattened; ``params`` stays the positional
        ``?`` channel and is empty for a named query.
        """
        from iai_mcp.lillibrain.sql.executor import run  # noqa: PLC0415

        # Determine table name and stmt type from the SQL prefix so we can
        # manage the row-key cache without re-parsing the full AST here.
        sql_upper = sql.lstrip().upper()
        is_select = sql_upper.startswith("SELECT ")
        is_update = sql_upper.startswith("UPDATE ")
        is_mutating = is_update or sql_upper.startswith(("INSERT ", "DELETE "))

        # Extract table name for index lookup and cache management.
        _table_for_cache: str | None = None
        if is_mutating:
            import re as _re_local  # noqa: PLC0415
            _m = _re_local.match(
                r"(?:UPDATE|INSERT\s+(?:OR\s+\w+\s+)?INTO|DELETE\s+FROM)\s+(\w+)",
                sql.lstrip(), _re_local.IGNORECASE,
            )
            if _m:
                _table_for_cache = _m.group(1).lower()

        _table_for_select: str | None = None
        if is_select:
            import re as _re_local  # noqa: PLC0415
            _ms = _re_local.search(
                r"FROM\s+(\w+)", sql.lstrip(), _re_local.IGNORECASE,
            )
            if _ms:
                _table_for_select = _ms.group(1).lower()

        # Supply the id→row_key index for SELECT and UPDATE point-lookups.
        id_index: "dict[str, int] | None" = None
        if is_update and _table_for_cache:
            id_index = self._get_id_index_for_update(_table_for_cache)
        elif is_select and _table_for_select:
            id_index = self._get_id_index_for_update(_table_for_select)

        if self._in_transaction:
            orig = self._suppress_pager_txn()
            try:
                rows, col_names = run(
                    sql, params, self._pager, self._catalog, self._root_map,
                    id_index=id_index, named_params=named_params,
                )
            finally:
                self._restore_pager_txn(orig)
        else:
            try:
                rows, col_names = run(
                    sql, params, self._pager, self._catalog, self._root_map,
                    id_index=id_index, named_params=named_params,
                )
            except BaseException:
                # The executor opened begin_write() for this autocommit DML
                # and did not reach commit(). Roll the pager back so the
                # journal is closed and the next statement is not rejected
                # with "A write transaction is already in progress".
                try:
                    self._pager.rollback()
                except RuntimeError:
                    pass  # pager had no open journal (failure before begin_write)
                raise

        # Invalidate the id→row_key cache only on INSERT and DELETE.
        # UPDATE does NOT change row keys — only payload bytes — so the cache
        # remains valid across provenance batch-updates and can be reused for
        # all subsequent point-lookups in the same executemany batch.
        if _table_for_cache and (
            sql_upper.startswith("INSERT ") or sql_upper.startswith("DELETE ")
        ):
            self._invalidate_id_cache(_table_for_cache)

        return LilliBrainCursor(rows, col_names)

    # ------------------------------------------------------------------
    # Public execute interface
    # ------------------------------------------------------------------

    def execute(self, sql: str, params=()) -> LilliBrainCursor:
        """Execute a single SQL statement and return a LilliBrainCursor.

        Transaction-control keywords (BEGIN / COMMIT / ROLLBACK) are
        intercepted before dispatch to the engine executor. PRAGMA statements
        that have no engine equivalent are handled inline. All other DML and
        DDL is routed through the executor via _run().
        """
        with self._exec_lock:
            sql_stripped = sql.strip()
            sql_upper = sql_stripped.upper()

            # --- Read-only guard: reject writes when PRAGMA query_only=ON ---
            if self._query_only and sql_upper.startswith(
                ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ", "DROP ")
            ):
                raise sqlite3.OperationalError("attempt to write a readonly database")

            # --- Transaction control ---
            if sql_upper in (
                "BEGIN",
                "BEGIN DEFERRED",
                "BEGIN IMMEDIATE",
                "BEGIN EXCLUSIVE",
            ):
                if self._in_transaction:
                    raise sqlite3.OperationalError(
                        "cannot start a transaction within a transaction"
                    )
                self._pager.begin_write()
                self._in_transaction = True
                return LilliBrainCursor([], [])

            if sql_upper == "COMMIT":
                self._pager.commit()
                self._in_transaction = False
                return LilliBrainCursor([], [])

            if sql_upper == "ROLLBACK":
                self._pager.rollback()
                self._in_transaction = False
                return LilliBrainCursor([], [])

            # --- PRAGMA: journal_mode — switch to write-ahead mode when =WAL is set ---
            if sql_upper.startswith("PRAGMA JOURNAL_MODE"):
                # Only the assignment form (contains '=') switches mode; a bare
                # PRAGMA journal_mode read-only query returns the current mode row
                # without mutating state.
                if "=" in sql_upper:
                    mode_part = sql_upper.split("=", 1)[1].strip()
                    if mode_part == "WAL":
                        self._pager.enable_wal_mode()
                return LilliBrainCursor([{"journal_mode": "wal"}], ["journal_mode"])

            # --- PRAGMA foreign_keys (read-only query) — report as enabled ---
            if sql_upper == "PRAGMA FOREIGN_KEYS":
                return LilliBrainCursor([{"foreign_keys": 1}], ["foreign_keys"])

            # --- PRAGMA busy_timeout — round-trip the value. Writes are serialized
            # through a connection-level lock, so the timeout is advisory; the value
            # is stored and reported back for drop-in parity with sqlite3. ---
            if sql_upper.startswith("PRAGMA BUSY_TIMEOUT"):
                if "=" in sql_stripped:
                    try:
                        self._busy_timeout = int(sql_stripped.split("=", 1)[1].strip())
                    except ValueError:
                        pass
                    return LilliBrainCursor([], [])
                return LilliBrainCursor(
                    [{"busy_timeout": getattr(self, "_busy_timeout", 0)}],
                    ["busy_timeout"],
                )

            # --- PRAGMA cache_size — bound the pager's resident page cache.
            # SQLite semantics: a negative value is KiB of memory, a positive
            # value is a page count. Convert to a max-page count and set the
            # pager bound so the cache no longer grows to the full working set.
            # A zero or unparseable value leaves the pager unbounded (no crash).
            # The empty-cursor return shape is preserved for store call-site
            # parity; only the page-cache side effect is new. ---
            if sql_upper.startswith("PRAGMA CACHE_SIZE"):
                if "=" in sql_stripped:
                    page_size = getattr(self._pager, "_page_size", 0) or 1
                    raw = sql_stripped.split("=", 1)[1].strip().rstrip(";").strip()
                    try:
                        value = int(raw)
                    except ValueError:
                        value = 0
                    if value < 0:
                        max_pages: int | None = max(1, (abs(value) * 1024) // page_size)
                    elif value > 0:
                        max_pages = max(1, value)
                    else:
                        max_pages = None  # unbounded
                    self._pager.set_max_cached_pages(max_pages)
                    return LilliBrainCursor([], [])
                return LilliBrainCursor([], [])

            # --- PRAGMA wal_checkpoint — trigger real checkpoint in WAL mode ---
            if sql_upper.startswith("PRAGMA WAL_CHECKPOINT"):
                self._pager.checkpoint_wal()
                return LilliBrainCursor([], [])

            # --- PRAGMA query_only — enforce read-only contract on the engine ---
            if sql_upper.startswith("PRAGMA QUERY_ONLY"):
                if "=" in sql_upper:
                    rhs = sql_upper.split("=", 1)[1].strip().rstrip(";").strip()
                    self._query_only = rhs in ("ON", "1", "TRUE")
                    return LilliBrainCursor([], [])
                return LilliBrainCursor(
                    [{"query_only": int(self._query_only)}], ["query_only"]
                )

            # --- PRAGMAs accepted as no-ops ---
            if any(sql_upper.startswith(p) for p in self._NOOP_PRAGMA_PREFIXES):
                return LilliBrainCursor([], [])

            # VACUUM: no-op until compaction is implemented.
            if sql_upper == "VACUUM":
                return LilliBrainCursor([], [])

            # --- PRAGMA table_info → synthesize from catalog ---
            m_ti = _re.search(r"PRAGMA\s+TABLE_INFO\s*\(\s*(\w+)\s*\)", sql_stripped, _re.IGNORECASE)
            if m_ti:
                table = m_ti.group(1)
                rows = self._table_info_rows(table)
                return LilliBrainCursor(
                    rows, ["cid", "name", "type", "notnull", "dflt_value", "pk"]
                )

            # --- sqlite_master enumeration → synthesize from catalog ---
            if "SQLITE_MASTER" in sql_upper:
                tables = sorted(self._catalog._tables)  # type: ignore[attr-defined]
                rows = [{"name": t} for t in tables if not t.startswith("sqlite_")]
                return LilliBrainCursor(rows, ["name"])

            # --- CREATE TABLE → register_table (allocates root + persists DDL) ---
            if sql_upper.startswith("CREATE TABLE"):
                # CTAS: CREATE TABLE <name> AS SELECT ...
                # Detect "AS" after the table name by scanning for " AS " before
                # the first "(" (a regular CREATE TABLE always has "(" before any "AS").
                ctas_match = _re.match(
                    r"CREATE\s+TABLE\s+(\w+)\s+AS\s+(SELECT\b.*)",
                    sql_stripped,
                    _re.IGNORECASE | _re.DOTALL,
                )
                if ctas_match:
                    new_table = ctas_match.group(1)
                    select_sql = ctas_match.group(2)
                    self._execute_ctas(new_table, select_sql, params)
                    return LilliBrainCursor([], [])
                if self._in_transaction:
                    orig = self._suppress_pager_txn()
                    try:
                        register_table(self._pager, self._catalog, self._root_map, sql_stripped)
                    finally:
                        self._restore_pager_txn(orig)
                else:
                    register_table(self._pager, self._catalog, self._root_map, sql_stripped)
                return LilliBrainCursor([], [])

            # --- ALTER TABLE → apply via executor (catalog update) + persist DDL to meta table ---
            if sql_upper.startswith("ALTER TABLE"):
                # A RENAME re-keys the catalog AND the pager root_map (the B+tree
                # root page is unchanged — only the name→root mapping moves). Read
                # the old root before _run re-keys the catalog.
                is_rename = " RENAME " in f" {sql_upper} "
                rename_old = rename_new = None
                rename_root: int | None = None
                if is_rename:
                    rename_old, rename_new = _extract_rename_names(sql_stripped)
                    rename_root = self._root_map.get(rename_old)

                # Route through executor first so the live catalog is updated.
                result = self._run(sql_stripped, [])

                if is_rename and rename_root is not None:
                    # Re-key the root mapping (move, not copy).
                    self._root_map[rename_new] = self._root_map.pop(rename_old)

                # Persist the DDL to the meta table so re-open replays it correctly.
                # When an outer transaction is open, write the meta row directly
                # (pager.begin_write/commit suppressed) rather than via _write_meta_ddl_row
                # (which opens its own transaction).
                existing_rows = _read_meta_rows(self._pager)
                ddl_rows = [r for r in existing_rows if r and r[0] == "ddl"]
                seq = len(ddl_rows)
                if self._in_transaction:
                    # Write inside the existing outer transaction (direct cursor —
                    # _write_meta_ddl_row / _write_meta_root_row would open their own
                    # begin_write/commit, illegal under an open transaction).
                    cursor = _meta_cursor(self._pager)
                    meta_key = _meta_next_key(self._pager)
                    payload = encode_record(["ddl", seq, sql_stripped])
                    cursor.insert(meta_key, payload)
                    if is_rename and rename_root is not None:
                        root_key = _meta_next_key(self._pager)
                        root_payload = encode_record(["root", rename_new, rename_root])
                        cursor.insert(root_key, root_payload)
                        # Tombstone every stale root row for the old name so a
                        # re-open cannot re-alias it into root_map (mirrors the
                        # DROP path at connection.py:1020-1035).
                        for stale_key in _find_root_meta_keys(self._pager, rename_old):
                            cursor.delete(stale_key)
                else:
                    _write_meta_ddl_row(self._pager, seq, sql_stripped)
                    if is_rename and rename_root is not None:
                        # Write the new-name root row first, then tombstone every
                        # stale root row for the old name so re-open cannot
                        # re-alias it into root_map (mirrors the DROP path).
                        _write_meta_root_row(self._pager, rename_new, rename_root)
                        stale_root_keys = _find_root_meta_keys(self._pager, rename_old)
                        if stale_root_keys:
                            cursor = _meta_cursor(self._pager)
                            self._pager.begin_write()
                            for stale_key in stale_root_keys:
                                cursor.delete(stale_key)
                            self._pager.commit()
                return result

            # --- DROP TABLE → apply via executor (catalog deregister) + free root_map + persist ---
            if sql_upper.startswith("DROP TABLE"):
                drop_name = _extract_drop_name(sql_stripped)
                # Route through executor first so the catalog drops the table
                # (this raises CatalogError for a non-IF-EXISTS missing table —
                # the persist + tombstone below never run in that case).
                result = self._run(sql_stripped, [])

                # Free the live root mapping for the dropped name.
                self._root_map.pop(drop_name, None)

                # Persist a DROP DDL row so re-open replays the catalog deregister,
                # AND tombstone every stale ``root`` meta row for the dropped name so
                # the re-open replay cannot resurrect the mapping (a later CREATE of
                # the same name must land on a fresh root, not the dropped one).
                existing_rows = _read_meta_rows(self._pager)
                ddl_rows = [r for r in existing_rows if r and r[0] == "ddl"]
                seq = len(ddl_rows)
                stale_root_keys = _find_root_meta_keys(self._pager, drop_name)
                if self._in_transaction:
                    cursor = _meta_cursor(self._pager)
                    meta_key = _meta_next_key(self._pager)
                    payload = encode_record(["ddl", seq, sql_stripped])
                    cursor.insert(meta_key, payload)
                    for root_key in stale_root_keys:
                        cursor.delete(root_key)
                else:
                    _write_meta_ddl_row(self._pager, seq, sql_stripped)
                    if stale_root_keys:
                        cursor = _meta_cursor(self._pager)
                        self._pager.begin_write()
                        for root_key in stale_root_keys:
                            cursor.delete(root_key)
                        self._pager.commit()
                return result

            # --- INSERT: inject vec_label AUTOINCREMENT value if absent ---
            # --- All other DML / DDL (SELECT, INSERT, UPDATE, DELETE…) ---
            # A Mapping param is a :name dict — route it through unflattened as
            # named_params (list()-ing it would collapse the dict to its keys).
            # The positional channel stays empty for a named query; a named
            # SELECT skips the INSERT-only vec_label rewrite via its early-return.
            if isinstance(params, Mapping):
                final_sql, _final_params, injected_label = self._maybe_inject_vec_label(
                    sql_stripped, []
                )
                cursor = self._run(final_sql, [], named_params=dict(params))
                cursor.lastrowid = injected_label
                return cursor

            final_sql, final_params, injected_label = self._maybe_inject_vec_label(
                sql_stripped, list(params)
            )
            cursor = self._run(final_sql, final_params)
            cursor.lastrowid = injected_label
            return cursor

    def executemany(self, sql: str, seq_of_params) -> LilliBrainCursor:
        """Execute sql once per parameter tuple in seq_of_params.

        If an outer transaction is already open, each row's DML runs under it
        (executor internal begin/commit suppressed). If no outer transaction is
        open, opens one begin_write() / commit() pair wrapping all rows so
        the batch is atomic.
        """
        with self._exec_lock:
            seq = list(seq_of_params)
            if not seq:
                return LilliBrainCursor([], [])

            if self._in_transaction:
                # Already in outer transaction — each _run() call suppresses its own txn.
                for params in seq:
                    row_sql, row_params, _ = self._maybe_inject_vec_label(sql, list(params))
                    self._run(row_sql, row_params)
            else:
                # No outer transaction — wrap the whole batch in one pager transaction.
                orig = self._suppress_pager_txn()
                self._restore_pager_txn(orig)  # restore real methods to open the real txn
                self._pager.begin_write()
                self._in_transaction = True
                # Re-suppress so individual _run() calls don't double-begin.
                orig2 = self._suppress_pager_txn()
                try:
                    for params in seq:
                        row_sql, row_params, _ = self._maybe_inject_vec_label(sql, list(params))
                        self._run(row_sql, row_params)
                except BaseException:
                    self._restore_pager_txn(orig2)
                    self._pager.rollback()
                    self._in_transaction = False
                    raise
                else:
                    self._restore_pager_txn(orig2)
                    self._pager.commit()
                    self._in_transaction = False

            return LilliBrainCursor([], [])

    def cursor(self) -> LilliBrainCursor:
        """Return a connection-bound cursor.

        The cursor starts empty; calling ``.execute(sql, params)`` on it runs
        the statement through this connection and mirrors the result onto the
        same object — the access pattern pandas' read path drives.
        """
        return LilliBrainCursor([], [], connection=self)

    def commit(self) -> None:
        """Commit the current write transaction if one is open."""
        with self._exec_lock:
            if self._in_transaction:
                self._pager.commit()
                self._in_transaction = False

    def close(self) -> None:
        """Rollback any open transaction and close the pager.

        Matches stdlib sqlite3.Connection.close(): close is NOT a commit point —
        any uncommitted transaction is discarded, not persisted.
        """
        with self._exec_lock:
            if self._in_transaction:
                try:
                    self._pager.rollback()
                except RuntimeError:
                    pass
                self._in_transaction = False
            self._pager.close()

    def __enter__(self) -> "LilliBrainConnection":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        # Matches stdlib sqlite3.Connection.__exit__: commit on success, rollback
        # on error, but leave the connection open and reusable — do NOT close the
        # pager. Callers that need to close after a with-block must call .close()
        # explicitly.
        with self._exec_lock:
            if exc_type is None:
                self.commit()
            else:
                if self._in_transaction:
                    try:
                        self._pager.rollback()
                    except RuntimeError:
                        pass
                    self._in_transaction = False

    # ------------------------------------------------------------------
    # vec_label AUTOINCREMENT helpers
    # ------------------------------------------------------------------

    # Regex to parse INSERT column list from INSERT INTO <tbl> (<cols>) VALUES ...
    _INSERT_RE = _re.compile(
        r"INSERT\s+(?:OR\s+\w+\s+)?INTO\s+(\w+)\s*\(([^)]+)\)",
        _re.IGNORECASE,
    )

    def _has_autoincrement_vec_label(self, table: str) -> bool:
        """Return True if table has a column named vec_label that is the INTEGER PK."""
        try:
            columns = self._catalog.get(table)  # type: ignore[attr-defined]
        except Exception:
            return False
        return any(
            col.name == "vec_label" and col.primary_key
            for col in columns
        )

    def _ensure_hwm_loaded(self, table: str) -> None:
        """Load the persisted high-water mark for table from the meta table if not cached."""
        if table not in self._vec_label_hwm:
            hwm = _read_hwm_from_meta(self._pager, table)
            self._vec_label_hwm[table] = hwm
            self._hwm_meta_key[table] = _find_hwm_meta_key(self._pager, table)

    def set_vec_label_hwm(self, table: str, value: int) -> None:
        """Set table's vec_label high-water mark to an absolute value, persisting it.

        Called after a verbatim bulk copy that wrote explicit vec_label values, so
        the next auto-assigned label is value + 1 and cannot collide with a copied
        row. Monotonic: a value not greater than the persisted mark is a no-op, so
        a stale max can never lower a higher persisted mark. Mirrors the engine's
        set_vec_label_hwm so the migrator can drive either backend through one call.
        """
        with self._exec_lock:
            self._ensure_hwm_loaded(table)
            if value <= self._vec_label_hwm[table]:
                return
            old_key = self._hwm_meta_key.get(table)
            _update_hwm_in_meta(self._pager, table, old_key, value)
            if old_key is None:
                self._hwm_meta_key[table] = _find_hwm_meta_key(self._pager, table)
            self._vec_label_hwm[table] = value

    def _next_vec_label(self, table: str) -> int:
        """Return the next vec_label for table, increment and persist the HWM.

        The high-water mark is persisted to the meta table immediately so it
        survives close/re-open. When an outer transaction is open, the HWM
        write runs under that transaction (no new begin_write). When no outer
        transaction is open, `_update_hwm_in_meta` opens its own.
        """
        self._ensure_hwm_loaded(table)
        hwm = self._vec_label_hwm[table]
        new_label = hwm + 1
        old_key = self._hwm_meta_key.get(table)
        if self._in_transaction:
            # Write HWM meta row directly under the existing open transaction
            # (no new begin_write — pager already has one open).
            cursor = _meta_cursor(self._pager)
            payload = encode_record(["hwm", table, new_label])
            if old_key is None:
                meta_key = _meta_next_key(self._pager)
                cursor.insert(meta_key, payload)
                self._hwm_meta_key[table] = meta_key
            else:
                cursor.insert(old_key, payload)
        else:
            # No outer transaction — persist with its own begin/commit.
            _update_hwm_in_meta(self._pager, table, old_key, new_label)
            if old_key is None:
                self._hwm_meta_key[table] = _find_hwm_meta_key(self._pager, table)
        self._vec_label_hwm[table] = new_label
        return new_label

    def _maybe_inject_vec_label(
        self, sql: str, params: list
    ) -> tuple[str, list, int | None]:
        """If sql is an INSERT into a vec_label-AUTOINCREMENT table and vec_label
        is not in the supplied column list, prepend vec_label and its next value.

        Returns (possibly-rewritten sql, possibly-extended params, injected_label).
        injected_label is the value prepended as vec_label (which becomes the
        cursor's lastrowid), or None when nothing was injected.
        """
        m = self._INSERT_RE.match(sql)
        if m is None:
            return sql, params, None
        table = m.group(1)
        col_names_raw = [c.strip() for c in m.group(2).split(",")]
        col_names_lower = [c.lower() for c in col_names_raw]
        if "vec_label" in col_names_lower:
            return sql, params, None  # caller supplied vec_label — leave as-is
        if not self._has_autoincrement_vec_label(table):
            return sql, params, None  # table does not use AUTOINCREMENT vec_label

        next_label = self._next_vec_label(table)
        # Rewrite: INSERT INTO <tbl> (vec_label, <cols>) VALUES (?, <original ?>s)
        # Replace the column list and add one leading '?' in the VALUES clause.
        new_col_list = "vec_label, " + m.group(2)
        # Replace the first match of the column-list portion in the SQL.
        new_sql = sql[: m.start(2)] + new_col_list + sql[m.end(2):]
        # Add leading '?' for vec_label in the VALUES clause.
        new_sql = _re.sub(
            r"VALUES\s*\(",
            "VALUES (??, ",
            new_sql,
            count=1,
            flags=_re.IGNORECASE,
        ).replace("??", "?", 1)
        new_params = [next_label] + params
        logger.debug(
            "_inject_vec_label: table=%r label=%d", table, next_label
        )
        return new_sql, new_params, next_label

    # ------------------------------------------------------------------
    # PRAGMA table_info helper
    # ------------------------------------------------------------------

    def raw_conn(self, *, read_only: bool = False) -> "LilliBrainRawConn":
        """Return a thin auto-commit adapter backed by this connection.

        Provides a sqlite3.Connection-compatible surface for raw read/write
        access without opening a second pager on the same file. Each execute()
        call on the returned adapter commits immediately (auto-commit semantics),
        matching the usage pattern of stdlib sqlite3 in direct-access helpers.

        When read_only=True the returned adapter enforces the RO constraint at
        its own layer, leaving this connection's _query_only state untouched.
        """
        return LilliBrainRawConn(self, read_only=read_only)

    def _execute_ctas(
        self, new_table: str, select_sql: str, params: tuple | list
    ) -> None:
        """Implement CREATE TABLE <new_table> AS SELECT ... (CTAS).

        Executes the SELECT against the live catalog, derives column names and
        TEXT affinity from the result, creates the new table with register_table,
        then bulk-inserts the rows. The column affinity defaults to TEXT because
        the parser cannot derive storage types from arbitrary SELECT expressions;
        this matches sqlite3's CREATE TABLE AS behaviour for expression columns.
        """
        from iai_mcp.lillibrain.sql.executor import run as _run_sql  # noqa: PLC0415

        # Execute the SELECT to get rows + column names.
        rows, col_names = _run_sql(
            select_sql,
            list(params),
            self._pager,
            self._catalog,
            self._root_map,
        )

        # Build a conventional CREATE TABLE DDL from the result column names.
        # All columns default to TEXT affinity (sqlite3 parity for CTAS).
        col_defs = ", ".join(f"{c} TEXT" for c in col_names)
        create_ddl = f"CREATE TABLE {new_table} ({col_defs})"

        if self._in_transaction:
            orig = self._suppress_pager_txn()
            try:
                register_table(self._pager, self._catalog, self._root_map, create_ddl)
            finally:
                self._restore_pager_txn(orig)
        else:
            register_table(self._pager, self._catalog, self._root_map, create_ddl)

        # Insert the rows from the SELECT result.
        if rows and col_names:
            placeholders = ", ".join("?" * len(col_names))
            col_list = ", ".join(col_names)
            insert_sql = f"INSERT INTO {new_table} ({col_list}) VALUES ({placeholders})"
            for row in rows:
                row_params = [row.get(c) for c in col_names]
                self._run(insert_sql, row_params)

        logger.debug(
            "CTAS: created %r with %d columns, %d rows",
            new_table, len(col_names), len(rows),
        )

    def _table_info_rows(self, table: str) -> list[dict]:
        """Synthesize PRAGMA table_info rows from catalog metadata."""
        try:
            columns = self._catalog.get(table)  # type: ignore[attr-defined]
        except Exception:
            return []
        result = []
        for i, col in enumerate(columns):
            result.append({
                "cid": i,
                "name": col.name,
                "type": col.affinity if hasattr(col, "affinity") else col.col_type,
                "notnull": int(not col.nullable),
                "dflt_value": None,
                "pk": int(col.primary_key),
            })
        return result


# ---------------------------------------------------------------------------
# LilliBrainRawConn — auto-commit adapter for direct raw access
# ---------------------------------------------------------------------------


class LilliBrainRawConn:
    """Auto-commit adapter providing a sqlite3.Connection-compatible surface.

    Used for raw direct read access without opening a second pager on the same
    file. Production paths include daemon-down recall and runtime-graph RO
    export; test fixtures use the same surface. Each execute() call auto-commits
    (matching the stdlib sqlite3 usage pattern). Backed by the same open
    LilliBrainConnection as HippoDB, or a temporary pager when HippoDB is
    closed (see _OwnedLilliBrainRawConn).

    read_only is enforced at the ADAPTER level: PRAGMA query_only=ON sets
    the adapter's own _ro flag and is intercepted before reaching the
    underlying LilliBrainConnection, so a RO adapter opened on a live
    (shared) writer connection never mutates the writer's _query_only state.
    """

    #: Set row_factory to support ``conn.row_factory = sqlite3.Row`` assignments.
    row_factory = None

    def __init__(self, conn: LilliBrainConnection, *, read_only: bool = False) -> None:
        self._conn = conn
        self._ro: bool = read_only

    def execute(self, sql: str, params=()) -> LilliBrainCursor:
        """Execute sql with params under auto-commit semantics.

        PRAGMA query_only is intercepted here: it sets this adapter's _ro flag
        without delegating to the underlying connection, so the shared writer
        connection's _query_only state is never mutated by a RO adapter.
        Write SQL is blocked when _ro is True with the same error sqlite3 raises.
        """
        sql_stripped = sql.strip()
        sql_upper = sql_stripped.upper()
        # Intercept PRAGMA query_only at the adapter — do NOT forward to the
        # shared writer connection.
        if sql_upper.startswith("PRAGMA QUERY_ONLY"):
            if "=" in sql_upper:
                rhs = sql_upper.split("=", 1)[1].strip().rstrip(";").strip()
                self._ro = rhs in ("ON", "1", "TRUE")
                return LilliBrainCursor([], [])
            return LilliBrainCursor(
                [{"query_only": int(self._ro)}], ["query_only"]
            )
        if self._ro and sql_upper.startswith(
            ("INSERT ", "UPDATE ", "DELETE ", "CREATE ", "ALTER ", "DROP ")
        ):
            raise sqlite3.OperationalError("attempt to write a readonly database")
        return self._conn.execute(sql, params)

    def commit(self) -> None:
        """No-op: each execute already committed."""

    def close(self) -> None:
        """No-op: pager lifecycle managed by HippoDB."""


class _OwnedLilliBrainRawConn(LilliBrainRawConn):
    """LilliBrainRawConn that owns its connection and closes the pager on close().

    Used when HippoDB has already closed the original connection and a temporary
    pager is opened solely for raw direct access.
    """

    def close(self) -> None:
        """Close the underlying pager (this conn owns it)."""
        try:
            self._conn.close()
        except Exception:  # noqa: BLE001
            pass


class _OwnedEngineRawConn:
    """Raw adapter that owns a freshly opened native engine connection.

    Used on a registry miss (the store's HippoDB is closed) when the live
    storage driver is the native engine. The file on disk was written by the
    native engine's pager, whose page format — including the per-page checksum —
    differs from the reference pager's; opening the file through the reference
    pager and writing to it produces pages the engine rejects on its next open
    (checksum mismatch). So the raw path must reopen the SAME engine that owns
    the file. This adapter holds that engine connection and releases it (and its
    exclusive file lock) on close().

    The native engine's own raw adapter enforces the read-only gate; this object
    delegates execute/commit to it and adds ownership of the underlying
    connection's lifecycle.

    A writer-mode adapter registers its engine connection under the store path so
    that a later in-process open at the same path borrows the shared connection
    instead of attempting a second exclusive-lock open (which the single-writer
    lock would reject). It deregisters on close. A read-only adapter takes no
    advisory lock and is not registered (it is one of potentially many concurrent
    readers).
    """

    row_factory = None

    def __init__(
        self, engine_conn, *, read_only: bool = False, path: str | None = None
    ) -> None:
        self._engine_conn = engine_conn
        self._read_only = read_only
        self._path = path
        self._raw = engine_conn.raw_conn(read_only=read_only)
        # A writer holds the exclusive lock; publish it so a later open borrows
        # this connection rather than colliding with the lock. Readers take no
        # lock and stay unregistered.
        if not read_only and path is not None:
            register_lilli_conn(path, engine_conn)

    def execute(self, sql: str, params=()):
        return self._raw.execute(sql, params)

    def refresh(self) -> bool:
        """Re-arm a read-only adapter to the newest committed snapshot in place.

        Returns True when the visible snapshot moved. Read-only adapters only —
        the engine rejects a refresh on a writer connection.
        """
        return bool(self._raw.refresh())

    def col_index_ready(self, table: str) -> bool:
        return bool(self._raw.col_index_ready(table))

    def id_index_ready(self, table: str) -> bool:
        return bool(self._raw.id_index_ready(table))

    def cells_visited_count(self) -> int:
        return int(self._raw.cells_visited_count())

    def commit(self) -> None:
        # A read-only open has no write transaction to commit; commit is a no-op
        # so a reader's commit() (the stdlib usage pattern) never errors.
        if self._read_only:
            return
        self._engine_conn.commit()

    def close(self) -> None:
        # Drop this adapter's own raw-conn clone of the shared backing connection
        # first, so the engine refcount can reach the last holder: the raw adapter
        # holds an Arc clone that, if left live, keeps the strong-count above one
        # and blocks the refcount-gated file/lock release below.
        raw_close = getattr(self._raw, "close", None)
        if callable(raw_close):
            try:
                raw_close()
            except Exception:  # noqa: BLE001
                pass
        if not self._read_only and self._path is not None:
            try:
                self._engine_conn.close()
            finally:
                # Always drop this owner's registry slot on close: the engine
                # handle is now closed and can no longer be borrowed. A borrower
                # that is still live keeps the backing store (file/lock) alive on
                # its own — the engine releases the file only at the true last
                # holder's close/drop, independent of the registry. The registry
                # is a borrow-routing optimization for the owner's open window,
                # not the keeper of the file/lock lifecycle; leaving a closed
                # owner handle in it would only hand out a dead connection.
                deregister_lilli_conn(self._path, self._engine_conn)
            return
        try:
            self._engine_conn.close()
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "_OwnedEngineRawConn":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Release the engine connection (and its exclusive advisory lock) on
        # every exit path, including an exception — a writer raw open must not
        # rely on garbage collection to free the lock. Does not suppress.
        self.close()
        return False


# ---------------------------------------------------------------------------
# Connection registry — maps db_path to the active LilliBrainConnection
# ---------------------------------------------------------------------------

#: Registry populated by HippoDB when LILLI_STORAGE_DRIVER=lilli.
#: Allows raw-access helpers to obtain the existing connection for a path
#: without opening a second pager on the same file.
_LILLI_CONN_REGISTRY: dict[str, "LilliBrainConnection"] = {}


def register_lilli_conn(path: str, conn: "LilliBrainConnection") -> None:
    """Register a connection under its canonical path string.

    If a different live connection is already registered for this path, the
    existing slot is preserved to avoid orphaning the first writer. This
    prevents a second HippoDB open on the same file (e.g. a transient RO open)
    from silently overwriting the writer's registry entry and then removing it
    on close, which would force the next open_store_conn call into a second
    pager on the same live B+tree file.
    """
    existing = _LILLI_CONN_REGISTRY.get(path)
    if existing is not None and existing is not conn:
        # A different live connection owns this slot — do not overwrite it.
        return
    _LILLI_CONN_REGISTRY[path] = conn


def deregister_lilli_conn(path: str, conn: "LilliBrainConnection | None" = None) -> None:
    """Remove a connection from the registry on close.

    When conn is provided only removes the slot if it still maps to that
    specific connection object, so a transient secondary open cannot deregister
    the primary writer's slot on its own close().
    """
    if conn is None:
        _LILLI_CONN_REGISTRY.pop(path, None)
    elif _LILLI_CONN_REGISTRY.get(path) is conn:
        _LILLI_CONN_REGISTRY.pop(path, None)


def get_lilli_engine_conn(path: str):
    """Return the live engine connection registered for ``path``, or None.

    Used by a second open at the same path in the same process to reuse the
    already-open connection instead of opening a second pager on the file (which
    would fail the engine's exclusive advisory lock). The returned object shares
    the registered connection's backing store but is a distinct handle: holding
    it keeps the shared backing connection live, so the first opener's close()
    does not tear the store out from under this borrower — the file/lock releases
    only when the last shared handle closes. The borrower does not own the
    lifecycle (it does not register or deregister).

    A registered connection whose backing store was already closed (its file/lock
    released) is treated as absent: it is dropped from the registry and ``None``
    is returned, so the caller opens a fresh connection rather than borrowing a
    released handle that can serve no I/O.
    """
    conn = _LILLI_CONN_REGISTRY.get(path)
    if conn is None:
        return None
    is_closed = getattr(conn, "is_closed", False)
    if is_closed:
        # Stale entry — the owner closed the engine handle without deregistering.
        # Drop it so the next open creates a live connection instead of borrowing
        # a released one.
        if _LILLI_CONN_REGISTRY.get(path) is conn:
            _LILLI_CONN_REGISTRY.pop(path, None)
        return None
    # Hand out a distinct shared handle (a clone of the backing store) when the
    # engine supports it, so the borrow is its own holder of the shared
    # connection and an owner close cannot release the file while it is live.
    share = getattr(conn, "share", None)
    if callable(share):
        return share()
    return conn


def get_lilli_raw_conn(path: str, *, read_only: bool = False) -> "LilliBrainRawConn | None":
    """Return a LilliBrainRawConn for path if the lilli driver owns it, else None.

    When the owning HippoDB is still open and read_only=False (the writer path),
    returns an adapter backed by the existing shared connection (no second pager).
    When the owning HippoDB is still open and read_only=True, opens a dedicated
    lock-free handle via engine.Connection.open_read_only so that a read scan
    does not contend on the live writer connection's exclusive mutex.  When the
    owning HippoDB has been closed (registry entry removed), opens a temporary
    fresh connection on the file if it exists and is not a sqlite3 database.

    The caller is responsible for closing the returned adapter via its close()
    method to free the underlying pager (both the registry-hit RO path and the
    registry-miss path return an _OwnedEngineRawConn that owns its engine handle).

    When read_only=True the returned adapter enforces the RO constraint at its
    own layer; the live writer connection's write path is never touched.
    """
    conn = get_lilli_engine_conn(path)
    if conn is not None:
        if not read_only:
            # Writer path: borrow the shared connection — no second pager.
            return conn.raw_conn(read_only=False)
        # Read-only path on a registry hit: open a dedicated lock-free handle
        # so a long read scan does not share the writer connection's mutex.
        # Mirror the same open/wrap pattern as the registry-miss branch below.
        import sqlite3 as _sqlite3  # noqa: PLC0415
        try:
            from iai_mcp_native import engine as _engine  # noqa: PLC0415

            engine_conn = _engine.Connection.open_read_only(str(path), 0)
            # Adopt the live writer's built index pairs (Arc clones stamped
            # with the write-generation; the engine verifies the stamp and
            # the column set per table before installing). Keeps a fresh
            # reader from paying its own lazy whole-table index builds on
            # the first indexed read — the writer maintains these
            # incrementally across every committed write, so they are
            # always current at the reader's snapshot when the stamps match.
            # NEVER wait on a busy writer for this: adoption is only an
            # accelerator, and a blocking export stalls every RO open (and
            # with it the recall path) for the length of the writer's op.
            try:
                _try_export = getattr(conn, "try_export_built_indexes", None)
                _snap = (
                    _try_export()
                    if callable(_try_export)
                    else conn.export_built_indexes()
                )
                if _snap is not None:
                    engine_conn.adopt_built_indexes(_snap)
            except Exception:  # noqa: BLE001 -- adoption is an accelerator;
                # the lazy build stays the always-correct fallback.
                pass
            return _OwnedEngineRawConn(engine_conn, read_only=True, path=str(path))
        except _sqlite3.DatabaseError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise _sqlite3.DatabaseError(
                f"file is not a database (corrupt lilli file): {path}"
            ) from exc

    # Registry miss — HippoDB may have been closed already.
    # Try opening a fresh temporary connection if the path is a known B+tree file.
    from pathlib import Path as _Path  # noqa: PLC0415

    p = _Path(path)
    if not p.exists():
        return None
    # Heuristic: sqlite3 files start with "SQLite format 3\x00"; LilliBrain files do not.
    try:
        with p.open("rb") as _fh:
            header = _fh.read(16)
    except OSError:
        return None
    if header[:6] == b"SQLite":
        return None  # genuine sqlite3 file — let the original sqlite3.connect handle it

    # File exists and is not sqlite3 — reopen the SAME engine that wrote it.
    # The on-disk file was produced by the native engine pager (the live storage
    # driver), whose page format the reference pager does not share, so the raw
    # path must reopen through the native engine; opening the file through the
    # reference pager and writing to it would produce pages the engine rejects on
    # its next open. A corrupt or unreadable file raises here; re-raise as
    # sqlite3.DatabaseError so callers see the same exception type as stdlib's
    # "file is not a database" and the diagnostic is "corrupt database", not
    # "no engine connection".
    import sqlite3 as _sqlite3  # noqa: PLC0415
    try:
        from iai_mcp_native import engine as _engine  # noqa: PLC0415

        if read_only:
            # Lock-free read-only open: a reader process (the ANN-index rebuild
            # worker, a daemon-down recall reader) reads committed rows while a
            # writer process holds the store's exclusive lock. The engine takes
            # no advisory lock on this path, so it never blocks on the writer.
            engine_conn = _engine.Connection.open_read_only(str(path), 0)
        else:
            engine_conn = _engine.Connection.open(str(path), 0)
        return _OwnedEngineRawConn(engine_conn, read_only=read_only, path=str(path))
    except _sqlite3.DatabaseError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise _sqlite3.DatabaseError(
            f"file is not a database (corrupt lilli file): {path}"
        ) from exc
