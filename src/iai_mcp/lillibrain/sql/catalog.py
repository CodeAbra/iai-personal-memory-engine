"""In-memory schema catalog: column metadata and table registry.

Populated by DDL statements (CREATE TABLE, ALTER TABLE ADD COLUMN, CREATE INDEX).
CREATE INDEX is accepted and recorded as index metadata but does not create a
scan path — the planner always emits a full scan in this implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iai_mcp.lillibrain.sql.ast_nodes import DDLStmt


# ---------------------------------------------------------------------------
# Catalog error
# ---------------------------------------------------------------------------


class CatalogError(ValueError):
    """Raised when a table or column is not found in the catalog."""


# ---------------------------------------------------------------------------
# Column metadata
# ---------------------------------------------------------------------------


_SENTINEL = object()  # sentinel for "no default provided" (distinct from None)


@dataclass
class ColumnMeta:
    """Metadata for one column in the schema catalog."""

    name: str
    affinity: str = "TEXT"       # "INTEGER", "TEXT", "REAL", "BLOB", "NUMERIC"
    nullable: bool = True
    primary_key: bool = False
    default: object = field(default_factory=lambda: _SENTINEL)


# ---------------------------------------------------------------------------
# Type-affinity derivation
# ---------------------------------------------------------------------------

_AFFINITY_MAP: dict[str, str] = {
    "INTEGER": "INTEGER",
    "INT":     "INTEGER",
    "REAL":    "REAL",
    "FLOAT":   "REAL",
    "DOUBLE":  "REAL",
    "BLOB":    "BLOB",
    "NONE":    "BLOB",
    "NUMERIC": "NUMERIC",
    "DECIMAL": "NUMERIC",
    "BOOLEAN": "NUMERIC",
    "DATE":    "NUMERIC",
    "DATETIME":"NUMERIC",
    "TEXT":    "TEXT",
    "CHAR":    "TEXT",
    "CLOB":    "TEXT",
    "VARCHAR": "TEXT",
}


def _affinity_for(type_token: str) -> str:
    """Return the column affinity for a declared type token.

    Follows the sqlite3 affinity rules: match the longest-prefix type token
    against the affinity map; fall back to TEXT for unknown types.
    """
    upper = type_token.upper()
    # Direct lookup first
    if upper in _AFFINITY_MAP:
        return _AFFINITY_MAP[upper]
    # Substring match for compound type names (e.g. "VARYING CHARACTER")
    for keyword, affinity in _AFFINITY_MAP.items():
        if keyword in upper:
            return affinity
    return "TEXT"


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


@dataclass
class Catalog:
    """Registry mapping table names to their ordered column metadata list."""

    _tables: dict[str, list[ColumnMeta]] = field(default_factory=dict)
    # Index metadata: table → list of (index_name, column, is_partial) tuples.
    # Stored for completeness; never used to select an index scan path.
    _indexes: dict[str, list[tuple[str, str, bool]]] = field(default_factory=dict)

    def register(self, table: str, columns: list[ColumnMeta]) -> None:
        """Register (or replace) a table with the given column list."""
        self._tables[table] = list(columns)

    def column_names(self, table: str) -> list[str]:
        """Return the column names for table in declaration order.

        Raises CatalogError if the table has not been registered.
        """
        if table not in self._tables:
            raise CatalogError(f"catalog: unknown table {table!r}")
        return [col.name for col in self._tables[table]]

    def get(self, table: str) -> list[ColumnMeta]:
        """Return the ColumnMeta list for table.

        Raises CatalogError if the table has not been registered.
        """
        if table not in self._tables:
            raise CatalogError(f"catalog: unknown table {table!r}")
        return list(self._tables[table])

    def apply_ddl(self, stmt: DDLStmt) -> None:  # type: ignore[name-defined]
        """Update the catalog from a parsed DDL statement.

        Handles CREATE TABLE (registers columns), ALTER TABLE ADD COLUMN
        (appends one column), and CREATE INDEX (records index metadata without
        creating any scan path). PRAGMA statements are silently ignored.
        """
        kind = stmt.kind

        if kind == "create_table":
            self._apply_create_table(stmt.raw)
        elif kind == "alter_add_column":
            self._apply_alter_add_column(stmt.raw)
        elif kind == "alter_drop_column":
            self._apply_alter_drop_column(stmt.raw)
        elif kind == "alter_rename_table":
            self._apply_alter_rename(stmt.raw)
        elif kind == "drop_table":
            self._apply_drop_table(stmt.raw)
        elif kind == "create_index":
            self._apply_create_index(stmt.raw)
        # PRAGMA and other DDL kinds are silently accepted.

    # ---------------------------------------------------------------------------
    # DDL application helpers
    # ---------------------------------------------------------------------------

    def _apply_create_table(self, raw: str) -> None:
        """Parse the raw CREATE TABLE SQL and register the column list."""
        from iai_mcp.lillibrain.sql.lexer import TT, tokenize  # noqa: PLC0415

        tokens = tokenize(raw)
        pos = 0

        def peek() -> str:
            return tokens[pos].value.upper() if pos < len(tokens) else ""

        def advance() -> str:
            nonlocal pos
            val = tokens[pos].value
            pos += 1
            return val

        def expect(value: str) -> None:
            got = advance().upper()
            if got != value.upper():
                raise CatalogError(f"CREATE TABLE: expected {value!r}, got {got!r}")

        # CREATE TABLE [IF NOT EXISTS] <name> (...)
        expect("CREATE")
        expect("TABLE")
        if peek() == "IF":
            advance()  # IF
            expect("NOT")
            expect("EXISTS")

        table_name = advance()  # table name

        expect("(")

        columns: list[ColumnMeta] = []
        while peek() not in (")", ""):
            # column name
            col_tok = tokens[pos]
            # Handle table-level constraints (PRIMARY KEY, UNIQUE, CHECK, FOREIGN)
            if col_tok.value.upper() in ("PRIMARY", "UNIQUE", "CHECK", "FOREIGN", "CONSTRAINT"):
                is_primary = col_tok.value.upper() == "PRIMARY"
                # Consume until the opening paren, then read the column list.
                pk_col_names: list[str] = []
                while pos < len(tokens) and tokens[pos].value != "(":
                    pos += 1
                if pos < len(tokens) and tokens[pos].value == "(":
                    pos += 1  # skip "("
                    depth = 1
                    while pos < len(tokens) and depth > 0:
                        v = tokens[pos].value
                        if v == "(":
                            depth += 1
                            pos += 1
                        elif v == ")":
                            depth -= 1
                            if depth == 0:
                                pos += 1
                                break
                            pos += 1
                        elif v == "," and depth == 1:
                            pos += 1
                        else:
                            if is_primary and depth == 1:
                                pk_col_names.append(v)
                            pos += 1
                # Mark those columns as primary_key in the already-parsed list.
                if is_primary:
                    pk_set = {n.lower() for n in pk_col_names}
                    for cm in columns:
                        if cm.name.lower() in pk_set:
                            cm.primary_key = True
                            cm.nullable = False  # PRIMARY KEY implies NOT NULL
                if peek() == ",":
                    advance()
                continue

            col_name = advance()
            # type token (may be absent for implicit BLOB/TEXT)
            affinity = "TEXT"
            if (pos < len(tokens)
                    and tokens[pos].type in (TT.KEYWORD, TT.IDENT)
                    and tokens[pos].value.upper() not in (
                        ",", ")", "PRIMARY", "NOT", "UNIQUE", "DEFAULT",
                        "REFERENCES", "CHECK", "CONSTRAINT", "AUTOINCREMENT",
                    )):
                type_token = advance()
                affinity = _affinity_for(type_token)

            # Column constraints (PRIMARY KEY, NOT NULL, UNIQUE, DEFAULT, ...)
            nullable = True
            primary_key = False
            col_default: object = _SENTINEL
            while peek() not in (",", ")", "") and not tokens[pos].type == TT.EOF:
                kw = peek()
                if kw == "PRIMARY":
                    advance(); advance()  # PRIMARY KEY
                    primary_key = True
                elif kw == "NOT":
                    advance()  # NOT
                    if peek() == "NULL":
                        advance()
                        nullable = False
                elif kw == "AUTOINCREMENT":
                    advance()
                elif kw == "UNIQUE":
                    advance()
                elif kw == "DEFAULT":
                    advance()
                    # Consume and store the default value token.
                    # A hex blob literal x'' is tokenized as two tokens (IDENT "x"
                    # followed by STRING "'...'"). Consume both when the first token
                    # is the single-character hex-blob prefix.
                    if pos < len(tokens):
                        first_default = advance()
                        if (first_default.upper() == "X"
                                and pos < len(tokens)
                                and tokens[pos].type.name == "STRING"):
                            hex_str = advance()  # consume the '' string part of x''
                            # Decode x'<hex>' literal to bytes; x'' → empty blob.
                            inner = hex_str.strip("'")
                            col_default = bytes.fromhex(inner) if inner else b""
                        elif first_default.startswith("'") and first_default.endswith("'"):
                            col_default = first_default[1:-1]  # strip surrounding quotes
                        elif first_default.upper() == "NULL":
                            col_default = None
                        else:
                            # Numeric literal or unquoted keyword default
                            try:
                                col_default = int(first_default)
                            except ValueError:
                                try:
                                    col_default = float(first_default)
                                except ValueError:
                                    col_default = first_default
                elif kw == "REFERENCES":
                    advance()
                    advance()  # referenced table
                    if peek() == "(":
                        advance()  # (
                        advance()  # col
                        advance()  # )
                else:
                    break

            columns.append(
                ColumnMeta(col_name, affinity, nullable, primary_key, col_default)
            )

            if peek() == ",":
                advance()

        self._tables[table_name] = columns

    def _apply_alter_add_column(self, raw: str) -> None:
        """Parse ALTER TABLE ADD COLUMN and append the column."""
        from iai_mcp.lillibrain.sql.lexer import TT, tokenize  # noqa: PLC0415

        tokens = tokenize(raw)
        pos = 0

        def advance() -> str:
            nonlocal pos
            val = tokens[pos].value
            pos += 1
            return val

        def expect(value: str) -> None:
            got = advance().upper()
            if got != value.upper():
                raise CatalogError(f"ALTER TABLE: expected {value!r}, got {got!r}")

        def peek() -> str:
            return tokens[pos].value.upper() if pos < len(tokens) else ""

        expect("ALTER")
        expect("TABLE")
        table_name = advance()
        expect("ADD")
        if peek() == "COLUMN":
            advance()
        col_name = advance()

        affinity = "TEXT"
        if (pos < len(tokens)
                and tokens[pos].type in (TT.KEYWORD, TT.IDENT)
                and tokens[pos].value.upper() not in (",", ")")):
            affinity = _affinity_for(advance())

        col = ColumnMeta(col_name, affinity)
        if table_name not in self._tables:
            raise CatalogError(f"catalog: unknown table {table_name!r} in ALTER TABLE")
        self._tables[table_name].append(col)

    def _apply_alter_drop_column(self, raw: str) -> None:
        """Parse ALTER TABLE DROP COLUMN and remove the column from the catalog.

        The physical B+tree rows retain their encoded column positions, but the
        catalog no longer includes the dropped column in SELECT projections or
        INSERT layouts. Columns that are not present in the catalog are silently
        ignored on re-open (the raw positional value is never decoded).
        """
        from iai_mcp.lillibrain.sql.lexer import tokenize  # noqa: PLC0415

        tokens = tokenize(raw)
        pos = 0

        def advance() -> str:
            nonlocal pos
            val = tokens[pos].value
            pos += 1
            return val

        def peek() -> str:
            return tokens[pos].value.upper() if pos < len(tokens) else ""

        def expect(value: str) -> None:
            got = advance().upper()
            if got != value.upper():
                raise CatalogError(f"ALTER TABLE DROP: expected {value!r}, got {got!r}")

        expect("ALTER")
        expect("TABLE")
        table_name = advance()
        expect("DROP")
        if peek() == "COLUMN":
            advance()
        col_name = advance()

        if table_name not in self._tables:
            return  # table unknown — silently ignore
        self._tables[table_name] = [
            cm for cm in self._tables[table_name] if cm.name != col_name
        ]

    def _apply_alter_rename(self, raw: str) -> None:
        """Parse ALTER TABLE <old> RENAME TO <new> and re-key the catalog entry.

        Moves the column list (and any index metadata) from <old> to <new>. The
        table's stored rows are untouched — only the name→columns mapping moves.
        Raises CatalogError if <old> is not registered (sqlite3 parity: rename of
        a missing table is a no-such-table error).
        """
        from iai_mcp.lillibrain.sql.lexer import tokenize  # noqa: PLC0415

        tokens = tokenize(raw)
        pos = 0

        def advance() -> str:
            nonlocal pos
            val = tokens[pos].value
            pos += 1
            return val

        def expect(value: str) -> None:
            got = advance().upper()
            if got != value.upper():
                raise CatalogError(f"ALTER TABLE RENAME: expected {value!r}, got {got!r}")

        expect("ALTER")
        expect("TABLE")
        old_name = advance()
        expect("RENAME")
        expect("TO")
        new_name = advance()

        if old_name not in self._tables:
            raise CatalogError(f"catalog: unknown table {old_name!r} in ALTER TABLE RENAME")

        self._tables[new_name] = self._tables.pop(old_name)
        if old_name in self._indexes:
            self._indexes[new_name] = self._indexes.pop(old_name)

    def _apply_drop_table(self, raw: str) -> None:
        """Parse DROP TABLE [IF EXISTS] <name> and deregister the table.

        Removes the table's column list (and index metadata) from the catalog.
        With IF EXISTS, dropping an absent table is a no-op; without it, a missing
        table raises CatalogError (sqlite3 parity: no-such-table error).
        """
        from iai_mcp.lillibrain.sql.lexer import tokenize  # noqa: PLC0415

        tokens = tokenize(raw)
        pos = 0

        def advance() -> str:
            nonlocal pos
            val = tokens[pos].value
            pos += 1
            return val

        def peek() -> str:
            return tokens[pos].value.upper() if pos < len(tokens) else ""

        def expect(value: str) -> None:
            got = advance().upper()
            if got != value.upper():
                raise CatalogError(f"DROP TABLE: expected {value!r}, got {got!r}")

        expect("DROP")
        expect("TABLE")
        if_exists = False
        if peek() == "IF":
            advance()  # IF
            expect("EXISTS")
            if_exists = True
        name = advance()

        if name not in self._tables:
            if if_exists:
                return  # IF EXISTS + absent table — no-op
            raise CatalogError(f"catalog: unknown table {name!r} in DROP TABLE")

        del self._tables[name]
        self._indexes.pop(name, None)

    def _apply_create_index(self, raw: str) -> None:
        """Record CREATE INDEX metadata without creating a scan path."""
        from iai_mcp.lillibrain.sql.lexer import TT, tokenize  # noqa: PLC0415

        tokens = tokenize(raw)
        pos = 0

        def peek() -> str:
            return tokens[pos].value.upper() if pos < len(tokens) else ""

        def advance() -> str:
            nonlocal pos
            val = tokens[pos].value
            pos += 1
            return val

        # CREATE INDEX [IF NOT EXISTS] <name> ON <table>(<col>) [WHERE ...]
        advance()  # CREATE
        advance()  # INDEX
        if peek() == "IF":
            advance(); advance(); advance()  # IF NOT EXISTS
        index_name = advance()
        advance()  # ON
        table_name = advance()
        advance()  # (
        col_name = advance()
        advance()  # )
        is_partial = peek() == "WHERE"
        entry = (index_name, col_name, is_partial)
        self._indexes.setdefault(table_name, []).append(entry)
