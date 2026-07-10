"""Recursive-descent parser: SQL string → AST statement node.

Accepts exactly the statement catalogue used by lilli (SELECT, INSERT, UPDATE,
DELETE, CREATE TABLE/INDEX, ALTER TABLE ADD COLUMN, PRAGMA) and rejects every
out-of-subset construct (JOIN, WITH/CTE, window OVER, GROUP BY, subqueries,
UNION/INTERSECT/EXCEPT, DISTINCT) with a descriptive ParseError at parse time.

The parser never receives a rejected shape — rejection occurs at parse time so
the executor never evaluates an unsupported statement.
"""
from __future__ import annotations

from iai_mcp.lillibrain.sql.ast_nodes import (
    BinOp,
    Coalesce,
    ColumnRef,
    DDLStmt,
    DeleteStmt,
    ExcludedRef,
    Expr,
    FuncCall,
    InList,
    InsertStmt,
    IsNull,
    Literal,
    NamedParam,
    SelectStmt,
    Stmt,
    UnaryOp,
    UpdateStmt,
)
from iai_mcp.lillibrain.sql.lexer import TT, Token, tokenize


# ---------------------------------------------------------------------------
# Parse error
# ---------------------------------------------------------------------------


class ParseError(SyntaxError):
    """Raised when the SQL string does not match the supported grammar subset."""


# ---------------------------------------------------------------------------
# Parser bounds — protect against pathological input
# ---------------------------------------------------------------------------

_MAX_NESTING_DEPTH: int = 50    # maximum parenthesis nesting in expressions
_MAX_IN_LIST_SIZE: int = 1000   # maximum items in an IN (...) list

# Sentinel returned by _parse_select_item for a COUNT(*) aggregate item in a
# grouped select list, so the select-list loop can route it to group_select.
_COUNT_STAR_MARKER: str = "\x00COUNT(*)"

# Keywords that introduce unsupported join forms
_JOIN_KEYWORDS: frozenset[str] = frozenset({
    "JOIN", "LEFT", "RIGHT", "INNER", "OUTER", "CROSS", "FULL", "NATURAL",
})


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------


class Parser:
    """Recursive-descent parser for the lilli SQL subset."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._depth = 0  # tracks parenthesis nesting for DoS guard

    # ---------------------------------------------------------------------------
    # Token navigation helpers
    # ---------------------------------------------------------------------------

    def _peek(self) -> Token:
        """Return the current token without consuming it."""
        return self._tokens[self._pos]

    def _peek_value(self) -> str:
        """Return the upper-cased value of the current token."""
        return self._tokens[self._pos].value.upper()

    def _advance(self) -> Token:
        """Consume and return the current token."""
        tok = self._tokens[self._pos]
        self._pos += 1
        return tok

    def _expect(self, value: str) -> Token:
        """Consume the current token; raise ParseError if its value doesn't match."""
        tok = self._peek()
        if tok.value.upper() != value.upper():
            raise ParseError(
                f"expected {value!r}, got {tok.value!r} at position {self._pos}"
            )
        return self._advance()

    def _expect_type(self, tt: TT) -> Token:
        """Consume the current token; raise ParseError if its type doesn't match."""
        tok = self._peek()
        if tok.type != tt:
            raise ParseError(
                f"expected token type {tt.name}, got {tok.type.name} ({tok.value!r})"
            )
        return self._advance()

    def _expect_name(self) -> Token:
        """Consume and return the current token as an identifier name.

        Accepts both IDENT and KEYWORD tokens so that SQL reserved words
        (e.g. key, value, type, text) can be used as column names in positions
        where the grammar unambiguously expects an identifier — INSERT column
        lists, UPDATE SET targets, ON CONFLICT key lists, SELECT columns, and
        CREATE TABLE column names.
        """
        tok = self._peek()
        if tok.type not in (TT.IDENT, TT.KEYWORD):
            raise ParseError(
                f"expected identifier or name, got {tok.type.name} ({tok.value!r})"
            )
        return self._advance()

    def _at_eof(self) -> bool:
        return self._peek().type == TT.EOF

    def _at_keyword(self, *values: str) -> bool:
        """Return True if the current token's value matches any of the given keywords.

        Checks both KEYWORD and IDENT token types so that words not in the
        lexer's keyword set are still caught by reject guards.
        """
        tok = self._peek()
        return (
            tok.type in (TT.KEYWORD, TT.IDENT)
            and tok.value.upper() in {v.upper() for v in values}
        )

    def _at_punct(self, value: str) -> bool:
        return self._peek().type == TT.PUNCT and self._peek().value == value

    # ---------------------------------------------------------------------------
    # Top-level dispatch
    # ---------------------------------------------------------------------------

    def parse(self) -> Stmt:
        """Parse the full token list and return the statement AST node."""
        tok = self._peek()
        if tok.type == TT.EOF:
            raise ParseError("empty SQL statement")

        upper = tok.value.upper()

        if upper == "WITH":
            raise ParseError(
                "CTEs (WITH ... AS) are not supported in this SQL subset"
            )

        if upper == "EXPLAIN":
            # Consume EXPLAIN and optional QUERY PLAN, then parse the inner
            # statement as a DDLStmt passthrough so the connection layer returns
            # an empty rowset without crashing.  sqlite3 consumers that call
            # EXPLAIN only to verify the planner doesn't error are satisfied by
            # a non-raising empty result.
            self._advance()  # consume EXPLAIN
            if self._at_keyword("QUERY"):
                self._advance()  # QUERY
                if self._at_keyword("PLAN"):
                    self._advance()  # PLAN
            # Consume all remaining tokens; return a DDLStmt passthrough.
            raw_parts: list[str] = []
            while not self._at_eof():
                raw_parts.append(self._advance().value)
            return DDLStmt(kind="pragma", raw=" ".join(raw_parts))

        if upper == "SELECT":
            stmt = self._parse_select()
        elif upper == "INSERT":
            stmt = self._parse_insert()
        elif upper == "UPDATE":
            stmt = self._parse_update()
        elif upper == "DELETE":
            stmt = self._parse_delete()
        elif upper in ("CREATE", "ALTER", "DROP", "PRAGMA"):
            stmt = self._parse_ddl()
        else:
            raise ParseError(f"unsupported statement type: {tok.value!r}")

        # Reject set operations after a parsed SELECT
        if self._at_keyword("UNION", "INTERSECT", "EXCEPT"):
            kw = self._peek().value.upper()
            raise ParseError(
                f"{kw} is not supported; set operations are outside the supported grammar"
            )

        # Any token left after a fully parsed statement is an unsupported tail
        # (e.g. a second ORDER BY key, a trailing clause). Fail closed here so
        # such a clause raises instead of being silently dropped.
        if not self._at_eof():
            tok = self._peek()
            raise ParseError(
                f"unexpected trailing tokens after statement: "
                f"{tok.value!r} at position {self._pos}"
            )

        return stmt

    # ---------------------------------------------------------------------------
    # SELECT
    # ---------------------------------------------------------------------------

    def _parse_select(self) -> SelectStmt:
        """Parse a SELECT statement."""
        self._expect("SELECT")

        # Reject DISTINCT
        if self._at_keyword("DISTINCT"):
            raise ParseError(
                "DISTINCT is not supported in this SQL subset"
            )

        # Parse the select-list: *, col list, COUNT(*), or SUM(col)
        aggregate: str | None = None
        aggregate_column: str | None = None
        count_alias: str | None = None
        columns: list[str] = []
        select_exprs: list[tuple[Expr, str]] = []
        # Grouped select list: ordered (kind, name) items where kind is "col"
        # for a plain grouped column and "count" for a COUNT(*) [AS alias] item.
        group_select: list[tuple[str, str]] = []
        group_count_alias: str | None = None

        if self._peek().type == TT.PUNCT and self._peek().value == "*":
            self._advance()
            columns = ["*"]
        elif self._at_keyword("COUNT") and self._count_star_is_sole_select():
            self._advance()  # consume COUNT
            self._expect("(")
            self._expect("*")
            self._expect(")")
            aggregate = "count"
            aggregate_column = None
            # A sole COUNT(*) may carry an output label (``COUNT(*) AS n``); the
            # alias names the result column the consumer reads the count by. An
            # unaliased scalar count keeps the default "COUNT(*)" label.
            count_alias = self._parse_optional_alias()
        elif self._at_keyword("SUM"):
            self._advance()  # consume SUM
            self._expect("(")
            col_tok = self._expect_name()
            aggregate_column = col_tok.value
            self._expect(")")
            aggregate = "sum"
        elif self._at_keyword("MIN", "MAX"):
            agg_kw = self._advance().value.lower()  # consume MIN / MAX
            self._expect("(")
            col_tok = self._expect_name()
            aggregate_column = col_tok.value
            self._expect(")")
            aggregate = agg_kw
        else:
            # Heterogeneous column list: each item is a plain column, a
            # ``col [AS] alias``, a ``COALESCE(a, b) [AS] alias``, or a
            # ``COUNT(*) [AS alias]`` aggregate alongside grouped columns. Items
            # are parsed in source order. A list mixing a COUNT(*) item with one
            # or more plain columns is a grouped projection (carried in
            # group_select); a list that is aliased / has a COALESCE is carried
            # in select_exprs; an all-plain list stays on the ``columns`` fast
            # path.
            items: list[tuple["Expr | str", str | None]] = [self._parse_select_item()]
            while self._at_punct(","):
                self._advance()
                items.append(self._parse_select_item())

            has_count = any(node == _COUNT_STAR_MARKER for node, _ in items)
            if has_count:
                # Grouped projection: plain columns take their first-row value per
                # group; the COUNT(*) item carries the group count under its alias
                # (or the default "COUNT(*)" label). Each item is recorded as
                # (kind, output_name, source_name) so positional access matches
                # sqlite3's column order. A COALESCE in a grouped list is outside
                # the supported subset (the consumers never mix them).
                for node, alias in items:
                    if node == _COUNT_STAR_MARKER:
                        group_count_alias = alias or "COUNT(*)"
                        group_select.append(("count", group_count_alias, ""))
                    elif isinstance(node, str):
                        group_select.append(("col", alias or node, node))
                    else:
                        raise ParseError(
                            "a grouped COUNT(*) select list supports only plain "
                            "columns beside the aggregate"
                        )
            elif all(isinstance(node, str) and alias is None for node, alias in items):
                # Pure plain-column list — stay on the fast path.
                columns = [node for node, _ in items]  # type: ignore[misc]
            else:
                for node, alias in items:
                    if isinstance(node, str):
                        # Plain column (possibly aliased) -> ColumnRef.
                        select_exprs.append((ColumnRef(node), alias or node))
                    elif alias is None:
                        # Literal expressions without an alias use their string
                        # representation as the output name (sqlite3 parity:
                        # SELECT 1 → column named "1"; SELECT 'x' → column "x").
                        if isinstance(node, Literal):
                            lit_name = str(node.value) if node.value is not None else "NULL"
                            select_exprs.append((node, lit_name))
                        else:
                            raise ParseError(
                                "a COALESCE projection requires an output name "
                                "(use AS <name>)"
                            )
                    else:
                        select_exprs.append((node, alias))

        # FROM <table>
        self._expect("FROM")
        table_tok = self._peek()
        if table_tok.type not in (TT.IDENT, TT.KEYWORD):
            raise ParseError(f"expected table name, got {table_tok.value!r}")
        table = self._advance().value

        # Reject joins after the table name
        self._reject_join()

        # Optional WHERE
        where = self._parse_where()

        # Optional GROUP BY <col> [HAVING COUNT(*) <op> <int>]. A grouped query
        # buckets rows by the group column after the WHERE filter; a bare select
        # column not in the GROUP BY takes its first-row-in-group value (sqlite3
        # semantics). HAVING filters the resulting groups by their COUNT(*).
        group_by_col: str | None = None
        having_op: str | None = None
        having_value: int | None = None
        if self._at_keyword("GROUP"):
            self._advance()
            self._expect("BY")
            group_by_col = self._expect_name().value
            if self._at_keyword("HAVING"):
                self._advance()
                self._expect("COUNT")
                self._expect("(")
                self._expect("*")
                self._expect(")")
                op_tok = self._peek()
                if op_tok.type != TT.OP or op_tok.value not in (
                    ">=", ">", "<=", "<", "=", "!=", "<>"
                ):
                    raise ParseError(
                        f"HAVING COUNT(*) expects a comparison operator, got "
                        f"{op_tok.value!r}"
                    )
                having_op = self._advance().value
                if having_op == "<>":
                    having_op = "!="
                val_tok = self._expect_type(TT.INTEGER)
                having_value = int(val_tok.value)
        elif self._at_keyword("HAVING"):
            # HAVING without GROUP BY is rejected (matches sqlite3 for the shapes
            # under test — a bare HAVING aggregate is outside the supported set).
            raise ParseError("HAVING without GROUP BY is not supported")

        # Optional ORDER BY <col> [ASC|DESC] [, <col> [ASC|DESC]] — up to two
        # keys. A third comma-separated key is left unconsumed so the
        # end-of-input assertion in parse() rejects it (three-key ORDER BY is
        # outside this subset; the consumers need at most two).
        order_by_col: str | None = None
        order_by_desc = False
        order_by_col2: str | None = None
        order_by_desc2 = False
        order_by_datetime = False
        order_by_datetime2 = False
        if self._at_keyword("ORDER"):
            self._advance()
            self._expect("BY")
            order_by_col, order_by_datetime = self._parse_order_by_key()
            if self._at_keyword("DESC"):
                self._advance()
                order_by_desc = True
            elif self._at_keyword("ASC"):
                self._advance()
            if self._at_punct(","):
                self._advance()
                order_by_col2, order_by_datetime2 = self._parse_order_by_key()
                if self._at_keyword("DESC"):
                    self._advance()
                    order_by_desc2 = True
                elif self._at_keyword("ASC"):
                    self._advance()

        # Optional LIMIT — an integer literal or a '?' placeholder bound from
        # params at execution time (the consumer's keyset / recency queries use
        # LIMIT ?). limit_is_param signals the placeholder form; limit stays None
        # and the executor pops the bound value after the WHERE params.
        # An optional OFFSET clause after LIMIT is also accepted.
        limit: int | None = None
        limit_is_param = False
        offset: int | None = None
        offset_is_param = False
        if self._at_keyword("LIMIT"):
            self._advance()
            if self._peek().type == TT.PLACEHOLDER:
                self._advance()
                limit_is_param = True
            else:
                limit_tok = self._expect_type(TT.INTEGER)
                limit = int(limit_tok.value)
            # Optional OFFSET clause immediately after the LIMIT value.
            if self._at_keyword("OFFSET"):
                self._advance()
                if self._peek().type == TT.PLACEHOLDER:
                    self._advance()
                    offset_is_param = True
                else:
                    offset_tok = self._expect_type(TT.INTEGER)
                    offset = int(offset_tok.value)

        return SelectStmt(
            table=table,
            columns=columns,
            where=where,
            limit=limit,
            limit_is_param=limit_is_param,
            offset=offset,
            offset_is_param=offset_is_param,
            aggregate=aggregate,
            aggregate_column=aggregate_column,
            count_alias=count_alias,
            order_by_col=order_by_col,
            order_by_desc=order_by_desc,
            order_by_col2=order_by_col2,
            order_by_desc2=order_by_desc2,
            order_by_datetime=order_by_datetime,
            order_by_datetime2=order_by_datetime2,
            select_exprs=select_exprs,
            group_by_col=group_by_col,
            group_count_alias=group_count_alias,
            group_select=group_select,
            having_op=having_op,
            having_value=having_value,
        )

    def _parse_order_by_key(self) -> tuple[str, bool]:
        """Read one ORDER BY key, returning ``(column_name, is_datetime_wrapped)``.

        Accepts both a plain ``<col>`` key and a ``datetime(<col>)`` key. For the
        wrapped form the inner argument must be a single bare column name; the
        underlying column name is returned with the flag set so the executor
        sorts by that column's normalised timestamp value. A plain key returns
        the flag unset and the raw column name.
        """
        if (
            self._at_keyword("DATETIME")
            and self._pos + 1 < len(self._tokens)
            and self._tokens[self._pos + 1].type == TT.PUNCT
            and self._tokens[self._pos + 1].value == "("
        ):
            self._advance()  # consume DATETIME
            self._expect("(")
            col = self._expect_name().value
            self._expect(")")
            return (col, True)
        return (self._expect_name().value, False)

    def _parse_select_item(self) -> tuple["Expr | str", str | None]:
        """Parse one SELECT-list item: a plain column, ``col [AS] alias``,
        ``COALESCE(a, b) [AS] alias``, or a ``COUNT(*) [AS] alias`` aggregate.

        Returns ``(node, alias)`` where ``node`` is the bare column name (str)
        for a plain column, a ``Coalesce`` expression node, or the
        ``_COUNT_STAR_MARKER`` sentinel for a COUNT(*) aggregate item; ``alias``
        is the output name or None. A function call other than the supported
        ``COALESCE`` / ``COUNT(*)`` (or a window ``OVER``) raises ParseError so
        unsupported projections are rejected at parse time rather than silently
        dropped.
        """
        # COUNT(*) aggregate item inside a grouped select list (``c, COUNT(*)``).
        if self._at_keyword("COUNT") and self._peek_next_is_open_paren():
            self._advance()  # consume COUNT
            self._expect("(")
            self._expect("*")
            self._expect(")")
            alias = self._parse_optional_alias()
            return (_COUNT_STAR_MARKER, alias)

        # COALESCE(<expr>, <fallback>) projection.
        if self._at_keyword("COALESCE") and self._peek_next_is_open_paren():
            self._advance()  # consume COALESCE
            self._expect("(")
            args: list[Expr] = [self._parse_primary()]
            while self._at_punct(","):
                self._advance()
                args.append(self._parse_primary())
            self._expect(")")
            if len(args) != 2:
                raise ParseError(
                    f"COALESCE expects exactly 2 arguments, got {len(args)}"
                )
            alias = self._parse_optional_alias()
            return (Coalesce(args=args), alias)

        # Scalar literal in the select list (e.g. SELECT 1 FROM t).
        # The _parse_primary path already handles INTEGER, REAL, STRING, BLOB,
        # NULL, TRUE, FALSE — reuse it here so literal-only select items work
        # without duplicating literal-parsing logic.
        tok = self._peek()
        if tok.type in (TT.INTEGER, TT.REAL, TT.STRING, TT.BLOB, TT.PLACEHOLDER, TT.NAMED_PARAM):
            lit_expr = self._parse_primary()
            alias = self._parse_optional_alias()
            return (lit_expr, alias)
        # NULL / TRUE / FALSE keywords used as literals in the select list.
        if tok.type == TT.KEYWORD and tok.value.upper() in ("NULL", "TRUE", "FALSE"):
            lit_expr = self._parse_primary()
            alias = self._parse_optional_alias()
            return (lit_expr, alias)

        # Plain column name.
        name = self._expect_name().value
        # Reject a function-call '(' after a non-COALESCE identifier — window
        # functions appear as ident() OVER (...); catch before FROM is expected.
        if self._at_punct("("):
            self._advance()  # consume (
            depth = 1
            while not self._at_eof() and depth > 0:
                if self._at_punct("("):
                    depth += 1
                elif self._at_punct(")"):
                    depth -= 1
                self._advance()
            if self._at_keyword("OVER"):
                raise ParseError(
                    "window functions (OVER) are not supported in this SQL subset"
                )
            raise ParseError(
                "function calls are not supported in the SELECT list; "
                "use COUNT(*) or SUM(col) for aggregates, or COALESCE(col, lit)"
            )
        alias = self._parse_optional_alias()
        return (name, alias)

    def _peek_next_is_open_paren(self) -> bool:
        """True when the token after the current one is an opening parenthesis."""
        nxt = self._pos + 1
        return (
            nxt < len(self._tokens)
            and self._tokens[nxt].type == TT.PUNCT
            and self._tokens[nxt].value == "("
        )

    def _count_star_is_sole_select(self) -> bool:
        """True when ``COUNT(*)`` at the current position is the only select item.

        The current token is COUNT. A sole-aggregate select (``SELECT COUNT(*)
        FROM ...``) is followed by FROM; a grouped select list (``SELECT c,
        COUNT(*) ...`` or ``SELECT COUNT(*), c ...``) puts a comma after the
        COUNT(*) item (optionally after an ``[AS] alias``). The sole form routes
        to the scalar-aggregate path; the mixed form routes to group_select.
        """
        # Require COUNT ( * ) at the current position; anything else is not the
        # sole-aggregate shape and falls through to the mixed-list parser.
        toks = self._tokens
        p = self._pos
        if not (
            p + 3 < len(toks)
            and toks[p + 1].type == TT.PUNCT and toks[p + 1].value == "("
            and toks[p + 2].type == TT.PUNCT and toks[p + 2].value == "*"
            and toks[p + 3].type == TT.PUNCT and toks[p + 3].value == ")"
        ):
            return False
        # Token after ``)``: FROM -> sole aggregate; an alias then FROM is also
        # sole; a comma -> mixed grouped list.
        after = p + 4
        if after >= len(toks):
            return True
        nxt = toks[after]
        if nxt.type == TT.PUNCT and nxt.value == ",":
            return False
        # ``COUNT(*) AS n FROM`` / ``COUNT(*) n FROM`` is still a sole aggregate
        # only when no comma follows the (optional) alias before FROM.
        if nxt.value.upper() == "AS" and after + 1 < len(toks):
            after += 2  # skip AS <alias>
        elif nxt.type in (TT.IDENT, TT.KEYWORD) and nxt.value.upper() != "FROM":
            after += 1  # skip a bare alias
        if after < len(toks) and toks[after].type == TT.PUNCT and toks[after].value == ",":
            return False
        return True

    def _parse_optional_alias(self) -> str | None:
        """Consume an ``AS <name>`` (or a bare alias name) after a select item.

        ``AS`` is optional in sqlite3, so a bare identifier that is not a clause
        keyword (FROM, etc.) is also accepted as the alias. Returns the alias or
        None when the next token starts the FROM clause / a comma / EOF.
        """
        if self._at_keyword("AS"):
            self._advance()
            return self._expect_name().value
        # A bare alias: a following identifier that does not begin the next
        # clause. FROM always terminates the select list; a comma starts the
        # next item. Anything else identifier-shaped here is a bare alias.
        tok = self._peek()
        if (
            tok.type in (TT.IDENT, TT.KEYWORD)
            and not self._at_keyword("FROM")
        ):
            return self._advance().value
        return None

    # ---------------------------------------------------------------------------
    # INSERT
    # ---------------------------------------------------------------------------

    def _parse_insert(self) -> InsertStmt:
        """Parse an INSERT statement."""
        self._expect("INSERT")

        or_ignore = False
        or_replace = False
        if self._at_keyword("OR"):
            self._advance()
            if self._at_keyword("IGNORE"):
                self._advance()
                or_ignore = True
            elif self._at_keyword("REPLACE"):
                self._advance()
                or_replace = True
            else:
                raise ParseError(
                    f"expected IGNORE or REPLACE after OR, got "
                    f"{self._peek().value!r} at position {self._pos}"
                )

        self._expect("INTO")
        table = self._expect_name().value

        # Column list
        self._expect("(")
        columns: list[str] = [self._expect_name().value]
        while self._at_punct(","):
            self._advance()
            columns.append(self._expect_name().value)
        self._expect(")")

        self._expect("VALUES")

        # Values list
        self._expect("(")
        values: list = [self._parse_primary()]
        while self._at_punct(","):
            self._advance()
            values.append(self._parse_primary())
        self._expect(")")

        conflict_keys: list[str] = []
        conflict_action: str | None = None
        set_clauses: list[tuple[str, object]] = []

        # Optional ON CONFLICT
        if self._at_keyword("ON"):
            self._advance()
            self._expect("CONFLICT")
            self._expect("(")
            conflict_keys.append(self._expect_name().value)
            while self._at_punct(","):
                self._advance()
                conflict_keys.append(self._expect_name().value)
            self._expect(")")
            self._expect("DO")

            if self._at_keyword("NOTHING"):
                self._advance()
                conflict_action = "nothing"
            elif self._at_keyword("UPDATE"):
                self._advance()
                conflict_action = "update"
                self._expect("SET")
                set_clauses = self._parse_set_clauses(allow_excluded=True)
            else:
                raise ParseError(
                    f"expected NOTHING or UPDATE after DO, got {self._peek().value!r}"
                )

        return InsertStmt(
            table=table,
            columns=columns,
            values=values,
            or_ignore=or_ignore,
            or_replace=or_replace,
            conflict_keys=conflict_keys,
            conflict_action=conflict_action,
            set_clauses=set_clauses,
        )

    # ---------------------------------------------------------------------------
    # UPDATE
    # ---------------------------------------------------------------------------

    def _parse_update(self) -> UpdateStmt:
        """Parse an UPDATE statement."""
        self._expect("UPDATE")
        table = self._expect_name().value
        self._expect("SET")
        set_clauses = self._parse_set_clauses(allow_excluded=False)
        where = self._parse_where()
        return UpdateStmt(table=table, set_clauses=set_clauses, where=where)

    # ---------------------------------------------------------------------------
    # DELETE
    # ---------------------------------------------------------------------------

    def _parse_delete(self) -> DeleteStmt:
        """Parse a DELETE statement."""
        self._expect("DELETE")
        self._expect("FROM")
        table = self._expect_name().value
        where = self._parse_where()
        return DeleteStmt(table=table, where=where)

    # ---------------------------------------------------------------------------
    # DDL and PRAGMA
    # ---------------------------------------------------------------------------

    def _parse_ddl(self) -> DDLStmt:
        """Parse a DDL or PRAGMA statement; preserve raw SQL, return DDLStmt."""
        first = self._peek().value.upper()
        # Consume all remaining tokens to capture raw SQL
        raw_parts: list[str] = []
        while not self._at_eof():
            raw_parts.append(self._advance().value)

        raw = " ".join(raw_parts)

        if first == "CREATE":
            # Determine CREATE TABLE vs CREATE INDEX by the second keyword
            # in the raw token stream (after CREATE). A full-string "INDEX"
            # check would misclassify CREATE TABLE statements that contain a
            # column named "*_index".
            raw_words = raw.split()
            # raw starts with CREATE; next significant word is TABLE or INDEX
            # (possibly preceded by IF NOT EXISTS for CREATE INDEX IF NOT EXISTS).
            second = raw_words[1].upper() if len(raw_words) > 1 else ""
            if second == "INDEX" or (second == "UNIQUE" and len(raw_words) > 2 and raw_words[2].upper() == "INDEX"):
                kind = "create_index"
            else:
                kind = "create_table"
        elif first == "ALTER":
            # Distinguish RENAME TO from ADD / DROP COLUMN by scanning the raw
            # tokens. RENAME is checked first: it moves the table's B+tree root
            # under a new name (a catalog + root_map re-key, not a column edit).
            raw_upper = raw.upper()
            if " RENAME " in raw_upper:
                kind = "alter_rename_table"
            elif " DROP " in raw_upper:
                kind = "alter_drop_column"
            else:
                kind = "alter_add_column"
        elif first == "DROP":
            kind = "drop_table"
        else:
            kind = "pragma"

        return DDLStmt(kind=kind, raw=raw)

    # ---------------------------------------------------------------------------
    # SET clause (UPDATE and ON CONFLICT DO UPDATE)
    # ---------------------------------------------------------------------------

    def _parse_set_clauses(
        self, *, allow_excluded: bool
    ) -> list[tuple[str, object]]:
        """Parse col=expr, col=expr, ... returning a list of (col, expr) pairs."""
        clauses: list[tuple[str, object]] = []
        col = self._expect_name().value
        self._expect("=")
        val = self._parse_set_rhs(allow_excluded=allow_excluded)
        clauses.append((col, val))

        while self._at_punct(","):
            self._advance()
            col = self._expect_name().value
            self._expect("=")
            val = self._parse_set_rhs(allow_excluded=allow_excluded)
            clauses.append((col, val))

        return clauses

    def _parse_set_rhs(self, *, allow_excluded: bool) -> object:
        """Parse the right-hand side of a SET assignment."""
        # Check for excluded.col form
        if (
            allow_excluded
            and self._peek().type == TT.KEYWORD
            and self._peek().value.upper() == "EXCLUDED"
        ):
            self._advance()  # consume EXCLUDED
            self._expect(".")
            col_tok = self._expect_name()
            return ExcludedRef(col_name=col_tok.value)
        return self._parse_primary()

    # ---------------------------------------------------------------------------
    # WHERE clause
    # ---------------------------------------------------------------------------

    def _parse_where(self) -> object | None:
        """Parse optional WHERE clause; returns Expr or None."""
        if not self._at_keyword("WHERE"):
            return None
        self._advance()
        return self._parse_expr()

    # ---------------------------------------------------------------------------
    # Expression grammar (precedence: OR → AND → NOT → comparison → primary)
    # ---------------------------------------------------------------------------

    def _parse_expr(self) -> object:
        """Parse a boolean expression at the OR precedence level."""
        return self._parse_or()

    def _parse_or(self) -> object:
        """Parse OR-level boolean expression."""
        left = self._parse_and()
        while self._at_keyword("OR"):
            self._advance()
            right = self._parse_and()
            left = BinOp(op="OR", left=left, right=right)
        return left

    def _parse_and(self) -> object:
        """Parse AND-level boolean expression."""
        left = self._parse_not()
        while self._at_keyword("AND"):
            self._advance()
            right = self._parse_not()
            left = BinOp(op="AND", left=left, right=right)
        return left

    def _parse_not(self) -> object:
        """Parse NOT-level boolean expression."""
        if self._at_keyword("NOT"):
            self._advance()
            # NOT LIKE is handled at comparison level — just wrap the comparison
            operand = self._parse_comparison()
            return UnaryOp(op="NOT", operand=operand)
        return self._parse_comparison()

    def _parse_comparison(self) -> object:
        """Parse a comparison, IS NULL, IS NOT NULL, IN, or LIKE expression."""
        left = self._parse_primary()

        # OVER after an identifier — reject window functions
        if self._at_keyword("OVER"):
            raise ParseError(
                "window functions (OVER) are not supported in this SQL subset"
            )

        # IS [NOT] NULL
        if self._at_keyword("IS"):
            self._advance()
            negated = False
            if self._at_keyword("NOT"):
                self._advance()
                negated = True
            self._expect("NULL")
            return IsNull(operand=left, negated=negated)

        # IN (...)
        if self._at_keyword("IN"):
            self._advance()
            self._expect("(")
            # Reject (SELECT ...) subquery
            if self._at_keyword("SELECT"):
                raise ParseError(
                    "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead"
                )
            items = self._parse_in_items()
            self._expect(")")
            return InList(operand=left, items=items)

        # NOT IN (...)
        if self._at_keyword("NOT") and self._tokens[self._pos + 1].value.upper() == "IN":
            self._advance()  # NOT
            self._advance()  # IN
            self._expect("(")
            if self._at_keyword("SELECT"):
                raise ParseError(
                    "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead"
                )
            items = self._parse_in_items()
            self._expect(")")
            return UnaryOp(op="NOT", operand=InList(operand=left, items=items))

        # LIKE / NOT LIKE (infix forms)
        if self._at_keyword("LIKE"):
            self._advance()
            right = self._parse_primary()
            return BinOp(op="LIKE", left=left, right=right)

        # NOT LIKE (infix form): col NOT LIKE pattern
        if self._at_keyword("NOT") and self._pos + 1 < len(self._tokens) and self._tokens[self._pos + 1].value.upper() == "LIKE":
            self._advance()  # NOT
            self._advance()  # LIKE
            right = self._parse_primary()
            return BinOp(op="NOT LIKE", left=left, right=right)

        # Binary comparison operator
        if self._peek().type == TT.OP:
            op = self._advance().value
            right = self._parse_primary()
            return BinOp(op=op, left=left, right=right)

        return left

    def _parse_in_items(self) -> list:
        """Parse a variable-length comma-separated list for IN (...)."""
        items = [self._parse_primary()]
        while self._at_punct(","):
            self._advance()
            if self._at_keyword("SELECT"):
                raise ParseError(
                    "subquery in IN (...) is not supported; use parameterized IN (?, ...) instead"
                )
            items.append(self._parse_primary())
            if len(items) > _MAX_IN_LIST_SIZE:
                raise ParseError(
                    f"IN list exceeds maximum allowed size of {_MAX_IN_LIST_SIZE}; "
                    "reduce the number of items or use a temporary table"
                )
        return items

    def _parse_primary(self) -> object:
        """Parse a primary expression: literal, placeholder, column ref, or sub-expr."""
        tok = self._peek()

        # Parenthesised expression
        if tok.type == TT.PUNCT and tok.value == "(":
            self._advance()
            self._depth += 1
            if self._depth > _MAX_NESTING_DEPTH:
                raise ParseError(
                    f"expression nesting depth exceeds maximum of {_MAX_NESTING_DEPTH}; "
                    "simplify the predicate"
                )
            # Reject (SELECT ...) subquery
            if self._at_keyword("SELECT"):
                raise ParseError(
                    "subquery is not supported in this SQL subset"
                )
            expr = self._parse_expr()
            self._expect(")")
            self._depth -= 1
            return expr

        # Placeholder
        if tok.type == TT.PLACEHOLDER:
            self._advance()
            return Literal(value="?")

        # Named placeholder (:name) — resolved from a dict param at bind time.
        if tok.type == TT.NAMED_PARAM:
            self._advance()
            return NamedParam(name=tok.value[1:])

        # String literal
        if tok.type == TT.STRING:
            self._advance()
            # Strip surrounding single quotes
            raw = tok.value
            if raw.startswith("'") and raw.endswith("'"):
                raw = raw[1:-1]
            return Literal(value=raw)

        # Hex-BLOB literal: x'<hex>' -> bytes. Strip the leading x/X and quotes,
        # then decode the hex body (empty body -> b""). An odd-length body fails
        # closed via bytes.fromhex (ValueError) rather than silently truncating.
        if tok.type == TT.BLOB:
            self._advance()
            return Literal(value=bytes.fromhex(tok.value[2:-1]))

        # Integer literal
        if tok.type == TT.INTEGER:
            self._advance()
            return Literal(value=int(tok.value))

        # Real literal
        if tok.type == TT.REAL:
            self._advance()
            return Literal(value=float(tok.value))

        # Keywords that map to literal values
        if tok.type == TT.KEYWORD:
            upper = tok.value.upper()
            if upper == "NULL":
                self._advance()
                from iai_mcp.lillibrain.sql.types import SqlNull
                return Literal(value=SqlNull)
            if upper == "TRUE":
                self._advance()
                return Literal(value=1)
            if upper == "FALSE":
                self._advance()
                return Literal(value=0)

        # COALESCE(<expr>, <fallback>) — two-argument first-non-NULL function.
        # Matched before the bare-identifier fall-through so the trailing call
        # parens are consumed (otherwise COALESCE parses as a ColumnRef and the
        # '(' becomes an unconsumed trailing token).
        if (
            tok.type in (TT.IDENT, TT.KEYWORD)
            and tok.value.upper() == "COALESCE"
            and self._pos + 1 < len(self._tokens)
            and self._tokens[self._pos + 1].type == TT.PUNCT
            and self._tokens[self._pos + 1].value == "("
        ):
            self._advance()  # consume COALESCE
            self._expect("(")
            args = [self._parse_primary()]
            while self._at_punct(","):
                self._advance()
                args.append(self._parse_primary())
            self._expect(")")
            if len(args) != 2:
                raise ParseError(
                    f"COALESCE expects exactly 2 arguments, got {len(args)}"
                )
            from iai_mcp.lillibrain.sql.ast_nodes import Coalesce  # noqa: PLC0415
            return Coalesce(args=args)

        # datetime(<expr>) — single-argument scalar timestamp function. Matched
        # before the bare-identifier fall-through so the trailing call parens are
        # consumed (otherwise datetime parses as a ColumnRef and the '(' becomes
        # an unconsumed trailing token). The single argument is parsed on the
        # depth-guarded path so nested datetime(datetime(...)) trips the nesting
        # bound rather than recursing without limit.
        if (
            tok.type in (TT.IDENT, TT.KEYWORD)
            and tok.value.upper() == "DATETIME"
            and self._pos + 1 < len(self._tokens)
            and self._tokens[self._pos + 1].type == TT.PUNCT
            and self._tokens[self._pos + 1].value == "("
        ):
            self._advance()  # consume DATETIME
            self._depth += 1
            if self._depth > _MAX_NESTING_DEPTH:
                raise ParseError(
                    f"expression nesting depth exceeds maximum of {_MAX_NESTING_DEPTH}; "
                    "simplify the predicate"
                )
            self._expect("(")
            args = [self._parse_primary()]
            while self._at_punct(","):
                self._advance()
                args.append(self._parse_primary())
            self._expect(")")
            self._depth -= 1
            if len(args) != 1:
                raise ParseError(
                    f"datetime() expects exactly 1 argument, got {len(args)}"
                )
            return FuncCall(name="datetime", args=args)

        # EXCLUDED keyword in SET clause (handled by _parse_set_rhs; guard here)
        if tok.type == TT.KEYWORD and tok.value.upper() == "EXCLUDED":
            # If we reach here from _parse_primary in an expression context, it is invalid
            raise ParseError(
                f"unexpected keyword EXCLUDED in expression context"
            )

        # Identifier — column reference, possibly followed by . (table.col).
        # Accepts both IDENT and KEYWORD tokens so that SQL reserved words used
        # as column names (e.g. key, value, type, text) parse correctly in WHERE,
        # SELECT column list, and other expression positions.
        if tok.type in (TT.IDENT, TT.KEYWORD):
            name = self._advance().value
            # table.col form (e.g. excluded.col is handled above; this catches other dotted refs)
            if self._at_punct("."):
                self._advance()
                col = self._expect_name().value
                return ColumnRef(name=f"{name}.{col}")
            # Check OVER immediately after an identifier — window function guard
            if self._at_keyword("OVER"):
                raise ParseError(
                    f"window functions (OVER) are not supported in this SQL subset"
                )
            return ColumnRef(name=name)

        raise ParseError(f"unexpected token {tok.value!r} in expression")

    # ---------------------------------------------------------------------------
    # Reject-list guards
    # ---------------------------------------------------------------------------

    def _reject_join(self) -> None:
        """Raise ParseError if the next token introduces a join."""
        if self._at_keyword(*_JOIN_KEYWORDS):
            tok_val = self._peek().value
            raise ParseError(
                f"JOIN ({tok_val}) is not supported; only single-table queries are allowed"
            )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def parse_sql(sql: str) -> Stmt:
    """Tokenize and parse a SQL string, returning the statement AST node.

    Raises ParseError on unsupported constructs, SyntaxError on illegal tokens.
    """
    tokens = tokenize(sql)
    parser = Parser(tokens)
    return parser.parse()
