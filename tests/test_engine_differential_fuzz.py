"""Differential fuzz: the same statement stream runs against lilliengine and
stdlib sqlite3; result sets, and success/error outcomes must match within the
supported dialect. A divergence is an engine bug — or must be pinned as a
documented dialect decision, never left silent.

Statement generation stays strictly inside the engine's grammar (SELECT /
INSERT / UPDATE / DELETE, single-key ORDER BY, LIMIT, COUNT(*), IN lists,
IS NULL, AND/OR); values flow through qmark parameters so both drivers see
identical binding. IAI_FUZZ_EXAMPLES scales depth (default 50; raise for
soak runs).
"""
from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile

import pytest
from hypothesis import HealthCheck, example, given, settings
from hypothesis import strategies as st

from iai_mcp_native import engine

COLS = ("c0", "c1", "c2")
DDL = "CREATE TABLE t (c0 INTEGER, c1 TEXT, c2 REAL)"

_ints = st.one_of(st.none(), st.integers(min_value=-(2**62), max_value=2**62))
_texts = st.one_of(st.none(), st.text(max_size=40))
# Halves stay exactly representable in binary floats: equality is meaningful.
_reals = st.one_of(st.none(), st.integers(-2000, 2000).map(lambda i: i / 2.0))
_VALUE = {"c0": _ints, "c1": _texts, "c2": _reals}
_CMP_OPS = ("=", "!=", "<", ">", "<=", ">=")


@st.composite
def _where(draw, depth: int = 0):
    col = draw(st.sampled_from(COLS))
    col_ref = f"t.{col}" if draw(st.booleans()) else col
    kind = draw(
        st.sampled_from(
            ["cmp", "in", "isnull", "notnull"] + (["and", "or"] if depth < 2 else [])
        )
    )
    if kind == "cmp":
        val = draw(_VALUE[col])
        return f"{col_ref} {draw(st.sampled_from(_CMP_OPS))} ?", [val]
    if kind == "in":
        vals = draw(st.lists(_VALUE[col], min_size=1, max_size=5))
        marks = ", ".join("?" for _ in vals)
        return f"{col_ref} IN ({marks})", vals
    if kind == "isnull":
        return f"{col_ref} IS NULL", []
    if kind == "notnull":
        return f"{col_ref} IS NOT NULL", []
    left_sql, left_params = draw(_where(depth=depth + 1))
    right_sql, right_params = draw(_where(depth=depth + 1))
    return f"({left_sql}) {kind.upper()} ({right_sql})", left_params + right_params


@st.composite
def _op(draw):
    kind = draw(
        st.sampled_from(
            ["insert", "insert", "insert", "update", "delete", "select", "count"]
        )
    )
    if kind == "insert":
        row = [draw(_VALUE[c]) for c in COLS]
        return ("INSERT INTO t (c0, c1, c2) VALUES (?, ?, ?)", row, False)
    if kind == "update":
        col = draw(st.sampled_from(COLS))
        val = draw(_VALUE[col])
        w_sql, w_params = draw(_where())
        return (f"UPDATE t SET {col} = ? WHERE {w_sql}", [val] + w_params, False)
    if kind == "delete":
        w_sql, w_params = draw(_where())
        return (f"DELETE FROM t WHERE {w_sql}", w_params, False)
    if kind == "count":
        w_sql, w_params = draw(_where())
        return (f"SELECT COUNT(*) FROM t WHERE {w_sql}", w_params, True)
    cols_list = list(COLS)
    if draw(st.booleans()):
        cols_list[0] = f"t.{cols_list[0]}"
    cols = ", ".join(cols_list)
    w_sql, w_params = draw(_where())
    order = draw(st.one_of(st.none(), st.sampled_from(COLS)))
    limit = draw(st.one_of(st.none(), st.integers(0, 10)))
    sql = f"SELECT {cols} FROM t WHERE {w_sql}"
    ordered = False
    if order is not None:
        # ORDER BY never qualifies — the parser rejects a dotted ORDER BY
        # column (see test_select_dotted_order_by_currently_unsupported);
        # qualifying here would be a generator bug, not coverage.
        sql += f" ORDER BY {order} {draw(st.sampled_from(['ASC', 'DESC']))}"
        ordered = True
    if limit is not None:
        sql += f" LIMIT {limit}"
    return (sql, w_params, True)


def _canon(rows) -> list[tuple]:
    out = []
    for row in rows:
        vals = []
        for i in range(len(row) if hasattr(row, "__len__") else 3):
            vals.append(row[i])
        out.append(tuple(vals))
    return out


_SORT_KEY = repr  # total order over mixed None/int/float/str tuples


def _run(conn, sql: str, params: list, is_select: bool):
    try:
        cur = conn.execute(sql, tuple(params))
        return ("ok", _canon(cur.fetchall()) if is_select else None)
    except Exception as exc:  # noqa: BLE001 -- outcome class is the comparison unit
        return ("err", type(exc).__name__)


@settings(
    max_examples=int(os.environ.get("IAI_FUZZ_EXAMPLES", "50")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@example(
    ops=[
        ("INSERT INTO t (c0, c1, c2) VALUES (?, ?, ?)", [1, "hi", 2.5], False),
        ("SELECT c0, c1, c2 FROM t WHERE t.c0 = ?", [1], True),
    ]
)
@example(
    ops=[
        ("INSERT INTO t (c0, c1, c2) VALUES (?, ?, ?)", [1, "hi", 2.5], False),
        ("SELECT t.c0, c1, c2 FROM t WHERE c0 = ?", [1], True),
    ]
)
@given(ops=st.lists(_op(), min_size=1, max_size=20))
def test_engine_matches_sqlite3_on_random_streams(ops):
    tmp = tempfile.mkdtemp(prefix="lilli-fuzz-")
    lilli = engine.Connection.open(os.path.join(tmp, "t.lilli"), 0)
    ref = sqlite3.connect(os.path.join(tmp, "t.sqlite3"))
    try:
        lilli.execute(DDL)
        ref.execute(DDL)

        for step, (sql, params, is_select) in enumerate(ops):
            got = _run(lilli, sql, params, is_select)
            want = _run(ref, sql, params, is_select)
            assert got[0] == want[0], (
                f"step {step}: outcome diverged on {sql!r} params={params!r}: "
                f"lilli={got} sqlite3={want}"
            )
            if got[0] == "ok" and is_select:
                got_rows, want_rows = got[1], want[1]
                if " ORDER BY " in sql:
                    # A single non-unique key leaves peer order among ties
                    # unspecified in both engines — do not tighten the row-set
                    # compare below to element-wise or ties false-RED.
                    order_col = sql.split(" ORDER BY ", 1)[1].split()[0]
                    ki = COLS.index(order_col)
                    assert [r[ki] for r in got_rows] == [r[ki] for r in want_rows], (
                        f"step {step}: ORDER BY key sequence diverged on {sql!r}"
                    )
                    assert sorted(got_rows, key=_SORT_KEY) == sorted(
                        want_rows, key=_SORT_KEY
                    ), f"step {step}: ORDER BY row set diverged on {sql!r}"
                else:
                    assert sorted(got_rows, key=_SORT_KEY) == sorted(
                        want_rows, key=_SORT_KEY
                    ), f"step {step}: row set diverged on {sql!r} params={params!r}"

        final_l = _run(lilli, "SELECT c0, c1, c2 FROM t", [], True)
        final_r = _run(ref, "SELECT c0, c1, c2 FROM t", [], True)
        assert final_l[0] == final_r[0] == "ok"
        assert sorted(final_l[1], key=_SORT_KEY) == sorted(final_r[1], key=_SORT_KEY), (
            "final table state diverged after the stream"
        )
    finally:
        lilli.close()
        ref.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# UPDATE SET right-hand-side expression evaluation — own DDL, own table, so a
# divergence here is never affinity-coercion noise from the c0/c1/c2 schema.
# ---------------------------------------------------------------------------

_SET_RHS_DDL = (
    "CREATE TABLE u (id INTEGER PRIMARY KEY, a INTEGER, b INTEGER, c TEXT, "
    "d INTEGER, ts TEXT)"
)
_SET_RHS_INT_COLS = ("a", "b", "d")
_SET_RHS_DATA_COLS = ("a", "b", "c", "d", "ts")

_u_ids = st.integers(min_value=1, max_value=200)
_u_int = st.one_of(st.none(), st.integers(min_value=-1000, max_value=1000))
_u_text = st.one_of(st.none(), st.text(max_size=20))
_u_ts = st.one_of(
    st.none(),
    st.sampled_from(["2026-01-01T00:00:00", "2026-06-15T12:30:45", "not-a-date"]),
)


@st.composite
def _set_rhs(draw):
    # rowid is deliberately excluded — a rowid RHS on this id-PK-alias shape
    # would assert against lilli's rowid on the by-design divergent alias
    # shape; the rowid arm below uses its own non-PK DDL instead.
    kind = draw(
        st.sampled_from(
            (
                "bare_col",
                "dotted_col",
                "coalesce_lit",
                "coalesce_col",
                "paren_col",
                "swap",
                "datetime_col",
            )
        )
    )
    if kind == "swap":
        return "a = b, b = a", []
    if kind == "datetime_col":
        return "c = DATETIME(ts)", []
    target = draw(st.sampled_from(_SET_RHS_INT_COLS))
    if kind == "bare_col":
        src = draw(st.sampled_from(_SET_RHS_INT_COLS))
        return f"{target} = {src}", []
    if kind == "dotted_col":
        src = draw(st.sampled_from(_SET_RHS_INT_COLS))
        return f"{target} = u.{src}", []
    if kind == "coalesce_lit":
        src = draw(st.sampled_from(_SET_RHS_INT_COLS))
        val = draw(st.integers(min_value=-1000, max_value=1000))
        return f"{target} = COALESCE({src}, ?)", [val]
    if kind == "coalesce_col":
        src = draw(st.sampled_from(_SET_RHS_INT_COLS))
        fallback = draw(st.sampled_from(_SET_RHS_INT_COLS))
        return f"{target} = COALESCE({src}, {fallback})", []
    src = draw(st.sampled_from(_SET_RHS_INT_COLS))  # paren_col
    return f"{target} = ({src})", []


@st.composite
def _u_where(draw):
    kind = draw(st.sampled_from(["id_eq", "col_isnull", "col_notnull", "true"]))
    if kind == "id_eq":
        return "id = ?", [draw(_u_ids)]
    if kind == "col_isnull":
        return f"{draw(st.sampled_from(_SET_RHS_DATA_COLS))} IS NULL", []
    if kind == "col_notnull":
        return f"{draw(st.sampled_from(_SET_RHS_DATA_COLS))} IS NOT NULL", []
    return "1 = 1", []


@st.composite
def _u_op(draw):
    kind = draw(
        st.sampled_from(["insert", "insert", "insert", "update_expr", "delete", "select"])
    )
    if kind == "insert":
        row = [
            draw(_u_ids),
            draw(_u_int),
            draw(_u_int),
            draw(_u_text),
            draw(_u_int),
            draw(_u_ts),
        ]
        return ("INSERT INTO u (id, a, b, c, d, ts) VALUES (?, ?, ?, ?, ?, ?)", row, False)
    if kind == "update_expr":
        rhs_sql, rhs_params = draw(_set_rhs())
        w_sql, w_params = draw(_u_where())
        return (f"UPDATE u SET {rhs_sql} WHERE {w_sql}", rhs_params + w_params, False)
    if kind == "delete":
        w_sql, w_params = draw(_u_where())
        return (f"DELETE FROM u WHERE {w_sql}", w_params, False)
    w_sql, w_params = draw(_u_where())
    cols = ", ".join(_SET_RHS_DATA_COLS)
    return (f"SELECT {cols} FROM u WHERE {w_sql}", w_params, True)


def _set_rhs_ops(rows, update_sql, update_params=()):
    """Build a minimal (insert rows..., update, [select]) op stream for @example pins."""
    ops = [
        ("INSERT INTO u (id, a, b, c, d, ts) VALUES (?, ?, ?, ?, ?, ?)", list(row), False)
        for row in rows
    ]
    ops.append((update_sql, list(update_params), False))
    return ops


@settings(
    max_examples=int(os.environ.get("IAI_FUZZ_SET_RHS_EXAMPLES", "50")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@example(
    ops=_set_rhs_ops(
        [(1, 7, 3, "hi", None, "2026-01-01T00:00:00")],
        "UPDATE u SET a = b, b = a WHERE id = ?",
        (1,),
    )
)
@example(
    ops=_set_rhs_ops(
        [(1, 5, None, "hi", 9, "2026-01-01T00:00:00")],
        "UPDATE u SET b = COALESCE(a, d) WHERE id = ?",
        (1,),
    )
)
@example(
    ops=_set_rhs_ops(
        [(1, 5, None, "hi", None, "2026-01-01T00:00:00")],
        "UPDATE u SET c = DATETIME(ts) WHERE id = ?",
        (1,),
    )
)
@example(
    ops=_set_rhs_ops(
        [(1, 5, None, "hi", None, "2026-01-01T00:00:00")],
        "UPDATE u SET b = u.a WHERE id = ?",
        (1,),
    )
)
@given(ops=st.lists(_u_op(), min_size=1, max_size=15))
def test_engine_matches_sqlite3_on_set_rhs_streams(ops):
    tmp = tempfile.mkdtemp(prefix="lilli-fuzz-setrhs-")
    lilli = engine.Connection.open(os.path.join(tmp, "u.lilli"), 0)
    ref = sqlite3.connect(os.path.join(tmp, "u.sqlite3"))
    try:
        lilli.execute(_SET_RHS_DDL)
        ref.execute(_SET_RHS_DDL)

        for step, (sql, params, is_select) in enumerate(ops):
            got = _run(lilli, sql, params, is_select)
            want = _run(ref, sql, params, is_select)
            assert got[0] == want[0], (
                f"step {step}: outcome diverged on {sql!r} params={params!r}: "
                f"lilli={got} sqlite3={want}"
            )
            if got[0] == "ok" and is_select:
                assert sorted(got[1], key=_SORT_KEY) == sorted(want[1], key=_SORT_KEY), (
                    f"step {step}: row set diverged on {sql!r} params={params!r}"
                )

        final_sql = f"SELECT {', '.join(_SET_RHS_DATA_COLS)} FROM u"
        final_l = _run(lilli, final_sql, [], True)
        final_r = _run(ref, final_sql, [], True)
        assert final_l[0] == final_r[0] == "ok"
        assert sorted(final_l[1], key=_SORT_KEY) == sorted(final_r[1], key=_SORT_KEY), (
            "final table state diverged after the stream"
        )
    finally:
        lilli.close()
        ref.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# rowid RHS heal — a plain non-PK vec_label shape only. On an INTEGER PRIMARY
# KEY alias shape lilli's vec_label high-water mark is independent of the raw
# storage key by design; asserting stdlib parity there is a guaranteed
# false-RED and must never be wired in.
# ---------------------------------------------------------------------------

_ROWID_DDL = "CREATE TABLE v (vec_label INTEGER, id TEXT NOT NULL UNIQUE)"
_ROWID_SPELLINGS = ("rowid", "ROWID", "RowId")


@st.composite
def _rowid_heal_case(draw):
    n_before = draw(st.integers(min_value=0, max_value=5))
    delete_mask = draw(st.lists(st.booleans(), min_size=n_before, max_size=n_before))
    spelling = draw(st.sampled_from(_ROWID_SPELLINGS))
    return n_before, delete_mask, spelling


@settings(
    max_examples=int(os.environ.get("IAI_FUZZ_ROWID_EXAMPLES", "50")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@example(case=(0, [], "rowid"))
@example(case=(0, [], "ROWID"))
@example(case=(0, [], "RowId"))
@example(case=(3, [True, False, True], "rowid"))
@given(case=_rowid_heal_case())
def test_engine_matches_sqlite3_rowid_heal_on_damaged_shape(case):
    n_before, delete_mask, spelling = case
    tmp = tempfile.mkdtemp(prefix="lilli-fuzz-rowid-")
    lilli = engine.Connection.open(os.path.join(tmp, "v.lilli"), 0)
    ref = sqlite3.connect(os.path.join(tmp, "v.sqlite3"))
    try:
        for conn in (lilli, ref):
            conn.execute(_ROWID_DDL)
            for i in range(n_before):
                conn.execute(
                    "INSERT INTO v (vec_label, id) VALUES (?, ?)", (i, f"pad{i}")
                )
            for i, doomed in enumerate(delete_mask):
                if doomed:
                    conn.execute("DELETE FROM v WHERE id = ?", (f"pad{i}",))
            conn.execute("INSERT INTO v (vec_label, id) VALUES (?, ?)", (None, "target"))

        ref.execute("UPDATE v SET vec_label = rowid WHERE vec_label IS NULL")
        lilli.execute(f"UPDATE v SET vec_label = {spelling} WHERE vec_label IS NULL")

        want_rows = ref.execute("SELECT vec_label, id FROM v").fetchall()
        got_rows = [
            (r[0], r[1]) for r in lilli.execute("SELECT vec_label, id FROM v").fetchall()
        ]
        assert all(vl is not None for vl, _ in want_rows), f"stdlib fixture invariant broken: {want_rows!r}"
        assert all(vl is not None for vl, _ in got_rows), (
            f"heal left a NULL vec_label ({spelling=}, n_before={n_before}, "
            f"delete_mask={delete_mask}): lilli={got_rows!r}"
        )
        want = sorted(want_rows)
        got = sorted(got_rows)
        assert got == want, (
            f"damaged-shape rowid heal mismatch ({spelling=}, n_before={n_before}, "
            f"delete_mask={delete_mask}): lilli={got!r} stdlib={want!r}"
        )
    finally:
        lilli.close()
        ref.close()
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Filtered-COUNT(*) cache freshness after a non-indexed-column mutation. The
# generator above never declares a secondary/partial index, so the
# col-indexed cache-freshness path is otherwise unreachable. `where_sql` is
# sampled from the fixed known-arming shape set (never drawn free-form) so
# every example satisfies the ColIndex precondition by construction; a
# free-form predicate would leave the cache unarmed on most draws and make
# the arming assertion below meaningless.
# ---------------------------------------------------------------------------


def _records_ddl() -> tuple[str, list[str]]:
    from iai_mcp.hippo import _table

    return _table._DDL_RECORDS, list(_table._DDL_RECORDS_INDEXES)


_RECORDS_INSERT_SQL = (
    "INSERT INTO records (id, tier, created_at, embedding, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)

# 7 known-arming predicate shapes, single-sourced against
# tests/test_lilliengine_count_cache_freshness.py by the drift guard below.
_PREDICATE_SHAPES = {
    "single": "tombstoned_at IS NULL",
    "conjunction": "tombstoned_at IS NULL AND COALESCE(embedding_pending, 0) = 0",
    "disjunction": "tombstoned_at IS NOT NULL OR embedding_pending = 1",
    "coalesce_wrapped": "COALESCE(embedding_pending, 0) = 0",
    "early_column": "tombstoned_at IS NULL AND embedding_pending = 0",
    "trailing_column": "embedding_pending = 0 AND tombstoned_at IS NULL",
    "operand_swapped": "COALESCE(embedding_pending, 0) = 0 AND tombstoned_at IS NULL",
}


def _records_rows(n_rows: int) -> list[tuple]:
    return [
        (f"r{i}", "episodic", f"2026-01-{i + 1:02d}T00:00:00", b"\x00\x01", None, i % 2)
        for i in range(n_rows)
    ]


def _records_count(conn, where_sql: str) -> int:
    return conn.execute(f"SELECT COUNT(*) FROM records WHERE {where_sql}").fetchone()[0]


def _records_scan_count(conn, where_sql: str) -> int:
    return len(conn.execute(f"SELECT id FROM records WHERE {where_sql}").fetchall())


@st.composite
def _cache_freshness_case(draw):
    where_sql = draw(st.sampled_from(list(_PREDICATE_SHAPES.values())))
    n_rows = draw(st.integers(min_value=3, max_value=20))
    tombstone_ids = draw(
        st.lists(st.integers(min_value=0, max_value=n_rows - 1), unique=True, max_size=n_rows)
    )
    if tombstone_ids:
        restore_ids = draw(
            st.lists(st.sampled_from(tombstone_ids), unique=True, max_size=len(tombstone_ids))
        )
    else:
        restore_ids = []
    return where_sql, n_rows, tombstone_ids, restore_ids


@settings(
    max_examples=int(os.environ.get("IAI_FUZZ_CACHE_EXAMPLES", "50")),
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
@example(case=(_PREDICATE_SHAPES["single"], 12, [0, 1, 2], [1]))
@example(case=(_PREDICATE_SHAPES["conjunction"], 12, [0, 1, 2], [1]))
@example(case=(_PREDICATE_SHAPES["disjunction"], 12, [0, 1, 2], [1]))
@example(case=(_PREDICATE_SHAPES["coalesce_wrapped"], 12, [0, 1, 2], [1]))
@given(case=_cache_freshness_case())
def test_engine_matches_sqlite3_on_cache_freshness(case):
    where_sql, n_rows, tombstone_ids, restore_ids = case
    tmp = tempfile.mkdtemp(prefix="lilli-fuzz-cache-")
    ddl, indexes = _records_ddl()
    rows = _records_rows(n_rows)
    eng = engine.Connection.open(os.path.join(tmp, "records.lilli"), 384)
    sq = sqlite3.connect(os.path.join(tmp, "records.sqlite3"))
    try:
        for conn in (eng, sq):
            conn.execute(ddl)
            for stmt in indexes:
                conn.execute(stmt)
            for row in rows:
                conn.execute(_RECORDS_INSERT_SQL, row)
            if isinstance(conn, sqlite3.Connection):
                conn.commit()

        # Cache-engagement precondition: a repeat of the identical filtered
        # COUNT must not re-scan, or the DDL/shape never armed the cache and
        # every freshness assertion below would pass vacuously.
        eng.reset_full_scan_count()
        _records_count(eng, where_sql)
        first_scan_count = eng.full_scan_count()
        _records_count(eng, where_sql)
        armed_delta = eng.full_scan_count() - first_scan_count
        assert armed_delta == 0, (
            f"filtered-COUNT cache never armed for {where_sql!r} "
            f"(second identical COUNT re-scanned {armed_delta} times) — vacuous example"
        )

        def _assert_fresh(step: str) -> None:
            lilli_count = _records_count(eng, where_sql)
            lilli_scan = _records_scan_count(eng, where_sql)
            stdlib_count = _records_count(sq, where_sql)
            assert lilli_count == lilli_scan, (
                f"{step}: stale filtered COUNT for {where_sql!r}: "
                f"COUNT(*)={lilli_count} row-scan={lilli_scan}"
            )
            assert lilli_count == stdlib_count, (
                f"{step}: lilli/stdlib COUNT mismatch for {where_sql!r}: "
                f"lilli={lilli_count} stdlib={stdlib_count}"
            )

        _assert_fresh("post-arm")

        for idx in tombstone_ids:
            row_id = f"r{idx}"
            for conn in (eng, sq):
                conn.execute(
                    "UPDATE records SET tombstoned_at = ? WHERE id = ?",
                    ("2026-02-01T00:00:00", row_id),
                )
                if isinstance(conn, sqlite3.Connection):
                    conn.commit()
            _assert_fresh(f"tombstone id={row_id}")

        for idx in restore_ids:
            row_id = f"r{idx}"
            for conn in (eng, sq):
                conn.execute("UPDATE records SET tombstoned_at = NULL WHERE id = ?", (row_id,))
                if isinstance(conn, sqlite3.Connection):
                    conn.commit()
            _assert_fresh(f"restore id={row_id}")
    finally:
        eng.close()
        sq.close()
        shutil.rmtree(tmp, ignore_errors=True)


def test_cache_freshness_predicate_shapes_match_count_cache_source():
    """Drift guard: keeps the fuzzer's known-arming shape copy single-sourced
    against tests/test_lilliengine_count_cache_freshness.py, so an upstream
    change to the arming set cannot silently leave this arm vacuous."""
    from tests.test_lilliengine_count_cache_freshness import _PREDICATE_SHAPES as source_shapes

    assert _PREDICATE_SHAPES == source_shapes


@pytest.mark.xfail(strict=True, reason="parser rejects a dotted ORDER BY column")
def test_select_dotted_order_by_currently_unsupported(tmp_path):
    """`ORDER BY t.c0` parses on stdlib but is a parse-time reject on lilli —
    unlike WHERE and the select-list, ORDER BY has no dotted-column production.
    An unresolved divergence, not an accepted design decision like the
    wrong-qualifier case. strict-xfail so this trips loudly the moment the parser
    gains the production.
    """
    conn = engine.Connection.open(str(tmp_path / "t.lilli"), 0)
    try:
        conn.execute(DDL)
        conn.execute("SELECT c0, c1, c2 FROM t ORDER BY t.c0 ASC")
    finally:
        conn.close()


def test_add_column_duplicate_rejects_on_both_drivers(tmp_path):
    """A duplicate ALTER TABLE ADD COLUMN must fail with the same error class on
    both drivers, never silently accept or diverge in outcome."""
    lilli = engine.Connection.open(str(tmp_path / "dup.lilli"), 0)
    ref = sqlite3.connect(str(tmp_path / "dup.sqlite3"))
    try:
        for conn in (lilli, ref):
            conn.execute(DDL)
        got = _run(lilli, "ALTER TABLE t ADD COLUMN c0 INTEGER", [], False)
        want = _run(ref, "ALTER TABLE t ADD COLUMN c0 INTEGER", [], False)
        assert got[0] == want[0] == "err", (
            f"duplicate ADD COLUMN must be rejected on both drivers: "
            f"lilli={got} sqlite3={want}"
        )
        assert got[1] == want[1], (
            f"duplicate ADD COLUMN error class diverged: lilli={got[1]} sqlite3={want[1]}"
        )
    finally:
        lilli.close()
        ref.close()


def test_adjacent_giant_integers_do_not_collapse_in_where(tmp_path):
    # Pinned fuzzer counterexample: past 2^53 adjacent i64s round to the same
    # f64, and a comparison path that promotes integers to floats silently
    # matches the WRONG row — sqlite3 leaves the row alone, the engine
    # nulled it. Both statements must behave identically.
    conn = engine.Connection.open(str(tmp_path / "t.lilli"), 0)
    ref = sqlite3.connect(str(tmp_path / "t.sqlite3"))
    try:
        for c in (conn, ref):
            c.execute(DDL)
            c.execute(
                "INSERT INTO t (c0, c1, c2) VALUES (?, ?, ?)",
                (-4611686018427387649, None, None),
            )
            c.execute(
                "UPDATE t SET c0 = ? WHERE c0 = ?",
                (None, -4611686018427387648),
            )
        got = _canon(conn.execute("SELECT c0, c1, c2 FROM t").fetchall())
        want = _canon(ref.execute("SELECT c0, c1, c2 FROM t").fetchall())
        assert got == want == [(-4611686018427387649, None, None)]
    finally:
        conn.close()
        ref.close()


# PRAGMA index_list parity is out of scope for this generator: sqlite3
# materializes sqlite_autoindex_* rows for UNIQUE column constraints that the
# engine does not emit (see tests/test_engine_pragma_index_list.py). Do not
# add a PRAGMA index_list arm here asserting full row-set parity.

UNSUPPORTED = [
    "SELECT c0 FROM t JOIN u ON t.c0 = u.c0",
    "SELECT DISTINCT c0 FROM t",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "SELECT c0 FROM t WHERE c0 IN (SELECT c0 FROM t)",
    "SELECT c0 FROM t UNION SELECT c0 FROM t",
    "SELECT c0, COUNT(*) OVER () FROM t",
]


@pytest.mark.parametrize("sql", UNSUPPORTED)
def test_unsupported_constructs_reject_cleanly(sql, tmp_path):
    conn = engine.Connection.open(str(tmp_path / "t.lilli"), 0)
    try:
        conn.execute(DDL)
        # A clean rejection (parse-time error), never a silent misexecution.
        # The exact message varies by where the parser bails; erroring at all
        # is the property under test.
        with pytest.raises(Exception):
            conn.execute(sql)
    finally:
        conn.close()
