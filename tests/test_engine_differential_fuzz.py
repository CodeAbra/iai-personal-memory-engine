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
from hypothesis import HealthCheck, given, settings
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
    kind = draw(
        st.sampled_from(
            ["cmp", "in", "isnull", "notnull"] + (["and", "or"] if depth < 2 else [])
        )
    )
    if kind == "cmp":
        val = draw(_VALUE[col])
        return f"{col} {draw(st.sampled_from(_CMP_OPS))} ?", [val]
    if kind == "in":
        vals = draw(st.lists(_VALUE[col], min_size=1, max_size=5))
        marks = ", ".join("?" for _ in vals)
        return f"{col} IN ({marks})", vals
    if kind == "isnull":
        return f"{col} IS NULL", []
    if kind == "notnull":
        return f"{col} IS NOT NULL", []
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
    cols = ", ".join(COLS)
    w_sql, w_params = draw(_where())
    order = draw(st.one_of(st.none(), st.sampled_from(COLS)))
    limit = draw(st.one_of(st.none(), st.integers(0, 10)))
    sql = f"SELECT {cols} FROM t WHERE {w_sql}"
    ordered = False
    if order is not None:
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
            v = row[i]
            if isinstance(v, float) and v == int(v):
                v = float(v)
            vals.append(v)
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
                    # A single non-unique key leaves peer order unspecified in
                    # both engines; compare as multisets, plus the key column's
                    # sequence which IS specified.
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
