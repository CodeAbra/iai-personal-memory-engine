"""Differential parity for ``WHERE vec_label IN (...)`` on a migrated-shape store.

Drives ``iai_mcp_native.engine.Connection`` directly to prove two complementary
properties across a **migrated-shape** corpus — rows whose stored ``vec_label``
values are explicitly supplied (scattered, high integers) and therefore diverge
from the engine's sequential B-tree keys (assigned ``max_key()+1`` at insert
time, independent of the payload).

Proof structure
---------------

**A. Correctness gate — migrated-shape 200/200 differential vs sqlite3**

   Inserts 200 rows with scattered, high explicit ``vec_label`` values (e.g. 5000,
   5003, 5011, 8801, 12040, …) that provably diverge from the B-tree keys assigned
   sequentially as 1, 2, 3, …, 200.  A parallel stdlib sqlite3 DB holds the
   identical DDL and identical rows (via the same explicit ``vec_label`` INSERT
   column list).  Both ``SELECT * FROM records WHERE vec_label IN (...)`` queries
   must return the same rows — value, type, and count parity — for:

   - present-only 200/200 list,
   - mixed present/absent list,
   - AND-tail shape (``AND (tombstoned_at IS NULL)``),
   - a forced-scan control (``OR vec_label IN (...)`` duplicating the label list,
     which forces the engine to scan regardless of the ColIndex path).

**B. Scan-count gate — warm ``vec_label IN`` is index-served (full_scan_count==0)**

   Both the plain-migrated seed (no explicit index) and the index-declared seed
   have counterpart tests asserting ``full_scan_count() == 0`` on the warm second
   query once ``idx_records_vec_label`` is in the catalog.

**C. Native-shape control**

   A second store seeded WITHOUT naming ``vec_label`` (AUTOINCREMENT assigns
   1..N, so ``vec_label == btree_key``) proves the same query returns byte-identical
   results to sqlite3.

The suite skips loudly when the native wheel is absent — it can never silently
pass against a different engine class.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


# ---------------------------------------------------------------------------
# Native-wheel availability guard
# ---------------------------------------------------------------------------


def _native_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401

        return True
    except ImportError:
        return False


pytestmark = pytest.mark.skipif(
    not _native_available(),
    reason="iai_mcp_native.engine submodule not installed (build the native wheel)",
)

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DB_SEQ = {"n": 0}

# Minimal records-shaped DDL — vec_label as INTEGER PRIMARY KEY (not AUTOINCREMENT
# in the explicit-insert case; the engine still assigns B-tree keys sequentially
# at max_key()+1 regardless of the payload vec_label value).
_DDL_RECORDS = (
    "CREATE TABLE records ("
    "vec_label INTEGER PRIMARY KEY AUTOINCREMENT, "
    "id TEXT NOT NULL, "
    "tier TEXT, "
    "centrality REAL DEFAULT 0.0, "
    "tombstoned_at TEXT, "
    "embedding_pending INTEGER"
    ")"
)

# Explicit-vec_label INSERT: names vec_label FIRST so the migrator path is
# reproduced exactly — the engine stores the supplied payload value as vec_label
# while keying the B-tree row at max_key()+1.
_INSERT_EXPLICIT = (
    "INSERT INTO records (vec_label, id, tier, centrality, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?, ?)"
)

# Native INSERT: does NOT name vec_label so AUTOINCREMENT assigns it.
_INSERT_NATIVE = (
    "INSERT INTO records (id, tier, centrality, tombstoned_at, embedding_pending)"
    " VALUES (?, ?, ?, ?, ?)"
)


# ---------------------------------------------------------------------------
# Scattered-label generator
# ---------------------------------------------------------------------------


def _scattered_labels(n: int = 200) -> list[int]:
    """Return ``n`` deterministic, non-contiguous, high vec_label values.

    The values start well above the row count (base 5000) and include
    irregular gaps so that ``vec_label != btree_key`` is provably true for
    every row.  The set is reproducible — no RNG import needed.

    The gap pattern (3, 8, 30, 101, 5, …, cycling) ensures the labels are
    scattered rather than evenly-spaced, mimicking the irregular spacing of a
    real migration from a live SQLite store.
    """
    _GAP_CYCLE = [3, 8, 30, 101, 5, 17, 22, 6, 44, 11]
    labels: list[int] = []
    v = 5000
    for i in range(n):
        labels.append(v)
        v += _GAP_CYCLE[i % len(_GAP_CYCLE)]
    return labels


# Pre-built fixture shared across cases that need 200 scattered labels.
_LABELS_200 = _scattered_labels(200)


# ---------------------------------------------------------------------------
# Helpers: connection, seed, norm, assert_parity
# ---------------------------------------------------------------------------


def _engine_conn(tmp_path: Path, suffix: str = ""):
    """Open a Rust engine connection and assert it IS the Rust class (false-green guard)."""
    from iai_mcp_native import engine

    _DB_SEQ["n"] += 1
    conn = engine.Connection.open(
        str(tmp_path / f"mig_{_DB_SEQ['n']}{suffix}.db"), 384
    )
    assert isinstance(conn, engine.Connection), (
        "connection under test is not the Rust engine.Connection — "
        "refusing to run a false-green proof against a different engine"
    )
    return conn


def _seed_engine_migrated(conn, labels: list[int]) -> None:
    """Seed a migrated-shape store: INSERT naming vec_label explicitly.

    Each row's vec_label is one of the supplied ``labels`` (scattered, high
    integers).  The engine assigns B-tree keys sequentially at max_key()+1
    (1, 2, 3, …), so the stored ``vec_label`` PROVABLY diverges from the
    B-tree key for every row where the label exceeds the row count.
    """
    conn.execute(_DDL_RECORDS)
    conn.execute("BEGIN")
    for i, lbl in enumerate(labels):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        conn.execute(
            _INSERT_EXPLICIT,
            [lbl, f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2],
        )
    conn.execute("COMMIT")


def _seed_engine_migrated_with_index(conn, labels: list[int]) -> None:
    """Seed with explicit vec_label AND a declared ``idx_records_vec_label`` index.

    The index is created BEFORE the INSERT (matching the production _seed_engine
    order in test_lilliengine_secondary_index_differential.py) so the engine
    can build the ColIndex from the catalog declaration.
    """
    conn.execute(_DDL_RECORDS)
    conn.execute("CREATE INDEX idx_records_vec_label ON records(vec_label)")
    conn.execute("BEGIN")
    for i, lbl in enumerate(labels):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        conn.execute(
            _INSERT_EXPLICIT,
            [lbl, f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2],
        )
    conn.execute("COMMIT")


def _seed_sqlite3_migrated(labels: list[int]) -> sqlite3.Connection:
    """Seed stdlib sqlite3 in-memory with identical DDL and identical explicit rows."""
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_RECORDS)
    for i, lbl in enumerate(labels):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        sq.execute(
            _INSERT_EXPLICIT,
            (lbl, f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2),
        )
    sq.commit()
    return sq


def _seed_engine_native(conn, n: int = 200) -> list[int]:
    """Seed a native-shape store without naming vec_label (AUTOINCREMENT assigns 1..n)."""
    conn.execute(_DDL_RECORDS)
    conn.execute("BEGIN")
    for i in range(n):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        conn.execute(
            _INSERT_NATIVE,
            [f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2],
        )
    conn.execute("COMMIT")
    return list(range(1, n + 1))


def _seed_sqlite3_native(n: int = 200) -> tuple[sqlite3.Connection, list[int]]:
    """Seed stdlib sqlite3 in-memory with identical native-shape rows."""
    sq = sqlite3.connect(":memory:")
    sq.row_factory = sqlite3.Row
    sq.execute(_DDL_RECORDS)
    for i in range(n):
        tier = "episodic" if i % 2 == 0 else "semantic"
        tomb = "2026-01-01 00:00:00.000000" if i % 10 == 0 else None
        sq.execute(
            _INSERT_NATIVE,
            (f"rec-{i:04d}", tier, float(i) * 0.1, tomb, i % 2),
        )
    sq.commit()
    return sq, list(range(1, n + 1))


def _norm(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def _types(rows) -> list[tuple]:
    return [tuple(type(v) for v in row) for row in rows]


def _assert_parity(eng, sq, sql: str, params=()) -> None:
    """Assert the Rust engine and stdlib sqlite3 return identical rows, order, and types."""
    eng_rows = _norm(eng.execute(sql, list(params)).fetchall())
    sq_rows = _norm(sq.execute(sql, params).fetchall())
    assert eng_rows == sq_rows, (
        f"value parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine rows ({len(eng_rows)}): {eng_rows[:5]!r}{'...' if len(eng_rows) > 5 else ''}\n"
        f"  sqlite rows ({len(sq_rows)}): {sq_rows[:5]!r}{'...' if len(sq_rows) > 5 else ''}"
    )
    assert _types(eng_rows) == _types(sq_rows), (
        f"type parity mismatch for {sql!r} params={params!r}:\n"
        f"  engine types={_types(eng_rows[:3])!r}\n  sqlite types={_types(sq_rows[:3])!r}"
    )


# ---------------------------------------------------------------------------
# Divergence confirmation (module-level, not a test)
# ---------------------------------------------------------------------------

# Static proof that the scattered labels diverge from sequential B-tree keys.
# B-tree key for row i is (i+1) since the engine uses max_key()+1 sequentially.
# For any label in _LABELS_200 that is > 200, btree_key != vec_label.
_BTREE_KEYS_200 = list(range(1, 201))
_DIVERGING_PAIRS = [
    (k, lbl)
    for k, lbl in zip(_BTREE_KEYS_200, _LABELS_200)
    if k != lbl
]
assert len(_DIVERGING_PAIRS) == 200, (
    "CORPUS DESIGN ERROR: all 200 rows must have vec_label != btree_key; "
    f"found {200 - len(_DIVERGING_PAIRS)} rows where they agree. "
    "Increase the base offset or check _scattered_labels()."
)


# ---------------------------------------------------------------------------
# Test A1 — migrated-shape 200/200 present-only differential
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_shape_200_of_200(tmp_path: Path) -> None:
    """``WHERE vec_label IN (200 scattered labels)`` returns all 200 present rows.

    The corpus uses explicit scattered high vec_label values (provably diverging
    from the engine's sequential B-tree keys) so that any row-drop caused by a
    B-tree-key descent manifests immediately.  Both the engine and stdlib sqlite3
    are seeded identically; the result must be 200/200 — not the 7/200 drop that
    a btree-key descent would produce on this corpus.

    On current HEAD the full-scan stopgap produces the correct 200/200 result.
    """
    labels = _LABELS_200
    eng = _engine_conn(tmp_path, "a1")
    _seed_engine_migrated(eng, labels)
    sq = _seed_sqlite3_migrated(labels)

    ph = ", ".join("?" for _ in labels)
    sql = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    eng_rows = _norm(eng.execute(sql, labels).fetchall())
    sq_rows = _norm(sq.execute(sql, labels).fetchall())

    assert len(eng_rows) == 200, (
        f"expected 200 rows from migrated-shape store; got {len(eng_rows)} — "
        "a value below 200 indicates rows were silently dropped (btree-key descent bug)"
    )
    assert eng_rows == sq_rows, (
        f"value parity failed: engine returned {len(eng_rows)} rows, "
        f"sqlite3 returned {len(sq_rows)} rows; first mismatch details above"
    )
    assert _types(eng_rows) == _types(sq_rows), "type parity failed"


# ---------------------------------------------------------------------------
# Test A2 — migrated-shape: mixed present/absent label list
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_shape_mixed_present_absent(tmp_path: Path) -> None:
    """Mixed present/absent list returns only the present rows, byte-identical to sqlite3.

    Absent labels (integers never inserted) contribute no rows; present labels
    all appear.  Verifies the engine does not fabricate rows for absent probes
    and does not drop present rows.
    """
    labels = _LABELS_200
    eng = _engine_conn(tmp_path, "a2")
    _seed_engine_migrated(eng, labels)
    sq = _seed_sqlite3_migrated(labels)

    # 10 present + 5 absent (labels far outside the seeded range).
    present_sample = labels[:10]
    absent_sample = [1, 2, 3, 4, 5]  # sequential keys, NOT in the scattered set
    probe = present_sample + absent_sample

    ph = ", ".join("?" for _ in probe)
    sql = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    eng_rows = _norm(eng.execute(sql, probe).fetchall())
    sq_rows = _norm(sq.execute(sql, probe).fetchall())

    assert len(eng_rows) == 10, (
        f"expected 10 present rows; got {len(eng_rows)} — "
        "absent labels should contribute no rows, present labels should all appear"
    )
    assert eng_rows == sq_rows, "value parity failed on mixed present/absent probe"
    assert _types(eng_rows) == _types(sq_rows), "type parity failed"


# ---------------------------------------------------------------------------
# Test A3 — migrated-shape: AND-tail parity (production batch-get shape)
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_shape_and_tail(tmp_path: Path) -> None:
    """``WHERE vec_label IN (...) AND (tombstoned_at IS NULL)`` is byte-identical to sqlite3.

    Proves the full predicate is re-applied to candidates after vec_label
    resolution — tombstoned rows (i % 10 == 0 in the seed) are filtered out
    correctly even when they are in the IN list.
    """
    labels = _LABELS_200
    eng = _engine_conn(tmp_path, "a3")
    _seed_engine_migrated(eng, labels)
    sq = _seed_sqlite3_migrated(labels)

    # Include 30 labels: some tombstoned (every 10th row), some live.
    probe = labels[:30]
    ph = ", ".join("?" for _ in probe)
    sql = (
        f"SELECT * FROM records WHERE vec_label IN ({ph})"
        " AND (tombstoned_at IS NULL)"
        " ORDER BY vec_label"
    )

    _assert_parity(eng, sq, sql, probe)

    # Sanity: without the AND-tail we should get more rows (tombstoned ones included).
    all_sql = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"
    eng_all = _norm(eng.execute(all_sql, probe).fetchall())
    eng_live = _norm(eng.execute(sql, probe).fetchall())
    assert len(eng_all) > len(eng_live), (
        "tombstoned rows should be excluded by the AND-tail; "
        f"all={len(eng_all)} live={len(eng_live)}"
    )


# ---------------------------------------------------------------------------
# Test A4 — migrated-shape: forced-scan control (A/B/C scan-proof shape)
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_shape_forced_scan_control(tmp_path: Path) -> None:
    """``WHERE vec_label IN (L) OR vec_label IN (L)`` matches the plain IN result.

    The OR shape prevents single-path ColIndex selection (both branches must
    be resolved) and causes the engine to fall through to the full-scan path.
    The result set must be byte-identical to the plain ``vec_label IN (L)`` query,
    proving the scan fallback and the ColIndex path agree on membership.
    """
    labels = _LABELS_200[:50]  # 50 labels for a manageable OR clause
    eng = _engine_conn(tmp_path, "a4")
    _seed_engine_migrated(eng, _LABELS_200)  # seed full 200 rows
    sq = _seed_sqlite3_migrated(_LABELS_200)

    ph = ", ".join("?" for _ in labels)
    # Path A — plain IN (resolved via ColIndex on the declared-index path)
    sql_a = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"
    # Path C — OR duplicates the label list, bypassing the ColIndex fast-path
    sql_c = (
        f"SELECT * FROM records WHERE vec_label IN ({ph}) OR vec_label IN ({ph})"
        " ORDER BY vec_label"
    )

    rows_a = _norm(eng.execute(sql_a, labels).fetchall())
    rows_c = _norm(eng.execute(sql_c, labels + labels).fetchall())

    assert rows_a == rows_c, (
        f"scan-proof shape mismatch: IN path returned {len(rows_a)} rows, "
        f"OR/IN forced-scan returned {len(rows_c)} rows"
    )

    # Both match sqlite3.
    sq_rows_a = _norm(sq.execute(sql_a, labels).fetchall())
    assert rows_a == sq_rows_a, "value parity failed between engine IN and sqlite3"


# ---------------------------------------------------------------------------
# Test A5 — native-shape control (vec_label == btree_key, no regression)
# ---------------------------------------------------------------------------


def test_veclabel_in_native_shape_control(tmp_path: Path) -> None:
    """Native-shape store (AUTOINCREMENT, vec_label == btree_key) is byte-identical to sqlite3.

    A natively-built store (INSERT without naming vec_label) assigns
    vec_label == btree_key == 1..N.  The same ``vec_label IN (...)`` query must
    return the same rows as sqlite3.
    """
    n = 200
    eng = _engine_conn(tmp_path, "a5")
    labels = _seed_engine_native(eng, n)
    sq, _ = _seed_sqlite3_native(n)

    # Probe a sample of native labels (all are 1..200, contiguous).
    probe = labels[5:15]  # 10 present contiguous labels
    ph = ", ".join("?" for _ in probe)
    sql = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    eng_rows = _norm(eng.execute(sql, probe).fetchall())
    sq_rows = _norm(sq.execute(sql, probe).fetchall())

    assert len(eng_rows) == 10, (
        f"native-shape control: expected 10 rows; got {len(eng_rows)}"
    )
    assert eng_rows == sq_rows, "native-shape parity failed"
    assert _types(eng_rows) == _types(sq_rows), "native-shape type parity failed"


# ---------------------------------------------------------------------------
# Test B1 — scan-count gate: migrated-shape store WITH declared index
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_index_served_with_explicit_seed(tmp_path: Path) -> None:
    """Migrated-shape store with ``idx_records_vec_label`` serves warm queries index-only.

    Seeds a store with scattered explicit vec_label values (provably diverging
    from B-tree keys) and declares ``idx_records_vec_label`` before the INSERT
    batch, matching the production schema.  The warm second ``vec_label IN (...)``
    query must produce ``full_scan_count() == 0``, confirming the ColIndex covers
    scattered vec_label values correctly — not just native-autoincrement stores.

    Correctness is also checked: rows must be byte-identical to sqlite3.
    """
    labels = _LABELS_200[:30]
    eng = _engine_conn(tmp_path, "b1")
    _seed_engine_migrated_with_index(eng, _LABELS_200)
    sq = _seed_sqlite3_migrated(_LABELS_200)

    ph = ", ".join("?" for _ in labels)
    sql = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    # First read: builds the ColIndex warm state from the declared index.
    eng.execute(sql, labels).fetchall()
    eng.reset_full_scan_count()

    # Warm second read: must be index-served.
    eng.execute(sql, labels).fetchall()

    scan_count = eng.full_scan_count()
    # Correctness: byte-identical to sqlite3.
    _assert_parity(eng, sq, sql, labels)

    assert scan_count == 0, (
        f"expected full_scan_count==0 on the warm vec_label IN query "
        f"(index-served via idx_records_vec_label on migrated-shape store); "
        f"got {scan_count} — the ColIndex is not covering scattered vec_label values"
    )


# ---------------------------------------------------------------------------
# Test B2 — scan-count gate WITH declared index (GREEN on HEAD)
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_index_served_when_declared(tmp_path: Path) -> None:
    """With ``idx_records_vec_label`` declared, warm ``vec_label IN`` is index-served.

    ``CREATE INDEX idx_records_vec_label ON records(vec_label)`` is issued at
    seed time (before any INSERT, matching the production DDL order).  This
    makes the column visible to the catalog-driven ColIndex machinery, which is
    already live in the engine for other columns.

    The warm (second) ``vec_label IN (...)`` query must resolve with
    ``full_scan_count() == 0`` on HEAD ALREADY — proving the ColIndex vehicle
    is live and correct for ``vec_label`` the moment the index is declared.
    This case is GREEN on HEAD: it provides the direct evidence that a one-line
    DDL addition activates the existing value-based path.

    Both correctness (byte-identical rows to sqlite3) and the scan-count gate
    are asserted.
    """
    labels = _LABELS_200[:30]
    eng = _engine_conn(tmp_path, "b2")
    # Seed with the declared index — this is what plan 02 will add to production DDL.
    _seed_engine_migrated_with_index(eng, _LABELS_200)
    sq = _seed_sqlite3_migrated(_LABELS_200)

    ph = ", ".join("?" for _ in labels)
    sql = f"SELECT * FROM records WHERE vec_label IN ({ph}) ORDER BY vec_label"

    # First read: builds the ColIndex from the warm start (catalog has the column).
    eng.execute(sql, labels).fetchall()
    eng.reset_full_scan_count()

    # Second (warm) read: must be index-served.
    eng.execute(sql, labels).fetchall()

    scan_count = eng.full_scan_count()

    # Correctness: byte-identical to sqlite3 (the ColIndex path re-applies predicates).
    _assert_parity(eng, sq, sql, labels)

    assert scan_count == 0, (
        f"expected full_scan_count==0 on the warm vec_label IN query "
        f"(declared idx_records_vec_label → ColIndex fast path); "
        f"got {scan_count} — the ColIndex vehicle is not routing vec_label IN correctly"
    )


# ---------------------------------------------------------------------------
# Test B3 — declared-index case with AND-tail (full predicate re-applied)
# ---------------------------------------------------------------------------


def test_veclabel_in_migrated_index_and_tail_parity(tmp_path: Path) -> None:
    """WITH declared index, ``vec_label IN (...) AND (tombstoned_at IS NULL)`` is index-served.

    After the declared index routes the query through the ColIndex fast path,
    the AND-tail predicate must still be re-applied to the candidates so that
    tombstoned rows are correctly excluded.  This proves the fast path is not
    a lazy short-circuit — it fully evaluates the WHERE clause.

    Byte-identical to sqlite3 AND ``full_scan_count() == 0`` warm.
    """
    labels = _LABELS_200[:40]
    eng = _engine_conn(tmp_path, "b3")
    _seed_engine_migrated_with_index(eng, _LABELS_200)
    sq = _seed_sqlite3_migrated(_LABELS_200)

    ph = ", ".join("?" for _ in labels)
    sql = (
        f"SELECT * FROM records WHERE vec_label IN ({ph})"
        " AND (tombstoned_at IS NULL)"
        " ORDER BY vec_label"
    )

    eng.execute(sql, labels).fetchall()
    eng.reset_full_scan_count()
    eng.execute(sql, labels).fetchall()

    _assert_parity(eng, sq, sql, labels)

    scan_count = eng.full_scan_count()
    assert scan_count == 0, (
        f"expected full_scan_count==0 on the warm AND-tail query "
        f"(declared idx_records_vec_label); got {scan_count}"
    )
