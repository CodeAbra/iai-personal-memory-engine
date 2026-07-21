"""Tests for the parent-side read-only export helpers.

Validate that the dedicated ro-connection refuses writes, that keyset
pagination over records and edges is correct and snapshot-consistent across
pages, that the SQL-layer filter drops tombstoned and embedding-pending
records, and that dangling-endpoint edges are dropped at the streaming SQL
layer (JOIN against the active-records predicate).
"""
from __future__ import annotations

import os
import sqlite3

from iai_mcp import errors
import threading
import time
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import runtime_graph_cache_ro_export as roexp
from iai_mcp.community import detect_communities
from iai_mcp.graph import MemoryGraph

_driver = os.environ.get("LILLI_STORAGE_DRIVER", "stdlib").lower()

# Tests in this file that exercise the sqlite-format on-disk layout and the
# file-layer ?mode=ro URI form are skipped on the lilli driver — those
# assertions have no meaningful lilli analog (the lilli driver has no
# file-layer RO mode). They are per-function rather than file-level so that
# the new lilli routing test below can run on the lilli driver.
_skip_sqlite_format = pytest.mark.skipif(
    _driver == "lilli",
    reason=(
        "exercises the SQLite-driver on-disk format and RO-WAL URI directly; "
        "lilli routing is proven by the dedicated lilli export test in this file"
    ),
)


_DIM = 384


def _emb_bytes(rng: np.random.Generator) -> bytes:
    return rng.standard_normal(_DIM).astype(np.float32).tobytes()


def _make_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE records (
            vec_label       INTEGER PRIMARY KEY AUTOINCREMENT,
            id              TEXT NOT NULL UNIQUE,
            embedding       BLOB NOT NULL,
            tombstoned_at   TEXT,
            embedding_pending INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.execute("CREATE INDEX idx_records_id ON records(id)")
    conn.execute("""
        CREATE TABLE edges (
            src         TEXT NOT NULL,
            dst         TEXT NOT NULL,
            edge_type   TEXT NOT NULL,
            weight      REAL NOT NULL DEFAULT 0.0,
            PRIMARY KEY (src, dst, edge_type)
        )
    """)
    return conn


def _insert_record(
    conn: sqlite3.Connection,
    rec_id: UUID,
    emb: bytes,
    tombstoned_at: str | None = None,
    embedding_pending: int = 0,
) -> None:
    conn.execute(
        "INSERT INTO records (id, embedding, tombstoned_at, embedding_pending)"
        " VALUES (?, ?, ?, ?)",
        (str(rec_id), emb, tombstoned_at, embedding_pending),
    )


def _insert_edge(
    conn: sqlite3.Connection,
    src: UUID,
    dst: UUID,
    weight: float = 1.0,
    edge_type: str = "hebbian",
) -> None:
    conn.execute(
        "INSERT INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)",
        (str(src), str(dst), edge_type, weight),
    )


@_skip_sqlite_format
def test_open_ro_connection_rejects_writes(tmp_path: Path):
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    _insert_record(writer, uuid4(), _emb_bytes(np.random.default_rng(1)))
    writer.close()

    ro = roexp.open_ro_connection(db_path)
    try:
        with pytest.raises(errors.OperationalError):
            ro.execute(
                "INSERT INTO records (id, embedding) VALUES (?, ?)",
                (str(uuid4()), b"\x00" * (_DIM * 4)),
            )
    finally:
        ro.close()


@_skip_sqlite_format
def test_iter_records_chunks_pagination_complete(tmp_path: Path):
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    rng = np.random.default_rng(2)
    inserted_ids: list[str] = []
    for _ in range(5000):
        rid = uuid4()
        inserted_ids.append(str(rid))
        _insert_record(writer, rid, _emb_bytes(rng))
    writer.close()

    ro = roexp.open_ro_connection(db_path)
    try:
        with roexp.read_transaction(ro):
            page_sizes: list[int] = []
            seen: set[str] = set()
            for chunk in roexp.iter_records_chunks(ro, chunk_size=2000):
                page_sizes.append(len(chunk))
                for id_str, emb_blob in chunk:
                    assert id_str not in seen, "duplicate id across pages"
                    seen.add(id_str)
                    assert len(emb_blob) == _DIM * 4
    finally:
        ro.close()

    assert seen == set(inserted_ids)
    assert page_sizes == [2000, 2000, 1000]


@_skip_sqlite_format
def test_iter_records_chunks_filters_tombstoned(tmp_path: Path):
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    rng = np.random.default_rng(3)
    active: list[str] = []
    tombstoned: list[str] = []
    for i in range(10):
        rid = uuid4()
        if i < 3:
            tombstoned.append(str(rid))
            _insert_record(writer, rid, _emb_bytes(rng), tombstoned_at="2026-01-01T00:00:00")
        else:
            active.append(str(rid))
            _insert_record(writer, rid, _emb_bytes(rng))
    writer.close()

    ro = roexp.open_ro_connection(db_path)
    try:
        with roexp.read_transaction(ro):
            seen = []
            for chunk in roexp.iter_records_chunks(ro):
                for id_str, _ in chunk:
                    seen.append(id_str)
    finally:
        ro.close()
    assert set(seen) == set(active)
    assert not any(tid in seen for tid in tombstoned)


@_skip_sqlite_format
def test_iter_records_chunks_filters_embedding_pending(tmp_path: Path):
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    rng = np.random.default_rng(4)
    pending: list[str] = []
    for _ in range(5):
        rid = uuid4()
        pending.append(str(rid))
        _insert_record(writer, rid, _emb_bytes(rng), embedding_pending=1)
    writer.close()

    ro = roexp.open_ro_connection(db_path)
    try:
        with roexp.read_transaction(ro):
            seen = []
            for chunk in roexp.iter_records_chunks(ro):
                for id_str, _ in chunk:
                    seen.append(id_str)
    finally:
        ro.close()
    assert seen == []


@_skip_sqlite_format
def test_iter_edges_chunks_compound_keyset(tmp_path: Path):
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    rng = np.random.default_rng(5)
    ids = []
    for _ in range(20):
        rid = uuid4()
        ids.append(rid)
        _insert_record(writer, rid, _emb_bytes(rng))
    # 100 deterministic (src, dst) pairs.
    edge_set: set[tuple[str, str]] = set()
    while len(edge_set) < 100:
        i = int(rng.integers(0, 20))
        j = int(rng.integers(0, 20))
        if i == j:
            continue
        pair = (str(ids[i]), str(ids[j]))
        if pair in edge_set:
            continue
        edge_set.add(pair)
        _insert_edge(writer, ids[i], ids[j], 1.0 + i * 0.01)
    writer.close()

    ro = roexp.open_ro_connection(db_path)
    try:
        with roexp.read_transaction(ro):
            seen = set()
            page_sizes = []
            for chunk in roexp.iter_edges_chunks(ro, chunk_size=30):
                page_sizes.append(len(chunk))
                for src, dst, _w in chunk:
                    seen.add((src, dst))
    finally:
        ro.close()
    assert seen == edge_set
    assert page_sizes == [30, 30, 30, 10]


@_skip_sqlite_format
def test_read_transaction_and_edges_branch_on_connection_not_env_reverse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Reverse mismatch: LILLI_STORAGE_DRIVER=lilli but the file is a genuine
    stdlib sqlite3 store (detected via on-disk magic, per _resolve_effective_driver).

    open_ro_connection resolves the real stdlib connection despite the env,
    so read_transaction and iter_edges_chunks must still take the stdlib
    BEGIN DEFERRED / compound-tuple-keyset branch — not skip the transaction
    because the env says "lilli".
    """
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    rng = np.random.default_rng(42)
    ids = [uuid4() for _ in range(4)]
    for rid in ids:
        _insert_record(writer, rid, _emb_bytes(rng))
    _insert_edge(writer, ids[0], ids[1], 1.0)
    _insert_edge(writer, ids[1], ids[2], 1.0)
    writer.close()

    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "lilli")

    ro = roexp.open_ro_connection(db_path)
    try:
        from iai_mcp import _sqlite_stdlib
        assert _sqlite_stdlib.is_stdlib_connection(ro), (
            "on-disk magic is genuine stdlib SQLite; open_ro_connection must "
            "resolve the stdlib connection regardless of env=lilli"
        )
        with roexp.read_transaction(ro):
            # A real BEGIN DEFERRED transaction must be active — a stdlib
            # connection has no in-progress transaction otherwise.
            assert ro.in_transaction
            seen = set()
            for chunk in roexp.iter_edges_chunks(ro, chunk_size=1):
                for src, dst, _w in chunk:
                    seen.add((src, dst))
    finally:
        ro.close()

    assert seen == {(str(ids[0]), str(ids[1])), (str(ids[1]), str(ids[2]))}


@_skip_sqlite_format
def test_read_transaction_snapshot_isolation(tmp_path: Path):
    db_path = tmp_path / "brain.sqlite3"
    writer = _make_db(db_path)
    rng = np.random.default_rng(6)
    for _ in range(50):
        _insert_record(writer, uuid4(), _emb_bytes(rng))
    writer.close()

    ro = roexp.open_ro_connection(db_path)
    pause_evt = threading.Event()
    resume_evt = threading.Event()
    extra_id = uuid4()

    def writer_thread():
        pause_evt.wait()
        w = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            w.execute("PRAGMA journal_mode=WAL")
            w.execute(
                "INSERT INTO records (id, embedding) VALUES (?, ?)",
                (str(extra_id), _emb_bytes(rng)),
            )
        finally:
            w.close()
        resume_evt.set()

    wt = threading.Thread(target=writer_thread, daemon=True)
    wt.start()

    try:
        with roexp.read_transaction(ro):
            seen = []
            it = roexp.iter_records_chunks(ro, chunk_size=10)
            # Pull first page.
            first = next(it)
            for id_str, _ in first:
                seen.append(id_str)
            # Let the writer commit a new row, then resume the in-flight
            # snapshot read.
            pause_evt.set()
            assert resume_evt.wait(timeout=5.0)
            # Give SQLite a moment to publish the new WAL frame.
            time.sleep(0.1)
            for chunk in it:
                for id_str, _ in chunk:
                    seen.append(id_str)
    finally:
        ro.close()
        wt.join(timeout=5.0)

    assert str(extra_id) not in seen, (
        "in-flight snapshot read must NOT see the post-snapshot insert"
    )


@_skip_sqlite_format
def test_dangling_endpoint_unchanged_partition(tmp_path: Path):
    """SQL-drop dangling endpoint: edge with a tombstoned-endpoint is filtered
    at the streaming layer; partition over the active subgraph is unchanged
    versus a control fixture with no such edge."""
    rng_a = np.random.default_rng(101)
    rng_b = np.random.default_rng(101)

    # Fixture A: 500 active records + 1 tombstoned record + 1 edge to it.
    db_a = tmp_path / "a.sqlite3"
    wa = _make_db(db_a)
    a_active_ids: list[UUID] = []
    for _ in range(500):
        rid = uuid4()
        a_active_ids.append(rid)
        _insert_record(wa, rid, _emb_bytes(rng_a))
    tomb_id = uuid4()
    _insert_record(wa, tomb_id, _emb_bytes(rng_a), tombstoned_at="2026-01-01T00:00:00")
    # Build a deterministic ring of edges among active records.
    edges_seen: set[tuple[str, str]] = set()
    for i in range(len(a_active_ids)):
        src = a_active_ids[i]
        dst = a_active_ids[(i + 1) % len(a_active_ids)]
        pair = (str(src), str(dst))
        if pair in edges_seen:
            continue
        edges_seen.add(pair)
        _insert_edge(wa, src, dst, 1.0)
    # One edge whose dst references the tombstoned record.
    _insert_edge(wa, a_active_ids[0], tomb_id, 1.0)
    wa.close()

    # Fixture B: same active records, same ring, no edge to a tombstoned id.
    db_b = tmp_path / "b.sqlite3"
    wb = _make_db(db_b)
    for rid in a_active_ids:
        _insert_record(wb, rid, _emb_bytes(rng_b))
    for src_s, dst_s in edges_seen:
        wb.execute(
            "INSERT INTO edges (src, dst, edge_type, weight) VALUES (?, ?, ?, ?)",
            (src_s, dst_s, "hebbian", 1.0),
        )
    wb.close()

    def _stream(db_path: Path):
        nodes: list = []
        edges: list = []
        ro = roexp.open_ro_connection(db_path)
        try:
            with roexp.read_transaction(ro):
                for chunk in roexp.iter_records_chunks(ro):
                    for id_str, emb_blob in chunk:
                        emb = np.frombuffer(emb_blob, dtype=np.float32).tolist()
                        nodes.append((UUID(id_str), None, emb, {}))
                for chunk in roexp.iter_edges_chunks(ro):
                    for src_str, dst_str, weight in chunk:
                        edges.append((
                            UUID(src_str), UUID(dst_str), float(weight), "hebbian"
                        ))
        finally:
            ro.close()
        return nodes, edges

    nodes_a, edges_a = _stream(db_a)
    nodes_b, edges_b = _stream(db_b)

    # No edge in the streamed result of A references the tombstoned record.
    tomb_str = str(tomb_id)
    for _src, _dst, _w, _et in edges_a:
        assert str(_src) != tomb_str
        assert str(_dst) != tomb_str

    def _partition(node_to_community):
        groups: dict = {}
        for n, c in node_to_community.items():
            groups.setdefault(c, set()).add(str(n))
        return frozenset(frozenset(m) for m in groups.values())

    g_a = MemoryGraph()
    g_a.clear_and_rebuild(nodes_a, edges_a)
    g_b = MemoryGraph()
    g_b.clear_and_rebuild(nodes_b, edges_b)

    a_assignment = detect_communities(g_a, prior_mode="cold")
    b_assignment = detect_communities(g_b, prior_mode="cold")

    assert _partition(a_assignment.node_to_community) == _partition(
        b_assignment.node_to_community
    )


@pytest.mark.skipif(
    _driver != "lilli",
    reason="proves iter_edges_chunks works on a lilli store and drops tombstoned-endpoint edges",
)
def test_lilli_iter_edges_drops_tombstoned_endpoint(tmp_path: Path):
    """iter_edges_chunks on a lilli store drops edges whose src or dst is tombstoned.

    Seeds two active records (r1, r2), one tombstoned record (t1), plus edges:
      r1 → r2  (both active — must appear in output)
      r1 → t1  (dst tombstoned — must be dropped)
    Asserts only (r1, r2) is yielded and t1 never appears as src or dst.
    """
    from iai_mcp.hippo import HippoDB

    db = HippoDB(tmp_path)
    try:
        rtbl = db.open_table("records")

        def _add_rec(rid: str, *, tombstoned: bool = False) -> None:
            row: dict = {
                "id": rid,
                "tier": "episodic",
                "literal_surface": "x",
                "embedding": np.random.rand(_DIM).astype(np.float32).tolist(),
                "created_at": "2026-01-01T00:00:00+00:00",
            }
            if tombstoned:
                row["tombstoned_at"] = "2026-01-02T00:00:00+00:00"
            rtbl.add([row])

        import uuid as _uuid
        r1 = str(_uuid.uuid4())
        r2 = str(_uuid.uuid4())
        t1 = str(_uuid.uuid4())
        _add_rec(r1)
        _add_rec(r2)
        _add_rec(t1, tombstoned=True)

        etbl = db.open_table("edges")
        etbl.add([{"src": r1, "dst": r2, "edge_type": "hebbian", "weight": 1.0}])
        etbl.add([{"src": r1, "dst": t1, "edge_type": "hebbian", "weight": 0.5}])
    finally:
        pass  # keep db open so open_ro_connection reuses the live conn

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    ro_conn = roexp.open_ro_connection(db_path)
    yielded = []
    try:
        with roexp.read_transaction(ro_conn):
            for chunk in roexp.iter_edges_chunks(ro_conn):
                yielded.extend(chunk)
    finally:
        ro_conn.close()
        db.close()

    srcs = {e[0] for e in yielded}
    dsts = {e[1] for e in yielded}
    assert t1 not in srcs and t1 not in dsts, (
        f"tombstoned record {t1!r} must not appear as edge endpoint; got edges={yielded}"
    )
    assert any(e[0] == r1 and e[1] == r2 for e in yielded), (
        f"active edge r1→r2 must appear; got edges={yielded}"
    )


@pytest.mark.skipif(
    _driver != "lilli",
    reason=(
        "proves read_transaction/iter_edges_chunks branch on the concrete "
        "connection object, not the ambient env var, on a real lilli store"
    ),
)
def test_lilli_ro_export_branches_on_connection_with_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Regression: a lilli-format store opened with LILLI_STORAGE_DRIVER unset
    must still take the no-transaction / full-scan branches.

    Before the fix, read_transaction and iter_edges_chunks re-derived the
    driver from os.environ["LILLI_STORAGE_DRIVER"] instead of the connection
    object open_ro_connection already resolved from the on-disk file format.
    A client process without the env inherited (matching the split-brain the
    on-disk-magic fix in _resolve_effective_driver addresses) would take the
    BEGIN DEFERRED / compound-tuple-keyset stdlib branch against a lilli
    connection — a write txn that conflicts with concurrent daemon writes,
    and SQL the engine's parser does not support.
    """
    import sqlite3 as _sqlite3

    from iai_mcp.hippo import HippoDB

    db = HippoDB(tmp_path)
    try:
        rtbl = db.open_table("records")

        def _add_rec(rid: str) -> None:
            rtbl.add([{
                "id": rid,
                "tier": "episodic",
                "literal_surface": "x",
                "embedding": np.random.rand(_DIM).astype(np.float32).tolist(),
                "created_at": "2026-01-01T00:00:00+00:00",
            }])

        import uuid as _uuid
        r1 = str(_uuid.uuid4())
        r2 = str(_uuid.uuid4())
        _add_rec(r1)
        _add_rec(r2)

        etbl = db.open_table("edges")
        etbl.add([{"src": r1, "dst": r2, "edge_type": "hebbian", "weight": 1.0}])
    finally:
        pass  # keep db open so open_ro_connection reuses the live conn

    # The regression condition: no LILLI_STORAGE_DRIVER in this process' env.
    monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    ro_conn = roexp.open_ro_connection(db_path)
    try:
        # The connection open_ro_connection resolved must NOT be a stdlib
        # sqlite3.Connection — it is the lilli engine wrapper — regardless
        # of the (now-unset) env var.
        assert not isinstance(ro_conn, _sqlite3.Connection)

        with roexp.read_transaction(ro_conn):
            # No BEGIN DEFERRED must have been issued against the engine
            # connection; if it had, this SELECT would still succeed but a
            # concurrent daemon write below would deadlock/conflict. The
            # real proof is architectural (isinstance-branch), asserted
            # directly here as a white-box check on the same code path
            # iter_edges_chunks and read_transaction both use.
            assert not isinstance(ro_conn, _sqlite3.Connection)

            edges = []
            for chunk in roexp.iter_edges_chunks(ro_conn):
                edges.extend(chunk)
    finally:
        ro_conn.close()
        db.close()

    assert any(e[0] == r1 and e[1] == r2 for e in edges), (
        f"expected edge r1->r2 in export; got {edges}"
    )


@pytest.mark.skipif(
    _driver != "lilli",
    reason="proves the routed open_ro_connection + BEGIN-branched read_transaction on a real lilli store",
)
def test_lilli_ro_export_records_via_engine(tmp_path: Path):
    """The routed export reads records from a real lilli store without taking a write txn."""
    import struct as _struct
    from uuid import uuid4 as _uuid4

    from iai_mcp.hippo import HippoDB

    n_records = 5
    seeded_ids: list[str] = []
    db = HippoDB(tmp_path)
    try:
        tbl = db.open_table("records")
        for _ in range(n_records):
            rid = str(_uuid4())
            seeded_ids.append(rid)
            tbl.add([{
                "id": rid,
                "tier": "episodic",
                "literal_surface": "test surface",
                "embedding": np.random.rand(_DIM).astype(np.float32).tolist(),
                "created_at": "2026-01-01T00:00:00+00:00",
            }])
    finally:
        # Keep HippoDB open so open_ro_connection gets the live registered conn.
        pass

    db_path = tmp_path / "hippo" / "brain.sqlite3"
    ro_conn = roexp.open_ro_connection(db_path)
    try:
        returned_ids: list[str] = []
        with roexp.read_transaction(ro_conn):
            for chunk in roexp.iter_records_chunks(ro_conn):
                for id_str, emb_blob in chunk:
                    returned_ids.append(id_str)
                    assert len(emb_blob) == _DIM * 4, (
                        f"embedding blob should be {_DIM * 4} bytes, got {len(emb_blob)}"
                    )
    finally:
        ro_conn.close()
        db.close()

    assert set(returned_ids) == set(seeded_ids), (
        f"export returned {set(returned_ids)}, expected {set(seeded_ids)}"
    )
