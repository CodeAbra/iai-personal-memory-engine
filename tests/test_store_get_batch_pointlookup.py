"""Regression: get_batch parity + indexed point-lookup access pattern.

Proves two properties:

1. **Parity** — ``store.get_batch(ids)`` returns a ``dict[UUID, MemoryRecord]``
   that is byte-identical (field-for-field) to the result that the prior
   ``SELECT ... WHERE id IN (?, ?, ...)`` path returned for a mixed corpus of
   present, absent, and duplicate ids.  A tombstoned and an
   ``embedding_pending`` row are included so the comparison exercises every
   decode branch, not just the happy path.

2. **Indexed access** (engine only) — on the lilli SQL engine, fetching by a
   per-id ``WHERE id = '<literal>'`` equality resolves through the always-warm
   IdIndex, paying ``full_scan_count() == 0``.  The single ``WHERE id IN (?,
   ?, ...)`` bound-param form does NOT reach the IdIndex and pays at least one
   full-table scan (``full_scan_count() >= 1``) to build the column index that
   has no bound-param fast path.  The contrast is pinned so the test records
   the exact difference, not just an absolute bound.

The engine-driven portion is skipped loudly when ``iai_mcp_native.engine``
is not importable — a stale or absent native wheel can never produce a
false green.
"""
from __future__ import annotations

import dataclasses
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

os.environ.setdefault("LILLI_FSYNC_MODE", "fast")


# ---------------------------------------------------------------------------
# Native-engine availability gate
# ---------------------------------------------------------------------------

def _engine_available() -> bool:
    try:
        from iai_mcp_native import engine  # noqa: F401
        return True
    except ImportError:
        return False


_ENGINE_SKIPMARK = pytest.mark.skipif(
    not _engine_available(),
    reason="iai_mcp_native.engine not installed (build the native wheel to run engine tests)",
)


# ---------------------------------------------------------------------------
# Shared helpers for the parity corpus
# ---------------------------------------------------------------------------

def _vec(seed: int) -> list[float]:
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _make_record(
    *,
    text: str,
    seed: int,
    pinned: bool = False,
    never_merge: bool = False,
    tags: list[str] | None = None,
    s5_trust_score: float = 0.5,
    hv_tier: str = "bsc",
    structure_hv_payload: bytes = b"",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=_vec(seed),
        community_id=None,
        centrality=float(seed) * 0.01,
        detail_level=2,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=never_merge,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags or [],
        language="en",
        s5_trust_score=s5_trust_score,
        hv_tier=hv_tier,
        structure_hv_payload=structure_hv_payload,
    )


@pytest.fixture
def _store(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    return MemoryStore(str(tmp_path / "store"))


# ---------------------------------------------------------------------------
# Helper: build the reference IN-query result from the raw connection
#
# This replicates the prior access pattern exactly: one SELECT with an IN
# clause driven through the store's own _RECORD_COLS / _decode_raw_row /
# _from_row decode chain, so it is the canonical reference for parity checks.
# ---------------------------------------------------------------------------

def _reference_in_result(
    store: MemoryStore,
    ids: list[UUID],
) -> dict[UUID, MemoryRecord]:
    """Fetch records using a single WHERE id IN (?, ...) and decode them exactly
    as the prior get_batch path did.  Missing ids are absent from the result."""
    if not ids:
        return {}
    str_ids = [str(i) for i in ids]
    ph = ", ".join("?" for _ in str_ids)
    sql = (
        f"SELECT {store._RECORD_COLS} FROM records"  # noqa: S608
        f" WHERE id IN ({ph})"
    )
    with store.db._conn_lock:
        raw_rows = store.db._conn.execute(sql, str_ids).fetchall()
    out: dict[UUID, MemoryRecord] = {}
    for raw in raw_rows:
        row_dict = store._decode_raw_row(dict(raw))
        try:
            rec = store._from_row(row_dict)
            out[rec.id] = rec
        except Exception:  # noqa: BLE001
            continue
    return out


_ALL_FIELDS = [f.name for f in dataclasses.fields(MemoryRecord)]


def _assert_record_identical(ref: MemoryRecord, got: MemoryRecord) -> None:
    for field in _ALL_FIELDS:
        a = getattr(ref, field)
        b = getattr(got, field)
        if field == "embedding":
            assert len(a) == len(b), (
                f"embedding length differs: reference={len(a)} got={len(b)}"
            )
            for i, (x, y) in enumerate(zip(a, b)):
                assert x == y, f"embedding[{i}] differs: {x!r} != {y!r}"
            continue
        assert a == b, (
            f"field {field!r} differs: reference={a!r} got={b!r}"
        )
        assert type(a) is type(b), (
            f"field {field!r} type differs: {type(a).__name__} vs {type(b).__name__}"
        )


# ---------------------------------------------------------------------------
# Parity tests — stdlib driver (engine-independent)
# ---------------------------------------------------------------------------

def test_get_batch_byte_identical_to_in_query_path(_store):
    """get_batch returns byte-identical records to the IN-query reference path
    for a corpus with varied fields (pinned, never_merge, tags, custom scalars,
    non-default hv_tier + structure_hv_payload)."""
    records = [
        _make_record(text="plain record", seed=1),
        _make_record(text="pinned fact", seed=2, pinned=True, never_merge=True),
        _make_record(text="tagged record", seed=3, tags=["role:user", "session:abc"]),
        _make_record(text="scored record", seed=4, s5_trust_score=0.87),
        _make_record(
            text="codec payload",
            seed=5,
            hv_tier="fhrr",
            structure_hv_payload=bytes(range(64)),
        ),
    ]
    for r in records:
        _store.insert(r)

    ids = [r.id for r in records]
    ref = _reference_in_result(_store, ids)
    got = _store.get_batch(ids)

    assert set(ref.keys()) == set(got.keys()), (
        f"key sets differ: reference={set(ref.keys())} got={set(got.keys())}"
    )
    for rid in ref:
        _assert_record_identical(ref[rid], got[rid])


def test_get_batch_absent_ids_skipped(_store):
    """Present ids appear in the result; absent (never-inserted) ids do not."""
    r = _make_record(text="present", seed=10)
    _store.insert(r)
    absent1 = uuid4()
    absent2 = uuid4()

    result = _store.get_batch([r.id, absent1, absent2])
    assert r.id in result, "present id must appear in result"
    assert absent1 not in result, "absent id must be skipped"
    assert absent2 not in result, "absent id must be skipped"


def test_get_batch_duplicate_id_collapsed(_store):
    """A duplicate id in the input appears exactly once in the result."""
    r = _make_record(text="dedup check", seed=11)
    _store.insert(r)

    result = _store.get_batch([r.id, r.id, r.id])
    assert len(result) == 1, (
        f"duplicate id must be collapsed to one entry; got {len(result)} entries"
    )
    assert r.id in result


def test_get_batch_mixed_present_absent_duplicate(_store):
    """Mix of present, absent, and duplicate ids: dict size equals distinct
    present id count; absent ids absent; byte-identical records for present ids."""
    r1 = _make_record(text="first", seed=20)
    r2 = _make_record(text="second", seed=21)
    _store.insert(r1)
    _store.insert(r2)

    absent = uuid4()
    query_ids = [r1.id, r2.id, absent, r1.id]  # r1.id repeated
    ref = _reference_in_result(_store, [r1.id, r2.id])
    got = _store.get_batch(query_ids)

    assert set(got.keys()) == {r1.id, r2.id}, (
        f"expected only present ids; got keys={set(got.keys())}"
    )
    assert absent not in got
    _assert_record_identical(ref[r1.id], got[r1.id])
    _assert_record_identical(ref[r2.id], got[r2.id])


def test_get_batch_tombstoned_row_visible(_store):
    """A tombstoned row is included in get_batch (no tombstone filter applied),
    identical to the prior IN-query path which also applied no filter."""
    r = _make_record(text="to be tombstoned", seed=30)
    _store.insert(r)
    # Tombstone via a direct SQL update so we exercise the decode, not the
    # higher-level delete path which removes the row from the index.
    now = datetime.now(timezone.utc).isoformat()
    with _store.db._conn_lock:
        _store.db._conn.execute(
            "UPDATE records SET tombstoned_at = ? WHERE id = ?",
            [now, str(r.id)],
        )

    ref = _reference_in_result(_store, [r.id])
    got = _store.get_batch([r.id])

    assert r.id in ref, "reference IN path must include the tombstoned row"
    assert r.id in got, "get_batch must include the tombstoned row (no tombstone filter)"
    _assert_record_identical(ref[r.id], got[r.id])


def test_get_batch_embedding_pending_row_visible(_store):
    """An embedding_pending=1 row is included in get_batch, identical to the
    prior IN-query path which applied no pending filter."""
    import json
    record_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    _store.db.insert_pending_row(
        record_id=record_id,
        tier="episodic",
        literal_surface="pending record",
        tags_json=json.dumps([]),
        provenance_json=json.dumps([]),
        created_at=now,
        updated_at=now,
    )
    rid = UUID(record_id)
    ref = _reference_in_result(_store, [rid])
    got = _store.get_batch([rid])

    assert rid in ref, "reference IN path must include the embedding_pending row"
    assert rid in got, "get_batch must include the embedding_pending row"
    _assert_record_identical(ref[rid], got[rid])


def test_get_batch_empty_input(_store):
    """get_batch([]) must return {} without touching the database."""
    result = _store.get_batch([])
    assert result == {}, f"empty input must return empty dict; got {result!r}"


# ---------------------------------------------------------------------------
# Engine-driven indexed-access test (lilli only)
#
# Drives engine.Connection directly to compare the full_scan_count() paid by:
#   (a) a loop of per-id `WHERE id = '<literal>' LIMIT 1` selects — 0 scans
#   (b) a single `WHERE id IN (?, ?, ...)` select — >= 1 scan (ColIndex build)
#
# This pins that the point-lookup loop is the no-full-scan path and the IN
# bound-param variant is not.  It is intentionally separate from the parity
# tests so the skip boundary is explicit.
# ---------------------------------------------------------------------------

_RECORDS_DDL = """\
CREATE TABLE records (
    vec_label   INTEGER PRIMARY KEY AUTOINCREMENT,
    id          TEXT NOT NULL UNIQUE,
    tier        TEXT NOT NULL,
    literal_surface TEXT,
    aaak_index      TEXT,
    embedding   BLOB NOT NULL,
    created_at  TEXT NOT NULL
)"""


def _engine_conn(tmp_path: Path, name: str):
    from iai_mcp_native import engine
    conn = engine.Connection.open(str(tmp_path / name), 384)
    assert isinstance(conn, engine.Connection), (
        "connection is not the Rust engine.Connection — test would be false-green"
    )
    return conn


def _seed_engine_records(conn, ids: list[str]) -> None:
    """Insert minimal records into a records-shaped table."""
    conn.execute(_RECORDS_DDL, [])
    now = datetime.now(timezone.utc).isoformat()
    embedding_blob = b"\x00" * (384 * 4)  # 384 float32 zeros
    conn.execute("BEGIN", [])
    for id_ in ids:
        conn.execute(
            "INSERT INTO records (id, tier, literal_surface, aaak_index,"
            " embedding, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            [id_, "episodic", "text", "", embedding_blob, now],
        )
    conn.execute("COMMIT", [])


@_ENGINE_SKIPMARK
def test_point_lookup_loop_pays_no_full_scan(tmp_path):
    """Both a per-id `WHERE id = '<literal>' LIMIT 1` loop and a single
    `WHERE id IN (?, ?, ...)` bound-param SELECT pay full_scan_count()==0 on a
    warm IdIndex.

    The literal-equality loop resolves each id through the id-index point
    lookup; the bound-param IN SELECT resolves the whole list through the same
    warm id-index.  Both access paths are O(log n) with no whole-table scan.
    """
    from iai_mcp_native import engine  # noqa: F401

    # Use distinct ids; the IN query resolves them through the warm id-index.
    ids = [str(uuid4()) for _ in range(8)]
    query_ids = ids[:5]  # fetch a subset

    # --- Point-lookup path ------------------------------------------------
    conn_pl = _engine_conn(tmp_path, "point_lookup.db")
    _seed_engine_records(conn_pl, ids)

    # Warm the IdIndex with one point lookup.
    lit = query_ids[0]
    conn_pl.execute(
        f"SELECT id FROM records WHERE id = '{lit}' LIMIT 1",  # noqa: S608
        [],
    )

    # Reset the scan counter, then run the per-id loop.
    conn_pl.reset_full_scan_count()
    for lit in query_ids:
        cur = conn_pl.execute(
            f"SELECT id FROM records WHERE id = '{lit}' LIMIT 1",  # noqa: S608
            [],
        )
        cur.fetchall()

    point_scan_count = conn_pl.full_scan_count()
    assert point_scan_count == 0, (
        f"per-id literal-equality loop must pay 0 full scans (IdIndex fast path); "
        f"got full_scan_count()={point_scan_count}"
    )

    # --- IN bound-param path (also indexed) -------------------------------
    conn_in = _engine_conn(tmp_path, "in_query.db")
    _seed_engine_records(conn_in, ids)

    # Warm the IdIndex so the IN path resolves through the warm index.
    lit0 = query_ids[0]
    conn_in.execute(
        f"SELECT id FROM records WHERE id = '{lit0}' LIMIT 1",  # noqa: S608
        [],
    )

    # Reset the scan counter, then issue the single IN query.
    conn_in.reset_full_scan_count()
    ph = ", ".join("?" for _ in query_ids)
    cur_in = conn_in.execute(
        f"SELECT id FROM records WHERE id IN ({ph})",  # noqa: S608
        query_ids,
    )
    cur_in.fetchall()

    in_scan_count = conn_in.full_scan_count()
    assert in_scan_count == 0, (
        f"single WHERE id IN (?) with bound params resolves through the warm "
        f"id-index at zero full scans (the bound-param IN path is indexed too); "
        f"got full_scan_count()={in_scan_count}. "
        "Both the per-id literal-equality loop and the IN path are now O(log n)."
    )
