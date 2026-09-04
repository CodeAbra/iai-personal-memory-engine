from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _make_rec(
    tier: str = "episodic",
    text: str = "user message",
    seed: int = 0,
    tags: list[str] | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=_random_vec(seed),
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=tags or [],
        language="en",
    )

def _insert_pending(store, seed: int = 0, text: str = "pending turn", tier: str = "episodic") -> str:
    from datetime import datetime, timezone
    record_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.db.insert_pending_row(
        record_id=record_id,
        tier=tier,
        literal_surface=text,
        tags_json=json.dumps([]),
        provenance_json=json.dumps([]),
        created_at=now,
        updated_at=now,
    )
    return record_id

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    yield s

def test_get_batch_exists(store):
    assert hasattr(store, "get_batch"), "MemoryStore.get_batch does not exist"

def test_get_batch_single_id(store):
    r = _make_rec(text="specific user turn", seed=1)
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result, f"id {r.id} not in get_batch result"
    assert result[r.id].literal_surface == r.literal_surface

def test_get_batch_created_at_populated(store):
    r = _make_rec(seed=2)
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result
    assert result[r.id].created_at is not None, "created_at must be populated"
    assert isinstance(result[r.id].created_at, datetime), (
        f"created_at must be datetime, got {type(result[r.id].created_at)}"
    )

def test_get_batch_embedding_decoded(store):
    vec = _random_vec(99)
    r = _make_rec(seed=99)
    r = MemoryRecord(
        id=r.id, tier=r.tier, literal_surface=r.literal_surface,
        aaak_index="", embedding=vec, community_id=None, centrality=0.0,
        detail_level=2, pinned=False, stability=0.0, difficulty=0.0,
        last_reviewed=None, never_decay=False, never_merge=False,
        provenance=[], created_at=r.created_at, updated_at=r.updated_at,
        tags=[], language="en",
    )
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result
    emb = result[r.id].embedding
    assert len(emb) == EMBED_DIM, (
        f"embedding must have {EMBED_DIM} floats, got {len(emb)} elements "
        f"(first element type: {type(emb[0]).__name__!r})"
    )
    assert isinstance(emb[0], float), (
        f"embedding elements must be float, got {type(emb[0]).__name__!r} — "
        "BLOB was not decoded via np.frombuffer"
    )

def test_get_batch_returns_all_present_ids(store):
    """get_batch returns a record for every present id and skips absent ids."""
    records = [_make_rec(seed=100 + i) for i in range(10)]
    for r in records:
        store.insert(r)

    ids = [r.id for r in records]
    result = store.get_batch(ids)
    assert len(result) == 10, f"Expected 10 records, got {len(result)}"
    for r in records:
        assert r.id in result, f"Record {r.id} missing from get_batch result"
    unknown = uuid4()
    result2 = store.get_batch(ids + [unknown])
    assert unknown not in result2, "Unknown id must not appear in get_batch result"

def test_get_batch_indexed_lookup(store):
    """get_batch resolves the whole id set through chunked `id IN (...)`
    statements on ONE borrowed reader connection — the engine serves `id IN`
    through the same id-index probes as a per-id equality, so this is the
    indexed fast path with a handful of round trips instead of one statement
    per id.  The result is still correct for a single present id."""
    import inspect
    src = inspect.getsource(store.get_batch)
    assert "id IN (" in src, (
        "get_batch must resolve ids through the engine's indexed `id IN` "
        "fast path"
    )
    assert src.count("ro_conn") == 1, (
        "get_batch must borrow ONE reader connection for the whole batch, "
        "never one per id"
    )

    r = _make_rec(seed=200)
    store.insert(r)
    result = store.get_batch([r.id])
    assert r.id in result, f"Record {r.id} must be returned by get_batch"
    assert result[r.id].literal_surface == r.literal_surface

def test_get_batch_unknown_ids_absent(store):
    r = _make_rec(seed=300)
    store.insert(r)
    unknown = uuid4()
    result = store.get_batch([r.id, unknown])
    assert r.id in result
    assert unknown not in result, "Unknown id must not appear in result"

def test_get_batch_empty_ids(store):
    result = store.get_batch([])
    assert result == {}

def test_recent_pending_markers_exists(store):
    assert hasattr(store, "recent_pending_markers"), (
        "MemoryStore.recent_pending_markers does not exist"
    )

def test_recent_pending_markers_pending_record_surfaces(store):
    record_id = _insert_pending(store, seed=400, text="pending user turn")
    result = store.recent_pending_markers(n=10)
    result_id_strs = {str(rec.id) for rec in result}
    assert record_id in result_id_strs, (
        "A pending record (embedding_pending=1) must appear in recent_pending_markers"
    )

def test_recent_pending_markers_role_user_surfaces(store):
    r = _make_rec(tier="episodic", tags=["role:user"], seed=500)
    store.insert(r)
    result = store.recent_pending_markers(n=10)
    result_ids = {rec.id for rec in result}
    assert r.id in result_ids, (
        "A role:user episodic record must appear in recent_pending_markers"
    )

def test_recent_pending_markers_no_all_records(store, monkeypatch):
    r = _make_rec(tier="episodic", tags=["role:user"], seed=600)
    store.insert(r)

    def fail_all_records():
        raise AssertionError("recent_pending_markers must not call all_records()")

    monkeypatch.setattr(store, "all_records", fail_all_records)
    store.recent_pending_markers(n=10)

def test_recent_pending_markers_role_not_starved(store):
    n = 10
    user_turn = _make_rec(tier="episodic", tags=["role:user"], seed=700)
    store.insert(user_turn)

    for i in range(n + 1):
        ambient = _make_rec(tier="episodic", tags=["role:system"], seed=701 + i)
        store.insert(ambient)

    result = store.recent_pending_markers(n=n)
    result_ids = {rec.id for rec in result}
    assert user_turn.id in result_ids, (
        f"role:user turn must appear in recent_pending_markers(n={n}) even after "
        f"{n+1} ambient writes (filter must be in SQL, not post-LIMIT Python)"
    )

@pytest.fixture
def legacy_store(tmp_path, monkeypatch):
    """A store opened against the legacy driver explicitly, for the query-planner
    assertion below (EXPLAIN QUERY PLAN has no native-engine analog)."""
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", "stdlib")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    s = MemoryStore(str(tmp_path / "store"))
    yield s

def test_recent_pending_markers_explain_search_using_index(legacy_store):
    """Asserts the SQLite query planner chose idx_records_pending via EXPLAIN
    QUERY PLAN; the native engine has no cost-based planner or EXPLAIN
    surface, so this test runs against the legacy driver explicitly. The
    actual pending-marker query correctness is proven on both drivers by the
    sibling tests in this file."""
    store = legacy_store
    for i in range(5):
        store.insert(_make_rec(seed=800 + i))
    for i in range(3):
        store.insert(_make_rec(tier="episodic", tags=["role:user"], seed=810 + i))
    for i in range(2):
        _insert_pending(store, seed=820 + i, text=f"pending turn {i}")

    db = store.db

    _EXPLAIN_READ_A = (
        "EXPLAIN QUERY PLAN"
        " SELECT id, tier, literal_surface, aaak_index, embedding,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending"
        " FROM records WHERE embedding_pending = 1"
        " ORDER BY rowid DESC LIMIT ?"
    )
    _EXPLAIN_READ_B = (
        "EXPLAIN QUERY PLAN"
        " SELECT id, tier, literal_surface, aaak_index, embedding,"
        " community_id, centrality, detail_level, pinned,"
        " stability, difficulty, last_reviewed, never_decay, never_merge,"
        " provenance_json, created_at, updated_at, tags_json, language,"
        " s5_trust_score, profile_modulation_gain_json, schema_version,"
        " hv_tier, structure_hv_payload,"
        " COALESCE(embedding_pending, 0) AS embedding_pending"
        " FROM records WHERE tier='episodic' AND tags_json LIKE ?"
        " ORDER BY rowid DESC LIMIT ?"
    )

    with db._conn_lock:
        plan_a = db._conn.execute(_EXPLAIN_READ_A, (10,)).fetchall()
        plan_b = db._conn.execute(_EXPLAIN_READ_B, ('%"role:user"%', 40)).fetchall()

    plan_a_lines = [" ".join(str(v) for v in row) for row in plan_a]
    plan_b_lines = [" ".join(str(v) for v in row) for row in plan_b]

    has_index_a = any("USING INDEX" in line.upper() for line in plan_a_lines)
    has_scan_a = any("SCAN RECORDS" in line.upper() for line in plan_a_lines)
    assert has_index_a, (
        f"READ A (pending) must SEARCH USING INDEX (idx_records_pending). "
        f"EXPLAIN plan: {plan_a_lines}"
    )
    assert not has_scan_a, (
        f"READ A must not do a full SCAN of records. "
        f"EXPLAIN plan: {plan_a_lines}"
    )

    has_scan_b = any("SCAN RECORDS" in line.upper() for line in plan_b_lines)
    assert not has_scan_b, (
        f"READ B must not do a full SCAN of records (must be index-backed on tier). "
        f"EXPLAIN plan: {plan_b_lines}"
    )

def test_recent_pending_markers_large_pending_backlog_bounded(store):
    from iai_mcp.store import MemoryStore
    pending_sql = MemoryStore._PENDING_READ_SQL
    assert "LIMIT ?" in pending_sql, (
        f"_PENDING_READ_SQL must contain 'LIMIT ?' for CC2-H4 bounding: {pending_sql!r}"
    )

    n = 10
    from datetime import datetime, timezone
    for i in range(25):
        record_id = str(uuid4())
        now = datetime.now(timezone.utc).isoformat()
        store.db.insert_pending_row(
            record_id=record_id,
            tier="episodic",
            literal_surface=f"pending turn {i}",
            tags_json=json.dumps([]),
            provenance_json=json.dumps([]),
            created_at=now,
            updated_at=now,
        )

    result = store.recent_pending_markers(n=n)
    assert len(result) <= n, (
        f"recent_pending_markers(n={n}) must return at most {n} records, "
        f"got {len(result)} (SQL LIMIT not being honoured)"
    )

def test_idx_records_pending_exists(store):
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_records_pending'"
        ).fetchall()
    assert rows, (
        "idx_records_pending partial index must exist after MemoryStore init"
    )

def test_recent_pending_markers_dedup(store):
    from datetime import datetime, timezone
    record_id = str(uuid4())
    now = datetime.now(timezone.utc).isoformat()
    store.db.insert_pending_row(
        record_id=record_id,
        tier="episodic",
        literal_surface="pending role:user turn",
        tags_json=json.dumps(["role:user"]),
        provenance_json=json.dumps([]),
        created_at=now,
        updated_at=now,
    )
    result = store.recent_pending_markers(n=20)
    result_id_strs = [str(rec.id) for rec in result]
    count = result_id_strs.count(record_id)
    assert count == 1, (
        f"Record that is both pending and role:user appeared {count} times (expected 1)"
    )
