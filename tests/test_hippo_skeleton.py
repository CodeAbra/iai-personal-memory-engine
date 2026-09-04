from __future__ import annotations

import stat
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
from iai_mcp.hippo import _schema as pa
import pytest

from iai_mcp.hippo import (
    HippoDB,
    HippoTableList,
    _validate_table_name,
)
from iai_mcp.types import EMBED_DIM


_CANONICAL_TABLES = sorted([
    "_hippo_meta",
    "budget_ledger",
    "edges",
    "events",
    "proc_transitions",
    "ratelimit_ledger",
    "record_tags",
    "records",
])

_RECORDS_COLUMNS = sorted([
    "vec_label", "id", "tier", "literal_surface", "aaak_index", "embedding",
    "structure_hv", "community_id", "centrality", "detail_level", "pinned",
    "stability", "difficulty", "last_reviewed", "never_decay", "never_merge",
    "tombstoned_at", "schema_bypass", "labile_until", "provenance_json",
    "created_at", "updated_at", "tags_json", "language", "s5_trust_score",
    "profile_modulation_gain_json", "schema_version", "wing", "room",
    "drawer", "valence",
])


def _edge_row(**overrides) -> dict:
    row = {
        "src": str(uuid4()),
        "dst": str(uuid4()),
        "edge_type": "hebbian",
        "weight": 0.5,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    row.update(overrides)
    return row


def test_hippo_db_opens_on_tmp_path(tmp_path: Path) -> None:
    db = HippoDB(tmp_path)
    try:
        assert (tmp_path / "hippo" / "brain.sqlite3").exists()
    finally:
        db.close()


def test_wal_mode_enabled(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        result = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert result == "wal"


def test_foreign_keys_enabled(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        result = db._conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert result == 1


def test_all_canonical_tables_exist(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        names = sorted(db.table_names())
    assert names == _CANONICAL_TABLES


def test_list_tables_shape(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        result = db.list_tables()
    assert isinstance(result, HippoTableList)
    assert isinstance(result.tables, list)
    for name in _CANONICAL_TABLES:
        assert name in result.tables


def test_list_tables_iterable(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        result = db.list_tables()
    assert sorted(list(result)) == _CANONICAL_TABLES


def test_meta_seeded(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        rows = db._conn.execute("SELECT key, value FROM _hippo_meta").fetchall()
    meta = {r[0]: r[1] for r in rows}
    assert meta.get("schema_version") == "1"
    assert meta.get("embed_dim") == str(EMBED_DIM)


def test_records_schema_columns(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        schema_names = sorted(tbl.schema.names)
    for col in _RECORDS_COLUMNS:
        assert col in schema_names, f"Missing column: {col}"


def test_records_embedding_schema_type(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        embed_field = tbl.schema.field("embedding")
    assert embed_field.type.list_size == EMBED_DIM


def test_edges_primary_key(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        rows = db._conn.execute("PRAGMA table_info(edges)").fetchall()
    pk_cols = {r["name"] for r in rows if r["pk"] > 0}
    assert pk_cols == {"src", "dst", "edge_type"}


def test_count_rows_empty(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        assert tbl.count_rows() == 0


def test_edges_add_and_count_roundtrip(tmp_path: Path) -> None:
    rows = [_edge_row() for _ in range(3)]
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        tbl.add(rows)
        assert tbl.count_rows() == 3


def test_edges_merge_insert_upsert(tmp_path: Path) -> None:
    row = _edge_row(weight=0.5)
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        tbl.add([row])
        assert tbl.count_rows() == 1

        row["weight"] = 0.9
        tbl.merge_insert(["src", "dst", "edge_type"]).when_matched_update_all().execute([row])
        assert tbl.count_rows() == 1

        df = tbl.to_pandas()
        assert abs(float(df["weight"].iloc[0]) - 0.9) < 1e-6


def test_table_update_changes_values(tmp_path: Path) -> None:
    row = _edge_row(weight=0.1)
    src = row["src"]
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        tbl.add([row])
        tbl.update(where=f"src = '{src}'", values={"weight": 0.77})
        df = tbl.to_pandas()
    assert abs(float(df[df["src"] == src]["weight"].iloc[0]) - 0.77) < 1e-6


def test_table_delete_removes_rows(tmp_path: Path) -> None:
    row = _edge_row()
    src = row["src"]
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        tbl.add([row, _edge_row()])
        assert tbl.count_rows() == 2
        tbl.delete(where=f"src = '{src}'")
        assert tbl.count_rows() == 1


def test_search_returns_chainable_query(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        tbl.add([_edge_row(edge_type="episodic")])
        df = tbl.search().where("edge_type = 'episodic'").limit(5).to_pandas()
    assert len(df) == 1


def test_search_with_vector_returns_query(tmp_path: Path) -> None:
    from iai_mcp.hippo import HippoQuery
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        q = tbl.search(vector=vec)
    assert isinstance(q, HippoQuery)


def test_list_versions_stub(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        versions = tbl.list_versions()
    assert len(versions) == 1
    assert versions[0]["version"] == 1
    assert "ts" in versions[0]


def test_optimize_returns_noop_dict(tmp_path: Path) -> None:
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("records")
        result = tbl.optimize()
    assert result == {"compaction": "noop_hippo"}


def test_add_columns_idempotent(tmp_path: Path) -> None:
    new_field = pa.field("test_extra_col", pa.string(), nullable=True)
    with HippoDB(tmp_path) as db:
        tbl = db.open_table("edges")
        tbl.add_columns([new_field])
        tbl.add_columns([new_field])
        names = [r["name"] for r in db._conn.execute("PRAGMA table_info(edges)").fetchall()]
    assert names.count("test_extra_col") == 1


def test_lock_file_created_on_open(tmp_path: Path) -> None:
    lock_path = tmp_path / "hippo" / ".lock"
    with HippoDB(tmp_path):
        assert lock_path.exists()
        mode = stat.S_IMODE(lock_path.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


def test_second_open_same_process_succeeds(tmp_path: Path) -> None:
    db1 = HippoDB(tmp_path)
    try:
        db2 = HippoDB(tmp_path)
        try:
            assert db1._lock_key == db2._lock_key
        finally:
            db2.close()
    finally:
        db1.close()


def test_lock_released_on_close(tmp_path: Path) -> None:
    db1 = HippoDB(tmp_path)
    db1.close()
    db2 = HippoDB(tmp_path)
    db2.close()


def test_lock_released_on_context_manager_exit(tmp_path: Path) -> None:
    with HippoDB(tmp_path):
        pass

    db2 = HippoDB(tmp_path)
    db2.close()


def test_reopen_at_same_path_persists_rows(tmp_path: Path) -> None:
    row = _edge_row()
    with HippoDB(tmp_path) as db:
        db.open_table("edges").add([row])

    with HippoDB(tmp_path) as db:
        count = db.open_table("edges").count_rows()
    assert count == 1


def test_validate_table_name_rejects_sql_injection() -> None:
    bad_names = ["foo; DROP TABLE", "foo bar", "123foo", "foo-bar", "foo.bar"]
    for name in bad_names:
        with pytest.raises(ValueError, match="Invalid table name"):
            _validate_table_name(name)


def test_access_mode_enum_has_two_members(tmp_path: Path) -> None:
    from iai_mcp.hippo import AccessMode
    members = list(AccessMode)
    assert len(members) == 2
    names = {m.name for m in members}
    assert names == {"EXCLUSIVE", "SHARED"}


def test_shared_open_raises_when_exclusive_held_same_process(tmp_path: Path) -> None:
    from iai_mcp.hippo import AccessMode, HippoDB, HippoLockHeldError
    db_ex = HippoDB(tmp_path, access_mode=AccessMode.EXCLUSIVE)
    try:
        with pytest.raises(HippoLockHeldError):
            HippoDB(tmp_path, access_mode=AccessMode.SHARED)
    finally:
        db_ex.close()


def test_two_shared_opens_in_same_process_succeed(tmp_path: Path) -> None:
    from iai_mcp.hippo import AccessMode, HippoDB
    db1 = HippoDB(tmp_path, access_mode=AccessMode.SHARED)
    try:
        db2 = HippoDB(tmp_path, access_mode=AccessMode.SHARED)
        db2.close()
    finally:
        db1.close()


def test_shared_read_only_has_no_hnsw(tmp_path: Path) -> None:
    from iai_mcp.hippo import AccessMode, HippoDB
    db = HippoDB(tmp_path, access_mode=AccessMode.SHARED, read_only=True)
    try:
        assert db._hnsw is None
    finally:
        db.close()


def test_busy_timeout_set_on_exclusive_connection(tmp_path: Path) -> None:
    db = HippoDB(tmp_path)
    try:
        val = db._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert val == 2000, f"Expected busy_timeout=2000, got {val}"
    finally:
        db.close()


def test_memory_store_accepts_access_mode(tmp_path: Path) -> None:
    from iai_mcp.hippo import AccessMode
    from iai_mcp.store import MemoryStore
    store = MemoryStore(tmp_path, access_mode=AccessMode.SHARED)
    try:
        assert store.db._access_mode == AccessMode.SHARED
    finally:
        store.close()


def test_validate_table_name_accepts_valid_identifiers() -> None:
    valid_names = ["records", "my_table", "_internal", "Table123", "a"]
    for name in valid_names:
        result = _validate_table_name(name)
        assert result == name
