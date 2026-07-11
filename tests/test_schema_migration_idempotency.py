
from __future__ import annotations

import pytest

@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch: pytest.MonkeyPatch):
    import keyring as _keyring

    fake: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake.get((s, u)))
    monkeypatch.setattr(
        _keyring, "set_password", lambda s, u, p: fake.__setitem__((s, u), p)
    )
    monkeypatch.setattr(
        _keyring, "delete_password", lambda s, u: fake.pop((s, u), None)
    )
    yield fake

def test_schema_migration_already_idempotent(tmp_path):
    from pathlib import Path  # noqa: F401 (imported for clarity; tmp_path is already a Path)

    from iai_mcp.store import RECORDS_TABLE, MemoryStore

    store1 = MemoryStore(path=tmp_path)
    tbl1 = store1.db.open_table(RECORDS_TABLE)
    cols_after_first_open = set(tbl1.schema.names)

    store2 = MemoryStore(path=tmp_path)
    tbl2 = store2.db.open_table(RECORDS_TABLE)
    cols_after_second_open = set(tbl2.schema.names)

    assert cols_after_first_open == cols_after_second_open, (
        f"Column sets diverged between first and second open.\n"
        f"  First open : {sorted(cols_after_first_open)}\n"
        f"  Second open: {sorted(cols_after_second_open)}\n"
        f"  Added      : {sorted(cols_after_second_open - cols_after_first_open)}\n"
        f"  Removed    : {sorted(cols_after_first_open - cols_after_second_open)}"
    )

    migration_columns = (
        "tombstoned_at",
        "schema_bypass",
        "labile_until",
        "wing",
        "room",
        "drawer",
    )
    for col in migration_columns:
        assert col in cols_after_second_open, (
            f"Expected column {col!r} missing after second open"
        )


def _try_role_migrator(store):
    """Run the role migrator if it exists; skip the caller if not yet present."""
    try:
        from iai_mcp.migrate import migrate_role_column  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pytest.skip(
            "migrate_role_column not yet present in iai_mcp.migrate — "
            "this case turns GREEN after the role column implementation ships the migrator"
        )
    return migrate_role_column(store)


def test_role_column_migration_idempotency_and_index_presence(tmp_path):
    """Re-opening the store and re-running the role migrator is a no-op.

    After the role column implementation:
    - The ``role`` column must be present exactly once in PRAGMA table_info.
    - The ``idx_records_role`` index must exist.
    - A second migrator run reports zero updated rows (idempotent).

    On HEAD this test skips (the role column and migrator do not exist yet).
    """
    from iai_mcp.store import MemoryStore

    # First open: column and index are created by _ensure_tables / _reconcile_columns.
    store1 = MemoryStore(path=tmp_path)

    with store1.db._conn_lock:
        pragma_rows = store1.db._conn.execute(
            "PRAGMA table_info(records)"
        ).fetchall()
    col_names = [r["name"] for r in pragma_rows]
    assert col_names.count("role") == 1, (
        f"expected exactly one 'role' column; got {col_names.count('role')} "
        f"(all columns: {col_names})"
    )

    # Probe the catalog directly via sqlite_master — both storage backends
    # populate it, whereas ``PRAGMA index_list`` is not implemented by every
    # backend. The index is created by the records-table DDL on open.
    with store1.db._conn_lock:
        indexes = store1.db._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = 'records'"
        ).fetchall()
    index_names = [
        (r["name"] if hasattr(r, "keys") else r[0]) for r in indexes
    ]
    assert "idx_records_role" in index_names, (
        f"idx_records_role not found in index list: {index_names}"
    )

    # First migrator run (backfill on empty store = no-op or 0 updated).
    first = _try_role_migrator(store1)

    # Second open: schema is already complete; column count stays the same.
    store2 = MemoryStore(path=tmp_path)
    with store2.db._conn_lock:
        pragma_rows2 = store2.db._conn.execute(
            "PRAGMA table_info(records)"
        ).fetchall()
    col_names2 = [r["name"] for r in pragma_rows2]
    assert col_names2.count("role") == 1, (
        f"role column duplicated after second open: count={col_names2.count('role')}"
    )

    # Second migrator run: must be a no-op (all rows already backfilled or none exist).
    try:
        from iai_mcp.migrate import migrate_role_column  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        pytest.skip("migrate_role_column not yet present")
    second = migrate_role_column(store2)
    assert second.get("updated", 0) == 0, (
        f"second migrator run updated rows unexpectedly: {second!r}"
    )


def test_fresh_store_open_has_single_role_column_after_boot_migration(tmp_path):
    """A fresh store open yields exactly one ``role`` column on every backend.

    ``MemoryStore.__init__`` runs ``_reconcile_columns`` (the canonical add-path,
    PRAGMA-guarded) and then ``_run_boot_migrations`` (which invokes the role
    migrator's belt-and-suspenders ALTER). The migrator must NOT append a second
    ``role`` column on a backend that applies ``ALTER ADD COLUMN`` without
    diffing the live schema — a duplicate column desyncs the row codec
    (``to_batches`` then raises on mismatched array widths).

    Re-running the migrator explicitly must also stay at one column.
    """
    from iai_mcp.store import MemoryStore
    from iai_mcp.migrate import migrate_role_column

    store = MemoryStore(path=tmp_path)

    def _role_count() -> int:
        with store.db._conn_lock:
            rows = store.db._conn.execute("PRAGMA table_info(records)").fetchall()
        return [r["name"] for r in rows].count("role")

    assert _role_count() == 1, (
        f"fresh store open produced {_role_count()} 'role' columns; "
        "the belt-and-suspenders ALTER must skip when the column already exists"
    )

    # An explicit second migrator run must not duplicate the column either.
    migrate_role_column(store)
    assert _role_count() == 1, (
        f"explicit migrator re-run produced {_role_count()} 'role' columns"
    )

    # The row codec must round-trip: insert + iter must not raise on column-width
    # mismatch (the symptom a duplicate column produces downstream).
    from iai_mcp.types import EMBED_DIM, MemoryRecord, SCHEMA_VERSION_CURRENT
    from datetime import datetime, timezone
    from uuid import uuid4

    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="codec round-trip probe",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": "x", "cue": "y", "session_id": "z"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["capture", "role:user"],
        language="en",
        schema_version=SCHEMA_VERSION_CURRENT,
        structure_hv=b"",
    )
    store.insert(rec)
    recovered = list(store.iter_records())
    assert any(str(r.id) == str(rec.id) for r in recovered), (
        "inserted record did not round-trip through iter_records"
    )


def test_store_open_triggers_role_backfill_for_legacy_rows(tmp_path):
    """Opening a store with NULL-role rows auto-backfills the role column.

    This test proves that ``MemoryStore.__init__`` triggers the role backfill
    migrator (via ``_run_boot_migrations``) so that rows written before the
    role write-path was active are automatically populated on the next open.
    A live prod store opened after deployment will therefore have its
    ``role:user`` rows backfilled without any manual migration step.
    """
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM, MemoryRecord, SCHEMA_VERSION_CURRENT
    from datetime import datetime, timezone
    from uuid import uuid4

    # First open: create the schema and insert a role:user record.
    store1 = MemoryStore(path=tmp_path)
    rec = MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="a user message",
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[{"ts": "x", "cue": "y", "session_id": "z"}],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=["capture", "role:user"],
        language="en",
        schema_version=SCHEMA_VERSION_CURRENT,
        structure_hv=b"",
    )
    store1.insert(rec)

    # Force role to NULL for this row AND clear the one-time backfill stamp —
    # together these reproduce a genuine pre-migrator store. The stamp is set on
    # the first boot migration (the empty-store open above); a real legacy store
    # written before the migrator existed never carries the stamp, so the next
    # open re-runs the (idempotent) backfill rather than short-circuiting on it.
    from iai_mcp.migrate._role_column import _BACKFILL_MARKER_KEY

    with store1.db._conn_lock:
        store1.db._conn.execute("UPDATE records SET role = NULL")
        store1.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = ?", (_BACKFILL_MARKER_KEY,)
        )
        store1.db._conn.commit()

    # Verify it is NULL before re-open.
    with store1.db._conn_lock:
        row = store1.db._conn.execute(
            "SELECT role FROM records WHERE id = ?", (str(rec.id),)
        ).fetchone()
    assert row is not None and row["role"] is None, (
        "precondition: role must be NULL before re-open"
    )

    # Re-open: _run_boot_migrations must backfill role from tags_json.
    store2 = MemoryStore(path=tmp_path)

    with store2.db._conn_lock:
        row2 = store2.db._conn.execute(
            "SELECT role FROM records WHERE id = ?", (str(rec.id),)
        ).fetchone()
    assert row2 is not None, f"record {rec.id} missing after re-open"
    assert row2["role"] == "user", (
        f"expected role='user' after re-open triggered boot migration; "
        f"got role={row2['role']!r}"
    )
