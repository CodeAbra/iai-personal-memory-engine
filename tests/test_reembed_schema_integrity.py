"""The reembed staging table must keep the canonical records constraints.

A staging table derived from the pyarrow schema loses
``vec_label INTEGER PRIMARY KEY AUTOINCREMENT`` and ``id UNIQUE`` (arrow
carries names and types, never SQL constraints). After the swap, inserts
leave ``vec_label`` NULL — the rowid alias is gone — and the next label-map
load crashes the daemon into a launchd restart loop. These tests pin the
three defenses: canonical staging DDL, the boot-time NULL backfill, and the
fail-loud label-map guard.
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest


def _make_record(store, text: str):
    from iai_mcp.types import MemoryRecord

    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * store.embed_dim,
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
        tags=[],
        language="en",
    )


def _forge_constraintless_records(store) -> None:
    """Recreate `records` the way the buggy migration did: CREATE TABLE AS
    SELECT drops every constraint (no PRIMARY KEY alias), then NULL out
    vec_label like a post-swap insert would have left it."""
    with store.db._conn_lock:
        conn = store.db._conn
        conn.execute("CREATE TABLE records_damaged AS SELECT * FROM records")
        conn.execute("DROP TABLE records")
        conn.execute("ALTER TABLE records_damaged RENAME TO records")
        conn.execute("UPDATE records SET vec_label = NULL")


def _staging_ddl(db) -> str:
    from iai_mcp.migrate import STAGING_TABLE

    with db._conn_lock:
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (STAGING_TABLE,),
        ).fetchone()
    assert row is not None, "staging table missing"
    return row["sql"] if not isinstance(row, tuple) else row[0]


def test_canonical_staging_keeps_constraints(tmp_path):
    from iai_mcp.migrate._reembed import _create_canonical_staging
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    try:
        _create_canonical_staging(store.db)
        ddl = _staging_ddl(store.db)
        assert "PRIMARY KEY AUTOINCREMENT" in ddl, (
            "staging lost the vec_label rowid alias — post-swap inserts "
            f"would leave vec_label NULL. DDL: {ddl[:200]}"
        )
        assert "UNIQUE" in ddl, f"staging lost the id UNIQUE constraint: {ddl[:200]}"
    finally:
        store.close()


def test_canonical_staging_covers_every_live_column(tmp_path):
    from iai_mcp.migrate import STAGING_TABLE
    from iai_mcp.migrate._reembed import _create_canonical_staging
    from iai_mcp.store import MemoryStore, RECORDS_TABLE

    store = MemoryStore(path=tmp_path / "store")
    try:
        _create_canonical_staging(store.db)
        live_cols = {f.name for f in store.db.open_table(RECORDS_TABLE).schema}
        staging_cols = {f.name for f in store.db.open_table(STAGING_TABLE).schema}
        missing = live_cols - staging_cols
        assert not missing, (
            f"staging is missing live records columns {sorted(missing)} — "
            "the staging insert would fail"
        )
    finally:
        store.close()


def test_boot_heal_backfills_null_vec_label(tmp_path):
    """Simulate the damaged post-migration store: a records table without the
    rowid alias gets rows with NULL vec_label; reopening the store must
    backfill them from the rowid instead of crash-looping."""
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    root = store.root
    try:
        rec = _make_record(store, "heal fixture row")
        store.insert(rec)
        _forge_constraintless_records(store)
        with store.db._conn_lock:
            row = store.db._conn.execute(
                "SELECT vec_label FROM records WHERE id = ?", (str(rec.id),)
            ).fetchone()
        assert row["vec_label"] is None
    finally:
        store.close()

    reopened = MemoryStore(path=root)
    try:
        with reopened.db._conn_lock:
            row = reopened.db._conn.execute(
                "SELECT vec_label FROM records WHERE id = ?", (str(rec.id),)
            ).fetchone()
        assert row["vec_label"] is not None, "boot heal did not backfill"
        # The label map must build cleanly on the healed store.
        label_map = reopened.db._load_label_map_from_sqlite()
        assert str(rec.id) in label_map
    finally:
        reopened.close()


def test_label_map_raises_loud_on_null_vec_label(tmp_path):
    from iai_mcp.hippo import HippoIntegrityError
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "store")
    try:
        rec = _make_record(store, "guard fixture row")
        store.insert(rec)
        _forge_constraintless_records(store)
        with pytest.raises(HippoIntegrityError, match="NULL vec_label"):
            store.db._load_label_map_from_sqlite()
    finally:
        # close() walks the same label-map path on the still-damaged table;
        # its raise is the guard working, not a test failure.
        import contextlib

        with contextlib.suppress(Exception):
            store.close()


def test_doctor_accepts_transitioning_state(monkeypatch):
    """The reconciler writes canonical DROWSY as legacy TRANSITIONING; a
    healthy parked daemon must not FAIL the state-file check."""
    import iai_mcp.doctor._lifecycle_checks as lc

    monkeypatch.setattr(
        "iai_mcp.daemon_state.load_state",
        lambda: {"fsm_state": "TRANSITIONING"},
    )
    result = lc.check_e_state_file_valid()
    assert result.passed, f"TRANSITIONING rejected: {result}"
