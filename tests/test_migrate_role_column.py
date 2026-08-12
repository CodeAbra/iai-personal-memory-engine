"""Tests for the role-column backfill migrator.

Verifies that the migrator that backfills the ``role`` column from existing
``tags_json`` data is correct, idempotent, lossless, and supports dry-run.

Key behaviors under test:

- **Backfill correctness**: a row whose tags carry ``role:user`` / ``role:assistant``
  / a non-standard role / no role gets its ``role`` column populated correctly
  (or left NULL when no role tag exists).
- **Idempotent**: re-running the migrator on an already-backfilled store
  reports zero rows updated and leaves the data unchanged.
- **Lossless**: ``literal_surface`` bytes are byte-for-byte identical before
  and after the migration — the migrator never reads or writes encrypted columns.
- **dry_run**: ``dry_run=True`` reports the would-update count without mutating
  any row.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest


def _migrator():
    """Return the migrate_role_column callable."""
    from iai_mcp.migrate import migrate_role_column  # type: ignore[attr-defined]
    return migrate_role_column


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolated_keyring(monkeypatch):
    import keyring as _keyring

    fake_store: dict[tuple[str, str], str] = {}
    monkeypatch.setattr(_keyring, "get_password", lambda s, u: fake_store.get((s, u)))
    monkeypatch.setattr(_keyring, "set_password", lambda s, u, p: fake_store.__setitem__((s, u), p))
    monkeypatch.setattr(_keyring, "delete_password", lambda s, u: fake_store.pop((s, u), None))
    yield fake_store


def _make_record(
    text: str = "hello",
    tags: "list[str] | None" = None,
    tier: str = "episodic",
):
    from iai_mcp.types import EMBED_DIM, MemoryRecord, SCHEMA_VERSION_CURRENT

    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
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
        tags=tags or [],
        language="en",
        schema_version=SCHEMA_VERSION_CURRENT,
        structure_hv=b"",
    )


def _seed_store_with_roles(tmp_path, monkeypatch):
    """Open a fresh store and insert records covering every role tag case.

    After insertion, the role column is reset to NULL for all rows so the
    fixture represents a legacy store (written before the role write-path was
    active). The migrator under test must derive and populate the role values.
    """
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.store import MemoryStore

    store = MemoryStore()

    rec_user = _make_record(text="user turn text", tags=["capture", "role:user"])
    rec_asst = _make_record(text="assistant turn text", tags=["capture", "role:assistant"])
    rec_tool = _make_record(text="tool output text", tags=["capture", "role:tool"])
    rec_none = _make_record(text="no role tag text", tags=["capture"])

    store.insert(rec_user)
    store.insert(rec_asst)
    store.insert(rec_tool)
    store.insert(rec_none)

    # Simulate a pre-write-path store: reset the role column to NULL AND clear
    # the one-time backfill stamp the boot migrator wrote on the (empty) open.
    # The migrator must re-derive role from tags_json for all such rows.
    with store.db._conn_lock:
        store.db._conn.execute("UPDATE records SET role = NULL")
        store.db._conn.execute(
            "DELETE FROM _hippo_meta WHERE key = 'role_backfill_done'"
        )
        store.db._conn.commit()

    return store, {
        "user": rec_user,
        "assistant": rec_asst,
        "tool": rec_tool,
        "none": rec_none,
    }


# ---------------------------------------------------------------------------
# Helper: derive role value from tags (mirrors the production derive logic)
# ---------------------------------------------------------------------------


def _expected_role(tags: "list[str]") -> "str | None":
    for t in (tags or []):
        if t.startswith("role:"):
            return t[len("role:"):]
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_migrate_role_column_symbol_will_exist():
    """The migrator symbol is importable from iai_mcp.migrate."""
    assert callable(_migrator())


def test_backfill_sets_correct_role_for_each_tag_variant(tmp_path, monkeypatch):
    """The migrator sets ``role`` to the suffix of the ``role:`` tag for every variant.

    Rows with ``role:user`` / ``role:assistant`` / ``role:tool`` tags get the
    correct string value; a row with no ``role:`` tag gets ``role=NULL``.
    """
    migrate_role_column = _migrator()
    store, records = _seed_store_with_roles(tmp_path, monkeypatch)

    # The migrator backfills role from tags for rows where role IS NULL.
    # Three of the four seeded records carry a role: tag (user/assistant/tool).
    # The fourth (no role tag) is left with role=NULL — not counted as updated.
    result = migrate_role_column(store)
    assert isinstance(result, dict)
    assert result.get("updated", 0) == 3, (
        f"expected 3 rows updated (role:user, role:assistant, role:tool); "
        f"got {result.get('updated')!r}"
    )

    # Verify each record's role column matches its tags.
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tags_json, role FROM records ORDER BY id"
        ).fetchall()

    id_to_row = {r["id"]: r for r in rows}
    for label, rec in records.items():
        rid = str(rec.id)
        assert rid in id_to_row, f"record {rid!r} ({label}) missing from store"
        db_role = id_to_row[rid]["role"]
        expected = _expected_role(rec.tags)
        assert db_role == expected, (
            f"role mismatch for {label!r} record {rid!r}: "
            f"got {db_role!r}, expected {expected!r}"
        )


def test_migrator_is_idempotent(tmp_path, monkeypatch):
    """Re-running the migrator on an already-backfilled store updates zero rows.

    The first run backfills the 3 role-tagged rows and writes the persisted
    completion stamp. The second run short-circuits on the stamp (a single keyed
    meta lookup, no row scan) and reports zero work — the data is unchanged.
    """
    migrate_role_column = _migrator()
    store, records = _seed_store_with_roles(tmp_path, monkeypatch)

    first = migrate_role_column(store)
    assert first.get("updated", 0) == 3, (
        f"first run must update the 3 rows with role tags; got {first.get('updated')!r}"
    )

    second = migrate_role_column(store)
    assert second.get("updated", 0) == 0, (
        f"second migrator run must update 0 rows (already backfilled); "
        f"got updated={second.get('updated')!r}"
    )
    # The second run takes the persisted-stamp fast path: no row scan, so it
    # reports no work rather than re-counting the already-set rows.
    assert second.get("note") == "already migrated", (
        f"second run must short-circuit on the persisted stamp; "
        f"got note={second.get('note')!r}"
    )

    # Data integrity unchanged: the 3 role-tagged rows still carry the correct
    # derived role after the idempotent second run.
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, tags_json, role FROM records ORDER BY id"
        ).fetchall()
    id_to_row = {r["id"]: r for r in rows}
    for label, rec in records.items():
        db_role = id_to_row[str(rec.id)]["role"]
        assert db_role == _expected_role(rec.tags), (
            f"role drifted for {label!r} after idempotent run: got {db_role!r}"
        )


def test_migrator_preserves_literal_surface_bytes(tmp_path, monkeypatch):
    """The migrator never reads or writes ``literal_surface`` — bytes are unchanged.

    This is the lossless invariant: ``role`` is derived purely from ``tags_json``
    (plaintext metadata). The encrypted verbatim content is untouched.
    """
    migrate_role_column = _migrator()
    store, records = _seed_store_with_roles(tmp_path, monkeypatch)

    # Snapshot the literal_surface for every record before migration.
    pre = {str(rec.id): rec.literal_surface for rec in records.values()}

    migrate_role_column(store)

    # Verify byte-for-byte equality after migration.
    for rid, pre_text in pre.items():
        from uuid import UUID
        fetched = store.get(UUID(rid))
        assert fetched is not None, f"record {rid!r} missing after migration"
        assert fetched.literal_surface == pre_text, (
            f"literal_surface changed for {rid!r}: "
            f"before={pre_text!r}, after={fetched.literal_surface!r}"
        )


def test_migrator_dry_run_reports_count_without_mutating(tmp_path, monkeypatch):
    """``dry_run=True`` reports the would-update count without mutating any row.

    After a dry-run pass, all ``role`` columns must still be NULL (no update
    was committed).
    """
    migrate_role_column = _migrator()
    store, _ = _seed_store_with_roles(tmp_path, monkeypatch)

    result = migrate_role_column(store, dry_run=True)
    assert isinstance(result, dict)
    would_update = result.get("updated", result.get("would_update", 0))
    assert would_update == 3, (
        f"dry_run should report 3 would-update rows (role-tagged records); "
        f"got {would_update!r}"
    )

    # No row should have a non-NULL role after a dry run.
    with store.db._conn_lock:
        rows = store.db._conn.execute(
            "SELECT id, role FROM records"
        ).fetchall()
    for row in rows:
        assert row["role"] is None, (
            f"dry_run mutated row {row['id']!r}: role={row['role']!r}"
        )


# ---------------------------------------------------------------------------
# Boot-cost regression: an already-migrated store must NOT scan ``records``.
# ---------------------------------------------------------------------------


def _lilli_scan_count_conn(store):
    """Return the engine connection if it exposes the scan-count probe, else None.

    The probe lives only on the lilli engine connection. On the stdlib sqlite3
    driver it is absent, so callers skip the scan-count assertion there.
    """
    conn = store.db._conn
    if hasattr(conn, "full_scan_count") and hasattr(conn, "reset_full_scan_count"):
        return conn
    return None


def test_boot_migrator_no_full_scan_on_migrated_store(tmp_path, monkeypatch):
    """An already-migrated store open pays ZERO full scans through the migrator.

    The persisted ``_hippo_meta`` stamp gates the migrator: after the one-time
    backfill, every subsequent invocation resolves on a single keyed meta lookup
    (a point read, not a scan of ``records``). This is the boot-cost regression
    the gate fixes — the prior implementation scanned ``records`` twice on every
    open. Asserted on the lilli engine, which exposes the scan-count probe.
    """
    migrate_role_column = _migrator()
    store, _ = _seed_store_with_roles(tmp_path, monkeypatch)

    conn = _lilli_scan_count_conn(store)
    if conn is None:
        pytest.skip("scan-count probe is lilli-engine only (stdlib driver lacks it)")

    # First run: backfills the legacy rows and writes the completion stamp.
    first = migrate_role_column(store)
    assert first.get("updated", 0) == 3, (
        f"first run must backfill the 3 role-tagged rows; got {first.get('updated')!r}"
    )

    # A second invocation simulates a post-migration boot: the stamp is set, so
    # the migrator must short-circuit WITHOUT touching ``records``.
    conn.reset_full_scan_count()
    second = migrate_role_column(store)
    assert second.get("updated", 0) == 0, (
        f"already-migrated store must update 0 rows; got {second.get('updated')!r}"
    )
    assert conn.full_scan_count() == 0, (
        f"boot migrator full-scanned an already-migrated store: "
        f"{conn.full_scan_count()} scans (the persisted stamp must gate the scan)"
    )


def test_backfill_stamp_persists_across_reopen(tmp_path, monkeypatch):
    """The completion stamp survives a close + re-open — no re-scan on reboot.

    Re-opening the store (a fresh boot) reads the persisted stamp, so the boot
    migration that ``MemoryStore.__init__`` runs takes the fast path and a
    follow-up migrator call scans ``records`` zero times.
    """
    migrate_role_column = _migrator()
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    from iai_mcp.store import MemoryStore

    store, _ = _seed_store_with_roles(tmp_path, monkeypatch)
    migrate_role_column(store)  # backfill + stamp
    store.close()

    # Fresh boot on the already-migrated store: __init__ runs the boot migration,
    # which must short-circuit on the persisted stamp.
    store2 = MemoryStore()
    try:
        # The stamp is present after the re-open.
        with store2.db._conn_lock:
            stamp = store2.db._conn.execute(
                "SELECT value FROM _hippo_meta WHERE key = 'role_backfill_done'"
            ).fetchone()
        assert stamp is not None, "completion stamp did not persist across re-open"

        conn = _lilli_scan_count_conn(store2)
        if conn is None:
            pytest.skip("scan-count probe is lilli-engine only (stdlib driver lacks it)")
        conn.reset_full_scan_count()
        migrate_role_column(store2)
        assert conn.full_scan_count() == 0, (
            f"re-opened already-migrated store full-scanned: "
            f"{conn.full_scan_count()} scans"
        )
    finally:
        store2.close()
