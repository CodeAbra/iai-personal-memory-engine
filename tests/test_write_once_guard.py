"""Write-once guard: HippoTable.update()/.delete() is the PRIMARY,
column-precise enforcement that events content is write-once (except its
ciphertext envelope column) and literal_surface is write-once -- values
arrive there as a column-name-keyed mapping, so it is proven complete for
every column shape. The connection-layer regex in hippo/_db.py is a
best-effort backstop for raw _conn.execute paths that bypass HippoTable
entirely; its own tests assert only the shapes it actually catches, not
completeness. Parametrized over both storage drivers -- each run opens a
FRESH store under its own driver (the on-disk format is decided at
first-open time from the env), never an env var layered on a native-format
file, so the stdlib branch is genuinely exercised."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp import errors
from iai_mcp.store import EVENTS_TABLE, RECORDS_TABLE, MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord


@pytest.fixture(params=["lilli", "stdlib"])
def driver(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("LILLI_STORAGE_DRIVER", request.param)
    return request.param


def _make_record(text: str = "guard test content") -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=[0.1] * EMBED_DIM,
        community_id=None,
        centrality=0.0,
        detail_level=1,
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


def _open_store(tmp_path: Path, driver: str) -> MemoryStore:
    store = MemoryStore(tmp_path, user_id="test")
    assert store.db._storage_driver == driver, (
        f"expected a genuine {driver!r}-driver store (on-disk format decided "
        f"at first open), got {store.db._storage_driver!r}"
    )
    return store


# --- HippoTable: PRIMARY, column-precise guard (proven complete) --------


def test_hippotable_update_events_content_column_raises(
    tmp_path: Path, driver: str
) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            store.db.open_table(EVENTS_TABLE).update(
                where="id = '1'", values={"kind": "x"}
            )
    finally:
        store.close()


def test_hippotable_update_events_mixed_columns_raises(
    tmp_path: Path, driver: str
) -> None:
    """One protected column in the mapping is enough to refuse the whole
    write -- there is no partial-column update path."""
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            store.db.open_table(EVENTS_TABLE).update(
                where="id = '1'", values={"data_json": "ct", "kind": "x"}
            )
    finally:
        store.close()


def test_hippotable_update_events_ciphertext_envelope_column_passes(
    tmp_path: Path, driver: str
) -> None:
    """A ciphertext-swap-only update (key rotation / redaction) must
    succeed -- the guard is column-aware, not a table-wide ban."""
    store = _open_store(tmp_path, driver)
    try:
        from iai_mcp.events import write_event

        write_event(store, kind="probe", data={"x": 1}, severity="info")
        tbl = store.db.open_table(EVENTS_TABLE)
        eid = str(tbl.to_pandas().iloc[0]["id"])
        before = tbl.to_pandas()
        before_ct = before[before["id"] == eid].iloc[0]["data_json"]
        # data_json is an encrypted column -- HippoTable.update() re-encrypts
        # it transparently under the store's own key (like a real key
        # rotation / redaction ciphertext swap). The point under test is
        # that the write is no longer refused, not the decrypt round-trip
        # (covered elsewhere by the crypto rotation tests) -- so assert only
        # that the stored ciphertext changed.
        tbl.update(where=f"id = '{eid}'", values={"data_json": "new plaintext for re-encryption"})
        after = tbl.to_pandas()
        after_ct = after[after["id"] == eid].iloc[0]["data_json"]
        assert after_ct != before_ct
    finally:
        store.close()


def test_hippotable_delete_events_raises(tmp_path: Path, driver: str) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="append-only"):
            store.db.open_table(EVENTS_TABLE).delete(where="id = '1'")
    finally:
        store.close()


def test_hippotable_update_literal_surface_raises(tmp_path: Path, driver: str) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            store.db.open_table(RECORDS_TABLE).update(
                where="id = '1'", values={"literal_surface": "corrupted"}
            )
    finally:
        store.close()


def test_hippotable_update_many_by_id_events_content_column_raises(
    tmp_path: Path, driver: str
) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            store.db.open_table(EVENTS_TABLE).update_many_by_id(
                [("1", {"kind": "x"})]
            )
    finally:
        store.close()


# --- connection-layer backstop: only what it actually catches -----------


def test_raw_connection_delete_events_bareword_raises(
    tmp_path: Path, driver: str
) -> None:
    """The never-regress case: a future change that drops the
    connection-layer guard and keeps only the HippoTable one would let this
    raw execute() through undetected."""
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="append-only"):
            with store.db._conn_lock:
                store.db._conn.execute("DELETE FROM events WHERE id = ?", ("1",))
    finally:
        store.close()


def test_raw_connection_update_events_content_column_raises(
    tmp_path: Path, driver: str
) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            with store.db._conn_lock:
                store.db._conn.execute(
                    "UPDATE events SET kind = ? WHERE id = ?", ("x", "1")
                )
    finally:
        store.close()


def test_raw_connection_update_events_ciphertext_envelope_column_passes(
    tmp_path: Path, driver: str
) -> None:
    """At the backstop layer: the same ciphertext-only statement
    HippoTable.update() issues internally must not be re-blocked here, or
    every legitimate crypto-rotation/migration write breaks again."""
    store = _open_store(tmp_path, driver)
    try:
        from iai_mcp.events import write_event

        write_event(store, kind="probe", data={"x": 1}, severity="info")
        tbl = store.db.open_table(EVENTS_TABLE)
        eid = str(tbl.to_pandas().iloc[0]["id"])
        with store.db._conn_lock:
            store.db._conn.execute(
                "UPDATE events SET data_json = ? WHERE id = ?",
                ("iai:enc:v1:swap", eid),
            )
    finally:
        store.close()


@pytest.mark.parametrize(
    "quoted_form",
    ['"events"', "`events`", "[events]"],
    ids=["double-quoted", "backtick", "bracket"],
)
def test_raw_connection_quoted_events_identifier_raises(
    tmp_path: Path, driver: str, quoted_form: str
) -> None:
    """A quoted table identifier must not bypass the backstop, for both
    UPDATE and DELETE."""
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation):
            with store.db._conn_lock:
                store.db._conn.execute(
                    f"UPDATE {quoted_form} SET kind = ? WHERE id = ?", ("x", "1")
                )
        with pytest.raises(errors.CanonicalSourceViolation, match="append-only"):
            with store.db._conn_lock:
                store.db._conn.execute(
                    f"DELETE FROM {quoted_form} WHERE id = ?", ("1",)
                )
    finally:
        store.close()


def test_raw_connection_leading_comment_events_update_raises(
    tmp_path: Path, driver: str
) -> None:
    """A leading SQL comment must not defeat the backstop."""
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            with store.db._conn_lock:
                store.db._conn.execute(
                    "-- trace-id: abc\nUPDATE events SET kind = ? WHERE id = ?",
                    ("x", "1"),
                )
    finally:
        store.close()


@pytest.mark.parametrize(
    "quoted_form",
    ['"literal_surface"', "`literal_surface`", "[literal_surface]"],
    ids=["double-quoted", "backtick", "bracket"],
)
def test_raw_connection_quoted_literal_surface_assignment_raises(
    tmp_path: Path, driver: str, quoted_form: str
) -> None:
    """A quoted column identifier must not defeat the write-once check on
    literal_surface."""
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            with store.db._conn_lock:
                store.db._conn.execute(
                    f"UPDATE records SET {quoted_form} = ? WHERE id = ?",
                    ("corrupted", "1"),
                )
    finally:
        store.close()


def test_literal_surface_bare_update_rewrite_raises(tmp_path: Path, driver: str) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            with store.db._conn_lock:
                store.db._conn.execute(
                    "UPDATE records SET literal_surface = ? WHERE id = ?",
                    ("corrupted", "1"),
                )
    finally:
        store.close()


def test_literal_surface_on_conflict_upsert_rewrite_raises(
    tmp_path: Path, driver: str
) -> None:
    store = _open_store(tmp_path, driver)
    try:
        with pytest.raises(errors.CanonicalSourceViolation, match="write-once"):
            with store.db._conn_lock:
                store.db._conn.execute(
                    "INSERT INTO records (id, literal_surface) VALUES (?, ?) "
                    "ON CONFLICT (id) DO UPDATE SET "
                    "literal_surface=excluded.literal_surface",
                    ("1", "corrupted"),
                )
    finally:
        store.close()


# --- legitimate writes are unaffected -------------------------------------


def test_legitimate_non_protected_column_update_passes(
    tmp_path: Path, driver: str
) -> None:
    store = _open_store(tmp_path, driver)
    try:
        rec = _make_record()
        store.insert(rec)
        flush_record_buffer(store)
        with store.db._conn_lock:
            store.db._conn.execute(
                "UPDATE records SET centrality = ? WHERE id = ?",
                (0.5, str(rec.id)),
            )
            row = store.db._conn.execute(
                "SELECT centrality FROM records WHERE id = ?", (str(rec.id),)
            ).fetchone()
        assert float(row["centrality"]) == 0.5
    finally:
        store.close()


def test_normal_record_insert_still_succeeds(tmp_path: Path, driver: str) -> None:
    store = _open_store(tmp_path, driver)
    try:
        rec = _make_record("normal insert path")
        store.insert(rec)
        flush_record_buffer(store)
        retrieved = store.get(rec.id)
        assert retrieved is not None
        assert retrieved.literal_surface == "normal insert path"
    finally:
        store.close()


# --- guard-drift tripwire --------------------------------------------------


def test_events_mutable_column_allowlists_stay_in_sync() -> None:
    """The primary guard's allow-list (hippo/__init__._ENCRYPTED_EVENTS_COLUMNS)
    and the backstop's allow-list (hippo/_db._EVENTS_MUTABLE_COLUMNS) must
    name the same columns -- a change to one without the other silently
    reopens either a false-block on a legitimate ciphertext swap or a
    false-pass gap."""
    from iai_mcp.hippo import _ENCRYPTED_EVENTS_COLUMNS
    from iai_mcp.hippo._db import _EVENTS_MUTABLE_COLUMNS

    assert set(_ENCRYPTED_EVENTS_COLUMNS) == _EVENTS_MUTABLE_COLUMNS


# --- crypto rotation/migration callers: a guard violation must propagate,
# never degrade into a silently-counted per-row failure --------------------


def test_canonical_source_violation_not_caught_by_narrow_os_value_runtime_except() -> None:
    """Every crypto rotation/migration call site that legitimately rewrites
    literal_surface or an events ciphertext column wraps its update() call
    in `except (OSError, ValueError, RuntimeError)`, treating a caught
    exception as one countable re-encrypt failure rather than an abort.
    CanonicalSourceViolation must never become a subclass of any of those
    three, or a genuine guard violation at one of those call sites would
    silently degrade into a counted failure instead of propagating -- see
    the two end-to-end tests below for the propagation itself on the
    reachable literal_surface paths."""
    assert not issubclass(
        errors.CanonicalSourceViolation, (OSError, ValueError, RuntimeError)
    )


def test_cli_crypto_rotate_records_guard_violation_propagates_uncaught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A broken encrypt_field (returns its input unchanged, not
    ciphertext-shaped) makes HippoTable.update()'s literal_surface guard
    raise CanonicalSourceViolation on cmd_crypto_rotate's records branch.
    That branch's `except (OSError, ValueError, RuntimeError)` does not
    list CanonicalSourceViolation, so this must propagate uncaught out of
    cmd_crypto_rotate rather than being counted as one rotate_failures
    entry."""
    import argparse
    import os
    import secrets

    import iai_mcp.crypto as crypto_mod

    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path))
    monkeypatch.delenv("IAI_MCP_CRYPTO_PASSPHRASE", raising=False)
    key_path = tmp_path / ".crypto.key"
    key_path.write_bytes(secrets.token_bytes(32))
    os.chmod(key_path, 0o600)

    from iai_mcp.cli import cmd_crypto_rotate
    from iai_mcp.store import MemoryStore

    store = MemoryStore()
    store.insert(_make_record("rotate guard violation probe"))

    monkeypatch.setattr(
        crypto_mod,
        "encrypt_field",
        lambda plaintext, key, associated_data=b"": plaintext,
    )

    args = argparse.Namespace(user_id="default")
    with pytest.raises(errors.CanonicalSourceViolation):
        cmd_crypto_rotate(args)


def test_migrate_encryption_v2_to_v3_records_fallback_guard_violation_propagates_uncaught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """migrate_encryption_v2_to_v3's records branch normally re-encrypts via
    a batched merge_insert, which re-encrypts every row itself before the
    backstop ever inspects the submitted value -- a shape violation cannot
    reach the guard on that path. Its fallback loop (taken when the batch
    call raises) issues a direct HippoTable.update() per row instead, which
    checks the raw submitted value with no re-encrypt step ahead of it. A
    broken encrypt_field forces that fallback update to submit a
    plaintext-shaped literal_surface: the merge_insert failure is caught
    and logged, then the per-row `except (OSError, ValueError, RuntimeError)`
    around the fallback update does not list CanonicalSourceViolation, so
    the guard violation propagates uncaught out of
    migrate_encryption_v2_to_v3 rather than silently leaving the migration
    partially applied."""
    from iai_mcp.hippo._table import HippoMergeInsert
    from iai_mcp.migrate import migrate_encryption_v2_to_v3
    from iai_mcp.store import MemoryStore
    from tests.test_migrate_encryption import _make, _write_plaintext_row
    import iai_mcp.migrate._crypto_mig as crypto_mig_mod

    store = MemoryStore(path=tmp_path)
    rec = _make(text="migration guard violation probe")
    _write_plaintext_row(store, rec)

    monkeypatch.setattr(
        crypto_mig_mod,
        "encrypt_field",
        lambda plaintext, key, associated_data=b"": plaintext,
    )

    def _boom(self: "HippoMergeInsert", data: object) -> None:
        raise RuntimeError("forced merge_insert failure")

    monkeypatch.setattr(HippoMergeInsert, "execute", _boom)

    with pytest.raises(errors.CanonicalSourceViolation):
        migrate_encryption_v2_to_v3(store)
