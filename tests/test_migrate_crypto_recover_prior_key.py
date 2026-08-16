
from __future__ import annotations

import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.migrate import migrate_crypto_recover_prior_key
from iai_mcp.store import MemoryStore
from iai_mcp.store._buffers import flush_record_buffer
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT


def _minimal_record(literal: str, embedding: list[float] | None = None) -> MemoryRecord:
    rid = uuid4()
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=rid,
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * 384,
        structure_hv=b"\x00" * 1250,
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
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )


def _setup_rotated_store(
    tmp_path: Path,
) -> tuple[MemoryStore, MemoryRecord, MemoryRecord, MemoryRecord, bytes, int]:
    """Build a store with a tombstoned record, a pending-embedding record, and a
    live control record, all written under a prior key, then rotate the on-disk
    key so recovery must decrypt under the prior key and re-key under the new one.
    """
    root = tmp_path / "recover-preserve"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)

    store_a = MemoryStore(path=root, user_id="default")
    rec_dead = _minimal_record(
        "tombstoned-through-recover", embedding=[1.0] + [0.0] * 383
    )
    rec_pending = _minimal_record(
        "pending-through-recover", embedding=[0.0, 1.0] + [0.0] * 382
    )
    rec_live = _minimal_record(
        "survives-recover-live", embedding=[0.0, 0.0, 1.0] + [0.0] * 381
    )
    store_a.insert(rec_dead)
    store_a.insert(rec_pending)
    store_a.insert(rec_live)
    flush_record_buffer(store_a)

    with store_a.db.ro_conn() as conn:
        (row_count,) = conn.execute("SELECT COUNT(*) FROM records").fetchone()
    assert row_count == 3, (
        "fixture setup must land three distinct rows before recovery; a "
        "near-duplicate collapse here invalidates the whole gate"
    )

    now_iso = datetime.now(timezone.utc).isoformat()
    with store_a.db._conn_lock:
        store_a.db._conn.execute(
            "UPDATE records SET tombstoned_at = ?, live = 0 WHERE id = ?",
            (now_iso, str(rec_dead.id)),
        )
        store_a.db._conn.execute(
            "UPDATE records SET embedding_pending = 1 WHERE id = ?",
            (str(rec_pending.id),),
        )
        store_a.db._conn.commit()

    with store_a.db.ro_conn() as conn:
        (live_label,) = conn.execute(
            "SELECT vec_label FROM records WHERE id = ?", (str(rec_live.id),)
        ).fetchone()

    del store_a

    key_b = secrets.token_bytes(32)
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")

    return store_b, rec_dead, rec_pending, rec_live, key_a, int(live_label)


def test_recover_prior_key_preserves_tombstone(tmp_path: Path) -> None:
    store_b, rec_dead, rec_pending, rec_live, key_a, live_label = (
        _setup_rotated_store(tmp_path)
    )

    with store_b.db.ro_conn() as conn:
        tomb, live = conn.execute(
            "SELECT tombstoned_at, live FROM records WHERE id = ?",
            (str(rec_dead.id),),
        ).fetchone()
    assert tomb is not None and int(live) == 0, (
        "a tombstoned record must remain marked dead before recovery runs; "
        "otherwise the fixture never exercised the tombstone path"
    )

    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)

    assert out.get("no_op") is False, (
        "recovery must actually stage and swap; a no-op short-circuit would "
        "pass every downstream assertion without exercising the staging path"
    )
    assert out.get("records_staged") == 3

    with store_b.db.ro_conn() as conn:
        tomb, live = conn.execute(
            "SELECT tombstoned_at, live FROM records WHERE id = ?",
            (str(rec_dead.id),),
        ).fetchone()
    assert tomb is not None and int(live) == 0, (
        "a record tombstoned before recovery must remain tombstoned after it"
    )

    with store_b.db.ro_conn() as conn:
        (drift,) = conn.execute(
            "SELECT COUNT(*) FROM records WHERE tombstoned_at IS NOT NULL AND live = 1"
        ).fetchone()
    assert int(drift) == 0, "no record may emerge both tombstoned and live"

    with store_b.db.ro_conn() as conn:
        (post_label,) = conn.execute(
            "SELECT vec_label FROM records WHERE id = ?", (str(rec_live.id),)
        ).fetchone()
    assert post_label == live_label, (
        "a record's index label must survive recovery unchanged so the "
        "vector index rebuild that follows stays consistent"
    )

    got_live = store_b.get(rec_live.id)
    assert got_live is not None
    assert got_live.literal_surface == "survives-recover-live"


def test_recover_prior_key_preserves_pending_embedding(tmp_path: Path) -> None:
    store_b, rec_dead, rec_pending, rec_live, key_a, _live_label = (
        _setup_rotated_store(tmp_path)
    )

    with store_b.db.ro_conn() as conn:
        (pending,) = conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?",
            (str(rec_pending.id),),
        ).fetchone()
    assert pending == 1, (
        "a pending-embedding record must remain pending before recovery runs; "
        "otherwise the fixture never exercised the pending-embedding path"
    )

    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)

    assert out.get("no_op") is False, (
        "recovery must actually stage and swap; a no-op short-circuit would "
        "pass every downstream assertion without exercising the staging path"
    )
    assert out.get("records_staged") == 3

    with store_b.db.ro_conn() as conn:
        (pending,) = conn.execute(
            "SELECT embedding_pending FROM records WHERE id = ?",
            (str(rec_pending.id),),
        ).fetchone()
    assert pending == 1, (
        "a record pending an embedding before recovery must not be silently "
        "marked ready after it"
    )

    got_live = store_b.get(rec_live.id)
    assert got_live is not None
    assert got_live.literal_surface == "survives-recover-live"


def test_recover_prior_key_atomic_swap_and_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "store"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    key_b = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)

    store_a = MemoryStore(path=root, user_id="default")
    rec = _minimal_record("verbatim-prior-key-recover")
    store_a.insert(rec)
    rid = rec.id
    del store_a

    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")
    # A wrong-generation record surfaces as a NAMED integrity error carrying
    # the recover-prior-key runbook — never a bare empty-message InvalidTag.
    from iai_mcp.hippo import HippoIntegrityError

    with pytest.raises(HippoIntegrityError, match="recover-prior-key"):
        store_b.get(rid)

    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)
    assert out.get("no_op") is False
    assert out.get("records_staged") == 1
    assert out.get("rows_needed_prior_key") == 1

    got = store_b.get(rid)
    assert got is not None
    assert got.literal_surface == "verbatim-prior-key-recover"

    out2 = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)
    assert out2.get("no_op") is True
    assert out2.get("reason") == "all_rows_decrypt_with_current_key"


def test_recover_prior_key_preserves_rowid_vec_label_alignment_on_gap(
    tmp_path: Path,
) -> None:
    """A store that has ever hard-deleted a record carries a non-contiguous
    `vec_label` sequence. Recovery staging must preserve `vec_label` as the
    `rowid` alias so the two never diverge, and the `id` uniqueness
    constraint must survive the swap.
    """
    from iai_mcp.errors import IntegrityError
    from iai_mcp.hippo._table import _decode_raw_row_embedding
    from iai_mcp.store import RECORDS_TABLE, _uuid_literal

    root = tmp_path / "recover-gap"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)

    store_a = MemoryStore(path=root, user_id="default")
    rec_first = _minimal_record("gap-fixture-first")
    rec_deleted = _minimal_record("gap-fixture-deleted")
    rec_third = _minimal_record("gap-fixture-third")
    rec_fourth = _minimal_record("gap-fixture-fourth")
    store_a.insert(rec_first)
    store_a.insert(rec_deleted)
    store_a.insert(rec_third)
    store_a.insert(rec_fourth)
    flush_record_buffer(store_a)

    store_a.db.open_table(RECORDS_TABLE).delete(
        f"id = '{_uuid_literal(rec_deleted.id)}'"
    )

    with store_a.db.ro_conn() as conn:
        labels = [
            int(row[0])
            for row in conn.execute(
                "SELECT vec_label FROM records ORDER BY vec_label"
            ).fetchall()
        ]
    assert len(labels) == 3, "hard-delete must leave exactly three surviving rows"
    gaps = [b - a for a, b in zip(labels, labels[1:])]
    assert any(gap > 1 for gap in gaps), (
        "hard-delete must leave a non-contiguous vec_label sequence "
        f"(got {labels}); otherwise the regression under test is not "
        "reproduced"
    )

    del store_a

    key_b = secrets.token_bytes(32)
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")

    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)
    assert out.get("no_op") is False
    assert out.get("records_staged") == 3

    del store_b

    store_c = MemoryStore(path=root, user_id="default")
    with store_c.db.ro_conn() as conn:
        rows = conn.execute("SELECT rowid, vec_label FROM records").fetchall()
    assert len(rows) == 3
    for rowid, vec_label in rows:
        assert int(rowid) == int(vec_label), (
            f"rowid {rowid} != vec_label {vec_label} after recovery on a "
            "store with a pre-existing hard-delete gap"
        )

    with store_c.db.ro_conn() as conn:
        post_labels = [
            int(row[0])
            for row in conn.execute(
                "SELECT vec_label FROM records ORDER BY vec_label"
            ).fetchall()
        ]
    assert post_labels == labels, (
        f"recovery renumbered index labels {labels} -> {post_labels}; the "
        "persisted vector index keys on the original values"
    )

    with store_c.db.ro_conn() as conn:
        surviving_ids = [
            row[0] for row in conn.execute("SELECT id FROM records").fetchall()
        ]
    assert len(set(surviving_ids)) == 3, (
        "exactly the three surviving records' distinct ids must remain "
        "after recovery"
    )

    with store_c.db.ro_conn() as conn:
        existing_row = conn.execute(
            "SELECT * FROM records WHERE id = ?", (str(rec_first.id),)
        ).fetchone()
    dup_row = _decode_raw_row_embedding(dict(existing_row))
    dup_row.pop("vec_label", None)
    with pytest.raises(IntegrityError, match="records.id"):
        store_c.db.open_table(RECORDS_TABLE).add([dup_row])

    rerun = migrate_crypto_recover_prior_key(store_c, key_a, dry_run=False)
    assert rerun.get("no_op") is True, (
        "a second recovery run against an already-recovered store must be a "
        "no-op, not a repeat stage"
    )


def test_recover_prior_key_refuses_on_duplicate_id_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source whose loaded `records` snapshot holds a duplicate `id` must
    never be silently amplified or collapsed by recovery — the pre-flight
    distinct-id guard raises fail-loud before any staging happens.

    The mismatch is injected via monkeypatch rather than a raw duplicate-id
    seed: the two storage drivers disagree on whether a raw `INSERT OR
    IGNORE` can even land a duplicate id (the native engine's OR IGNORE falls
    back to the `vec_label` PK and lets it through; stdlib's OR IGNORE honors
    `id UNIQUE` and no-ops it), so a driver-specific seed is not what this
    guard exercises. Monkeypatching the extracted `_duplicate_id_count` is
    driver-independent and tests the guard itself.
    """
    import iai_mcp.migrate._crypto_mig as crypto_mig

    root = tmp_path / "recover-dirty-source"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)

    store_a = MemoryStore(path=root, user_id="default")
    rec = _minimal_record("alice-dirty-source-fixture")
    store_a.insert(rec)
    flush_record_buffer(store_a)
    del store_a

    # Rotate the key so recovery is genuinely needed (a store where every row
    # already decrypts with the current key short-circuits before the
    # pre-flight guard ever runs).
    key_b = secrets.token_bytes(32)
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")

    monkeypatch.setattr(
        crypto_mig,
        "_duplicate_id_count",
        lambda df: (len(df) + 1, len(df)),
    )

    with pytest.raises(RuntimeError, match="duplicate id"):
        migrate_crypto_recover_prior_key(store_b, key_a, dry_run=False)

    # The refusal must not have created (or left behind) a staging table.
    from iai_mcp.migrate import CRYPTO_RECOVER_STAGING, _db_table_names_set

    assert CRYPTO_RECOVER_STAGING not in _db_table_names_set(store_b.db)


def test_recover_prior_key_dry_run_counts(tmp_path: Path) -> None:
    root = tmp_path / "store2"
    root.mkdir()
    key_a = secrets.token_bytes(32)
    key_b = secrets.token_bytes(32)
    kpath = root / ".crypto.key"
    kpath.write_bytes(key_a)
    os.chmod(kpath, 0o600)
    store_a = MemoryStore(path=root, user_id="default")
    store_a.insert(_minimal_record("dry-run-count"))
    del store_a
    kpath.write_bytes(key_b)
    os.chmod(kpath, 0o600)
    store_b = MemoryStore(path=root, user_id="default")
    out = migrate_crypto_recover_prior_key(store_b, key_a, dry_run=True)
    assert out.get("dry_run") is True
    assert out.get("would_stage") == 1
    assert out.get("rows_needing_prior_key") == 1
