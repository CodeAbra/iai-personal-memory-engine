from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord

def _make_record(**overrides) -> MemoryRecord:
    rng = np.random.RandomState(99)
    vec = rng.randn(EMBED_DIM).tolist()
    base = dict(
        id=uuid4(),
        tier="episodic",
        literal_surface="codec metadata test record",
        aaak_index="",
        embedding=vec,
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
        language="en",
    )
    base.update(overrides)
    return MemoryRecord(**base)

def _minimal_row(id_str: str, **overrides) -> dict:
    rng = np.random.RandomState(7)
    vec = rng.randn(EMBED_DIM).tolist()
    row = {
        "id": id_str,
        "tier": "episodic",
        "literal_surface": "injected row",
        "aaak_index": "",
        "embedding": vec,
        "community_id": "",
        "centrality": 0.0,
        "detail_level": 1,
        "pinned": 0,
        "stability": 0.0,
        "difficulty": 0.0,
        "last_reviewed": None,
        "never_decay": 0,
        "never_merge": 0,
        "provenance_json": "[]",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "tags_json": "[]",
        "language": "en",
        "s5_trust_score": 0.5,
        "profile_modulation_gain_json": "{}",
        "schema_version": 5,
        "structure_hv": b"",
        "tombstoned_at": None,
        "schema_bypass": 0,
        "labile_until": None,
        "wing": None,
        "room": None,
        "drawer": None,
    }
    row.update(overrides)
    return row

def test_memoryrecord_hv_tier_default_bsc() -> None:
    rec = _make_record()
    assert rec.hv_tier == "bsc"

def test_memoryrecord_structure_hv_payload_default_empty() -> None:
    rec = _make_record()
    assert rec.structure_hv_payload == b""
    assert isinstance(rec.structure_hv_payload, bytes)

def test_memoryrecord_hv_tier_accepts_three_tiers() -> None:
    for tier in ("bsc", "fhrr", "sparse_vsa"):
        rec = _make_record(hv_tier=tier)
        assert rec.hv_tier == tier

def test_memoryrecord_hv_tier_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="HV_TIER_ENUM"):
        _make_record(hv_tier="garbage")

def test_memoryrecord_structure_hv_payload_must_be_bytes() -> None:
    with pytest.raises(ValueError, match="structure_hv_payload must be bytes"):
        _make_record(structure_hv_payload="not bytes")

def test_hv_tier_enum_frozen() -> None:
    from iai_mcp.types import HV_TIER_ENUM

    assert isinstance(HV_TIER_ENUM, frozenset)
    assert HV_TIER_ENUM == frozenset({"bsc", "fhrr", "sparse_vsa"})
    assert len(HV_TIER_ENUM) == 3

def test_schema_version_v5_in_accepted() -> None:
    from iai_mcp.types import (
        SCHEMA_VERSION_ACCEPTED,
        SCHEMA_VERSION_CURRENT,
        SCHEMA_VERSION_V5,
        SCHEMA_VERSION_V6,
    )

    assert SCHEMA_VERSION_V5 == 5
    assert SCHEMA_VERSION_V6 == 6
    # V5 remains an accepted on-disk version (backward compat) even though the
    # current write version has advanced to V6.
    assert SCHEMA_VERSION_V5 in SCHEMA_VERSION_ACCEPTED
    assert SCHEMA_VERSION_V6 in SCHEMA_VERSION_ACCEPTED
    assert SCHEMA_VERSION_CURRENT == SCHEMA_VERSION_V6

def test_structure_hv_bytes_unchanged() -> None:
    from iai_mcp.types import STRUCTURE_HV_BYTES

    assert STRUCTURE_HV_BYTES == 1250

def test_fresh_hippo_db_has_new_columns(tmp_path: Path) -> None:
    from iai_mcp.hippo import HippoDB

    with HippoDB(tmp_path) as db:
        cols = {
            row["name"]
            for row in db._conn.execute("PRAGMA table_info(records)").fetchall()  # nosemgrep
        }
    assert "hv_tier" in cols
    assert "structure_hv_payload" in cols

def test_migration_idempotent_on_existing_db(tmp_path: Path) -> None:
    from iai_mcp.hippo import HippoDB
    from iai_mcp.migrate import _migrate_add_hv_tier_columns

    with HippoDB(tmp_path) as db:
        conn = db._conn
        _migrate_add_hv_tier_columns(conn)
        _migrate_add_hv_tier_columns(conn)

def test_existing_records_default_to_bsc_after_migration(tmp_path: Path) -> None:

    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec = _make_record()
        store.insert(rec)
        flush_record_buffer(store)
        raw = store.db._conn.execute(  # nosemgrep
            "SELECT hv_tier FROM records WHERE id = ?", (str(rec.id),)
        ).fetchone()
        assert raw is not None
        assert raw["hv_tier"] == "bsc"
    finally:
        store.close()

def test_inserted_bsc_record_round_trips(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path, user_id="test")
    try:
        rec = _make_record(hv_tier="bsc", structure_hv_payload=b"")
        store.insert(rec)
        flush_record_buffer(store)
        fetched = store.get(rec.id)
        assert fetched is not None
        assert fetched.hv_tier == "bsc"
        assert fetched.structure_hv_payload == b""
    finally:
        store.close()

def test_inserted_fhrr_record_round_trips(tmp_path: Path) -> None:
    payload = b"\x00" * 10000

    store1 = MemoryStore(tmp_path, user_id="test")
    rec_id = None
    try:
        rec = _make_record(hv_tier="fhrr", structure_hv_payload=payload)
        rec_id = rec.id
        store1.insert(rec)
        flush_record_buffer(store1)
    finally:
        store1.close()

    store2 = MemoryStore(tmp_path, user_id="test")
    try:
        fetched = store2.get(rec_id)
        assert fetched is not None
        assert fetched.hv_tier == "fhrr"
        assert fetched.structure_hv_payload == payload
    finally:
        store2.close()

def test_inserted_sparse_vsa_record_round_trips(tmp_path: Path) -> None:
    payload = b"\x01\x00" * 20

    store1 = MemoryStore(tmp_path, user_id="test")
    rec_id = None
    try:
        rec = _make_record(hv_tier="sparse_vsa", structure_hv_payload=payload)
        rec_id = rec.id
        store1.insert(rec)
        flush_record_buffer(store1)
    finally:
        store1.close()

    store2 = MemoryStore(tmp_path, user_id="test")
    try:
        fetched = store2.get(rec_id)
        assert fetched is not None
        assert fetched.hv_tier == "sparse_vsa"
        assert fetched.structure_hv_payload == payload
    finally:
        store2.close()

def test_from_row_emits_telemetry_on_invalid_hv_tier(tmp_path: Path) -> None:
    from iai_mcp import events

    store = MemoryStore(tmp_path, user_id="test")
    try:
        row_id = str(uuid4())
        row = _minimal_row(row_id, hv_tier="garbage")

        result = store._from_row(row)
        assert result is not None
        assert result.hv_tier == "bsc"
        assert result.structure_hv_payload == b""

        emitted = events.query_events(store, kind="codec_marker_missing", limit=10)
        assert len(emitted) >= 1, "Expected at least one codec_marker_missing event"
        reasons = [e["data"].get("reason", "") for e in emitted]
        assert any("HV_TIER_ENUM" in r for r in reasons), (
            f"Expected 'HV_TIER_ENUM' in reason; got {reasons}"
        )
    finally:
        store.close()

def test_from_row_handles_missing_hv_tier_silently(tmp_path: Path) -> None:
    from iai_mcp import events

    store = MemoryStore(tmp_path, user_id="test")
    try:
        row_id = str(uuid4())
        row = _minimal_row(row_id)
        row.pop("hv_tier", None)

        result = store._from_row(row)
        assert result is not None
        assert result.hv_tier == "bsc"
        assert result.structure_hv_payload == b""

        emitted = events.query_events(store, kind="codec_marker_missing", limit=10)
        assert len(emitted) == 0, (
            f"Expected no telemetry for missing hv_tier key; got {emitted}"
        )
    finally:
        store.close()

def test_from_row_emits_telemetry_on_wrong_type_payload(tmp_path: Path) -> None:
    from iai_mcp import events

    store = MemoryStore(tmp_path, user_id="test")
    try:
        row_id = str(uuid4())
        row = _minimal_row(row_id, hv_tier="fhrr", structure_hv_payload="not bytes")

        result = store._from_row(row)
        assert result is not None
        assert result.hv_tier == "bsc"
        assert result.structure_hv_payload == b""

        emitted = events.query_events(store, kind="codec_marker_missing", limit=10)
        assert len(emitted) >= 1, "Expected at least one codec_marker_missing event"
        reasons = [e["data"].get("reason", "") for e in emitted]
        assert any("expected bytes" in r for r in reasons), (
            f"Expected 'expected bytes' in reason; got {reasons}"
        )
    finally:
        store.close()
