from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import uuid4


from iai_mcp.store import (
    CPU_HAS_AVX2,
    RECORDS_TABLE,
    EDGES_TABLE,
    EVENTS_TABLE,
    BUDGET_TABLE,
    RATELIMIT_TABLE,
    MemoryStore,
)
from iai_mcp.hippo import HippoDB
from iai_mcp.types import MemoryRecord


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path)


def _make_record(embed_dim: int = 384) -> MemoryRecord:
    import datetime
    now = datetime.datetime.now(datetime.timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface="hello hippo",
        aaak_index="",
        embedding=[0.01 * i for i in range(embed_dim)],
        community_id=None,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=1.0,
        difficulty=0.3,
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
        schema_version=2,
        structure_hv=b"",
    )


def test_store_db_is_hippo(tmp_path):
    store = _make_store(tmp_path)
    assert isinstance(store.db, HippoDB), (
        f"Expected store.db to be HippoDB, got {type(store.db)}"
    )


def test_cpu_has_avx2_constant():
    assert CPU_HAS_AVX2 is True, (
        "CPU_HAS_AVX2 must be True on the Hippo backend (no AVX2 dependency)"
    )


def test_table_names_contains_all_tables(tmp_path):
    store = _make_store(tmp_path)
    names = set(store._table_names())
    expected = {RECORDS_TABLE, EDGES_TABLE, EVENTS_TABLE, BUDGET_TABLE, RATELIMIT_TABLE}
    missing = expected - names
    assert not missing, f"Missing tables after open: {missing}"


def test_insert_get_round_trip(tmp_path):
    store = _make_store(tmp_path)
    record = _make_record(embed_dim=store.embed_dim)
    store.insert(record)
    retrieved = store.get(record.id)
    assert retrieved is not None, "get() returned None after insert()"
    assert retrieved.literal_surface == "hello hippo"
    assert str(retrieved.id) == str(record.id)


def test_async_writes_lifecycle(tmp_path):
    store = _make_store(tmp_path)

    async def _run():
        await store.enable_async_writes(coalesce_ms=10, max_batch=8, max_queue_size=64)
        assert store._write_queue is not None, "_write_queue must be set after enable"
        await store.enable_async_writes()
        await store.disable_async_writes()
        assert store._write_queue is None, "_write_queue must be None after disable"
        assert store._async_conn is None, "_async_conn must stay None on Hippo path"

    asyncio.run(_run())


def test_ensure_tables_reads_embed_dim(tmp_path):
    store1 = _make_store(tmp_path)
    dim1 = store1.embed_dim
    store1.db.close()

    store2 = _make_store(tmp_path)
    dim2 = store2.embed_dim
    assert dim1 == dim2, f"embed_dim mismatch on re-open: {dim1} vs {dim2}"
