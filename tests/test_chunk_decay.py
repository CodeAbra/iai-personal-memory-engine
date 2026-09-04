"""Behavioral coverage for decay_proc_chunks: tombstone retirement, tier
scope, and bidirectional dry_run."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd

from iai_mcp.lilli.cycle.chunk import (
    CHUNK_DECAY_AGE_DAYS,
    CHUNK_DECAY_STALENESS_DAYS,
    decay_proc_chunks,
    persist_proc_chunk,
)
from iai_mcp.lilli.cycle.proc_mine import (
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    CofirePairCandidate,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord, SCHEMA_VERSION_CURRENT

_MINT_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)
# 500 days clears both CHUNK_DECAY_AGE_DAYS and CHUNK_DECAY_STALENESS_DAYS
# with a wide margin -- eligibility does not depend on the exact chosen values.
_DECAY_NOW = _MINT_TS + timedelta(days=500)


def _fresh_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(path=tmp_path / "operator-home" / ".iai-mcp")


def _candidate(pair: tuple[str, str]) -> CofirePairCandidate:
    return CofirePairCandidate(
        pair=pair,
        source="retrieval_cofired",
        count=PAIR_COUNT_FLOOR,
        session_count=MIN_DISTINCT_SESSIONS,
        sessions=frozenset({"s1", "s2", "s3"}),
        first_ts=_MINT_TS,
        last_ts=_MINT_TS + timedelta(minutes=5),
    )


def _row_for(df: pd.DataFrame, rid) -> dict | None:
    sub = df[df["id"] == str(rid)]
    return None if sub.empty else sub.iloc[0].to_dict()


def _is_tombstoned(row: dict) -> bool:
    val = row.get("tombstoned_at")
    return val is not None and not pd.isna(val)


def _backdate(store: MemoryStore, record_id, *, created_at: datetime, last_reviewed: datetime | None) -> None:
    tbl = store.db.open_table(RECORDS_TABLE)
    tbl.update(
        where=f"id = '{record_id}'",
        values={
            "created_at": created_at.isoformat(),
            "last_reviewed": last_reviewed.isoformat() if last_reviewed is not None else None,
        },
    )


def _insert_semantic_record(store: MemoryStore, *, created_at: datetime, last_reviewed: datetime | None):
    rec = MemoryRecord(
        id=uuid4(),
        tier="semantic",
        literal_surface="alice prefers dark roast coffee",
        aaak_index="",
        embedding=[0.01] * store._embed_dim,
        community_id=None,
        centrality=0.0,
        detail_level=1,
        pinned=False,
        stability=0.5,
        difficulty=0.3,
        last_reviewed=last_reviewed,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=created_at,
        updated_at=created_at,
        tags=[],
        language="en",
        s5_trust_score=0.5,
        profile_modulation_gain={},
        schema_version=SCHEMA_VERSION_CURRENT,
    )
    store.insert(rec)
    return rec.id


def test_decay_constants_are_positive_ints():
    assert isinstance(CHUNK_DECAY_AGE_DAYS, int) and CHUNK_DECAY_AGE_DAYS > 0
    assert isinstance(CHUNK_DECAY_STALENESS_DAYS, int) and CHUNK_DECAY_STALENESS_DAYS > 0


def test_forced_stale_procedural_chunk_is_retired(tmp_path):
    store = _fresh_store(tmp_path)
    chunk_id = persist_proc_chunk(store, _candidate(("alice", "bob")))
    assert chunk_id is not None

    _backdate(store, chunk_id, created_at=_MINT_TS, last_reviewed=_MINT_TS)

    result = decay_proc_chunks(store, now=_DECAY_NOW, dry_run=False)
    assert result["quarantined"] == 1
    assert result["retired"] == 1

    tbl = store.db.open_table(RECORDS_TABLE)
    row = _row_for(tbl.to_pandas(), chunk_id)
    assert row is not None  # decay NEVER deletes; it tombstones in place
    assert _is_tombstoned(row)
    assert not bool(row["live"])


def test_reinforced_chunk_survives_same_decay_pass(tmp_path):
    store = _fresh_store(tmp_path)
    chunk_id = persist_proc_chunk(store, _candidate(("alice", "bob")))
    assert chunk_id is not None

    # created_at is old enough to clear the age clause on its own; only the
    # fresh last_reviewed keeps this chunk out of the staleness clause.
    fresh_reviewed = _DECAY_NOW - timedelta(days=5)
    _backdate(store, chunk_id, created_at=_MINT_TS, last_reviewed=fresh_reviewed)

    result = decay_proc_chunks(store, now=_DECAY_NOW, dry_run=False)
    assert result["quarantined"] == 0
    assert result["retired"] == 0

    tbl = store.db.open_table(RECORDS_TABLE)
    row = _row_for(tbl.to_pandas(), chunk_id)
    assert row is not None
    assert not _is_tombstoned(row)
    assert bool(row["live"])


def test_tier_scope_negative_control_semantic_record_survives(tmp_path):
    store = _fresh_store(tmp_path)
    chunk_id = persist_proc_chunk(store, _candidate(("alice", "bob")))
    assert chunk_id is not None
    _backdate(store, chunk_id, created_at=_MINT_TS, last_reviewed=_MINT_TS)

    semantic_id = _insert_semantic_record(store, created_at=_MINT_TS, last_reviewed=_MINT_TS)

    result = decay_proc_chunks(store, now=_DECAY_NOW, dry_run=False)
    assert result["quarantined"] == 1  # only the procedural chunk

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()

    chunk_row = _row_for(df, chunk_id)
    assert chunk_row is not None
    assert _is_tombstoned(chunk_row)

    semantic_row = _row_for(df, semantic_id)
    assert semantic_row is not None
    assert not _is_tombstoned(semantic_row)
    assert bool(semantic_row["live"])


def test_dry_run_is_bidirectionally_load_bearing(tmp_path):
    store = _fresh_store(tmp_path)
    chunk_id = persist_proc_chunk(store, _candidate(("alice", "bob")))
    assert chunk_id is not None
    _backdate(store, chunk_id, created_at=_MINT_TS, last_reviewed=_MINT_TS)

    dry_result = decay_proc_chunks(store, now=_DECAY_NOW, dry_run=True)
    assert dry_result["quarantined"] == 1
    assert dry_result["retired"] == 0

    tbl = store.db.open_table(RECORDS_TABLE)
    row = _row_for(tbl.to_pandas(), chunk_id)
    assert row is not None
    assert not _is_tombstoned(row)
    assert bool(row["live"])

    live_result = decay_proc_chunks(store, now=_DECAY_NOW, dry_run=False)
    assert live_result["quarantined"] == 1
    assert live_result["retired"] == 1

    row = _row_for(store.db.open_table(RECORDS_TABLE).to_pandas(), chunk_id)
    assert row is not None
    assert _is_tombstoned(row)
    assert not bool(row["live"])


def test_chunk_module_never_reads_erasure_env():
    repo_root = Path(__file__).resolve().parent.parent
    source = (repo_root / "src" / "iai_mcp" / "lilli" / "cycle" / "chunk.py").read_text(
        encoding="utf-8"
    )
    assert "IAI_MCP_ERASURE_" not in source
    assert "PYTEST_CURRENT_TEST" not in source
