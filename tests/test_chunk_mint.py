"""Behavioral coverage for the procedural chunk minter and proc_transitions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pandas as pd
import pytest

from iai_mcp import errors
from iai_mcp.lilli.cycle.proc_mine import (
    MIN_DISTINCT_SESSIONS,
    PAIR_COUNT_FLOOR,
    CofirePairCandidate,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore, flush_record_buffer

_BASE_TS = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _fresh_store(tmp_path: Path) -> "tuple[MemoryStore, Path]":
    home = tmp_path / "operator-home"
    store_root = home / ".iai-mcp"
    store = MemoryStore(path=store_root)
    return store, home


def _candidate(
    pair: tuple[str, str],
    source: str = "retrieval_cofired",
    count: int | None = None,
    session_count: int | None = None,
    first_ts: datetime = _BASE_TS,
    last_ts: datetime | None = None,
) -> CofirePairCandidate:
    return CofirePairCandidate(
        pair=pair,
        source=source,
        count=PAIR_COUNT_FLOOR if count is None else count,
        session_count=MIN_DISTINCT_SESSIONS if session_count is None else session_count,
        sessions=frozenset({"s1", "s2", "s3"}),
        first_ts=first_ts,
        last_ts=last_ts or first_ts + timedelta(minutes=5),
    )


def _row_for(df: pd.DataFrame, rid) -> dict | None:
    sub = df[df["id"] == str(rid)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()


def _is_tombstoned(row: dict) -> bool:
    val = row.get("tombstoned_at")
    return val is not None and not pd.isna(val)


# --- proc_transitions composite-PK smoke ---


def test_proc_transitions_merge_insert_twice_yields_one_row_advanced(tmp_path):
    store, _ = _fresh_store(tmp_path)
    alice_id = str(uuid4())
    bob_id = str(uuid4())
    tbl = store.db.open_table("proc_transitions")

    tbl.merge_insert(["src", "dst", "source"]).execute(
        [
            {
                "src": alice_id,
                "dst": bob_id,
                "source": "retrieval_cofired",
                "count": 3,
                "session_count": 2,
                "first_ts": _BASE_TS.isoformat(),
                "last_ts": _BASE_TS.isoformat(),
                "chunk_id": None,
                "updated_at": _BASE_TS.isoformat(),
            }
        ]
    )
    advanced_ts = (_BASE_TS + timedelta(hours=1)).isoformat()
    tbl.merge_insert(["src", "dst", "source"]).execute(
        [
            {
                "src": alice_id,
                "dst": bob_id,
                "source": "retrieval_cofired",
                "count": 7,
                "session_count": 4,
                "first_ts": _BASE_TS.isoformat(),
                "last_ts": advanced_ts,
                "chunk_id": None,
                "updated_at": advanced_ts,
            }
        ]
    )

    rows = tbl.to_pandas()
    matching = rows[
        (rows["src"] == alice_id) & (rows["dst"] == bob_id) & (rows["source"] == "retrieval_cofired")
    ]
    assert len(matching) == 1
    row = matching.iloc[0]
    assert int(row["count"]) == 7
    assert row["last_ts"] == advanced_ts


def test_proc_transitions_plain_add_on_existing_pk_raises_integrity_error(tmp_path):
    store, _ = _fresh_store(tmp_path)
    alice_id = str(uuid4())
    bob_id = str(uuid4())
    tbl = store.db.open_table("proc_transitions")

    row = {
        "src": alice_id,
        "dst": bob_id,
        "source": "retrieval_cofired",
        "count": 1,
        "session_count": 1,
        "first_ts": _BASE_TS.isoformat(),
        "last_ts": _BASE_TS.isoformat(),
        "chunk_id": None,
        "updated_at": _BASE_TS.isoformat(),
    }
    tbl.add([row])
    with pytest.raises(errors.IntegrityError):
        tbl.add([row])


# --- gate + mint + directed transition row ---


def test_under_floor_candidate_mints_nothing(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    records_before = store.db.open_table(RECORDS_TABLE).count_rows()
    transitions_before = store.db.open_table("proc_transitions").count_rows()

    under = _candidate(("alice", "bob"), count=PAIR_COUNT_FLOOR - 1)
    result = persist_proc_chunk(store, under)

    assert result is None
    assert store.db.open_table(RECORDS_TABLE).count_rows() == records_before
    assert store.db.open_table("proc_transitions").count_rows() == transitions_before


def test_under_session_floor_candidate_mints_nothing(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    records_before = store.db.open_table(RECORDS_TABLE).count_rows()
    transitions_before = store.db.open_table("proc_transitions").count_rows()

    # count clears its own floor -- only the session-spread clause is under.
    under = _candidate(
        ("alice", "bob"), count=PAIR_COUNT_FLOOR, session_count=MIN_DISTINCT_SESSIONS - 1
    )
    result = persist_proc_chunk(store, under)

    assert result is None
    assert store.db.open_table(RECORDS_TABLE).count_rows() == records_before
    assert store.db.open_table("proc_transitions").count_rows() == transitions_before


def test_over_floor_candidate_mints_readable_procedural_record(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    candidate = _candidate(("alice", "bob"))

    chunk_id = persist_proc_chunk(store, candidate)

    assert chunk_id is not None
    rec = store.get(chunk_id)
    assert rec is not None
    assert rec.tier == "procedural"
    assert rec.detail_level < 3
    assert rec.never_decay is False


def test_direction_preserved_across_two_proc_transitions_rows(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    forward = _candidate(("alice", "bob"))
    backward = _candidate(("bob", "alice"))

    forward_id = persist_proc_chunk(store, forward)
    backward_id = persist_proc_chunk(store, backward)

    assert forward_id is not None
    assert backward_id is not None
    assert forward_id != backward_id

    rows = store.db.open_table("proc_transitions").to_pandas()
    fwd_rows = rows[(rows["src"] == "alice") & (rows["dst"] == "bob")]
    bwd_rows = rows[(rows["src"] == "bob") & (rows["dst"] == "alice")]
    assert len(fwd_rows) == 1
    assert len(bwd_rows) == 1


def test_source_discriminator_yields_two_rows_for_same_pair(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    cand_s1 = _candidate(("alice", "bob"), source="s1")
    cand_s2 = _candidate(("alice", "bob"), source="s2")

    id1 = persist_proc_chunk(store, cand_s1)
    id2 = persist_proc_chunk(store, cand_s2)

    assert id1 is not None
    assert id2 is not None
    assert id1 != id2

    rows = store.db.open_table("proc_transitions").to_pandas()
    matching = rows[(rows["src"] == "alice") & (rows["dst"] == "bob")]
    assert len(matching) == 2
    assert set(matching["source"]) == {"s1", "s2"}


# --- reinforce on repeat sighting ---


def test_repeat_candidate_reinforces_no_duplicate(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    first = _candidate(("alice", "bob"), count=PAIR_COUNT_FLOOR, session_count=MIN_DISTINCT_SESSIONS)

    first_id = persist_proc_chunk(store, first)
    assert first_id is not None

    # Backdate the chunk's decay clock generously so reinforcement's freshening
    # is observably distinguishable from the mint-time value.
    tbl = store.db.open_table(RECORDS_TABLE)
    stale_ts = (_BASE_TS - timedelta(days=400)).isoformat()
    tbl.update(
        where=f"id = '{first_id}'",
        values={"last_reviewed": stale_ts, "created_at": stale_ts},
    )
    backdated_row = _row_for(tbl.to_pandas(), first_id)
    assert backdated_row is not None
    backdated_last_reviewed = backdated_row["last_reviewed"]

    second = _candidate(
        ("alice", "bob"),
        count=PAIR_COUNT_FLOOR + 5,
        session_count=MIN_DISTINCT_SESSIONS + 2,
        first_ts=_BASE_TS,
        last_ts=_BASE_TS + timedelta(days=1),
    )
    second_id = persist_proc_chunk(store, second)

    assert second_id == first_id
    assert store.db.open_table(RECORDS_TABLE).count_rows(
        filter="tier = 'procedural'"
    ) == 1
    assert store.db.open_table("proc_transitions").count_rows() == 1

    rows = store.db.open_table("proc_transitions").to_pandas()
    matching = rows[(rows["src"] == "alice") & (rows["dst"] == "bob")]
    assert len(matching) == 1
    row = matching.iloc[0]
    assert int(row["count"]) == PAIR_COUNT_FLOOR + 5

    row = _row_for(tbl.to_pandas(), first_id)
    assert row is not None
    assert not _is_tombstoned(row)
    # Same read path on both sides of the comparison -- type-agnostic
    # regardless of what dtype the driver hands back for a TEXT timestamp.
    assert row["last_reviewed"] != backdated_last_reviewed


def test_repeat_candidate_under_real_buffered_window_no_duplicate_mint(tmp_path, monkeypatch):
    """The conftest autoflush fixture flushes after every store.insert() in
    every other test, masking the production buffered-visibility window.
    This test opts out via IAI_MCP_TEST_NO_AUTOFLUSH=1 (set BEFORE the first
    insert) so the second sighting's dedup read genuinely races an unflushed
    buffer -- the exact condition persist_proc_chunk's liveness check must
    survive without minting a second chunk for the same (src,dst,source)."""
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    monkeypatch.setenv("IAI_MCP_TEST_NO_AUTOFLUSH", "1")
    store, _ = _fresh_store(tmp_path)
    first = _candidate(("alice", "bob"), count=PAIR_COUNT_FLOOR, session_count=MIN_DISTINCT_SESSIONS)

    first_id = persist_proc_chunk(store, first)
    assert first_id is not None

    second = _candidate(
        ("alice", "bob"),
        count=PAIR_COUNT_FLOOR + 5,
        session_count=MIN_DISTINCT_SESSIONS + 2,
        first_ts=_BASE_TS,
        last_ts=_BASE_TS + timedelta(days=1),
    )
    second_id = persist_proc_chunk(store, second)

    assert second_id == first_id

    # Settle whatever is still buffered (production's own eventual flush,
    # e.g. the 500-row threshold) so the on-disk state is assertable. The
    # mint-vs-reinforce decision above already happened under the real
    # buffered window -- this flush does not retroactively fix a wrong call.
    flush_record_buffer(store)
    live_procedural = store.db.open_table(RECORDS_TABLE).count_rows(
        filter="tier = 'procedural' AND tombstoned_at IS NULL"
    )
    assert live_procedural == 1


def test_different_source_same_pair_mints_separately_not_a_repeat(tmp_path):
    from iai_mcp.lilli.cycle.chunk import persist_proc_chunk

    store, _ = _fresh_store(tmp_path)
    cand_s1 = _candidate(("alice", "bob"), source="s1")
    cand_s1_repeat = _candidate(("alice", "bob"), source="s1", count=PAIR_COUNT_FLOOR + 3)
    cand_s2 = _candidate(("alice", "bob"), source="s2")

    id_s1 = persist_proc_chunk(store, cand_s1)
    id_s1_repeat = persist_proc_chunk(store, cand_s1_repeat)
    id_s2 = persist_proc_chunk(store, cand_s2)

    assert id_s1 == id_s1_repeat
    assert id_s2 != id_s1
    assert store.db.open_table(RECORDS_TABLE).count_rows(
        filter="tier = 'procedural'"
    ) == 2
