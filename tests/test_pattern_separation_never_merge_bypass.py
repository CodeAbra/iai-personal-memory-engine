from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from iai_mcp.events import query_events
from iai_mcp.store import (
    RECORDS_TABLE,
    MemoryStore,
)
from iai_mcp.types import MemoryRecord

EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]
NEAR_DUP_DEFAULT: float = 0.92

def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(
            f"cos_target must be in [-1.0, 1.0], got {cos_target}"
        )
    if embed_dim < 2:
        raise ValueError(f"embed_dim must be >= 2, got {embed_dim}")
    residual = math.sqrt(max(0.0, 1.0 - cos_target * cos_target))
    return [cos_target, residual] + [0.0] * (embed_dim - 2)

def _make_record(
    *,
    embedding: list[float],
    tier: str = "episodic",
    literal_surface: str = "alice prefers tea over coffee",
    **overrides,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    base = dict(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
        community_id=None,
        centrality=0.5,
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
        language="en",
    )
    base.update(overrides)
    return MemoryRecord(**base)

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def _events_in_insert_order(store: MemoryStore) -> list[dict]:
    events = query_events(store, kind="pattern_separation_pass", limit=1000)
    return list(reversed(events))

@pytest.fixture(autouse=True)
def _reset_patsep_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for var in (
        "IAI_MCP_PATSEP_NEAR_DUP_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_THRESHOLD",
        "IAI_MCP_PATSEP_LINK_INITIAL_WEIGHT",
        "IAI_MCP_PATSEP_TOP_K",
        "IAI_MCP_PATSEP_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("IAI_MCP_EMBED_DIM", str(EMBED_DIM))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp"))

@pytest.fixture
def fresh_store(tmp_path: Path) -> MemoryStore:
    return _make_store(tmp_path)

def test_never_merge_new_candidate_bypasses_skip(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING, never_merge=False)
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(
        embedding=near_dup_embedding,
        never_merge=True,
        literal_surface="alice identity anchor — never merge",
    )
    b_id_before = rec_b.id
    fresh_store.insert(rec_b)

    assert rec_b.id == b_id_before, (
        f"never_merge=True new candidate must keep its own id; "
        f"got rec_b.id={rec_b.id}, expected {b_id_before}"
    )
    assert rec_b.id != a_id, (
        "never_merge=True new candidate must not be collapsed into existing record"
    )

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"never_merge=True candidate must INSERT as its own row; "
        f"records table has {tbl.count_rows()} rows (expected 2)"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "insert", (
        f"event for never_merge=True candidate must report action=insert; "
        f"got body={b_event_body}"
    )

def test_never_merge_existing_record_bypasses_skip(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(
        embedding=REFERENCE_EMBEDDING,
        never_merge=True,
        literal_surface="alice identity anchor — must never absorb duplicates",
    )
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(embedding=near_dup_embedding, never_merge=False)
    b_id_before = rec_b.id
    fresh_store.insert(rec_b)

    assert rec_b.id == b_id_before, (
        f"new candidate near existing never_merge=True record must keep its id; "
        f"got rec_b.id={rec_b.id}, expected {b_id_before}"
    )
    assert rec_b.id != a_id, (
        "new candidate must not be collapsed into a never_merge=True existing record"
    )

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"never_merge=True existing record must not absorb new inserts; "
        f"records table has {tbl.count_rows()} rows (expected 2)"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "insert", (
        f"event for insert near never_merge=True existing record must be action=insert; "
        f"got body={b_event_body}"
    )

def test_directive_new_candidate_bypasses_skip(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A directive-flagged candidate must never be folded into an existing
    near-duplicate, same as a never_merge lock -- inserted directly via
    store.insert() so this isolates the store-level gate from capture.py's
    own dedup (a different call site entirely)."""
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING, never_merge=False)
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(
        embedding=near_dup_embedding,
        directive=True,
        literal_surface="alice always wants terse commit messages",
    )
    b_id_before = rec_b.id
    fresh_store.insert(rec_b)

    assert rec_b.id == b_id_before, (
        f"directive=True new candidate must keep its own id; "
        f"got rec_b.id={rec_b.id}, expected {b_id_before}"
    )
    assert rec_b.id != a_id, (
        "directive=True new candidate must not be collapsed into existing record"
    )

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"directive=True candidate must INSERT as its own row; "
        f"records table has {tbl.count_rows()} rows (expected 2)"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "insert", (
        f"event for directive=True candidate must report action=insert; "
        f"got body={b_event_body}"
    )

def test_neither_pinned_skip_merge_still_fires(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING, never_merge=False)
    fresh_store.insert(rec_a)
    a_id = rec_a.id

    near_dup_embedding = _make_embedding_at_cosine(0.97)
    rec_b = _make_record(embedding=near_dup_embedding, never_merge=False)
    fresh_store.insert(rec_b)

    assert rec_b.id == a_id, (
        f"SKIP-merge must fire when neither party is pinned; "
        f"expected rec_b.id={a_id}, got {rec_b.id}"
    )

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 1, (
        f"SKIP-merge must collapse near-duplicate to 1 row when neither pinned; "
        f"got {tbl.count_rows()} rows"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events; got {len(events)}"
    )
    b_event_body = events[1]["data"]
    assert b_event_body["action"] == "skip", (
        f"event for standard near-dup must report action=skip; "
        f"got body={b_event_body}"
    )
    assert b_event_body["near_dup_hit_id"] == str(a_id), b_event_body

def test_cross_tier_near_dup_never_folds(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dedup is a within-tier operation: a semantic summary quotes its
    episodic members (cos ~0.96 by construction), and folding it into an
    episode would collapse knowledge into the very memory it summarizes."""
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    episodic = _make_record(embedding=REFERENCE_EMBEDDING, tier="episodic")
    fresh_store.insert(episodic)

    summary = _make_record(
        embedding=_make_embedding_at_cosine(0.99),
        tier="semantic",
        literal_surface="summary of the tea preference cluster",
    )
    summary_id_before = summary.id
    fresh_store.insert(summary)

    assert summary.id == summary_id_before, (
        "semantic record folded into an episodic near-neighbor"
    )
    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    from iai_mcp.store._buffers import flush_record_buffer
    flush_record_buffer(fresh_store)
    assert tbl.count_rows() == 2


def test_same_tier_near_dup_still_folds(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    first = _make_record(embedding=REFERENCE_EMBEDDING, tier="semantic")
    fresh_store.insert(first)

    second = _make_record(
        embedding=_make_embedding_at_cosine(0.99), tier="semantic",
    )
    fresh_store.insert(second)

    assert second.id == first.id, (
        "same-tier near-duplicate must still fold into the survivor"
    )
