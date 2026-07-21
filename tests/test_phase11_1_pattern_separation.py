from __future__ import annotations

import inspect
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from iai_mcp.events import query_events
from iai_mcp.store import (
    EDGES_TABLE,
    RECORDS_TABLE,
    GateAction,
    MemoryStore,
)
from iai_mcp.types import MemoryRecord

EMBED_DIM = 4
REFERENCE_EMBEDDING: list[float] = [1.0, 0.0, 0.0, 0.0]
NEAR_DUP_DEFAULT: float = 0.92
LINK_DEFAULT: float = 0.70
LINK_WEIGHT_DEFAULT: float = 0.10

_UUID_REGEX: re.Pattern[str] = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

def _make_embedding_at_cosine(
    cos_target: float, embed_dim: int = EMBED_DIM,
) -> list[float]:
    if not (-1.0 <= cos_target <= 1.0):
        raise ValueError(
            f"cos_target must be in [-1.0, 1.0], got {cos_target}"
        )
    if embed_dim < 2:
        raise ValueError(
            f"embed_dim must be >= 2, got {embed_dim}"
        )
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

def _gate_body_source() -> str:
    return inspect.getsource(MemoryStore.pattern_separation_gate)

def _insert_body_source() -> str:
    return inspect.getsource(MemoryStore.insert)

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

def _events_in_insert_order(store: MemoryStore) -> list[dict]:
    events = query_events(store, kind="pattern_separation_pass", limit=1000)
    return list(reversed(events))

def test_near_duplicate_cohort_collapses_to_reinforce(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    for _ in range(10):
        rec_dup = _make_record(embedding=REFERENCE_EMBEDDING)
        pre_id = rec_dup.id
        fresh_store.insert(rec_dup)
        assert rec_dup.id == A_id, (
            f"SKIP path must mutate rec.id to existing-record uuid; "
            f"got rec_dup.id={rec_dup.id}, expected A_id={A_id}"
        )
        assert rec_dup.id != pre_id, (
            f"caller-transparent merge: rec.id must change away from the "
            f"fresh uuid {pre_id} after a SKIP collapse"
        )

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 1, (
        f"near-duplicate collapse must keep exactly 1 row; "
        f"got {tbl.count_rows()}"
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    self_loops = edges_df[
        (edges_df["edge_type"] == "hebbian")
        & (edges_df["src"] == str(A_id))
        & (edges_df["dst"] == str(A_id))
    ]
    assert len(self_loops) >= 1, (
        f"reinforce_record must persist a self-loop hebbian edge; "
        f"got rows={edges_df.to_dict('records')}"
    )
    self_loop_weight = float(self_loops["weight"].iloc[0])
    assert self_loop_weight >= 0.9, (
        f"10 reinforces x delta=0.1 must accumulate to ~1.0; "
        f"got weight={self_loop_weight}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 11, (
        f"expected 11 pattern_separation_pass events (1 insert + 10 skip), "
        f"got {len(events)}"
    )
    first_body = events[0]["data"]
    assert first_body["action"] == "insert", first_body
    assert first_body["near_dup_hit_id"] is None, first_body
    for idx, ev in enumerate(events[1:], start=1):
        body = ev["data"]
        assert body["action"] == "skip", (
            f"event {idx} must be a skip; got body={body}"
        )
        assert body["near_dup_hit_id"] == str(A_id), (
            f"event {idx} must reference A_id={A_id} as the near-dup hit; "
            f"got body={body}"
        )
        assert body["dry_run_mode"] is False, body
        assert body["near_dup_cos"] is not None and body["near_dup_cos"] >= 0.99, body

def test_link_seeding_creates_pattern_separation_seed_edge(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    rec_b = _make_record(embedding=_make_embedding_at_cosine(0.80))
    B_id = rec_b.id
    fresh_store.insert(rec_b)

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"link-band insert must NOT collapse; both A and B land. "
        f"Got {tbl.count_rows()} rows."
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    seed_edges = edges_df[edges_df["edge_type"] == "pattern_separation_seed"]
    assert len(seed_edges) == 1, (
        f"expected exactly 1 pattern_separation_seed edge between A and B, "
        f"got {len(seed_edges)} -> {seed_edges.to_dict('records')}"
    )

    edge_row = seed_edges.iloc[0]
    actual_endpoints = tuple(sorted([edge_row["src"], edge_row["dst"]]))
    expected_endpoints = tuple(sorted([str(A_id), str(B_id)]))
    assert actual_endpoints == expected_endpoints, (
        f"seed-edge endpoints must canonicalise to {expected_endpoints}; "
        f"got {actual_endpoints}"
    )

    weight = float(edge_row["weight"])
    assert abs(weight - LINK_WEIGHT_DEFAULT) < 1e-6, (
        f"seed-edge weight must equal link_initial_weight={LINK_WEIGHT_DEFAULT} "
        f"to within 1e-6; got weight={weight}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events (one per insert), "
        f"got {len(events)}"
    )
    body_b = events[1]["data"]
    assert body_b["action"] == "insert", body_b
    assert body_b["edges_seeded"] == 1, body_b
    assert body_b["near_dup_hit_id"] is None, body_b

def test_independent_insert_no_edge_seeded(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a)

    orthogonal = [0.0, 0.0, 1.0, 0.0]
    rec_c = _make_record(embedding=orthogonal)
    fresh_store.insert(rec_c)

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 2, (
        f"independent insert must NOT collapse; both A and C land. "
        f"Got {tbl.count_rows()} rows."
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    seed_edges = edges_df[edges_df["edge_type"] == "pattern_separation_seed"]
    assert len(seed_edges) == 0, (
        f"cos=0.0 < link_threshold=0.70 must NOT seed any seed-edge; "
        f"got {len(seed_edges)} -> {seed_edges.to_dict('records')}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 2, (
        f"expected 2 pattern_separation_pass events, got {len(events)}"
    )
    body_c = events[1]["data"]
    assert body_c["action"] == "insert", body_c
    assert body_c["edges_seeded"] == 0, body_c
    assert body_c["near_dup_hit_id"] is None, body_c

def test_dry_run_preserves_regular_insert(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "true")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a)
    inserted_ids: set[str] = {str(rec_a.id)}

    for _ in range(9):
        rec_dup = _make_record(embedding=REFERENCE_EMBEDDING)
        pre_id = rec_dup.id
        fresh_store.insert(rec_dup)
        assert rec_dup.id == pre_id, (
            f"dry-run R7: rec.id must NOT be mutated; "
            f"got {rec_dup.id}, expected fresh {pre_id}"
        )
        inserted_ids.add(str(rec_dup.id))

    tbl = fresh_store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == 10, (
        f"dry-run R7: all 10 inserts must land; got {tbl.count_rows()}"
    )

    edges_tbl = fresh_store.db.open_table(EDGES_TABLE)
    edges_df = edges_tbl.to_pandas()
    seed_edges = edges_df[edges_df["edge_type"] == "pattern_separation_seed"]
    assert len(seed_edges) == 0, (
        f"dry-run R7: zero seed edges; got {len(seed_edges)}"
    )

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 10, (
        f"dry-run R7: exactly 10 pattern_separation_pass events; "
        f"got {len(events)}"
    )

    first_body = events[0]["data"]
    assert first_body["action"] == "insert", first_body
    assert first_body["near_dup_hit_id"] is None, first_body
    assert first_body["dry_run_mode"] is True, first_body

    for idx, ev in enumerate(events[1:], start=1):
        body = ev["data"]
        assert body["action"] == "skip", (
            f"event {idx}: dry-run must still report the SKIP decision; "
            f"got body={body}"
        )
        assert body["dry_run_mode"] is True, body
        assert body["near_dup_hit_id"] in inserted_ids, (
            f"event {idx}: near_dup_hit_id must reference an already-inserted "
            f"record; got {body['near_dup_hit_id']!r}, inserted={inserted_ids}"
        )
        assert body["near_dup_cos"] is not None and body["near_dup_cos"] >= 0.99, body

def test_event_body_shape_and_field_types(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    A_id = rec_a.id
    fresh_store.insert(rec_a)

    rec_a2 = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a2)

    rec_b = _make_record(embedding=_make_embedding_at_cosine(0.80))
    B_id = rec_b.id
    fresh_store.insert(rec_b)

    rec_c = _make_record(embedding=[0.0, 0.0, 1.0, 0.0])
    fresh_store.insert(rec_c)

    events = _events_in_insert_order(fresh_store)
    assert len(events) == 4, (
        f"expected exactly 4 pattern_separation_pass events (1 per insert), "
        f"got {len(events)}"
    )

    expected_keys = {
        "action",
        "near_dup_hit_id",
        "near_dup_cos",
        "edges_seeded",
        "top_k_probed",
        "threshold_near_dup",
        "threshold_link",
        "dry_run_mode",
        "ann_prefilter_cos",
        "exact_confirm_cos",
    }

    for idx, ev in enumerate(events):
        body = ev["data"]
        assert set(body.keys()) == expected_keys, (
            f"event {idx} body keys must equal {sorted(expected_keys)}; "
            f"got extra={set(body.keys()) - expected_keys}, "
            f"missing={expected_keys - set(body.keys())}, body={body}"
        )
        assert isinstance(body["action"], str) and body["action"] in {"insert", "skip"}, body
        if body["near_dup_hit_id"] is not None:
            assert isinstance(body["near_dup_hit_id"], str), body
            assert _UUID_REGEX.match(body["near_dup_hit_id"]), (
                f"event {idx} near_dup_hit_id must be canonical UUID; "
                f"got {body['near_dup_hit_id']!r}"
            )
        if body["near_dup_cos"] is not None:
            assert isinstance(body["near_dup_cos"], float), body
        assert isinstance(body["edges_seeded"], int) and not isinstance(body["edges_seeded"], bool), body
        assert body["edges_seeded"] >= 0, body
        assert isinstance(body["top_k_probed"], int) and not isinstance(body["top_k_probed"], bool), body
        assert body["top_k_probed"] >= 0, body
        assert isinstance(body["threshold_near_dup"], float), body
        assert abs(body["threshold_near_dup"] - 0.92) < 1e-9, body
        assert isinstance(body["threshold_link"], float), body
        assert abs(body["threshold_link"] - 0.70) < 1e-9, body
        assert isinstance(body["dry_run_mode"], bool), body
        assert body["dry_run_mode"] is False, body
        if body["ann_prefilter_cos"] is not None:
            assert isinstance(body["ann_prefilter_cos"], float), body
        if body["exact_confirm_cos"] is not None:
            assert isinstance(body["exact_confirm_cos"], float), body

    a_body = events[0]["data"]
    assert a_body["action"] == "insert", a_body
    assert a_body["near_dup_hit_id"] is None, a_body
    assert a_body["edges_seeded"] == 0, a_body

    a2_body = events[1]["data"]
    assert a2_body["action"] == "skip", a2_body
    assert a2_body["near_dup_hit_id"] == str(A_id), a2_body
    assert a2_body["near_dup_cos"] is not None and a2_body["near_dup_cos"] >= 0.99, a2_body

    b_body = events[2]["data"]
    assert b_body["action"] == "insert", b_body
    assert b_body["edges_seeded"] == 1, b_body
    assert b_body["near_dup_hit_id"] is None, b_body

    c_body = events[3]["data"]
    assert c_body["action"] == "insert", c_body
    assert c_body["edges_seeded"] == 0, c_body
    assert c_body["near_dup_hit_id"] is None, c_body

def test_gate_does_not_mutate_record_embedding(
    fresh_store: MemoryStore, monkeypatch: pytest.MonkeyPatch,
) -> None:
    assign_pattern = re.compile(r"record\.embedding\s*=(?!=)")
    gate_src = _gate_body_source()
    insert_src = _insert_body_source()
    gate_hits = assign_pattern.findall(gate_src)
    insert_hits = assign_pattern.findall(insert_src)
    assert gate_hits == [], (
        f"R4: pattern_separation_gate body MUST NOT assign to record.embedding; "
        f"found {len(gate_hits)} match(es) -> {gate_hits}"
    )
    assert insert_hits == [], (
        f"R4: MemoryStore.insert body MUST NOT assign to record.embedding; "
        f"found {len(insert_hits)} match(es) -> {insert_hits}"
    )

    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    rec_isolated = _make_record(embedding=REFERENCE_EMBEDDING)
    original_embedding = list(rec_isolated.embedding)
    action_empty, payload_empty = fresh_store.pattern_separation_gate(rec_isolated)
    assert rec_isolated.embedding == original_embedding, (
        f"R4: gate on empty store mutated record.embedding; "
        f"original={original_embedding}, after={rec_isolated.embedding}"
    )
    assert action_empty == GateAction.INSERT, (
        f"empty store must return INSERT, got {action_empty}"
    )
    assert payload_empty == [], (
        f"empty store INSERT payload must be []; got {payload_empty}"
    )

    rec_a = _make_record(embedding=REFERENCE_EMBEDDING)
    fresh_store.insert(rec_a)

    rec_new = _make_record(embedding=REFERENCE_EMBEDDING)
    original_new_embedding = list(rec_new.embedding)
    action_hit, payload_hit = fresh_store.pattern_separation_gate(rec_new)
    assert rec_new.embedding == original_new_embedding, (
        f"R4: gate with matching hit mutated record.embedding; "
        f"original={original_new_embedding}, after={rec_new.embedding}"
    )
    assert action_hit == GateAction.SKIP, (
        f"matching hit must return SKIP, got {action_hit}"
    )
    assert isinstance(payload_hit, UUID), (
        f"SKIP payload must be the existing-record UUID; got {payload_hit!r}"
    )
    assert payload_hit == rec_a.id, (
        f"SKIP payload must equal A_id={rec_a.id}; got {payload_hit}"
    )
