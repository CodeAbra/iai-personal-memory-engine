from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.daemon import _load_erasure_config
from iai_mcp.events import query_events
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

FROZEN_NOW = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)

HIGH_UTILITY_N = 5
LOW_UTILITY_N = 10
PROTECTED_N = 4
TOTAL_N = HIGH_UTILITY_N + LOW_UTILITY_N + PROTECTED_N

def _make_record(
    *,
    tier: str,
    centrality: float,
    pinned: bool,
    never_decay: bool,
    last_reviewed: datetime | None,
    created_at: datetime,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=[0.01] * embed_dim,
        community_id=None,
        centrality=centrality,
        detail_level=1,
        pinned=pinned,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=last_reviewed,
        never_decay=never_decay,
        never_merge=False,
        provenance=[],
        created_at=created_at,
        updated_at=created_at,
        language="en",
    )

def _build_three_cohort_store(
    store: MemoryStore, now: datetime,
) -> dict[str, list[UUID]]:
    age_60d = now - timedelta(days=60)
    recent_review = now - timedelta(days=7)
    embed_dim = store._embed_dim

    cohort_ids: dict[str, list[UUID]] = {
        "high_utility": [],
        "low_utility": [],
        "protected": [],
    }

    for _ in range(HIGH_UTILITY_N):
        rec = _make_record(
            tier="episodic",
            centrality=0.5,
            pinned=False,
            never_decay=False,
            last_reviewed=recent_review,
            created_at=age_60d,
            embed_dim=embed_dim,
            literal_surface="alice's notes on graph topology stability",
        )
        store.insert(rec)
        cohort_ids["high_utility"].append(rec.id)

    for _ in range(LOW_UTILITY_N):
        rec = _make_record(
            tier="episodic",
            centrality=0.005,
            pinned=False,
            never_decay=False,
            last_reviewed=None,
            created_at=age_60d,
            embed_dim=embed_dim,
            literal_surface="bob mentioned the weather on a forgotten day",
        )
        store.insert(rec)
        cohort_ids["low_utility"].append(rec.id)

    for i in range(PROTECTED_N):
        is_pinned = i < 2
        is_never_decay = not is_pinned
        rec = _make_record(
            tier="episodic",
            centrality=0.005,
            pinned=is_pinned,
            never_decay=is_never_decay,
            last_reviewed=None,
            created_at=age_60d,
            embed_dim=embed_dim,
            literal_surface=(
                "alice locked this as a permanent reminder"
                if is_pinned
                else "bob marked this never_decay for the long memory"
            ),
        )
        store.insert(rec)
        cohort_ids["protected"].append(rec.id)

    return cohort_ids

@pytest.fixture
def iai_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PYTHON_KEYRING_BACKEND", "keyring.backends.fail.Keyring")
    monkeypatch.setenv("IAI_MCP_CRYPTO_PASSPHRASE", "test-phase11-passphrase")
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / ".iai-mcp" / "lancedb"))
    import keyring.core

    keyring.core._keyring_backend = None
    yield tmp_path
    keyring.core._keyring_backend = None

@pytest.fixture
def pipeline(iai_home, tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "false")

    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )

    store = MemoryStore()
    cohort_ids = _build_three_cohort_store(store, FROZEN_NOW)

    tbl = store.db.open_table(RECORDS_TABLE)
    assert tbl.count_rows() == TOTAL_N, (
        f"three-cohort fixture must insert exactly {TOTAL_N} records, "
        f"got {tbl.count_rows()}"
    )

    pipe = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
    )
    return pipe, store, cohort_ids

def _row_for(df, rid: UUID) -> dict | None:
    sub = df[df["id"] == str(rid)]
    if sub.empty:
        return None
    return sub.iloc[0].to_dict()

def _is_tombstoned(row: dict) -> bool:
    import pandas as pd

    val = row.get("tombstoned_at")
    return val is not None and not pd.isna(val)

def test_low_utility_cohort_tombstoned_after_one_pass(pipeline):
    pipe, store, cohort_ids = pipeline

    ok, payload = pipe._step_erasure_agent(None)
    assert ok is True, payload
    assert payload.get("dry_run") is False, (
        f"fixture should disable dry-run, got payload={payload}"
    )
    assert payload.get("count_quarantined") == LOW_UTILITY_N, (
        f"expected count_quarantined={LOW_UTILITY_N}, got {payload}"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    assert df.shape[0] == TOTAL_N, (
        f"tombstoning sets a column, must not delete rows; "
        f"expected {TOTAL_N}, got {df.shape[0]}"
    )

    for rid in cohort_ids["low_utility"]:
        row = _row_for(df, rid)
        assert row is not None, f"low-utility row {rid} disappeared"
        assert _is_tombstoned(row), (
            f"low-utility row {rid} should be tombstoned, "
            f"got tombstoned_at={row.get('tombstoned_at')!r}"
        )

    for rid in cohort_ids["high_utility"]:
        row = _row_for(df, rid)
        assert row is not None, f"high-utility row {rid} disappeared"
        assert not _is_tombstoned(row), (
            f"high-utility row {rid} should NOT be tombstoned, "
            f"got tombstoned_at={row.get('tombstoned_at')!r}"
        )

    for rid in cohort_ids["protected"]:
        row = _row_for(df, rid)
        assert row is not None, f"protected row {rid} disappeared"
        assert not _is_tombstoned(row), (
            f"protected row {rid} should NOT be tombstoned (R3 carve-out), "
            f"got tombstoned_at={row.get('tombstoned_at')!r} "
            f"pinned={row.get('pinned')} never_decay={row.get('never_decay')}"
        )

def test_protected_cohort_survives_multiple_passes(pipeline):
    pipe, store, cohort_ids = pipeline

    for pass_idx in range(3):
        ok, payload = pipe._step_erasure_agent(None)
        assert ok is True, f"pass {pass_idx}: {payload}"

        tbl = store.db.open_table(RECORDS_TABLE)
        df = tbl.to_pandas()
        for rid in cohort_ids["protected"]:
            row = _row_for(df, rid)
            assert row is not None, (
                f"pass {pass_idx}: protected row {rid} disappeared"
            )
            assert not _is_tombstoned(row), (
                f"pass {pass_idx}: protected row {rid} was tombstoned; "
                f"R3 carve-out failed. "
                f"pinned={row.get('pinned')} never_decay={row.get('never_decay')}"
            )

def test_aged_tombstones_dropped_after_second_pass(pipeline, monkeypatch):
    pipe, store, cohort_ids = pipeline

    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )

    ok, _ = pipe._step_erasure_agent(None)
    assert ok is True

    cfg = _load_erasure_config()
    ttl = cfg.tombstone_ttl_sec

    fast_forward = FROZEN_NOW + timedelta(seconds=ttl + 60)
    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: fast_forward,
    )

    ok2, payload2 = pipe._step_optimize_hippo(None)
    assert ok2 is True, payload2
    assert payload2.get("count_dropped_by_erasure") == LOW_UTILITY_N, (
        f"expected {LOW_UTILITY_N} drops, got {payload2}"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()

    surviving_ids = set(df["id"].tolist())
    for rid in cohort_ids["low_utility"]:
        assert str(rid) not in surviving_ids, (
            f"low-utility row {rid} should be dropped after TTL fast-forward, "
            f"but is still present"
        )

    for rid in cohort_ids["high_utility"]:
        row = _row_for(df, rid)
        assert row is not None, f"high-utility row {rid} disappeared"
        assert not _is_tombstoned(row)
    for rid in cohort_ids["protected"]:
        row = _row_for(df, rid)
        assert row is not None, f"protected row {rid} disappeared"
        assert not _is_tombstoned(row)

    assert df.shape[0] == HIGH_UTILITY_N + PROTECTED_N, (
        f"expected {HIGH_UTILITY_N + PROTECTED_N} surviving rows "
        f"({HIGH_UTILITY_N} high-utility + {PROTECTED_N} protected), "
        f"got {df.shape[0]}"
    )

def test_dry_run_mode_emits_event_no_mutation(
    iai_home, tmp_path, monkeypatch,
):
    monkeypatch.setenv("IAI_MCP_ERASURE_DRY_RUN", "true")
    monkeypatch.setattr(
        "iai_mcp.lilli.cycle.sleep_pipeline._utc_now", lambda: FROZEN_NOW,
    )

    store = MemoryStore()
    cohort_ids = _build_three_cohort_store(store, FROZEN_NOW)

    pipe = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
    )

    ok, payload = pipe._step_erasure_agent(None)
    assert ok is True, payload
    assert payload.get("dry_run") is True, (
        f"dry-run env var should land in payload, got {payload}"
    )
    assert payload.get("count_quarantined") == LOW_UTILITY_N, (
        f"dry-run must still count the eligibility set "
        f"(expected {LOW_UTILITY_N}), got {payload}"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    assert df.shape[0] == TOTAL_N
    for _, row in df.iterrows():
        assert not _is_tombstoned(row.to_dict()), (
            f"dry-run wrote a tombstone on row id={row.get('id')}; "
            f"mutation path must be inert when dry_run=True (R7)"
        )

    events = query_events(store, kind="erasure_agent_pass", limit=10)
    assert len(events) >= 1, (
        f"no erasure_agent_pass event emitted in dry-run mode, "
        f"got events={events}"
    )
    body = events[0]["data"]
    assert body.get("dry_run_mode") is True, body
    assert body.get("count_quarantined") == LOW_UTILITY_N, body
    assert len(cohort_ids["low_utility"]) == LOW_UTILITY_N

def test_erasure_event_body_shape_and_uniqueness(pipeline):
    pipe, store, _ = pipeline

    ok, _ = pipe._step_erasure_agent(None)
    assert ok is True

    events = query_events(store, kind="erasure_agent_pass", limit=10)
    assert len(events) == 1, (
        f"exactly one erasure_agent_pass event per pass (R5), "
        f"got {len(events)} -> {events}"
    )

    body = events[0]["data"]

    required_keys = {
        "count_quarantined",
        "count_dropped",
        "total_records_after",
        "threshold_used",
        "dry_run_mode",
    }
    missing = required_keys - set(body.keys())
    assert not missing, (
        f"erasure_agent_pass body missing required keys {sorted(missing)}; "
        f"got body={body}"
    )

    assert isinstance(body["count_quarantined"], int), body
    assert isinstance(body["count_dropped"], int), body
    assert isinstance(body["total_records_after"], int), body
    assert isinstance(body["threshold_used"], float), body
    assert isinstance(body["dry_run_mode"], bool), body

    assert body["count_quarantined"] == LOW_UTILITY_N, body
    assert body["count_dropped"] == 0, body
    assert body["total_records_after"] == TOTAL_N, body
    assert body["threshold_used"] == 0.02, body
    assert body["dry_run_mode"] is False, body
