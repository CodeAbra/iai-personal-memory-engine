from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from iai_mcp.daemon import (
    _load_reconsolidation_config,
)
from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import default_state, save_state
from iai_mcp.reconsolidation_critic import PROMPT_TEMPLATE, call_critic
from iai_mcp.lilli.cycle.sleep_pipeline import (
    STEP_PHASE,
    SleepPhase,
    SleepPipeline,
    SleepStep,
)
from iai_mcp.store import RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD",
        "IAI_MCP_LABILE_WINDOW_SEC",
        "IAI_MCP_RECONSOLIDATION_TIER1",
        "IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD",
        "IAI_MCP_RECONSOLIDATION_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

def _make_record(
    *,
    embed_dim: int,
    literal: str = "alice prefers tea over coffee",
    embedding: list[float] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal,
        aaak_index="",
        embedding=embedding if embedding is not None else [0.01] * embed_dim,
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
        language="en",
        tags=["t"],
    )

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def _stub_centroids(
    monkeypatch: pytest.MonkeyPatch,
    centroids: dict[uuid.UUID, list[float]],
) -> None:
    assignment = SimpleNamespace(community_centroids=centroids)
    rich_club: list = []
    node_payload: dict = {}
    max_degree: int = 0
    fake = (assignment, rich_club, node_payload, max_degree)

    def _fake_try_load(store: Any) -> tuple:
        return fake

    monkeypatch.setattr(
        "iai_mcp.runtime_graph_cache.try_load", _fake_try_load,
    )

def test_R1_schema_bypass_column_exists_and_default_false(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    tbl = store.db.open_table(RECORDS_TABLE)
    assert "schema_bypass" in tbl.schema.names
    field = tbl.schema.field("schema_bypass")
    from iai_mcp.hippo import _schema as pa
    assert field.type.equals(pa.bool_()), (
        f"schema_bypass column type must be bool, got {field.type}"
    )

    rec = _make_record(embed_dim=store._embed_dim)
    store.insert(rec)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert bool(row["schema_bypass"]) is False

def test_R1_schema_bypass_true_when_cosine_meets_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "0.85")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    centroid_axis0 = [1.0] + [0.0] * (embed_dim - 1)
    _stub_centroids(monkeypatch, {uuid.uuid4(): centroid_axis0})

    rec_aligned = _make_record(
        embed_dim=embed_dim,
        literal="aligned with centroid",
        embedding=list(centroid_axis0),
    )
    store.insert(rec_aligned)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row_a = df[df["id"] == str(rec_aligned.id)].iloc[0]
    assert bool(row_a["schema_bypass"]) is True, (
        "aligned-to-centroid insert must tag schema_bypass=True"
    )

    orthogonal = [0.0, 1.0] + [0.0] * (embed_dim - 2)
    rec_far = _make_record(
        embed_dim=embed_dim,
        literal="orthogonal to centroid",
        embedding=orthogonal,
    )
    store.insert(rec_far)
    df = tbl.to_pandas()
    row_b = df[df["id"] == str(rec_far.id)].iloc[0]
    assert bool(row_b["schema_bypass"]) is False, (
        "orthogonal insert must leave schema_bypass=False"
    )

def test_R2_labile_until_set_by_reinforce_record_is_retrieval_true(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim)
    store.insert(rec)
    tbl = store.db.open_table(RECORDS_TABLE)

    df = tbl.to_pandas()
    row_pre = df[df["id"] == str(rec.id)].iloc[0]
    assert row_pre["labile_until"] is None or (
        hasattr(row_pre["labile_until"], "to_pydatetime")
        and str(row_pre["labile_until"]) == "NaT"
    ), f"fresh insert labile_until must be null/NaT, got {row_pre['labile_until']!r}"

    store.reinforce_record(rec.id)
    df = tbl.to_pandas()
    row_default = df[df["id"] == str(rec.id)].iloc[0]
    val_default = row_default["labile_until"]
    is_null_default = val_default is None or str(val_default) == "NaT"
    assert is_null_default, (
        f"default reinforce_record must NOT stamp labile_until, "
        f"got {val_default!r}"
    )

    before = datetime.now(timezone.utc)
    store.reinforce_record(rec.id, is_retrieval=True)
    after = datetime.now(timezone.utc)
    df = tbl.to_pandas()
    row_post = df[df["id"] == str(rec.id)].iloc[0]
    stamped = row_post["labile_until"]
    if hasattr(stamped, "to_pydatetime"):
        stamped = stamped.to_pydatetime()
    if isinstance(stamped, str):
        stamped = datetime.fromisoformat(stamped.replace("+00:00", "").rstrip("Z"))
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    expected_low = before + timedelta(seconds=3600) - timedelta(seconds=10)
    expected_high = after + timedelta(seconds=3600) + timedelta(seconds=10)
    assert expected_low <= stamped <= expected_high, (
        f"labile_until={stamped} not in [{expected_low}, {expected_high}]"
    )

def test_R3_reconsolidation_step_in_enum_and_order() -> None:
    assert SleepStep.RECONSOLIDATION.value == 9
    assert STEP_PHASE[SleepStep.RECONSOLIDATION] == SleepPhase.REM
    order = SleepPipeline._STEP_ORDER
    idx_recon = order.index(SleepStep.RECONSOLIDATION)
    idx_cluster = order.index(SleepStep.CLUSTER_REPLAY)
    idx_crisis = order.index(SleepStep.CRISIS_RECLUSTER)
    assert idx_recon == idx_cluster + 1, (
        f"RECONSOLIDATION must follow CLUSTER_REPLAY; "
        f"got idx_recon={idx_recon}, idx_cluster={idx_cluster}"
    )
    assert idx_recon < idx_crisis, (
        f"RECONSOLIDATION must precede CRISIS_RECLUSTER; "
        f"got idx_recon={idx_recon}, idx_crisis={idx_crisis}"
    )

def test_R3_step_body_emits_reconsolidation_pass_event_with_correct_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "true")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "0.5")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim, literal="alice loves haiku")
    store.insert(rec)

    store.reinforce_record(rec.id, is_retrieval=True)

    def _stub_batched(items: Any, *args: Any, **kwargs: Any) -> dict:
        return {rid: 0.9 for rid, _surface in items}

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _stub_batched,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True
    assert payload["records_scanned"] >= 1
    assert payload["records_reconsolidated"] >= 1
    assert payload["dry_run"] is False

    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1, (
        f"_step_reconsolidation must emit exactly one event, got {len(events)}"
    )
    body = events[0]["data"]
    expected_keys = {
        "records_scanned",
        "records_reconsolidated",
        "critic_calls",
        "dry_run_mode",
    }
    assert set(body.keys()) == expected_keys, (
        f"event body keys must be {expected_keys}, got {set(body.keys())}"
    )
    assert isinstance(body["records_scanned"], int)
    assert isinstance(body["records_reconsolidated"], int)
    assert isinstance(body["critic_calls"], int)
    assert isinstance(body["dry_run_mode"], bool)
    assert body["records_scanned"] >= 1
    assert body["records_reconsolidated"] >= 1
    assert body["critic_calls"] >= 1
    assert body["dry_run_mode"] is False

def test_R3_tier1_false_skips_critic_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "false")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim)
    store.insert(rec)
    store.reinforce_record(rec.id, is_retrieval=True)

    def _raise_critic(*args: Any, **kwargs: Any) -> dict:
        raise AssertionError(
            "evaluate_batch_reconsolidation must NOT be called when "
            "reconsolidation_tier1=False"
        )

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _raise_critic,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True
    assert payload["records_reconsolidated"] == 0

    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1
    body = events[0]["data"]
    assert body["critic_calls"] == 0
    assert body["records_reconsolidated"] == 0
    assert body["records_scanned"] >= 1

def test_R4_schema_bypass_tagging_does_not_run_on_pattern_separation_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_PATSEP_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "0.85")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    emb = [1.0] + [0.0] * (embed_dim - 1)
    rec_a = _make_record(
        embed_dim=embed_dim, literal="first record", embedding=emb,
    )
    store.insert(rec_a)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row_a_pre = df[df["id"] == str(rec_a.id)].iloc[0]
    assert bool(row_a_pre["schema_bypass"]) is False

    _stub_centroids(monkeypatch, {uuid.uuid4(): emb})

    rec_b = _make_record(
        embed_dim=embed_dim, literal="near-dup", embedding=emb,
    )
    store.insert(rec_b)

    df = tbl.to_pandas()
    rows_a = df[df["id"] == str(rec_a.id)]
    assert len(rows_a) == 1
    row_a_post = rows_a.iloc[0]
    assert bool(row_a_post["schema_bypass"]) is False, (
        "schema_bypass was unexpectedly toggled True on a SKIP branch; "
        "the centroid probe must only fire on GateAction.INSERT"
    )

@pytest.mark.parametrize(
    "env_var, bad_value",
    [
        ("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "1.5"),
        ("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "not-a-float"),
        ("IAI_MCP_LABILE_WINDOW_SEC", "-1"),
        ("IAI_MCP_LABILE_WINDOW_SEC", "0"),
        ("IAI_MCP_LABILE_WINDOW_SEC", "not-an-int"),
        ("IAI_MCP_RECONSOLIDATION_TIER1", "maybe"),
        ("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "-0.1"),
        ("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "1.1"),
        ("IAI_MCP_RECONSOLIDATION_DRY_RUN", "banana"),
    ],
)
def test_R5_invalid_env_var_raises_ValueError_naming_the_var(
    monkeypatch: pytest.MonkeyPatch, env_var: str, bad_value: str,
) -> None:
    monkeypatch.setenv(env_var, bad_value)
    with pytest.raises(ValueError, match=env_var):
        _load_reconsolidation_config()

def test_R6_dry_run_skips_all_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "true")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "true")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "0.5")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")

    store = _make_store(tmp_path)
    rec = _make_record(embed_dim=store._embed_dim, literal="dry-run record")
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    tbl.update(
        where=f"id = '{str(rec.id)}'",
        values={"labile_until": future},
    )

    def _stub_batched(items: Any, *args: Any, **kwargs: Any) -> dict:
        return {rid: 0.9 for rid, _surface in items}

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _stub_batched,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True
    assert payload["records_scanned"] >= 1
    assert payload["dry_run"] is True

    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1
    body = events[0]["data"]
    assert body["dry_run_mode"] is True
    assert body["critic_calls"] >= 1
    assert body["records_reconsolidated"] >= 1

    fetched = store.get(rec.id)
    assert fetched is not None
    for entry in fetched.provenance:
        assert "reconsolidated_at" not in entry, (
            f"dry-run must NOT write provenance, got entry={entry!r}"
        )

def test_R6_dry_run_skips_schema_bypass_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "true")
    monkeypatch.setenv("IAI_MCP_SCHEMA_BYPASS_COS_THRESHOLD", "0.85")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    centroid = [1.0] + [0.0] * (embed_dim - 1)
    _stub_centroids(monkeypatch, {uuid.uuid4(): centroid})

    rec = _make_record(
        embed_dim=embed_dim,
        literal="aligned but dry-run",
        embedding=list(centroid),
    )
    store.insert(rec)

    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.to_pandas()
    row = df[df["id"] == str(rec.id)].iloc[0]
    assert bool(row["schema_bypass"]) is False, (
        "dry-run schema-bypass must NOT mutate the row"
    )

    events = query_events(store, kind="schema_bypass_pass", limit=10)
    assert len(events) >= 1
    body = events[0]["data"]
    assert body["dry_run_mode"] is True
    assert body["tagged"] is False
    assert float(body["max_cos"]) >= 0.85

def test_PROMPT_TEMPLATE_contains_required_slots() -> None:
    assert "{literal_surface}" in PROMPT_TEMPLATE
    assert "{current_summary}" in PROMPT_TEMPLATE
    formatted = PROMPT_TEMPLATE.format(
        literal_surface="x", current_summary="y",
    )
    assert isinstance(formatted, str) and len(formatted) > 0
    assert "{literal_surface}" not in formatted
    assert "{current_summary}" not in formatted
    assert "{" not in formatted and "}" not in formatted

def test_call_critic_tier0_fallback_returns_0_when_gate_denies(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    err = call_critic(
        "memory text",
        "",
        store,
        llm_enabled=False,
        has_api_key=False,
    )
    assert err == 0.0

if __name__ == "__main__":  # pragma: no cover -- direct-run convenience
    raise SystemExit(pytest.main([__file__, "-v"]))
