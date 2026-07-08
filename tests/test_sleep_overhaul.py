from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.ashby_step import (
    BreachInfo,
    EssentialVariableTracker,
    TopologySnapshot,
)
from iai_mcp.daemon import (
    _load_sleep_overhaul_config,
)
from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import (
    LifecycleStateRecord,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.lilli.cycle.sleep_pipeline import (
    MAX_PAIRS_PER_CLUSTER,
    STEP_PHASE,
    SleepPhase,
    SleepPipeline,
    SleepStep,
)
from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

@pytest.fixture(autouse=True)
def _isolate_iai_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    for var in (
        "IAI_MCP_RICH_CLUB_RATIO_FLOOR",
        "IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO",
        "IAI_MCP_EDGE_DENSITY_FLOOR",
        "IAI_MCP_CLUSTER_WINDOW_SEC",
        "IAI_MCP_CRISIS_DROP_QUARTILE",
        "IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT",
        "IAI_MCP_SLEEP_OVERHAUL_DRY_RUN",
    ):
        monkeypatch.delenv(var, raising=False)

def _make_record(
    *,
    embed_dim: int,
    literal_surface: str = "alice prefers tea over coffee",
    last_reviewed: datetime | None = None,
    community_id: uuid.UUID | None = None,
) -> MemoryRecord:
    rng = random.Random(hash(literal_surface))
    raw = [rng.gauss(0.0, 1.0) for _ in range(embed_dim)]
    mag = math.sqrt(sum(x * x for x in raw))
    embedding = [x / mag for x in raw] if mag > 0 else raw
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=literal_surface,
        aaak_index="",
        embedding=embedding,
        community_id=community_id,
        centrality=0.5,
        detail_level=1,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=last_reviewed,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        language="en",
    )

def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )

def test_r1_step_phase_mapping() -> None:
    assert SleepPhase.NREM is not None
    assert SleepPhase.REM is not None
    assert STEP_PHASE[SleepStep.SCHEMA_MINE] == SleepPhase.NREM
    assert STEP_PHASE[SleepStep.DREAM_DECAY] == SleepPhase.REM
    assert set(STEP_PHASE.keys()) == set(SleepStep)

    nrem_steps = {s for s, p in STEP_PHASE.items() if p == SleepPhase.NREM}
    rem_steps = {s for s, p in STEP_PHASE.items() if p == SleepPhase.REM}
    assert nrem_steps == {
        SleepStep.SCHEMA_MINE,
        SleepStep.KNOB_TUNE,
        SleepStep.OPTIMIZE_HIPPO,
        SleepStep.HIPPO_CLEANUP,
    }
    assert rem_steps == {
        SleepStep.DREAM_DECAY,
        SleepStep.ERASURE_AGENT,
        SleepStep.CLUSTER_REPLAY,
        SleepStep.RECONSOLIDATION,
        SleepStep.USER_MODEL_UPDATE,
        SleepStep.DMN_REFLECTION,
        SleepStep.CRISIS_RECLUSTER,
        SleepStep.CLUSTER_SUMMARY,
        SleepStep.RECALL_INDEX_REBUILD,
        SleepStep.CURIOSITY_DRAIN,
    }

def test_r2_step_order_nrem_before_rem() -> None:
    order = SleepPipeline._STEP_ORDER
    assert len(order) == 14
    assert order[-1] == SleepStep.RECALL_INDEX_REBUILD

    nrem_positions = [
        order.index(s)
        for s in (
            SleepStep.SCHEMA_MINE,
            SleepStep.KNOB_TUNE,
            SleepStep.OPTIMIZE_HIPPO,
            SleepStep.HIPPO_CLEANUP,
        )
    ]
    rem_positions = [
        order.index(s)
        for s in (
            SleepStep.DREAM_DECAY,
            SleepStep.ERASURE_AGENT,
            SleepStep.CLUSTER_REPLAY,
            SleepStep.RECONSOLIDATION,
            SleepStep.USER_MODEL_UPDATE,
            SleepStep.DMN_REFLECTION,
            SleepStep.CRISIS_RECLUSTER,
            SleepStep.CLUSTER_SUMMARY,
            SleepStep.RECALL_INDEX_REBUILD,
            SleepStep.CURIOSITY_DRAIN,
        )
    ]
    assert max(nrem_positions) < min(rem_positions)

    assert SleepStep.CLUSTER_REPLAY.value == 7
    assert SleepStep.CRISIS_RECLUSTER.value == 8
    assert SleepStep.RECONSOLIDATION.value == 9
    assert order.index(SleepStep.RECONSOLIDATION) == (
        order.index(SleepStep.CLUSTER_REPLAY) + 1
    )
    assert SleepStep.USER_MODEL_UPDATE.value == 10
    assert order.index(SleepStep.USER_MODEL_UPDATE) == (
        order.index(SleepStep.RECONSOLIDATION) + 1
    )
    assert SleepStep.DMN_REFLECTION.value == 11
    assert order.index(SleepStep.DMN_REFLECTION) == (
        order.index(SleepStep.USER_MODEL_UPDATE) + 1
    )
    assert order.index(SleepStep.CRISIS_RECLUSTER) == len(order) - 4
    assert SleepStep.CLUSTER_SUMMARY.value == 12
    assert SleepStep.RECALL_INDEX_REBUILD.value == 13
    assert SleepStep.CURIOSITY_DRAIN.value == 14
    assert order[-3] == SleepStep.CLUSTER_SUMMARY
    assert order[-2] == SleepStep.CURIOSITY_DRAIN
    assert order[-1] == SleepStep.RECALL_INDEX_REBUILD

def test_r3_cluster_replay_batches_intra_cluster_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CLUSTER_WINDOW_SEC", "300")
    monkeypatch.setenv("IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT", "0.05")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    tbl = store.db.open_table(RECORDS_TABLE)

    now = datetime.now(timezone.utc)
    cluster_offsets = [
        [-30, -60, -90, -120],
        [-430, -460, -490],
        [-830, -860, -890],
    ]
    record_ids: list[uuid.UUID] = []
    for cluster in cluster_offsets:
        for off in cluster:
            rec = _make_record(
                embed_dim=embed_dim,
                literal_surface=f"alice record at {off}s",
            )
            store.insert(rec)
            ts = now + timedelta(seconds=off)
            tbl.update(
                where=f"id = '{str(rec.id)}'",
                values={"last_reviewed": ts},
            )
            record_ids.append(rec.id)
    assert len(record_ids) == 10

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_cluster_replay(interrupt_check=None)
    assert done is True
    assert payload["clusters_replayed"] == 3
    assert payload["dry_run"] is False

    events = query_events(store, kind="cluster_replay_pass", limit=5)
    assert len(events) >= 1
    body = events[0]["data"]
    assert body["clusters_replayed"] == 3
    assert body["window_sec"] == 300
    assert body["lookback_windows"] == 5
    assert body["dry_run_mode"] is False

    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    cluster_edges = edges[edges["edge_type"] == "hebbian_cluster_replay"]
    assert len(cluster_edges) > 0, (
        "non-dry-run CLUSTER_REPLAY must create hebbian_cluster_replay edges"
    )

    assert "max_pairs_per_cluster_applied" in body
    assert body["max_pairs_per_cluster_applied"] == 0
    assert MAX_PAIRS_PER_CLUSTER == 100

def test_r4_essential_variable_tracker_detects_rich_club_breach() -> None:
    class _Cfg:
        rich_club_ratio_floor = 0.05
        community_count_ceiling_ratio = 0.9
        edge_density_floor = 0.001

    tracker = EssentialVariableTracker(_Cfg())

    breach_snapshot = TopologySnapshot(
        rich_club_ratio=0.01,
        community_count=500,
        edge_density=0.01,
        total_nodes=1000,
    )
    breaches = tracker.check(breach_snapshot)
    assert set(breaches.keys()) == {
        "rich_club_ratio",
        "community_count",
        "edge_density",
    }
    rc = breaches["rich_club_ratio"]
    assert isinstance(rc, BreachInfo)
    assert rc.direction == "floor_breach"
    assert rc.observed_value == pytest.approx(0.01)
    assert rc.threshold == pytest.approx(0.05)
    assert breaches["community_count"] is None
    assert breaches["edge_density"] is None

    healthy = TopologySnapshot(
        rich_club_ratio=0.5,
        community_count=10,
        edge_density=0.5,
        total_nodes=1000,
    )
    healthy_result = tracker.check(healthy)
    assert all(v is None for v in healthy_result.values())

    empty = TopologySnapshot(
        rich_club_ratio=0.0,
        community_count=0,
        edge_density=0.0,
        total_nodes=0,
    )
    empty_result = tracker.check(empty)
    assert all(v is None for v in empty_result.values())

def test_r5_crisis_recluster_conditional_on_crisis_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    lifecycle_path = tmp_path / "lifecycle.json"

    state: LifecycleStateRecord = default_state()
    state["crisis_mode"] = False
    save_state(state, lifecycle_path)
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline._step_crisis_recluster(interrupt_check=None)
    assert done is True
    assert payload["communities_dropped"] == 0
    events_a = query_events(store, kind="crisis_recluster_pass", limit=10)
    assert len(events_a) == 0, (
        f"crisis_mode=False path must NOT emit crisis_recluster_pass, "
        f"got {len(events_a)} event(s)"
    )

    tbl = store.db.open_table(RECORDS_TABLE)
    for i in range(100):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice rec {i}",
        )
        store.insert(rec)
        tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    state = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)

    pipeline_b = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    done, payload = pipeline_b._step_crisis_recluster(interrupt_check=None)
    assert done is True
    assert payload["communities_dropped"] == 25, (
        f"expected 25 communities dropped (25% of 100), got {payload}"
    )

    final_state = load_state(lifecycle_path)
    assert final_state["crisis_mode"] is False, (
        "non-dry-run CRISIS_RECLUSTER must clear crisis_mode"
    )

    events_b = query_events(store, kind="crisis_recluster_pass", limit=10)
    assert len(events_b) == 1, (
        f"expected exactly 1 crisis_recluster_pass event, got {len(events_b)}"
    )
    body = events_b[0]["data"]
    assert body["communities_dropped"] == 25
    assert body["dry_run_mode"] is False

@pytest.mark.parametrize(
    "var_name,bad_value",
    [
        ("IAI_MCP_RICH_CLUB_RATIO_FLOOR", "2.0"),
        ("IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO", "-0.1"),
        ("IAI_MCP_EDGE_DENSITY_FLOOR", "not_a_float"),
        ("IAI_MCP_CLUSTER_WINDOW_SEC", "0"),
        ("IAI_MCP_CRISIS_DROP_QUARTILE", "1.0"),
        ("IAI_MCP_CLUSTER_REPLAY_INITIAL_WEIGHT", "5.0"),
        ("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "maybe"),
    ],
)
def test_r6_env_var_fail_loud_naming(
    monkeypatch: pytest.MonkeyPatch,
    var_name: str,
    bad_value: str,
) -> None:
    monkeypatch.setenv(var_name, bad_value)
    with pytest.raises(ValueError) as exc_info:
        _load_sleep_overhaul_config()
    assert var_name in str(exc_info.value), (
        f"ValueError message {str(exc_info.value)!r} must contain {var_name!r}"
    )

def test_r7_dry_run_no_mutation_all_three_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "true")
    monkeypatch.setenv("IAI_MCP_CLUSTER_WINDOW_SEC", "300")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    records_tbl = store.db.open_table(RECORDS_TABLE)

    now = datetime.now(timezone.utc)
    for off in (-30, -60, -90, -120):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice rec {off}",
        )
        store.insert(rec)
        records_tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"last_reviewed": now + timedelta(seconds=off)},
        )

    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    pipeline._step_cluster_replay(interrupt_check=None)

    events1 = query_events(store, kind="cluster_replay_pass", limit=5)
    assert events1, "cluster_replay_pass event must still emit in dry_run"
    body1 = events1[0]["data"]
    assert body1["dry_run_mode"] is True
    assert body1["clusters_replayed"] == 1, (
        f"4 records in one window -> 1 cluster, got {body1}"
    )
    edges_after_p1 = store.db.open_table(EDGES_TABLE).to_pandas()
    if not edges_after_p1.empty:
        cluster_edges = edges_after_p1[
            edges_after_p1["edge_type"] == "hebbian_cluster_replay"
        ]
        assert len(cluster_edges) == 0, (
            "dry_run must not write hebbian_cluster_replay edges"
        )

    monkeypatch.setenv("IAI_MCP_RICH_CLUB_RATIO_FLOOR", "0.99")
    pipeline_p2 = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    try:
        pipeline_p2._run_essential_variable_tracker_hook()
    except Exception:
        pass
    events2 = query_events(store, kind="essential_variable_breach", limit=10)
    for e in events2:
        body2 = e["data"]
        assert body2["dry_run_mode"] is True
        assert body2["crisis_mode_set"] is False, (
            "dry_run breach event must report crisis_mode_set=False"
        )
    final_state = load_state(lifecycle_path)
    assert final_state["crisis_mode"] is False, (
        "dry_run must not flip crisis_mode"
    )

    state: LifecycleStateRecord = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    for i in range(100):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice c-rec {i}",
        )
        store.insert(rec)
        records_tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    pipeline_p3 = SleepPipeline(
        store=store,
        lifecycle_state_path=lifecycle_path,
    )
    pipeline_p3._step_crisis_recluster(interrupt_check=None)
    events3 = query_events(store, kind="crisis_recluster_pass", limit=5)
    assert events3, "crisis_recluster_pass must still emit in dry_run"
    body3 = events3[0]["data"]
    assert body3["dry_run_mode"] is True
    assert body3["records_reassigned"] == 0, (
        "dry_run must not reassign community_id on any record"
    )

    final_state_p3 = load_state(lifecycle_path)
    assert final_state_p3["crisis_mode"] is True, (
        "dry_run must not clear crisis_mode"
    )
