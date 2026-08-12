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
        "IAI_MCP_AVG_DEGREE_FLOOR",
        "IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR",
        "IAI_MCP_EV_MIN_NODES",
        "IAI_MCP_EV_ARM_AFTER_N",
        "IAI_MCP_EV_DISARM_AFTER_N",
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
        SleepStep.EMBEDDING_INTEGRITY,
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
        SleepStep.ENTITY_LINK,
        SleepStep.CURIOSITY_MINE,
    }

def test_r2_step_order_nrem_before_rem() -> None:
    order = SleepPipeline._STEP_ORDER
    assert len(order) == 16
    assert order[-1] == SleepStep.EMBEDDING_INTEGRITY
    assert order[-2] == SleepStep.CURIOSITY_MINE
    assert order[-3] == SleepStep.ENTITY_LINK

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
    assert order.index(SleepStep.CRISIS_RECLUSTER) == len(order) - 6
    assert SleepStep.CLUSTER_SUMMARY.value == 12
    assert SleepStep.RECALL_INDEX_REBUILD.value == 13
    assert SleepStep.ENTITY_LINK.value == 14
    assert SleepStep.CURIOSITY_MINE.value == 15
    assert order[-5] == SleepStep.CLUSTER_SUMMARY
    assert order[-4] == SleepStep.RECALL_INDEX_REBUILD
    assert order[-3] == SleepStep.ENTITY_LINK
    assert order[-2] == SleepStep.CURIOSITY_MINE
    assert order[-1] == SleepStep.EMBEDDING_INTEGRITY

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

def test_r4_essential_variable_tracker_reports_floor_breaches_and_clears_below_min_scale() -> None:
    class _Cfg:
        rich_club_ratio_floor = 0.05
        community_count_ceiling_ratio = 0.9
        edge_density_floor = 0.001
        avg_degree_floor = 2.0
        giant_component_fraction_floor = 0.5
        ev_min_nodes = 100

    tracker = EssentialVariableTracker(_Cfg())

    breach_snapshot = TopologySnapshot(
        rich_club_ratio=0.01,
        community_count=500,
        edge_density=0.01,
        total_nodes=1000,
        avg_degree=4.0,
        giant_component_fraction=0.9,
    )
    breaches = tracker.check(breach_snapshot)
    assert set(breaches.keys()) == {
        "rich_club_ratio",
        "community_count",
        "edge_density",
        "avg_degree",
        "giant_component_fraction",
    }
    rc = breaches["rich_club_ratio"]
    assert isinstance(rc, BreachInfo)
    assert rc.direction == "floor_breach"
    assert rc.observed_value == pytest.approx(0.01)
    assert rc.threshold == pytest.approx(0.05)
    assert breaches["community_count"] is None
    assert breaches["edge_density"] is None
    assert breaches["avg_degree"] is None
    assert breaches["giant_component_fraction"] is None

    healthy = TopologySnapshot(
        rich_club_ratio=0.5,
        community_count=10,
        edge_density=0.5,
        total_nodes=1000,
        avg_degree=4.0,
        giant_component_fraction=0.9,
    )
    healthy_result = tracker.check(healthy)
    assert all(v is None for v in healthy_result.values())

    empty = TopologySnapshot(
        rich_club_ratio=0.0,
        community_count=0,
        edge_density=0.0,
        total_nodes=0,
        avg_degree=0.0,
        giant_component_fraction=0.0,
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

def test_crisis_recluster_updates_are_batched_and_interruptible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The heavy path never issues one update() statement per record or per
    community: the drop phase nulls its members with a direct bound-parameter
    IN-UPDATE per batch, and the reassignment moves rows through
    update_many_by_id (distinct-per-row payloads under one commit per chunk).
    Interrupt checks between batches keep the step deferrable."""
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    lifecycle_path = tmp_path / "lifecycle.json"

    tbl = store.db.open_table(RECORDS_TABLE)
    n_records = 120
    for i in range(n_records):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice batched rec {i}",
        )
        store.insert(rec)
        tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    state = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    from iai_mcp.hippo import _table as _table_mod

    update_wheres: list[str] = []
    orig_update = _table_mod.HippoTable.update

    def _counting_update(self, *args, **kwargs):
        update_wheres.append(kwargs.get("where") or (args[0] if args else ""))
        return orig_update(self, *args, **kwargs)

    monkeypatch.setattr(_table_mod.HippoTable, "update", _counting_update)

    many_call_cids: list[list] = []
    orig_many = _table_mod.HippoTable.update_many_by_id

    def _counting_many(self, rows):
        many_call_cids.append([v.get("community_id") for (_rid, v) in rows])
        return orig_many(self, rows)

    monkeypatch.setattr(
        _table_mod.HippoTable, "update_many_by_id", _counting_many
    )

    done, payload = pipeline._step_crisis_recluster(interrupt_check=None)
    assert done is True

    # The heavy path issues no per-record update() statement: the drop phase
    # nulls members with a direct IN-UPDATE and the reassignment moves through
    # chunked update_many_by_id calls.
    assert len(update_wheres) == 0, (
        f"per-statement update() is back in the heavy path: {update_wheres}"
    )
    # The clear phase no longer routes NULL-clears through the id-point-lookup
    # seam: every update_many_by_id call now belongs to the reassignment and
    # assigns a real community id, never None.
    null_clears = [
        cid for call in many_call_cids for cid in call if cid is None
    ]
    assert not null_clears, (
        "clear phase must null members via a direct IN-UPDATE, not the seam"
    )
    # The reassignment still rides update_many_by_id with many ids per call.
    assert many_call_cids, "reassignment must route through update_many_by_id"
    assert all(len(call) > 1 for call in many_call_cids), (
        f"bulk reassignment calls degenerated to single rows: "
        f"{[len(c) for c in many_call_cids]}"
    )
    assert sum(len(call) for call in many_call_cids) >= n_records // 4

    # A tripped interrupt inside the DROP phase defers: re-fragment the
    # communities so the quartile is non-empty; check #2 is the clear batch.
    tbl = store.db.open_table(RECORDS_TABLE)
    rows = tbl._conn.execute("SELECT id FROM records").fetchall()
    tbl.update_many_by_id(
        [(str(rid), {"community_id": str(uuid.uuid4())}) for (rid,) in rows]
    )

    state = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    pipeline_c = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    calls = {"n": 0}

    def _trip_after_first(_probe=None):
        calls["n"] += 1
        return calls["n"] > 1

    done_c, _ = pipeline_c._step_crisis_recluster(
        interrupt_check=_trip_after_first,
    )
    assert done_c is False, (
        "an interrupt landing at the drop clear batch must defer the step"
    )

    # And in the REASSIGNMENT loop: checks run start(1), drop clear(2),
    # first reassignment chunk(3) — trip the third.
    state = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    pipeline_d = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)
    calls_d = {"n": 0}

    def _trip_late(_probe=None):
        calls_d["n"] += 1
        return calls_d["n"] > 2

    done_d, _ = pipeline_d._step_crisis_recluster(interrupt_check=_trip_late)
    assert done_d is False, (
        "an interrupt landing between reassignment chunks must defer"
    )


def test_crisis_recluster_clear_phase_nulls_dropped_members(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clear phase nulls each dropped community's members with a direct
    bound-parameter IN-UPDATE (not the id-point-lookup seam). With the
    reassignment disabled, the nulls are observable: exactly the dropped
    quartile's members end with community_id IS NULL, and update_many_by_id is
    never called."""
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    lifecycle_path = tmp_path / "lifecycle.json"

    tbl = store.db.open_table(RECORDS_TABLE)
    n_records = 120
    for i in range(n_records):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice clear {i}",
        )
        store.insert(rec)
        tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    def _count_null() -> int:
        with store.db._conn_lock:
            got = tbl._conn.execute(
                "SELECT COUNT(*) FROM records WHERE community_id IS NULL"
            ).fetchall()
        return int(got[0][0])

    null_before = _count_null()

    # Disable the reassignment so the clear-phase nulls are not overwritten.
    import iai_mcp.runtime_graph_cache as _rgc

    def _no_reassign(*_a, **_k):
        raise RuntimeError("reassignment disabled for the clear-phase probe")

    monkeypatch.setattr(_rgc, "compute_assignment_in_child", _no_reassign)

    # The clear phase must not touch the id-point-lookup seam.
    from iai_mcp.hippo import _table as _table_mod

    many_calls: list[int] = []
    orig_many = _table_mod.HippoTable.update_many_by_id

    def _counting_many(self, rows):
        many_calls.append(len(rows))
        return orig_many(self, rows)

    monkeypatch.setattr(
        _table_mod.HippoTable, "update_many_by_id", _counting_many
    )

    state = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    done, payload = pipeline._step_crisis_recluster(interrupt_check=None)
    assert done is True

    dropped = int(payload["communities_dropped"])
    assert dropped == n_records // 4, (
        f"the smallest quartile is {n_records // 4} single-member communities, "
        f"got {dropped}"
    )
    null_after = _count_null()
    assert null_after - null_before == dropped, (
        "the clear phase must null exactly the dropped communities' members "
        f"({dropped}); saw a delta of {null_after - null_before}"
    )
    assert many_calls == [], (
        "the clear phase must null via a direct IN-UPDATE, never "
        f"update_many_by_id: {many_calls}"
    )


def test_update_many_by_id_one_txn_and_matched_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    tbl = store.db.open_table(RECORDS_TABLE)

    recs = []
    for i in range(3):
        rec = _make_record(
            embed_dim=embed_dim,
            literal_surface=f"alice bulk {i}",
        )
        store.insert(rec)
        recs.append(rec)

    import iai_mcp.hippo as hippo_pkg

    txn_calls = {"n": 0}
    orig_txn = hippo_pkg._txn

    def _counting_txn(conn):
        txn_calls["n"] += 1
        return orig_txn(conn)

    monkeypatch.setattr(hippo_pkg, "_txn", _counting_txn)

    cid = str(uuid.uuid4())
    matched = tbl.update_many_by_id(
        [(str(r.id), {"community_id": cid}) for r in recs[:2]]
        + [(str(uuid.uuid4()), {"community_id": cid})]
    )
    assert matched == 2, (
        f"expected 2 matched rows (third id does not exist), got {matched}"
    )
    assert txn_calls["n"] == 1, (
        f"the whole batch must ride ONE transaction/commit, saw {txn_calls['n']}"
    )

    with store.db._conn_lock:
        got = tbl._conn.execute(
            "SELECT COUNT(*) FROM records WHERE community_id = ?", (cid,)
        ).fetchall()
    assert int(got[0][0]) == 2


def test_update_many_by_id_validates_before_writing(
    tmp_path: Path,
) -> None:
    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    tbl = store.db.open_table(RECORDS_TABLE)

    rec = _make_record(embed_dim=embed_dim, literal_surface="alice guard")
    store.insert(rec)

    cid = str(uuid.uuid4())
    with pytest.raises(ValueError):
        tbl.update_many_by_id(
            [
                (str(rec.id), {"community_id": cid}),
                (str(rec.id), {"no_such_column": 1}),
            ]
        )
    with pytest.raises(ValueError):
        tbl.update_many_by_id(
            [(str(rec.id), {"embedding": [0.0] * embed_dim})]
        )
    # Prepare-time validation rejects the batch before any statement runs.
    with store.db._conn_lock:
        got = tbl._conn.execute(
            "SELECT COUNT(*) FROM records WHERE community_id = ?", (cid,)
        ).fetchall()
    assert int(got[0][0]) == 0

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
        ("IAI_MCP_AVG_DEGREE_FLOOR", "not_a_float"),
        ("IAI_MCP_GIANT_COMPONENT_FRACTION_FLOOR", "1.1"),
        ("IAI_MCP_EV_MIN_NODES", "0"),
        ("IAI_MCP_EV_ARM_AFTER_N", "101"),
        ("IAI_MCP_EV_DISARM_AFTER_N", "0"),
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
    # This fixture only seeds 4 records; the tracker's default ev_min_nodes=100
    # guard would otherwise report all-clear vacuously and the breach/dry-run
    # event-path assertions below would pass on nothing.
    monkeypatch.setenv("IAI_MCP_EV_MIN_NODES", "1")

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
    assert events2, (
        "the forced rich_club floor must actually drive a breach through "
        "the dry-run event path (ev_min_nodes=1 keeps this fixture's 4 "
        "records from vacuously passing the min-scale guard)"
    )
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
