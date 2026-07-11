from __future__ import annotations

import math
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import default_state, load_state, save_state
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord

N_NODES = 220


@pytest.fixture(autouse=True)
def _isolate_iai_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "iai-mcp-store"))
    monkeypatch.delenv("IAI_MCP_EMBED_MODEL", raising=False)
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    for var in (
        "IAI_MCP_RICH_CLUB_RATIO_FLOOR",
        "IAI_MCP_COMMUNITY_COUNT_CEILING_RATIO",
        "IAI_MCP_EDGE_DENSITY_FLOOR",
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
    literal_surface: str,
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


def _make_store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(
        path=str(tmp_path / "iai-mcp-store"),
        user_id="alice",
        read_consistency_interval=timedelta(seconds=0),
    )


def _edge_row(*, src: str, dst: str, weight: float = 1.0) -> dict:
    return {
        "src": src,
        "dst": dst,
        "edge_type": "hebbian",
        "weight": weight,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def _insert_edges(store: MemoryStore, rows: list[dict]) -> None:
    if not rows:
        return
    tbl = store.db.open_table(EDGES_TABLE)
    tbl.merge_insert(["src", "dst", "edge_type"]).when_matched_update_all().execute(rows)


def _seed_records(store: MemoryStore, n: int, *, prefix: str) -> list[str]:
    embed_dim = store._embed_dim
    ids: list[str] = []
    for i in range(n):
        rec = _make_record(
            embed_dim=embed_dim, literal_surface=f"{prefix} rec {i}",
        )
        store.insert(rec)
        ids.append(str(rec.id))
    return ids


def _seed_ring_plus_chords(store: MemoryStore, n: int, *, prefix: str) -> list[str]:
    """Connected sparse topology: ring + a few chords, avg_degree ~4, one component."""
    ids = _seed_records(store, n, prefix=prefix)
    rows: list[dict] = []
    for i in range(n):
        rows.append(_edge_row(src=ids[i], dst=ids[(i + 1) % n]))
    rng = random.Random(42)
    for i in range(n):
        j = rng.randrange(n)
        if j != i:
            rows.append(_edge_row(src=ids[i], dst=ids[j]))
    _insert_edges(store, rows)
    return ids


def _seed_ring_with_light_chords(
    store: MemoryStore, n: int, *, prefix: str, n_chords: int = 40,
) -> list[str]:
    """Ring + a modest number of extra chords: enough redundancy that the
    CORRECT (tombstone/orphan-excluding) avg_degree survives a ~30% node
    removal above the 2.0 floor, while the INCORRECT phantom-inflated
    computation (dividing the same surviving edges by the full pre-removal
    node count) drops well below it -- a genuine RED/GREEN discriminator
    for the tombstone-filter + no-phantom-nodes fix, unlike a denser
    topology whose redundancy tolerates the bug too."""
    ids = _seed_records(store, n, prefix=prefix)
    edge_pairs: set[tuple[int, int]] = set()
    for i in range(n):
        edge_pairs.add(tuple(sorted((i, (i + 1) % n))))
    rng = random.Random(42)
    added = 0
    while added < n_chords:
        i = rng.randrange(n)
        j = rng.randrange(n)
        if i == j:
            continue
        pair = tuple(sorted((i, j)))
        if pair in edge_pairs:
            continue
        edge_pairs.add(pair)
        added += 1
    rows = [_edge_row(src=ids[a], dst=ids[b]) for a, b in edge_pairs]
    _insert_edges(store, rows)
    return ids


def _seed_shattered_islands(store: MemoryStore, n: int, *, prefix: str) -> list[str]:
    """Many small disconnected islands (pairs) -- giant fraction << 0.5."""
    ids = _seed_records(store, n, prefix=prefix)
    rows: list[dict] = []
    for i in range(0, n - 1, 2):
        rows.append(_edge_row(src=ids[i], dst=ids[i + 1]))
    _insert_edges(store, rows)
    return ids


def _make_pipeline(store: MemoryStore, lifecycle_path: Path) -> SleepPipeline:
    return SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)


def _gating_breach_events(store: MemoryStore) -> list[dict]:
    events = query_events(store, kind="essential_variable_breach", limit=50)
    return [e for e in events if e["data"].get("gates_crisis") is True]


class TestHealthyAtScale:
    def test_healthy_ring_topology_never_arms_crisis(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _seed_ring_plus_chords(store, N_NODES, prefix="healthy")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        for _ in range(4):
            pipeline = _make_pipeline(store, lifecycle_path)
            pipeline._run_essential_variable_tracker_hook()

        final_state = load_state(lifecycle_path)
        assert final_state["crisis_mode"] is False

        gating_events = _gating_breach_events(store)
        assert gating_events == [], (
            f"healthy topology must raise zero gating breaches, got "
            f"{gating_events}"
        )


class TestTombstonedAndOrphanEdgesExcluded:
    def test_orphan_edges_to_nonexistent_ids_do_not_fabricate_phantom_nodes(
        self, tmp_path: Path,
    ) -> None:
        """An edge referencing an id that has NO corresponding record row at
        all (hard-deleted, or a stale reference from before a record was
        purged) must not fabricate a phantom graph node -- add_edge() would
        otherwise silently create an adjacency entry for an id that was
        never add_node()'d, inflating total_nodes against a numerator (live
        edges) that can never reference it, deflating avg_degree and
        skewing giant_component_fraction. Measured on the real 38k-record
        prod corpus copy: total_nodes read 41836 (vs 38641 actual records)
        purely from this phantom-node effect, before any HIPPO_CLEANUP
        orphan-edge sweep runs.
        """
        store = _make_store(tmp_path)
        ids = _seed_ring_with_light_chords(store, N_NODES, prefix="orphan")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        # Edges pointing at ids with no record row at all (never inserted) --
        # the dominant real-corpus mechanism (measured there: ~39k of ~91k
        # edges referenced fully-missing ids, vs ~16k referencing merely
        # tombstoned-but-present rows).
        orphan_edge_count = int(N_NODES * 0.3)
        orphan_rows = [
            _edge_row(src=ids[i], dst=str(uuid.uuid4()))
            for i in range(orphan_edge_count)
        ]
        _insert_edges(store, orphan_rows)

        pipeline = _make_pipeline(store, lifecycle_path)
        pipeline._run_essential_variable_tracker_hook()

        events = query_events(
            store, kind="essential_variable_breach", limit=50,
        )
        for e in events:
            assert e["data"]["total_nodes"] == N_NODES, (
                f"orphan edges to nonexistent ids must not fabricate "
                f"phantom graph nodes -- total_nodes must equal the real "
                f"record count ({N_NODES}), got {e['data']['total_nodes']}"
            )
        gating_events = _gating_breach_events(store)
        assert gating_events == [], (
            f"orphan edges to nonexistent ids must not skew avg_degree/"
            f"giant_component_fraction on an otherwise healthy graph, got "
            f"{gating_events}"
        )
        assert load_state(lifecycle_path)["crisis_mode"] is False

    def test_tombstoned_records_excluded_from_graph_nodes(
        self, tmp_path: Path,
    ) -> None:
        """A tombstoned record must not count as a graph node -- it is a
        dead memory, not part of the live topology HIPPO_CLEANUP maintains.
        """
        store = _make_store(tmp_path)
        ids = _seed_ring_with_light_chords(store, N_NODES, prefix="tomb")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        tbl = store.db.open_table(RECORDS_TABLE)
        tombstoned_ids = set(ids[: int(N_NODES * 0.3)])
        for rid in tombstoned_ids:
            tbl.update(
                where=f"id = '{rid}'",
                values={
                    "tombstoned_at": datetime.now(timezone.utc).isoformat(),
                },
            )

        pipeline = _make_pipeline(store, lifecycle_path)
        pipeline._run_essential_variable_tracker_hook()

        events = query_events(
            store, kind="essential_variable_breach", limit=50,
        )
        assert events, (
            "expected at least one breach event (rich_club_ratio is "
            "expected to breach as telemetry-only) to carry the "
            "total_nodes assertion"
        )
        for e in events:
            assert e["data"]["total_nodes"] == N_NODES - len(tombstoned_ids), (
                "total_nodes must reflect only live (non-tombstoned) "
                f"records, got {e['data']['total_nodes']}"
            )
        assert load_state(lifecycle_path)["crisis_mode"] is False


class TestFragmentedArmsAfterHysteresis:
    def test_fragmented_topology_arms_at_exactly_arm_after_n(
        self, tmp_path: Path,
    ) -> None:
        store = _make_store(tmp_path)
        _seed_shattered_islands(store, N_NODES, prefix="frag")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        arm_after_n = 3
        for cycle in range(1, arm_after_n + 1):
            pipeline = _make_pipeline(store, lifecycle_path)
            pipeline._run_essential_variable_tracker_hook()
            state = load_state(lifecycle_path)
            if cycle < arm_after_n:
                assert state["crisis_mode"] is False, (
                    f"cycle {cycle}: crisis must not arm before arm_after_n"
                )
            else:
                assert state["crisis_mode"] is True, (
                    f"cycle {cycle}: crisis must arm at exactly arm_after_n"
                )

        gating_events = _gating_breach_events(store)
        assert gating_events, "fragmented topology must raise a gating breach"
        arming_events = [
            e for e in gating_events if e["data"].get("crisis_mode_set") is True
        ]
        assert arming_events, "at least one event must carry crisis_mode_set=True"
        for e in arming_events:
            assert e["data"]["gates_crisis"] is True
            assert e["data"]["variable_name"] in (
                "avg_degree", "giant_component_fraction", "community_count",
            )


class TestFlapKill:
    def test_single_bad_cycle_after_disarm_does_not_rearm(
        self, tmp_path: Path,
    ) -> None:
        store = _make_store(tmp_path)
        ids = _seed_shattered_islands(store, N_NODES, prefix="flap")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        for _ in range(3):
            pipeline = _make_pipeline(store, lifecycle_path)
            pipeline._run_essential_variable_tracker_hook()
        assert load_state(lifecycle_path)["crisis_mode"] is True

        rows: list[dict] = []
        for i in range(len(ids)):
            rows.append(_edge_row(src=ids[i], dst=ids[(i + 1) % len(ids)]))
        rng = random.Random(7)
        for i in range(len(ids)):
            j = rng.randrange(len(ids))
            if j != i:
                rows.append(_edge_row(src=ids[i], dst=ids[j]))
        _insert_edges(store, rows)

        disarm_after_n = 3
        for _ in range(disarm_after_n):
            pipeline = _make_pipeline(store, lifecycle_path)
            pipeline._run_essential_variable_tracker_hook()
        assert load_state(lifecycle_path)["crisis_mode"] is False
        assert load_state(lifecycle_path)["crisis_mode_since_ts"] is None

        tbl = store.db.open_table(EDGES_TABLE)
        edges_df = tbl.to_pandas()
        healthy_edge = (
            (edges_df["src"] == ids[0]) & (edges_df["dst"] == ids[(1) % len(ids)])
        )
        tbl.delete(where=(
            f"src = '{ids[0]}' AND dst = '{ids[(1) % len(ids)]}'"
        ))

        pipeline = _make_pipeline(store, lifecycle_path)
        pipeline._run_essential_variable_tracker_hook()
        assert load_state(lifecycle_path)["crisis_mode"] is False, (
            "a single bad cycle right after disarm must not re-arm crisis "
            "(this is the direct regression for the ~90s flap)"
        )


class TestCounterPersistenceAcrossRestarts:
    def test_hysteresis_counters_survive_new_pipeline_instance(
        self, tmp_path: Path,
    ) -> None:
        store = _make_store(tmp_path)
        _seed_shattered_islands(store, N_NODES, prefix="restart")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        pipeline_a = _make_pipeline(store, lifecycle_path)
        pipeline_a._run_essential_variable_tracker_hook()
        state_after_1 = load_state(lifecycle_path)
        assert state_after_1["essential_variable_consecutive_breaches"] == 1
        assert state_after_1["crisis_mode"] is False

        pipeline_b = _make_pipeline(store, lifecycle_path)
        pipeline_b._run_essential_variable_tracker_hook()
        state_after_2 = load_state(lifecycle_path)
        assert state_after_2["essential_variable_consecutive_breaches"] == 2, (
            "counter must continue across a brand-new SleepPipeline instance "
            "over the same lifecycle path (restart-survival)"
        )
        assert state_after_2["crisis_mode"] is False

        pipeline_c = _make_pipeline(store, lifecycle_path)
        pipeline_c._run_essential_variable_tracker_hook()
        state_after_3 = load_state(lifecycle_path)
        assert state_after_3["crisis_mode"] is True
        assert state_after_3["essential_variable_consecutive_breaches"] == 3, (
            "counters must survive the arm cycle itself, not be clobbered by "
            "the crisis-mode writer's own load-modify-save"
        )


class TestScipyFallback:
    def test_scipy_fallback_matches_native_gating_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins

        store = _make_store(tmp_path)
        _seed_shattered_islands(store, N_NODES, prefix="fallback")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        real_import = builtins.__import__

        def _blocking_import(name, *args, **kwargs):
            if name == "iai_mcp_native.graph":
                raise ImportError("native graph module blocked for test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocking_import)

        arm_after_n = 3
        for cycle in range(1, arm_after_n + 1):
            pipeline = _make_pipeline(store, lifecycle_path)
            pipeline._run_essential_variable_tracker_hook()

        assert load_state(lifecycle_path)["crisis_mode"] is True, (
            "scipy fallback must reach the identical gating outcome as the "
            "native path on the fragmented fixture"
        )


class TestGiantComponentComputationFailureObservable:
    def test_component_computation_failure_logs_warning_and_fails_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # A broken component sensor must fail-healthy (never arm crisis) AND
        # be observable at warning level so a persistent failure is not
        # silent in prod logs.
        import logging as _logging
        from iai_mcp.graph import MemoryGraph

        store = _make_store(tmp_path)
        _seed_shattered_islands(store, N_NODES, prefix="broken-sensor")
        lifecycle_path = tmp_path / "lifecycle.json"
        save_state(default_state(), lifecycle_path)

        def _raise_csr(self):
            raise RuntimeError("synthetic component-search failure")

        monkeypatch.setattr(MemoryGraph, "to_csr_arrays", _raise_csr)

        with caplog.at_level(
            _logging.WARNING,
            logger="iai_mcp.lilli.cycle.sleep_pipeline._essential_variable",
        ):
            pipeline = _make_pipeline(store, lifecycle_path)
            pipeline._run_essential_variable_tracker_hook()

        # Fail-open value preserved: even a shattered fixture must not arm
        # crisis on the giant-component axis when the sensor is broken.
        assert load_state(lifecycle_path)["crisis_mode"] is False

        warnings = [
            r for r in caplog.records
            if r.levelno >= _logging.WARNING
            and "giant_component_fraction" in r.getMessage()
        ]
        assert warnings, (
            "a broken component sensor must log at warning level so a "
            "persistent failure is observable"
        )
