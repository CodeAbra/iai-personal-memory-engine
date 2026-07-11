"""Golden-snapshot parity gates for the three column-streaming sleep steps.

These lock the EXACT output of CRISIS_RECLUSTER, RECONSOLIDATION, and
CLUSTER_REPLAY beyond the count-only assertions the existing suites carry, so a
rewrite that streams record columns (instead of materializing the full corpus)
can be proven byte-identical:

- CRISIS:          the community-partition shape after the cold re-cluster (not
                   just communities_dropped==25).
- RECONSOLIDATION: the exact set of record ids handed to the critic, and the
                   records_scanned count over all labile rows (not just >=1).
- CLUSTER_REPLAY:  the cluster-membership signature carried by the
                   hebbian_cluster_replay edges (which records grouped into the
                   same temporal window), plus the temporal_sequence edge set
                   (currently empty — see note in the cluster_replay test).

The fixtures use per-text deterministic embeddings so the snapshots reproduce
across runs. Record ids are random per run, so the cluster-membership gate keys
edges by each record's last_reviewed offset (which IS fixed by the fixture)
rather than by raw id.
"""

from __future__ import annotations

import math
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from iai_mcp.events import query_events
from iai_mcp.lifecycle_state import (
    LifecycleStateRecord,
    default_state,
    load_state,
    save_state,
)
from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
from iai_mcp.store import EDGES_TABLE, RECORDS_TABLE, MemoryStore
from iai_mcp.types import MemoryRecord


@pytest.fixture(autouse=True)
def _isolate_iai_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
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
        "IAI_MCP_RECONSOLIDATION_TIER1",
        "IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD",
        "IAI_MCP_RECONSOLIDATION_DRY_RUN",
        "IAI_MCP_LABILE_WINDOW_SEC",
    ):
        monkeypatch.delenv(var, raising=False)


def _make_record(
    *,
    embed_dim: int,
    text: str = "alice prefers tea over coffee",
    last_reviewed: datetime | None = None,
    community_id: uuid.UUID | None = None,
) -> MemoryRecord:
    # Deterministic per-text embedding so the snapshots reproduce exactly.
    rng = random.Random(hash(text))
    raw = [rng.gauss(0.0, 1.0) for _ in range(embed_dim)]
    mag = math.sqrt(sum(x * x for x in raw))
    embedding = [x / mag for x in raw] if mag > 0 else raw
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid.uuid4(),
        tier="episodic",
        literal_surface=text,
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


def _community_partition_shape(store: MemoryStore) -> list[int]:
    """Sorted multiset of community sizes after a step runs.

    Community ids are fresh uuids per run, but the partition SHAPE (how records
    group into communities) is deterministic for a fixed (node, edge) set.
    """
    tbl = store.db.open_table(RECORDS_TABLE)
    df = tbl.search().to_pandas()
    by_cid: Counter[str] = Counter()
    for _, row in df.iterrows():
        cid = row.get("community_id")
        by_cid[str(cid)] += 1
    return sorted(by_cid.values())


def _edge_offset_signature(
    store: MemoryStore,
    edge_type: str,
    rid_to_offset: dict[str, int],
) -> frozenset[frozenset[int]]:
    """Undirected membership signature for one edge type, keyed by offset.

    Pairs are made undirected (frozenset of the two endpoint offsets) so the
    signature pins cluster MEMBERSHIP — which is window-determined and stable —
    independent of intra-cluster ordering (which the pre-rewrite pandas sort
    does not fix deterministically).
    """
    edges = store.db.open_table(EDGES_TABLE).to_pandas()
    if edges.empty or "edge_type" not in edges.columns:
        return frozenset()
    sel = edges[edges["edge_type"] == edge_type]
    return frozenset(
        frozenset((rid_to_offset[str(row["src"])], rid_to_offset[str(row["dst"])]))
        for _, row in sel.iterrows()
    )


def test_crisis_recluster_community_partition_is_byte_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_SLEEP_OVERHAUL_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_CRISIS_DROP_QUARTILE", "0.25")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim
    tbl = store.db.open_table(RECORDS_TABLE)

    # 100 records, each with a unique community_id (100 groups of size 1).
    for i in range(100):
        rec = _make_record(embed_dim=embed_dim, text=f"alice rec {i}")
        store.insert(rec)
        tbl.update(
            where=f"id = '{str(rec.id)}'",
            values={"community_id": str(uuid.uuid4())},
        )

    lifecycle_path = tmp_path / "lifecycle.json"
    state: LifecycleStateRecord = default_state()
    state["crisis_mode"] = True
    save_state(state, lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    done, payload = pipeline._step_crisis_recluster(interrupt_check=None)
    assert done is True
    assert payload["communities_dropped"] == 25

    # The cold re-cluster reassigns every record from scratch on the same
    # (node, edge) set, so the partition shape is deterministic. With no
    # inter-record edges among the 100 records, the re-cluster collapses them
    # into a single community.
    shape = _community_partition_shape(store)
    assert sum(shape) == 100, "every record must keep a community_id"
    assert shape == GOLDEN_CRISIS_PARTITION_SHAPE, (
        f"community partition shape changed: got {shape}, "
        f"expected {GOLDEN_CRISIS_PARTITION_SHAPE}"
    )

    final_state = load_state(lifecycle_path)
    assert final_state["crisis_mode"] is False


def test_reconsolidation_critic_pool_and_scan_count_are_byte_stable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_DRY_RUN", "false")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_TIER1", "true")
    monkeypatch.setenv("IAI_MCP_LABILE_WINDOW_SEC", "3600")
    monkeypatch.setenv("IAI_MCP_RECONSOLIDATION_ERROR_THRESHOLD", "0.5")

    store = _make_store(tmp_path)
    embed_dim = store._embed_dim

    expected_rids: set[str] = set()
    for i in range(7):
        rec = _make_record(embed_dim=embed_dim, text=f"alice memory {i}")
        store.insert(rec)
        store.reinforce_record(rec.id, is_retrieval=True)
        expected_rids.add(str(rec.id))

    captured_pools: list[list[tuple[Any, Any]]] = []

    def _capturing_critic(items: Any, *args: Any, **kwargs: Any) -> dict:
        pool = list(items)
        captured_pools.append(pool)
        return {rid: 0.9 for rid, _surface in pool}

    monkeypatch.setattr(
        "iai_mcp.reconsolidation_critic.evaluate_batch_reconsolidation",
        _capturing_critic,
    )

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    done, payload = pipeline._step_reconsolidation(interrupt_check=None)
    assert done is True

    # records_scanned counts every labile row.
    assert payload["records_scanned"] == 7

    # The critic was handed exactly the labile rid-set (order-independent).
    assert len(captured_pools) == 1
    critic_rids = frozenset(str(rid) for rid, _surface in captured_pools[0])
    assert critic_rids == frozenset(expected_rids), (
        "reconsolidation critic pool diverged from the labile rid-set"
    )

    events = query_events(store, kind="reconsolidation_pass", limit=10)
    assert len(events) == 1
    assert events[0]["data"]["records_scanned"] == 7


def test_cluster_replay_membership_signature_is_byte_stable(
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
    rid_to_offset: dict[str, int] = {}
    for cluster in cluster_offsets:
        for off in cluster:
            rec = _make_record(embed_dim=embed_dim, text=f"alice rec at {off}s")
            store.insert(rec)
            ts = now + timedelta(seconds=off)
            tbl.update(
                where=f"id = '{str(rec.id)}'",
                values={"last_reviewed": ts},
            )
            rid_to_offset[str(rec.id)] = off
    assert len(rid_to_offset) == 10

    lifecycle_path = tmp_path / "lifecycle.json"
    save_state(default_state(), lifecycle_path)
    pipeline = SleepPipeline(store=store, lifecycle_state_path=lifecycle_path)

    done, payload = pipeline._step_cluster_replay(interrupt_check=None)
    assert done is True
    assert payload["clusters_replayed"] == 3

    # The hebbian_cluster_replay edges encode cluster membership: every pair of
    # records that fell into the same temporal window. This is the
    # window-walk's observable output and is determined by the sort over
    # last_reviewed — exactly what the streaming rewrite must reproduce.
    membership = _edge_offset_signature(
        store, "hebbian_cluster_replay", rid_to_offset,
    )
    assert membership == GOLDEN_CLUSTER_MEMBERSHIP, (
        f"cluster membership changed: got {sorted(sorted(p) for p in membership)}, "
        f"expected {sorted(sorted(p) for p in GOLDEN_CLUSTER_MEMBERSHIP)}"
    )

    # The temporal_sequence trajectory edges are currently never written (the
    # edge_type is not accepted by boost_edges; the call is swallowed). Pin the
    # empty set so the rewrite reproduces it and any accidental re-enablement is
    # caught. See deferred-items.md.
    trajectory = _edge_offset_signature(
        store, "temporal_sequence", rid_to_offset,
    )
    assert trajectory == frozenset(), (
        "temporal_sequence edges must stay absent (byte-identical to current)"
    )
    assert payload["sequential_pairs"] == 0


# --- Golden snapshots captured from the current (pre-rewrite) source ---------
#
# CRISIS: the 100-record fixture has no inter-record edges, so the cold
# re-cluster collapses all records into one community.
GOLDEN_CRISIS_PARTITION_SHAPE: list[int] = [100]

# CLUSTER_REPLAY: undirected membership signature keyed by last_reviewed offset.
# Cluster A {-30,-60,-90,-120} -> all 6 pairs; B {-430,-460,-490} -> 3 pairs;
# C {-830,-860,-890} -> 3 pairs. Total 12, fully window-determined.
GOLDEN_CLUSTER_MEMBERSHIP: frozenset[frozenset[int]] = frozenset(
    frozenset(pair)
    for pair in (
        (-30, -60), (-30, -90), (-30, -120),
        (-60, -90), (-60, -120), (-90, -120),
        (-430, -460), (-430, -490), (-460, -490),
        (-830, -860), (-830, -890), (-860, -890),
    )
)
