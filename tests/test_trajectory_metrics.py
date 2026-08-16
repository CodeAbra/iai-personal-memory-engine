from __future__ import annotations

from datetime import datetime, timedelta, timezone


from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore

def test_metric_names_covers_m1_to_m6():
    from iai_mcp.trajectory import METRIC_NAMES

    assert set(METRIC_NAMES) == {"m1", "m2", "m3", "m4", "m5", "m6"}

def test_record_session_metrics_writes_6_events(tmp_path):
    from iai_mcp.trajectory import record_session_metrics

    store = MemoryStore(path=tmp_path)
    record_session_metrics(
        store, session_id="s1",
        metrics={"m1": 3.0, "m2": 0.7, "m3": 2000.0, "m4": 0.2, "m5": 1.0, "m6": 0.85},
    )
    events = query_events(store, kind="trajectory_metric")
    assert len(events) == 6
    metrics = {e["data"]["metric"] for e in events}
    assert metrics == {"m1", "m2", "m3", "m4", "m5", "m6"}

def test_record_session_metrics_ignores_bad_keys(tmp_path):
    from iai_mcp.trajectory import record_session_metrics

    store = MemoryStore(path=tmp_path)
    record_session_metrics(
        store, session_id="s-bad",
        metrics={"m1": 1.0, "m7_bogus": 42.0},
    )
    events = query_events(store, kind="trajectory_metric")
    metrics = {e["data"]["metric"] for e in events}
    assert "m7_bogus" not in metrics
    assert "m1" in metrics

def test_aggregate_trajectory_groups_by_metric(tmp_path):
    from iai_mcp.trajectory import aggregate_trajectory, record_session_metrics

    store = MemoryStore(path=tmp_path)
    for i in range(3):
        record_session_metrics(
            store, session_id=f"s{i}",
            metrics={"m1": float(i), "m2": float(i) * 0.1},
        )
    out = aggregate_trajectory(store)
    assert "m1" in out
    assert "m2" in out
    assert len(out["m1"]) == 3
    assert len(out["m2"]) == 3

def test_aggregate_trajectory_since_filter(tmp_path):
    from iai_mcp.trajectory import aggregate_trajectory, record_session_metrics

    store = MemoryStore(path=tmp_path)
    record_session_metrics(store, "s1", metrics={"m1": 1.0})
    future = datetime.now(timezone.utc) + timedelta(hours=1)
    out = aggregate_trajectory(store, since=future)
    assert sum(len(v) for v in out.values()) == 0

def test_m1_clarifying_questions_signal(tmp_path):
    from iai_mcp.trajectory import compute_m1_clarifying_questions_per_session

    store = MemoryStore(path=tmp_path)
    for _ in range(3):
        write_event(
            store, kind="curiosity_question",
            data={}, session_id="s-m1",
        )
    val = compute_m1_clarifying_questions_per_session(store, "s-m1")
    assert val == 3.0

def test_m3_token_budget_signal(tmp_path):
    from iai_mcp.trajectory import compute_m3_token_budget

    store = MemoryStore(path=tmp_path)
    for toks in (1000, 2000, 3000):
        write_event(
            store, kind="session_started",
            data={"total_cached_tokens": toks}, session_id="s-m3",
        )
    val = compute_m3_token_budget(store, "s-m3")
    assert val == 2000.0

def test_m3_token_budget_empty(tmp_path):
    from iai_mcp.trajectory import compute_m3_token_budget

    store = MemoryStore(path=tmp_path)
    assert compute_m3_token_budget(store, "s-empty") == 0.0

def test_m3_reads_what_session_start_writes(tmp_path):
    """The live serve path and the M3 metric must share one event kind."""
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.session import assemble_session_start
    from iai_mcp.trajectory import compute_m3_token_budget

    store = MemoryStore(path=tmp_path)
    assignment = CommunityAssignment(
        node_to_community={}, community_centroids={}, modularity=0.0,
        backend="flat", top_communities=[], mid_regions={},
    )
    assemble_session_start(store, assignment, [], session_id="s-live")
    assert compute_m3_token_budget(store, "s-live") > 0.0
