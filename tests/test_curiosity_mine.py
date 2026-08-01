"""Nightly curiosity mining: deferred recall inputs become pending questions.

The recall hot path buffers ``deferred_curiosity_input`` events (hit ids +
scores + cue + turn); the CURIOSITY_MINE sleep step re-ranks each cue against
the current store and mints only on a live contradiction among the fresh top
candidates. A topic that gained records after the snapshot is active — the
miner skips it and resolves any pending question on that cue. The watermark
makes re-runs idempotent, per-session and per-run caps bound the output, and
the TTL expires questions whose topic simply went quiet.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp.events import query_events, write_event
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord


def _rec(vec=None, text="rec"):
    vec = vec or [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=2,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=[],
        language="en",
    )


def _write_deferred(store, *, cue, scores, session_id="s1", hit_ids=None, turn=1):
    write_event(
        store,
        kind="deferred_curiosity_input",
        data={
            "hit_ids": [str(x) for x in (hit_ids or [])],
            "scores": scores,
            "cue": cue,
            "session_id": session_id,
            "turn": turn,
        },
        severity="info",
        session_id=session_id,
    )


# Entropy is normalized to [0, 1] by log2(n_nonzero); for two scores the
# divisor is 1, so the p/(1-p) calibration is the raw Shannon value:
#   [0.5, 0.5]   -> 1.000  (>= ENTROPY_HIGH 0.9, tier "question")
#   [0.75, 0.25] -> 0.811  (in [0.7, 0.9), tier "inline")
#   [0.9, 0.1]   -> 0.469  (in [0.4, 0.7), silent log)
#   [0.99, 0.01] -> 0.081  (< ENTROPY_LOW 0.4, skipped)
# Many-candidate distributions land on the same scale: 10 equal scores -> 1.0
# (max uncertainty), one dominant among 10 -> well below the top tier.


def test_entropy_normalized_across_candidate_counts():
    from iai_mcp.curiosity import ENTROPY_LOW, compute_entropy

    assert abs(compute_entropy([0.5, 0.5]) - 1.0) < 1e-9
    assert abs(compute_entropy([0.75, 0.25]) - 0.811) < 1e-3
    assert abs(compute_entropy([0.1] * 10) - 1.0) < 1e-9
    dominant = compute_entropy([0.9] + [0.01] * 9)
    assert dominant < ENTROPY_LOW, (
        f"one dominant candidate among 10 must be low uncertainty, got {dominant}"
    )
    assert compute_entropy([0.7]) == 0.0
    assert compute_entropy([]) == 0.0


def _contradiction_fixture(store, *, text_a="the daemon restarts nightly",
                           text_b="the daemon restarts weekly"):
    """Two records joined by a live contradicts edge — the v2 mint trigger."""
    from iai_mcp.store import flush_edge_buffer

    a = _rec(vec=[1.0] + [0.0] * (EMBED_DIM - 1), text=text_a)
    b = _rec(vec=[0.0, 1.0] + [0.0] * (EMBED_DIM - 2), text=text_b)
    store.insert(a)
    store.insert(b)
    store.boost_edges([(a.id, b.id)], edge_type="contradicts", delta=1.0)
    flush_edge_buffer(store)
    return a, b


def test_contradiction_mints_two_sided_question(tmp_path):
    from iai_mcp.curiosity import pending_questions, process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    a, b = _contradiction_fixture(store)
    _write_deferred(
        store, cue="daemon restart schedule", scores=[0.5, 0.5], hit_ids=[a.id],
    )

    result = process_deferred_inputs(store)

    assert result["curiosity_minted"] == 1
    qs = pending_questions(store)
    assert len(qs) == 1
    q = qs[0]
    assert q.tier == "question"
    assert "disagree" in q.text
    assert "nightly" in q.text and "weekly" in q.text
    assert set(q.triggered_by_record_ids) == {a.id, b.id}


def test_dense_topic_without_contradiction_logs_silently(tmp_path):
    from iai_mcp.curiosity import pending_questions, process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    for i in range(3):
        store.insert(_rec(vec=[0.0] * i + [1.0] + [0.0] * (EMBED_DIM - 1 - i),
                          text=f"dense knowledge {i}"))
    _write_deferred(store, cue="dense topic", scores=[0.5, 0.5])

    result = process_deferred_inputs(store)

    assert result["curiosity_minted"] == 0
    assert result["curiosity_silent"] == 1
    assert pending_questions(store) == []
    assert len(query_events(store, kind="curiosity_silent_log")) == 1


def test_topic_active_after_snapshot_skips_and_resolves(tmp_path):
    from iai_mcp.curiosity import pending_questions, process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    _write_deferred(store, cue="daemon restart schedule", scores=[0.5, 0.5])
    assert process_deferred_inputs(store)["curiosity_minted"] == 1
    assert len(pending_questions(store)) == 1

    _write_deferred(store, cue="daemon restart schedule", scores=[0.5, 0.5])
    store.insert(_rec(vec=[0.0, 0.0, 1.0] + [0.0] * (EMBED_DIM - 3),
                      text="clarified: restarts happen nightly at 3am"))

    result = process_deferred_inputs(store)

    assert result["curiosity_minted"] == 0
    assert result["curiosity_resolved"] == 1
    assert pending_questions(store) == [], (
        "answered topic must resolve its pending question"
    )
    runs = query_events(store, kind="curiosity_mine_run", limit=1)
    assert runs[0]["data"]["skipped_active"] == 1


def test_confident_input_skipped_entirely(tmp_path):
    from iai_mcp.curiosity import process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _write_deferred(store, cue="confident", scores=[0.99, 0.01])

    result = process_deferred_inputs(store)

    assert result["curiosity_minted"] == 0
    assert result["curiosity_silent"] == 0


def test_watermark_makes_rerun_idempotent(tmp_path):
    from iai_mcp.curiosity import pending_questions, process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    _write_deferred(store, cue="what deploy", scores=[0.5, 0.5])

    first = process_deferred_inputs(store)
    second = process_deferred_inputs(store)

    assert first["curiosity_minted"] == 1
    assert second["curiosity_scanned"] == 0
    assert second["curiosity_minted"] == 0
    assert len(pending_questions(store)) == 1


def test_legacy_events_without_scores_are_skipped_past(tmp_path):
    from iai_mcp.curiosity import process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    write_event(
        store,
        kind="deferred_curiosity_input",
        data={"hit_ids": [], "cue": "pre-upgrade event", "session_id": "s1"},
        severity="info",
        session_id="s1",
    )

    first = process_deferred_inputs(store)
    second = process_deferred_inputs(store)

    assert first["curiosity_scanned"] == 1
    assert first["curiosity_minted"] == 0
    assert second["curiosity_scanned"] == 0


def test_per_session_cap_bounds_minting(tmp_path):
    from iai_mcp.curiosity import MINE_MAX_PER_SESSION, process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    for i in range(MINE_MAX_PER_SESSION + 2):
        _write_deferred(store, cue=f"distinct cue {i}", scores=[0.5, 0.5])

    result = process_deferred_inputs(store)

    assert result["curiosity_minted"] == MINE_MAX_PER_SESSION


def test_sessions_capped_independently(tmp_path):
    from iai_mcp.curiosity import process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    _write_deferred(store, cue="cue a", scores=[0.5, 0.5], session_id="sA")
    _write_deferred(store, cue="cue b", scores=[0.5, 0.5], session_id="sB")

    result = process_deferred_inputs(store)

    assert result["curiosity_minted"] == 2


def test_already_asked_cue_not_reminted(tmp_path):
    from iai_mcp.curiosity import pending_questions, process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    _write_deferred(store, cue="same cue", scores=[0.5, 0.5])
    assert process_deferred_inputs(store)["curiosity_minted"] == 1

    _write_deferred(store, cue="same cue", scores=[0.5, 0.5])
    result = process_deferred_inputs(store)

    assert result["curiosity_scanned"] == 1
    assert result["curiosity_minted"] == 0
    assert len(pending_questions(store)) == 1


def test_pending_questions_expire_by_ttl(tmp_path, monkeypatch):
    from iai_mcp import curiosity

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    _write_deferred(store, cue="short lived", scores=[0.5, 0.5])
    curiosity.process_deferred_inputs(store)
    assert len(curiosity.pending_questions(store)) == 1

    monkeypatch.setattr(curiosity, "PENDING_TTL_DAYS", 0)
    assert curiosity.pending_questions(store) == []


def test_mine_run_event_records_watermark(tmp_path):
    from iai_mcp.curiosity import process_deferred_inputs

    store = MemoryStore(path=tmp_path)
    _contradiction_fixture(store)
    _write_deferred(store, cue="anything", scores=[0.5, 0.5])
    process_deferred_inputs(store)

    runs = query_events(store, kind="curiosity_mine_run", limit=5)
    assert len(runs) == 1
    data = runs[0]["data"]
    assert data["scanned"] == 1
    assert data["minted"] == 1
    datetime.fromisoformat(data["through_ts"])


def test_recall_defers_scores_and_turn(tmp_path):
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.events import flush_event_buffer
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import recall_for_response

    def _unit(hot):
        v = [0.0] * EMBED_DIM
        v[hot] = 1.0
        return v

    store = MemoryStore(path=tmp_path / "store")
    graph = MemoryGraph()
    recs = []
    for i in range(3):
        rec = _rec(vec=_unit(i), text=f"deploy note {i}")
        store.insert(rec)
        recs.append(rec)
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    cid = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: _unit(0)},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )

    class _Emb:
        DIM = EMBED_DIM

        def embed(self, text):
            return _unit(0)

        def embed_batch(self, texts):
            return [_unit(0) for _ in texts]

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_Emb(),
        cue="deploy note",
        session_id="s-defer",
        turn=3,
    )
    flush_event_buffer(store)

    events = query_events(store, kind="deferred_curiosity_input", limit=5)
    assert events, "non-verbatim recall with hits must buffer curiosity input"
    data = events[0]["data"]
    assert data["turn"] == 3
    assert isinstance(data["scores"], list) and data["scores"]
    assert len(data["scores"]) == len(data["hit_ids"])
    assert all(isinstance(s, float) for s in data["scores"])


def test_deferred_payload_excludes_recency_markers(tmp_path):
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.events import flush_event_buffer
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import recall_for_response
    from uuid import UUID

    def _unit(hot):
        v = [0.0] * EMBED_DIM
        v[hot] = 1.0
        return v

    store = MemoryStore(path=tmp_path / "store")
    graph = MemoryGraph()
    recs = []
    for i in range(3):
        rec = _rec(vec=_unit(i), text=f"deploy note {i}")
        store.insert(rec)
        recs.append(rec)
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    cid = uuid4()
    assignment = CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: _unit(0)},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )

    marker_id = UUID(int=888_001)
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=str(marker_id),
        tier="episodic",
        literal_surface="pending recency marker text",
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json="[]",
    )

    class _Emb:
        DIM = EMBED_DIM

        def embed(self, text):
            return _unit(0)

        def embed_batch(self, texts):
            return [_unit(0) for _ in texts]

    recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=_Emb(),
        cue="deploy note",
        session_id="s-marker",
        turn=1,
    )
    flush_event_buffer(store)

    events = query_events(store, kind="deferred_curiosity_input", limit=5)
    assert events
    data = events[0]["data"]
    assert str(marker_id) not in data["hit_ids"], (
        "recency marker leaked into the curiosity payload"
    )
    assert all(s > 0.0 for s in data["scores"]), (
        "zero-score entries in curiosity payload — entropy signal poisoned"
    )


def test_knob_tune_ignores_zero_curiosity_signal(tmp_path, monkeypatch):
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline

    store = MemoryStore(path=tmp_path / "store")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    pipeline = SleepPipeline(
        store=store,
        lifecycle_state_path=tmp_path / "lifecycle_state.json",
        event_log=LifecycleEventLog(log_dir=log_dir),
        quarantine_ttl_hours=24.0,
    )

    saves = []
    monkeypatch.setattr("iai_mcp.user_model.save", lambda um: saves.append(um))

    done, _payload = pipeline._step_knob_tune(None)

    assert done is True
    assert saves == [], (
        "zero curiosity_bridge edges is absence of signal; the soft knob "
        "must not be nudged"
    )
