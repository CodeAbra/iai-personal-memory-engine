from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord

def _make_record(
    *,
    text: str = "hello",
    vec: list[float] | None = None,
    tags: list[str] | None = None,
    detail_level: int = 2,
    tier: str = "episodic",
    language: str = "en",
) -> MemoryRecord:
    if vec is None:
        vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=vec,
        community_id=None,
        centrality=0.0,
        detail_level=detail_level,
        pinned=False,
        stability=0.0,
        difficulty=0.0,
        last_reviewed=None,
        never_decay=False,
        never_merge=False,
        provenance=[],
        created_at=now,
        updated_at=now,
        tags=list(tags or []),
        language=language,
    )

def _hit_for(rec: MemoryRecord, score: float = 0.9) -> MemoryHit:
    return MemoryHit(
        record_id=rec.id,
        score=score,
        reason="test",
        literal_surface=rec.literal_surface,
        adjacent_suggestions=[],
    )

def test_s4_module_defines_rho_097():
    from iai_mcp import s4

    assert s4.S4_VIGILANCE_RHO == 0.97

def test_s4_exports_on_read_check():
    from iai_mcp import s4

    assert hasattr(s4, "on_read_check")
    assert callable(s4.on_read_check)

def test_s4_exports_monotropic_proactive_check():
    from iai_mcp import s4

    assert hasattr(s4, "monotropic_proactive_check")
    assert callable(s4.monotropic_proactive_check)

def test_global_daily_scan_not_implemented():
    from iai_mcp import s4

    assert not hasattr(s4, "daily_scan")
    assert not hasattr(s4, "session_exit_sweep")
    import inspect

    members = {name for name, _ in inspect.getmembers(s4)}
    assert "daily_scan" not in members
    assert "session_exit_sweep" not in members

def test_s4_on_read_returns_empty_when_consistent(tmp_path):
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    recs = []
    for i in range(5):
        vec = [0.0] * EMBED_DIM
        vec[i] = 1.0
        r = _make_record(text=f"rec {i}", vec=vec, tags=[f"topic_{i}"])
        store.insert(r)
        recs.append(r)

    hits = [_hit_for(r, score=0.5) for r in recs]
    result = on_read_check(store, hits, session_id="test")
    assert result == []

def test_s4_on_read_respects_contradicts_edge(tmp_path):
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v1 = [0.0] * EMBED_DIM
    v1[0] = 1.0
    v2 = [0.0] * EMBED_DIM
    v2[1] = 1.0
    r1 = _make_record(text="X is true", vec=v1, tags=["claim"])
    r2 = _make_record(text="X is false", vec=v2, tags=["claim"])
    store.insert(r1)
    store.insert(r2)
    store.add_contradicts_edge(r1.id, r2.id)

    hits = [_hit_for(r1), _hit_for(r2)]
    result = on_read_check(store, hits, session_id="test")
    assert len(result) == 1
    hint = result[0]
    assert hint["kind"] == "s4_contradiction"
    assert set(hint["source_ids"]) == {str(r1.id), str(r2.id)}
    assert "inconsistency" in hint["text"].lower()

def test_s4_on_read_uses_rho_097(tmp_path):
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    import math

    theta_low = math.acos(0.95)
    v_a = [math.cos(0.0)] + [0.0] * (EMBED_DIM - 1)
    v_b = [math.cos(theta_low), math.sin(theta_low)] + [0.0] * (EMBED_DIM - 2)
    r1 = _make_record(text="claim A", vec=v_a, tags=["topic"])
    r2 = _make_record(text="claim B", vec=v_b, tags=["topic"])
    store.insert(r1)
    store.insert(r2)

    hits = [_hit_for(r1), _hit_for(r2)]
    result = on_read_check(store, hits, session_id="test")
    assert result == []

def test_s4_on_read_detects_polarity_contradiction(tmp_path):
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v1 = [1.0] + [0.0] * (EMBED_DIM - 1)
    v2 = [0.99] + [0.01] + [0.0] * (EMBED_DIM - 2)
    r1 = _make_record(text="X is good", vec=v1, tags=["topic", "positive"])
    r2 = _make_record(text="X is bad", vec=v2, tags=["topic", "negative"])
    store.insert(r1)
    store.insert(r2)

    hits = [_hit_for(r1), _hit_for(r2)]
    result = on_read_check(store, hits, session_id="test")
    assert len(result) == 1
    hint = result[0]
    assert hint["kind"] == "s4_contradiction"
    assert set(hint["source_ids"]) == {str(r1.id), str(r2.id)}

def test_s4_on_read_writes_event(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v1 = [0.0] * EMBED_DIM
    v1[0] = 1.0
    v2 = [0.0] * EMBED_DIM
    v2[1] = 1.0
    r1 = _make_record(text="asserted", vec=v1, tags=["claim"])
    r2 = _make_record(text="retracted", vec=v2, tags=["claim"])
    store.insert(r1)
    store.insert(r2)
    store.add_contradicts_edge(r1.id, r2.id)

    hits = [_hit_for(r1), _hit_for(r2)]
    on_read_check(store, hits, session_id="s-test")

    events = query_events(store, kind="s4_contradiction")
    assert len(events) >= 1
    ev = events[0]
    assert ev["kind"] == "s4_contradiction"
    assert ev["session_id"] == "s-test"

def test_s4_on_read_empty_hits_returns_empty(tmp_path):
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    assert on_read_check(store, [], session_id="t") == []

def test_s4_on_read_single_hit_returns_empty(tmp_path):
    from iai_mcp.s4 import on_read_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v = [1.0] + [0.0] * (EMBED_DIM - 1)
    r = _make_record(vec=v)
    store.insert(r)
    assert on_read_check(store, [_hit_for(r)], session_id="t") == []

def test_monotropic_check_gate_profile_depth(tmp_path):
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v = [0.1] * EMBED_DIM
    new_rec = _make_record(
        text="new deep-interest fact",
        vec=v,
        tags=["domain:coding"],
        detail_level=5,
    )
    store.insert(new_rec)
    profile_state = {"monotropism_depth": {"coding": 0.5}}

    result = monotropic_proactive_check(
        store, new_rec, profile_state, session_id="t"
    )
    assert result == []

def test_monotropic_check_gate_detail_level(tmp_path):
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v = [0.1] * EMBED_DIM
    new_rec = _make_record(vec=v, tags=["domain:coding"], detail_level=3)
    store.insert(new_rec)
    profile_state = {"monotropism_depth": {"coding": 0.9}}

    result = monotropic_proactive_check(
        store, new_rec, profile_state, session_id="t"
    )
    assert result == []

def test_monotropic_check_within_domain_only(tmp_path):
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v_other = [1.0] + [0.0] * (EMBED_DIM - 1)
    other = _make_record(
        text="tomato care", vec=v_other, tags=["domain:gardening"]
    )
    store.insert(other)

    new_rec = _make_record(
        text="refactor method", vec=v_other, tags=["domain:coding"], detail_level=5
    )
    store.insert(new_rec)

    profile_state = {"monotropism_depth": {"coding": 0.9}}
    result = monotropic_proactive_check(
        store, new_rec, profile_state, session_id="t"
    )
    assert result == []

def test_monotropic_check_pairwise_scan_skip_above_100(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    for i in range(101):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        rec = _make_record(
            text=f"rec {i}", vec=vec, tags=["domain:coding"], detail_level=1
        )
        store.insert(rec)

    vec = [1.0] + [0.0] * (EMBED_DIM - 1)
    new_rec = _make_record(
        text="new", vec=vec, tags=["domain:coding"], detail_level=5
    )
    store.insert(new_rec)

    profile_state = {"monotropism_depth": {"coding": 0.9}}
    result = monotropic_proactive_check(
        store, new_rec, profile_state, session_id="t"
    )
    assert result == []
    events = query_events(store, kind="s4_monotropic_skip")
    assert len(events) >= 1

def test_monotropic_check_emits_event_on_hit(tmp_path):
    from iai_mcp.events import query_events
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v1 = [1.0] + [0.0] * (EMBED_DIM - 1)
    existing = _make_record(
        text="fact A", vec=v1, tags=["domain:coding"], detail_level=2
    )
    store.insert(existing)

    new_rec = _make_record(
        text="fact A again",
        vec=v1,
        tags=["domain:coding"],
        detail_level=5,
    )
    store.insert(new_rec)

    profile_state = {"monotropism_depth": {"coding": 0.9}}
    result = monotropic_proactive_check(
        store, new_rec, profile_state, session_id="s-mp"
    )
    assert len(result) >= 1
    events = query_events(store, kind="s4_monotropic_contradiction")
    assert len(events) >= 1
    assert events[0]["data"]["domain"] == "domain:coding"

def test_monotropic_check_missing_domain_tag_returns_empty(tmp_path):
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v = [1.0] + [0.0] * (EMBED_DIM - 1)
    new_rec = _make_record(text="x", vec=v, tags=[], detail_level=5)
    store.insert(new_rec)
    profile_state = {"monotropism_depth": {"coding": 0.9}}
    assert monotropic_proactive_check(store, new_rec, profile_state, session_id="t") == []

def test_monotropic_check_malformed_profile_state_degrades(tmp_path):
    from iai_mcp.s4 import monotropic_proactive_check
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v = [1.0] + [0.0] * (EMBED_DIM - 1)
    new_rec = _make_record(vec=v, tags=["domain:coding"], detail_level=5)
    store.insert(new_rec)
    profile_state = {"monotropism_depth": [0.9]}
    assert monotropic_proactive_check(store, new_rec, profile_state, session_id="t") == []

def test_recall_response_has_hints_field():
    from iai_mcp.types import RecallResponse

    resp = RecallResponse(hits=[], anti_hits=[], activation_trace=[], budget_used=0)
    assert hasattr(resp, "hints")
    assert resp.hints == []

def test_s4_on_read_hint_populated_in_recall(tmp_path):
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path)
    v = [1.0] + [0.0] * (EMBED_DIM - 1)
    r1 = _make_record(text="asserted X", vec=v, tags=["claim"])
    r2 = _make_record(text="retracted X", vec=v, tags=["claim"])
    store.insert(r1)
    store.insert(r2)
    store.add_contradicts_edge(r1.id, r2.id)

    class _Emb:
        DIM = EMBED_DIM

        def embed(self, text):
            return v

        def embed_batch(self, texts):
            return [v for _ in texts]

    graph, assignment, rc = build_runtime_graph(store)
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=_Emb(),
        cue="X",
        session_id="t",
        budget_tokens=1500,
    )
    assert hasattr(resp, "hints")
    assert len(resp.hints) >= 1
    h = resp.hints[0]
    assert h["kind"] == "s4_contradiction"
    assert isinstance(h["source_ids"], list)
    assert "text" in h
