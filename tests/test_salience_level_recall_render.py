"""Read-path proof for salience_level through every memory_recall hit.

Companion to test_epistemic_status_recall_render.py's coverage of the same
8 MemoryHit construction sites across pipeline.py, retrieve.py, and
core/__init__.py: each must thread the source record's salience_level onto
the returned hit, and the field must reach the JSON response via the
single _hit_to_json chokepoint. Correctness is proven by value-equality
against a fixture with explicitly distinct per-record levels, not by a
bare non-None check. The shared 8-site count-and-keyword AST guard lives
in test_epistemic_status_recall_render.py and is widened there, not
duplicated here.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import numpy as np

from iai_mcp import pipeline
from iai_mcp import retrieve
from iai_mcp.core._serializers import _hit_to_json
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord


def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def _unit_vec(idx: int, sign: float = 1.0) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[idx] = sign
    return v


def _mixed_vec(primary_idx: int, primary_weight: float, secondary_idx: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[primary_idx] = primary_weight
    v[secondary_idx] = (1.0 - primary_weight**2) ** 0.5
    return v


def _mk_rec(
    text: str,
    embedding: list[float],
    salience_level: str = "unflagged",
    tags: list[str] | None = None,
) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier="episodic",
        literal_surface=text,
        aaak_index="",
        embedding=embedding,
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
        tags=tags or [],
        language="en",
        salience_level=salience_level,
    )


def test_retrieve_recall_hits_and_anti_hits_carry_salience_level(tmp_path):
    """Sites 5 (hits) + 6 (anti_hits): full retrieve.recall() round trip.

    Four fixture records with four DISTINCT salience_level values, ranked
    deterministically by cosine to the cue so exactly two land as hits and
    two land as anti_hits.
    """
    store = MemoryStore(path=tmp_path / "recall-store")

    h1 = _mk_rec("alice's release shipped on schedule", _unit_vec(0), "critical")
    h2 = _mk_rec("alice thinks the next release slips", _mixed_vec(0, 0.9, 1), "notable")
    a1 = _mk_rec("alice's unrelated hobby project", _unit_vec(1), "unflagged")
    a2 = _mk_rec("alice's opposite-direction note", _unit_vec(0, sign=-1.0), "critical")
    for rec in (h1, h2, a1, a2):
        store.insert(rec)

    stored: dict[UUID, str] = {
        h1.id: "critical", h2.id: "notable", a1.id: "unflagged", a2.id: "critical",
    }
    surfaces: dict[UUID, str] = {r.id: r.literal_surface for r in (h1, h2, a1, a2)}

    response = retrieve.recall(
        store,
        cue_embedding=_unit_vec(0),
        cue_text="ignored — a valid vector is supplied directly",
        session_id="s1",
        k_hits=2,
        k_anti=2,
        mode="verbatim",
        allow_cue_reembed=False,
    )

    matched = [
        h for h in (list(response.hits) + list(response.anti_hits))
        if h.record_id in stored
    ]
    assert len(matched) >= 4, (
        f"expected all 4 fixture records to surface as hits/anti_hits, got "
        f"{len(matched)}: {[str(h.record_id) for h in matched]}"
    )
    distinct_values = {h.salience_level for h in matched}
    assert len(distinct_values) >= 2, (
        f"matched hits carry only {distinct_values!r} — fixture is vacuously uniform"
    )

    for h in matched:
        assert h.salience_level == stored[h.record_id], (
            f"record {h.record_id}: hit carries {h.salience_level!r}, "
            f"stored value was {stored[h.record_id]!r}"
        )
        assert h.literal_surface == surfaces[h.record_id], (
            "literal_surface must stay byte-identical alongside the passthrough field"
        )


def test_pipeline_find_anti_hits_carries_salience_level(tmp_path):
    """Site 1: the contradicts-edge anti-hit loop in _find_anti_hits."""
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import _find_anti_hits

    store = MemoryStore(path=tmp_path / "anti-hit-store")

    primary = _mk_rec("alice's release ships Tuesday", _unit_vec(0), "unflagged")
    contradicting = _mk_rec("alice's release actually slipped", _unit_vec(1), "critical")
    store.insert(primary)
    store.insert(contradicting)

    tbl = store.db.open_table("edges")
    tbl.add([{
        "src": str(primary.id),
        "dst": str(contradicting.id),
        "edge_type": "contradicts",
        "weight": 1.0,
        "updated_at": datetime.now(timezone.utc),
    }])

    graph = MemoryGraph()
    seed_hit = MemoryHit(
        record_id=primary.id, score=0.9, reason="seed",
        literal_surface=primary.literal_surface, adjacent_suggestions=[],
    )

    anti = _find_anti_hits([seed_hit], store, graph, k=3, records_cache=None)

    assert len(anti) == 1, f"expected 1 anti-hit, got {len(anti)}"
    assert anti[0].record_id == contradicting.id
    assert anti[0].salience_level == "critical"
    assert anti[0].literal_surface == contradicting.literal_surface


def test_pipeline_find_anti_hits_populated_cache_still_recovers_salience_level(tmp_path):
    """Graph-cache-sourced regression: a contradicts-edge neighbour resolved
    through a POPULATED records_cache (SimpleRecordView -- no
    salience_level field) renders None straight out of _find_anti_hits
    itself; the shared backfill helper that recall_for_response and
    recall_for_benchmark run before returning must recover the real stored
    value. The prior test at this file's
    test_pipeline_find_anti_hits_carries_salience_level passes
    records_cache=None, exercising only the store.get_batch branch, which
    hides this drop entirely.
    """
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import SimpleRecordView, _backfill_hit_metadata, _find_anti_hits

    store = MemoryStore(path=tmp_path / "anti-hit-cache-store")

    primary = _mk_rec("alice's release ships Tuesday", _unit_vec(0), "unflagged")
    contradicting = _mk_rec("alice's release actually slipped", _unit_vec(1), "critical")
    store.insert(primary)
    store.insert(contradicting)

    tbl = store.db.open_table("edges")
    tbl.add([{
        "src": str(primary.id),
        "dst": str(contradicting.id),
        "edge_type": "contradicts",
        "weight": 1.0,
        "updated_at": datetime.now(timezone.utc),
    }])

    graph = MemoryGraph()
    seed_hit = MemoryHit(
        record_id=primary.id, score=0.9, reason="seed",
        literal_surface=primary.literal_surface, adjacent_suggestions=[],
    )

    records_cache = {
        contradicting.id: SimpleRecordView(
            id=contradicting.id,
            embedding=None,
            literal_surface=contradicting.literal_surface,
            centrality=0.0,
            tier="episodic",
        ),
    }

    anti = _find_anti_hits(
        [seed_hit], store, graph, k=3, records_cache=records_cache,
    )
    assert len(anti) == 1, f"expected 1 anti-hit, got {len(anti)}"
    assert anti[0].record_id == contradicting.id
    assert anti[0].salience_level is None, (
        "precondition: a SimpleRecordView-sourced hit has no salience_level "
        "field, so it must render None (not a silently-defaulted 'unflagged') "
        "before the backfill runs — that distinction is what lets a caller "
        "tell 'not yet enriched' apart from 'genuinely unflagged'"
    )

    # _find_anti_hits alone does not repair the field -- that happens one
    # layer up, in the shared backfill helper called by the recall entry
    # points. Apply it explicitly here to prove the repair mechanism itself
    # recovers the true stored value from a graph-view resolution.
    _backfill_hit_metadata([], anti, store)
    assert anti[0].salience_level == "critical", (
        f"backfill must recover the real stored salience_level for the "
        f"anti-hit, got {anti[0].salience_level!r}"
    )


def test_recall_for_response_anti_hit_from_graph_cache_carries_salience_level(tmp_path):
    """End-to-end regression through the real recall_for_response call site,
    not a hand-rolled call to the backfill helper. Both the primary hit and
    its contradicting neighbour are graph-view-sourced (SimpleRecordView,
    no salience_level field) via a real build_runtime_graph pass -- the
    anti-hit must still carry the true stored level once recall_for_response
    returns. Deleting the widened backfill call from recall_for_response
    regresses this test.

    The contradicting record stays at the neutral default level so the
    salience rank boost cannot perturb which side of the hits/anti_hits
    split it lands on -- the split itself is enforced by
    the same graph-topology fixture shape (an orthogonal cue embedding plus
    paired filler edges) the epistemic_status sibling test uses, and giving
    the fragile record a non-default level here would conflate a boost bug
    with a threading bug. The primary record, whose position in `hits` is
    reinforced (not contested) by a higher level, carries the non-default
    value instead.
    """
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "e2e-anti-hit-store")
    fillers = []
    for i in range(60):
        f = _mk_rec(f"unrelated filler record {i}", _random_vec(9000 + i))
        store.insert(f)
        fillers.append(f)

    primary = _mk_rec("alice's release ships Tuesday", _unit_vec(0), "notable")
    contradicting = _mk_rec("alice's release actually slipped", _unit_vec(1), "unflagged")
    store.insert(primary)
    store.insert(contradicting)

    tbl = store.db.open_table("edges")
    edge_rows = [{
        "src": str(primary.id),
        "dst": str(contradicting.id),
        "edge_type": "contradicts",
        "weight": 1.0,
        "updated_at": datetime.now(timezone.utc),
    }]
    for j in range(0, len(fillers) - 1, 2):
        edge_rows.append({
            "src": str(fillers[j].id),
            "dst": str(fillers[j + 1].id),
            "edge_type": "hebbian",
            "weight": 1.0,
            "updated_at": datetime.now(timezone.utc),
        })
    tbl.add(edge_rows)

    g, a, rc = retrieve.build_runtime_graph(store)
    import iai_mcp.runtime_graph_cache as _rgc
    _rgc.save(store, a, rc)

    embedder = Embedder()
    pipeline._last_recall_latency_ms = 0.0
    response = pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue="ignored — a valid vector is supplied directly",
        session_id="s1", budget_tokens=2000, mode="concept",
        cue_embedding=_unit_vec(0),
    )

    _hit_ids = {h.record_id for h in response.hits}
    assert primary.id in _hit_ids, (
        "fixture precondition: primary must rank into hits for its "
        "contradicts edge to be looked up by _find_anti_hits at all"
    )
    assert contradicting.id not in _hit_ids, (
        "fixture precondition: contradicting must stay OUT of hits or the "
        "anti-hit path is never exercised"
    )

    anti_matches = [h for h in response.anti_hits if h.record_id == contradicting.id]
    assert len(anti_matches) == 1, (
        f"contradicting record not surfaced as an anti-hit: "
        f"{[(h.record_id, h.reason) for h in response.anti_hits]}"
    )
    anti_hit = anti_matches[0]
    assert anti_hit.salience_level == "unflagged", (
        f"end-to-end recall_for_response anti-hit dropped salience_level, "
        f"got {anti_hit.salience_level!r}"
    )
    assert anti_hit.literal_surface == contradicting.literal_surface


def test_recall_for_benchmark_graph_view_hit_carries_salience_level(tmp_path):
    """recall_for_benchmark has no session_id-gated enrichment loop the way
    recall_for_response does; a primary hit sourced from the graph-view
    records_cache (SimpleRecordView, no salience_level field) renders None
    unless the shared backfill helper runs at the end of this path. Prove
    it recovers the real stored value. Paired with the scored-hits test
    below, this covers both recall entry points.
    """
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "benchmark-store")
    for i in range(10):
        store.insert(_mk_rec(f"unrelated filler record {i}", _random_vec(8000 + i)))

    embedder = Embedder()
    target_text = "alice confirmed the audit passed cleanly"
    target = _mk_rec(target_text, list(embedder.embed(target_text)), "critical")
    store.insert(target)

    g, a, rc = retrieve.build_runtime_graph(store)
    import iai_mcp.runtime_graph_cache as _rgc
    _rgc.save(store, a, rc)

    response = pipeline.recall_for_benchmark(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue=target_text, session_id="s1",
        k_hits=10, mode="concept",
    )

    matches = [h for h in response.hits if h.record_id == target.id]
    assert len(matches) == 1, (
        f"target record not surfaced in recall_for_benchmark hits: "
        f"{[(h.record_id, h.reason) for h in response.hits]}"
    )
    hit = matches[0]
    assert hit.salience_level == "critical", (
        f"graph-view-sourced benchmark hit dropped salience_level, "
        f"got {hit.salience_level!r}"
    )
    assert hit.literal_surface == target_text


def test_pipeline_scored_hits_carry_salience_level(tmp_path):
    """Site 3: the main scored-hits loop inside recall_for_response. This
    is the second of the two "both recall entry points" proofs (paired
    with the benchmark test above).
    """
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "scored-hits-store")
    for i in range(10):
        store.insert(_mk_rec(f"unrelated filler record {i}", _random_vec(5000 + i)))

    embedder = Embedder()
    target_text = "alice confirmed the deployment finished successfully"
    target = _mk_rec(target_text, list(embedder.embed(target_text)), "notable")
    store.insert(target)

    g, a, rc = retrieve.build_runtime_graph(store)
    import iai_mcp.runtime_graph_cache as _rgc
    _rgc.save(store, a, rc)

    pipeline._last_recall_latency_ms = 0.0
    response = pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue=target_text, session_id="s1",
        budget_tokens=2000, mode="concept",
    )

    matches = [h for h in response.hits if h.record_id == target.id]
    assert len(matches) == 1, (
        f"target record not surfaced in scored hits: "
        f"{[(h.record_id, h.reason) for h in response.hits]}"
    )
    hit = matches[0]
    assert hit.salience_level == "notable"
    assert hit.literal_surface == target_text


def test_pipeline_pending_recency_marker_carries_salience_level(tmp_path):
    """Site 4: the pending-recency marker union inside recall_for_response.

    insert_pending_row does not accept a salience_level argument, so the
    marker's genuine value is whatever recent_pending_markers() independently
    returns (the MemoryRecord default) — read directly from the store rather
    than assumed, so the check stays a real value-equality proof.
    """
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "pending-marker-store")
    for i in range(5):
        store.insert(_mk_rec(f"unrelated active record {i}", _random_vec(6000 + i)))

    pending_id = uuid4()
    now = datetime.now(timezone.utc)
    store.db.insert_pending_row(
        record_id=str(pending_id),
        tier="episodic",
        literal_surface="alice's pending marker just captured",
        provenance_json="[]",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
        tags_json="[]",
    )

    markers = store.recent_pending_markers(n=10)
    marker_rec = next((m for m in markers if m.id == pending_id), None)
    assert marker_rec is not None, "pending marker not visible via recent_pending_markers"
    expected_level = marker_rec.salience_level

    embedder = Embedder()
    g, a, rc = retrieve.build_runtime_graph(store)
    import iai_mcp.runtime_graph_cache as _rgc
    _rgc.save(store, a, rc)

    pipeline._last_recall_latency_ms = 0.0
    response = pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue="a completely unrelated cue", session_id="s1",
        budget_tokens=2000, mode="concept",
    )

    pm_hits = [h for h in response.hits if h.record_id == pending_id]
    assert len(pm_hits) == 1, (
        f"pending marker not surfaced: "
        f"{[(h.record_id, h.reason) for h in response.hits]}"
    )
    hit = pm_hits[0]
    assert hit.reason == "pending-recency"
    assert hit.salience_level == expected_level
    assert hit.literal_surface == "alice's pending marker just captured"


def test_hit_to_json_serializer_carries_salience_level():
    """The single _hit_to_json chokepoint every JSON-facing recall response
    (dispatch's main hits/anti_hits, first_turn_recall, and the
    crisis-degraded direct hit) routes through must carry salience_level —
    value equality against a hit with a non-default level, not a presence
    check."""
    hit = MemoryHit(
        record_id=uuid4(), score=0.42, reason="test",
        literal_surface="alice's flagged record", adjacent_suggestions=[],
        salience_level="critical",
    )
    payload = _hit_to_json(hit)
    assert payload["salience_level"] == "critical", (
        f"_hit_to_json dropped salience_level, got {payload.get('salience_level')!r}"
    )


def test_memoryhit_dataclass_carries_salience_level_default_none():
    """A MemoryHit constructed without salience_level stays None (not
    'unflagged'), preserving the "not yet enriched" vs "genuinely
    unflagged" distinction the backfill relies on."""
    hit = MemoryHit(
        record_id=uuid4(), score=0.1, reason="test",
        literal_surface="unset", adjacent_suggestions=[],
    )
    assert hit.salience_level is None
