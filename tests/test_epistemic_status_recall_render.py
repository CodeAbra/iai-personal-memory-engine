"""Read-path proof for epistemic_status through every memory_recall hit.

Guards the 8 MemoryHit construction sites across pipeline.py, retrieve.py,
and core/__init__.py: each must thread the source record's epistemic_status
onto the returned hit, and the field must reach the JSON response via the
single _hit_to_json chokepoint. Correctness is proven by value-equality
against a fixture with explicitly distinct per-record statuses, not by a
bare non-None check.
"""
from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest

from iai_mcp import pipeline
from iai_mcp import retrieve
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryHit, MemoryRecord

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src" / "iai_mcp"


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
    epistemic_status: str = "unknown",
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
        epistemic_status=epistemic_status,
    )


def test_retrieve_recall_hits_and_anti_hits_carry_epistemic_status(tmp_path):
    """Site 5 (hits) + site 6 (anti_hits): full retrieve.recall() round trip.

    Four fixture records with four DISTINCT epistemic_status values,
    ranked deterministically by cosine to the cue so exactly two land as
    hits and two land as anti_hits.
    """
    store = MemoryStore(path=tmp_path / "recall-store")

    h1 = _mk_rec("alice's release shipped on schedule", _unit_vec(0), "fact")
    h2 = _mk_rec("alice thinks the next release slips", _mixed_vec(0, 0.9, 1), "estimate")
    a1 = _mk_rec("alice's unrelated hobby project", _unit_vec(1), "hypothesis")
    a2 = _mk_rec("alice's opposite-direction note", _unit_vec(0, sign=-1.0), "opinion")
    for rec in (h1, h2, a1, a2):
        store.insert(rec)

    stored: dict[UUID, str] = {
        h1.id: "fact", h2.id: "estimate", a1.id: "hypothesis", a2.id: "opinion",
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
    distinct_values = {h.epistemic_status for h in matched}
    assert len(distinct_values) >= 2, (
        f"matched hits carry only {distinct_values!r} — fixture is vacuously uniform"
    )

    for h in matched:
        assert h.epistemic_status == stored[h.record_id], (
            f"record {h.record_id}: hit carries {h.epistemic_status!r}, "
            f"stored value was {stored[h.record_id]!r}"
        )
        assert h.literal_surface == surfaces[h.record_id], (
            "literal_surface must stay byte-identical alongside the passthrough field"
        )


def test_pipeline_find_anti_hits_carries_epistemic_status(tmp_path):
    """Site 1: the contradicts-edge anti-hit loop in _find_anti_hits."""
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import _find_anti_hits

    store = MemoryStore(path=tmp_path / "anti-hit-store")

    primary = _mk_rec("alice's release ships Tuesday", _unit_vec(0), "fact")
    contradicting = _mk_rec("alice's release actually slipped", _unit_vec(1), "hypothesis")
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
    assert anti[0].epistemic_status == "hypothesis"
    assert anti[0].literal_surface == contradicting.literal_surface


def test_pipeline_find_anti_hits_populated_cache_still_recovers_epistemic_status(tmp_path):
    """Regression: a contradicts-edge neighbour resolved through a POPULATED
    records_cache (SimpleRecordView -- no epistemic_status field) renders
    None straight out of _find_anti_hits itself; the shared backfill helper
    that recall_for_response/recall_for_benchmark run before returning must
    recover the real stored value. The prior test at this file's
    test_pipeline_find_anti_hits_carries_epistemic_status passes
    records_cache=None, exercising only the store.get_batch branch, which
    hides this drop entirely.
    """
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.pipeline import SimpleRecordView, _backfill_hit_metadata, _find_anti_hits

    store = MemoryStore(path=tmp_path / "anti-hit-cache-store")

    primary = _mk_rec("alice's release ships Tuesday", _unit_vec(0), "fact")
    contradicting = _mk_rec("alice's release actually slipped", _unit_vec(1), "hypothesis")
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

    # _find_anti_hits alone does not repair the field -- that happens one
    # layer up, in the shared backfill helper called by the recall entry
    # points. Apply it explicitly here to prove the repair mechanism itself
    # works against a populated (graph-view) cache resolution, independent
    # of whether _find_anti_hits's own output happens to be None or already
    # correct (a future SimpleRecordView improvement should not break this).
    _backfill_hit_metadata([], anti, store)
    assert anti[0].epistemic_status == "hypothesis", (
        f"backfill must recover the real stored status for the anti-hit, "
        f"got {anti[0].epistemic_status!r}"
    )


def test_recall_for_response_anti_hit_from_graph_cache_carries_epistemic_status(tmp_path):
    """End-to-end WR-01 regression through the real recall_for_response call
    site, not a hand-rolled call to the backfill helper. Both the primary
    hit and its contradicting neighbour are graph-view-sourced
    (SimpleRecordView, no epistemic_status field) via a real
    build_runtime_graph pass -- the anti-hit must still carry the true
    stored status once recall_for_response returns. Deleting the
    _backfill_hit_metadata call from recall_for_response regresses
    this test; the hand-rolled helper test above does not catch that.

    The contradicting record uses an ORTHOGONAL embedding (unit_vec(1) vs
    the cue's unit_vec(0)) so it does not rank as a hit on cosine alone --
    it must surface exclusively through the contradicts-edge anti-hit path,
    the same shape as the review's failure scenario. Filler records are
    paired off with their own (non-contradicts) edges so the graph's degree
    normalization does not single the contradicting record out as the only
    node with an edge and rank it into the hit set on structural centrality
    alone -- without this, the contradicting record's sole degree bonus
    outranks the filler noise floor and it lands in `hits` instead of
    `anti_hits`, which would trivially pass without exercising the anti-hit
    backfill path at all. A real build_runtime_graph pass (not a hand-built
    MemoryGraph) puts both records in the graph-view records_cache as
    SimpleRecordView.
    """
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "e2e-anti-hit-store")
    fillers = []
    for i in range(60):
        f = _mk_rec(f"unrelated filler record {i}", _random_vec(9000 + i))
        store.insert(f)
        fillers.append(f)

    primary = _mk_rec("alice's release ships Tuesday", _unit_vec(0), "fact")
    contradicting = _mk_rec("alice's release actually slipped", _unit_vec(1), "hypothesis")
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
        "anti-hit path is never exercised (a record already in hits is "
        "excluded from anti-hit consideration for itself)"
    )

    anti_matches = [h for h in response.anti_hits if h.record_id == contradicting.id]
    assert len(anti_matches) == 1, (
        f"contradicting record not surfaced as an anti-hit: "
        f"{[(h.record_id, h.reason) for h in response.anti_hits]}"
    )
    anti_hit = anti_matches[0]
    assert anti_hit.epistemic_status == "hypothesis", (
        f"end-to-end recall_for_response anti-hit dropped epistemic_status, "
        f"got {anti_hit.epistemic_status!r}"
    )
    assert anti_hit.literal_surface == contradicting.literal_surface


def test_recall_for_benchmark_graph_view_hit_carries_epistemic_status(tmp_path):
    """recall_for_benchmark has no session_id-gated enrichment loop the way
    recall_for_response does -- before the fix it returned
    _apply_post_rank_pipeline's output verbatim, so a primary hit sourced
    from the graph-view records_cache (SimpleRecordView, no
    epistemic_status field) rendered None. Prove the shared backfill helper
    call at the end of this path recovers the real stored value.
    """
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "benchmark-store")
    for i in range(10):
        store.insert(_mk_rec(f"unrelated filler record {i}", _random_vec(8000 + i)))

    embedder = Embedder()
    target_text = "alice confirmed the audit passed cleanly"
    target = _mk_rec(target_text, list(embedder.embed(target_text)), "estimate")
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
    assert hit.epistemic_status == "estimate", (
        f"graph-view-sourced benchmark hit dropped epistemic_status, "
        f"got {hit.epistemic_status!r}"
    )
    assert hit.literal_surface == target_text


def test_pipeline_scored_hits_carry_epistemic_status(tmp_path):
    """Site 3: the main scored-hits loop inside recall_for_response."""
    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "scored-hits-store")
    for i in range(10):
        store.insert(_mk_rec(f"unrelated filler record {i}", _random_vec(5000 + i)))

    embedder = Embedder()
    target_text = "alice confirmed the deployment finished successfully"
    target = _mk_rec(target_text, list(embedder.embed(target_text)), "fact")
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
    assert hit.epistemic_status == "fact"
    assert hit.literal_surface == target_text


def test_pipeline_pending_recency_marker_carries_epistemic_status(tmp_path):
    """Site 4: the pending-recency marker union inside recall_for_response.

    insert_pending_row does not accept an epistemic_status argument, so the
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
    expected_status = marker_rec.epistemic_status

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
    assert hit.epistemic_status == expected_status
    assert hit.literal_surface == "alice's pending marker just captured"


def test_memoryhit_construction_site_count_and_keyword_ast_guard():
    """Source-wide guard: exactly 8 MemoryHit(...) call sites across
    pipeline.py, retrieve.py, and core/__init__.py, and every one of them
    carries both the epistemic_status and salience_level keywords. Catches
    a future 9th site, or an edit that drops either keyword from an
    existing one, independent of whether that site is otherwise reachable
    end to end above."""
    targets = [
        _SRC_ROOT / "pipeline.py",
        _SRC_ROOT / "retrieve.py",
        _SRC_ROOT / "core" / "__init__.py",
    ]

    sites: list[tuple[str, int]] = []
    missing_kw: list[tuple[str, int]] = []
    missing_kw2: list[tuple[str, int]] = []

    for path in targets:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name != "MemoryHit":
                continue
            sites.append((str(path.relative_to(_REPO_ROOT)), node.lineno))
            if not any(kw.arg == "epistemic_status" for kw in node.keywords):
                missing_kw.append((str(path.relative_to(_REPO_ROOT)), node.lineno))
            if not any(kw.arg == "salience_level" for kw in node.keywords):
                missing_kw2.append((str(path.relative_to(_REPO_ROOT)), node.lineno))

    assert len(sites) == 8, (
        f"expected exactly 8 MemoryHit(...) construction sites across "
        f"pipeline.py/retrieve.py/core/__init__.py, found {len(sites)}: {sites}"
    )
    assert not missing_kw, (
        f"MemoryHit construction sites missing the epistemic_status keyword: {missing_kw}"
    )
    assert not missing_kw2, (
        f"MemoryHit construction sites missing the salience_level keyword: {missing_kw2}"
    )


_DRIVER_PARAMS = [
    pytest.param("stdlib", id="stdlib"),
    pytest.param("lilli", id="lilli"),
]


def _set_driver(monkeypatch: pytest.MonkeyPatch, driver: str) -> None:
    if driver == "stdlib":
        monkeypatch.delenv("LILLI_STORAGE_DRIVER", raising=False)
    else:
        pytest.importorskip("iai_mcp_native")
        monkeypatch.setenv("LILLI_STORAGE_DRIVER", driver)


@pytest.mark.parametrize("driver", _DRIVER_PARAMS)
def test_backfill_recovers_unknown_created_at_from_db(tmp_path, monkeypatch, driver):
    """A graph-node payload missing `created_at` must represent as
    `SimpleRecordView.created_at is None` (not a fabricated `now()`), and
    `_backfill_hit_metadata`'s `captured_at is None` guard must be the SOLE
    trigger that re-fetches the DB-authoritative value.
    """
    from iai_mcp.pipeline import SimpleRecordView, _backfill_hit_metadata, _payload_created_at

    _set_driver(monkeypatch, driver)
    store = MemoryStore(path=tmp_path / f"unknown-created-at-{driver}")

    frozen_dt = datetime(2024, 1, 15, 0, 0, 0, tzinfo=timezone.utc)
    frozen_iso = frozen_dt.isoformat()
    rec = _mk_rec("alice's frozen-timestamp record", _unit_vec(0), "fact")
    rec.created_at = frozen_dt
    rec.updated_at = frozen_dt
    store.insert(rec)

    node = {"embedding": rec.embedding, "surface": rec.literal_surface}
    view = SimpleRecordView(
        id=rec.id,
        embedding=node["embedding"],
        literal_surface=node["surface"],
        centrality=0.0,
        tier="episodic",
        created_at=_payload_created_at(node.get("created_at")),
    )
    assert view.created_at is None, (
        "representation vacuity guard: a node payload missing the "
        "'created_at' key must yield SimpleRecordView.created_at is None, "
        "not a fabricated now()"
    )

    hit = MemoryHit(
        record_id=rec.id, score=0.9, reason="probe",
        literal_surface=rec.literal_surface, adjacent_suggestions=[],
        epistemic_status="fact", salience_level="unflagged", session_id="s1",
    )
    assert hit.captured_at is None, (
        "precondition: captured_at is unset before the backfill runs"
    )

    _backfill_hit_metadata([hit], [], store)

    assert hit.captured_at == frozen_iso, (
        f"backfill must recover the DB-authoritative created_at, "
        f"got {hit.captured_at!r}, expected {frozen_iso!r}"
    )


@pytest.mark.perf
def test_epistemic_status_recall_latency_report(tmp_path):
    """Opt-in latency evidence: epistemic_status is a dict-key/dataclass-field
    passthrough, not new compute — no new store query, decrypt, or I/O was
    added by this plan's diff (a getattr-on-an-already-materialized-record
    read at each of the 8 sites, plus one dict-key write in _hit_to_json).
    Reported for the record, not gated on a wall-clock ceiling."""
    import time as _time

    from iai_mcp.embed import Embedder

    store = MemoryStore(path=tmp_path / "latency-store")
    for i in range(50):
        store.insert(_mk_rec(f"latency filler record {i}", _random_vec(7000 + i)))

    embedder = Embedder()
    g, a, rc = retrieve.build_runtime_graph(store)
    import iai_mcp.runtime_graph_cache as _rgc
    _rgc.save(store, a, rc)

    # Warm call — model + graph caches.
    pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue="latency filler record 0", session_id="s1",
        budget_tokens=1500, mode="concept",
    )

    t0 = _time.perf_counter()
    pipeline.recall_for_response(
        store=store, graph=g, assignment=a, rich_club=rc,
        embedder=embedder, cue="latency filler record 1", session_id="s1",
        budget_tokens=1500, mode="concept",
    )
    elapsed_ms = (_time.perf_counter() - t0) * 1000.0

    print(f"epistemic_status passthrough recall latency (warm): {elapsed_ms:.2f}ms")
