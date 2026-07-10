from __future__ import annotations

import hashlib
import random
from datetime import datetime, timedelta, timezone
from uuid import UUID, uuid4

import pytest

from iai_mcp.types import EMBED_DIM, MemoryRecord

class _DispatchEmbedder:

    DIM = EMBED_DIM

    def __init__(self) -> None:
        self.fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self.fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self.fixed:
            return list(self.fixed[text])
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _make_record(
    *,
    literal_surface: str,
    embedding: list[float],
    created_at: datetime,
    tier: str = "episodic",
    detail_level: int = 1,
    tags: list[str] | None = None,
    language: str = "en",
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=literal_surface,
        aaak_index="",
        embedding=list(embedding),
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
        created_at=created_at,
        updated_at=created_at,
        tags=tags or [],
        language=language,
    )

def _add_contradicts_edge_raw(
    store, *, src: UUID, dst: UUID, when: datetime | None = None
) -> None:
    tbl = store.db.open_table("edges")
    tbl.add([{
        "src": str(src),
        "dst": str(dst),
        "edge_type": "contradicts",
        "weight": 1.0,
        "updated_at": when or datetime.now(timezone.utc),
    }])

@pytest.fixture
def fresh_store(tmp_path):
    from iai_mcp.store import MemoryStore
    return MemoryStore(path=tmp_path / "lancedb")

def test_valid_from_set_from_created_at(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "valid_from anchor cue"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    rec = _make_record(
        literal_surface="anchor record about valid_from",
        embedding=cue_vec,
        created_at=t0,
    )
    rec.aaak_index = generate_aaak_index(rec)
    fresh_store.insert(rec)

    from iai_mcp import retrieve
    from iai_mcp.embed import embedder_for_store  # noqa: F401  — patched above
    from iai_mcp.pipeline import recall_for_response

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=embedder,
        cue=cue_text,
        session_id="t1",
        budget_tokens=1500,
        mode="concept",
    )
    assert len(resp.hits) >= 1
    hit = next((h for h in resp.hits if h.record_id == rec.id), None)
    assert hit is not None, f"target record not in hits: {[h.record_id for h in resp.hits]}"
    assert hit.valid_from == t0
    assert hit.valid_to is None

def test_valid_to_none_when_no_contradiction(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "preferences about editor"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec1 = _make_record(
        literal_surface="user prefers Vim for editing",
        embedding=cue_vec,
        created_at=t0,
    )
    rec1.aaak_index = generate_aaak_index(rec1)
    fresh_store.insert(rec1)

    t1 = t0 + timedelta(hours=1)
    rec2 = _make_record(
        literal_surface="unrelated note about coffee",
        embedding=embedder.embed("unrelated note about coffee"),
        created_at=t1,
    )
    rec2.aaak_index = generate_aaak_index(rec2)
    fresh_store.insert(rec2)

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store, graph=graph, assignment=assignment, rich_club=rc,
        embedder=embedder, cue=cue_text, session_id="t2",
        budget_tokens=1500, mode="concept",
    )
    hit = next((h for h in resp.hits if h.record_id == rec1.id), None)
    assert hit is not None
    assert hit.valid_to is None

def test_valid_to_set_when_newer_contradiction(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "editor preference"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec_old = _make_record(
        literal_surface="user prefers Vim",
        embedding=cue_vec,
        created_at=t0,
    )
    rec_old.aaak_index = generate_aaak_index(rec_old)
    fresh_store.insert(rec_old)

    t1 = t0 + timedelta(days=30)
    rec_new = _make_record(
        literal_surface="user switched to Helix",
        embedding=cue_vec,
        created_at=t1,
    )
    rec_new.aaak_index = generate_aaak_index(rec_new)
    fresh_store.insert(rec_new)
    _add_contradicts_edge_raw(fresh_store, src=rec_old.id, dst=rec_new.id)

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store, graph=graph, assignment=assignment, rich_club=rc,
        embedder=embedder, cue=cue_text, session_id="t3",
        budget_tokens=4000, mode="concept",
    )
    hit_old = next((h for h in resp.hits if h.record_id == rec_old.id), None)
    hit_new = next((h for h in resp.hits if h.record_id == rec_new.id), None)
    assert hit_old is not None and hit_new is not None, (
        f"missing target record in hits: ids={[h.record_id for h in resp.hits]}"
    )
    assert hit_old.valid_to == t1, (
        f"old.valid_to should be new.created_at ({t1}), got {hit_old.valid_to}"
    )
    assert hit_new.valid_to is None, (
        f"new has no newer contradicter; expected None, got {hit_new.valid_to}"
    )

def test_older_contradiction_ignored_for_valid_to(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "anomaly cue"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=10)
    rec_older = _make_record(
        literal_surface="older anomaly target",
        embedding=cue_vec,
        created_at=t0,
    )
    rec_older.aaak_index = generate_aaak_index(rec_older)
    fresh_store.insert(rec_older)

    rec = _make_record(
        literal_surface="newer record anomaly main",
        embedding=cue_vec,
        created_at=t1,
    )
    rec.aaak_index = generate_aaak_index(rec)
    fresh_store.insert(rec)

    _add_contradicts_edge_raw(fresh_store, src=rec.id, dst=rec_older.id)

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store, graph=graph, assignment=assignment, rich_club=rc,
        embedder=embedder, cue=cue_text, session_id="t4",
        budget_tokens=4000, mode="concept",
    )
    hit = next((h for h in resp.hits if h.record_id == rec.id), None)
    assert hit is not None
    assert hit.valid_to is None, (
        f"older dst must be ignored; expected None, got {hit.valid_to}"
    )

def test_multiple_contradictions_use_oldest_newer(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "double contradiction cue"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=5)
    t2 = t0 + timedelta(days=10)

    rec_orig = _make_record(
        literal_surface="original belief X",
        embedding=cue_vec,
        created_at=t0,
    )
    rec_orig.aaak_index = generate_aaak_index(rec_orig)
    fresh_store.insert(rec_orig)

    rec_a = _make_record(
        literal_surface="first contradiction A",
        embedding=cue_vec,
        created_at=t1,
    )
    rec_a.aaak_index = generate_aaak_index(rec_a)
    fresh_store.insert(rec_a)

    rec_b = _make_record(
        literal_surface="second contradiction B",
        embedding=cue_vec,
        created_at=t2,
    )
    rec_b.aaak_index = generate_aaak_index(rec_b)
    fresh_store.insert(rec_b)

    _add_contradicts_edge_raw(fresh_store, src=rec_orig.id, dst=rec_a.id)
    _add_contradicts_edge_raw(fresh_store, src=rec_orig.id, dst=rec_b.id)

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store, graph=graph, assignment=assignment, rich_club=rc,
        embedder=embedder, cue=cue_text, session_id="t5",
        budget_tokens=4000, mode="concept",
    )
    hit_orig = next((h for h in resp.hits if h.record_id == rec_orig.id), None)
    assert hit_orig is not None
    assert hit_orig.valid_to == t1, (
        f"valid_to should be MIN of newer contradicters' created_at "
        f"({t1}); got {hit_orig.valid_to}"
    )

def test_stale_record_downweighted_not_hidden(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index
    from iai_mcp.retrieve import STALE_DOWNWEIGHT_FACTOR
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "downweight test cue"
    surface_text = "shared verbatim surface for stale vs fresh"
    surface_vec = embedder.embed(surface_text)
    embedder.set_fixed(cue_text, surface_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=30)

    rec_old = _make_record(
        literal_surface=surface_text,
        embedding=surface_vec,
        created_at=t0,
    )
    rec_old.aaak_index = generate_aaak_index(rec_old)
    fresh_store.insert(rec_old)

    rec_new = _make_record(
        literal_surface=surface_text,
        embedding=surface_vec,
        created_at=t1,
    )
    rec_new.aaak_index = generate_aaak_index(rec_new)
    fresh_store.insert(rec_new)

    _add_contradicts_edge_raw(fresh_store, src=rec_old.id, dst=rec_new.id)

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store, graph=graph, assignment=assignment, rich_club=rc,
        embedder=embedder, cue=cue_text, session_id="t6",
        budget_tokens=4000, mode="concept",
    )
    hit_old = next((h for h in resp.hits if h.record_id == rec_old.id), None)
    hit_new = next((h for h in resp.hits if h.record_id == rec_new.id), None)
    assert hit_old is not None and hit_new is not None, (
        f"both records should be in hits: ids={[h.record_id for h in resp.hits]}"
    )
    assert hit_old.valid_to is not None, "stale record should carry valid_to"
    assert hit_old.score == pytest.approx(
        hit_new.score * STALE_DOWNWEIGHT_FACTOR, rel=1e-3
    ), (
        f"stale should be downweighted: expected {hit_new.score * STALE_DOWNWEIGHT_FACTOR}, "
        f"got {hit_old.score} (fresh score={hit_new.score})"
    )

def test_downweight_reranks_order(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "reranking probe cue alpha"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=30)

    rec_stale = _make_record(
        literal_surface="stale answer surface text",
        embedding=cue_vec,
        created_at=t0,
    )
    rec_stale.aaak_index = generate_aaak_index(rec_stale)
    fresh_store.insert(rec_stale)

    fresh_vec = list(cue_vec)
    for i in range(8):
        fresh_vec[i] = -fresh_vec[i]
    nrm = sum(x * x for x in fresh_vec) ** 0.5
    if nrm > 0:
        fresh_vec = [x / nrm for x in fresh_vec]

    rec_fresh = _make_record(
        literal_surface="fresh alternative answer surface text",
        embedding=fresh_vec,
        created_at=t1,
    )
    rec_fresh.aaak_index = generate_aaak_index(rec_fresh)
    fresh_store.insert(rec_fresh)

    _add_contradicts_edge_raw(fresh_store, src=rec_stale.id, dst=rec_fresh.id)

    from iai_mcp import core
    response = core.dispatch(
        fresh_store, "memory_recall",
        {"cue": cue_text, "session_id": "t7", "cue_embedding": cue_vec,
         "budget_tokens": 4000},
    )
    hits = response["hits"]
    assert len(hits) >= 2, f"expected both records in hits, got {len(hits)}"

    ids = [h["record_id"] for h in hits]
    pos_stale = ids.index(str(rec_stale.id))
    pos_fresh = ids.index(str(rec_fresh.id))
    assert pos_fresh < pos_stale, (
        f"fresh should outrank stale after downweight; "
        f"stale_pos={pos_stale} fresh_pos={pos_fresh} "
        f"stale_score={hits[pos_stale]['score']} fresh_score={hits[pos_fresh]['score']}"
    )

def test_episodic_schema_byte_identical(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    records_tbl = fresh_store.db.open_table("records")
    before_schema_repr = repr(records_tbl.schema)

    cue_text = "schema invariance probe"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rec = _make_record(
        literal_surface="schema invariance record",
        embedding=cue_vec,
        created_at=t0,
    )
    rec.aaak_index = generate_aaak_index(rec)
    fresh_store.insert(rec)

    from iai_mcp.retrieve import contradict
    contradict(
        fresh_store,
        original_id=rec.id,
        new_fact="updated belief after schema probe",
        new_embedding=cue_vec,
    )

    from iai_mcp import core
    _ = core.dispatch(
        fresh_store, "memory_recall",
        {"cue": cue_text, "session_id": "t8", "cue_embedding": cue_vec},
    )

    after_records_tbl = fresh_store.db.open_table("records")
    after_schema_repr = repr(after_records_tbl.schema)
    assert before_schema_repr == after_schema_repr, (
        f"records table schema must be byte-identical (write-once preserved); "
        f"before:\n{before_schema_repr}\nafter:\n{after_schema_repr}"
    )

def test_json_wire_carries_new_keys(fresh_store, monkeypatch):
    from iai_mcp import core
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "wire schema check cue"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=30)

    rec_old = _make_record(
        literal_surface="old wire-test record",
        embedding=cue_vec,
        created_at=t0,
    )
    rec_old.aaak_index = generate_aaak_index(rec_old)
    fresh_store.insert(rec_old)

    rec_new = _make_record(
        literal_surface="new wire-test record",
        embedding=cue_vec,
        created_at=t1,
    )
    rec_new.aaak_index = generate_aaak_index(rec_new)
    fresh_store.insert(rec_new)
    _add_contradicts_edge_raw(fresh_store, src=rec_old.id, dst=rec_new.id)

    response = core.dispatch(
        fresh_store, "memory_recall",
        {"cue": cue_text, "session_id": "t9", "cue_embedding": cue_vec,
         "budget_tokens": 4000},
    )
    assert response["hits"], "expected non-empty hits"
    for entry in response["hits"]:
        assert "valid_from" in entry, f"hit missing valid_from: {entry}"
        assert "valid_to" in entry, f"hit missing valid_to: {entry}"
        if entry["valid_from"] is not None:
            assert isinstance(entry["valid_from"], str)
            datetime.fromisoformat(entry["valid_from"])
        if entry["valid_to"] is not None:
            assert isinstance(entry["valid_to"], str)
            datetime.fromisoformat(entry["valid_to"])

    for entry in response["anti_hits"]:
        assert "valid_from" in entry, f"anti_hit missing valid_from: {entry}"
        assert "valid_to" in entry, f"anti_hit missing valid_to: {entry}"

def test_reason_string_marked_stale(fresh_store, monkeypatch):
    from iai_mcp import embed as _embed_mod
    from iai_mcp.aaak import generate_aaak_index
    from iai_mcp import retrieve
    from iai_mcp.pipeline import recall_for_response

    embedder = _DispatchEmbedder()
    monkeypatch.setattr(_embed_mod, "embedder_for_store", lambda _s: embedder)

    cue_text = "reason suffix cue"
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    t1 = t0 + timedelta(days=30)
    surface_text = "reason suffix shared surface text"

    rec_old = _make_record(
        literal_surface=surface_text,
        embedding=cue_vec,
        created_at=t0,
    )
    rec_old.aaak_index = generate_aaak_index(rec_old)
    fresh_store.insert(rec_old)

    rec_new = _make_record(
        literal_surface=surface_text,
        embedding=cue_vec,
        created_at=t1,
    )
    rec_new.aaak_index = generate_aaak_index(rec_new)
    fresh_store.insert(rec_new)
    _add_contradicts_edge_raw(fresh_store, src=rec_old.id, dst=rec_new.id)

    graph, assignment, rc = retrieve.build_runtime_graph(fresh_store)
    resp = recall_for_response(
        store=fresh_store, graph=graph, assignment=assignment, rich_club=rc,
        embedder=embedder, cue=cue_text, session_id="t10",
        budget_tokens=4000, mode="concept",
    )
    hit_old = next((h for h in resp.hits if h.record_id == rec_old.id), None)
    hit_new = next((h for h in resp.hits if h.record_id == rec_new.id), None)
    assert hit_old is not None and hit_new is not None
    assert " · stale" in hit_old.reason, (
        f"stale hit reason missing ' · stale' suffix: {hit_old.reason!r}"
    )
    assert " · stale" not in hit_new.reason, (
        f"fresh hit reason should NOT carry stale marker: {hit_new.reason!r}"
    )
