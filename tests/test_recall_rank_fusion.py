"""Rank fusion: knowledge-aware boost + warm lexical lane in recall.

Two levers, both bounded and killable: (1) knowledge-grade sources
(semantic summaries, doc:* chunks) get a soft multiplier at final rank —
a nudge past equal raw turns, never a filter; (2) a warm BM25 lane fuses
into the rank and the seed set so a vocabulary-gap target can surface,
while the recall path never pays the index rebuild.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from uuid import uuid4

from iai_mcp import core
from iai_mcp.store import MemoryStore, flush_record_buffer
from iai_mcp.types import EMBED_DIM, MemoryRecord
from tests._helpers import stub_embedder_for_store


def _unit(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = 1.0
    return v


def _mix(i: int, j: int, cos_to_i: float) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i] = cos_to_i
    v[j] = math.sqrt(1.0 - cos_to_i * cos_to_i)
    return v


class _MappedEmbedder:
    """Deterministic embedder: exact vectors for registered surfaces, a far
    orthogonal vector for anything else. The store re-embeds surfaces on
    insert, so crafted cosines must come through the embedder, not the
    record field."""

    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self._mapping = dict(mapping)

    def embed(self, text: str) -> list[float]:
        vec = self._mapping.get(text)
        if vec is not None:
            return list(vec)
        fallback = [0.0] * EMBED_DIM
        fallback[EMBED_DIM - 1] = 1.0
        return fallback


def _install_embedder(monkeypatch, mapping: dict[str, list[float]]) -> None:
    stub_embedder_for_store(monkeypatch, _MappedEmbedder(mapping))


def _rec(text: str, emb: list[float], *, tier: str = "episodic", tags: list[str] | None = None) -> MemoryRecord:
    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
        literal_surface=text,
        aaak_index="",
        embedding=emb,
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
    )


def _dispatch(store: MemoryStore, cue: str, cue_vec: list[float]) -> dict:
    resp = core.dispatch(store, "memory_recall", {
        "cue": cue,
        "session_id": "rank-fusion-test",
        "budget_tokens": 2000,
        "cue_embedding": cue_vec,
    })
    assert "error" not in resp, f"recall dispatch errored: {resp.get('error')}"
    return resp


def test_semantic_summary_outranks_when_literal_preservation_relaxed(tmp_path, monkeypatch):
    """The semantic-summary boost respects the literal_preservation knob:
    it fires only when the profile relaxes past the strong default."""
    monkeypatch.setitem(core._profile_state, "literal_preservation", "medium")
    raw_text = "a raw conversational turn about deployment gates"
    knowledge_text = "the deployment gate requires a green suite"
    _install_embedder(monkeypatch, {
        raw_text: _unit(0), knowledge_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    knowledge = _rec(knowledge_text, _mix(0, 1, 0.98), tier="semantic")
    store.insert(raw)
    store.insert(knowledge)
    flush_record_buffer(store)

    resp = _dispatch(store, "rank fusion knowledge probe", _unit(0))
    assert resp["hits"][0]["record_id"] == str(knowledge.id), (
        "a 2%-closer raw turn must lose to knowledge under the default boost: "
        f"{[h['record_id'] for h in resp['hits']]}"
    )


def test_semantic_boost_stands_down_under_strong_literal_preservation(tmp_path, monkeypatch):
    """Under the strong default the raw original must keep outranking the
    summary — literal_preservation IS the raw-vs-summary preference."""
    monkeypatch.setitem(core._profile_state, "literal_preservation", "strong")
    raw_text = "a raw conversational turn about deployment gates"
    knowledge_text = "the deployment gate requires a green suite"
    _install_embedder(monkeypatch, {
        raw_text: _unit(0), knowledge_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    knowledge = _rec(knowledge_text, _mix(0, 1, 0.98), tier="semantic")
    store.insert(raw)
    store.insert(knowledge)
    flush_record_buffer(store)

    resp = _dispatch(store, "rank fusion strong-lp probe", _unit(0))
    assert resp["hits"][0]["record_id"] == str(raw.id)


def test_doc_chunk_counts_as_knowledge(tmp_path, monkeypatch):
    raw_text = "raw chatter mentioning the calibration flow"
    doc_text = "calibration flow: step one, zero the sensor"
    _install_embedder(monkeypatch, {
        raw_text: _unit(0), doc_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    doc = _rec(doc_text, _mix(0, 1, 0.98), tags=["doc:manual.md"])
    store.insert(raw)
    store.insert(doc)
    flush_record_buffer(store)

    resp = _dispatch(store, "rank fusion doc probe", _unit(0))
    assert resp["hits"][0]["record_id"] == str(doc.id)


def test_tier_boost_env_disables(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_TIER_BOOST", "1.0")
    monkeypatch.setitem(core._profile_state, "literal_preservation", "medium")
    raw_text = "a raw conversational turn about deployment gates"
    knowledge_text = "the deployment gate requires a green suite"
    _install_embedder(monkeypatch, {
        raw_text: _unit(0), knowledge_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    knowledge = _rec(knowledge_text, _mix(0, 1, 0.98), tier="semantic")
    store.insert(raw)
    store.insert(knowledge)
    flush_record_buffer(store)

    resp = _dispatch(store, "rank fusion boost-off probe", _unit(0))
    assert resp["hits"][0]["record_id"] == str(raw.id), (
        "with the boost disabled the closer record must win on cosine"
    )


def _seed_vocab_gap_world(store: MemoryStore, monkeypatch):
    """Five weakly-related fillers plus one vocabulary-gap target whose
    embedding shares nothing with the cue but whose surface holds the rare
    identifier."""
    filler_texts = [
        f"weakly related note number {i} about general status" for i in range(5)
    ]
    target_text = "the zephyrblatt rotor calibration constant lives in the flux config"
    mapping = {t: _mix(0, 2 + i, 0.25) for i, t in enumerate(filler_texts)}
    mapping[target_text] = _unit(9)
    _install_embedder(monkeypatch, mapping)
    fillers = [_rec(t, mapping[t]) for t in filler_texts]
    target = _rec(target_text, _unit(9))
    for r in fillers + [target]:
        store.insert(r)
    flush_record_buffer(store)
    return fillers, target


def test_warm_lexical_lane_surfaces_vocab_gap_target(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "0.5")
    store = MemoryStore(path=tmp_path)
    _fillers, target = _seed_vocab_gap_world(store, monkeypatch)

    store.lexical_search("warm the lane", k=1)
    resp = _dispatch(store, "zephyrblatt_rotor config", _unit(0))
    assert resp["hits"][0]["record_id"] == str(target.id), (
        "the vocabulary-gap target must surface first via BM25 fusion: "
        f"{[h['record_id'] for h in resp['hits']]}"
    )


def test_lex_fusion_kill_switch(tmp_path, monkeypatch):
    monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "0.5")
    monkeypatch.setenv("IAI_MCP_LEX_FUSION_OFF", "true")
    store = MemoryStore(path=tmp_path)
    _fillers, target = _seed_vocab_gap_world(store, monkeypatch)

    store.lexical_search("warm the lane", k=1)
    resp = _dispatch(store, "zephyrblatt_rotor config", _unit(0))
    assert resp["hits"][0]["record_id"] != str(target.id), (
        "with fusion off the zero-cosine target must not lead"
    )


def test_recall_never_builds_the_lexical_index(tmp_path, monkeypatch):
    """A cold index yields nothing on the recall path — no inline build, no
    background thread, no store reads. Warmth comes from the scoped-search
    surface / nightly warm-up plus the per-insert feed."""
    store = MemoryStore(path=tmp_path)
    _seed_vocab_gap_world(store, monkeypatch)

    from iai_mcp.store import _lexical_index as _li

    def _forbidden_build(self, rows, generation):
        raise AssertionError("recall path must never build the lexical index")

    monkeypatch.setattr(_li.LexicalIndex, "build", _forbidden_build)

    resp = _dispatch(store, "zephyrblatt_rotor config", _unit(0))
    assert resp["hits"], "cold lexical lane must not break recall"


def test_insert_feed_keeps_warm_index_current(tmp_path, monkeypatch):
    """The per-insert feed restamps the generation, so fusion survives
    writes between full builds — including finding the freshly written
    record itself."""
    monkeypatch.setenv("IAI_MCP_LEX_MIN_IDF", "0.5")
    store = MemoryStore(path=tmp_path)
    _seed_vocab_gap_world(store, monkeypatch)
    store.lexical_search("warm the lane", k=1)

    late = _rec("a late note about the quarzfeder spring tension", _unit(20))
    store.insert(late)
    flush_record_buffer(store)

    pairs = store.lexical_query_warm("quarzfeder", min_idf=0.5)
    assert pairs and pairs[0][0] == str(late.id), (
        f"the fed record must be findable without a rebuild: {pairs}"
    )


def test_common_word_cue_gated_out_of_fusion(tmp_path, monkeypatch):
    store = MemoryStore(path=tmp_path)
    _seed_vocab_gap_world(store, monkeypatch)
    store.lexical_search("warm the lane", k=1)

    assert store.lexical_query_warm("status", min_idf=4.0) == [], (
        "a ubiquitous-token cue carries no lexical signal worth fusing"
    )
    assert store.lexical_query_warm("zephyrblatt", min_idf=0.5), (
        "a rare-token cue must pass the gate"
    )


def test_supplied_cue_embedding_drives_warm_path_rank(tmp_path, monkeypatch):
    """The tool schema promises the cue is embedded server-side ONLY when
    cue_embedding is absent. The cue text below is unknown to the embedder
    (its fallback is orthogonal to every record), so the crafted margins can
    reach the rank only through the supplied vector."""
    monkeypatch.setitem(core._profile_state, "literal_preservation", "medium")
    raw_text = "a raw conversational turn about deployment gates"
    knowledge_text = "the deployment gate requires a green suite"
    _install_embedder(monkeypatch, {
        raw_text: _unit(0), knowledge_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    knowledge = _rec(knowledge_text, _mix(0, 1, 0.98), tier="semantic")
    store.insert(raw)
    store.insert(knowledge)
    flush_record_buffer(store)

    resp = _dispatch(store, "cue text the embedder has never seen", _unit(0))
    by_id = {h["record_id"]: h for h in resp["hits"]}
    raw_hit = by_id.get(str(raw.id))
    know_hit = by_id.get(str(knowledge.id))
    assert raw_hit is not None and know_hit is not None, resp["hits"]
    assert "cos 1.000" in raw_hit["reason"], (
        f"supplied cue vector must reach scoring: {raw_hit['reason']}"
    )
    assert "cos 0.980" in know_hit["reason"], know_hit["reason"]


def test_zero_cue_embedding_falls_back_to_server_side_embed(tmp_path, monkeypatch):
    cue_text = "which gate colour is the deployment probe"
    raw_text = "a raw conversational turn about deployment gates"
    knowledge_text = "the deployment gate requires a green suite"
    _install_embedder(monkeypatch, {
        cue_text: _unit(0),
        raw_text: _unit(0), knowledge_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    knowledge = _rec(knowledge_text, _mix(0, 1, 0.98), tier="semantic")
    store.insert(raw)
    store.insert(knowledge)
    flush_record_buffer(store)

    resp = _dispatch(store, cue_text, [0.0] * EMBED_DIM)
    by_id = {h["record_id"]: h for h in resp["hits"]}
    raw_hit = by_id.get(str(raw.id))
    assert raw_hit is not None, resp["hits"]
    assert "cos 1.000" in raw_hit["reason"], (
        f"zero vector must defer to the server-side embed of the cue text: "
        f"{raw_hit['reason']}"
    )


def _cue_fallback_world(tmp_path, monkeypatch, cue_text):
    raw_text = "a raw conversational turn about deployment gates"
    knowledge_text = "the deployment gate requires a green suite"
    _install_embedder(monkeypatch, {
        cue_text: _unit(0),
        raw_text: _unit(0), knowledge_text: _mix(0, 1, 0.98),
    })
    store = MemoryStore(path=tmp_path)
    raw = _rec(raw_text, _unit(0))
    knowledge = _rec(knowledge_text, _mix(0, 1, 0.98), tier="semantic")
    store.insert(raw)
    store.insert(knowledge)
    flush_record_buffer(store)
    return store, raw


def test_wrong_length_cue_embedding_falls_back_to_server_side_embed(tmp_path, monkeypatch):
    cue_text = "which gate colour is the deployment probe"
    store, raw = _cue_fallback_world(tmp_path, monkeypatch, cue_text)
    resp = _dispatch(store, cue_text, [1.0] * 100)
    by_id = {h["record_id"]: h for h in resp["hits"]}
    raw_hit = by_id.get(str(raw.id))
    assert raw_hit is not None, resp["hits"]
    assert "cos 1.000" in raw_hit["reason"], (
        f"a wrong-dim vector must defer to the server-side embed: "
        f"{raw_hit['reason']}"
    )


def test_non_finite_cue_embedding_falls_back_to_server_side_embed(tmp_path, monkeypatch):
    cue_text = "which gate colour is the deployment probe"
    store, raw = _cue_fallback_world(tmp_path, monkeypatch, cue_text)
    resp = _dispatch(store, cue_text, [float("nan")] * EMBED_DIM)
    by_id = {h["record_id"]: h for h in resp["hits"]}
    raw_hit = by_id.get(str(raw.id))
    assert raw_hit is not None, resp["hits"]
    assert "cos 1.000" in raw_hit["reason"], (
        f"a NaN vector must defer to the server-side embed, never flatten "
        f"the head: {raw_hit['reason']}"
    )
