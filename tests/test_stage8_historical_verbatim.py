from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4


from iai_mcp.types import EMBED_DIM, MemoryRecord

class _BenchEmbedder:

    DIM = EMBED_DIM

    def __init__(self, base_seed: int = 0) -> None:
        self._base_seed = base_seed
        self._fixed: dict[str, list[float]] = {}

    def set_fixed(self, text: str, vec: list[float]) -> None:
        self._fixed[text] = list(vec)

    def embed(self, text: str) -> list[float]:
        if text in self._fixed:
            return list(self._fixed[text])
        import hashlib
        import random
        digest = hashlib.sha256(
            f"{self._base_seed}:{text}".encode("utf-8")
        ).hexdigest()
        rng = random.Random(int(digest[:16], 16))
        v = [rng.random() * 2 - 1 for _ in range(self.DIM)]
        norm = sum(x * x for x in v) ** 0.5
        return [x / norm for x in v] if norm > 0 else v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

def _make_rec(vec: list[float], text: str, tags: list[str]) -> MemoryRecord:
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
        tags=tags,
        language="en",
    )

def _high_cos_variant(base_vec: list[float], noise_seed: int, noise_scale: float = 0.20) -> list[float]:
    import hashlib
    import random
    digest = hashlib.sha256(f"{noise_seed}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    noise = [rng.random() * 2 - 1 for _ in range(len(base_vec))]
    n_norm = sum(x * x for x in noise) ** 0.5
    if n_norm > 0:
        noise = [x / n_norm for x in noise]
    mixed = [
        (1.0 - noise_scale) * b + noise_scale * n
        for b, n in zip(base_vec, noise, strict=False)
    ]
    m_norm = sum(x * x for x in mixed) ** 0.5
    if m_norm > 0:
        mixed = [x / m_norm for x in mixed]
    return mixed

def _seed_bench_scenario(tmp_path, n_filler: int = 8):
    from iai_mcp.retrieve import build_runtime_graph, contradict
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _BenchEmbedder(base_seed=24)

    cue_text = "Quote the original ETA wording."
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    gold_vec = _high_cos_variant(cue_vec, noise_seed=1001, noise_scale=0.30)
    gold_text = "The fix ships in week 14."
    embedder.set_fixed(gold_text, gold_vec)
    gold = _make_rec(gold_vec, gold_text, tags=["topic:bug_fix_eta"])
    store.insert(gold)
    gold_id_post_insert = gold.id

    wrong_vec = _high_cos_variant(cue_vec, noise_seed=2002, noise_scale=0.15)
    corr_text = "Fix ETA revised: week 18."
    embedder.set_fixed(corr_text, wrong_vec)
    receipt = contradict(store, gold_id_post_insert, corr_text, list(wrong_vec))
    wrong_id = receipt.new_record_id
    assert wrong_id != gold_id_post_insert, (
        f"test fixture invariant broken: contradict() deduped WRONG into "
        f"GOLD (id={wrong_id}). Increase noise_scale or change seeds."
    )

    no_edge_vec = _high_cos_variant(cue_vec, noise_seed=3003, noise_scale=0.35)
    no_edge_text = "Unrelated auth tokens are rotated weekly."
    embedder.set_fixed(no_edge_text, no_edge_vec)
    no_edge = _make_rec(no_edge_vec, no_edge_text, tags=["topic:auth"])
    store.insert(no_edge)
    no_edge_id_post_insert = no_edge.id

    tag_pool = [
        ["topic:auth"], ["topic:db"], ["topic:web"],
        ["topic:net"], ["topic:cli"],
    ]
    for i in range(n_filler):
        vec = embedder.embed(f"filler-{i}")
        tags = list(tag_pool[i % len(tag_pool)])
        rec = _make_rec(vec, text=f"unrelated fact {i}", tags=tags)
        store.insert(rec)

    graph, assignment, rich_club = build_runtime_graph(store)
    return (
        store, embedder, graph, assignment, rich_club,
        gold_id_post_insert, wrong_id, no_edge_id_post_insert, cue_vec,
    )

def test_bench_failing_cue_corrector_ranks_above_original(tmp_path):
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, _cue_vec) = _seed_bench_scenario(tmp_path)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="Quote the original ETA wording.",
        session_id="bench-probe",
        k_hits=10, mode="concept",
    )
    assert len(resp.hits) >= 2, (
        f"need at least 2 hits to compare ranks; got {len(resp.hits)}"
    )

    hit_ids = [h.record_id for h in resp.hits]
    assert gold_id in hit_ids, (
        f"GOLD (superseded original) must be in top-10 hits "
        f"(historical_verbatim@10=1.000); not found. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface) for h in resp.hits]}"
    )
    assert wrong_id in hit_ids, (
        f"Corrector (current truth) must also be present in hits. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface) for h in resp.hits]}"
    )
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"Corrector (current truth) must rank above the superseded original "
        f"(current-fact primacy). corrector_rank={wrong_rank} gold_rank={gold_rank}. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface, round(h.score, 3)) for h in resp.hits[:5]]}"
    )

def test_superseded_original_in_top10_below_corrector_buried_cosine(tmp_path):
    from iai_mcp.retrieve import build_runtime_graph, contradict
    from iai_mcp.store import MemoryStore
    from iai_mcp.pipeline import recall_for_benchmark

    store = MemoryStore(path=tmp_path / "lancedb")
    embedder = _BenchEmbedder(base_seed=77)

    cue_text = "Quote the original ETA wording."
    cue_vec = embedder.embed(cue_text)
    embedder.set_fixed(cue_text, cue_vec)

    gold_vec = _high_cos_variant(cue_vec, noise_seed=5101, noise_scale=0.62)
    gold_text = "The fix ships in week 14."
    embedder.set_fixed(gold_text, gold_vec)
    gold = _make_rec(gold_vec, gold_text, tags=["topic:bug_fix_eta"])
    store.insert(gold)
    gold_id = gold.id

    wrong_vec = _high_cos_variant(cue_vec, noise_seed=6202, noise_scale=0.12)
    corr_text = "Fix ETA revised: week 18."
    embedder.set_fixed(corr_text, wrong_vec)
    receipt = contradict(store, gold_id, corr_text, list(wrong_vec))
    wrong_id = receipt.new_record_id
    assert wrong_id != gold_id

    for i in range(20):
        dvec = _high_cos_variant(cue_vec, noise_seed=7000 + i, noise_scale=0.40)
        embedder.set_fixed(f"distractor-{i}", dvec)
        store.insert(_make_rec(dvec, text=f"distractor fact {i}", tags=["topic:misc"]))

    graph, assignment, rich_club = build_runtime_graph(store)

    import numpy as np
    cv = np.asarray(cue_vec, dtype=np.float32); cv /= np.linalg.norm(cv) + 1e-12
    cos_scored = []
    for r in store.all_records():
        v = np.asarray(r.embedding, dtype=np.float32); v /= np.linalg.norm(v) + 1e-12
        cos_scored.append((float(v @ cv), r.id))
    cos_scored.sort(key=lambda t: -t[0])
    gold_cos_rank = next(i for i, (_, rid) in enumerate(cos_scored, 1) if rid == gold_id)
    assert gold_cos_rank > 5, (
        f"fixture invariant broken: GOLD must be buried (cos_rank>5) for this "
        f"test to exercise the anchor; got cos_rank={gold_cos_rank}"
    )

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=cue_text, session_id="bench-probe",
        k_hits=10, mode="concept",
    )
    hit_ids = [h.record_id for h in resp.hits]

    assert gold_id in hit_ids, (
        f"Anchor must surface the buried superseded original (cos_rank="
        f"{gold_cos_rank}) into top-10 on the historical cue; not found. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface, round(h.score,3)) for h in resp.hits]}"
    )
    assert wrong_id in hit_ids, (
        f"Corrector (current truth) must be in results on historical cue."
    )
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"Corrector must rank above the superseded original (current-fact primacy). "
        f"corrector_rank={wrong_rank} gold_rank={gold_rank}. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface, round(h.score,3)) for h in resp.hits[:5]]}"
    )

def test_neutral_cue_does_not_apply_downweight(tmp_path):
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, _cue_vec) = _seed_bench_scenario(tmp_path)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="What's the ETA?",
        session_id="bench-probe",
        k_hits=5, mode="concept",
    )

    gold_hit = next((h for h in resp.hits if h.record_id == gold_id), None)
    wrong_hit = next((h for h in resp.hits if h.record_id == wrong_id), None)
    assert gold_hit is not None, "GOLD must appear in neutral cue results"
    if wrong_hit is not None:
        gap = abs(gold_hit.score - wrong_hit.score)
        from iai_mcp.pipeline import HISTORICAL_VERBATIM_DOWNWEIGHT
        assert gap < HISTORICAL_VERBATIM_DOWNWEIGHT - 0.02, (
            f"On NEUTRAL cue the score gap should NOT reflect a "
            f"~{HISTORICAL_VERBATIM_DOWNWEIGHT} downweight; got gap={gap:.4f}. "
            f"GOLD={gold_hit.score:.4f} WRONG={wrong_hit.score:.4f}"
        )

def test_russian_historical_cue_corrector_ranks_above_original(tmp_path):
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, cue_vec) = _seed_bench_scenario(tmp_path)
    ru_cue = "приведи оригинальную формулировку"
    embedder.set_fixed(ru_cue, cue_vec)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=ru_cue,
        session_id="bench-probe",
        k_hits=10, mode="concept",
    )
    hit_ids = [h.record_id for h in resp.hits]
    assert gold_id in hit_ids, (
        f"RU historical cue: GOLD (superseded original) must be in top-10; not found. "
        f"Hits: {[(str(h.record_id)[:8], h.literal_surface) for h in resp.hits]}"
    )
    assert wrong_id in hit_ids, (
        f"RU historical cue: corrector must also be in top-10."
    )
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"RU historical cue: corrector must rank above the superseded original "
        f"(current-fact primacy). corrector_rank={wrong_rank} gold_rank={gold_rank}."
    )

def test_record_without_contradicts_edge_unaffected_by_downweight(tmp_path):
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     _gold_id, _wrong_id, no_edge_id, cue_vec) = _seed_bench_scenario(tmp_path)

    neutral_cue = "current eta status"
    embedder.set_fixed(neutral_cue, cue_vec)
    from iai_mcp.cue_router import _classify_cue
    assert _classify_cue(neutral_cue)[1] != "historical_verbatim", (
        "test fixture invariant: neutral cue must not classify as historical"
    )

    hist = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="Quote the original ETA wording.",
        session_id="bench-probe",
        k_hits=20, mode="concept",
    )
    neutral = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=neutral_cue,
        session_id="bench-probe",
        k_hits=20, mode="concept",
    )
    no_edge_hist = next((h for h in hist.hits if h.record_id == no_edge_id), None)
    no_edge_neutral = next(
        (h for h in neutral.hits if h.record_id == no_edge_id), None,
    )
    assert no_edge_hist is not None, (
        "NO_EDGE record should still appear in the ranked list (k_hits=20 "
        "covers all 11 records) — the historical machinery is targeted, not blanket"
    )
    assert no_edge_neutral is not None, "NO_EDGE must appear on the neutral cue"
    assert abs(no_edge_hist.score - no_edge_neutral.score) < 1e-6, (
        f"NO_EDGE score changed between historical and neutral cues "
        f"(hist={no_edge_hist.score:.6f} neutral={no_edge_neutral.score:.6f}); "
        f"it has no contradicts edge so the superseded-original anchor "
        f"must not touch it."
    )

def test_historical_verbatim_downweight_constant_is_module_level(tmp_path):
    from iai_mcp import pipeline as _pipeline_mod

    assert hasattr(_pipeline_mod, "HISTORICAL_VERBATIM_DOWNWEIGHT"), (
        "pipeline.py must export HISTORICAL_VERBATIM_DOWNWEIGHT module constant"
    )
    assert isinstance(_pipeline_mod.HISTORICAL_VERBATIM_DOWNWEIGHT, float)
    assert 0.0 < _pipeline_mod.HISTORICAL_VERBATIM_DOWNWEIGHT < 1.0, (
        f"HISTORICAL_VERBATIM_DOWNWEIGHT must be in (0, 1) for stability, "
        f"got {_pipeline_mod.HISTORICAL_VERBATIM_DOWNWEIGHT}"
    )

def test_bench_harness_calls_classify_cue_for_intent(tmp_path):
    from iai_mcp.pipeline import recall_for_benchmark

    (store, embedder, graph, assignment, rich_club,
     gold_id, wrong_id, _no_edge_id, _cue_vec) = _seed_bench_scenario(tmp_path)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue="Quote the original ETA wording.",
        session_id="bench-probe",
        k_hits=200,
        mode="concept",
    )
    assert len(resp.hits) >= 2
    hit_ids = [h.record_id for h in resp.hits]
    assert gold_id in hit_ids[:10], (
        "bench-shaped call (mode='concept', k_hits=200) must route "
        "historical_verbatim intent and include the original in top-10"
    )
    assert wrong_id in hit_ids, "corrector must be present in results"
    gold_rank = hit_ids.index(gold_id)
    wrong_rank = hit_ids.index(wrong_id)
    assert wrong_rank < gold_rank, (
        f"bench-shaped call: corrector must rank above original (current-fact primacy). "
        f"corrector_rank={wrong_rank} gold_rank={gold_rank}."
    )
