from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

import numpy as np
import psutil
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make
from _recall_helpers import diff_recall_quality_baseline_entry

import iai_mcp.pipeline as _pipeline_mod
from iai_mcp.embed import Embedder
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM, MemoryRecord

# Opt-in maintenance path: a normal run diffs fresh-vs-committed and fails
# on regression instead of silently resetting the baseline.
_UPDATE_BASELINE_ENV = "IAI_MCP_UPDATE_BASELINE"

RNG_SEED = 20260601
N_SMALL = 1_000
N_LARGE = 10_000
N_TRIALS = 12

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recall_quality_baseline.json"

LEXICAL_GENERIC_CUE = "hello"
LEXICAL_SPECIFIC_CUE = "specialized technical framework review"

_GOLD_TEXT_TEMPLATE = "reference gold doc {i}"

def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _make_gold_record(i: int, vec: list[float]) -> MemoryRecord:
    from datetime import datetime, timezone
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=_GOLD_TEXT_TEMPLATE.format(i=i),
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
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        tags=[],
        language="en",
    )

def _populate_bulk(store: MemoryStore, n: int, rng_seed: int = RNG_SEED) -> None:
    rng = np.random.default_rng(rng_seed)
    for i in range(n):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v = (v / np.linalg.norm(v)).tolist()
        rec = _make(
            text=f"User record {i} filler content for profiling harness",
            vec=v,
        )
        store.insert(rec)

def _reset_auto_depth() -> None:
    _pipeline_mod._last_recall_latency_ms = 0.0

def _p50_p95(samples: list[float]) -> tuple[float, float]:
    s = sorted(samples)
    p50 = s[len(s) // 2]
    p95 = s[int(len(s) * 0.95)]
    return p50, p95

def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")

def _print_table(title: str, rows: list[tuple]) -> None:
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")
    header = f"{'Stage/Cell':<44} {'p50 ms':>8} {'p95 ms':>8}"
    print(header)
    print("-" * 62)
    for label, p50, p95 in rows:
        print(f"{label:<44} {p50:>8.1f} {p95:>8.1f}")
    print(f"{'='*72}\n")

def _build_store(root: Path, n: int) -> MemoryStore:
    store_path = root / f"store-{n}"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    _populate_bulk(store, n, rng_seed=RNG_SEED)
    return store

def _time_build_runtime_graph_miss(store: MemoryStore) -> float:
    try:
        cache_path = Path(store.path) / "runtime_graph_cache.json"
        if cache_path.exists():
            cache_path.unlink()
    except Exception:
        pass
    t0 = time.perf_counter()
    from iai_mcp.retrieve import build_runtime_graph
    build_runtime_graph(store)
    return (time.perf_counter() - t0) * 1000.0

def _time_edges_topology_scan(store: MemoryStore) -> float:
    tbl = store.db.open_table("edges")
    t0 = time.perf_counter()
    _ = tbl.to_pandas()
    return (time.perf_counter() - t0) * 1000.0

def _time_temporal_validity_dirty(store: MemoryStore, insert_rec: MemoryRecord) -> float:
    from iai_mcp.retrieve import build_temporal_validity_maps, _tv_cache_dirty
    store.insert(insert_rec)
    _tv_cache_dirty[store] = True
    t0 = time.perf_counter()
    build_temporal_validity_maps(store)
    return (time.perf_counter() - t0) * 1000.0

def _time_temporal_validity_clean(store: MemoryStore) -> float:
    from iai_mcp.retrieve import build_temporal_validity_maps, _tv_cache_dirty
    _tv_cache_dirty[store] = True
    build_temporal_validity_maps(store)
    _tv_cache_dirty[store] = False
    t0 = time.perf_counter()
    build_temporal_validity_maps(store)
    return (time.perf_counter() - t0) * 1000.0

def _build_graph_for_store(store: MemoryStore):
    from iai_mcp.retrieve import build_runtime_graph
    return build_runtime_graph(store)

def _time_anti_hits_scan(store: MemoryStore, graph, cue_vec: list[float]) -> float:
    from iai_mcp.pipeline import _find_anti_hits
    from iai_mcp.types import MemoryHit
    hits = store.query_similar(cue_vec, k=5)
    memory_hits = [
        MemoryHit(
            record_id=r.id,
            score=s,
            reason="cosine",
            literal_surface=r.literal_surface or "",
            adjacent_suggestions=[],
        )
        for r, s in hits
    ]
    t0 = time.perf_counter()
    _find_anti_hits(memory_hits, store, graph, k=3)
    return (time.perf_counter() - t0) * 1000.0

def _time_schema_evidence_scan(store: MemoryStore) -> float:
    t0 = time.perf_counter()
    _ = store.db.open_table("edges").to_pandas()
    return (time.perf_counter() - t0) * 1000.0

def _time_profile_modulates_large_batch(store: MemoryStore) -> float:
    from iai_mcp.pipeline import PROFILE_SENTINEL_UUID
    dummy_ids = [UUID(int=9_000_000 + i) for i in range(6)]
    pairs = [(dummy_id, PROFILE_SENTINEL_UUID) for dummy_id in dummy_ids]
    deltas = [1.0] * 6
    t0 = time.perf_counter()
    store.boost_edges(pairs, edge_type="profile_modulates", delta=deltas)
    return (time.perf_counter() - t0) * 1000.0

def _time_embed(cue_text: str) -> float:
    embedder = Embedder()
    t0 = time.perf_counter()
    embedder.embed(cue_text)
    return (time.perf_counter() - t0) * 1000.0

def _time_end_to_end_recall(
    store: MemoryStore, graph, assignment, rich_club, embedder: Embedder, cue_text: str,
) -> float:
    from iai_mcp.pipeline import recall_for_response
    _reset_auto_depth()
    t0 = time.perf_counter()
    recall_for_response(
        store=store, graph=graph, assignment=assignment, rich_club=rich_club,
        embedder=embedder, cue=cue_text, session_id="user", budget_tokens=1500,
        mode="concept",
    )
    return (time.perf_counter() - t0) * 1000.0

@pytest.mark.slow
def test_per_stage_latency_profile(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    embedder = Embedder()

    embed_generic_ms = _time_embed(LEXICAL_GENERIC_CUE)
    embed_specific_ms = _time_embed(LEXICAL_SPECIFIC_CUE)

    results_rows = []

    for n_records, n_label in [(N_SMALL, "N=1k"), (N_LARGE, "N=10k")]:
        store = _build_store(tmp_path, n_records)
        graph, assignment, rich_club = _build_graph_for_store(store)

        cue_vec_generic = embedder.embed(LEXICAL_GENERIC_CUE)
        cue_vec_specific = embedder.embed(LEXICAL_SPECIFIC_CUE)

        for cue_text, cue_vec, cue_label in [
            (LEXICAL_GENERIC_CUE, cue_vec_generic, "lexical-generic"),
            (LEXICAL_SPECIFIC_CUE, cue_vec_specific, "lexical-specific"),
        ]:
            e2e_hit_samples = []
            for _ in range(N_TRIALS):
                ms = _time_end_to_end_recall(store, graph, assignment, rich_club, embedder, cue_text)
                e2e_hit_samples.append(ms)
            p50, p95 = _p50_p95(e2e_hit_samples)
            label = f"E2E-recall-HIT {cue_label} {n_label}"
            results_rows.append((label, p50, p95))

            miss_decrypt_samples = []
            for _ in range(N_TRIALS):
                ms = _time_build_runtime_graph_miss(store)
                miss_decrypt_samples.append(ms)
                _reset_auto_depth()
            p50, p95 = _p50_p95(miss_decrypt_samples)
            label = f"S1-decrypt-MISS {cue_label} {n_label}"
            results_rows.append((label, p50, p95))

            edges_topo_samples = []
            for _ in range(N_TRIALS):
                ms = _time_edges_topology_scan(store)
                edges_topo_samples.append(ms)
            p50, p95 = _p50_p95(edges_topo_samples)
            label = f"S2-edges-topology {n_label}"
            results_rows.append((label, p50, p95))

            tv_dirty_samples = []
            for i in range(N_TRIALS):
                dirty_rec = _make(
                    text=f"User profiling dirty insert {n_label} trial {i}",
                    vec=_random_vec(99_000 + i),
                )
                ms = _time_temporal_validity_dirty(store, dirty_rec)
                tv_dirty_samples.append(ms)
                _reset_auto_depth()
            p50, p95 = _p50_p95(tv_dirty_samples)
            label = f"S3a-temporal-validity-DIRTY {n_label}"
            results_rows.append((label, p50, p95))

            tv_clean_samples = []
            for _ in range(N_TRIALS):
                ms = _time_temporal_validity_clean(store)
                tv_clean_samples.append(ms)
            p50, p95 = _p50_p95(tv_clean_samples)
            label = f"S3b-temporal-validity-HIT {n_label}"
            results_rows.append((label, p50, p95))

            graph, assignment, rich_club = _build_graph_for_store(store)

            anti_hits_samples = []
            for _ in range(N_TRIALS):
                ms = _time_anti_hits_scan(store, graph, cue_vec)
                anti_hits_samples.append(ms)
                _reset_auto_depth()
            p50, p95 = _p50_p95(anti_hits_samples)
            label = f"S4-anti-hits-edges-scan {cue_label} {n_label}"
            results_rows.append((label, p50, p95))

            schema_ev_samples = []
            for _ in range(N_TRIALS):
                ms = _time_schema_evidence_scan(store)
                schema_ev_samples.append(ms)
            p50, p95 = _p50_p95(schema_ev_samples)
            label = f"S5-schema-evidence-edges-scan {n_label}"
            results_rows.append((label, p50, p95))

            profile_mod_samples = []
            for _ in range(N_TRIALS):
                ms = _time_profile_modulates_large_batch(store)
                profile_mod_samples.append(ms)
            p50, p95 = _p50_p95(profile_mod_samples)
            label = f"S6-profile-modulates-edges-scan {n_label}"
            results_rows.append((label, p50, p95))

    print(f"\n  Embed baseline: generic={embed_generic_ms:.1f}ms  specific={embed_specific_ms:.1f}ms")
    _print_table("Per-Stage Latency Profile (E2E + 6 Full-Table Scans)", results_rows)

    for label, p50, p95 in results_rows:
        assert np.isfinite(p50), f"Non-finite p50 for {label}"
        assert np.isfinite(p95), f"Non-finite p95 for {label}"
        assert p50 >= 0.0, f"Negative p50 for {label}"

def _exact_top_k(store: MemoryStore, cue_vec: list[float], k: int) -> list[str]:
    records_tbl = store.db.open_table("records")
    df = records_tbl.search().select(["id", "embedding", "embedding_pending"]).limit(
        int(records_tbl.count_rows())
    ).to_pandas()
    df = df[df["embedding_pending"].fillna(0).astype(int) == 0]
    if df.empty:
        return []
    ids = df["id"].tolist()
    embs = np.array([
        list(e) if hasattr(e, "__iter__") else [0.0] * EMBED_DIM
        for e in df["embedding"].tolist()
    ], dtype=np.float32)
    cue = np.asarray(cue_vec, dtype=np.float32)
    cnorm = float(np.linalg.norm(cue))
    if cnorm > 0:
        cue = cue / cnorm
    cos = np.matmul(embs, cue)
    order = np.argsort(-cos, kind="stable")[:k]
    return [str(ids[i]) for i in order]

def _ann_top_k(store: MemoryStore, cue_vec: list[float], k: int) -> list[str]:
    results = store.query_similar(cue_vec, k=k)
    return [str(r.id) for r, _s in results]

def _recall_at_k(ann_ids: list[str], exact_ids: list[str], k: int) -> float:
    ann_set = set(ann_ids[:k])
    exact_set = set(exact_ids[:k])
    if not exact_set:
        return 1.0
    return len(ann_set & exact_set) / len(exact_set)

def _rss_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / (1024 * 1024)

def _estimate_ann_top200_cosine_threshold(store: MemoryStore, cue_vec: list[float]) -> float:
    cue = np.asarray(cue_vec, dtype=np.float32)
    cue = cue / np.linalg.norm(cue)
    records_tbl = store.db.open_table("records")
    df = records_tbl.search().select(["embedding", "embedding_pending"]).limit(
        int(records_tbl.count_rows())
    ).to_pandas()
    df = df[df["embedding_pending"].fillna(0).astype(int) == 0]
    if df.empty:
        return 0.0
    embs = np.array([list(e) for e in df["embedding"].tolist()], dtype=np.float32)
    cos = np.matmul(embs, cue)
    sorted_cos = np.sort(cos)[::-1]
    return float(sorted_cos[min(199, len(sorted_cos) - 1)])

def build_gate_b_reference_store(
    store_root: Path,
    n_records: int,
    cue_vec_generic: list[float],
    cue_vec_specific: list[float],
) -> "tuple[MemoryStore, dict[str, UUID]]":
    """Build the exact reference store this module's fixture-producing test
    seeds, at the given record count. Any consumer of the committed
    recall_quality_baseline.json fixture must build from THIS function --
    a differently-shaped reference store (different gold-record graph
    degree, different filler count) changes graph-based ranking and makes
    a fresh-vs-committed diff compare two different systems, not the same
    system at two points in time. Returns the store and the stable gold
    UUIDs by role."""
    store_root.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_root))

    rng = np.random.default_rng(RNG_SEED)
    for i in range(n_records):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v = (v / np.linalg.norm(v)).tolist()
        rec = _make(text=f"User record {i} filler content for gate-b baseline", vec=v)
        store.insert(rec)

    cue_spec_arr = np.asarray(cue_vec_specific, dtype=np.float32)
    cue_spec_arr /= np.linalg.norm(cue_spec_arr)
    cue_gen_arr = np.asarray(cue_vec_generic, dtype=np.float32)
    cue_gen_arr /= np.linalg.norm(cue_gen_arr)

    hub_gold_id = UUID(int=1)
    store.insert(_make_gold_record(1, list(cue_gen_arr)))

    hub_node_id = UUID(int=2)
    rng4 = np.random.default_rng(44444)
    hub_node_vec = rng4.random(EMBED_DIM).astype(np.float32)
    hub_node_vec /= np.linalg.norm(hub_node_vec)
    store.insert(_make_gold_record(2, hub_node_vec.tolist()))
    store.boost_edges([(hub_node_id, hub_gold_id)], edge_type="hebbian", delta=[3.0])
    for extra_i in range(12):
        store.boost_edges([(hub_node_id, UUID(int=1000 + extra_i))], edge_type="hebbian", delta=[1.0])

    seed_id = UUID(int=3)
    store.insert(_make_gold_record(3, list(cue_spec_arr)))

    intermediate_id = UUID(int=4)
    inter_component = 0.4 * cue_spec_arr
    rng5 = np.random.default_rng(55555)
    inter_noise = rng5.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_spec_arr) * cue_spec_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_full = inter_component + 0.9165 * inter_noise
    inter_full /= np.linalg.norm(inter_full)
    store.insert(_make_gold_record(4, inter_full.tolist()))
    for extra_j in range(10):
        store.boost_edges([(intermediate_id, UUID(int=2000 + extra_j))], edge_type="hebbian", delta=[1.0])

    two_hop_gold_id = UUID(int=5)
    rng6 = np.random.default_rng(66666)
    noise = rng6.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_spec_arr) * cue_spec_arr
    noise /= np.linalg.norm(noise)
    target_cosine = 0.02
    orth_magnitude = float(np.sqrt(max(0.0, 1.0 - target_cosine**2)))
    two_hop_vec = target_cosine * cue_spec_arr + orth_magnitude * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))

    store.boost_edges([(seed_id, intermediate_id)], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(intermediate_id, two_hop_gold_id)], edge_type="hebbian", delta=[5.0])
    rng7 = np.random.default_rng(88888)
    for extra_k in range(8):
        target_id = UUID(int=3000 + extra_k)
        target_vec = rng7.random(EMBED_DIM).astype(np.float32)
        target_vec = (target_vec / np.linalg.norm(target_vec)).tolist()
        store.insert(_make_gold_record(3000 + extra_k, target_vec))
        store.boost_edges([(two_hop_gold_id, target_id)], edge_type="hebbian", delta=[2.0])

    contradict_a_id = UUID(int=6)
    contradict_b_id = UUID(int=7)
    rng3 = np.random.default_rng(77777)
    ca_vec = rng3.random(EMBED_DIM).astype(np.float32)
    ca_vec = (ca_vec / np.linalg.norm(ca_vec)).tolist()
    cb_vec = rng3.random(EMBED_DIM).astype(np.float32)
    cb_vec = (cb_vec / np.linalg.norm(cb_vec)).tolist()
    store.insert(_make_gold_record(6, ca_vec))
    store.insert(_make_gold_record(7, cb_vec))
    store.boost_edges([(contradict_a_id, contradict_b_id)], edge_type="contradicts", delta=[1.0])

    gold_ids = {
        "hub_gold": hub_gold_id,
        "hub_node": hub_node_id,
        "seed": seed_id,
        "intermediate": intermediate_id,
        "two_hop_gold": two_hop_gold_id,
        "contradict_a": contradict_a_id,
        "contradict_b": contradict_b_id,
    }
    return store, gold_ids

@pytest.mark.slow
def test_ef_k_linchpin_and_gate_b_fixture(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)
    _reset_auto_depth()

    embedder = Embedder()
    cue_vec_generic = embedder.embed(LEXICAL_GENERIC_CUE)
    cue_vec_specific = embedder.embed(LEXICAL_SPECIFIC_CUE)

    fixture_data: dict = {"rng_seed": RNG_SEED, "stable_key_scheme": "UUID(int=i)"}

    for n_records, n_label in [(N_SMALL, "n1k"), (N_LARGE, "n10k")]:
        print(f"\n{'='*60}")
        print(f"  Linchpin + Gate-B: {n_label}")
        print(f"{'='*60}")

        store_root = tmp_path / f"linchpin-{n_label}"
        store, gold_ids = build_gate_b_reference_store(
            store_root, n_records, cue_vec_generic, cue_vec_specific,
        )
        hub_gold_id = gold_ids["hub_gold"]
        hub_node_id = gold_ids["hub_node"]
        seed_id = gold_ids["seed"]
        intermediate_id = gold_ids["intermediate"]
        two_hop_gold_id = gold_ids["two_hop_gold"]
        contradict_a_id = gold_ids["contradict_a"]
        contradict_b_id = gold_ids["contradict_b"]

        cue_spec_arr = np.asarray(cue_vec_specific, dtype=np.float32)
        cue_spec_arr /= np.linalg.norm(cue_spec_arr)

        _reset_auto_depth()
        from iai_mcp.retrieve import build_runtime_graph
        graph, assignment, rich_club = build_runtime_graph(store)

        hub_in_rich_club = hub_node_id in rich_club or hub_gold_id in rich_club
        print(f"  hub_in_rich_club: {hub_in_rich_club} (rich_club size={len(rich_club)})")

        ann_boundary = _estimate_ann_top200_cosine_threshold(store, cue_vec_specific)
        two_hop_vec_arr = np.asarray(store.get(two_hop_gold_id).embedding, dtype=np.float32)
        two_hop_vec_arr /= np.linalg.norm(two_hop_vec_arr)
        gold_cosine_vs_cue = float(np.dot(two_hop_vec_arr, cue_spec_arr))
        two_hop_outside_ann_top200 = gold_cosine_vs_cue < ann_boundary
        print(f"  two-hop gold cosine vs specific cue: {gold_cosine_vs_cue:.4f}")
        print(f"  ANN top-200 boundary (specific cue): {ann_boundary:.4f}")
        print(f"  two-hop gold outside ANN top-200: {two_hop_outside_ann_top200}")

        spread_from_seed = graph.two_hop_neighborhood([seed_id], top_k=5)
        two_hop_reachable = two_hop_gold_id in spread_from_seed
        print(f"  two-hop gold reachable from seed via 2-hop: {two_hop_reachable}")

        ef50_behaviour = "unknown"
        ef50_result_count = 0
        try:
            results_ef50 = store.query_similar(cue_vec_generic, k=200)
            ef50_result_count = len(results_ef50)
            if ef50_result_count == 200:
                ef50_behaviour = "returned-200"
            elif ef50_result_count < 200:
                ef50_behaviour = f"returned-{ef50_result_count}-lt-200"
            else:
                ef50_behaviour = f"returned-{ef50_result_count}"
        except Exception as exc:
            ef50_behaviour = f"raised: {type(exc).__name__}: {exc}"
        print(f"  ef=50 k=200 behaviour: {ef50_behaviour}")

        rss_before_ef_raise = _rss_mb()
        store.db._hnsw.set_ef(200)
        rss_after_ef_raise = _rss_mb()
        memory_delta_ef_raise_mb = rss_after_ef_raise - rss_before_ef_raise

        recall_at_200_results = {}
        for cue_text, cue_vec, cue_label in [
            (LEXICAL_GENERIC_CUE, cue_vec_generic, "lexical-generic"),
            (LEXICAL_SPECIFIC_CUE, cue_vec_specific, "lexical-specific"),
        ]:
            ann_ids = _ann_top_k(store, cue_vec, k=200)
            exact_ids = _exact_top_k(store, cue_vec, k=200)
            overlap = _recall_at_k(ann_ids, exact_ids, k=min(200, len(exact_ids), len(ann_ids)))
            recall_at_200_results[cue_label] = round(overlap, 4)
            print(f"  recall@200 {cue_label} {n_label}: {overlap:.4f}")

        store.db._hnsw.set_ef(50)
        k3_ef50_samples = []
        k3_ef50_top_set = None
        for trial in range(N_TRIALS):
            t0 = time.perf_counter()
            r3 = store.query_similar(cue_vec_generic, k=3)
            k3_ef50_samples.append((time.perf_counter() - t0) * 1000.0)
            if trial == 0:
                k3_ef50_top_set = frozenset(str(rec.id) for rec, _ in r3)

        store.db._hnsw.set_ef(200)
        k3_ef200_samples = []
        k3_ef200_top_set = None
        for trial in range(N_TRIALS):
            t0 = time.perf_counter()
            r3 = store.query_similar(cue_vec_generic, k=3)
            k3_ef200_samples.append((time.perf_counter() - t0) * 1000.0)
            if trial == 0:
                k3_ef200_top_set = frozenset(str(rec.id) for rec, _ in r3)

        k3_p50_ef50, k3_p95_ef50 = _p50_p95(k3_ef50_samples)
        k3_p50_ef200, k3_p95_ef200 = _p50_p95(k3_ef200_samples)
        k3_latency_delta_p50 = k3_p50_ef200 - k3_p50_ef50
        k3_latency_delta_p95 = k3_p95_ef200 - k3_p95_ef50
        k3_top_set_changed = k3_ef50_top_set != k3_ef200_top_set
        k3_change_direction = "toward-exact-or-unchanged"

        print(f"  k=3 ef=50 p50={k3_p50_ef50:.1f}ms p95={k3_p95_ef50:.1f}ms")
        print(f"  k=3 ef=200 p50={k3_p50_ef200:.1f}ms p95={k3_p95_ef200:.1f}ms")
        print(f"  k=3 latency delta (p50): {k3_latency_delta_p50:+.1f}ms")
        print(f"  k=3 top-set changed at ef=200: {k3_top_set_changed} ({k3_change_direction})")
        print(f"  memory delta (RSS) ef 50->200: {memory_delta_ef_raise_mb:+.1f}MB")

        store.db._hnsw.set_ef(50)
        _reset_auto_depth()

        from iai_mcp.pipeline import recall_for_response

        graph, assignment, rich_club = build_runtime_graph(store)
        _reset_auto_depth()

        reference_cues = [
            {
                "cue": LEXICAL_GENERIC_CUE,
                "cue_label": "lexical-generic",
                "must_hit": True,
                "two_hop_only": False,
                "hub_sensitive": True,
                "expected_stable_keys": [str(hub_gold_id)],
            },
            {
                "cue": LEXICAL_SPECIFIC_CUE,
                "cue_label": "lexical-specific",
                "must_hit": True,
                "two_hop_only": True,
                "hub_sensitive": False,
                "expected_stable_keys": [str(two_hop_gold_id)],
            },
        ]

        cue_records = []
        for ref in reference_cues:
            cue_text = ref["cue"]
            _reset_auto_depth()
            response = recall_for_response(
                store=store,
                graph=graph,
                assignment=assignment,
                rich_club=rich_club,
                embedder=embedder,
                cue=cue_text,
                session_id="user",
                budget_tokens=1500,
                mode="concept",
            )
            returned_hit_ids = [str(h.record_id) for h in response.hits]
            expected = ref["expected_stable_keys"]
            r5 = sum(1 for ek in expected if ek in returned_hit_ids[:5]) / max(len(expected), 1)
            r10 = sum(1 for ek in expected if ek in returned_hit_ids[:10]) / max(len(expected), 1)
            anti_hit_ids = [str(h.record_id) for h in response.anti_hits]
            anti_hit_surfaced = (
                str(contradict_b_id) in anti_hit_ids or str(contradict_a_id) in anti_hit_ids
            )

            print(
                f"  Cue '{cue_text}': recall@5={r5:.2f} recall@10={r10:.2f} "
                f"anti_hit_surfaced={anti_hit_surfaced}"
            )
            print(f"    top-10 returned: {returned_hit_ids[:10]}")

            cue_records.append({
                "cue": cue_text,
                "cue_label": ref["cue_label"],
                "must_hit": ref["must_hit"],
                "two_hop_only": ref["two_hop_only"],
                "hub_sensitive": ref["hub_sensitive"],
                "expected_stable_keys": expected,
                "recall_at_5": round(r5, 4),
                "recall_at_10": round(r10, 4),
                "returned_hit_ids_top10": returned_hit_ids[:10],
                "anti_hit_surfaced": anti_hit_surfaced,
            })

        fixture_data[n_label] = {
            "n_records": n_records,
            "reference_cues": cue_records,
            "contradicts_pair_stable_keys": [str(contradict_a_id), str(contradict_b_id)],
            "hub_sensitive_cue_label": "lexical-generic",
            "two_hop_only_cue_label": "lexical-specific",
            "ef_50_behaviour": ef50_behaviour,
            "ef_50_result_count": ef50_result_count,
            "recall_at_200": recall_at_200_results,
            "recall_at_200_note": (
                "Random uniform unit-vector filler in 384-d produces near-tied cosines "
                "(std ~0.051). ANN at ef=200 struggles to rank near-tied neighbors consistently. "
                "Production clustered embeddings yield higher recall@200. "
                "The threshold for 200th-nearest-neighbor cosine at N=10k is ~0.10."
            ),
            "ann_boundary_specific_cue": round(ann_boundary, 4),
            "two_hop_gold_cosine_vs_cue": round(gold_cosine_vs_cue, 4),
            "two_hop_gold_outside_ann_top200": two_hop_outside_ann_top200,
            "two_hop_gold_reachable_via_2hop": two_hop_reachable,
            "small_k_ef_blast_radius": {
                "k": 3,
                "ef_from": 50,
                "ef_to": 200,
                "p50_ms_ef50": round(k3_p50_ef50, 2),
                "p95_ms_ef50": round(k3_p95_ef50, 2),
                "p50_ms_ef200": round(k3_p50_ef200, 2),
                "p95_ms_ef200": round(k3_p95_ef200, 2),
                "latency_delta_p50_ms": round(k3_latency_delta_p50, 2),
                "latency_delta_p95_ms": round(k3_latency_delta_p95, 2),
                "top_set_changed": k3_top_set_changed,
                "change_direction": k3_change_direction,
                "memory_delta_rss_mb": round(memory_delta_ef_raise_mb, 2),
            },
        }

    for n_label in ("n1k", "n10k"):
        entry = fixture_data[n_label]
        assert "reference_cues" in entry, f"Missing reference_cues in {n_label}"
        assert "contradicts_pair_stable_keys" in entry
        assert "recall_at_200" in entry
        assert "small_k_ef_blast_radius" in entry
        two_hop_cues = [c for c in entry["reference_cues"] if c.get("two_hop_only")]
        hub_cues = [c for c in entry["reference_cues"] if c.get("hub_sensitive")]
        assert len(two_hop_cues) >= 1, f"No two_hop_only cue in {n_label}"
        assert len(hub_cues) >= 1, f"No hub_sensitive cue in {n_label}"
        assert entry["ef_50_behaviour"] != "unknown"
        for cue_label in ("lexical-generic", "lexical-specific"):
            assert cue_label in entry["recall_at_200"], (
                f"Missing recall@200 for {cue_label} in {n_label}"
            )
        assert entry["two_hop_gold_reachable_via_2hop"], (
            f"Two-hop gold not reachable via 2-hop at {n_label} -- seeding failed"
        )

    # Load the OLD on-disk fixture BEFORE any overwrite: a diff against
    # freshly-written data would always compare fresh-vs-itself and could
    # never fail on a real regression.
    old_fixture: dict | None = None
    if FIXTURE_PATH.exists():
        with open(FIXTURE_PATH) as f:
            old_fixture = json.load(f)

    update_baseline = os.environ.get(_UPDATE_BASELINE_ENV) == "1"

    if old_fixture is not None and not update_baseline:
        regression_violations: list[str] = []
        for n_label in ("n1k", "n10k"):
            if n_label in old_fixture and n_label in fixture_data:
                regression_violations.extend(
                    diff_recall_quality_baseline_entry(
                        fixture_data[n_label], old_fixture[n_label]
                    )
                )
        assert not regression_violations, (
            "recall-quality regression vs the committed baseline "
            f"(set {_UPDATE_BASELINE_ENV}=1 to intentionally reset it):\n"
            + "\n".join(regression_violations)
        )

    if update_baseline or old_fixture is None:
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FIXTURE_PATH, "w", encoding="utf-8") as f:
            json.dump(fixture_data, f, indent=2)
        print(f"\n  Gate-B fixture written: {FIXTURE_PATH}")
    else:
        print(
            "\n  Gate-B fixture NOT overwritten (no regression, no update "
            f"requested): {FIXTURE_PATH}"
        )

    print("\n  Gate-B fixture validation: PASSED")
