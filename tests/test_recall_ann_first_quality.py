from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_store import _make

from iai_mcp.embed import Embedder
from iai_mcp.store import MemoryStore
from iai_mcp.types import EMBED_DIM

RNG_SEED = 20260601
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recall_quality_baseline.json"

RICH_CLUB_CAP = 50

LEXICAL_GENERIC_CUE = "hello"
LEXICAL_SPECIFIC_CUE = "specialized technical framework review"

def _random_vec(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.random(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()

def _monkeypatch_env(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "home"
    fake_home.mkdir(exist_ok=True)
    monkeypatch.setenv("IAI_MCP_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("IAI_DAEMON_SOCKET_PATH", str(tmp_path / "daemon.sock"))
    monkeypatch.setenv("IAI_MCP_RECALL_SAMPLE_RATE", "1.0")

def _make_gold_record(i: int, vec: list[float]) -> object:
    from iai_mcp.types import MemoryRecord
    return MemoryRecord(
        id=UUID(int=i),
        tier="episodic",
        literal_surface=f"User reference gold doc {i}",
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

def _build_reference_store(store_path: Path, n_records: int) -> MemoryStore:
    store = MemoryStore(str(store_path))

    rng = np.random.default_rng(RNG_SEED)
    for i in range(n_records):
        v = rng.random(EMBED_DIM).astype(np.float32)
        v = (v / np.linalg.norm(v)).tolist()
        rec = _make(text=f"User record {i} filler content for gate-b baseline", vec=v)
        store.insert(rec)

    embedder = Embedder()
    cue_gen_arr = np.asarray(embedder.embed(LEXICAL_GENERIC_CUE), dtype=np.float32)
    cue_gen_arr /= np.linalg.norm(cue_gen_arr)
    cue_spec_arr = np.asarray(embedder.embed(LEXICAL_SPECIFIC_CUE), dtype=np.float32)
    cue_spec_arr /= np.linalg.norm(cue_spec_arr)

    store.insert(_make_gold_record(1, list(cue_gen_arr)))

    rng4 = np.random.default_rng(44444)
    hub_vec = rng4.random(EMBED_DIM).astype(np.float32)
    hub_vec /= np.linalg.norm(hub_vec)
    store.insert(_make_gold_record(2, hub_vec.tolist()))
    store.boost_edges([(UUID(int=2), UUID(int=1))], edge_type="hebbian", delta=[3.0])
    for extra_i in range(12):
        store.boost_edges([(UUID(int=2), UUID(int=1000 + extra_i))], edge_type="hebbian", delta=[1.0])

    store.insert(_make_gold_record(3, list(cue_spec_arr)))

    rng5 = np.random.default_rng(55555)
    inter_noise = rng5.random(EMBED_DIM).astype(np.float32)
    inter_noise -= np.dot(inter_noise, cue_spec_arr) * cue_spec_arr
    inter_noise /= np.linalg.norm(inter_noise)
    inter_vec = 0.4 * cue_spec_arr + 0.9165 * inter_noise
    inter_vec /= np.linalg.norm(inter_vec)
    store.insert(_make_gold_record(4, inter_vec.tolist()))
    for extra_j in range(10):
        store.boost_edges([(UUID(int=4), UUID(int=2000 + extra_j))], edge_type="hebbian", delta=[1.0])

    rng6 = np.random.default_rng(66666)
    noise = rng6.random(EMBED_DIM).astype(np.float32)
    noise -= np.dot(noise, cue_spec_arr) * cue_spec_arr
    noise /= np.linalg.norm(noise)
    target_cosine = 0.02
    orth_mag = float(np.sqrt(max(0.0, 1.0 - target_cosine**2)))
    two_hop_vec = target_cosine * cue_spec_arr + orth_mag * noise
    two_hop_vec /= np.linalg.norm(two_hop_vec)
    store.insert(_make_gold_record(5, two_hop_vec.tolist()))
    store.boost_edges([(UUID(int=3), UUID(int=4))], edge_type="hebbian", delta=[5.0])
    store.boost_edges([(UUID(int=4), UUID(int=5))], edge_type="hebbian", delta=[5.0])
    rng7 = np.random.default_rng(77001)
    for _boost_i in range(50):
        _raw = rng7.random(EMBED_DIM).astype(np.float32)
        _raw -= np.dot(_raw, cue_spec_arr) * cue_spec_arr
        _raw /= np.linalg.norm(_raw)
        _brec = _make_gold_record(100 + _boost_i, _raw.tolist())
        _brec.never_merge = True
        store.insert(_brec)
        store.boost_edges(
            [(UUID(int=5), UUID(int=100 + _boost_i))], edge_type="hebbian", delta=[2.0]
        )

    rng3 = np.random.default_rng(77777)
    ca_vec = rng3.random(EMBED_DIM).astype(np.float32)
    ca_vec = (ca_vec / np.linalg.norm(ca_vec)).tolist()
    cb_vec = rng3.random(EMBED_DIM).astype(np.float32)
    cb_vec = (cb_vec / np.linalg.norm(cb_vec)).tolist()
    store.insert(_make_gold_record(6, ca_vec))
    store.insert(_make_gold_record(7, cb_vec))
    store.boost_edges([(UUID(int=6), UUID(int=7))], edge_type="contradicts", delta=[1.0])

    return store

def _prime_cache(store: MemoryStore) -> tuple:
    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc

    graph, assignment, rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, assignment, rc, max_degree=int(getattr(graph, "_max_degree", 0) or 0))
    return graph, assignment, rc

def _full_graph_recall(
    store: MemoryStore,
    graph,
    assignment,
    rc,
    cue: str,
    budget: int = 2000,
) -> set[str]:
    import iai_mcp.pipeline as _pm
    from iai_mcp.pipeline import recall_for_response
    from iai_mcp.embed import Embedder

    _pm._last_recall_latency_ms = 0.0
    embedder = Embedder()
    resp = recall_for_response(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=rc,
        embedder=embedder,
        cue=cue,
        session_id="gate-b-baseline",
        budget_tokens=budget,
        mode="concept",
        profile_state=None,
    )
    return {str(h.record_id) for h in resp.hits}

def _bounded_recall(store: MemoryStore, cue: str, budget: int = 2000) -> dict:
    import iai_mcp.pipeline as _pm
    from iai_mcp import core

    _pm._last_recall_latency_ms = 0.0
    params = {
        "cue": cue,
        "session_id": "gate-b-test",
        "budget_tokens": budget,
    }
    return core.dispatch(store, "memory_recall", params)

def _run_gate_b_parity(
    store: MemoryStore,
    graph,
    assignment,
    rc,
    n_label: str,
    monkeypatch,
) -> None:
    from iai_mcp import core as _core
    monkeypatch.setitem(_core._profile_state, "literal_preservation", "medium")

    embedder = Embedder()

    failures = []
    telemetry = []

    for cue, cue_label, hub_sensitive in [
        (LEXICAL_GENERIC_CUE, "lexical-generic", True),
        (LEXICAL_SPECIFIC_CUE, "lexical-specific", False),
    ]:
        full_hits = _full_graph_recall(store, graph, assignment, rc, cue)

        bounded_resp = _bounded_recall(store, cue)
        bounded_hits = {h["record_id"] for h in bounded_resp.get("hits", [])}

        if not bounded_resp.get("ann_path_used", False):
            failures.append(
                f"  [FAIL] {cue_label} {n_label}: ann_path_used=False"
            )

        gold_stable_keys = {
            str(UUID(int=i)) for i in range(1, 8)
        }
        full_gold = full_hits & gold_stable_keys
        bounded_gold = bounded_hits & gold_stable_keys

        dropped = full_gold - bounded_gold

        if dropped:
            failures.append(
                f"  [FAIL CC-C] {cue_label} {n_label}: "
                f"bounded dropped gold keys that full-graph kept: {dropped}. "
                f"full_gold={full_gold}, bounded_gold={bounded_gold}"
            )
        else:
            telemetry.append(
                f"  [PASS] {cue_label} {n_label}: "
                f"bounded parity. full_gold={full_gold} "
                f"bounded_gold={bounded_gold}"
            )

        hub_key = str(UUID(int=1))
        if hub_sensitive and hub_key not in bounded_hits:
            failures.append(
                f"  [FAIL HUB] {cue_label} {n_label}: "
                f"hub-sensitive gold {hub_key} not in bounded hits "
                f"(off-path cache prime should give non-empty rich-club)"
            )
        elif hub_sensitive:
            telemetry.append(f"  [PASS HUB] {cue_label} {n_label}: hub gold surfaced")

        u5 = str(UUID(int=5))
        u5_in_full = u5 in full_hits
        u5_in_bounded = u5 in bounded_hits
        if u5_in_full and not u5_in_bounded:
            failures.append(
                f"  [FAIL U5] {cue_label} {n_label}: "
                f"U5 in full={u5_in_full} but in bounded={u5_in_bounded} — "
                "bounded dropped a two-hop-only gold hit that full-graph kept"
            )
        else:
            telemetry.append(
                f"  [TELEMETRY U5] {cue_label} {n_label}: "
                f"U5 in full={u5_in_full}, in bounded={u5_in_bounded} "
                "(bounded recovering more than full-graph, via the exact-similarity "
                "authority, is an improvement, not a regression)"
            )

    print(f"\n  --- Gate B {n_label} Results ---")
    for msg in telemetry:
        print(msg)

    if failures:
        raise AssertionError(f"Gate B {n_label} failures:\n" + "\n".join(failures))

def test_gate_b_anti_hit_uncapped_contradicts(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    store_path = tmp_path / "anti-hit-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = MemoryStore(str(store_path))
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    for i in range(200):
        store.insert(_make(text=f"User filler anti-hit {i}", vec=_random_vec(9000 + i)))

    embedder = Embedder()
    cue_vec = embedder.embed(LEXICAL_GENERIC_CUE)
    cue_arr = np.asarray(cue_vec, dtype=np.float32)
    cue_arr /= np.linalg.norm(cue_arr)

    source_id = UUID(int=80_001)
    source_rec = _make_gold_record(80_001, list(cue_arr))
    store.insert(source_rec)

    target_id = UUID(int=80_002)
    rng_at = np.random.default_rng(11111)
    base_vec = rng_at.random(EMBED_DIM).astype(np.float32)
    base_vec -= np.dot(base_vec, cue_arr) * cue_arr
    base_vec /= np.linalg.norm(base_vec)
    target_rec = _make_gold_record(80_002, base_vec.tolist())
    store.insert(target_rec)

    store.boost_edges([(source_id, target_id)], edge_type="contradicts", delta=[1.0])

    import iai_mcp.retrieve as _retrieve
    import iai_mcp.runtime_graph_cache as _rgc
    _g, _a, _rc = _retrieve.build_runtime_graph(store)
    _rgc.save(store, _a, _rc)

    import iai_mcp.pipeline as _pm
    from iai_mcp import core
    _pm._last_recall_latency_ms = 0.0
    resp = core.dispatch(store, "memory_recall", {
        "cue": LEXICAL_GENERIC_CUE,
        "session_id": "test",
        "budget_tokens": 2000,
    })

    hit_ids = {h["record_id"] for h in resp.get("hits", [])}
    anti_hit_ids = {h["record_id"] for h in resp.get("anti_hits", [])}

    assert str(source_id) in hit_ids, (
        f"ANTI-HIT: source record {source_id} not in hits; hits={hit_ids}"
    )
    assert str(target_id) in anti_hit_ids, (
        f"ANTI-HIT: contradicts target {target_id} not in anti_hits; "
        f"anti_hits={anti_hit_ids}. The 62-04 UNCAPPED contradicts path "
        "(incident_edges top_k=None) must surface this record."
    )
    assert resp.get("ann_path_used", False), "ann_path_used must be True"

def test_gate_b_n1k(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    assert FIXTURE_PATH.exists(), f"Gate-B fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        baseline = json.load(f)
    n_records = baseline["n1k"]["n_records"]

    store_path = tmp_path / "gate-b-n1k-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = _build_reference_store(store_path, n_records)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    graph, assignment, rc = _prime_cache(store)

    _run_gate_b_parity(store, graph, assignment, rc, "N=1k", monkeypatch)

@pytest.mark.slow
def test_gate_b_n10k(tmp_path, monkeypatch):
    _monkeypatch_env(monkeypatch, tmp_path)

    assert FIXTURE_PATH.exists(), f"Gate-B fixture not found: {FIXTURE_PATH}"
    with open(FIXTURE_PATH) as f:
        baseline = json.load(f)
    n_records = baseline["n10k"]["n_records"]

    store_path = tmp_path / "gate-b-n10k-store"
    store_path.mkdir(parents=True, exist_ok=True)
    store = _build_reference_store(store_path, n_records)
    monkeypatch.setenv("IAI_MCP_STORE", str(store_path))

    graph, assignment, rc = _prime_cache(store)

    _run_gate_b_parity(store, graph, assignment, rc, "N=10k", monkeypatch)
