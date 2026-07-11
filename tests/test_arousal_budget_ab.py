from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _inline_production_route(cue: str) -> tuple[str, str]:
    digest = hashlib.md5(str(cue).encode("utf-8")).digest()
    cue_hash_hex = digest[:4].hex()
    if os.environ.get("IAI_MCP_AROUSAL_USE_SHADOW") == "1":
        return ("arousal_shadow", cue_hash_hex)
    route = "arousal_real" if (digest[0] & 1) else "arousal_shadow"
    return (route, cue_hash_hex)


def test_route_determinism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    from bench import contradiction_longitudinal_claude as bench

    helper = bench._bench_arousal_route_for_cue  # noqa: SLF001

    cues = ["one", "two", "three", "русский cue", "x" * 100]
    for cue in cues:
        r1, h1 = helper(cue)
        r2, h2 = helper(cue)
        assert r1 == r2, f"determinism broken for cue={cue!r}"
        assert h1 == h2


def test_route_split_balance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    from bench import contradiction_longitudinal_claude as bench

    helper = bench._bench_arousal_route_for_cue  # noqa: SLF001

    n = 10000
    real = 0
    shadow = 0
    for i in range(n):
        route, _ = helper(f"cue-{i}")
        if route == "arousal_real":
            real += 1
        elif route == "arousal_shadow":
            shadow += 1
        else:  # pragma: no cover — sanity guard
            pytest.fail(f"unexpected route {route!r}")
    expected_per_arm = n / 2
    tolerance = n * 0.02
    assert abs(real - expected_per_arm) < tolerance, f"real={real} skewed"
    assert abs(shadow - expected_per_arm) < tolerance, f"shadow={shadow} skewed"


def test_env_override_shadow_forces_all_to_shadow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IAI_MCP_AROUSAL_USE_SHADOW", "1")
    from bench import contradiction_longitudinal_claude as bench

    helper = bench._bench_arousal_route_for_cue  # noqa: SLF001

    for cue in [f"cue-{i}" for i in range(100)]:
        route, cue_hash = helper(cue)
        assert route == "arousal_shadow", (
            f"shadow override broken for cue={cue!r}: got {route}"
        )
        expected_hash = hashlib.md5(cue.encode("utf-8")).digest()[:4].hex()
        assert cue_hash == expected_hash


@pytest.mark.parametrize(
    "cue",
    [
        "alpha", "beta", "gamma", "delta",
        "Quote the original launch announcement.",
        "русский cue",
        "ceo_name probe",
        "x",
    ],
)
def test_bench_production_route_parity(
    cue: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    from bench import contradiction_longitudinal_claude as bench

    bench_route, bench_hash = bench._bench_arousal_route_for_cue(cue)  # noqa: SLF001
    prod_route, prod_hash = _inline_production_route(cue)

    assert bench_route == prod_route, (
        f"route drift for cue={cue!r}: bench={bench_route} prod={prod_route}"
    )
    assert bench_hash == prod_hash, (
        f"hash drift for cue={cue!r}: bench={bench_hash} prod={prod_hash}"
    )


class _FakeEmbedder:

    def __init__(self, vec: list[float] | None = None) -> None:
        from iai_mcp.types import EMBED_DIM

        self.DIM = EMBED_DIM
        if vec is None:
            vec = [1.0] + [0.0] * (EMBED_DIM - 1)
        self._vec = vec

    def embed(self, text: str) -> list[float]:
        return list(self._vec)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vec) for _ in texts]


def _make_record(vec: list[float], text: str = "rec", tier: str = "episodic"):
    from iai_mcp.types import MemoryRecord

    now = datetime.now(timezone.utc)
    return MemoryRecord(
        id=uuid4(),
        tier=tier,
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


def _build_store_and_graph(tmp_path, n: int):
    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import EMBED_DIM

    store = MemoryStore(path=tmp_path / "lancedb")
    recs = []
    for i in range(n):
        vec = [0.0] * EMBED_DIM
        vec[i % EMBED_DIM] = 1.0
        rec = _make_record(vec, text=f"rec{i}")
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    return store, graph, recs


def _flat_assignment(recs):
    from iai_mcp.community import CommunityAssignment
    from iai_mcp.types import EMBED_DIM

    cid = uuid4()
    centroid = [1.0] + [0.0] * (EMBED_DIM - 1)
    return CommunityAssignment(
        node_to_community={r.id: cid for r in recs},
        community_centroids={cid: centroid},
        modularity=0.0,
        backend="flat",
        top_communities=[cid],
        mid_regions={cid: [r.id for r in recs]},
    )


def test_telemetry_event_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)

    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.pipeline import _recall_core

    store, graph, recs = _build_store_and_graph(tmp_path, n=10)
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="probe-cue-alpha",
        session_id="test-session",
    )

    flush_event_buffer(store)

    rows = query_events(store, kind="retrieval_arousal_ab", limit=10)
    assert rows, "no retrieval_arousal_ab event written"

    ev = rows[0]
    assert ev["kind"] == "retrieval_arousal_ab"
    data = ev.get("data") or {}

    required_keys = {
        "cue_hash", "route", "n_hits", "budget_tokens_used",
        "max_hops_used", "rank_threshold_used", "arousal_level",
        "arousal_mode", "top_hit_id",
    }
    missing = required_keys - data.keys()
    assert not missing, f"telemetry missing keys: {missing}"

    assert data["route"] in {"arousal_real", "arousal_shadow", "arousal_skip"}
    assert isinstance(data["cue_hash"], str) and len(data["cue_hash"]) == 8


def test_rank_threshold_filters_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from iai_mcp.events import flush_event_buffer, query_events
    from iai_mcp.pipeline import _recall_core
    from iai_mcp.types import EMBED_DIM

    from iai_mcp.graph import MemoryGraph
    from iai_mcp.store import MemoryStore

    store = MemoryStore(path=tmp_path / "lancedb")
    recs = []
    good_count = 5
    bad_count = 55
    for i in range(good_count):
        vec = [0.0] * EMBED_DIM
        vec[0] = 1.0
        rec = _make_record(vec, text=f"good{i}")
        store.insert(rec)
        recs.append(rec)
    for i in range(bad_count):
        vec = [0.0] * EMBED_DIM
        vec[1] = 1.0
        rec = _make_record(vec, text=f"bad{i}")
        store.insert(rec)
        recs.append(rec)
    graph = MemoryGraph()
    for rec in recs:
        graph.add_node(rec.id, community_id=None, embedding=list(rec.embedding))
        graph.set_node_payload(rec.id, {
            "embedding": list(rec.embedding),
            "surface": rec.literal_surface,
            "centrality": 0.0,
            "tier": rec.tier,
            "tags": [],
            "language": "en",
        })
    assignment = _flat_assignment(recs)
    embedder = _FakeEmbedder()

    monkeypatch.setenv("IAI_MCP_AROUSAL_USE_SHADOW", "1")
    _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue="rank-threshold-probe",
        session_id="test-shadow",
    )
    flush_event_buffer(store)
    shadow_events = query_events(store, kind="retrieval_arousal_ab", limit=5)
    assert shadow_events, "shadow probe wrote no event"
    shadow_data = shadow_events[0]["data"]
    assert shadow_data["route"] == "arousal_shadow", (
        f"force-shadow env not honored: route={shadow_data['route']}"
    )
    assert shadow_data["rank_threshold_used"] == 0.0, (
        f"shadow rank_threshold_used should be 0.0: got "
        f"{shadow_data['rank_threshold_used']}"
    )

    monkeypatch.delenv("IAI_MCP_AROUSAL_USE_SHADOW", raising=False)
    real_cue = None
    for i in range(100):
        cand = f"real-cue-{i}"
        digest = hashlib.md5(cand.encode()).digest()
        if digest[0] & 1:
            real_cue = cand
            break
    assert real_cue is not None, "could not find a cue with arousal_real route"

    _recall_core(
        store=store,
        graph=graph,
        assignment=assignment,
        rich_club=[],
        embedder=embedder,
        cue=real_cue,
        session_id="test-real",
    )
    flush_event_buffer(store)
    real_events = query_events(store, kind="retrieval_arousal_ab", limit=5)
    real_data = real_events[0]["data"]
    assert real_data["route"] == "arousal_real", (
        f"unexpected route on real cue: {real_data['route']}"
    )
    assert real_data["rank_threshold_used"] > 0.0, (
        f"real route rank_threshold_used should be > 0: got "
        f"{real_data['rank_threshold_used']}"
    )


def _build_synthetic_csv(
    tmp_path: Path,
    real_hits_per_seed: int,
    real_misses_per_seed: int,
    shadow_hits_per_seed: int,
    shadow_misses_per_seed: int,
    seeds: list[int],
) -> Path:
    csv_path = tmp_path / "contradiction_longitudinal_synth.csv"
    header = [
        "probe_id", "seed", "n_slice", "condition", "topic",
        "pipeline_rank", "cosine_rank",
        "pipeline_hit_at_k", "cosine_hit_at_k",
        "s4_contradiction_emitted", "anti_hits_count", "hint_kinds",
        "pipeline_top1_text",
        "route", "cue_hash",
        "arousal_route", "arousal_cue_hash",
    ]
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        i = 0
        for seed in seeds:
            for _ in range(real_hits_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "1",
                    "pipeline_hit_at_k": "1",
                    "arousal_route": "arousal_real",
                    "arousal_cue_hash": "aaaa1111",
                })
                i += 1
            for _ in range(real_misses_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "-1",
                    "pipeline_hit_at_k": "0",
                    "arousal_route": "arousal_real",
                    "arousal_cue_hash": "aaaa2222",
                })
                i += 1
            for _ in range(shadow_hits_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "1",
                    "pipeline_hit_at_k": "1",
                    "arousal_route": "arousal_shadow",
                    "arousal_cue_hash": "bbbb1111",
                })
                i += 1
            for _ in range(shadow_misses_per_seed):
                w.writerow({
                    **{k: "" for k in header},
                    "probe_id": f"p{i}",
                    "seed": str(seed),
                    "pipeline_rank": "-1",
                    "pipeline_hit_at_k": "0",
                    "arousal_route": "arousal_shadow",
                    "arousal_cue_hash": "bbbb2222",
                })
                i += 1
    return csv_path


def test_analyzer_math_synthetic_csv(tmp_path: Path) -> None:
    csv_path = _build_synthetic_csv(
        tmp_path,
        real_hits_per_seed=18, real_misses_per_seed=2,
        shadow_hits_per_seed=5, shadow_misses_per_seed=15,
        seeds=[13, 42, 137],
    )
    analyzer = REPO_ROOT / "bench" / "analyze_arousal_ab.py"
    assert analyzer.exists(), f"analyzer not built yet: {analyzer}"

    result = subprocess.run(
        [sys.executable, str(analyzer), str(csv_path.parent)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"analyzer exited {result.returncode}: stderr={result.stderr}"
    )

    summary_path = csv_path.parent / "AROUSAL-AB-SUMMARY.json"
    assert summary_path.exists(), f"summary not written: {summary_path}"
    summary = json.loads(summary_path.read_text())
    assert summary["cross_seed_mean_delta"] > 0.5, (
        f"expected large positive delta; got {summary['cross_seed_mean_delta']}"
    )
    assert summary["ship_gate_hit"] is True
    assert summary.get("verdict") == "keep"
