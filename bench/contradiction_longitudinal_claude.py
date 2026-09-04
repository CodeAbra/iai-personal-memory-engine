#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
_SRC_PATH = str(Path(__file__).resolve().parent.parent / "src")
_ROOT_PATH = str(Path(__file__).resolve().parent.parent)
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)
if _ROOT_PATH not in sys.path:
    sys.path.insert(0, _ROOT_PATH)

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4


PRODUCTION_STORE = Path.home() / ".iai-mcp"

BENCH_PASSPHRASE = "iai-mcp-bench-falsifiability-deterministic-2026"

DEFAULT_STORE_DIR = "/tmp/iai-mcp-bench-claude/store"
DEFAULT_OUTPUT_DIR = "bench/results"
DEFAULT_SEEDS = [13, 42, 137]
DEFAULT_K = 10
CANDIDATE_POOL_SIZE = 200
WARMUP_PASSES = 5
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_CI = 0.95
WILCOXON_MIN_N = 30
MAX_RANK_REGRESSION = 20


SCALE_PRESETS = {
    "smoke":      (4,       2,         2),
    "mvp":        (200,     50,        50),
    "honest":     (1000,    250,       250),
    "stress":     (5000,    1000,      1000),
}


SYNTHETIC_FLIPS: list[tuple[str, str, str, str, str]] = [
    ("launch_date",
     "The product launches on 2026-06-01.",
     "Correction: launch moved to 2026-09-01.",
     "2026-09-01", "2026-06-01"),
    ("ceo_name",
     "The new CEO is Sarah Williams.",
     "Update: the actual CEO is Marcus Chen.",
     "Marcus Chen", "Sarah Williams"),
    ("hq_city",
     "Headquarters in Austin, Texas.",
     "Headquarters relocated to Boulder, Colorado.",
     "Boulder, Colorado", "Austin, Texas"),
    ("price_usd",
     "Annual subscription is $499.",
     "Pricing updated: annual subscription is now $349.",
     "$349", "$499"),
    ("supplier_name",
     "We source components from Acme Industries.",
     "Switched supplier: components now from Northwind Manufacturing.",
     "Northwind Manufacturing", "Acme Industries"),
    ("api_version",
     "Public API is at version 2.3.",
     "API rolled forward to version 3.0.",
     "3.0", "2.3"),
    ("conference_city",
     "Annual conference will be in Berlin.",
     "Conference venue changed: it will be in Lisbon.",
     "Lisbon", "Berlin"),
    ("bug_fix_eta",
     "The fix ships in week 14.",
     "Fix ETA revised: week 18.",
     "week 18", "week 14"),
    ("dependency_lib",
     "We use OpenSSL for crypto.",
     "Migrated crypto layer from OpenSSL to BoringSSL.",
     "BoringSSL", "OpenSSL"),
    ("budget_ceiling",
     "Q3 budget ceiling is $50k.",
     "Budget ceiling revised down to $35k.",
     "$35k", "$50k"),
]

FILLER_SENTENCES: list[str] = [
    "The team meets every Tuesday for sync.",
    "Documentation is hosted internally on the wiki.",
    "Quarterly reviews happen at the end of each quarter.",
    "The build pipeline is configured in CI.",
    "Linting runs on every pull request.",
    "Database backups happen nightly.",
    "Monitoring dashboards live in the ops portal.",
    "Onboarding takes about two weeks.",
    "Release notes are published on the changelog page.",
    "Code reviews require at least one approval.",
    "The road map is reviewed monthly.",
    "Customer feedback comes in through the support tracker.",
    "Performance regressions are caught by the perf suite.",
    "Security audits run twice a year.",
    "Vendor renewals are tracked in the procurement spreadsheet.",
    "Incident response follows the documented runbook.",
    "The on-call rotation is one week per engineer.",
    "Demos happen at the all-hands meeting.",
    "The design system is the single source for components.",
    "Translations are handled by an external agency.",
]


@dataclass
class CorpusEntry:

    session_id: str
    role: str
    text: str
    is_flip_original: bool = False
    is_flip_correction: bool = False
    topic: str = ""


@dataclass
class Probe:

    probe_id: str
    cue: str
    expects: str
    condition: str
    topic: str
    flip_original_id: str = ""
    flip_correction_id: str = ""


@dataclass
class ProbeResult:

    probe_id: str
    seed: int
    n_slice: int
    condition: str
    topic: str
    pipeline_rank: int
    cosine_rank: int
    pipeline_hit_at_k: bool
    cosine_hit_at_k: bool
    pipeline_top1_text: str
    s4_contradiction_emitted: bool = False
    anti_hits_count: int = 0
    hint_kinds: str = ""
    route: str = ""
    cue_hash: str = ""
    arousal_route: str = ""
    arousal_cue_hash: str = ""


def _refuse_production_store(store_path: Path) -> None:
    resolved_store = store_path.expanduser().resolve()
    resolved_prod = PRODUCTION_STORE.expanduser().resolve()
    if resolved_store == resolved_prod or resolved_prod in resolved_store.parents:
        print(
            f"[setup-gate] REFUSE: --store-dir resolves to {resolved_store}, "
            f"which is inside production store {resolved_prod}.",
            file=sys.stderr,
        )
        sys.exit(2)


def _gather_env_metadata(store_dir: Path, seed_list: list[int]) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent

    def _git(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(repo_root)] + cmd, text=True
            ).strip()
        except Exception:
            return "unknown"

    sha = _git(["rev-parse", "--short", "HEAD"])
    dirty = _git(["status", "--porcelain"]) != ""

    def _pkg_version(pkg: str) -> str:
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            return "unknown"

    cpu_brand = "unknown"
    try:
        cpu_brand = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        pass

    ram_gb = "unknown"
    try:
        bytes_ = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip())
        ram_gb = f"{bytes_ / (1024**3):.1f}"
    except Exception:
        pass

    embedder_model = os.environ.get("IAI_MCP_EMBED_MODEL", "bge-small-en-v1.5")

    return {
        "cpu_brand": cpu_brand,
        "cpu_cores_physical": os.cpu_count() or "unknown",
        "ram_gb": ram_gb,
        "os": platform.system(),
        "os_version": platform.release(),
        "python_version": platform.python_version(),
        "iai_mcp_git_sha": sha,
        "iai_mcp_git_dirty": dirty,
        "lance_version": _pkg_version("lance"),
        "lancedb_version": _pkg_version("lancedb"),
        "pyarrow_version": _pkg_version("pyarrow"),
        "sentence_transformers_version": _pkg_version("sentence-transformers"),
        "embedder_model": embedder_model,
        "seed_list": seed_list,
        "iai_mcp_store": str(store_dir),
        "wall_clock_start_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_corpus(
    seed: int,
    n_sessions: int,
    n_probes_pre: int,
    n_probes_post: int,
) -> tuple[list[CorpusEntry], list[Probe]]:
    rng = random.Random(seed)

    n_flips_max = min(len(SYNTHETIC_FLIPS), n_sessions // 4)
    n_flips_max = max(n_flips_max, 1)
    flips = rng.sample(SYNTHETIC_FLIPS, n_flips_max)

    entries: list[CorpusEntry] = []
    flip_metadata: dict[str, dict[str, str]] = {}

    session_idx = 0
    for topic, original, correction, gold_after, gold_before in flips:
        remaining = n_sessions - session_idx
        if remaining < 2:
            break
        max_filler_before = max(0, min(rng.randint(0, 3), remaining - 2))
        for _ in range(max_filler_before):
            entries.append(CorpusEntry(
                session_id=f"s{session_idx:05d}-{seed}",
                role="user",
                text=rng.choice(FILLER_SENTENCES),
            ))
            session_idx += 1

        orig_session = f"s{session_idx:05d}-{seed}"
        entries.append(CorpusEntry(
            session_id=orig_session,
            role="user",
            text=original,
            is_flip_original=True,
            topic=topic,
        ))
        session_idx += 1

        remaining = n_sessions - session_idx
        max_filler_between = max(0, min(rng.randint(0, 2), remaining - 1))
        for _ in range(max_filler_between):
            entries.append(CorpusEntry(
                session_id=f"s{session_idx:05d}-{seed}",
                role="user",
                text=rng.choice(FILLER_SENTENCES),
            ))
            session_idx += 1

        corr_session = f"s{session_idx:05d}-{seed}"
        entries.append(CorpusEntry(
            session_id=corr_session,
            role="user",
            text=correction,
            is_flip_correction=True,
            topic=topic,
        ))
        session_idx += 1

        flip_metadata[topic] = {
            "orig_session": orig_session,
            "corr_session": corr_session,
            "gold_before": gold_before,
            "gold_after": gold_after,
            "original_text": original,
            "correction_text": correction,
        }

    while len(entries) < n_sessions:
        entries.append(CorpusEntry(
            session_id=f"s{len(entries):05d}-{seed}",
            role="user",
            text=rng.choice(FILLER_SENTENCES),
        ))

    probes: list[Probe] = []
    topics = list(flip_metadata.keys())
    if not topics:
        return entries, probes

    for i in range(n_probes_post):
        topic = rng.choice(topics)
        meta = flip_metadata[topic]
        probes.append(Probe(
            probe_id=f"post-{i:04d}-{seed}",
            cue=_post_flip_cue_for(topic),
            expects=meta["gold_after"],
            condition="post_flip",
            topic=topic,
        ))

    for i in range(n_probes_pre):
        topic = rng.choice(topics)
        meta = flip_metadata[topic]
        probes.append(Probe(
            probe_id=f"hist-{i:04d}-{seed}",
            cue=_historical_verbatim_cue_for(topic),
            expects=meta["gold_before"],
            condition="historical_verbatim",
            topic=topic,
        ))

    return entries, probes


def _post_flip_cue_for(topic: str) -> str:
    cues = {
        "launch_date": "When does the product launch?",
        "ceo_name": "Who is the current CEO?",
        "hq_city": "Where are the headquarters?",
        "price_usd": "What does the annual subscription cost?",
        "supplier_name": "Who supplies our components?",
        "api_version": "What is the current API version?",
        "conference_city": "Where will the annual conference be held?",
        "bug_fix_eta": "When does the fix ship?",
        "dependency_lib": "Which crypto library do we use?",
        "budget_ceiling": "What is the Q3 budget ceiling?",
    }
    return cues.get(topic, f"What is the current {topic}?")


def _historical_verbatim_cue_for(topic: str) -> str:
    cues = {
        "launch_date": "Quote the original announcement about the launch date.",
        "ceo_name": "Quote the first CEO announcement verbatim.",
        "hq_city": "Quote the original headquarters location announcement.",
        "price_usd": "Quote the original pricing line.",
        "supplier_name": "Quote the first supplier statement.",
        "api_version": "Quote the original API version statement.",
        "conference_city": "Quote the original conference venue announcement.",
        "bug_fix_eta": "Quote the original ETA wording.",
        "dependency_lib": "Quote the original crypto library statement.",
        "budget_ceiling": "Quote the original Q3 budget ceiling line.",
    }
    return cues.get(topic, f"Quote the original {topic} announcement.")


def _bench_efe_route_for_cue(cue: str) -> tuple[str, str]:
    digest = hashlib.md5(str(cue).encode("utf-8")).digest()
    cue_hash_hex = digest[:4].hex()
    if os.environ.get("IAI_MCP_EFE_USE_SHADOW") == "1":
        route = "efe_shadow"
    else:
        route = "efe_real" if (digest[0] & 1) else "efe_shadow"
    return route, cue_hash_hex


def _bench_arousal_route_for_cue(cue: str) -> tuple[str, str]:
    digest = hashlib.md5(str(cue).encode("utf-8")).digest()
    cue_hash_hex = digest[:4].hex()
    if os.environ.get("IAI_MCP_AROUSAL_USE_SHADOW") == "1":
        route = "arousal_shadow"
    else:
        route = "arousal_real" if (digest[0] & 1) else "arousal_shadow"
    return route, cue_hash_hex


def run_one_seed(
    seed: int,
    entries: list[CorpusEntry],
    probes: list[Probe],
    n_slice: int,
    store_dir_for_seed: Path,
    embedder_key: str,
    k_hits: int,
) -> tuple[list[ProbeResult], dict[str, Any]]:
    from iai_mcp.embed import Embedder
    from iai_mcp.lifecycle_event_log import LifecycleEventLog
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph, contradict
    from iai_mcp.lilli.cycle.sleep_pipeline import SleepPipeline
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    gate = {"insert_ok": False, "contradict_ok": False, "sleep_ok": False,
            "errors": []}

    store_dir_for_seed.mkdir(parents=True, exist_ok=True)
    os.environ["IAI_MCP_STORE"] = str(store_dir_for_seed)
    if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = BENCH_PASSPHRASE

    store = MemoryStore(path=store_dir_for_seed / "lancedb")
    embedder = Embedder(model_key=embedder_key)

    _ = embedder.embed_batch(["warm-up " + str(i) for i in range(WARMUP_PASSES)])

    text_to_id: dict[str, UUID] = {}
    topic_to_orig_id: dict[str, UUID] = {}
    topic_to_corr_text: dict[str, str] = {}
    try:
        all_texts = [e.text for e in entries]
        embeddings = embedder.embed_batch(all_texts)
        now = datetime.now(timezone.utc)
        for entry, emb in zip(entries, embeddings):
            rec_id = uuid4()
            rec = MemoryRecord(
                id=rec_id,
                tier="episodic",
                literal_surface=entry.text,
                aaak_index="",
                embedding=list(emb),
                community_id=None,
                centrality=0.0,
                detail_level=2,
                pinned=False,
                stability=0.0,
                difficulty=0.0,
                last_reviewed=None,
                never_decay=False,
                never_merge=False,
                provenance=[{"ts": now.isoformat(), "cue": entry.text[:60],
                             "session_id": entry.session_id}],
                created_at=now,
                updated_at=now,
                tags=["bench-claude", f"role:{entry.role}",
                      f"session:{entry.session_id}"],
                language="en",
            )
            store.insert(rec)
            text_to_id[entry.text] = rec_id
            if entry.is_flip_original:
                topic_to_orig_id[entry.topic] = rec_id
            if entry.is_flip_correction:
                topic_to_corr_text[entry.topic] = entry.text
        gate["insert_ok"] = True
    except Exception as exc:
        gate["errors"].append(f"insert: {exc!r}")
        return [], gate

    try:
        for topic, orig_id in topic_to_orig_id.items():
            corr_text = topic_to_corr_text.get(topic, "")
            if not corr_text:
                continue
            cue_emb = list(embedder.embed_batch([corr_text])[0])
            contradict(store, orig_id, corr_text, cue_emb)
        gate["contradict_ok"] = True
    except Exception as exc:
        gate["errors"].append(f"contradict: {exc!r}")
        return [], gate

    if n_slice > 0:
        try:
            iso_state_path = store_dir_for_seed / "lifecycle_state.json"
            iso_log_dir = store_dir_for_seed / "logs"
            iso_log_dir.mkdir(exist_ok=True)
            event_log = LifecycleEventLog(log_dir=iso_log_dir)
            pipeline = SleepPipeline(
                store=store,
                lifecycle_state_path=iso_state_path,
                event_log=event_log,
                quarantine_ttl_hours=0.001,
            )
            for _ in range(n_slice):
                result = pipeline.force_run()
                if result.get("failed_step"):
                    gate["errors"].append(
                        f"sleep_pipeline failed_step={result.get('failed_step')}"
                    )
                    break
            gate["sleep_ok"] = not gate["errors"]
        except Exception as exc:
            gate["errors"].append(f"sleep: {exc!r}")
            return [], gate
    else:
        gate["sleep_ok"] = True

    try:
        graph, assignment, rich_club = build_runtime_graph(store)
    except Exception as exc:
        gate["errors"].append(f"build_runtime_graph: {exc!r}")
        return [], gate

    probe_results: list[ProbeResult] = []
    for probe in probes:
        try:
            cue_emb = list(embedder.embed_batch([probe.cue])[0])

            _route, _cue_hash = _bench_efe_route_for_cue(probe.cue)
            _arousal_route, _arousal_cue_hash = _bench_arousal_route_for_cue(probe.cue)

            from iai_mcp.cue_router import _classify_cue
            _bench_mode_unused, _bench_intent, _bench_label_unused = _classify_cue(probe.cue)

            resp = recall_for_benchmark(
                store=store, graph=graph, assignment=assignment,
                rich_club=rich_club, embedder=embedder,
                cue=probe.cue, session_id="bench-probe",
                k_hits=max(k_hits, CANDIDATE_POOL_SIZE),
                mode="concept",
            )
            pipe_hits = list(resp.hits) if hasattr(resp, "hits") else []
            pipe_rank = _rank_of(pipe_hits, probe.expects)

            cosine_hits = _cosine_baseline_topk(
                store, embedder, cue_emb, k=max(k_hits, CANDIDATE_POOL_SIZE)
            )
            cos_rank = _rank_of(cosine_hits, probe.expects)

            pipe_top1 = pipe_hits[0].literal_surface if pipe_hits else ""

            anti_hits_count = len(getattr(resp, "anti_hits", []) or [])
            hints = list(getattr(resp, "hints", []) or [])
            kinds = []
            for h in hints:
                if isinstance(h, dict):
                    k = h.get("kind", "")
                    if k:
                        kinds.append(k)
            s4_emitted = "s4_contradiction" in kinds

            probe_results.append(ProbeResult(
                probe_id=probe.probe_id,
                seed=seed,
                n_slice=n_slice,
                condition=probe.condition,
                topic=probe.topic,
                pipeline_rank=pipe_rank,
                cosine_rank=cos_rank,
                pipeline_hit_at_k=(0 < pipe_rank <= k_hits),
                cosine_hit_at_k=(0 < cos_rank <= k_hits),
                pipeline_top1_text=pipe_top1,
                s4_contradiction_emitted=s4_emitted,
                anti_hits_count=anti_hits_count,
                hint_kinds=",".join(kinds[:10]),
                route=_route,
                cue_hash=_cue_hash,
                arousal_route=_arousal_route,
                arousal_cue_hash=_arousal_cue_hash,
            ))
        except Exception as exc:
            gate["errors"].append(f"probe {probe.probe_id}: {exc!r}")

    return probe_results, gate


def _rank_of(hits: list, expected_substring: str) -> int:
    for i, h in enumerate(hits, start=1):
        if hasattr(h, "literal_surface"):
            surface = h.literal_surface
        elif isinstance(h, dict):
            surface = h.get("literal_surface", "")
        else:
            surface = ""
        if surface and expected_substring in surface:
            return i
    return -1


def _cosine_baseline_topk(store, embedder, cue_emb, k: int) -> list:
    import numpy as np

    @dataclass
    class _Hit:
        literal_surface: str

    rows = store.all_records()
    if not rows:
        return []
    cue = np.asarray(cue_emb, dtype=np.float32)
    cue_norm = cue / (np.linalg.norm(cue) + 1e-12)
    scored: list[tuple[float, str]] = []
    for r in rows:
        emb = np.asarray(getattr(r, "embedding", []), dtype=np.float32)
        if emb.size == 0:
            continue
        emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
        scored.append((float(emb_norm @ cue_norm), getattr(r, "literal_surface", "")))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [_Hit(literal_surface=t[1]) for t in scored[:k]]


def reciprocal_rank(rank: int) -> float:
    return 0.0 if rank <= 0 else 1.0 / rank


def bootstrap_ci_delta_mrr(
    pipe_ranks: list[int], cos_ranks: list[int],
    resamples: int = BOOTSTRAP_RESAMPLES, ci: float = BOOTSTRAP_CI,
    seed: int = 0,
) -> tuple[float, float, float]:
    n = len(pipe_ranks)
    if n == 0 or n != len(cos_ranks):
        return (0.0, 0.0, 0.0)
    deltas = [reciprocal_rank(p) - reciprocal_rank(c)
              for p, c in zip(pipe_ranks, cos_ranks)]
    point = sum(deltas) / n
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [deltas[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((1 - ci) / 2 * resamples)]
    hi = means[int((1 + ci) / 2 * resamples) - 1]
    return (point, lo, hi)


def wilcoxon_signed_rank_p(pipe_ranks: list[int], cos_ranks: list[int]) -> float | None:
    n = sum(1 for p, c in zip(pipe_ranks, cos_ranks)
            if reciprocal_rank(p) != reciprocal_rank(c))
    if n < WILCOXON_MIN_N:
        return None
    try:
        from scipy.stats import wilcoxon
        diffs = [reciprocal_rank(p) - reciprocal_rank(c)
                 for p, c in zip(pipe_ranks, cos_ranks)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat = wilcoxon(diffs, zero_method="wilcox", alternative="greater")
        return float(stat.pvalue)
    except Exception:
        return None


def aggregate(
    all_results: list[ProbeResult],
    seeds: list[int],
    n_slices: list[int],
    k_hits: int,
    a_threshold: float = 0.98,
    floor_mode: str = "relaxed",
) -> dict[str, Any]:
    summary: dict[str, Any] = {"per_cell": [], "cross_seed": {}, "gates": {}}

    per_cell_b_deltas: dict[int, list[float]] = {n: [] for n in n_slices}
    per_cell_a_hit: dict[int, list[float]] = {n: [] for n in n_slices}
    per_cell_a_floor_violations: dict[int, list[int]] = {n: [] for n in n_slices}
    catastrophic_b: dict[int, list[int]] = {n: [] for n in n_slices}

    for seed in seeds:
        for n in n_slices:
            cell_results = [r for r in all_results if r.seed == seed and r.n_slice == n]

            b_probes = [r for r in cell_results if r.condition == "post_flip"]
            pipe_ranks_b = [r.pipeline_rank for r in b_probes]
            cos_ranks_b = [r.cosine_rank for r in b_probes]

            point, lo, hi = bootstrap_ci_delta_mrr(
                pipe_ranks_b, cos_ranks_b, seed=seed
            )
            wilcoxon_p = wilcoxon_signed_rank_p(pipe_ranks_b, cos_ranks_b)
            max_regression = max(
                (r.cosine_rank - r.pipeline_rank for r in b_probes
                 if r.pipeline_rank > 0 and r.cosine_rank > 0),
                default=0,
            )
            rr_at_1_pipe = sum(1 for r in b_probes if r.pipeline_rank == 1) / max(len(b_probes), 1)
            rr_at_1_cos = sum(1 for r in b_probes if r.cosine_rank == 1) / max(len(b_probes), 1)

            a_probes = [r for r in cell_results if r.condition == "historical_verbatim"]
            a_hits_pipe = sum(1 for r in a_probes if r.pipeline_hit_at_k) / max(len(a_probes), 1)
            a_hits_cos = sum(1 for r in a_probes if r.cosine_hit_at_k) / max(len(a_probes), 1)
            if floor_mode == "strict":
                a_floor_viols = sum(
                    1 for r in a_probes
                    if any(marker in r.pipeline_top1_text
                           for marker in ("Correction", "Update:", "revised", "Switched",
                                          "relocated", "Migrated", "rolled forward",
                                          "venue changed"))
                )
            else:
                a_floor_viols = sum(1 for r in a_probes if not r.pipeline_hit_at_k)

            n_b = len(b_probes)
            hint_emit_rate = (
                sum(1 for r in b_probes if r.s4_contradiction_emitted) / n_b
                if n_b > 0 else 0.0
            )
            anti_hits_coverage = (
                sum(1 for r in b_probes if r.anti_hits_count > 0) / n_b
                if n_b > 0 else 0.0
            )
            mean_anti_hits = (
                statistics.mean(r.anti_hits_count for r in b_probes)
                if n_b > 0 else 0.0
            )

            cell = {
                "seed": seed, "n_slice": n,
                "n_b_probes": n_b, "n_a_probes": len(a_probes),
                "metric_b": {
                    "delta_mrr_point": round(point, 6),
                    "delta_mrr_ci_lo": round(lo, 6),
                    "delta_mrr_ci_hi": round(hi, 6),
                    "wilcoxon_p": wilcoxon_p,
                    "max_rank_regression": max_regression,
                    "rr_at_1_pipeline": round(rr_at_1_pipe, 4),
                    "rr_at_1_cosine": round(rr_at_1_cos, 4),
                },
                "metric_b_revised": {
                    "hint_emission_rate": round(hint_emit_rate, 4),
                    "anti_hits_coverage": round(anti_hits_coverage, 4),
                    "mean_anti_hits_count": round(mean_anti_hits, 4),
                },
                "metric_a": {
                    "hit_at_k_pipeline": round(a_hits_pipe, 4),
                    "hit_at_k_cosine": round(a_hits_cos, 4),
                    "k": k_hits,
                    "catastrophic_floor_violations": a_floor_viols,
                },
            }
            summary["per_cell"].append(cell)

            per_cell_b_deltas[n].append(point)
            per_cell_a_hit[n].append(a_hits_pipe)
            per_cell_a_floor_violations[n].append(a_floor_viols)
            catastrophic_b[n].append(max_regression)

    cross: dict[str, Any] = {}
    for n in n_slices:
        deltas = per_cell_b_deltas[n]
        if len(deltas) < 2:
            cross[f"n_{n}"] = {"delta_mrr_mean": deltas[0] if deltas else 0.0,
                                "delta_mrr_stdev": 0.0, "robust": deltas and deltas[0] > 0}
        else:
            mean = statistics.mean(deltas)
            stdev = statistics.stdev(deltas)
            cross[f"n_{n}"] = {
                "delta_mrr_mean": round(mean, 6),
                "delta_mrr_stdev": round(stdev, 6),
                "delta_mrr_min": round(min(deltas), 6),
                "delta_mrr_max": round(max(deltas), 6),
                "robust": (mean - stdev) > 0,
            }
    summary["cross_seed"] = cross

    gates: dict[str, dict[str, Any]] = {}
    for cell in summary["per_cell"]:
        key = f"seed{cell['seed']}_n{cell['n_slice']}"
        b = cell["metric_b"]
        b_rev = cell["metric_b_revised"]
        a = cell["metric_a"]
        gate_b_classical = (
            b["delta_mrr_ci_lo"] > 0 and
            b["max_rank_regression"] < MAX_RANK_REGRESSION and
            b["rr_at_1_pipeline"] >= b["rr_at_1_cosine"]
        )
        gate_b_contract = (
            b_rev["hint_emission_rate"] >= 0.80 or
            b_rev["anti_hits_coverage"] >= 0.80
        )
        a_baseline = max(a["hit_at_k_cosine"], 1e-6)
        gate_a = (
            (a["hit_at_k_pipeline"] / a_baseline >= a_threshold) and
            a["catastrophic_floor_violations"] == 0
        )
        gates[key] = {
            "gate_a": gate_a,
            "gate_b_classical": gate_b_classical,
            "gate_b_contract": gate_b_contract,
        }
    summary["gates"]["per_cell"] = gates

    cross_gate = all(c["robust"] for c in cross.values())
    summary["gates"]["cross_seed_robust"] = cross_gate

    summary["gates"]["overall_pass"] = (
        all(g["gate_a"] and g["gate_b_contract"] for g in gates.values())
    )

    return summary


def write_outputs(
    output_dir: Path, run_id: str,
    summary: dict[str, Any], all_results: list[ProbeResult],
    env: dict[str, Any], duration_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / f"contradiction_longitudinal_{run_id}.md"
    json_path = output_dir / f"contradiction_longitudinal_{run_id}.json"
    csv_path = output_dir / f"contradiction_longitudinal_{run_id}.csv"

    env_with_duration = {**env, "wall_clock_duration_seconds": round(duration_seconds, 2)}
    json_blob = {"env": env_with_duration, "summary": summary}
    json_path.write_text(json.dumps(json_blob, indent=2, default=str))

    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "probe_id", "seed", "n_slice", "condition", "topic",
            "pipeline_rank", "cosine_rank",
            "pipeline_hit_at_k", "cosine_hit_at_k",
            "s4_contradiction_emitted", "anti_hits_count", "hint_kinds",
            "pipeline_top1_text",
            "route", "cue_hash",
            "arousal_route", "arousal_cue_hash",
        ])
        for r in all_results:
            w.writerow([
                r.probe_id, r.seed, r.n_slice, r.condition, r.topic,
                r.pipeline_rank, r.cosine_rank,
                int(r.pipeline_hit_at_k), int(r.cosine_hit_at_k),
                int(r.s4_contradiction_emitted), r.anti_hits_count, r.hint_kinds,
                r.pipeline_top1_text[:200],
                r.route, r.cue_hash,
                r.arousal_route, r.arousal_cue_hash,
            ])

    overall = "PASS" if summary["gates"]["overall_pass"] else "FAIL"
    lines = [
        f"# Contradiction-longitudinal falsifiability bench — {overall}",
        "",
        f"**Run ID:** {run_id}",
        f"**Duration:** {duration_seconds:.1f}s",
        "",
        "## Environment",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for k, v in env_with_duration.items():
        lines.append(f"| `{k}` | {v} |")
    lines += ["", "## Cross-seed (B robustness)", "",
              "| N slice | ΔMRR mean | stdev | min | max | robust? |",
              "|---|---|---|---|---|---|"]
    for n_key, c in summary["cross_seed"].items():
        lines.append(
            f"| {n_key} | {c.get('delta_mrr_mean', 0):.4f} | "
            f"{c.get('delta_mrr_stdev', 0):.4f} | "
            f"{c.get('delta_mrr_min', 0):.4f} | "
            f"{c.get('delta_mrr_max', 0):.4f} | "
            f"{'YES' if c.get('robust') else 'NO'} |"
        )
    lines += ["", "## Per-cell detail", "",
              "| seed | N | A hit@k (pipe / cos) | A floor | "
              "B-class ΔMRR (CI) | B-contract hint% / anti-hits% | "
              "gate A | gate B-class | gate B-contract |",
              "|---|---|---|---|---|---|---|---|---|"]
    gates_pc = summary["gates"]["per_cell"]
    for cell in summary["per_cell"]:
        key = f"seed{cell['seed']}_n{cell['n_slice']}"
        g = gates_pc.get(key, {})
        b = cell["metric_b"]
        b_rev = cell["metric_b_revised"]
        a = cell["metric_a"]
        lines.append(
            f"| {cell['seed']} | {cell['n_slice']} | "
            f"{a['hit_at_k_pipeline']:.3f} / {a['hit_at_k_cosine']:.3f} | "
            f"{a['catastrophic_floor_violations']} | "
            f"{b['delta_mrr_point']:.4f} "
            f"({b['delta_mrr_ci_lo']:.4f}, {b['delta_mrr_ci_hi']:.4f}) | "
            f"{b_rev['hint_emission_rate']:.3f} / {b_rev['anti_hits_coverage']:.3f} | "
            f"{'PASS' if g.get('gate_a') else 'FAIL'} | "
            f"{'PASS' if g.get('gate_b_classical') else 'FAIL'} | "
            f"{'PASS' if g.get('gate_b_contract') else 'FAIL'} |"
        )
    lines += [
        "",
        "**Cross-seed robust gate (B-classical only):** "
        f"{'PASS' if summary['gates']['cross_seed_robust'] else 'FAIL (expected: B-class is not the architectural promise)'}",
        f"**Overall verdict (uses gate_a + gate_b_contract):** {overall}",
        "",
        "## Notes on metric design",
        "",
        "- **Metric A (verbatim preserved)** tests REQUIREMENTS.md MEM-05 — the system's promise that contradiction = reconsolidation, never overwrite. Pipeline beating cosine here = real architectural advantage.",
        "- **Metric B-classical (rank current above cosine)** tests an expectation the system does not promise: it uses dual-route + inhibitory edges + hints, not rerank. Expect ΔMRR ≈ 0; this is a feature, not a bug.",
        "- **Metric B-contract (s4_contradiction hint OR anti_hits ≥80%)** tests what the system actually promises (REQUIREMENTS.md MEM-08, MCP-01 dual-route). Cosine cannot do either; pipeline either signals contradictions or it doesn't.",
        "",
    ]
    md_path.write_text("\n".join(lines))

    print(f"[bench] Wrote: {md_path}")
    print(f"[bench] Wrote: {json_path}")
    print(f"[bench] Wrote: {csv_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Contradiction-longitudinal falsifiability bench (Claude impl)",
    )
    parser.add_argument("--scale", choices=list(SCALE_PRESETS.keys()), default="smoke")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR,
                        help=f"Bench-isolated IAI_MCP_STORE (default {DEFAULT_STORE_DIR})")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="≥3 RNG seeds (DESIGN §5.2 mandatory)")
    parser.add_argument("--n-slices", nargs="+", type=int, default=[0, 1],
                        help="Sleep cycles to run per seed (DESIGN §3.2)")
    parser.add_argument("--k-hits", type=int, default=DEFAULT_K)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedder", default="bge-small-en-v1.5")
    parser.add_argument("--a-threshold", type=float, default=0.98,
                        help="Metric A: A_after >= a_threshold * A_baseline")
    parser.add_argument("--floor-mode", choices=["strict", "relaxed"], default="relaxed",
                        help=("Catastrophic floor for Metric A. "
                              "strict = top-1 must NOT be a correction marker (DESIGN.md literal); "
                              "relaxed = original must be in top-k (more lenient, default)."))
    args = parser.parse_args(argv)

    if len(args.seeds) < 3:
        print(f"[setup-gate] REFUSE: need ≥3 seeds, got {len(args.seeds)} "
              f"(DESIGN §5.2 mandatory).", file=sys.stderr)
        return 2

    store_dir = Path(args.store_dir).expanduser().resolve()
    _refuse_production_store(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    n_sessions, n_pre, n_post = SCALE_PRESETS[args.scale]

    env = _gather_env_metadata(store_dir, args.seeds)
    env.update({
        "scale": args.scale, "n_sessions": n_sessions,
        "n_probes_pre": n_pre, "n_probes_post": n_post,
        "n_slices": args.n_slices, "k_hits": args.k_hits,
        "embedder_model": args.embedder, "a_threshold": args.a_threshold,
        "candidate_pool_size": CANDIDATE_POOL_SIZE,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    })

    run_id = (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        f"-{env['iai_mcp_git_sha']}"
        f"-seeds{'-'.join(map(str, args.seeds))}"
        f"-scale_{args.scale}"
    )

    print(f"[bench] Run ID: {run_id}")
    print(f"[bench] Scale: {args.scale}  ({n_sessions} sessions, "
          f"{n_pre} pre + {n_post} post probes)")
    print(f"[bench] Seeds: {args.seeds}  N-slices: {args.n_slices}")
    print(f"[bench] Store: {store_dir}")
    print(f"[bench] git: {env['iai_mcp_git_sha']} "
          f"({'dirty' if env['iai_mcp_git_dirty'] else 'clean'})")

    t0 = time.time()
    all_results: list[ProbeResult] = []
    setup_failed = False

    for seed in args.seeds:
        entries, probes = generate_corpus(seed, n_sessions, n_pre, n_post)
        for n_slice in args.n_slices:
            cell_dir = store_dir / f"seed{seed}_n{n_slice}"
            print(f"[bench]   seed={seed} n_slice={n_slice}  "
                  f"(corpus={len(entries)} probes={len(probes)})  ...", flush=True)
            cell_t0 = time.time()
            results, gate = run_one_seed(
                seed=seed, entries=entries, probes=probes, n_slice=n_slice,
                store_dir_for_seed=cell_dir,
                embedder_key=args.embedder, k_hits=args.k_hits,
            )
            cell_dt = time.time() - cell_t0
            ok = gate["insert_ok"] and gate["contradict_ok"] and gate["sleep_ok"]
            print(f"[bench]     -> {'ok' if ok else 'FAILED'}  "
                  f"({cell_dt:.1f}s, {len(results)} probes)  "
                  f"errors={gate['errors'][:2]}", flush=True)
            if not ok:
                setup_failed = True
            all_results.extend(results)

    duration = time.time() - t0

    if setup_failed and not all_results:
        print(f"[bench] Setup-level failure: no probes ran. Exit 2.", file=sys.stderr)
        return 2

    summary = aggregate(
        all_results=all_results, seeds=args.seeds, n_slices=args.n_slices,
        k_hits=args.k_hits, a_threshold=args.a_threshold,
        floor_mode=args.floor_mode,
    )
    env["floor_mode"] = args.floor_mode

    write_outputs(
        output_dir=output_dir, run_id=run_id, summary=summary,
        all_results=all_results, env=env, duration_seconds=duration,
    )

    overall = summary["gates"]["overall_pass"]
    print(f"[bench] Overall: {'PASS' if overall else 'FAIL'}  "
          f"({duration:.1f}s)")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
