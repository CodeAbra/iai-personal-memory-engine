#!/usr/bin/env python3
"""Score this system against a portable longitudinal dataset file.

Reference implementation of the protocol embedded in the dataset (see
bench/longitudinal_export.py): entries are inserted strictly in order,
probes run against the recall pipeline, and rescue@k / poison@k / MRR are
computed exactly as the protocol text specifies. The dataset file is the
only input — this script never regenerates the corpus.
"""
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
import asyncio
import json
import os
import statistics
import time
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

PRODUCTION_STORE = Path.home() / ".iai-mcp"
BENCH_PASSPHRASE = "iai-mcp-bench-falsifiability-deterministic-2026"
DEFAULT_STORE_DIR = "/tmp/iai-mcp-longitudinal/store"
DEFAULT_KS = [1, 5, 10]
DEFAULT_POOL = 50
TIERS = ("T1-ambient", "T2-native")


def rank_of_substring(texts: "list[str]", needle: str) -> int:
    for i, text in enumerate(texts, start=1):
        if needle and needle in text:
            return i
    return -1


def rescue_at_k(texts: "list[str]", expects: str, k: int) -> int:
    rank = rank_of_substring(texts[:k], expects)
    return 1 if rank > 0 else 0


def poison_at_k(texts: "list[str]", expects: str, counterfact: str, k: int) -> int:
    top = texts[:k]
    expects_rank = rank_of_substring(top, expects)
    counter_rank = rank_of_substring(top, counterfact)
    if counter_rank <= 0:
        return 0
    if expects_rank <= 0:
        return 1
    return 1 if counter_rank < expects_rank else 0


def reciprocal_rank_of(texts: "list[str]", expects: str) -> float:
    rank = rank_of_substring(texts, expects)
    return 0.0 if rank <= 0 else 1.0 / rank


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


def _load_dataset(path: Path) -> "dict[str, Any]":
    payload = json.loads(path.read_text())
    for field in ("protocol", "entries", "probes"):
        if field not in payload:
            print(f"[setup-gate] REFUSE: dataset missing `{field}`.", file=sys.stderr)
            sys.exit(2)
    return payload


def _run_tier(
    tier: str,
    payload: "dict[str, Any]",
    store_dir: Path,
    pool: int,
) -> "list[dict[str, Any]]":
    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph, contradict
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    store_dir.mkdir(parents=True, exist_ok=True)
    if any(store_dir.iterdir()):
        print(
            f"[setup-gate] REFUSE: store dir {store_dir} is not empty; "
            f"remove it and rerun.",
            file=sys.stderr,
        )
        sys.exit(2)

    os.environ["IAI_MCP_STORE"] = str(store_dir)
    if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = BENCH_PASSPHRASE

    store = MemoryStore(path=store_dir / "lancedb")
    # Same insert pathway as the published retrieval bench; disable below is
    # the drain barrier that makes every row visible before the graph build.
    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))
    embedder = Embedder()

    entries = payload["entries"]
    texts = [e["text"] for e in entries]
    embeddings = embedder.embed_batch(texts)
    # Monotonic timestamps that all lie in the PAST at probe time (newest =
    # now): a future-dated correction would read as "not yet in effect" and
    # the staleness machinery would rightly skip it.
    base = datetime.now(timezone.utc) - timedelta(hours=len(entries))

    topic_to_orig_id: "dict[str, UUID]" = {}
    pending_corrections: "list[tuple[str, str, list[float]]]" = []
    for entry, emb in zip(entries, embeddings):
        ts = base + timedelta(hours=int(entry["order"]))
        rec_id = uuid4()
        rec = MemoryRecord(
            id=rec_id,
            tier="episodic",
            literal_surface=entry["text"],
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
            provenance=[{"ts": ts.isoformat(), "cue": entry["text"][:60],
                         "session_id": entry["session_id"]}],
            created_at=ts,
            updated_at=ts,
            tags=["bench-longitudinal", f"role:{entry['role']}",
                  f"session:{entry['session_id']}"],
            language="en",
        )
        store.insert(rec)
        if entry.get("flip") == "original" and entry.get("topic"):
            topic_to_orig_id[entry["topic"]] = rec_id
        if tier == "T2-native" and entry.get("flip") == "correction":
            pending_corrections.append(
                (entry.get("topic") or "", entry["text"], list(emb))
            )

    # Drain first: contradict must be able to read the original row.
    asyncio.run(store.disable_async_writes())
    for topic, corr_text, cue_emb in pending_corrections:
        if topic in topic_to_orig_id:
            contradict(store, topic_to_orig_id[topic], corr_text, cue_emb)

    graph, assignment, rich_club = build_runtime_graph(store)

    rows: "list[dict[str, Any]]" = []
    for probe in payload["probes"]:
        resp = recall_for_benchmark(
            store=store, graph=graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder,
            cue=probe["cue"], session_id="longitudinal-score",
            k_hits=pool, mode="concept",
        )
        hits = list(resp.hits) if hasattr(resp, "hits") else []
        hit_texts = [getattr(h, "literal_surface", "") or "" for h in hits]
        rows.append({
            "probe_id": probe["probe_id"],
            "condition": probe["condition"],
            "topic": probe["topic"],
            "tier": tier,
            "expects": probe["expects"],
            "counterfact": probe.get("counterfact", ""),
            "hit_texts": hit_texts,
        })
    return rows


def _aggregate(
    rows: "list[dict[str, Any]]", ks: "list[int]",
) -> "dict[str, Any]":
    out: "dict[str, Any]" = {}
    conditions = sorted({r["condition"] for r in rows})
    for condition in conditions:
        cond_rows = [r for r in rows if r["condition"] == condition]
        block: "dict[str, Any]" = {"n_probes": len(cond_rows)}
        for k in ks:
            block[f"rescue_at_{k}"] = round(statistics.mean(
                rescue_at_k(r["hit_texts"], r["expects"], k)
                for r in cond_rows
            ), 4)
        if condition == "post_flip":
            for k in ks:
                block[f"poison_at_{k}"] = round(statistics.mean(
                    poison_at_k(r["hit_texts"], r["expects"],
                                r["counterfact"], k)
                    for r in cond_rows
                ), 4)
        block["mrr"] = round(statistics.mean(
            reciprocal_rank_of(r["hit_texts"], r["expects"])
            for r in cond_rows
        ), 4)
        out[condition] = block
    return out


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True,
                        help="Portable dataset JSON from longitudinal_export.py")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR,
                        help=f"Bench-isolated store root (default {DEFAULT_STORE_DIR})")
    parser.add_argument("--tiers", nargs="+", choices=list(TIERS),
                        default=list(TIERS))
    parser.add_argument("--ks", nargs="+", type=int, default=DEFAULT_KS)
    parser.add_argument("--pool", type=int, default=DEFAULT_POOL,
                        help="Hits requested per probe; must be >= max(ks)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    if args.pool < max(args.ks):
        print(f"[setup-gate] REFUSE: --pool {args.pool} < max(--ks) "
              f"{max(args.ks)}.", file=sys.stderr)
        return 2

    store_root = Path(args.store_dir).expanduser().resolve()
    _refuse_production_store(store_root)

    dataset_path = Path(args.dataset).expanduser().resolve()
    payload = _load_dataset(dataset_path)
    print(f"[score] dataset={dataset_path.name} "
          f"protocol={payload['protocol'].get('name', '?')} "
          f"entries={len(payload['entries'])} probes={len(payload['probes'])}")

    t0 = time.time()
    all_rows: "list[dict[str, Any]]" = []
    results: "dict[str, Any]" = {}
    for tier in args.tiers:
        tier_dir = store_root / tier
        print(f"[score] tier={tier} ...", flush=True)
        tier_t0 = time.time()
        rows = _run_tier(tier, payload, tier_dir, args.pool)
        all_rows.extend(rows)
        results[tier] = _aggregate(rows, sorted(args.ks))
        print(f"[score]   -> {len(rows)} probes in "
              f"{time.time() - tier_t0:.1f}s", flush=True)

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "dataset": dataset_path.name,
        "protocol": payload["protocol"].get("name", ""),
        "seed": payload.get("seed"),
        "scale": payload.get("scale"),
        "ks": sorted(args.ks),
        "pool": args.pool,
        "duration_seconds": round(time.time() - t0, 1),
        "results": results,
        "per_probe": [
            {k: v for k, v in r.items() if k != "hit_texts"}
            | {"top3": r["hit_texts"][:3]}
            for r in all_rows
        ],
    }, indent=1, ensure_ascii=False))

    for tier, blocks in results.items():
        for condition, block in blocks.items():
            print(f"[score] {tier} {condition}: "
                  + " ".join(f"{k}={v}" for k, v in block.items()))
    print(f"[score] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
