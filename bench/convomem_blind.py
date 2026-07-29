#!/usr/bin/env python3
"""ConvoMem blind-run orchestrator: public-API retrieval, no per-dataset tuning.

Pre-mixed conversational-memory test cases (Salesforce/ConvoMem): each case
carries a question, the answer, verbatim evidence messages, and a haystack of
conversations with filler mixed in at a stated context size. Every haystack
message is inserted as one record through the system's own insert pathway;
the question runs through the recall pipeline and is scored on whether the
evidence messages appear in the top-k (exact-text match, as the dataset's
message_evidences are verbatim copies of haystack messages).

Category directories (user_evidence, assistant_facts_evidence,
changing_evidence, preference_evidence, implicit_connection_evidence,
abstention_evidence) and evidence counts are part of the dataset layout;
abstention cases have no retrievable evidence and are skipped with a
disclosed count.
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
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

PRODUCTION_STORE = Path.home() / ".iai-mcp"
BENCH_PASSPHRASE = "iai-mcp-bench-falsifiability-deterministic-2026"
BASE_URL = (
    "https://huggingface.co/datasets/Salesforce/ConvoMem/resolve/main/"
    "core_benchmark/pre_mixed_testcases"
)
DEFAULT_CACHE = Path.home() / ".cache" / "iai-mcp-bench" / "convomem"
DEFAULT_STORE_DIR = "/tmp/iai-mcp-convomem/store"

CATEGORIES = {
    "user_evidence": [1, 2, 3, 4, 5, 6],
    "assistant_facts_evidence": [1, 2, 3, 4, 5, 6],
    "changing_evidence": [2, 3, 4, 5, 6],
    "implicit_connection_evidence": [1, 2, 3],
    "preference_evidence": [1, 2],
}


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


def _fetch_batch(cache_dir: Path, category: str, n_evidence: int) -> Path:
    rel = f"{category}/{n_evidence}_evidence/batched_000.json"
    local = cache_dir / rel
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/{rel}"
        print(f"[convomem] downloading {rel} ...", flush=True)
        with urllib.request.urlopen(url, timeout=300) as resp:
            payload = resp.read()
        if len(payload) < 100:
            print(f"[setup-gate] REFUSE: {url} returned {len(payload)} bytes.",
                  file=sys.stderr)
            sys.exit(2)
        local.write_bytes(payload)
    return local


def rank_of_text(ranked_texts: "list[str]", needle: str) -> int:
    needle_norm = " ".join(needle.split())
    for i, text in enumerate(ranked_texts, start=1):
        if needle_norm and needle_norm in " ".join(text.split()):
            return i
    return -1


def evidence_recall_at_k(
    ranked_texts: "list[str]", evidences: "list[str]", k: int,
) -> float:
    if not evidences:
        return 0.0
    top = ranked_texts[:k]
    found = sum(1 for e in evidences if rank_of_text(top, e) > 0)
    return found / len(evidences)


def any_evidence_hit_at_k(
    ranked_texts: "list[str]", evidences: "list[str]", k: int,
) -> int:
    top = ranked_texts[:k]
    return 1 if any(rank_of_text(top, e) > 0 for e in evidences) else 0


def _run_one_case(
    case: "dict[str, Any]",
    store_dir: Path,
    pool: int,
    warm_lexical: bool = False,
    entity_link: bool = False,
) -> "dict[str, Any] | None":
    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    items = case.get("evidenceItems") or []
    if not items:
        return None
    item = items[0]
    question = str(item.get("question", "")).strip()
    evidences = [
        str(m.get("text", "")).strip()
        for m in (item.get("message_evidences") or [])
        if str(m.get("text", "")).strip()
    ]
    if not question or not evidences:
        return None

    texts: "list[str]" = []
    for conv in case.get("conversations") or []:
        for msg in conv.get("messages") or []:
            text = str(msg.get("text", "")).strip()
            if not text:
                continue
            speaker = str(msg.get("speaker", "")).strip()
            texts.append(f"{speaker}: {text}" if speaker else text)

    store_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IAI_MCP_STORE"] = str(store_dir)
    if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = BENCH_PASSPHRASE
    store = MemoryStore(path=store_dir / "lancedb")
    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))
    embedder = Embedder()

    base = datetime.now(timezone.utc) - timedelta(minutes=len(texts) + 1)
    embeddings = embedder.embed_batch(texts)
    for i, (surface, emb) in enumerate(zip(texts, embeddings)):
        ts = base + timedelta(minutes=i)
        store.insert(MemoryRecord(
            id=uuid4(),
            tier="episodic",
            literal_surface=surface,
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
            provenance=[{"ts": ts.isoformat(), "cue": surface[:60],
                         "session_id": "bench-convomem"}],
            created_at=ts,
            updated_at=ts,
            tags=["bench-convomem"],
            language="en",
        ))
    asyncio.run(store.disable_async_writes())
    if warm_lexical:
        try:
            store.lexical_search("lexical warm-up", k=1)
        except Exception:  # noqa: BLE001 -- warm-up is best-effort
            pass
    if entity_link:
        from iai_mcp.entity_link import mine_entity_edges
        mine_entity_edges(store)
    graph, assignment, rich_club = build_runtime_graph(store)

    resp = recall_for_benchmark(
        store=store, graph=graph, assignment=assignment,
        rich_club=rich_club, embedder=embedder,
        cue=question, session_id="bench-convomem",
        k_hits=pool, mode="concept",
    )
    hits = list(resp.hits) if hasattr(resp, "hits") else []
    ranked_texts = [getattr(h, "literal_surface", "") or "" for h in hits]

    return {
        "question": question,
        "n_evidence": len(evidences),
        "n_messages": len(texts),
        "context_size": int(case.get("contextSize", 0)),
        "evidence_recall_at_5": evidence_recall_at_k(ranked_texts, evidences, 5),
        "evidence_recall_at_10": evidence_recall_at_k(ranked_texts, evidences, 10),
        "any_hit_at_5": any_evidence_hit_at_k(ranked_texts, evidences, 5),
        "any_hit_at_10": any_evidence_hit_at_k(ranked_texts, evidences, 10),
    }


def _aggregate(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    metrics = (
        "evidence_recall_at_5", "evidence_recall_at_10",
        "any_hit_at_5", "any_hit_at_10",
    )

    def _block(subset: "list[dict[str, Any]]") -> "dict[str, Any]":
        return {
            m: round(statistics.mean(float(r[m]) for r in subset), 4)
            for m in metrics
        } | {"n": len(subset)}

    out: "dict[str, Any]" = {}
    if rows:
        out["overall"] = _block(rows)
        by_cat: "dict[str, Any]" = {}
        for cat in sorted({r["category"] for r in rows}):
            by_cat[cat] = _block([r for r in rows if r["category"] == cat])
        out["by_category"] = by_cat
        by_size: "dict[str, Any]" = {}
        for size in sorted({r["context_size"] for r in rows}):
            by_size[f"ctx_{size}"] = _block(
                [r for r in rows if r["context_size"] == size])
        out["by_context_size"] = by_size
    return out


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE),
                        help="dataset cache; batches auto-downloaded when missing")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    parser.add_argument("--categories", nargs="+",
                        choices=sorted(CATEGORIES), default=sorted(CATEGORIES))
    parser.add_argument("--max-context", type=int, default=30,
                        help="skip cases whose contextSize exceeds this")
    parser.add_argument("--cases-per-category", type=int, default=40,
                        help="cap on scored cases per category")
    parser.add_argument("--pool", type=int, default=50)
    parser.add_argument("--warm-lexical", action="store_true",
                        help="build the per-store lexical index after ingest "
                             "(matches a long-lived daemon post warm-up)")
    parser.add_argument("--entity-link", action="store_true",
                        help="mine entity_shared edges before the graph build "
                             "(matches a store after one consolidation cycle)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    store_root = Path(args.store_dir).expanduser().resolve()
    _refuse_production_store(store_root)
    cache_dir = Path(args.cache_dir).expanduser()

    t0 = time.time()
    all_rows: "list[dict[str, Any]]" = []
    skipped_oversize = 0
    for category in args.categories:
        batch_path = _fetch_batch(cache_dir, category, CATEGORIES[category][0])
        cases = json.loads(batch_path.read_text())
        scored = 0
        cat_t0 = time.time()
        for case_idx, case in enumerate(cases):
            if scored >= args.cases_per_category:
                break
            if int(case.get("contextSize", 0)) > args.max_context:
                skipped_oversize += 1
                continue
            row = _run_one_case(
                case,
                store_root / category / f"case-{case_idx:04d}",
                pool=args.pool,
                warm_lexical=args.warm_lexical,
                entity_link=args.entity_link,
            )
            if row is None:
                continue
            row["category"] = category
            all_rows.append(row)
            scored += 1
        print(
            f"[convomem] {category}: {scored} cases in "
            f"{time.time() - cat_t0:.1f}s",
            flush=True,
        )

    summary = _aggregate(all_rows)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "bench": "convomem-blind-retrieval",
        "max_context": args.max_context,
        "cases_per_category": args.cases_per_category,
        "pool": args.pool,
        "n_skipped_oversize": skipped_oversize,
        "duration_seconds": round(time.time() - t0, 1),
        "summary": summary,
        "per_case": all_rows,
    }, indent=1, ensure_ascii=False))

    print(f"[convomem] overall: {summary.get('overall')}")
    for label, block in (summary.get("by_category") or {}).items():
        print(f"[convomem]   {label}: {block}")
    print(f"[convomem] skipped oversize (> ctx {args.max_context}): "
          f"{skipped_oversize}")
    print(f"[convomem] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
