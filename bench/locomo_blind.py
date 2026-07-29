#!/usr/bin/env python3
"""LoCoMo blind-run orchestrator: public-API retrieval, no per-dataset tuning.

Long-conversation memory QA (Maharana et al. 2024): 10 conversations, each
up to ~35 sessions, questions annotated with the dialog turns that contain
the answer. Every turn is inserted as one record through the system's own
insert pathway; each question runs through the recall pipeline and is scored
on whether the annotated evidence turns appear in the top-k.

Adversarial questions (category 5) carry no evidence by design and are
excluded from retrieval metrics; their count is disclosed in the summary.
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
import random
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

PRODUCTION_STORE = Path.home() / ".iai-mcp"
BENCH_PASSPHRASE = "iai-mcp-bench-falsifiability-deterministic-2026"
DATA_URL = (
    "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
)
DEFAULT_CACHE = Path.home() / ".cache" / "iai-mcp-bench" / "locomo10.json"
DEFAULT_STORE_DIR = "/tmp/iai-mcp-locomo/store"
ADVERSARIAL_CATEGORY = 5

CATEGORY_LABELS = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
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


def _fetch_dataset(data_path: Path) -> "list[dict[str, Any]]":
    if not data_path.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[locomo] downloading dataset -> {data_path}", flush=True)
        with urllib.request.urlopen(DATA_URL, timeout=120) as resp:
            data_path.write_bytes(resp.read())
    return json.loads(data_path.read_text())


def _parse_session_datetime(raw: str, fallback: datetime) -> datetime:
    # Dataset format: "1:56 pm on 8 May, 2023" (occasionally without minutes).
    for fmt in ("%I:%M %p on %d %B, %Y", "%I %p on %d %B, %Y"):
        try:
            return datetime.strptime(raw.strip(), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return fallback


def _session_keys_in_order(conversation: "dict[str, Any]") -> "list[str]":
    keys = []
    for key in conversation:
        if key.startswith("session_") and not key.endswith("_date_time"):
            try:
                keys.append((int(key.split("_")[1]), key))
            except (IndexError, ValueError):
                continue
    return [k for _n, k in sorted(keys)]


def evidence_recall_at_k(
    ranked_dia_ids: "list[str]", evidence: "list[str]", k: int,
) -> float:
    if not evidence:
        return 0.0
    top = set(ranked_dia_ids[:k])
    return sum(1 for e in evidence if e in top) / len(evidence)


def any_evidence_hit_at_k(
    ranked_dia_ids: "list[str]", evidence: "list[str]", k: int,
) -> int:
    top = set(ranked_dia_ids[:k])
    return 1 if any(e in top for e in evidence) else 0


def _run_one_conversation(
    sample: "dict[str, Any]",
    store_dir: Path,
    pool: int,
    qa_limit: "int | None",
    embed_window: int = 0,
    embed_date: bool = False,
    entity_link: bool = False,
) -> "list[dict[str, Any]]":
    from iai_mcp.embed import Embedder
    from iai_mcp.pipeline import recall_for_benchmark
    from iai_mcp.retrieve import build_runtime_graph
    from iai_mcp.store import MemoryStore
    from iai_mcp.types import MemoryRecord

    store_dir.mkdir(parents=True, exist_ok=True)
    os.environ["IAI_MCP_STORE"] = str(store_dir)
    if not os.environ.get("IAI_MCP_CRYPTO_PASSPHRASE"):
        os.environ["IAI_MCP_CRYPTO_PASSPHRASE"] = BENCH_PASSPHRASE
    store = MemoryStore(path=store_dir / "lancedb")
    asyncio.run(store.enable_async_writes(coalesce_ms=50, max_batch=128))
    embedder = Embedder()

    conversation = sample["conversation"]
    fallback_base = datetime.now(timezone.utc) - timedelta(days=365)

    turns: "list[tuple[str, str, datetime]]" = []
    for s_idx, s_key in enumerate(_session_keys_in_order(conversation)):
        session_ts = _parse_session_datetime(
            str(conversation.get(f"{s_key}_date_time", "")),
            fallback_base + timedelta(days=s_idx),
        )
        for t_idx, turn in enumerate(conversation[s_key] or []):
            text = str(turn.get("text", "")).strip()
            dia_id = str(turn.get("dia_id", ""))
            if not text or not dia_id:
                continue
            speaker = str(turn.get("speaker", "")).strip()
            surface = f"{speaker}: {text}" if speaker else text
            turns.append((dia_id, surface, session_ts + timedelta(seconds=t_idx)))

    import numpy as np

    id_to_dia: "dict[str, str]" = {}
    if embed_window > 0:
        # Embed each turn with its conversational neighbors (same session
        # only): the stored surface stays the verbatim turn — only the
        # vector sees the context that binds a fragment to its topic.
        session_of = [t[0].split(":")[0] for t in turns]
        embed_texts = []
        for i, (_dia, surface, _ts) in enumerate(turns):
            lo = max(0, i - embed_window)
            hi = min(len(turns), i + embed_window + 1)
            parts = [
                turns[j][1] for j in range(lo, hi)
                if session_of[j] == session_of[i]
            ]
            embed_texts.append("\n".join(parts))
    else:
        embed_texts = [t[1] for t in turns]
    if embed_date:
        # Prefix the session date into the embedded text only — timestamps
        # are metadata the record already carries; this binds "when did X"
        # cues to the vector without touching the verbatim surface.
        embed_texts = [
            f"On {ts.strftime('%-d %B %Y')}: {text}"
            for (_dia, _surface, ts), text in zip(turns, embed_texts)
        ]
    embeddings = embedder.embed_batch(embed_texts)
    emb_matrix = np.asarray(embeddings, dtype=np.float32)
    emb_matrix /= (np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-12)
    dia_order = [t[0] for t in turns]
    for (dia_id, surface, ts), emb in zip(turns, embeddings):
        rec_id = uuid4()
        store.insert(MemoryRecord(
            id=rec_id,
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
                         "session_id": dia_id.split(":")[0]}],
            created_at=ts,
            updated_at=ts,
            tags=["bench-locomo", f"dia:{dia_id}"],
            language="en",
        ))
        id_to_dia[str(rec_id)] = dia_id

    asyncio.run(store.disable_async_writes())
    if entity_link:
        from iai_mcp.entity_link import mine_entity_edges
        mine_entity_edges(store)
    graph, assignment, rich_club = build_runtime_graph(store)

    qa_items = sample["qa"][:qa_limit] if qa_limit else sample["qa"]
    rows: "list[dict[str, Any]]" = []
    for qa in qa_items:
        category = int(qa.get("category", 0))
        evidence = [str(e) for e in (qa.get("evidence") or [])]
        if category == ADVERSARIAL_CATEGORY or not evidence:
            rows.append({
                "sample_id": sample.get("sample_id", ""),
                "category": category,
                "question": str(qa.get("question", "")),
                "skipped_no_evidence": True,
            })
            continue
        question = str(qa.get("question", ""))
        resp = recall_for_benchmark(
            store=store, graph=graph, assignment=assignment,
            rich_club=rich_club, embedder=embedder,
            cue=question, session_id="bench-locomo",
            k_hits=pool, mode="concept",
        )
        hits = list(resp.hits) if hasattr(resp, "hits") else []
        ranked_dia = [
            id_to_dia.get(str(h.record_id), "") for h in hits
        ]
        cue_vec = np.asarray(embedder.embed(question), dtype=np.float32)
        cue_vec /= (np.linalg.norm(cue_vec) + 1e-12)
        cos_order = np.argsort(-(emb_matrix @ cue_vec))[:pool]
        cos_ranked_dia = [dia_order[int(i)] for i in cos_order]
        rows.append({
            "sample_id": sample.get("sample_id", ""),
            "category": category,
            "question": question,
            "gold_answer": str(qa.get("answer", "")),
            "retrieved_top10": [
                getattr(h, "literal_surface", "") or "" for h in hits[:10]
            ],
            "n_evidence": len(evidence),
            "evidence_recall_at_5": evidence_recall_at_k(ranked_dia, evidence, 5),
            "evidence_recall_at_10": evidence_recall_at_k(ranked_dia, evidence, 10),
            "any_hit_at_5": any_evidence_hit_at_k(ranked_dia, evidence, 5),
            "any_hit_at_10": any_evidence_hit_at_k(ranked_dia, evidence, 10),
            "cos_any_hit_at_5": any_evidence_hit_at_k(cos_ranked_dia, evidence, 5),
            "cos_any_hit_at_10": any_evidence_hit_at_k(cos_ranked_dia, evidence, 10),
            "cos_evidence_recall_at_10": evidence_recall_at_k(
                cos_ranked_dia, evidence, 10),
        })
    return rows


def _claude_p(prompt: str, timeout: int = 120) -> str:
    """One subscription-billed `claude -p` call; empty string on failure."""
    try:
        proc = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=timeout,
        )
        return (proc.stdout or "").strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _judge_sample(
    rows: "list[dict[str, Any]]", n_sample: int, seed: int = 7,
) -> "dict[str, Any]":
    """Answer-accuracy prong over a disclosed sample: answer strictly from
    the retrieved top-10 turns, then a yes/no judge against gold. This is
    the axis third-party leaderboards report; retrieval metrics above stay
    the primary result."""
    candidates = [
        r for r in rows
        if not r.get("skipped_no_evidence") and r.get("retrieved_top10")
    ]
    rng = random.Random(seed)
    sample = rng.sample(candidates, min(n_sample, len(candidates)))
    correct = 0
    judged = 0
    for r in sample:
        context = "\n".join(r["retrieved_top10"])
        answer = _claude_p(
            "Answer the question using ONLY the conversation excerpts "
            "below. Be concise (one short sentence). If the excerpts do "
            "not contain the answer, say exactly: not found.\n\n"
            f"Excerpts:\n{context}\n\nQuestion: {r['question']}"
        )
        if not answer:
            continue
        verdict = _claude_p(
            "You are grading a QA answer. Reply with exactly YES or NO.\n"
            f"Question: {r['question']}\n"
            f"Gold answer: {r['gold_answer']}\n"
            f"Candidate answer: {answer}\n"
            "Does the candidate answer convey the gold answer?"
        )
        judged += 1
        if verdict.strip().upper().startswith("YES"):
            correct += 1
    return {
        "n_sampled": len(sample),
        "n_judged": judged,
        "accuracy": round(correct / judged, 4) if judged else None,
    }


def _aggregate(rows: "list[dict[str, Any]]") -> "dict[str, Any]":
    scored = [r for r in rows if not r.get("skipped_no_evidence")]
    skipped = len(rows) - len(scored)
    out: "dict[str, Any]" = {
        "n_scored": len(scored),
        "n_skipped_no_evidence": skipped,
    }
    metrics = (
        "evidence_recall_at_5", "evidence_recall_at_10",
        "any_hit_at_5", "any_hit_at_10",
        "cos_any_hit_at_5", "cos_any_hit_at_10",
        "cos_evidence_recall_at_10",
    )

    def _block(subset: "list[dict[str, Any]]") -> "dict[str, float]":
        return {
            m: round(statistics.mean(float(r[m]) for r in subset), 4)
            for m in metrics
        } | {"n": len(subset)}

    if scored:
        out["overall"] = _block(scored)
        by_cat: "dict[str, Any]" = {}
        for cat in sorted({r["category"] for r in scored}):
            label = CATEGORY_LABELS.get(cat, str(cat))
            by_cat[label] = _block([r for r in scored if r["category"] == cat])
        out["by_category"] = by_cat
    return out


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DEFAULT_CACHE),
                        help="dataset JSON path; auto-downloaded when missing")
    parser.add_argument("--store-dir", default=DEFAULT_STORE_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="cap on conversations evaluated (of 10)")
    parser.add_argument("--qa-limit", type=int, default=None,
                        help="cap on questions per conversation (smoke runs)")
    parser.add_argument("--pool", type=int, default=50,
                        help="hits requested per question")
    parser.add_argument("--embed-window", type=int, default=0,
                        help="embed each turn with +-N same-session neighbors "
                             "(surface stays the verbatim turn)")
    parser.add_argument("--embed-date", action="store_true",
                        help="prefix the session date into the embedded text "
                             "(surface stays the verbatim turn)")
    parser.add_argument("--entity-link", action="store_true",
                        help="mine entity_shared edges before the graph build "
                             "(matches a store after one consolidation cycle)")
    parser.add_argument("--judge", type=int, default=0,
                        help="answer-accuracy prong over N sampled questions "
                             "via subscription-billed claude -p (0 = off)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    store_root = Path(args.store_dir).expanduser().resolve()
    _refuse_production_store(store_root)

    samples = _fetch_dataset(Path(args.data).expanduser())
    if args.limit:
        samples = samples[: args.limit]

    t0 = time.time()
    all_rows: "list[dict[str, Any]]" = []
    for i, sample in enumerate(samples):
        conv_t0 = time.time()
        rows = _run_one_conversation(
            sample,
            store_root / f"conv-{i:02d}",
            pool=args.pool,
            qa_limit=args.qa_limit,
            embed_window=args.embed_window,
            embed_date=args.embed_date,
            entity_link=args.entity_link,
        )
        all_rows.extend(rows)
        print(
            f"[locomo] conv {i + 1}/{len(samples)} "
            f"({sample.get('sample_id', '?')}): {len(rows)} questions in "
            f"{time.time() - conv_t0:.1f}s",
            flush=True,
        )

    summary = _aggregate(all_rows)
    if args.judge > 0:
        judge_t0 = time.time()
        summary["judge"] = _judge_sample(all_rows, args.judge)
        summary["judge"]["duration_seconds"] = round(time.time() - judge_t0, 1)
        print(f"[locomo] judge: {summary['judge']}", flush=True)
    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "bench": "locomo-blind-retrieval",
        "n_conversations": len(samples),
        "pool": args.pool,
        "embed_window": args.embed_window,
        "embed_date": args.embed_date,
        "qa_limit": args.qa_limit,
        "duration_seconds": round(time.time() - t0, 1),
        "summary": summary,
        "per_question": [
            {k: v for k, v in r.items() if k != "retrieved_top10"}
            for r in all_rows
        ],
    }, indent=1, ensure_ascii=False))

    print(f"[locomo] overall: {summary.get('overall')}")
    for label, block in (summary.get("by_category") or {}).items():
        print(f"[locomo]   {label}: {block}")
    print(f"[locomo] skipped (adversarial/no-evidence): "
          f"{summary['n_skipped_no_evidence']}")
    print(f"[locomo] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
